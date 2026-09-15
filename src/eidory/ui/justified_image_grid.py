from __future__ import annotations

from eidory.ui.theme_colors import theme_color

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import (
    QEvent,
    QMimeData,
    QObject,
    QPoint,
    QRect,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import QColor, QDrag, QImage, QImageReader, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QAbstractScrollArea, QApplication

from eidory.core.media_types import is_supported_video
from eidory.models import ImageItem
from eidory.ui.collection_tree import IMAGE_IDS_MIME, CollectionTreeWidget
from eidory.ui.drop_import_box import payload_from_mime_data, payload_supports_mime_data


PixmapCacheKey = tuple[str, int, int, int]
RESIZE_LAYOUT_DELAY_MS = 35


class _PixmapLoadSignals(QObject):
    loaded = Signal(object, object)


class _PixmapLoadTask(QRunnable):
    def __init__(
        self,
        key: PixmapCacheKey,
        path: Path,
        max_side: int,
        signals: _PixmapLoadSignals,
    ):
        super().__init__()
        self.key = key
        self.path = path
        self.max_side = max_side
        self.signals = signals

    def run(self) -> None:
        image = JustifiedImageGridView._load_scaled_image(self.path, self.max_side)
        self.signals.loaded.emit(self.key, image)


class JustifiedImageGridView(QAbstractScrollArea):
    selectionChanged = Signal(object)
    selectionSetChanged = Signal(object)
    imageDoubleClicked = Signal(object)
    imagePreviewRequested = Signal(object)
    imageContextMenuRequested = Signal(object, object)
    filesDropped = Signal(object)
    dropPayloadDropped = Signal(object)

    def __init__(self, thumbnail_size: int = 180, spacing: int = 4):
        super().__init__()
        self._images: list[ImageItem] = []
        self._indexes_by_image_id: dict[int, int] = {}
        self._rects: list[QRect] = []
        self._row_ranges: list[tuple[int, int, int, int]] = []
        self._row_tops: list[int] = []
        self._target_height = thumbnail_size
        self._spacing = spacing
        self._selected_index = -1
        self._selected_indexes: set[int] = set()
        self._badges_by_image_id: dict[int, list[str]] = {}
        self._project_references: dict[int, list[str]] = {}
        self._selection_anchor = -1
        self._drag_start_position: QPoint | None = None
        self._drag_start_index = -1
        self._pixmap_cache: OrderedDict[PixmapCacheKey, QPixmap] = OrderedDict()
        self._pixmap_cache_bytes: dict[PixmapCacheKey, int] = {}
        self._pixmap_cache_total_bytes = 0
        self._pending_pixmap_loads: set[PixmapCacheKey] = set()
        self._loaded_pixmap_results: OrderedDict[PixmapCacheKey, QImage] = OrderedDict()
        self._loaded_pixmap_flush_pending = False
        self._layout_signature: tuple[tuple[object, ...], ...] = ()
        self._viewport_update_pending = False
        self._content_height = 0
        self._layout_viewport_width = -1
        self._resize_layout_timer = QTimer(self)
        self._resize_layout_timer.setSingleShot(True)
        self._resize_layout_timer.setInterval(RESIZE_LAYOUT_DELAY_MS)
        self._resize_layout_timer.timeout.connect(self._run_scheduled_layout_rebuild)
        self._pixmap_load_signals = _PixmapLoadSignals()
        self._pixmap_load_signals.loaded.connect(self._handle_async_pixmap_loaded)
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(2)
        self._cache_limit = 700
        self._cache_byte_limit = 180 * 1024 * 1024
        self._pending_load_limit = 48
        self._loaded_pixmap_flush_batch_size = 8
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.viewport().installEventFilter(self)
        self.verticalScrollBar().valueChanged.connect(self._handle_scroll_value_changed)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.viewport():
            if event.type() in {QEvent.Type.DragEnter, QEvent.Type.DragMove}:
                if self._supports_external_import_drop(event.mimeData()):
                    event.setDropAction(Qt.DropAction.CopyAction)
                    event.accept()
                    return True
            if event.type() == QEvent.Type.Drop:
                if self._supports_external_import_drop(event.mimeData()):
                    self.dropPayloadDropped.emit(payload_from_mime_data(event.mimeData()))
                    event.setDropAction(Qt.DropAction.CopyAction)
                    event.accept()
                    return True
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._handle_left_press(event.position().toPoint(), event.modifiers())
                event.accept()
                return True
            if event.type() == QEvent.Type.MouseMove:
                if self._handle_mouse_move(event.position().toPoint(), event.buttons()):
                    event.accept()
                    return True
            if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                self._drag_start_position = None
                self._drag_start_index = -1
        return super().eventFilter(watched, event)

    def rowCount(self) -> int:
        return len(self._images)

    def images(self) -> list[ImageItem]:
        return list(self._images)

    def current_index(self) -> int:
        return self._selected_index

    def selected_images(self) -> list[ImageItem]:
        return [
            self._images[index]
            for index in sorted(self._selected_indexes)
            if 0 <= index < len(self._images)
        ]

    def selected_image_ids(self) -> list[int]:
        return [image.id for image in self.selected_images()]

    def select_image_ids(self, image_ids: list[int]) -> None:
        previous_indexes, previous_index = set(self._selected_indexes), self._selected_index
        self._selected_indexes = {self._indexes_by_image_id[i] for i in image_ids if i in self._indexes_by_image_id}
        self._selected_index = min(self._selected_indexes) if self._selected_indexes else -1
        self._selection_anchor = self._selected_index
        self._update_selection_change(previous_indexes, previous_index)
        self._emit_selection()

    def set_images(
        self,
        images: list[ImageItem],
        *,
        selected_image_ids: list[int] | None = None,
        current_image_id: int | None = None,
        badges_by_image_id: dict[int, list[str]] | None = None,
    ) -> None:
        if selected_image_ids is None:
            selected_image_ids = self.selected_image_ids()
        if current_image_id is None:
            current = self.current_image()
            current_image_id = current.id if current is not None else None

        new_images = list(images)
        new_layout_signature = self._layout_keys(new_images)
        should_rebuild_layout = self._layout_signature != new_layout_signature
        if should_rebuild_layout:
            self._pending_pixmap_loads.clear()
            self._loaded_pixmap_results.clear()
        self._images = new_images
        self._layout_signature = new_layout_signature
        self._badges_by_image_id = dict(badges_by_image_id or {})
        self._rebuild_index_map()
        indexes_by_id = self._indexes_by_image_id
        self._selected_indexes = {
            indexes_by_id[image_id]
            for image_id in selected_image_ids
            if image_id in indexes_by_id
        }
        if current_image_id in indexes_by_id and indexes_by_id[current_image_id] in self._selected_indexes:
            self._selected_index = indexes_by_id[current_image_id]
        elif self._selected_indexes:
            self._selected_index = min(self._selected_indexes)
        else:
            self._selected_index = -1
        self._selection_anchor = self._selected_index
        if should_rebuild_layout:
            self._rebuild_layout()
        self.viewport().update()
        self._emit_selection()

    def set_project_references(self, references: dict[int, list[str]]) -> None:
        """Set saved membership for the active project, not temporary selection.

        Kept separate from result badges so filtering/sorting cannot erase it.
        The project controller replaces this map when project links change.
        """
        references = {image_id: list(names) for image_id, names in references.items() if names}
        if references != self._project_references:
            self._project_references = references
            self.viewport().update()

    def image_tooltip(self, image: ImageItem) -> str:
        parts = [image.file_path]
        nodes = self._project_references.get(image.id, [])
        if nodes:
            parts.append("本项目已选 · 所属节点：" + "、".join(nodes))
        badges = self._badges_by_image_id.get(image.id, [])
        if badges:
            parts.append("图片标记：" + "、".join(badges))
        return "\n".join(parts)

    def append_images(self, images: list[ImageItem]) -> None:
        if not images:
            return
        old_count = len(self._images)
        self._images.extend(images)
        self._layout_signature = self._layout_signature + self._layout_keys(images)
        for offset, image in enumerate(images):
            self._indexes_by_image_id[image.id] = old_count + offset
        if self._row_ranges:
            self._rebuild_layout_from(self._row_ranges[-1][2])
        else:
            self._rebuild_layout()
        self.viewport().update()

    def image_at(self, row: int) -> ImageItem | None:
        if row < 0 or row >= len(self._images):
            return None
        return self._images[row]

    def current_image(self) -> ImageItem | None:
        return self.image_at(self._selected_index)

    def select_image_id(self, image_id: int) -> None:
        index = self._indexes_by_image_id.get(int(image_id))
        if index is None:
            return
        self._select_single(index)
        self._ensure_index_visible(index)

    def set_thumbnail_size(self, size: int) -> None:
        next_size = max(80, min(420, size))
        if next_size == self._target_height:
            return
        self._target_height = next_size
        self._pending_pixmap_loads.clear()
        self._loaded_pixmap_results.clear()
        self._rebuild_layout()
        self.viewport().update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.viewport().width() != self._layout_viewport_width:
            self._resize_layout_timer.start()
        else:
            self._update_scroll_range()

    def _handle_scroll_value_changed(self, _value: int) -> None:
        self._schedule_viewport_update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self.viewport())
        painter.fillRect(event.rect(), self.palette().base())
        offset_y = self.verticalScrollBar().value()
        visible = self.viewport().rect()
        visible_content_top = offset_y + visible.top()
        visible_content_bottom = offset_y + visible.bottom()
        first_row = max(0, bisect_right(self._row_tops, visible_content_top) - 1)

        for row_index in range(first_row, len(self._row_ranges)):
            row_top, row_bottom, row_start, row_end = self._row_ranges[row_index]
            if row_top > visible_content_bottom:
                break
            if row_bottom < visible_content_top:
                continue
            for index in range(row_start, row_end):
                rect = self._rects[index]
                draw_rect = rect.translated(0, -offset_y)
                if not draw_rect.intersects(visible):
                    continue
                image = self._images[index]
                pixmap = self._pixmap_for(image, schedule_load=True)
                if pixmap.isNull():
                    self._draw_placeholder(painter, draw_rect, image)
                else:
                    painter.drawPixmap(draw_rect, pixmap)
                self._draw_badges(painter, draw_rect, image)
                if index in self._selected_indexes:
                    old_pen = painter.pen()
                    outer_pen = QPen(QColor("#86aaff"), 3)
                    inner_pen = QPen(QColor("#e0e9ff"), 1)
                    painter.setPen(outer_pen)
                    painter.drawRect(draw_rect.adjusted(1, 1, -2, -2))
                    painter.setPen(inner_pen)
                    painter.drawRect(draw_rect.adjusted(4, 4, -5, -5))
                    painter.save()
                    check = QRect(draw_rect.right() - 26, draw_rect.top() + 8, 18, 18)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor("#4f83ff"))
                    painter.drawRoundedRect(check, 5, 5)
                    painter.setPen(QPen(QColor("#ffffff"), 2))
                    painter.drawLine(check.left() + 4, check.top() + 9, check.left() + 7, check.top() + 12)
                    painter.drawLine(check.left() + 7, check.top() + 12, check.left() + 14, check.top() + 5)
                    painter.restore()
                    painter.setPen(old_pen)
                if index == self._selected_index and index not in self._selected_indexes:
                    pen = painter.pen()
                    focus_pen = QPen(QColor("#86aaff"), 2)
                    painter.setPen(focus_pen)
                    painter.drawRect(draw_rect.adjusted(0, 0, -1, -1))
                    painter.setPen(pen)
                self._draw_project_reference_badge(painter, draw_rect, image)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._handle_left_press(event.position().toPoint(), event.modifiers())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            index = self._index_at(event.position().toPoint())
            if index >= 0:
                self._select_single(index)
                self.imageDoubleClicked.emit(self._images[index])
                return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:
        index = self._index_at(event.pos())
        if index >= 0:
            if index not in self._selected_indexes:
                self._select_single(index)
            else:
                previous_index = self._selected_index
                self._selected_index = index
                self._selection_anchor = index
                self._update_selection_indexes({previous_index, index})
                self._emit_selection()
            self.imageContextMenuRequested.emit(self._images[index], event.globalPos())

    def mouseMoveEvent(self, event) -> None:
        if self._handle_mouse_move(event.position().toPoint(), event.buttons()):
            event.accept()
            return
        index = self._index_at(event.position().toPoint())
        if index >= 0:
            image = self._images[index]
            self.setToolTip(self.image_tooltip(image))
        else:
            self.setToolTip("")
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_position = None
            self._drag_start_index = -1
        super().mouseReleaseEvent(event)

    def dragEnterEvent(self, event) -> None:
        if self._supports_external_import_drop(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if self._supports_external_import_drop(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        if self._supports_external_import_drop(event.mimeData()):
            self.dropPayloadDropped.emit(payload_from_mime_data(event.mimeData()))
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            return
        super().dropEvent(event)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        bar = self.verticalScrollBar()
        bar.setValue(bar.value() - delta)
        event.accept()

    def keyPressEvent(self, event) -> None:
        if self._images and event.matches(QKeySequence.StandardKey.SelectAll):
            self._selection_anchor = 0
            self._select_range(len(self._images) - 1, additive=False)
            event.accept()
            return
        navigation = {Qt.Key.Key_Home, Qt.Key.Key_End, Qt.Key.Key_Left, Qt.Key.Key_Right,
                      Qt.Key.Key_Up, Qt.Key.Key_Down}
        if self._images and event.key() in navigation:
            index = max(0, self._selected_index)
            key = event.key()
            if key == Qt.Key.Key_Home:
                index = 0
            elif key == Qt.Key.Key_End:
                index = len(self._images) - 1
            elif key in {Qt.Key.Key_Left, Qt.Key.Key_Right}:
                index += -1 if key == Qt.Key.Key_Left else 1
            elif self._selected_index >= 0 and self._rects:
                current = self._rects[index].center()
                direction = -1 if key == Qt.Key.Key_Up else 1
                candidates = [(i, r.center()) for i, r in enumerate(self._rects)
                              if (r.center().y() - current.y()) * direction > 0]
                if candidates:
                    index = min(candidates, key=lambda pair: (abs(pair[1].y() - current.y()), abs(pair[1].x() - current.x())))[0]
            index = max(0, min(index, len(self._images) - 1))
            self._handle_selection_click(index, event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self._ensure_index_visible(index)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space:
            image = self.current_image()
            if image is not None:
                self.imagePreviewRequested.emit(image)
                event.accept()
                return
        super().keyPressEvent(event)

    def _handle_left_press(self, point: QPoint, modifiers: Qt.KeyboardModifier) -> None:
        self.setFocus()
        self._drag_start_position = point
        index = self._index_at(point)
        self._drag_start_index = index
        self._handle_selection_click(index, modifiers)

    def _handle_mouse_move(self, point: QPoint, buttons: Qt.MouseButton) -> bool:
        if (
            buttons & Qt.MouseButton.LeftButton
            and self._drag_start_position is not None
            and (point - self._drag_start_position).manhattanLength()
            >= QApplication.startDragDistance()
        ):
            self._start_image_drag()
            return True
        return False

    def _start_image_drag(self) -> None:
        if self._drag_start_index < 0:
            return
        if self._drag_start_index not in self._selected_indexes:
            self._select_single(self._drag_start_index)
        image_ids = self.selected_image_ids()
        if not image_ids:
            return
        mime = QMimeData()
        mime.setData(IMAGE_IDS_MIME, CollectionTreeWidget.encode_image_ids(image_ids))
        urls = self._selected_file_urls()
        if urls:
            mime.setUrls(urls)
        drag = QDrag(self.viewport())
        drag.setMimeData(mime)
        current = self.current_image()
        if current is not None:
            pixmap = self._cached_pixmap_for(current)
            if not pixmap.isNull():
                drag.setPixmap(pixmap.scaled(
                    96,
                    96,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
        drag.exec(Qt.DropAction.CopyAction, Qt.DropAction.CopyAction)
        self._drag_start_position = None
        self._drag_start_index = -1

    def _selected_file_urls(self) -> list[QUrl]:
        urls: list[QUrl] = []
        for image in self.selected_images():
            path = Path(image.file_path)
            if not image.is_missing and path.exists():
                urls.append(QUrl.fromLocalFile(str(path)))
        return urls

    @staticmethod
    def _supports_external_import_drop(mime_data: QMimeData) -> bool:
        if mime_data.hasFormat(IMAGE_IDS_MIME):
            return False
        return payload_supports_mime_data(mime_data)

    @staticmethod
    def _local_paths(urls) -> list[str]:
        paths: list[str] = []
        for url in urls:
            if url.isLocalFile():
                path = url.toLocalFile()
                if path:
                    paths.append(path)
        return paths

    def _handle_selection_click(self, index: int, modifiers: Qt.KeyboardModifier) -> None:
        if index < 0:
            self._clear_selection()
            return
        additive = bool(
            modifiers
            & (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.MetaModifier
            )
        )
        range_select = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if range_select and self._selection_anchor >= 0:
            self._select_range(index, additive=additive)
        elif additive:
            self._toggle_index(index)
        else:
            self._select_single(index)

    def _select_single(self, index: int) -> None:
        if index < 0 or index >= len(self._images):
            self._clear_selection()
            return
        if self._selected_indexes == {index} and self._selected_index == index:
            return
        previous_indexes = set(self._selected_indexes)
        previous_index = self._selected_index
        self._selected_indexes = {index}
        self._selected_index = index
        self._selection_anchor = index
        self._update_selection_change(previous_indexes, previous_index)
        self._emit_selection()

    def _toggle_index(self, index: int) -> None:
        if index < 0 or index >= len(self._images):
            return
        previous_indexes = set(self._selected_indexes)
        previous_index = self._selected_index
        if index in self._selected_indexes:
            self._selected_indexes.remove(index)
            if self._selected_index == index:
                self._selected_index = min(self._selected_indexes) if self._selected_indexes else -1
        else:
            self._selected_indexes.add(index)
            self._selected_index = index
        self._selection_anchor = index
        self._update_selection_change(previous_indexes, previous_index)
        self._emit_selection()

    def _select_range(self, index: int, *, additive: bool) -> None:
        if index < 0 or index >= len(self._images):
            return
        previous_indexes = set(self._selected_indexes)
        previous_index = self._selected_index
        start = min(self._selection_anchor, index)
        end = max(self._selection_anchor, index)
        indexes = set(range(start, end + 1))
        if additive:
            self._selected_indexes |= indexes
        else:
            self._selected_indexes = indexes
        self._selected_index = index
        self._update_selection_change(previous_indexes, previous_index)
        self._emit_selection()

    def _clear_selection(self) -> None:
        if not self._selected_indexes and self._selected_index == -1:
            return
        previous_indexes = set(self._selected_indexes)
        previous_index = self._selected_index
        self._selected_indexes.clear()
        self._selected_index = -1
        self._selection_anchor = -1
        self._update_selection_change(previous_indexes, previous_index)
        self._emit_selection()

    def _update_selection_change(self, previous_indexes: set[int], previous_index: int) -> None:
        changed_indexes = previous_indexes ^ self._selected_indexes
        if previous_index != self._selected_index:
            changed_indexes.add(previous_index)
            changed_indexes.add(self._selected_index)
        self._update_selection_indexes(changed_indexes)

    def _update_selection_indexes(self, indexes: set[int]) -> None:
        clean_indexes = {index for index in indexes if 0 <= index < len(self._rects)}
        if not clean_indexes:
            return
        if len(clean_indexes) > 160:
            self.viewport().update()
            return
        offset_y = self.verticalScrollBar().value()
        dirty_rect: QRect | None = None
        for index in clean_indexes:
            rect = self._rects[index].translated(0, -offset_y).adjusted(-6, -6, 6, 6)
            dirty_rect = rect if dirty_rect is None else dirty_rect.united(rect)
        if dirty_rect is None:
            return
        self.viewport().update(dirty_rect.intersected(self.viewport().rect()))

    def _emit_selection(self) -> None:
        self.selectionChanged.emit(self.current_image())
        self.selectionSetChanged.emit(self.selected_images())

    def _ensure_index_visible(self, index: int) -> None:
        if index < 0 or index >= len(self._rects):
            return
        rect = self._rects[index]
        bar = self.verticalScrollBar()
        top = rect.top()
        bottom = rect.bottom()
        visible_top = bar.value()
        visible_bottom = visible_top + self.viewport().height()
        if top < visible_top:
            bar.setValue(top)
        elif bottom > visible_bottom:
            bar.setValue(max(0, bottom - self.viewport().height()))

    def _index_at(self, point: QPoint) -> int:
        content_y = point.y() + self.verticalScrollBar().value()
        row_index = bisect_right(self._row_tops, content_y) - 1
        if row_index < 0 or row_index >= len(self._row_ranges):
            return -1
        row_top, row_bottom, row_start, row_end = self._row_ranges[row_index]
        if content_y < row_top or content_y >= row_bottom:
            return -1
        content_point = QPoint(point.x(), content_y)
        for index in range(row_start, row_end):
            rect = self._rects[index]
            if rect.contains(content_point):
                return index
        return -1

    def _rebuild_layout(self) -> None:
        self._resize_layout_timer.stop()
        self._rebuild_layout_from(0)

    def _run_scheduled_layout_rebuild(self) -> None:
        self._rebuild_layout()
        # The resize paint can precede the debounced layout. Repaint the whole
        # viewport now; selection/thumbnail updates only invalidate small areas.
        self.viewport().update()

    def _rebuild_layout_from(self, start_index: int) -> None:
        viewport_width = max(1, self.viewport().width())
        self._layout_viewport_width = viewport_width
        available_width = max(1, viewport_width)
        start_index = max(0, min(int(start_index), len(self._images)))
        if start_index <= 0 or not self._row_ranges or not self._rects:
            y = 0
            rects: list[QRect] = [QRect() for _ in self._images]
            row_ranges: list[tuple[int, int, int, int]] = []
            start_index = 0
        else:
            rects = self._rects
            if len(rects) < len(self._images):
                rects.extend(QRect() for _ in range(len(self._images) - len(rects)))
            elif len(rects) > len(self._images):
                del rects[len(self._images):]
            row_ranges = []
            y = 0
            for row in self._row_ranges:
                row_top, row_bottom, row_start, row_end = row
                if row_end <= start_index:
                    row_ranges.append(row)
                    y = row_bottom
                    continue
                start_index = row_start
                y = row_top
                break
            for index in range(start_index, len(rects)):
                rects[index] = QRect()
        row_indexes: list[int] = []
        row_aspects: list[float] = []
        row_width = 0.0

        def flush_row(*, justify: bool) -> None:
            nonlocal y, row_indexes, row_aspects
            if not row_indexes:
                return
            row_start = row_indexes[0]
            row_end = row_indexes[-1] + 1
            row_top = y
            y = self._layout_row(rects, row_indexes, row_aspects, y, available_width, justify)
            row_ranges.append((row_top, y, row_start, row_end))
            row_indexes = []
            row_aspects = []

        for index in range(start_index, len(self._images)):
            image = self._images[index]
            aspect = self._aspect_ratio(image)
            next_width = self._target_height * aspect
            spacing_width = self._spacing * len(row_indexes)
            if row_indexes and row_width + next_width + spacing_width >= available_width:
                flush_row(justify=True)
                row_width = 0.0
            row_indexes.append(index)
            row_aspects.append(aspect)
            row_width += next_width

        if row_indexes:
            flush_row(justify=False)

        self._rects = rects
        self._row_ranges = row_ranges
        self._row_tops = [row[0] for row in row_ranges]
        self._content_height = max(0, y - self._spacing)
        self._update_scroll_range()

    def _update_scroll_range(self) -> None:
        bar = self.verticalScrollBar()
        bar.setPageStep(self.viewport().height())
        bar.setRange(0, max(0, self._content_height - self.viewport().height()))

    def _rebuild_index_map(self) -> None:
        self._indexes_by_image_id = {image.id: index for index, image in enumerate(self._images)}

    @staticmethod
    def _layout_keys(images: list[ImageItem]) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                image.id,
                image.width,
                image.height,
                image.thumbnail_path,
                image.file_path,
                image.file_size,
                image.modified_time_ns,
                image.is_missing,
            )
            for image in images
        )

    def _layout_row(
        self,
        rects: list[QRect],
        row_indexes: list[int],
        row_aspects: list[float],
        y: int,
        available_width: int,
        justify: bool,
    ) -> int:
        if not row_indexes:
            return y
        if justify:
            total_aspect = sum(row_aspects)
            row_height = int((available_width - self._spacing * (len(row_indexes) - 1)) / total_aspect)
            row_height = max(48, min(row_height, self._target_height * 2))
        else:
            row_height = self._target_height

        x = 0
        for position, (index, aspect) in enumerate(zip(row_indexes, row_aspects)):
            if position == len(row_indexes) - 1 and justify:
                width = max(1, available_width - x)
            else:
                width = max(1, int(row_height * aspect))
            rects[index] = QRect(x, y, width, row_height)
            x += width + self._spacing
        return y + row_height + self._spacing

    def _aspect_ratio(self, image: ImageItem) -> float:
        if image.width and image.height and image.height > 0:
            return max(0.2, min(6.0, image.width / image.height))
        # Layout must stay metadata-only; image decoding is deferred to paint.
        return 1.0

    def _cached_pixmap_for(self, image: ImageItem) -> QPixmap:
        if image.is_missing:
            return QPixmap()
        key = self._pixmap_cache_key(image, image.thumbnail_path or image.file_path)
        if key is None:
            return QPixmap()
        cached = self._pixmap_cache.get(key)
        if cached is None:
            return QPixmap()
        self._pixmap_cache.move_to_end(key)
        return cached

    def _pixmap_for(self, image: ImageItem, *, schedule_load: bool = False) -> QPixmap:
        if image.is_missing:
            return QPixmap()
        source = image.thumbnail_path or image.file_path
        key = self._pixmap_cache_key(image, source)
        if key is None:
            return QPixmap()
        cached = self._pixmap_cache.get(key)
        if cached is not None:
            self._pixmap_cache.move_to_end(key)
            return cached

        if schedule_load:
            self._schedule_pixmap_load(key)
            return QPixmap()

        pixmap = QPixmap()
        if not image.is_missing:
            pixmap = self._load_scaled_pixmap(Path(source))

        self._cache_pixmap(key, pixmap)
        return pixmap

    def _load_scaled_pixmap(self, path: Path) -> QPixmap:
        image = self._load_scaled_image(path, max(256, int(self._target_height * 3)))
        return QPixmap.fromImage(image) if not image.isNull() else QPixmap()

    @staticmethod
    def _load_scaled_image(path: Path, max_side: int) -> QImage:
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        max_side = max(256, int(max_side))
        size = reader.size()
        if size.isValid() and max(size.width(), size.height()) > max_side:
            scaled_size = QSize(size.width(), size.height())
            scaled_size.scale(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio)
            reader.setScaledSize(scaled_size)
        return reader.read()

    def _pixmap_cache_key(self, image: ImageItem, source: str | None) -> PixmapCacheKey | None:
        if not source:
            return None
        max_side = max(256, int(self._target_height * 3))
        return (source, int(image.modified_time_ns or 0), int(image.file_size or 0), max_side)

    def _pixmap_keys_for_images(self, images: list[ImageItem]) -> set[PixmapCacheKey]:
        keys: set[PixmapCacheKey] = set()
        for image in images:
            if image.is_missing:
                continue
            key = self._pixmap_cache_key(image, image.thumbnail_path or image.file_path)
            if key is not None:
                keys.add(key)
        return keys

    def _schedule_pixmap_load(self, key: PixmapCacheKey) -> None:
        if key in self._pending_pixmap_loads:
            return
        if len(self._pending_pixmap_loads) >= self._pending_load_limit:
            return
        self._pending_pixmap_loads.add(key)
        source, _mtime, _size, max_side = key
        self._thread_pool.start(
            _PixmapLoadTask(key, Path(source), max_side, self._pixmap_load_signals)
        )

    def _handle_async_pixmap_loaded(self, key: object, image: object) -> None:
        cache_key = key
        if not isinstance(cache_key, tuple) or len(cache_key) != 4:
            return
        self._pending_pixmap_loads.discard(cache_key)
        if not isinstance(image, QImage):
            return
        self._loaded_pixmap_results[cache_key] = image
        self._schedule_loaded_pixmap_flush()

    def _schedule_loaded_pixmap_flush(self, *, delay_ms: int = 0) -> None:
        if self._loaded_pixmap_flush_pending:
            return
        self._loaded_pixmap_flush_pending = True
        QTimer.singleShot(delay_ms, self._flush_loaded_pixmap_results)

    def _flush_loaded_pixmap_results(self) -> None:
        self._loaded_pixmap_flush_pending = False
        processed = 0
        while self._loaded_pixmap_results and processed < self._loaded_pixmap_flush_batch_size:
            cache_key, image = self._loaded_pixmap_results.popitem(last=False)
            pixmap = QPixmap.fromImage(image) if not image.isNull() else QPixmap()
            self._cache_pixmap(cache_key, pixmap)
            processed += 1
        if processed:
            self._schedule_viewport_update()
        if self._loaded_pixmap_results:
            self._schedule_loaded_pixmap_flush(delay_ms=16)

    def _schedule_viewport_update(self) -> None:
        if self._viewport_update_pending:
            return
        self._viewport_update_pending = True
        QTimer.singleShot(0, self._run_scheduled_viewport_update)

    def _run_scheduled_viewport_update(self) -> None:
        self._viewport_update_pending = False
        self.viewport().update()

    def _cache_pixmap(self, key: PixmapCacheKey, pixmap: QPixmap) -> None:
        existing_bytes = self._pixmap_cache_bytes.pop(key, 0)
        self._pixmap_cache_total_bytes = max(0, self._pixmap_cache_total_bytes - existing_bytes)
        self._pixmap_cache[key] = pixmap
        self._pixmap_cache.move_to_end(key)
        pixmap_bytes = self._estimated_pixmap_bytes(pixmap)
        self._pixmap_cache_bytes[key] = pixmap_bytes
        self._pixmap_cache_total_bytes += pixmap_bytes
        while (
            len(self._pixmap_cache) > self._cache_limit
            or self._pixmap_cache_total_bytes > self._cache_byte_limit
        ):
            old_key, _old_pixmap = self._pixmap_cache.popitem(last=False)
            old_bytes = self._pixmap_cache_bytes.pop(old_key, 0)
            self._pixmap_cache_total_bytes = max(0, self._pixmap_cache_total_bytes - old_bytes)

    @staticmethod
    def _estimated_pixmap_bytes(pixmap: QPixmap) -> int:
        if pixmap.isNull():
            return 0
        return max(0, pixmap.width()) * max(0, pixmap.height()) * 4

    def _draw_placeholder(self, painter: QPainter, rect: QRect, image: ImageItem) -> None:
        painter.fillRect(rect, theme_color("canvas"))
        painter.setPen(theme_color("text"))
        if image.is_missing:
            text = "文件丢失"
        elif is_supported_video(image.file_path):
            text = f"视频\n{image.file_name}"
        else:
            text = "无缩略图"
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_project_reference_badge(self, painter: QPainter, rect: QRect, image: ImageItem) -> None:
        if image.id not in self._project_references or rect.width() < 12:
            return
        painter.save()
        painter.setClipRect(rect)
        metrics = painter.fontMetrics()
        available = max(1, rect.width() - 8)
        text = "✓ 本项目已选"
        if metrics.horizontalAdvance(text) + 10 > available:
            text = "✓ 已选"
        if metrics.horizontalAdvance(text) + 10 > available:
            text = "✓"
        width = min(available, metrics.horizontalAdvance(text) + 10)
        badge = QRect(rect.left() + 4, rect.top() + 4, width, metrics.height() + 6)
        painter.setBrush(QColor(66, 48, 22, 240))
        painter.setPen(QPen(QColor("#d5a14c"), 1))
        painter.drawRoundedRect(badge, 3, 3)
        painter.setPen(QColor("#ffe2a6"))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def _draw_badges(self, painter: QPainter, rect: QRect, image: ImageItem) -> None:
        badges = self._badges_by_image_id.get(image.id, [])
        if not badges:
            return
        from eidory.core.project_state import reference_badge_text
        text = reference_badge_text(badges)
        metrics = painter.fontMetrics()
        text = metrics.elidedText(text, Qt.TextElideMode.ElideRight, max(24, rect.width() - 14))
        padding_x = 5
        padding_y = 3
        badge_width = min(rect.width() - 8, metrics.horizontalAdvance(text) + padding_x * 2)
        badge_height = metrics.height() + padding_y * 2
        badge_rect = QRect(
            rect.left() + 4,
            rect.bottom() - badge_height - 4,
            max(1, badge_width),
            badge_height,
        )
        painter.fillRect(badge_rect, QColor(16, 20, 26, 205))
        painter.setPen(QColor("#f3f6fb"))
        painter.drawText(
            badge_rect.adjusted(padding_x, 0, -padding_x, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            text,
        )
