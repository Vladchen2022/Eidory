from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QImage, QImageReader, QKeySequence, QPainter, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QMenu,
)

from eidory.models import ImageItem


BOARD_WIDTH = 5200
BOARD_HEIGHT = 3600
MIN_ZOOM = 0.15
MAX_ZOOM = 16.0
GRID_SIZE = 36
PIXMAP_CACHE_LIMIT = 512
MAX_SOURCE_PIXMAP_SIDE = 2200
MAX_BOARD_THUMBNAIL_SIDE = 560
LAYOUT_CHANGED_IDLE_MS = 450
DEFAULT_LAYOUT_LEFT = 80.0
DEFAULT_LAYOUT_TOP = 110.0
DEFAULT_LAYOUT_ROW_WIDTH = 1600.0
DEFAULT_LAYOUT_GAP_X = 48.0
DEFAULT_LAYOUT_GAP_Y = 64.0


_BOARD_PIXMAP_CACHE: OrderedDict[tuple[str, int, int, int], QPixmap] = OrderedDict()


def _cached_pixmap(
    path: Path,
    *,
    max_side: int = MAX_SOURCE_PIXMAP_SIDE,
    modified_time_ns: int | None = None,
    file_size: int | None = None,
) -> QPixmap:
    max_side = max(128, int(max_side))
    if modified_time_ns is None or file_size is None:
        try:
            stat = path.stat()
        except OSError:
            return QPixmap()
        modified_time_ns = int(stat.st_mtime_ns)
        file_size = int(stat.st_size)
    key = (str(path), int(modified_time_ns), int(file_size), max_side)
    cached = _BOARD_PIXMAP_CACHE.get(key)
    if cached is not None:
        _BOARD_PIXMAP_CACHE.move_to_end(key)
        return QPixmap(cached)

    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid() and max(size.width(), size.height()) > max_side:
        scaled_size = QSize(size.width(), size.height())
        scaled_size.scale(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio)
        reader.setScaledSize(scaled_size)
    image = reader.read()
    pixmap = QPixmap.fromImage(image) if not image.isNull() else QPixmap()
    if pixmap.isNull():
        return pixmap
    _BOARD_PIXMAP_CACHE[key] = QPixmap(pixmap)
    _BOARD_PIXMAP_CACHE.move_to_end(key)
    while len(_BOARD_PIXMAP_CACHE) > PIXMAP_CACHE_LIMIT:
        _BOARD_PIXMAP_CACHE.popitem(last=False)
    return pixmap


class BoardImageItem(QGraphicsPixmapItem):
    MIN_DISPLAY_WIDTH = 90.0
    RESIZE_MARGIN = 12.0

    def __init__(
        self,
        *,
        image_id: int,
        pixmap: QPixmap,
        file_path: str,
        badges: list[str] | None = None,
        on_layout_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(pixmap)
        self.image_id = image_id
        self._source_pixmap = pixmap
        self._badges = self._normalize_badges(badges)
        self._on_layout_changed = on_layout_changed
        self._display_width = float(max(1, pixmap.width()))
        self._display_height = float(max(1, pixmap.height()))
        self._resize_zone = ""
        self._resizing = False
        self._resize_start_scene = QPointF()
        self._resize_start_width = self._display_width
        self._aspect_ratio = pixmap.width() / max(1, pixmap.height())
        self._pinned = False
        self._flipped = False
        self._grayscale = False
        self.setToolTip(file_path)
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.setCacheMode(QGraphicsItem.CacheMode.ItemCoordinateCache)
        self.setAcceptHoverEvents(True)
        self._apply_flags()

    @property
    def display_width(self) -> float:
        return self._display_width

    @property
    def display_height(self) -> float:
        return self._display_height

    @property
    def badge_text(self) -> str:
        if not self._badges:
            return ""
        if len(self._badges) == 1:
            return self._badges[0]
        return f"{self._badges[0]} +{len(self._badges) - 1}"

    @staticmethod
    def _normalize_badges(badges: list[str] | None) -> list[str]:
        normalized: list[str] = []
        for badge in badges or []:
            text = str(badge).strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    def set_display_size(self, width: float, _height: float | None = None) -> None:
        pixmap = self.pixmap()
        if pixmap.isNull():
            return
        width = max(self.MIN_DISPLAY_WIDTH, float(width))
        previous_width = self._display_width
        previous_height = self._display_height
        self.prepareGeometryChange()
        self._display_width = width
        self._display_height = width / max(0.001, self._aspect_ratio)
        self.setScale(self._display_width / max(1, pixmap.width()))
        if (
            abs(previous_width - self._display_width) > 0.001
            or abs(previous_height - self._display_height) > 0.001
        ):
            self._notify_layout_changed()

    def set_pinned(self, pinned: bool) -> None:
        pinned = bool(pinned)
        if self._pinned == pinned:
            return
        self._pinned = pinned
        self._apply_flags()
        self._notify_layout_changed()

    def toggle_pinned(self) -> bool:
        self.set_pinned(not self._pinned)
        return self._pinned

    def is_pinned(self) -> bool:
        return self._pinned

    def set_flipped(self, flipped: bool) -> None:
        flipped = bool(flipped)
        if self._flipped == flipped:
            return
        self._flipped = flipped
        self._refresh_display_pixmap()
        self._notify_layout_changed()

    def toggle_flipped(self) -> None:
        self.set_flipped(not self._flipped)

    def is_flipped(self) -> bool:
        return self._flipped

    def set_grayscale(self, grayscale: bool) -> None:
        grayscale = bool(grayscale)
        if self._grayscale == grayscale:
            return
        self._grayscale = grayscale
        self._refresh_display_pixmap()
        self._notify_layout_changed()

    def toggle_grayscale(self) -> None:
        self.set_grayscale(not self._grayscale)

    def is_grayscale(self) -> bool:
        return self._grayscale

    def _apply_flags(self) -> None:
        flags = (
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        if not self._pinned:
            flags |= QGraphicsItem.GraphicsItemFlag.ItemIsMovable
        self.setFlags(flags)

    def _refresh_display_pixmap(self) -> None:
        pixmap = self._source_pixmap
        if pixmap.isNull():
            return
        if self._grayscale:
            image = pixmap.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(image.convertToFormat(QImage.Format.Format_ARGB32))
        if self._flipped:
            pixmap = pixmap.transformed(QTransform().scale(-1, 1), Qt.TransformationMode.SmoothTransformation)
        current_width = self._display_width
        self.setPixmap(pixmap)
        self.set_display_size(current_width)

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._notify_layout_changed()
        return result

    def _notify_layout_changed(self) -> None:
        if self._on_layout_changed is not None:
            self._on_layout_changed()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        text = self.badge_text
        if not text:
            return
        rect = self.boundingRect()
        transform = painter.worldTransform()
        bottom_left = transform.map(rect.bottomLeft())
        bottom_right = transform.map(rect.bottomRight())
        top_left = transform.map(rect.topLeft())
        image_width = abs(bottom_right.x() - bottom_left.x())
        image_height = abs(bottom_left.y() - top_left.y())
        if image_width < 36 or image_height < 18:
            return
        painter.save()
        painter.resetTransform()
        font = painter.font()
        font.setPointSizeF(10.0)
        font.setBold(True)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        max_text_width = max(28, int(min(image_width - 12, 220)))
        text = metrics.elidedText(text, Qt.TextElideMode.ElideRight, max_text_width)
        padding_x = 6.0
        padding_y = 3.0
        text_width = metrics.horizontalAdvance(text)
        text_height = metrics.height()
        badge_width = min(image_width, text_width + padding_x * 2)
        badge_height = text_height + padding_y * 2
        badge_rect = QRectF(
            bottom_left.x(),
            bottom_left.y() - badge_height,
            badge_width,
            badge_height,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(20, 24, 30, 220))
        painter.drawRect(badge_rect)
        painter.setPen(QColor("#eef2f8"))
        painter.drawText(
            QRectF(
                badge_rect.left() + padding_x,
                badge_rect.top() + padding_y,
                badge_rect.width() - padding_x * 2,
                text_height,
            ),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            text,
        )
        painter.restore()

    def hoverMoveEvent(self, event) -> None:
        if self._pinned:
            self.unsetCursor()
            super().hoverMoveEvent(event)
            return
        self._resize_zone = self._resize_zone_at(event.pos())
        if self._resize_zone in {"corner", "right", "bottom"}:
            if self._resize_zone == "right":
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif self._resize_zone == "bottom":
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            else:
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if not self._resizing:
            self._resize_zone = ""
            self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if self._pinned:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            zone = self._resize_zone_at(event.pos())
            if zone in {"corner", "right", "bottom"}:
                self._resize_zone = zone
                self._resizing = True
                self._resize_start_scene = event.scenePos()
                self._resize_start_width = self._display_width
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._resizing:
            delta = event.scenePos() - self._resize_start_scene
            if self._resize_zone == "bottom":
                new_width = self._resize_start_width + delta.y() * self._aspect_ratio
            elif self._resize_zone == "corner":
                new_width = self._resize_start_width + max(delta.x(), delta.y() * self._aspect_ratio)
            else:
                new_width = self._resize_start_width + delta.x()
            self.set_display_size(new_width)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._resizing:
            self._resizing = False
            self._resize_zone = ""
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _resize_zone_at(self, pos: QPointF) -> str:
        rect = self.boundingRect()
        margin = self.RESIZE_MARGIN / max(0.001, self.scale())
        near_right = rect.right() - pos.x() <= margin
        near_bottom = rect.bottom() - pos.y() <= margin
        if near_right and near_bottom:
            return "corner"
        if near_right:
            return "right"
        if near_bottom:
            return "bottom"
        return ""


class BoardPlaceholderItem(QGraphicsRectItem):
    def __init__(
        self,
        rect: QRectF,
        *,
        on_layout_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(rect)
        self._on_layout_changed = on_layout_changed

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._notify_layout_changed()
        return result

    def _notify_layout_changed(self) -> None:
        if self._on_layout_changed is not None:
            self._on_layout_changed()


class ProjectBoardView(QGraphicsView):
    imageDoubleClicked = Signal(int)
    selectionChanged = Signal(list)
    layoutChanged = Signal()
    removeImagesRequested = Signal(list)
    undoRemovalRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#20262d"))
        self._image_items: dict[int, QGraphicsItem] = {}
        self._title_item: QGraphicsTextItem | None = None
        self._zoom = 1.0
        self._space_pan_active = False
        self._space_pan_moved = False
        self._panning = False
        self._last_pan_pos = QPoint()
        self._view_mode = "fit_all"
        self._last_fit_selection_ids: tuple[int, ...] = ()
        self._refit_timer_pending = False
        self._suppress_layout_changed = False
        self._layout_change_pending = False
        self._layout_change_timer = QTimer(self)
        self._layout_change_timer.setSingleShot(True)
        self._layout_change_timer.setInterval(LAYOUT_CHANGED_IDLE_MS)
        self._layout_change_timer.timeout.connect(self._emit_pending_layout_changed)
        self._scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self._scene.selectionChanged.connect(self._emit_selection_changed)

    def set_images(
        self,
        images: list[ImageItem],
        *,
        title: str = "",
        layout_payload: dict[str, Any] | None = None,
        badges_by_image_id: dict[int, list[str]] | None = None,
    ) -> None:
        normalized_badges = {
            int(image_id): [str(badge) for badge in badges]
            for image_id, badges in (badges_by_image_id or {}).items()
        }
        self.setUpdatesEnabled(False)
        previous_scene_signal_state = self._scene.blockSignals(True)
        self._suppress_layout_changed = True
        try:
            self._scene.clear()
            self._image_items.clear()
            self._scene.setSceneRect(0, 0, BOARD_WIDTH, BOARD_HEIGHT)
            self.resetTransform()
            self._zoom = 1.0
            self._view_mode = "fit_all"
            self._last_fit_selection_ids = ()

            stored_items = {}
            if isinstance(layout_payload, dict):
                raw_items = layout_payload.get("items", {})
                if isinstance(raw_items, dict):
                    stored_items = raw_items

            self._title_item = None
            if title:
                self._title_item = self._scene.addText(title)
                self._title_item.setDefaultTextColor(QColor("#eef2f8"))
                font = self._title_item.font()
                font.setPointSize(18)
                font.setBold(True)
                self._title_item.setFont(font)
                self._title_item.setPos(56, 36)

            default_x = DEFAULT_LAYOUT_LEFT
            default_y = DEFAULT_LAYOUT_TOP
            row_height = 0.0
            row_right = DEFAULT_LAYOUT_LEFT + DEFAULT_LAYOUT_ROW_WIDTH
            stored_positions: set[tuple[int, int]] = set()
            has_duplicate_stored_position = False
            for image in images:
                payload = stored_items.get(str(image.id), {})
                if not isinstance(payload, dict):
                    continue
                position = (
                    int(round(_safe_float(payload.get("x"), DEFAULT_LAYOUT_LEFT))),
                    int(round(_safe_float(payload.get("y"), DEFAULT_LAYOUT_TOP))),
                )
                if position in stored_positions:
                    has_duplicate_stored_position = True
                    break
                stored_positions.add(position)
            if (
                has_duplicate_stored_position
                or any(not isinstance(stored_items.get(str(image.id)), dict) for image in images)
            ):
                stored_bottom = DEFAULT_LAYOUT_TOP
                for image in images:
                    payload = stored_items.get(str(image.id), {})
                    if not isinstance(payload, dict):
                        continue
                    y = _safe_float(payload.get("y"), DEFAULT_LAYOUT_TOP)
                    height = max(80.0, _safe_float(payload.get("height"), 160.0))
                    stored_bottom = max(stored_bottom, y + height)
                if stored_bottom > DEFAULT_LAYOUT_TOP:
                    default_y = stored_bottom + DEFAULT_LAYOUT_GAP_Y
            placed_positions: set[tuple[int, int]] = set()
            for index, image in enumerate(images):
                item_payload = stored_items.get(str(image.id), {})
                pixmap = self._pixmap_for(
                    image,
                    prefer_source=False,
                    max_side=MAX_BOARD_THUMBNAIL_SIDE,
                )
                item_width, item_height = self._default_item_size(pixmap)
                if default_x > DEFAULT_LAYOUT_LEFT and default_x + item_width > row_right:
                    default_x = DEFAULT_LAYOUT_LEFT
                    default_y += row_height + DEFAULT_LAYOUT_GAP_Y
                    row_height = 0.0
                if isinstance(item_payload, dict):
                    x = _safe_float(item_payload.get("x"), default_x)
                    y = _safe_float(item_payload.get("y"), default_y)
                    item_width = max(80.0, _safe_float(item_payload.get("width"), item_width))
                    item_height = max(80.0, _safe_float(item_payload.get("height"), item_height))
                    position = (int(round(x)), int(round(y)))
                    if position in placed_positions:
                        item_payload = {}
                        x = float(default_x)
                        y = float(default_y)
                else:
                    x = float(default_x)
                    y = float(default_y)
                self._add_image_item(
                    image,
                    pixmap,
                    x,
                    y,
                    item_width,
                    item_height,
                    normalized_badges.get(image.id, []),
                )
                self._apply_item_payload_state(image.id, item_payload)
                placed_positions.add((int(round(x)), int(round(y))))
                row_height = max(row_height, item_height)
                default_x += item_width + DEFAULT_LAYOUT_GAP_X

            if not images:
                text = self._scene.addText("当前节点没有保存图片")
                text.setDefaultTextColor(QColor("#d8dee9"))
                text.setPos(80, 110)
        finally:
            self.setUpdatesEnabled(True)
            self._scene.blockSignals(previous_scene_signal_state)
            self._finish_layout_reset()

        self.selectionChanged.emit([])
        if images:
            self._schedule_refit_current_view_mode()

    def fit_all_images(self) -> None:
        self._view_mode = "fit_all"
        self._last_fit_selection_ids = ()
        self._fit_items_to_view(self._visible_image_items(), margin=80.0)

    def hide_selected_items(self) -> int:
        items = self._selected_image_items()
        for item in items:
            item.setVisible(False)
            item.setSelected(False)
        if items:
            self._schedule_layout_changed()
        return len(items)

    def show_all_items(self) -> None:
        changed = any(not item.isVisible() for item in self._image_items.values())
        for item in self._image_items.values():
            item.setVisible(True)
        if changed:
            self._schedule_layout_changed()
        self.fit_all_images()

    def toggle_selected_pinned(self) -> int:
        changed = 0
        for item in self._selected_image_items():
            if isinstance(item, BoardImageItem):
                item.toggle_pinned()
                changed += 1
        return changed

    def toggle_selected_flipped(self) -> int:
        changed = 0
        for item in self._selected_image_items():
            if isinstance(item, BoardImageItem):
                item.toggle_flipped()
                changed += 1
        return changed

    def toggle_selected_grayscale(self) -> int:
        changed = 0
        for item in self._selected_image_items():
            if isinstance(item, BoardImageItem):
                item.toggle_grayscale()
                changed += 1
        return changed

    def set_title_visible(self, visible: bool) -> None:
        if self._title_item is not None:
            self._title_item.setVisible(visible)

    def layout_payload(self) -> dict[str, Any]:
        self.flush_pending_layout_change()
        items: dict[str, dict[str, float]] = {}
        for image_id, item in self._image_items.items():
            if isinstance(item, BoardImageItem):
                width = item.display_width
                height = item.display_height
            else:
                rect = item.mapToScene(item.boundingRect()).boundingRect()
                width = rect.width()
                height = rect.height()
            items[str(image_id)] = {
                "x": float(item.pos().x()),
                "y": float(item.pos().y()),
                "width": float(width),
                "height": float(height),
                "visible": bool(item.isVisible()),
            }
            if isinstance(item, BoardImageItem):
                items[str(image_id)].update(
                    {
                        "pinned": item.is_pinned(),
                        "flipped": item.is_flipped(),
                        "grayscale": item.is_grayscale(),
                    }
                )
        return {"version": 1, "items": items}

    def fit_visible_images(self) -> None:
        selected_items = self._selected_image_items()
        if selected_items:
            self._fit_selected_items(selected_items)
        else:
            self.fit_all_images()

    def toggle_fit_selection_or_restore(self) -> None:
        selected_ids = tuple(sorted(self.selected_image_ids()))
        if (
            selected_ids
            and self._view_mode == "fit_selection"
            and selected_ids == self._last_fit_selection_ids
        ):
            self.fit_all_images()
            return
        selected_items = self._selected_image_items()
        if selected_items:
            self._fit_selected_items(selected_items)
            return
        self.fit_all_images()

    def reset_view(self) -> None:
        self.resetTransform()
        self._zoom = 1.0
        self._view_mode = "manual"
        self._last_fit_selection_ids = ()
        self.centerOn(BOARD_WIDTH / 2, BOARD_HEIGHT / 2)

    def wheelEvent(self, event) -> None:
        modifiers = event.modifiers()
        zoom_modifier = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        if modifiers & zoom_modifier:
            raw_delta = event.angleDelta().y()
            if raw_delta == 0:
                raw_delta = event.pixelDelta().y()
            delta = max(-160, min(160, raw_delta))
            if delta == 0:
                event.accept()
                return
            factor = 1.12 if delta > 0 else 1 / 1.12
            self._view_mode = "manual"
            self._last_fit_selection_ids = ()
            self._zoom_by(factor)
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event) -> None:
        undo_modifier = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        if event.matches(QKeySequence.StandardKey.Undo) or (
            event.key() == Qt.Key.Key_Z and bool(event.modifiers() & undo_modifier)
        ):
            self.undoRemovalRequested.emit()
            event.accept()
            return
        if event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}:
            image_ids = self.selected_image_ids()
            if image_ids:
                self.removeImagesRequested.emit(image_ids)
                event.accept()
                return
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan_active = True
            self._space_pan_moved = False
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            if event.key() in {Qt.Key.Key_Equal, Qt.Key.Key_Plus}:
                self._zoom_by(1.2)
                event.accept()
                return
            if event.key() in {Qt.Key.Key_Minus, Qt.Key.Key_Underscore}:
                self._zoom_by(1 / 1.2)
                event.accept()
                return
            if event.key() == Qt.Key.Key_0:
                self.reset_view()
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            should_fit = self._space_pan_active and not self._space_pan_moved
            self._space_pan_active = False
            self._panning = False
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self.viewport().unsetCursor()
            if should_fit:
                self.toggle_fit_selection_or_restore()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if self._space_pan_active and event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._last_pan_pos = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            current_pos = event.position().toPoint()
            delta = current_pos - self._last_pan_pos
            if abs(delta.x()) > 1 or abs(delta.y()) > 1:
                self._space_pan_moved = True
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._last_pan_pos = current_pos
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._panning and event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            if self._space_pan_active:
                self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        painter.fillRect(rect, QColor("#252b33"))
        minor_pen = QPen(QColor("#303842"))
        minor_pen.setWidth(0)
        major_pen = QPen(QColor("#3a4450"))
        major_pen.setWidth(0)

        left = int(rect.left()) - (int(rect.left()) % GRID_SIZE)
        top = int(rect.top()) - (int(rect.top()) % GRID_SIZE)
        right = int(rect.right())
        bottom = int(rect.bottom())

        x = left
        while x <= right:
            painter.setPen(major_pen if (x // GRID_SIZE) % 4 == 0 else minor_pen)
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
            x += GRID_SIZE

        y = top
        while y <= bottom:
            painter.setPen(major_pen if (y // GRID_SIZE) % 4 == 0 else minor_pen)
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)
            y += GRID_SIZE

    def mouseDoubleClickEvent(self, event) -> None:
        image_id = self._image_id_at(event.position().toPoint())
        if image_id is not None:
            self._select_image_id(image_id)
            self.toggle_fit_selection_or_restore()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:
        image_id = self._image_id_at(event.pos())
        if image_id is None:
            super().contextMenuEvent(event)
            return
        if image_id not in self.selected_image_ids():
            self._select_image_id(image_id)
        menu = QMenu(self)
        remove_action = menu.addAction("从当前项目移除")
        remove_action.setEnabled(bool(self.selected_image_ids()))
        chosen = menu.exec(event.globalPos())
        if chosen == remove_action:
            image_ids = self.selected_image_ids()
            if image_ids:
                self.removeImagesRequested.emit(image_ids)
            event.accept()
            return
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._view_mode in {"fit_all", "fit_selection"}:
            self._schedule_refit_current_view_mode()

    def selected_image_ids(self) -> list[int]:
        image_ids: list[int] = []
        seen: set[int] = set()
        for item in self._scene.selectedItems():
            image_id = self._image_id_for_item(item)
            if image_id is not None and image_id not in seen:
                image_ids.append(image_id)
                seen.add(image_id)
        return image_ids

    def _zoom_by(self, factor: float) -> None:
        next_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self._zoom * factor))
        if abs(next_zoom - self._zoom) < 0.001:
            return
        scale_factor = next_zoom / max(0.001, self._zoom)
        self.scale(scale_factor, scale_factor)
        self._zoom = next_zoom

    def _fit_selected_items(self, items: list[QGraphicsItem]) -> None:
        if not items:
            return
        self._view_mode = "fit_selection"
        self._last_fit_selection_ids = tuple(sorted(self.selected_image_ids()))
        self._fit_items_to_view(items, margin=10.0)

    def _refit_current_view_mode(self) -> None:
        if self._view_mode == "fit_selection":
            selected_items = self._selected_image_items()
            if selected_items:
                self._fit_items_to_view(selected_items, margin=10.0)
                return
        if self._view_mode == "fit_all":
            self._fit_items_to_view(self._visible_image_items(), margin=80.0)

    def _schedule_refit_current_view_mode(self) -> None:
        if self._refit_timer_pending:
            return
        self._refit_timer_pending = True
        QTimer.singleShot(0, self._run_scheduled_refit)

    def _run_scheduled_refit(self) -> None:
        self._refit_timer_pending = False
        self._refit_current_view_mode()

    def _finish_layout_reset(self) -> None:
        self._suppress_layout_changed = False
        self._layout_change_pending = False
        self._layout_change_timer.stop()

    def _schedule_layout_changed(self) -> None:
        if self._suppress_layout_changed or not self._image_items:
            return
        self._layout_change_pending = True
        self._layout_change_timer.start()

    def flush_pending_layout_change(self) -> bool:
        if not self._layout_change_pending:
            return False
        self._layout_change_timer.stop()
        self._emit_pending_layout_changed()
        return True

    def _emit_pending_layout_changed(self) -> None:
        if not self._layout_change_pending:
            return
        self._layout_change_pending = False
        if self._suppress_layout_changed or not self._image_items:
            return
        self.layoutChanged.emit()

    def _add_image_item(
        self,
        image: ImageItem,
        pixmap: QPixmap,
        x: float,
        y: float,
        width: float,
        height: float,
        badges: list[str],
    ) -> None:
        if pixmap.isNull():
            frame = BoardPlaceholderItem(
                QRectF(0, 0, width, height),
                on_layout_changed=self._schedule_layout_changed,
            )
            frame.setBrush(QColor("#171b21"))
            frame.setPen(QPen(QColor("#4b5563"), 1))
            item: QGraphicsItem = frame
            label = QGraphicsTextItem(image.file_name, frame)
            label.setDefaultTextColor(QColor("#d8dee9"))
            label.setTextWidth(width - 18)
            label.setPos(9, height / 2 - 20)
        else:
            item = BoardImageItem(
                image_id=image.id,
                pixmap=pixmap,
                file_path=image.file_path,
                badges=badges,
                on_layout_changed=self._schedule_layout_changed,
            )
            item.set_display_size(width, height)
        item.setPos(x, y)
        if not isinstance(item, BoardImageItem):
            item.setFlags(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            )
        item.setData(0, int(image.id))
        item.setZValue(len(self._image_items) + 1)
        self._scene.addItem(item)
        self._image_items[image.id] = item

    def _apply_item_payload_state(self, image_id: int, item_payload: object) -> None:
        if not isinstance(item_payload, dict):
            return
        item = self._image_items.get(image_id)
        if item is None:
            return
        if "visible" in item_payload:
            item.setVisible(bool(item_payload.get("visible")))
        if isinstance(item, BoardImageItem):
            if "pinned" in item_payload:
                item.set_pinned(bool(item_payload.get("pinned")))
            if "flipped" in item_payload:
                item.set_flipped(bool(item_payload.get("flipped")))
            if "grayscale" in item_payload:
                item.set_grayscale(bool(item_payload.get("grayscale")))

    def _selected_image_items(self) -> list[QGraphicsItem]:
        items: list[QGraphicsItem] = []
        seen: set[int] = set()
        for item in self._scene.selectedItems():
            image_id = self._image_id_for_item(item)
            if image_id is None or image_id in seen:
                continue
            image_item = self._image_items.get(image_id)
            if image_item is not None:
                items.append(image_item)
                seen.add(image_id)
        return items

    def _select_image_id(self, image_id: int) -> None:
        previous_ids = self.selected_image_ids()
        self._scene.blockSignals(True)
        try:
            for item in self._scene.selectedItems():
                item.setSelected(False)
            target = self._image_items.get(image_id)
            if target is not None:
                target.setSelected(True)
        finally:
            self._scene.blockSignals(False)
        selected_ids = self.selected_image_ids()
        if selected_ids != previous_ids:
            self.selectionChanged.emit(selected_ids)

    def _emit_selection_changed(self) -> None:
        self.selectionChanged.emit(self.selected_image_ids())

    def _visible_image_items(self) -> list[QGraphicsItem]:
        return [item for item in self._image_items.values() if item.isVisible()]

    def _fit_items_to_view(self, items: list[QGraphicsItem], *, margin: float) -> None:
        if not items:
            return
        rect = self._items_rect(items)
        if not rect.isValid() or rect.width() <= 1 or rect.height() <= 1:
            return
        viewport_rect = self.viewport().rect()
        available_width = max(1.0, float(viewport_rect.width()) - margin * 2)
        available_height = max(1.0, float(viewport_rect.height()) - margin * 2)
        next_zoom = min(available_width / rect.width(), available_height / rect.height())
        next_zoom = max(MIN_ZOOM, min(MAX_ZOOM, next_zoom))
        self.resetTransform()
        self.scale(next_zoom, next_zoom)
        self._zoom = next_zoom
        self.centerOn(rect.center())

    def _image_id_at(self, viewport_pos) -> int | None:
        item = self.itemAt(viewport_pos)
        return self._image_id_for_item(item)

    def _image_id_for_item(self, item: QGraphicsItem | None) -> int | None:
        current: QGraphicsItem | None = item
        while current is not None:
            image_id = current.data(0)
            if image_id is not None:
                return int(image_id)
            current = current.parentItem()
        return None

    @staticmethod
    def _items_rect(items: list[QGraphicsItem]) -> QRectF:
        rect = QRectF()
        for item in items:
            item_rect = item.mapToScene(item.boundingRect()).boundingRect()
            rect = item_rect if rect.isNull() else rect.united(item_rect)
        return rect

    @staticmethod
    def _pixmap_for(
        image: ImageItem,
        *,
        prefer_source: bool = False,
        max_side: int = MAX_SOURCE_PIXMAP_SIDE,
    ) -> QPixmap:
        candidates = (
            [image.file_path, image.thumbnail_path]
            if prefer_source
            else [image.thumbnail_path, image.file_path]
        )
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if candidate == image.file_path:
                pixmap = _cached_pixmap(
                    path,
                    max_side=max_side,
                    modified_time_ns=image.modified_time_ns,
                    file_size=image.file_size,
                )
            else:
                pixmap = _cached_pixmap(path, max_side=max_side)
            if not pixmap.isNull():
                return pixmap
        return QPixmap()

    @staticmethod
    def _default_item_size(pixmap: QPixmap) -> tuple[float, float]:
        target_height = 160.0
        if pixmap.isNull():
            return 220.0, target_height
        ratio = pixmap.width() / max(1, pixmap.height())
        width = max(120.0, min(360.0, target_height * ratio))
        return width, target_height


def _safe_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
