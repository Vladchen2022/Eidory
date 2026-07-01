from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageOps
from PySide6.QtCore import QEvent, QObject, QPoint, QRectF, QRunnable, QSize, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QIcon,
    QImage,
    QImageReader,
    QKeyEvent,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from eidory.core.image_loader import open_local_image
from eidory.core.linetop_processor import LineTopSettings, render_linetop_image
from eidory.core.media_types import is_supported_video
from eidory.core.metadata_store import MetadataStore
from eidory.models import ImageItem


PREVIEW_ICON_BUTTON_SIZE = QSize(32, 26)
PREVIEW_ICON_SIZE = QSize(18, 18)
INLINE_SOURCE_PREVIEW_MAX_BYTES = 512 * 1024
SOURCE_PREVIEW_REFINE_DELAY_MS = 320
LINETOP_PANEL_WIDTH = 330
LINETOP_RENDER_CACHE_LIMIT = 12


def _load_linetop_source_image(
    image_path: str,
    *,
    max_width: int | None = None,
    max_height: int | None = None,
    grayscale: bool = False,
    mirror_horizontal: bool = False,
) -> Image.Image:
    with open_local_image(image_path) as source:
        if max_width is not None and max_height is not None and source.format == "JPEG":
            source.draft("RGB", (max_width, max_height))
        loaded = ImageOps.exif_transpose(source)
        if max_width is not None and max_height is not None:
            loaded.thumbnail((max(1, max_width), max(1, max_height)), Image.Resampling.LANCZOS)
        if mirror_horizontal:
            loaded = ImageOps.mirror(loaded)
        if grayscale:
            loaded = ImageOps.grayscale(loaded).convert("RGBA")
        return loaded.convert("RGBA")


def _qimage_from_pillow(image: Image.Image) -> QImage:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(
        data,
        rgba.width,
        rgba.height,
        rgba.width * 4,
        QImage.Format.Format_RGBA8888,
    )
    return qimage.copy()


class _LineTopRenderSignals(QObject):
    loaded = Signal(object, object, str)


class _LineTopRenderTask(QRunnable):
    def __init__(
        self,
        *,
        token: object,
        image_path: str,
        max_width: int,
        max_height: int,
        settings: LineTopSettings,
        grayscale: bool,
        mirror_horizontal: bool,
        signals: _LineTopRenderSignals,
    ) -> None:
        super().__init__()
        self.token = token
        self.image_path = image_path
        self.max_width = max_width
        self.max_height = max_height
        self.settings = settings
        self.grayscale = grayscale
        self.mirror_horizontal = mirror_horizontal
        self.signals = signals

    def run(self) -> None:
        try:
            source = _load_linetop_source_image(
                self.image_path,
                max_width=self.max_width,
                max_height=self.max_height,
                grayscale=self.grayscale,
                mirror_horizontal=self.mirror_horizontal,
            )
            rendered = render_linetop_image(source, self.settings)
            qimage = _qimage_from_pillow(rendered)
            error = ""
        except Exception as exc:
            qimage = QImage()
            error = str(exc)
        try:
            self.signals.loaded.emit(self.token, qimage, error)
        except RuntimeError:
            return


def _load_preview_qimage(image_path: str, max_width: int, max_height: int) -> QImage:
    if max_width <= 0 or max_height <= 0:
        return QImage()
    path = Path(image_path)
    if not path.exists():
        return QImage()
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    source_size = reader.size()
    if source_size.isValid():
        scaled_size = QSize(source_size.width(), source_size.height())
        scaled_size.scale(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio)
        if scaled_size.width() < source_size.width() or scaled_size.height() < source_size.height():
            reader.setScaledSize(scaled_size)
    qimage = reader.read()
    if not qimage.isNull():
        return qimage
    try:
        with open_local_image(image_path) as image:
            if image.format == "JPEG":
                image.draft("RGB", (max_width, max_height))
            image = ImageOps.exif_transpose(image)
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            image = image.convert("RGBA")
            data = image.tobytes("raw", "RGBA")
            qimage = QImage(
                data,
                image.width,
                image.height,
                image.width * 4,
                QImage.Format.Format_RGBA8888,
            )
            return qimage.copy()
    except Exception:
        return QImage()


class _PreviewSourceLoadSignals(QObject):
    loaded = Signal(object, object, bool)


class _PreviewSourceLoadTask(QRunnable):
    def __init__(
        self,
        *,
        token: tuple[int, str, int, int, int, int],
        image_path: str,
        fallback_path: str | None,
        max_width: int,
        max_height: int,
        signals: _PreviewSourceLoadSignals,
    ) -> None:
        super().__init__()
        self.token = token
        self.image_path = image_path
        self.fallback_path = fallback_path
        self.max_width = max_width
        self.max_height = max_height
        self.signals = signals

    def run(self) -> None:
        fallback_used = False
        qimage = _load_preview_qimage(self.image_path, self.max_width, self.max_height)
        if qimage.isNull() and self.fallback_path:
            qimage = _load_preview_qimage(self.fallback_path, self.max_width, self.max_height)
            fallback_used = not qimage.isNull()
        try:
            self.signals.loaded.emit(self.token, qimage, fallback_used)
        except RuntimeError:
            return


def _preview_transform_icon(kind: str) -> QIcon:
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    line_color = QColor("#e4e9f1")
    fill_color = QColor("#e4e9f1")
    painter.setPen(QPen(line_color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.setBrush(Qt.BrushStyle.NoBrush)

    if kind == "flip":
        painter.drawLine(12, 4, 12, 20)
        painter.drawLine(5, 7, 10, 12)
        painter.drawLine(5, 17, 10, 12)
        painter.drawLine(19, 7, 14, 12)
        painter.drawLine(19, 17, 14, 12)
    elif kind == "grayscale":
        painter.setBrush(fill_color)
        painter.drawPie(5, 5, 14, 14, 90 * 16, 180 * 16)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(5, 5, 14, 14)
        painter.drawLine(12, 5, 12, 19)

    painter.end()
    return QIcon(pixmap)


def _configure_preview_icon_button(button: QPushButton, icon_name: str, name: str, tooltip: str) -> None:
    button.setText("")
    button.setIcon(_preview_transform_icon(icon_name))
    button.setIconSize(PREVIEW_ICON_SIZE)
    button.setFixedSize(PREVIEW_ICON_BUTTON_SIZE)
    button.setToolTip(tooltip)
    button.setAccessibleName(name)
    button.setAccessibleDescription(tooltip)


class PreviewNavigator(QWidget):
    centerRequested = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thumbnail = QPixmap()
        self._source_width = 1
        self._source_height = 1
        self._visible_rect = QRectF()
        self._image_rect = QRectF()
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.hide()

    def set_state(
        self,
        *,
        thumbnail: QPixmap,
        source_width: int,
        source_height: int,
        visible_rect: QRectF,
    ) -> None:
        if thumbnail.isNull() or source_width <= 0 or source_height <= 0:
            self.hide()
            return
        self._thumbnail = thumbnail
        self._source_width = max(1, source_width)
        self._source_height = max(1, source_height)
        self._visible_rect = visible_rect
        self._image_rect = QRectF(6, 6, thumbnail.width(), thumbnail.height())
        wanted_width = thumbnail.width() + 12
        wanted_height = thumbnail.height() + 12
        if self.width() != wanted_width or self.height() != wanted_height:
            self.resize(wanted_width, wanted_height)
        self.show()
        self.raise_()
        self.update()

    def paintEvent(self, event) -> None:
        if self._thumbnail.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        outer = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor(84, 94, 108, 210), 1))
        painter.setBrush(QColor(22, 25, 30, 190))
        painter.drawRoundedRect(outer, 5, 5)

        painter.drawPixmap(self._image_rect.toRect(), self._thumbnail)
        view_rect = self._mapped_visible_rect()
        if not view_rect.isEmpty():
            painter.setPen(QPen(QColor(255, 255, 255, 235), 2))
            painter.setBrush(QColor(255, 255, 255, 28))
            painter.drawRoundedRect(view_rect.adjusted(1, 1, -1, -1), 2, 2)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._request_center(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._request_center(event)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def _mapped_visible_rect(self) -> QRectF:
        if self._image_rect.isEmpty():
            return QRectF()
        scale_x = self._image_rect.width() / self._source_width
        scale_y = self._image_rect.height() / self._source_height
        visible = self._visible_rect.intersected(QRectF(0, 0, self._source_width, self._source_height))
        return QRectF(
            self._image_rect.left() + visible.left() * scale_x,
            self._image_rect.top() + visible.top() * scale_y,
            visible.width() * scale_x,
            visible.height() * scale_y,
        )

    def _request_center(self, event) -> None:
        position = event.position() if hasattr(event, "position") else event.pos()
        if self._image_rect.isEmpty() or not self._image_rect.contains(position):
            return
        x_ratio = (position.x() - self._image_rect.left()) / self._image_rect.width()
        y_ratio = (position.y() - self._image_rect.top()) / self._image_rect.height()
        self.centerRequested.emit(
            max(0.0, min(1.0, x_ratio)),
            max(0.0, min(1.0, y_ratio)),
        )


class PreviewImageView(QGraphicsView):
    zoomChanged = Signal(float, bool)
    doubleClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._message_item = QGraphicsTextItem()
        self._scene.addItem(self._pixmap_item)
        self._scene.addItem(self._message_item)
        self.setScene(self._scene)

        self._fit_to_window = True
        self._zoom_factor = 1.0
        self._original_width = 1
        self._original_height = 1
        self._source_width = 1
        self._source_height = 1
        self._has_pixmap = False
        self._navigator_thumb = QPixmap()

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundBrush(QColor("#2d3138"))
        self.setStyleSheet("background:#2d3138;color:#d8dee9;")
        self.setMinimumHeight(420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._message_item.setDefaultTextColor(QColor("#d8dee9"))
        self._smooth_timer = QTimer(self)
        self._smooth_timer.setSingleShot(True)
        self._smooth_timer.setInterval(90)
        self._smooth_timer.timeout.connect(self._restore_smooth_transform)
        self._navigator = PreviewNavigator(self.viewport())
        self._navigator.centerRequested.connect(self._center_on_navigator_ratio)
        self.horizontalScrollBar().valueChanged.connect(lambda _value: self._update_navigator())
        self.verticalScrollBar().valueChanged.connect(lambda _value: self._update_navigator())
        self.set_message("未选择图片")

    @property
    def fit_to_window(self) -> bool:
        return self._fit_to_window

    @property
    def zoom_factor(self) -> float:
        return self._zoom_factor

    def set_message(self, text: str) -> None:
        self._smooth_timer.stop()
        self._has_pixmap = False
        self._fit_to_window = True
        self._zoom_factor = 1.0
        self._navigator_thumb = QPixmap()
        self._navigator.hide()
        self._pixmap_item.setPixmap(QPixmap())
        self._pixmap_item.hide()
        self._message_item.setPlainText(text)
        self._message_item.show()
        self.resetTransform()
        self._scene.setSceneRect(QRectF(0, 0, max(1, self.viewport().width()), max(1, self.viewport().height())))
        self._center_message()
        self._update_drag_mode()

    def clear_pixmap(self) -> None:
        self._smooth_timer.stop()
        self._has_pixmap = False
        self._fit_to_window = True
        self._zoom_factor = 1.0
        self._navigator_thumb = QPixmap()
        self._navigator.hide()
        self._pixmap_item.setPixmap(QPixmap())
        self._pixmap_item.hide()
        self._message_item.hide()
        self.resetTransform()
        self._scene.setSceneRect(QRectF(0, 0, max(1, self.viewport().width()), max(1, self.viewport().height())))
        self._update_drag_mode()

    def set_pixmap(
        self,
        pixmap: QPixmap,
        *,
        original_width: int | None,
        original_height: int | None,
        fit_to_window: bool,
        zoom_factor: float,
    ) -> None:
        if pixmap.isNull():
            self.set_message("无法预览")
            return
        center_ratio = self._current_center_ratio()
        self._has_pixmap = True
        self._message_item.hide()
        self._pixmap_item.show()
        self._pixmap_item.setPixmap(pixmap)
        self._source_width = max(1, pixmap.width())
        self._source_height = max(1, pixmap.height())
        self._original_width = max(1, int(original_width or pixmap.width()))
        self._original_height = max(1, int(original_height or pixmap.height()))
        self._navigator_thumb = pixmap.scaled(
            170,
            130,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._scene.setSceneRect(QRectF(0, 0, self._source_width, self._source_height))
        if fit_to_window:
            self.fit_image_to_window(emit=False)
        else:
            self.set_zoom_factor(zoom_factor, emit=False)
            self._center_on_ratio(center_ratio)
        self._update_drag_mode()
        self._schedule_navigator_update()

    def fit_image_to_window(self, *, emit: bool = True) -> None:
        if not self._has_pixmap:
            return
        self._smooth_timer.stop()
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._fit_to_window = True
        self._zoom_factor = self.fit_zoom_factor()
        self._apply_zoom_transform(reset=True)
        self.centerOn(self._pixmap_item)
        self._update_drag_mode()
        if emit:
            self.zoomChanged.emit(self._zoom_factor, self._fit_to_window)

    def set_actual_size(self, *, emit: bool = True) -> None:
        if not self._has_pixmap:
            return
        self._smooth_timer.stop()
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._fit_to_window = False
        self._zoom_factor = 1.0
        self._apply_zoom_transform(reset=True)
        self.centerOn(self._pixmap_item)
        self._update_drag_mode()
        if emit:
            self.zoomChanged.emit(self._zoom_factor, self._fit_to_window)

    def set_zoom_factor(self, zoom_factor: float, *, emit: bool = True) -> None:
        if not self._has_pixmap:
            return
        self._fit_to_window = False
        self._zoom_factor = max(0.1, min(8.0, zoom_factor))
        self._apply_zoom_transform(reset=True)
        self._update_drag_mode()
        if emit:
            self.zoomChanged.emit(self._zoom_factor, self._fit_to_window)

    def fit_zoom_factor(self) -> float:
        if not self._has_pixmap:
            return 1.0
        viewport_width = max(1, self.viewport().width() - 2)
        viewport_height = max(1, self.viewport().height() - 2)
        return max(
            0.1,
            min(8.0, min(viewport_width / self._original_width, viewport_height / self._original_height)),
        )

    def zoom_by(self, wheel_delta: int) -> None:
        if not self._has_pixmap or wheel_delta == 0:
            return
        if self._fit_to_window:
            self._zoom_factor = self.fit_zoom_factor()
            self._fit_to_window = False
        step = 1.15 if wheel_delta > 0 else 1 / 1.15
        self._zoom_factor = max(0.1, min(8.0, self._zoom_factor * step))
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self._apply_zoom_transform(reset=False)
        self._update_drag_mode()
        self._smooth_timer.start()
        self.zoomChanged.emit(self._zoom_factor, self._fit_to_window)

    def can_pan(self) -> bool:
        return (
            self._has_pixmap
            and not self._fit_to_window
            and (self.horizontalScrollBar().maximum() > 0 or self.verticalScrollBar().maximum() > 0)
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._message_item.isVisible():
            self._scene.setSceneRect(QRectF(0, 0, max(1, self.viewport().width()), max(1, self.viewport().height())))
            self._center_message()
        elif self._fit_to_window:
            self.fit_image_to_window()
        else:
            self._position_navigator()
            self._schedule_navigator_update()

    def wheelEvent(self, event) -> None:
        self.zoom_by(event.angleDelta().y())
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _apply_zoom_transform(self, *, reset: bool) -> None:
        target_view_scale = self._view_scale_for_zoom(self._zoom_factor)
        if reset:
            self.resetTransform()
            self.scale(target_view_scale, target_view_scale)
            return
        current_view_scale = self.transform().m11()
        if current_view_scale <= 0:
            self.resetTransform()
            self.scale(target_view_scale, target_view_scale)
            return
        factor = target_view_scale / current_view_scale
        self.scale(factor, factor)

    def _view_scale_for_zoom(self, zoom_factor: float) -> float:
        return max(0.001, zoom_factor * self._original_width / self._source_width)

    def _restore_smooth_transform(self) -> None:
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._update_navigator()
        self.viewport().update()

    def _update_drag_mode(self) -> None:
        can_pan = self.can_pan()
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if can_pan else QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.CursorShape.OpenHandCursor if can_pan else Qt.CursorShape.ArrowCursor)
        self._update_navigator()
        self._schedule_navigator_update()

    def _center_message(self) -> None:
        rect = self._scene.sceneRect()
        text_rect = self._message_item.boundingRect()
        self._message_item.setPos(
            rect.center().x() - text_rect.width() / 2,
            rect.center().y() - text_rect.height() / 2,
        )

    def _current_center_ratio(self) -> tuple[float, float] | None:
        if not self._has_pixmap or self._source_width <= 0 or self._source_height <= 0:
            return None
        center = self.mapToScene(self.viewport().rect().center())
        return (
            max(0.0, min(1.0, center.x() / self._source_width)),
            max(0.0, min(1.0, center.y() / self._source_height)),
        )

    def _center_on_ratio(self, center_ratio: tuple[float, float] | None) -> None:
        if center_ratio is None:
            self.centerOn(self._pixmap_item)
            return
        self.centerOn(
            center_ratio[0] * self._source_width,
            center_ratio[1] * self._source_height,
        )

    def _center_on_navigator_ratio(self, x_ratio: float, y_ratio: float) -> None:
        if not self._has_pixmap:
            return
        self.centerOn(x_ratio * self._source_width, y_ratio * self._source_height)
        self._update_navigator()

    def _visible_source_rect(self) -> QRectF:
        if not self._has_pixmap:
            return QRectF()
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        return visible.intersected(QRectF(0, 0, self._source_width, self._source_height))

    def _position_navigator(self) -> None:
        if not hasattr(self, "_navigator") or self._navigator.isHidden():
            return
        margin = 14
        self._navigator.move(
            max(0, self.viewport().width() - self._navigator.width() - margin),
            max(0, self.viewport().height() - self._navigator.height() - margin),
        )

    def _update_navigator(self) -> None:
        if not hasattr(self, "_navigator"):
            return
        if not self.can_pan() or self._navigator_thumb.isNull():
            self._navigator.hide()
            return
        self._navigator.set_state(
            thumbnail=self._navigator_thumb,
            source_width=self._source_width,
            source_height=self._source_height,
            visible_rect=self._visible_source_rect(),
        )
        self._position_navigator()

    def _schedule_navigator_update(self) -> None:
        QTimer.singleShot(0, self._update_navigator)


class ImagePreviewDialog(QDialog):
    imageChanged = Signal(object)
    favoriteChanged = Signal(object)
    feedbackSaved = Signal(object, str)
    indexRemoved = Signal(object)

    def __init__(
        self,
        *,
        images: list[ImageItem],
        start_index: int,
        store: MetadataStore,
        semantic_query: str | None,
        model_name: str,
        model_revision: str,
        embedding_dim: int,
        thumbnail_dir: Path | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.images = list(images)
        self.index = max(0, min(start_index, len(self.images) - 1))
        self.store = store
        self.semantic_query = semantic_query
        self.model_name = model_name
        self.model_revision = model_revision
        self.embedding_dim = embedding_dim
        self.thumbnail_dir = Path(thumbnail_dir) if thumbnail_dir is not None else None
        self.fit_to_window = True
        self.zoom_factor = 1.0
        self.grayscale_preview = False
        self.mirrored_preview = False
        self._panning = False
        self._pan_start_pos = QPoint()
        self._pan_start_horizontal = 0
        self._pan_start_vertical = 0
        self._shortcuts: list[QShortcut] = []
        self._app_event_filter_installed = False
        self._preview_base_key: tuple[int, str, int, int] | None = None
        self._preview_base_pixmap = QPixmap()
        self._preview_base_is_fallback = False
        self._preview_variant_cache: dict[tuple[bool, bool], QPixmap] = {}
        self._preview_source_pending_token: tuple[int, str, int, int, int, int] | None = None
        self._preview_source_running_token: tuple[int, str, int, int, int, int] | None = None
        self._preview_source_queued_request: tuple[
            tuple[int, str, int, int, int, int],
            str,
            str | None,
            int,
            int,
        ] | None = None
        self._preview_source_signals = _PreviewSourceLoadSignals()
        self._preview_source_signals.loaded.connect(self._handle_preview_source_loaded)
        self._preview_source_thread_pool = QThreadPool.globalInstance()
        self._linetop_settings = LineTopSettings()
        self._linetop_show_original_compare = False
        self._linetop_render_cache: dict[tuple[object, ...], QPixmap] = {}
        self._linetop_render_pending_token: tuple[object, ...] | None = None
        self._linetop_render_running_token: tuple[object, ...] | None = None
        self._linetop_queued_render_request: tuple[
            tuple[object, ...],
            str,
            int,
            int,
            LineTopSettings,
            bool,
            bool,
        ] | None = None
        self._linetop_render_signals = _LineTopRenderSignals()
        self._linetop_render_signals.loaded.connect(self._handle_linetop_render_loaded)
        self._linetop_thread_pool = QThreadPool.globalInstance()
        self._linetop_render_timer = QTimer(self)
        self._linetop_render_timer.setSingleShot(True)
        self._linetop_render_timer.setInterval(120)
        self._linetop_render_timer.timeout.connect(self._render_current_image)
        self._zoom_refine_timer = QTimer(self)
        self._zoom_refine_timer.setSingleShot(True)
        self._zoom_refine_timer.setInterval(90)
        self._zoom_refine_timer.timeout.connect(self._render_current_image)
        self._preview_refine_timer = QTimer(self)
        self._preview_refine_timer.setSingleShot(True)
        self._preview_refine_timer.setInterval(SOURCE_PREVIEW_REFINE_DELAY_MS)
        self._preview_refine_timer.timeout.connect(self._render_current_image)

        self.setWindowTitle("Eidory 预览")
        self.resize(1200, 820)
        self.setMinimumSize(720, 520)

        self.image_view = PreviewImageView()
        self.image_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_view.viewport().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_view.customContextMenuRequested.connect(self._show_image_context_menu)
        self.image_view.viewport().customContextMenuRequested.connect(self._show_image_context_menu)
        self.image_view.doubleClicked.connect(self._fit_image_to_window)
        self.image_view.zoomChanged.connect(self._handle_image_zoom_changed)

        self.video_widget: QVideoWidget | None = None
        self.video_player: QMediaPlayer | None = None
        self.video_audio_output: QAudioOutput | None = None

        self.preview_stack = QStackedWidget()
        self.preview_stack.addWidget(self.image_view)
        self.advanced_panel = self._build_linetop_panel()
        self.advanced_panel.hide()

        self.info_label = QLabel("-")
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.previous_button = QPushButton("上一张")
        self.next_button = QPushButton("下一张")
        self.fit_button = QPushButton("适应窗口")
        self.actual_size_button = QPushButton("100%")
        self.grayscale_button = QPushButton()
        self.grayscale_button.setCheckable(True)
        _configure_preview_icon_button(
            self.grayscale_button,
            "grayscale",
            "黑白",
            "把当前图片切换为黑白显示",
        )
        self.mirror_button = QPushButton()
        self.mirror_button.setCheckable(True)
        _configure_preview_icon_button(
            self.mirror_button,
            "flip",
            "左右翻转",
            "左右镜像翻转当前图片",
        )
        self.video_play_pause_button = QPushButton("播放")
        self.video_position_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_position_slider.setRange(0, 0)
        self.video_time_label = QLabel("00:00 / 00:00")
        self.favorite_checkbox = QCheckBox("收藏")
        self.compare_toggle_button = QPushButton("对比切换")
        self.compare_toggle_button.setCheckable(True)
        self.compare_toggle_button.setToolTip("在原图和高级处理结果之间切换")
        self.compare_toggle_button.setEnabled(False)
        self.save_render_button = QPushButton("保存为")
        self.save_render_button.setToolTip("把当前高级处理结果另存为 PNG，不改动图库源图")
        self.save_render_button.setEnabled(False)
        self.advanced_toggle_button = QPushButton("高级")
        self.advanced_toggle_button.setCheckable(True)
        self.advanced_toggle_button.setToolTip("显示或隐藏高级功能区（Tab）")

        self.feedback_group = QButtonGroup(self)
        self.feedback_group.setExclusive(True)
        self.feedback_relevant_button = QPushButton("相关")
        self.feedback_irrelevant_button = QPushButton("不相关")
        self.feedback_ignored_button = QPushButton("忽略")
        self.feedback_buttons = {
            "relevant": self.feedback_relevant_button,
            "irrelevant": self.feedback_irrelevant_button,
            "ignored": self.feedback_ignored_button,
        }
        for button in self.feedback_buttons.values():
            button.setCheckable(True)
            button.hide()
            self.feedback_group.addButton(button)

        nav_buttons = [
            self.previous_button,
            self.next_button,
            self.fit_button,
            self.actual_size_button,
            self.grayscale_button,
            self.mirror_button,
            self.compare_toggle_button,
            self.save_render_button,
            self.advanced_toggle_button,
        ]
        for button in [*nav_buttons, *self.feedback_buttons.values()]:
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.open_original_button = QPushButton("打开源文件")
        self.reveal_button = QPushButton("Finder 中显示")
        self.copy_image_button = QPushButton("复制图片")
        self.copy_path_button = QPushButton("复制路径")
        self.remove_index_button = QPushButton("移除索引")

        self.video_controls_widget = QWidget()
        video_controls = QHBoxLayout(self.video_controls_widget)
        video_controls.setContentsMargins(0, 0, 0, 0)
        video_controls.addWidget(self.video_play_pause_button)
        video_controls.addWidget(self.video_position_slider, 1)
        video_controls.addWidget(self.video_time_label)

        controls = QHBoxLayout()
        controls.addWidget(self.previous_button)
        controls.addWidget(self.next_button)
        controls.addWidget(self.fit_button)
        controls.addWidget(self.actual_size_button)
        controls.addWidget(self.grayscale_button)
        controls.addWidget(self.mirror_button)
        controls.addSpacing(12)
        controls.addWidget(self.favorite_checkbox)
        controls.addSpacing(12)
        controls.addStretch(1)
        controls.addWidget(self.compare_toggle_button)
        controls.addWidget(self.save_render_button)
        controls.addWidget(self.advanced_toggle_button)

        layout = QVBoxLayout(self)
        preview_row = QHBoxLayout()
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.setSpacing(8)
        preview_row.addWidget(self.preview_stack, 1)
        preview_row.addWidget(self.advanced_panel)
        layout.addLayout(preview_row, 1)
        layout.addWidget(self.video_controls_widget)
        layout.addWidget(self.info_label)
        layout.addLayout(controls)

        self.previous_button.clicked.connect(lambda: self._move(-1))
        self.next_button.clicked.connect(lambda: self._move(1))
        self.fit_button.clicked.connect(self._fit_image_to_window)
        self.actual_size_button.clicked.connect(self._actual_size_image)
        self.grayscale_button.toggled.connect(self._set_grayscale_preview)
        self.mirror_button.toggled.connect(self._set_mirrored_preview)
        self.favorite_checkbox.toggled.connect(self._save_favorite)
        self.compare_toggle_button.toggled.connect(self._set_linetop_compare_original)
        self.save_render_button.clicked.connect(self._save_linetop_render_as)
        self.advanced_toggle_button.toggled.connect(self._set_linetop_panel_visible)
        self.feedback_relevant_button.clicked.connect(lambda: self._save_feedback("relevant"))
        self.feedback_irrelevant_button.clicked.connect(lambda: self._save_feedback("irrelevant"))
        self.feedback_ignored_button.clicked.connect(lambda: self._save_feedback("ignored"))
        self.open_original_button.clicked.connect(self._open_original)
        self.reveal_button.clicked.connect(self._reveal_in_finder)
        self.copy_image_button.clicked.connect(self._copy_current_image)
        self.copy_path_button.clicked.connect(self._copy_path)
        self.remove_index_button.clicked.connect(self._remove_current_index)
        self.video_play_pause_button.clicked.connect(self._toggle_video_playback)
        self.video_position_slider.sliderMoved.connect(self._seek_video)
        self._install_shortcuts()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._app_event_filter_installed = True

        self._refresh()

    def closeEvent(self, event) -> None:
        self._linetop_render_timer.stop()
        self._linetop_render_pending_token = None
        self._linetop_queued_render_request = None
        self._preview_source_pending_token = None
        self._preview_source_queued_request = None
        self._zoom_refine_timer.stop()
        self._preview_refine_timer.stop()
        if self._app_event_filter_installed:
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._app_event_filter_installed = False
        self._stop_video()
        self._clear_preview_pixmap_cache()
        self.image_view.clear_pixmap()
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._schedule_fit_to_window()
        image = self.current_image()
        if image is not None and not is_supported_video(image.file_path):
            self._preview_refine_timer.start()

    def current_image(self) -> ImageItem | None:
        if not self.images:
            return None
        return self.images[self.index]

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        image = self.current_image()
        if self.fit_to_window and (image is None or not is_supported_video(image.file_path)):
            self._render_current_image(use_thumbnail_first=True)
            self._preview_refine_timer.start()

    def eventFilter(self, watched, event) -> bool:
        if (
            self._is_preview_surface(watched)
            and event.type() == QEvent.Type.MouseButtonDblClick
            and event.button() == Qt.MouseButton.LeftButton
        ):
            image = self.current_image()
            if image is not None and not is_supported_video(image.file_path):
                self._fit_image_to_window()
            else:
                self.close()
            event.accept()
            return True
        if event.type() in {QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride}:
            if isinstance(event, QKeyEvent) and self._event_belongs_to_preview(watched):
                if self._is_close_shortcut(event):
                    if event.type() == QEvent.Type.KeyPress:
                        self.close()
                    event.accept()
                    return True
                if event.key() == Qt.Key.Key_Space:
                    if event.type() == QEvent.Type.KeyPress:
                        self._handle_space_pressed()
                    event.accept()
                    return True
                if event.key() == Qt.Key.Key_Tab:
                    if event.type() == QEvent.Type.KeyPress:
                        self._toggle_linetop_panel()
                    event.accept()
                    return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._is_close_shortcut(event):
            self.close()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space:
            self._handle_space_pressed()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Tab:
            self._toggle_linetop_panel()
            event.accept()
            return
        if event.key() in {Qt.Key.Key_Right, Qt.Key.Key_Down}:
            self._move(1)
            event.accept()
            return
        if event.key() in {Qt.Key.Key_Left, Qt.Key.Key_Up}:
            self._move(-1)
            event.accept()
            return
        super().keyPressEvent(event)

    def _move(self, delta: int) -> None:
        if not self.images:
            return
        next_index = self.index + delta
        if next_index < 0 or next_index >= len(self.images):
            return
        self.index = next_index
        self.fit_to_window = True
        self.zoom_factor = 1.0
        self._refresh()

    def _refresh(self) -> None:
        self._preview_refine_timer.stop()
        image = self.current_image()
        if image is None:
            self.info_label.setText("-")
            self._set_linetop_compare_original(False, rerender=False)
            self._clear_preview_pixmap_cache()
            self.image_view.set_message("未选择图片")
            self.video_controls_widget.hide()
            self.fit_button.setEnabled(False)
            self.actual_size_button.setEnabled(False)
            self.grayscale_button.setEnabled(False)
            self.mirror_button.setEnabled(False)
            self.copy_image_button.setEnabled(False)
            self.compare_toggle_button.setEnabled(False)
            self.save_render_button.setEnabled(False)
            self.advanced_toggle_button.setEnabled(False)
            self.remove_index_button.setEnabled(False)
            self._stop_video()
            return

        self.previous_button.setEnabled(self.index > 0)
        self.next_button.setEnabled(self.index < len(self.images) - 1)
        self.favorite_checkbox.blockSignals(True)
        self.favorite_checkbox.setChecked(image.is_favorite)
        self.favorite_checkbox.blockSignals(False)
        self.remove_index_button.setEnabled(True)
        self._refresh_feedback_buttons(image)
        self._update_info(image)
        if is_supported_video(image.file_path):
            self._set_linetop_panel_visible(False)
            self._render_current_video(image)
        else:
            self.advanced_toggle_button.setEnabled(True)
            self.compare_toggle_button.setEnabled(self._linetop_preview_active(image))
            self.save_render_button.setEnabled(self._linetop_preview_active(image))
            self._render_current_image(use_thumbnail_first=True)
            self._preview_refine_timer.start()
        self.imageChanged.emit(image)

    def _handle_space_pressed(self) -> None:
        image = self.current_image()
        if image is None:
            return
        if is_supported_video(image.file_path):
            self._toggle_video_playback()
            return
        self.close()

    def _event_belongs_to_preview(self, watched) -> bool:
        if watched is self:
            return True
        if isinstance(watched, QWidget) and self.isAncestorOf(watched):
            return True
        return QApplication.activeWindow() is self

    @staticmethod
    def _is_close_shortcut(event: QKeyEvent) -> bool:
        modifiers = event.modifiers()
        return (
            event.key() == Qt.Key.Key_Escape
            or
            event.matches(QKeySequence.StandardKey.Close)
            or (
                event.key() == Qt.Key.Key_W
                and bool(
                    modifiers
                    & (
                        Qt.KeyboardModifier.MetaModifier
                        | Qt.KeyboardModifier.ControlModifier
                    )
                )
            )
        )

    def _update_info(self, image: ImageItem) -> None:
        dimensions = "-"
        if image.width and image.height:
            dimensions = f"{image.width} x {image.height}"
        if image.duration_ms is not None:
            duration = self._format_video_time(image.duration_ms)
            dimensions = duration if dimensions == "-" else f"{dimensions} / {duration}"
        score = "-" if image.score is None else f"{image.score:.4f}"
        zoom_text = (
            "视频播放"
            if is_supported_video(image.file_path)
            else "适应窗口" if self.fit_to_window else f"{int(self.zoom_factor * 100)}%"
        )
        self.info_label.setText(
            f"{self.index + 1} / {len(self.images)}    "
            f"{image.file_name}    {dimensions}    相似度 {score}    {zoom_text}"
        )

    def _render_current_image(self, *, use_thumbnail_first: bool = False) -> None:
        image = self.current_image()
        if image is None:
            return
        self._stop_video()
        self.video_controls_widget.hide()
        self.preview_stack.setCurrentWidget(self.image_view)
        self.fit_button.setEnabled(True)
        self.actual_size_button.setEnabled(True)
        self.grayscale_button.setEnabled(True)
        self.mirror_button.setEnabled(True)
        self.copy_image_button.setEnabled(True)
        advanced_active = self._linetop_preview_active(image)
        self.compare_toggle_button.setEnabled(advanced_active)
        self.save_render_button.setEnabled(advanced_active)
        max_width, max_height = self._render_bounds()
        if advanced_active and not self._linetop_show_original_compare:
            self._request_linetop_render(image, max_width, max_height, use_thumbnail_first=use_thumbnail_first)
            return
        pixmap = QPixmap()
        if use_thumbnail_first:
            pixmap = self._cached_preview_base_pixmap(image)
            if pixmap.isNull():
                pixmap = self._load_quick_preview_pixmap(image, max_width, max_height)
            if pixmap.isNull() and image.file_size <= INLINE_SOURCE_PREVIEW_MAX_BYTES:
                pixmap = self._load_preview_source_pixmap(image, max_width, max_height)
            if pixmap.isNull():
                self.image_view.set_message("加载预览...")
                self._update_info(image)
                return
        else:
            pixmap = self._load_preview_source_pixmap(image, max_width, max_height)
        if pixmap.isNull():
            message = "加载预览..." if self._preview_source_pending_token is not None else "无法预览"
            self.image_view.set_message(message)
            return
        self.image_view.set_pixmap(
            pixmap,
            original_width=image.width,
            original_height=image.height,
            fit_to_window=self.fit_to_window,
            zoom_factor=self.zoom_factor,
        )
        self._update_info(image)

    def _ensure_video_preview(self) -> tuple[QVideoWidget, QMediaPlayer]:
        if self.video_widget is None:
            self.video_widget = QVideoWidget()
            self.video_widget.setMinimumHeight(420)
            self.video_widget.setStyleSheet("background:#2d3138;")
            self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.video_widget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
            self.video_widget.installEventFilter(self)
            self.preview_stack.addWidget(self.video_widget)
        if self.video_player is None:
            self.video_player = QMediaPlayer(self)
            self.video_audio_output = QAudioOutput(self)
            self.video_player.setAudioOutput(self.video_audio_output)
            self.video_player.positionChanged.connect(self._update_video_position)
            self.video_player.durationChanged.connect(self._update_video_duration)
            self.video_player.playbackStateChanged.connect(self._update_video_play_button)
        self.video_player.setVideoOutput(self.video_widget)
        return self.video_widget, self.video_player

    def _render_current_video(self, image: ImageItem) -> None:
        video_widget, video_player = self._ensure_video_preview()
        self.fit_button.setEnabled(False)
        self.actual_size_button.setEnabled(False)
        self.grayscale_button.setEnabled(False)
        self.mirror_button.setEnabled(False)
        self.copy_image_button.setEnabled(False)
        self._stop_pan()
        self.preview_stack.setCurrentWidget(video_widget)
        self.video_controls_widget.show()
        self.video_position_slider.setValue(0)
        self.video_position_slider.setRange(0, 0)
        self.video_time_label.setText("00:00 / 00:00")
        video_player.stop()
        video_player.setSource(QUrl())

        path = Path(image.file_path)
        if image.is_missing or not path.exists():
            self.info_label.setText(f"{image.file_name}    视频文件不存在")
            self.video_play_pause_button.setEnabled(False)
            return

        self.video_play_pause_button.setEnabled(True)
        video_player.setSource(QUrl.fromLocalFile(str(path)))
        video_player.play()
        self._update_info(image)

    def _render_bounds(self) -> tuple[int, int]:
        viewport = self.image_view.viewport().size()
        width = max(1, viewport.width() - 2)
        height = max(1, viewport.height() - 2)
        if self.fit_to_window:
            return width, height
        image = self.current_image()
        if image is not None and image.width and image.height:
            return max(1, int(image.width * self.zoom_factor)), max(1, int(image.height * self.zoom_factor))
        return max(1, int(width * self.zoom_factor)), max(1, int(height * self.zoom_factor))

    @staticmethod
    def _scale_pixmap_to_bounds(
        pixmap: QPixmap,
        max_width: int,
        max_height: int,
        *,
        smooth: bool = True,
    ) -> QPixmap:
        if pixmap.isNull() or max_width <= 0 or max_height <= 0:
            return pixmap
        return pixmap.scaled(
            max_width,
            max_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
            if smooth
            else Qt.TransformationMode.FastTransformation,
        )

    @staticmethod
    def _apply_preview_transforms(
        pixmap: QPixmap,
        *,
        grayscale: bool,
        mirror_horizontal: bool,
    ) -> QPixmap:
        if pixmap.isNull() or (not grayscale and not mirror_horizontal):
            return pixmap
        image = pixmap.toImage()
        if grayscale:
            image = image.convertToFormat(QImage.Format.Format_Grayscale8)
        if mirror_horizontal:
            image = image.flipped(Qt.Orientation.Horizontal)
        return QPixmap.fromImage(image)

    def _build_linetop_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(LINETOP_PANEL_WIDTH)
        panel.setObjectName("linetopAdvancedPanel")
        panel.setStyleSheet(
            """
            QWidget#linetopAdvancedPanel {
                background: #222832;
                border-left: 1px solid #3f4854;
            }
            QLabel {
                color: #e5eaf2;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #44505d;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
                background: #f3f6fb;
            }
            QPushButton {
                background: #343c47;
                color: #e5eaf2;
                border: 1px solid #4c5663;
                border-radius: 5px;
                padding: 5px 9px;
            }
            QPushButton:checked {
                background: #2f7df6;
                border-color: #4f96ff;
                color: white;
            }
            QCheckBox {
                color: #e5eaf2;
                spacing: 8px;
            }
            """
        )

        root = QVBoxLayout(panel)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        title = QLabel("高级功能区")
        title.setStyleSheet("font-weight:600;font-size:15px;")
        root.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll, 1)
        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        mode_label = QLabel("显示模式")
        mode_label.setStyleSheet("font-weight:600;")
        layout.addWidget(mode_label)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self.linetop_line_mode_button = QPushButton("细线稿")
        self.linetop_color_limit_mode_button = QPushButton("色阶限制")
        self.linetop_line_mode_button.setCheckable(True)
        self.linetop_color_limit_mode_button.setCheckable(True)
        self.linetop_mode_group = QButtonGroup(self)
        self.linetop_mode_group.setExclusive(True)
        self.linetop_mode_group.addButton(self.linetop_line_mode_button)
        self.linetop_mode_group.addButton(self.linetop_color_limit_mode_button)
        mode_row.addWidget(self.linetop_line_mode_button)
        mode_row.addWidget(self.linetop_color_limit_mode_button)
        layout.addLayout(mode_row)

        self.linetop_color_limit_row, self.linetop_color_limit_slider, self.linetop_color_limit_value = (
            self._add_linetop_slider(
                layout,
                "色阶限制",
                0,
                15,
                8,
                lambda value: f"{value:d}",
            )
        )
        self.linetop_color_grayscale_checkbox = QCheckBox("黑白")
        layout.addWidget(self.linetop_color_grayscale_checkbox)

        smart_label = QLabel("智能增强")
        smart_label.setStyleSheet("font-weight:600;")
        layout.addWidget(smart_label)
        self.linetop_smart_enhance_checkbox = QCheckBox("智能增强")
        self.linetop_enhanced_line_checkbox = QCheckBox("增强线稿引擎")
        layout.addWidget(self.linetop_smart_enhance_checkbox)
        layout.addWidget(self.linetop_enhanced_line_checkbox)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        preset_row.addWidget(QLabel("预设"))
        self.linetop_photo_preset_button = QPushButton("照片")
        self.linetop_illustration_preset_button = QPushButton("插画")
        self.linetop_photo_preset_button.setCheckable(True)
        self.linetop_illustration_preset_button.setCheckable(True)
        self.linetop_preset_group = QButtonGroup(self)
        self.linetop_preset_group.setExclusive(True)
        self.linetop_preset_group.addButton(self.linetop_photo_preset_button)
        self.linetop_preset_group.addButton(self.linetop_illustration_preset_button)
        preset_row.addWidget(self.linetop_photo_preset_button)
        preset_row.addWidget(self.linetop_illustration_preset_button)
        layout.addLayout(preset_row)

        params_label = QLabel("参数设置")
        params_label.setStyleSheet("font-weight:600;")
        layout.addWidget(params_label)
        self.linetop_contrast_row, self.linetop_contrast_slider, self.linetop_contrast_value = (
            self._add_linetop_slider(
                layout,
                "对比",
                50,
                300,
                100,
                lambda value: f"{value / 100:.2f}",
            )
        )
        self.linetop_brightness_row, self.linetop_brightness_slider, self.linetop_brightness_value = (
            self._add_linetop_slider(
                layout,
                "亮度/阈值倾向",
                -30,
                30,
                0,
                lambda value: f"{value / 100:.2f}",
            )
        )
        self.linetop_thickness_row, self.linetop_thickness_slider, self.linetop_thickness_value = (
            self._add_linetop_slider(
                layout,
                "线条粗细",
                0,
                30,
                0,
                lambda value: f"{value / 10:.1f}",
            )
        )

        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        self.linetop_reset_button = QPushButton("恢复默认值")
        reset_row.addWidget(self.linetop_reset_button)
        layout.addLayout(reset_row)
        layout.addStretch(1)

        self._linetop_color_only_widgets = [
            self.linetop_color_limit_row,
            self.linetop_color_grayscale_checkbox,
        ]
        self._linetop_line_only_widgets = [
            self.linetop_enhanced_line_checkbox,
            self.linetop_thickness_row,
        ]
        self._linetop_controls = [
            self.linetop_line_mode_button,
            self.linetop_color_limit_mode_button,
            self.linetop_color_limit_slider,
            self.linetop_color_grayscale_checkbox,
            self.linetop_smart_enhance_checkbox,
            self.linetop_enhanced_line_checkbox,
            self.linetop_photo_preset_button,
            self.linetop_illustration_preset_button,
            self.linetop_contrast_slider,
            self.linetop_brightness_slider,
            self.linetop_thickness_slider,
        ]
        for control in self._linetop_controls:
            if isinstance(control, QSlider):
                control.valueChanged.connect(self._queue_linetop_settings_changed)
            elif isinstance(control, QCheckBox):
                control.toggled.connect(self._queue_linetop_settings_changed)
            elif isinstance(control, QPushButton):
                control.toggled.connect(self._queue_linetop_settings_changed)
        self.linetop_reset_button.clicked.connect(self._reset_linetop_settings)
        self._sync_linetop_controls_from_settings()
        return panel

    def _add_linetop_slider(
        self,
        parent_layout: QVBoxLayout,
        title: str,
        minimum: int,
        maximum: int,
        value: int,
        formatter,
    ) -> tuple[QWidget, QSlider, QLabel]:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        label = QLabel(title)
        value_label = QLabel(formatter(value))
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(label, 1)
        header.addWidget(value_label)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.valueChanged.connect(lambda slider_value, value_label=value_label: value_label.setText(formatter(slider_value)))
        layout.addLayout(header)
        layout.addWidget(slider)
        parent_layout.addWidget(widget)
        return widget, slider, value_label

    def _current_linetop_settings_from_controls(self) -> LineTopSettings:
        mode = "line" if self.linetop_line_mode_button.isChecked() else "color_limit"
        preset = "photo" if self.linetop_photo_preset_button.isChecked() else "illustration"
        return LineTopSettings(
            mode=mode,
            opacity=1.0,
            edge_strength=self._linetop_settings.edge_strength,
            line_thickness=self.linetop_thickness_slider.value() / 10,
            overlay_contrast=self.linetop_contrast_slider.value() / 100,
            overlay_brightness=self.linetop_brightness_slider.value() / 100,
            color_limit_steps=self.linetop_color_limit_slider.value(),
            color_limit_grayscale=self.linetop_color_grayscale_checkbox.isChecked(),
            color_limit_shape_simplification=1,
            smart_enhance=self.linetop_smart_enhance_checkbox.isChecked(),
            smart_preset=preset,
            enhanced_line_engine=self.linetop_enhanced_line_checkbox.isChecked(),
        )

    def _sync_linetop_controls_from_settings(self) -> None:
        for control in self._linetop_controls:
            control.blockSignals(True)
        try:
            self.linetop_line_mode_button.setChecked(self._linetop_settings.mode == "line")
            self.linetop_color_limit_mode_button.setChecked(self._linetop_settings.mode == "color_limit")
            self.linetop_color_limit_slider.setValue(int(self._linetop_settings.color_limit_steps))
            self.linetop_color_grayscale_checkbox.setChecked(self._linetop_settings.color_limit_grayscale)
            self.linetop_smart_enhance_checkbox.setChecked(self._linetop_settings.smart_enhance)
            self.linetop_enhanced_line_checkbox.setChecked(self._linetop_settings.enhanced_line_engine)
            self.linetop_photo_preset_button.setChecked(self._linetop_settings.smart_preset == "photo")
            self.linetop_illustration_preset_button.setChecked(self._linetop_settings.smart_preset == "illustration")
            self.linetop_contrast_slider.setValue(int(round(self._linetop_settings.overlay_contrast * 100)))
            self.linetop_brightness_slider.setValue(int(round(self._linetop_settings.overlay_brightness * 100)))
            self.linetop_thickness_slider.setValue(int(round(self._linetop_settings.line_thickness * 10)))
        finally:
            for control in self._linetop_controls:
                control.blockSignals(False)
        self.linetop_color_limit_value.setText(f"{self.linetop_color_limit_slider.value():d}")
        self.linetop_contrast_value.setText(f"{self.linetop_contrast_slider.value() / 100:.2f}")
        self.linetop_brightness_value.setText(f"{self.linetop_brightness_slider.value() / 100:.2f}")
        self.linetop_thickness_value.setText(f"{self.linetop_thickness_slider.value() / 10:.1f}")
        self._refresh_linetop_control_visibility()

    def _refresh_linetop_control_visibility(self) -> None:
        mode = "line" if self.linetop_line_mode_button.isChecked() else "color_limit"
        for widget in self._linetop_line_only_widgets:
            widget.setVisible(mode == "line")
        for widget in self._linetop_color_only_widgets:
            widget.setVisible(mode == "color_limit")
        brightness_title = "亮度/阈值倾向" if mode == "line" else "亮度"
        label = self.linetop_brightness_row.findChild(QLabel)
        if label is not None:
            label.setText(brightness_title)

    def _queue_linetop_settings_changed(self) -> None:
        self._linetop_settings = self._current_linetop_settings_from_controls()
        self._refresh_linetop_control_visibility()
        self._linetop_render_cache.clear()
        image = self.current_image()
        if image is not None and self._linetop_preview_active(image) and not self._linetop_show_original_compare:
            self._linetop_render_timer.start()

    def _reset_linetop_settings(self) -> None:
        self._linetop_settings = LineTopSettings()
        self._sync_linetop_controls_from_settings()
        self._queue_linetop_settings_changed()

    def _linetop_preview_active(self, image: ImageItem | None = None) -> bool:
        image = image or self.current_image()
        return bool(
            not self.advanced_panel.isHidden()
            and image is not None
            and not is_supported_video(image.file_path)
        )

    def _toggle_linetop_panel(self) -> None:
        self._set_linetop_panel_visible(self.advanced_panel.isHidden())

    def _set_linetop_panel_visible(self, visible: bool) -> None:
        image = self.current_image()
        visible = bool(visible and image is not None and not is_supported_video(image.file_path))
        if self.advanced_toggle_button.isChecked() != visible:
            self.advanced_toggle_button.blockSignals(True)
            self.advanced_toggle_button.setChecked(visible)
            self.advanced_toggle_button.blockSignals(False)
        if not visible:
            self._set_linetop_compare_original(False, rerender=False)
        panel_is_visible = not self.advanced_panel.isHidden()
        if panel_is_visible == visible:
            self.compare_toggle_button.setEnabled(self._linetop_preview_active(image))
            self.save_render_button.setEnabled(self._linetop_preview_active(image))
            return
        self.advanced_panel.setVisible(visible)
        self.compare_toggle_button.setEnabled(self._linetop_preview_active(image))
        self.save_render_button.setEnabled(self._linetop_preview_active(image))
        self._linetop_render_timer.stop()
        self._linetop_render_pending_token = None
        self._linetop_queued_render_request = None
        if image is None or is_supported_video(image.file_path):
            return
        self._render_current_image(use_thumbnail_first=True)
        self._preview_refine_timer.start()

    def _set_linetop_compare_original(self, checked: bool, *, rerender: bool = True) -> None:
        checked = bool(checked and self._linetop_preview_active())
        self._linetop_show_original_compare = checked
        if self.compare_toggle_button.isChecked() != checked:
            self.compare_toggle_button.blockSignals(True)
            self.compare_toggle_button.setChecked(checked)
            self.compare_toggle_button.blockSignals(False)
        if checked:
            self._linetop_render_timer.stop()
            self._linetop_queued_render_request = None
        image = self.current_image()
        if rerender and image is not None and not is_supported_video(image.file_path):
            self._render_current_image(use_thumbnail_first=True)

    def _linetop_cache_key(self, image: ImageItem, target_width: int, target_height: int) -> tuple[object, ...]:
        return (
            image.id,
            image.file_path,
            image.file_size,
            image.modified_time_ns,
            max(1, int(target_width)),
            max(1, int(target_height)),
            self.grayscale_preview,
            self.mirrored_preview,
            *self._linetop_settings.cache_key(),
        )

    def _request_linetop_render(
        self,
        image: ImageItem,
        target_width: int,
        target_height: int,
        *,
        use_thumbnail_first: bool,
    ) -> None:
        cache_key = self._linetop_cache_key(image, target_width, target_height)
        cached = self._linetop_render_cache.get(cache_key)
        if cached is not None and not cached.isNull():
            self.image_view.set_pixmap(
                cached,
                original_width=image.width,
                original_height=image.height,
                fit_to_window=self.fit_to_window,
                zoom_factor=self.zoom_factor,
            )
            self._update_info(image)
            return
        if use_thumbnail_first:
            fallback = self._cached_preview_base_pixmap(image)
            if fallback.isNull():
                fallback = self._load_quick_preview_pixmap(image, target_width, target_height)
            if not fallback.isNull():
                self.image_view.set_pixmap(
                    fallback,
                    original_width=image.width,
                    original_height=image.height,
                    fit_to_window=self.fit_to_window,
                    zoom_factor=self.zoom_factor,
                )
            else:
                self.image_view.set_message("处理中...")
        else:
            self.image_view.set_message("处理中...")
        self._update_info(image)
        if self._linetop_render_pending_token == cache_key:
            return
        if self._linetop_render_running_token is not None:
            self._linetop_render_pending_token = cache_key
            if self._linetop_render_running_token == cache_key:
                return
            self._linetop_queued_render_request = (
                cache_key,
                image.file_path,
                target_width,
                target_height,
                self._linetop_settings,
                self.grayscale_preview,
                self.mirrored_preview,
            )
            return
        self._start_linetop_render_task(
            cache_key,
            image.file_path,
            target_width,
            target_height,
            self._linetop_settings,
            self.grayscale_preview,
            self.mirrored_preview,
        )

    def _start_linetop_render_task(
        self,
        token: tuple[object, ...],
        image_path: str,
        target_width: int,
        target_height: int,
        settings: LineTopSettings,
        grayscale: bool,
        mirror_horizontal: bool,
    ) -> None:
        self._linetop_render_pending_token = token
        self._linetop_render_running_token = token
        task = _LineTopRenderTask(
            token=token,
            image_path=image_path,
            max_width=target_width,
            max_height=target_height,
            settings=settings,
            grayscale=grayscale,
            mirror_horizontal=mirror_horizontal,
            signals=self._linetop_render_signals,
        )
        self._linetop_thread_pool.start(task)

    def _start_queued_linetop_render_if_needed(self) -> None:
        request = self._linetop_queued_render_request
        self._linetop_queued_render_request = None
        if request is None:
            return
        token, image_path, target_width, target_height, settings, grayscale, mirror_horizontal = request
        if token != self._linetop_render_pending_token:
            return
        image = self.current_image()
        if image is None or not self._linetop_preview_active(image) or self._linetop_show_original_compare:
            return
        if self._linetop_cache_key(image, target_width, target_height) != token:
            return
        cached = self._linetop_render_cache.get(token)
        if cached is not None and not cached.isNull():
            self._linetop_render_pending_token = None
            self.image_view.set_pixmap(
                cached,
                original_width=image.width,
                original_height=image.height,
                fit_to_window=self.fit_to_window,
                zoom_factor=self.zoom_factor,
            )
            self._update_info(image)
            return
        self._start_linetop_render_task(
            token,
            image_path,
            target_width,
            target_height,
            settings,
            grayscale,
            mirror_horizontal,
        )

    def _handle_linetop_render_loaded(self, token: object, qimage: QImage, error: str) -> None:
        if token == self._linetop_render_running_token:
            self._linetop_render_running_token = None
        if token != self._linetop_render_pending_token:
            self._start_queued_linetop_render_if_needed()
            return
        self._linetop_render_pending_token = None
        image = self.current_image()
        if image is None or not self._linetop_preview_active(image) or self._linetop_show_original_compare:
            self._start_queued_linetop_render_if_needed()
            return
        if error or qimage.isNull():
            self.image_view.set_message("无法处理图片")
            self._start_queued_linetop_render_if_needed()
            return
        pixmap = QPixmap.fromImage(qimage)
        if pixmap.isNull():
            self.image_view.set_message("无法处理图片")
            self._start_queued_linetop_render_if_needed()
            return
        if isinstance(token, tuple):
            self._linetop_render_cache[token] = QPixmap(pixmap)
            while len(self._linetop_render_cache) > LINETOP_RENDER_CACHE_LIMIT:
                oldest_key = next(iter(self._linetop_render_cache))
                self._linetop_render_cache.pop(oldest_key, None)
        self.image_view.set_pixmap(
            pixmap,
            original_width=image.width,
            original_height=image.height,
            fit_to_window=self.fit_to_window,
            zoom_factor=self.zoom_factor,
        )
        self._update_info(image)
        self._start_queued_linetop_render_if_needed()

    def _render_linetop_export_image(self, image: ImageItem) -> Image.Image:
        source = _load_linetop_source_image(
            image.file_path,
            grayscale=self.grayscale_preview,
            mirror_horizontal=self.mirrored_preview,
        )
        return render_linetop_image(source, self._linetop_settings)

    def _save_linetop_render_as(self) -> None:
        image = self.current_image()
        if image is None or not self._linetop_preview_active(image):
            return
        suffix = "line" if self._linetop_settings.mode == "line" else "color_limit"
        source_path = Path(image.file_path)
        default_path = source_path.with_name(f"{source_path.stem}_{suffix}.png")
        output_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存处理结果",
            str(default_path),
            "PNG Image (*.png)",
        )
        if not output_path:
            return
        target_path = Path(output_path)
        if target_path.suffix.lower() != ".png":
            target_path = target_path.with_suffix(".png")
        try:
            if target_path.resolve() == source_path.resolve():
                QMessageBox.warning(self, "Eidory", "不能覆盖图库源图片。请选择一个新的 PNG 文件名。")
                return
        except OSError:
            pass
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            rendered = self._render_linetop_export_image(image)
            rendered.save(target_path, "PNG")
        except Exception as exc:
            QMessageBox.warning(self, "Eidory", f"保存失败：{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _set_grayscale_preview(self, checked: bool) -> None:
        self.grayscale_preview = checked
        image = self.current_image()
        if image is not None and not is_supported_video(image.file_path):
            self._render_current_image(use_thumbnail_first=self._preview_base_pixmap.isNull())
            if self._preview_base_pixmap.isNull():
                self._preview_refine_timer.start()

    def _set_mirrored_preview(self, checked: bool) -> None:
        self.mirrored_preview = checked
        image = self.current_image()
        if image is not None and not is_supported_video(image.file_path):
            self._render_current_image(use_thumbnail_first=self._preview_base_pixmap.isNull())
            if self._preview_base_pixmap.isNull():
                self._preview_refine_timer.start()

    def _zoom_by(self, wheel_delta: int) -> None:
        image = self.current_image()
        if image is None or is_supported_video(image.file_path):
            return
        self.image_view.zoom_by(wheel_delta)

    def _fit_image_to_window(self) -> None:
        self._zoom_refine_timer.stop()
        self.fit_to_window = True
        self._stop_pan()
        self.image_view.fit_image_to_window()
        self.zoom_factor = self.image_view.zoom_factor
        image = self.current_image()
        if image is not None:
            self._update_info(image)

    def _actual_size_image(self) -> None:
        self._zoom_refine_timer.stop()
        self.fit_to_window = False
        self._stop_pan()
        self.image_view.set_actual_size()
        self.zoom_factor = self.image_view.zoom_factor
        image = self.current_image()
        if image is not None:
            self._update_info(image)

    def _handle_image_zoom_changed(self, zoom_factor: float, fit_to_window: bool) -> None:
        self.zoom_factor = zoom_factor
        self.fit_to_window = fit_to_window
        image = self.current_image()
        if image is None or is_supported_video(image.file_path):
            return
        self._update_info(image)
        if not fit_to_window:
            self._zoom_refine_timer.start()

    def _fit_zoom_factor(self) -> float:
        return self.image_view.fit_zoom_factor()

    def _schedule_fit_to_window(self) -> None:
        QTimer.singleShot(0, self._fit_current_media_to_window)

    def _fit_current_media_to_window(self) -> None:
        if not self.isVisible():
            return
        image = self.current_image()
        if image is None:
            return
        if is_supported_video(image.file_path):
            if self.video_widget is not None:
                self.video_widget.updateGeometry()
            return
        if self.fit_to_window:
            self._fit_image_to_window()

    def _can_pan(self) -> bool:
        return self.image_view.can_pan()

    def _start_pan(self, event) -> None:
        self._panning = False

    def _pan_to(self, event) -> None:
        return

    def _stop_pan(self) -> None:
        self._panning = False
        self.image_view._update_drag_mode()

    def _update_pan_cursor(self) -> None:
        self.image_view._update_drag_mode()

    def _is_preview_surface(self, watched) -> bool:
        if hasattr(self, "image_view") and watched is self.image_view:
            return True
        if hasattr(self, "image_view") and watched is self.image_view.viewport():
            return True
        if self.video_widget is not None and watched is self.video_widget:
            return True
        return False

    def _fit_render_bounds(self) -> tuple[int, int]:
        viewport = self.image_view.viewport().size()
        return max(1, viewport.width() - 2), max(1, viewport.height() - 2)

    def _load_quick_preview_pixmap(
        self,
        image: ImageItem,
        target_width: int,
        target_height: int,
    ) -> QPixmap:
        if not image.thumbnail_path:
            return QPixmap()
        thumbnail_path = Path(image.thumbnail_path)
        if not thumbnail_path.exists():
            return QPixmap()
        pixmap = QPixmap(str(thumbnail_path))
        if pixmap.isNull():
            return QPixmap()
        pixmap = self._scale_pixmap_to_bounds(
            pixmap,
            target_width,
            target_height,
            smooth=False,
        )
        return self._apply_preview_transforms(
            pixmap,
            grayscale=self.grayscale_preview,
            mirror_horizontal=self.mirrored_preview,
        )

    def _load_preview_source_pixmap(
        self,
        image: ImageItem,
        target_width: int,
        target_height: int,
    ) -> QPixmap:
        base_key = self._preview_base_key_for(image)
        if base_key != self._preview_base_key:
            self._clear_preview_pixmap_cache()
            self._preview_base_key = base_key

        needs_source_load = self._preview_base_pixmap.isNull() or (
            not self._preview_base_is_fallback
            and self._preview_pixmap_is_too_small(
                self._preview_base_pixmap,
                target_width,
                target_height,
            )
        )
        if needs_source_load:
            previous_base = self._preview_base_pixmap
            previous_fallback = self._preview_base_is_fallback
            load_width, load_height = self._source_load_bounds(image, target_width, target_height)
            if image.file_size > INLINE_SOURCE_PREVIEW_MAX_BYTES:
                self._request_preview_source_load(
                    image,
                    max_width=load_width,
                    max_height=load_height,
                )
                if previous_base.isNull():
                    return QPixmap()
                pixmap = previous_base
                fallback_used = previous_fallback
                self._preview_base_pixmap = pixmap
                self._preview_base_is_fallback = fallback_used
                self._preview_variant_cache.clear()
                return self._preview_variant_from_base()
            pixmap = self._load_preview_pixmap(image.file_path, load_width, load_height)
            fallback_used = False
            if pixmap.isNull():
                fallback = image.thumbnail_path if image.thumbnail_path and Path(image.thumbnail_path).exists() else None
                pixmap = QPixmap(fallback) if fallback else QPixmap()
                fallback_used = not pixmap.isNull()
            if pixmap.isNull():
                if previous_base.isNull():
                    return QPixmap()
                pixmap = previous_base
                fallback_used = previous_fallback
            self._preview_base_pixmap = pixmap
            self._preview_base_is_fallback = fallback_used
            self._preview_variant_cache.clear()

        return self._preview_variant_from_base()

    def _preview_variant_from_base(self) -> QPixmap:
        variant_key = (self.grayscale_preview, self.mirrored_preview)
        cached = self._preview_variant_cache.get(variant_key)
        if cached is not None and not cached.isNull():
            return cached

        pixmap = self._apply_preview_transforms(
            self._preview_base_pixmap,
            grayscale=self.grayscale_preview,
            mirror_horizontal=self.mirrored_preview,
        )
        self._preview_variant_cache[variant_key] = pixmap
        return pixmap

    def _request_preview_source_load(
        self,
        image: ImageItem,
        *,
        max_width: int,
        max_height: int,
    ) -> None:
        base_key = self._preview_base_key_for(image)
        token = (
            base_key[0],
            base_key[1],
            base_key[2],
            base_key[3],
            max_width,
            max_height,
        )
        if token == self._preview_source_pending_token:
            return
        fallback_path = (
            image.thumbnail_path
            if image.thumbnail_path and Path(image.thumbnail_path).exists()
            else None
        )
        self._preview_source_pending_token = token
        if self._preview_source_running_token is not None:
            if self._preview_source_running_token == token:
                return
            self._preview_source_queued_request = (
                token,
                image.file_path,
                fallback_path,
                max_width,
                max_height,
            )
            return
        self._start_preview_source_load_task(
            token,
            image.file_path,
            fallback_path,
            max_width,
            max_height,
        )

    def _start_preview_source_load_task(
        self,
        token: tuple[int, str, int, int, int, int],
        image_path: str,
        fallback_path: str | None,
        max_width: int,
        max_height: int,
    ) -> None:
        self._preview_source_pending_token = token
        self._preview_source_running_token = token
        task = _PreviewSourceLoadTask(
            token=token,
            image_path=image_path,
            fallback_path=fallback_path,
            max_width=max_width,
            max_height=max_height,
            signals=self._preview_source_signals,
        )
        self._preview_source_thread_pool.start(task)

    def _start_queued_preview_source_load_if_needed(self) -> None:
        request = self._preview_source_queued_request
        self._preview_source_queued_request = None
        if request is None:
            return
        token, image_path, fallback_path, max_width, max_height = request
        if token != self._preview_source_pending_token:
            return
        image = self.current_image()
        if image is None or self._preview_base_key_for(image) != (token[0], token[1], token[2], token[3]):
            return
        self._start_preview_source_load_task(
            token,
            image_path,
            fallback_path,
            max_width,
            max_height,
        )

    def _handle_preview_source_loaded(
        self,
        token: tuple[int, str, int, int, int, int],
        qimage: QImage,
        fallback_used: bool,
    ) -> None:
        if token == self._preview_source_running_token:
            self._preview_source_running_token = None
        if token != self._preview_source_pending_token:
            self._start_queued_preview_source_load_if_needed()
            return
        self._preview_source_pending_token = None
        image = self.current_image()
        if image is None:
            self._start_queued_preview_source_load_if_needed()
            return
        base_key = (token[0], token[1], token[2], token[3])
        if self._preview_base_key_for(image) != base_key:
            self._start_queued_preview_source_load_if_needed()
            return
        pixmap = QPixmap.fromImage(qimage) if not qimage.isNull() else QPixmap()
        if pixmap.isNull():
            if self._preview_base_pixmap.isNull():
                self.image_view.set_message("无法预览")
            self._start_queued_preview_source_load_if_needed()
            return
        self._preview_base_key = base_key
        self._preview_base_pixmap = pixmap
        self._preview_base_is_fallback = fallback_used
        self._preview_variant_cache.clear()
        self._render_current_image(use_thumbnail_first=True)
        self._start_queued_preview_source_load_if_needed()

    @staticmethod
    def _preview_base_key_for(image: ImageItem) -> tuple[int, str, int, int]:
        return (
            image.id,
            image.file_path,
            image.file_size,
            image.modified_time_ns,
        )

    def _cached_preview_base_pixmap(self, image: ImageItem) -> QPixmap:
        if self._preview_base_key != self._preview_base_key_for(image):
            return QPixmap()
        if self._preview_base_pixmap.isNull():
            return QPixmap()
        variant_key = (self.grayscale_preview, self.mirrored_preview)
        cached = self._preview_variant_cache.get(variant_key)
        if cached is not None and not cached.isNull():
            return cached
        pixmap = self._apply_preview_transforms(
            self._preview_base_pixmap,
            grayscale=self.grayscale_preview,
            mirror_horizontal=self.mirrored_preview,
        )
        self._preview_variant_cache[variant_key] = pixmap
        return pixmap

    @staticmethod
    def _preview_pixmap_is_too_small(pixmap: QPixmap, target_width: int, target_height: int) -> bool:
        if pixmap.isNull():
            return True
        return (
            target_width > int(pixmap.width() * 0.9)
            or target_height > int(pixmap.height() * 0.9)
        )

    @staticmethod
    def _source_load_bounds(
        image: ImageItem,
        target_width: int,
        target_height: int,
    ) -> tuple[int, int]:
        preload = 1.6
        wanted_width = max(1, int(target_width * preload))
        wanted_height = max(1, int(target_height * preload))
        if image.width and image.height:
            return min(image.width, wanted_width), min(image.height, wanted_height)
        return wanted_width, wanted_height

    def _clear_preview_pixmap_cache(self) -> None:
        self._preview_source_pending_token = None
        self._preview_source_queued_request = None
        self._preview_base_key = None
        self._preview_base_pixmap = QPixmap()
        self._preview_base_is_fallback = False
        self._preview_variant_cache.clear()

    def _toggle_video_playback(self) -> None:
        image = self.current_image()
        if image is None or not is_supported_video(image.file_path):
            return
        if self.video_player is None:
            return
        if self.video_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.video_player.pause()
        else:
            self.video_player.play()

    def _seek_video(self, position: int) -> None:
        if self.video_player is None:
            return
        self.video_player.setPosition(position)

    def _update_video_position(self, position: int) -> None:
        if not self.video_position_slider.isSliderDown():
            self.video_position_slider.setValue(position)
        self._update_video_time_label()

    def _update_video_duration(self, duration: int) -> None:
        self.video_position_slider.setRange(0, max(0, duration))
        self._update_video_time_label()

    def _update_video_play_button(self, state) -> None:
        self.video_play_pause_button.setText(
            "暂停" if state == QMediaPlayer.PlaybackState.PlayingState else "播放"
        )

    def _update_video_time_label(self) -> None:
        if self.video_player is None:
            self.video_time_label.setText("00:00 / 00:00")
            return
        self.video_time_label.setText(
            f"{self._format_video_time(self.video_player.position())} / "
            f"{self._format_video_time(self.video_player.duration())}"
        )

    def _stop_video(self) -> None:
        if self.video_player is None:
            self.video_play_pause_button.setText("播放")
            return
        self.video_player.stop()
        self.video_player.setSource(QUrl())
        self.video_play_pause_button.setText("播放")

    @staticmethod
    def _format_video_time(milliseconds: int) -> str:
        total_seconds = max(0, int(milliseconds / 1000))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _event_position(event) -> QPoint:
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()

    def _show_image_context_menu(self, position) -> None:
        menu = QMenu(self)
        fit_action = menu.addAction("适应窗口显示")
        actual_size_action = menu.addAction("实际大小 100%")
        copy_image_action = menu.addAction("复制图片")
        open_action = menu.addAction("打开原图")
        reveal_action = menu.addAction("Finder 中显示")
        remove_index_action = menu.addAction("移除当前索引")
        sender = self.sender()
        if hasattr(sender, "mapToGlobal"):
            global_position = sender.mapToGlobal(position)
        else:
            global_position = self.image_view.viewport().mapToGlobal(position)
        action = menu.exec(global_position)
        if action == fit_action:
            self._fit_image_to_window()
        elif action == actual_size_action:
            self._actual_size_image()
        elif action == copy_image_action:
            self._copy_current_image()
        elif action == open_action:
            self._open_original()
        elif action == reveal_action:
            self._reveal_in_finder()
        elif action == remove_index_action:
            self._remove_current_index()

    def _install_shortcuts(self) -> None:
        shortcuts = [
            (QKeySequence(Qt.Key.Key_Space), self._handle_space_pressed),
            (QKeySequence(QKeySequence.StandardKey.Close), self.close),
            (QKeySequence("Meta+W"), self.close),
            (QKeySequence("Ctrl+W"), self.close),
            (QKeySequence(Qt.Key.Key_Right), lambda: self._move(1)),
            (QKeySequence(Qt.Key.Key_Down), lambda: self._move(1)),
            (QKeySequence(Qt.Key.Key_Left), lambda: self._move(-1)),
            (QKeySequence(Qt.Key.Key_Up), lambda: self._move(-1)),
        ]
        for sequence, callback in shortcuts:
            shortcut = QShortcut(sequence, self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    def _refresh_feedback_buttons(self, image: ImageItem) -> None:
        can_feedback = bool(self.semantic_query)
        self.feedback_group.setExclusive(False)
        for button in self.feedback_buttons.values():
            button.blockSignals(True)
            button.setChecked(False)
            button.setEnabled(can_feedback)
            button.blockSignals(False)
        self.feedback_group.setExclusive(True)
        if not can_feedback or not self.semantic_query:
            return

        label = self.store.get_search_feedback(
            query=self.semantic_query,
            image_id=image.id,
            model_name=self.model_name,
            model_revision=self.model_revision,
            embedding_dim=self.embedding_dim,
        )
        if label in self.feedback_buttons:
            button = self.feedback_buttons[label]
            button.blockSignals(True)
            button.setChecked(True)
            button.blockSignals(False)

    def _save_favorite(self, checked: bool) -> None:
        image = self.current_image()
        if image is None:
            return
        self.store.update_favorite(image.id, checked)
        updated = replace(image, is_favorite=checked)
        self.images[self.index] = updated
        self.favoriteChanged.emit(updated)

    def _save_feedback(self, label: str) -> None:
        image = self.current_image()
        if image is None or not self.semantic_query:
            return
        self.store.upsert_search_feedback(
            query=self.semantic_query,
            image_id=image.id,
            model_name=self.model_name,
            model_revision=self.model_revision,
            embedding_dim=self.embedding_dim,
            score=image.score,
            label=label,
        )
        self._refresh_feedback_buttons(image)
        self.feedbackSaved.emit(image, label)

    def _open_original(self) -> None:
        image = self.current_image()
        if image is None:
            return
        path = Path(image.file_path)
        if image.is_missing or not path.exists():
            QMessageBox.warning(self, "Eidory", "源文件不存在，无法打开。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _reveal_in_finder(self) -> None:
        image = self.current_image()
        if image is None:
            return
        path = Path(image.file_path)
        if image.is_missing or not path.exists():
            QMessageBox.warning(self, "Eidory", "源文件不存在，无法在 Finder 中显示。")
            return
        subprocess.run(["open", "-R", str(path)], check=False)

    def _copy_current_image(self) -> None:
        image = self.current_image()
        if image is None or is_supported_video(image.file_path):
            return
        if self._linetop_preview_active(image) and not self._linetop_show_original_compare:
            try:
                rendered = self._render_linetop_export_image(image)
                QApplication.clipboard().setImage(_qimage_from_pillow(rendered))
            except Exception:
                QMessageBox.warning(self, "Eidory", "无法复制当前处理结果。")
            return
        width = image.width or self.image_view.viewport().width()
        height = image.height or self.image_view.viewport().height()
        pixmap = self._load_preview_pixmap(image.file_path, width, height)
        if pixmap.isNull():
            fallback = image.thumbnail_path if image.thumbnail_path and Path(image.thumbnail_path).exists() else None
            pixmap = QPixmap(fallback) if fallback else QPixmap()
        if pixmap.isNull():
            QMessageBox.warning(self, "Eidory", "无法复制当前图片。")
            return
        pixmap = self._apply_preview_transforms(
            pixmap,
            grayscale=self.grayscale_preview,
            mirror_horizontal=self.mirrored_preview,
        )
        QApplication.clipboard().setPixmap(pixmap)

    def _copy_path(self) -> None:
        image = self.current_image()
        if image is None:
            return
        QApplication.clipboard().setText(image.file_path)

    def _remove_current_index(self) -> None:
        image = self.current_image()
        if image is None:
            return
        answer = QMessageBox.question(
            self,
            "移除索引",
            f"只从 Eidory 移除“{image.file_name}”的索引记录，不删除源文件。继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        thumbnail_paths = self.store.remove_images_from_library([image.id])
        self._delete_thumbnail_files(thumbnail_paths)
        removed_image = image
        del self.images[self.index]
        if self.images:
            self.index = min(self.index, len(self.images) - 1)
            self._refresh()
        else:
            self._refresh()
            self.close()
        self.indexRemoved.emit(removed_image)

    def _delete_thumbnail_files(self, thumbnail_paths: list[str]) -> None:
        if self.thumbnail_dir is None:
            return
        thumbnail_root = self.thumbnail_dir.resolve()
        for thumbnail_path in thumbnail_paths:
            try:
                resolved = Path(thumbnail_path).resolve()
                if resolved.is_relative_to(thumbnail_root):
                    resolved.unlink(missing_ok=True)
            except Exception:
                continue

    @staticmethod
    def _load_preview_pixmap(image_path: str, max_width: int, max_height: int) -> QPixmap:
        if max_width <= 0 or max_height <= 0:
            return QPixmap()
        path = Path(image_path)
        if not path.exists():
            return QPixmap()
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        source_size = reader.size()
        if source_size.isValid():
            scaled_size = QSize(source_size.width(), source_size.height())
            scaled_size.scale(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio)
            if scaled_size.width() < source_size.width() or scaled_size.height() < source_size.height():
                reader.setScaledSize(scaled_size)
        qimage = reader.read()
        if not qimage.isNull():
            return QPixmap.fromImage(qimage)
        try:
            with open_local_image(image_path) as image:
                if image.format == "JPEG":
                    image.draft("RGB", (max_width, max_height))
                image = ImageOps.exif_transpose(image)
                image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                image = image.convert("RGBA")
                data = image.tobytes("raw", "RGBA")
                qimage = QImage(
                    data,
                    image.width,
                    image.height,
                    image.width * 4,
                    QImage.Format.Format_RGBA8888,
                )
                return QPixmap.fromImage(qimage.copy())
        except Exception:
            return QPixmap()
