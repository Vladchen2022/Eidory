from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageOps
from PySide6.QtCore import QEvent, QPoint, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QImage,
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
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from eidory.core.image_loader import open_local_image
from eidory.core.media_types import is_supported_video
from eidory.core.metadata_store import MetadataStore
from eidory.models import ImageItem


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
        self._preview_source_key: tuple[int, str, int, int, bool, bool] | None = None
        self._preview_source_pixmap = QPixmap()
        self._zoom_refine_timer = QTimer(self)
        self._zoom_refine_timer.setSingleShot(True)
        self._zoom_refine_timer.setInterval(90)
        self._zoom_refine_timer.timeout.connect(self._render_current_image)

        self.setWindowTitle("Eidory 预览")
        self.resize(1200, 820)
        self.setMinimumSize(720, 520)

        self.image_view = PreviewImageView()
        self.image_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_view.viewport().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_view.customContextMenuRequested.connect(self._show_image_context_menu)
        self.image_view.viewport().customContextMenuRequested.connect(self._show_image_context_menu)
        self.image_view.doubleClicked.connect(self.close)
        self.image_view.zoomChanged.connect(self._handle_image_zoom_changed)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(420)
        self.video_widget.setStyleSheet("background:#2d3138;")
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.video_widget.installEventFilter(self)
        self.video_player = QMediaPlayer(self)
        self.video_audio_output = QAudioOutput(self)
        self.video_player.setAudioOutput(self.video_audio_output)
        self.video_player.setVideoOutput(self.video_widget)

        self.preview_stack = QStackedWidget()
        self.preview_stack.addWidget(self.image_view)
        self.preview_stack.addWidget(self.video_widget)

        self.info_label = QLabel("-")
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.previous_button = QPushButton("上一张")
        self.next_button = QPushButton("下一张")
        self.fit_button = QPushButton("适应窗口")
        self.actual_size_button = QPushButton("100%")
        self.grayscale_button = QPushButton("黑白")
        self.grayscale_button.setCheckable(True)
        self.mirror_button = QPushButton("左右翻转")
        self.mirror_button.setCheckable(True)
        self.video_play_pause_button = QPushButton("播放")
        self.video_position_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_position_slider.setRange(0, 0)
        self.video_time_label = QLabel("00:00 / 00:00")
        self.favorite_checkbox = QCheckBox("收藏")

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
        controls.addWidget(self.open_original_button)
        controls.addWidget(self.reveal_button)
        controls.addWidget(self.copy_image_button)
        controls.addWidget(self.copy_path_button)
        controls.addWidget(self.remove_index_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.preview_stack, 1)
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
        self.video_player.positionChanged.connect(self._update_video_position)
        self.video_player.durationChanged.connect(self._update_video_duration)
        self.video_player.playbackStateChanged.connect(self._update_video_play_button)
        self._install_shortcuts()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._app_event_filter_installed = True

        self._refresh()

    def closeEvent(self, event) -> None:
        if self._app_event_filter_installed:
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._app_event_filter_installed = False
        self._stop_video()
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._schedule_fit_to_window()

    def current_image(self) -> ImageItem | None:
        if not self.images:
            return None
        return self.images[self.index]

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        image = self.current_image()
        if self.fit_to_window and (image is None or not is_supported_video(image.file_path)):
            self._render_current_image()

    def eventFilter(self, watched, event) -> bool:
        if (
            self._is_preview_surface(watched)
            and event.type() == QEvent.Type.MouseButtonDblClick
            and event.button() == Qt.MouseButton.LeftButton
        ):
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
        image = self.current_image()
        if image is None:
            self.info_label.setText("-")
            self._clear_preview_pixmap_cache()
            self.image_view.set_message("未选择图片")
            self.video_controls_widget.hide()
            self.fit_button.setEnabled(False)
            self.actual_size_button.setEnabled(False)
            self.grayscale_button.setEnabled(False)
            self.mirror_button.setEnabled(False)
            self.copy_image_button.setEnabled(False)
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
            self._render_current_video(image)
        else:
            self._render_current_image()
        self.imageChanged.emit(image)

    def _handle_space_pressed(self) -> None:
        image = self.current_image()
        if image is None:
            return
        if is_supported_video(image.file_path):
            self._toggle_video_playback()
            return
        self._fit_image_to_window()

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

    def _render_current_image(self) -> None:
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
        max_width, max_height = self._render_bounds()
        pixmap = self._load_preview_source_pixmap(image, max_width, max_height)
        if pixmap.isNull():
            self.image_view.set_message("无法预览")
            return
        self.image_view.set_pixmap(
            pixmap,
            original_width=image.width,
            original_height=image.height,
            fit_to_window=self.fit_to_window,
            zoom_factor=self.zoom_factor,
        )
        self._update_info(image)

    def _render_current_video(self, image: ImageItem) -> None:
        self.fit_button.setEnabled(False)
        self.actual_size_button.setEnabled(False)
        self.grayscale_button.setEnabled(False)
        self.mirror_button.setEnabled(False)
        self.copy_image_button.setEnabled(False)
        self._stop_pan()
        self.preview_stack.setCurrentWidget(self.video_widget)
        self.video_controls_widget.show()
        self.video_position_slider.setValue(0)
        self.video_position_slider.setRange(0, 0)
        self.video_time_label.setText("00:00 / 00:00")
        self.video_player.stop()
        self.video_player.setSource(QUrl())

        path = Path(image.file_path)
        if image.is_missing or not path.exists():
            self.info_label.setText(f"{image.file_name}    视频文件不存在")
            self.video_play_pause_button.setEnabled(False)
            return

        self.video_play_pause_button.setEnabled(True)
        self.video_player.setSource(QUrl.fromLocalFile(str(path)))
        self.video_player.play()
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

    def _set_grayscale_preview(self, checked: bool) -> None:
        self.grayscale_preview = checked
        image = self.current_image()
        if image is not None and not is_supported_video(image.file_path):
            self._render_current_image()

    def _set_mirrored_preview(self, checked: bool) -> None:
        self.mirrored_preview = checked
        image = self.current_image()
        if image is not None and not is_supported_video(image.file_path):
            self._render_current_image()

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
        if hasattr(self, "video_widget") and watched is self.video_widget:
            return True
        return False

    def _fit_render_bounds(self) -> tuple[int, int]:
        viewport = self.image_view.viewport().size()
        return max(1, viewport.width() - 2), max(1, viewport.height() - 2)

    def _load_preview_source_pixmap(
        self,
        image: ImageItem,
        target_width: int,
        target_height: int,
    ) -> QPixmap:
        key = (
            image.id,
            image.file_path,
            image.file_size,
            image.modified_time_ns,
            self.grayscale_preview,
            self.mirrored_preview,
        )
        if key != self._preview_source_key:
            self._clear_preview_pixmap_cache()
            self._preview_source_key = key

        if self._preview_source_pixmap.isNull() or self._preview_source_is_too_small(
            target_width,
            target_height,
        ):
            load_width, load_height = self._source_load_bounds(image, target_width, target_height)
            pixmap = self._load_preview_pixmap(image.file_path, load_width, load_height)
            if pixmap.isNull():
                fallback = (
                    image.thumbnail_path
                    if image.thumbnail_path and Path(image.thumbnail_path).exists()
                    else None
                )
                pixmap = QPixmap(fallback) if fallback else QPixmap()
            pixmap = self._apply_preview_transforms(
                pixmap,
                grayscale=self.grayscale_preview,
                mirror_horizontal=self.mirrored_preview,
            )
            self._preview_source_pixmap = pixmap
        return self._preview_source_pixmap

    def _preview_source_is_too_small(self, target_width: int, target_height: int) -> bool:
        if self._preview_source_pixmap.isNull():
            return True
        return (
            target_width > int(self._preview_source_pixmap.width() * 0.9)
            or target_height > int(self._preview_source_pixmap.height() * 0.9)
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
        self._preview_source_key = None
        self._preview_source_pixmap = QPixmap()

    def _toggle_video_playback(self) -> None:
        image = self.current_image()
        if image is None or not is_supported_video(image.file_path):
            return
        if self.video_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.video_player.pause()
        else:
            self.video_player.play()

    def _seek_video(self, position: int) -> None:
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
        self.video_time_label.setText(
            f"{self._format_video_time(self.video_player.position())} / "
            f"{self._format_video_time(self.video_player.duration())}"
        )

    def _stop_video(self) -> None:
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
