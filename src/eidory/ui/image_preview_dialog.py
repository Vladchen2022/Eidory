from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageOps
from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QKeyEvent, QKeySequence, QPixmap, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
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
from eidory.core.media_types import is_supported_video
from eidory.core.metadata_store import MetadataStore
from eidory.models import ImageItem


class ImagePreviewDialog(QDialog):
    imageChanged = Signal(object)
    favoriteChanged = Signal(object)
    feedbackSaved = Signal(object, str)

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
        self.fit_to_window = True
        self.zoom_factor = 1.0
        self._panning = False
        self._pan_start_pos = QPoint()
        self._pan_start_horizontal = 0
        self._pan_start_vertical = 0
        self._shortcuts: list[QShortcut] = []
        self._app_event_filter_installed = False

        self.setWindowTitle("Eidory 预览")
        self.resize(1200, 820)
        self.setMinimumSize(720, 520)

        self.image_label = QLabel("未选择图片")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background:#2d3138;color:#d8dee9;")
        self.image_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self._show_image_context_menu)
        self.image_label.installEventFilter(self)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setMinimumHeight(420)
        self.scroll_area.setStyleSheet("background:#2d3138;")
        self.scroll_area.viewport().installEventFilter(self)
        self.scroll_area.viewport().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.scroll_area.viewport().customContextMenuRequested.connect(self._show_image_context_menu)

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
        self.preview_stack.addWidget(self.scroll_area)
        self.preview_stack.addWidget(self.video_widget)

        self.info_label = QLabel("-")
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.previous_button = QPushButton("上一张")
        self.next_button = QPushButton("下一张")
        self.fit_button = QPushButton("适应窗口")
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
        ]
        for button in [*nav_buttons, *self.feedback_buttons.values()]:
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.open_original_button = QPushButton("打开源文件")
        self.reveal_button = QPushButton("Finder 中显示")
        self.copy_path_button = QPushButton("复制路径")

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
        controls.addSpacing(12)
        controls.addWidget(self.favorite_checkbox)
        controls.addSpacing(12)
        controls.addStretch(1)
        controls.addWidget(self.open_original_button)
        controls.addWidget(self.reveal_button)
        controls.addWidget(self.copy_path_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.preview_stack, 1)
        layout.addWidget(self.video_controls_widget)
        layout.addWidget(self.info_label)
        layout.addLayout(controls)

        self.previous_button.clicked.connect(lambda: self._move(-1))
        self.next_button.clicked.connect(lambda: self._move(1))
        self.fit_button.clicked.connect(self._fit_image_to_window)
        self.favorite_checkbox.toggled.connect(self._save_favorite)
        self.feedback_relevant_button.clicked.connect(lambda: self._save_feedback("relevant"))
        self.feedback_irrelevant_button.clicked.connect(lambda: self._save_feedback("irrelevant"))
        self.feedback_ignored_button.clicked.connect(lambda: self._save_feedback("ignored"))
        self.open_original_button.clicked.connect(self._open_original)
        self.reveal_button.clicked.connect(self._reveal_in_finder)
        self.copy_path_button.clicked.connect(self._copy_path)
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
            hasattr(self, "scroll_area")
            and watched == self.scroll_area.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            if self.fit_to_window:
                self._schedule_fit_to_window()
            return super().eventFilter(watched, event)
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
        if (
            (watched == self.scroll_area.viewport() or watched == self.image_label)
            and event.type() == QEvent.Type.Wheel
        ):
            self._zoom_by(event.angleDelta().y())
            event.accept()
            return True
        if watched == self.scroll_area.viewport() or watched == self.image_label:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                if self._can_pan():
                    self._start_pan(event)
                    event.accept()
                    return True
            if event.type() == QEvent.Type.MouseMove and self._panning:
                self._pan_to(event)
                event.accept()
                return True
            if event.type() == QEvent.Type.MouseButtonRelease and self._panning:
                self._stop_pan()
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
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("未选择图片")
            self.image_label.resize(self.scroll_area.viewport().size())
            self.video_controls_widget.hide()
            self._stop_video()
            return

        self.previous_button.setEnabled(self.index > 0)
        self.next_button.setEnabled(self.index < len(self.images) - 1)
        self.favorite_checkbox.blockSignals(True)
        self.favorite_checkbox.setChecked(image.is_favorite)
        self.favorite_checkbox.blockSignals(False)
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
        self.preview_stack.setCurrentWidget(self.scroll_area)
        self.fit_button.setEnabled(True)
        max_width, max_height = self._render_bounds()
        pixmap = self._load_preview_pixmap(
            image.file_path,
            max_width,
            max_height,
        )
        if pixmap.isNull():
            fallback = image.thumbnail_path if image.thumbnail_path and Path(image.thumbnail_path).exists() else None
            pixmap = QPixmap(fallback) if fallback else QPixmap()
        if pixmap.isNull():
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("无法预览")
            self.image_label.resize(self.scroll_area.viewport().size())
            return
        self.image_label.setText("")
        pixmap = self._scale_pixmap_to_bounds(pixmap, max_width, max_height)
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())
        self._update_pan_cursor()
        self._update_info(image)

    def _render_current_video(self, image: ImageItem) -> None:
        self.fit_button.setEnabled(False)
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
        viewport = self.scroll_area.viewport().size()
        width = max(1, viewport.width() - 2)
        height = max(1, viewport.height() - 2)
        if self.fit_to_window:
            return width, height
        return max(1, int(width * self.zoom_factor)), max(1, int(height * self.zoom_factor))

    @staticmethod
    def _scale_pixmap_to_bounds(pixmap: QPixmap, max_width: int, max_height: int) -> QPixmap:
        if pixmap.isNull() or max_width <= 0 or max_height <= 0:
            return pixmap
        return pixmap.scaled(
            max_width,
            max_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _zoom_by(self, wheel_delta: int) -> None:
        if wheel_delta == 0:
            return
        if self.fit_to_window:
            self.fit_to_window = False
            self.zoom_factor = 1.0
        step = 1.15 if wheel_delta > 0 else 1 / 1.15
        self.zoom_factor = max(0.1, min(8.0, self.zoom_factor * step))
        self._render_current_image()

    def _fit_image_to_window(self) -> None:
        self.fit_to_window = True
        self.zoom_factor = 1.0
        self._stop_pan()
        self._render_current_image()

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
        horizontal = self.scroll_area.horizontalScrollBar()
        vertical = self.scroll_area.verticalScrollBar()
        return not self.fit_to_window and (horizontal.maximum() > 0 or vertical.maximum() > 0)

    def _start_pan(self, event) -> None:
        self._panning = True
        self._pan_start_pos = self._event_position(event)
        self._pan_start_horizontal = self.scroll_area.horizontalScrollBar().value()
        self._pan_start_vertical = self.scroll_area.verticalScrollBar().value()
        self.scroll_area.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
        self.image_label.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _pan_to(self, event) -> None:
        delta = self._event_position(event) - self._pan_start_pos
        self.scroll_area.horizontalScrollBar().setValue(self._pan_start_horizontal - delta.x())
        self.scroll_area.verticalScrollBar().setValue(self._pan_start_vertical - delta.y())

    def _stop_pan(self) -> None:
        self._panning = False
        self._update_pan_cursor()

    def _update_pan_cursor(self) -> None:
        cursor = Qt.CursorShape.OpenHandCursor if self._can_pan() else Qt.CursorShape.ArrowCursor
        self.scroll_area.viewport().setCursor(cursor)
        self.image_label.setCursor(cursor)

    def _is_preview_surface(self, watched) -> bool:
        if hasattr(self, "image_label") and watched is self.image_label:
            return True
        if hasattr(self, "scroll_area") and watched is self.scroll_area.viewport():
            return True
        if hasattr(self, "video_widget") and watched is self.video_widget:
            return True
        return False

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
        open_action = menu.addAction("打开原图")
        reveal_action = menu.addAction("Finder 中显示")
        sender = self.sender()
        if hasattr(sender, "mapToGlobal"):
            global_position = sender.mapToGlobal(position)
        else:
            global_position = self.scroll_area.viewport().mapToGlobal(position)
        action = menu.exec(global_position)
        if action == fit_action:
            self._fit_image_to_window()
        elif action == open_action:
            self._open_original()
        elif action == reveal_action:
            self._reveal_in_finder()

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

    def _copy_path(self) -> None:
        image = self.current_image()
        if image is None:
            return
        QApplication.clipboard().setText(image.file_path)

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
