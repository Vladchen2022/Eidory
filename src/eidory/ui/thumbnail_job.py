"""Cancellable thumbnail repair without decoding on the GUI thread."""
import threading
from pathlib import Path
from weakref import ref
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QProgressBar, QPushButton
from eidory.core.media_types import is_supported_video


class ThumbnailJob(QObject):
    progress = Signal(int, int)
    finished = Signal(int, int, bool)

    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self.running = False
        self.cancelled = threading.Event()
        self.panel = QWidget()
        row = QHBoxLayout(self.panel)
        row.setContentsMargins(0, 0, 0, 0)
        self.bar = QProgressBar()
        self.bar.setFormat('重建缩略图 %v/%m')
        self.button = QPushButton('取消')
        self.button.clicked.connect(self.cancelled.set)
        row.addWidget(self.bar)
        row.addWidget(self.button)
        window.statusBar().addPermanentWidget(self.panel)
        self.panel.hide()
        self.progress.connect(self.update_progress)
        self.finished.connect(self.done)

    def start(self, images):
        if self.running:
            return
        self.running = True
        self.cancelled.clear()
        self.bar.setRange(0, len(images))
        self.bar.setValue(0)
        self.panel.show()
        w = self._window()
        def run():
            good = bad = 0
            try:
                for index, image in enumerate(images):
                    if self.cancelled.is_set():
                        break
                    try:
                        if image.is_missing or not Path(image.file_path).exists():
                            raise FileNotFoundError(image.file_path)
                        path = (w.thumbnailer.generate_video if is_supported_video(image.file_path) else w.thumbnailer.generate)(image.id, image.file_path)
                        w.store.update_thumbnail(image.id, str(path), 'ready')
                        good += 1
                    except Exception:
                        bad += 1
                        w.store.update_thumbnail(image.id, None, 'failed')
                    self.progress.emit(index + 1, len(images))
            finally:
                self.finished.emit(good, bad, self.cancelled.is_set())
        w._start_background_task(run, on_rejected=lambda: self.done(0, 0, True))

    def update_progress(self, value, total):
        self.bar.setRange(0, total)
        self.bar.setValue(value)

    def done(self, good, bad, cancelled):
        self.running = False
        self.panel.hide()
        w = self._window()
        if w._closing:
            return
        w._invalidate_near_duplicate_hash_cache()
        w.grid_view.clear_pixmap_cache()
        w._refresh_current_results_for_filters()
        text = f'缩略图{"已停止" if cancelled else "已完成"}：成功 {good}，失败 {bad}'
        w._record_operation_history(text)
        w.statusBar().showMessage(text)
