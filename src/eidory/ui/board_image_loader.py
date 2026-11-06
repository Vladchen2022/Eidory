"""Bounded QImage decoding. QPixmaps and scene mutations stay on the GUI thread."""
from collections import OrderedDict
import logging
import threading

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QImage, QImageReader
from eidory.ui.display_image import read_display_image
from eidory.core.image_loader import MAX_DECODED_IMAGE_PIXELS
from eidory.core.thumbnailer import upgrade_display_thumbnail


class _Signals(QObject):
    loaded = Signal(object, object, object)


class _Decode(QRunnable):
    def __init__(self, token, paths, side, cancelled, signals):
        super().__init__()
        self.token, self.paths, self.side = token, paths, side
        self.cancelled, self.signals = cancelled, signals

    def run(self):
        image, used = QImage(), ''
        try:
            if len(self.paths) >= 2 and not self.cancelled.is_set():
                upgrade_display_thumbnail(self.paths[0], self.paths[1])
            for path in self.paths:
                if self.cancelled.is_set():
                    break
                reader = QImageReader(path)
                reader.setAutoTransform(True)
                size = reader.size()
                if not self.side and size.width() * size.height() > MAX_DECODED_IMAGE_PIXELS:
                    continue
                if self.side and size.isValid() and max(size.width(), size.height()) > self.side:
                    size.scale(self.side, self.side, Qt.AspectRatioMode.KeepAspectRatio)
                    reader.setScaledSize(size)
                image = read_display_image(reader, path)
                if not image.isNull():
                    used = path
                    break
        except Exception:
            logging.getLogger('eidory.runtime').exception('Board image decoding failed')
        finally:
            # The loader outlives in-flight tasks; tokens discard old results.
            self.signals.loaded.emit(self.token, image, used)


class BoardImageLoader(QObject):
    loaded = Signal(object, object, object)

    def __init__(self, parent):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        # Tasks own a reference too. It must survive destruction of the loader;
        # Qt automatically disconnects the receiver if that happens first.
        self.signals = _Signals()
        self.signals.loaded.connect(self._done)
        self.queue = OrderedDict()
        self.active = {}
        self.stopped = False

    def request(self, token, paths, side, *, urgent=False):
        if self.stopped or token in self.active:
            return
        self.queue[token] = (paths, side)
        if urgent:
            self.queue.move_to_end(token, last=False)
        self._pump()

    def _pump(self):
        # Feed a small look-ahead batch. Waiting for a GUI round trip after
        # every pair leaves both decoder threads idle between paint events.
        while not self.stopped and self.queue and len(self.active) < 8:
            token, (paths, side) = self.queue.popitem(last=False)
            cancelled = threading.Event()
            self.active[token] = cancelled
            self.pool.start(_Decode(token, paths, side, cancelled, self.signals), 1 if side > 560 else 0)

    def _done(self, token, image, path):
        cancelled = self.active.pop(token, None)
        if not self.stopped and cancelled is not None and not cancelled.is_set():
            self.loaded.emit(token, image, path)
        self._pump()

    def reset(self):
        self.queue.clear()
        for cancelled in self.active.values():
            cancelled.set()

    def stop(self):
        self.stopped = True
        self.reset()
