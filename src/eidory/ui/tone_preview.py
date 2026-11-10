"""A bounded latest-value renderer, independent of full-resolution disk saves."""
from __future__ import annotations

import math
import time

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtGui import QImage


class LatestTonePreview(QObject):
    ready = Signal(object, object)
    failed = Signal(object, object)
    completed = Signal(object, object)

    def __init__(self, start, parent=None):
        super().__init__(parent)
        self.start = start
        self.serial = 0
        self.minimum_serial = 0
        self.owner = None
        self.pending = None
        self.running = False
        self.closed = False
        self.next_frame = 0.
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.launch)
        self.completed.connect(self.finished)

    def request(self, owner, cache, options):
        if self.closed:
            return
        if owner != self.owner:
            self.invalidate()
            self.owner = owner
        self.serial += 1
        self.pending = (self.serial, owner, cache, dict(options))
        if not self.running:
            self.schedule()

    def schedule(self):
        if self.pending and not self.closed:
            self.timer.start(max(0, math.ceil((self.next_frame-time.monotonic())*1000)))

    def launch(self):
        if self.running or not self.pending or self.closed:
            return
        request, self.pending = self.pending, None
        self.running = True
        self.next_frame = time.monotonic()+1/60
        def worker():
            try:
                pixels = request[2].render(request[3])
                data = pixels.tobytes()
                result = QImage(data, pixels.width, pixels.height, pixels.width*4, QImage.Format.Format_RGBA8888).copy()
            except Exception as exc:
                result = exc
            self.completed.emit(request[:2]+(request[3],), result)
        if not self.start(worker):
            self.running = False

    def finished(self, request, result):
        self.running = False
        # Publish completed frames monotonically even if the pointer has moved
        # again, or sustained input faster than rendering would starve the view.
        # Only the one latest waiting value is rendered next.
        if not self.closed and request[0] > self.minimum_serial and request[1] == self.owner:
            if isinstance(result, Exception):
                self.failed.emit(request[1:], result)
            else:
                self.ready.emit(request[1:], result)
        self.schedule()

    def invalidate(self):
        self.serial += 1
        self.minimum_serial = self.serial
        self.pending = None
        self.timer.stop()

    def close(self):
        self.closed = True
        self.invalidate()
