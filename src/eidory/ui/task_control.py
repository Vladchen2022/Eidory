"""Visible, cancellable foreground jobs with late-result isolation."""
import threading
import time
from weakref import ref
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton


class Job:
    def __init__(self, control, title, retry, batch=None):
        self.control = control
        self.title = title
        self.retry = retry
        self.batch = batch
        self.cancelled = threading.Event()
        self.completed = []
        self.lock = threading.Lock()
        self.started = time.monotonic()

    def emit(self, kind, payload):
        self.control.delivered.emit(self, kind, payload)

    def checkpoint(self, result):
        with self.lock:
            if not self.cancelled.is_set():
                self.completed.append(result)


class TaskControl(QObject):
    delivered = Signal(object, str, object)

    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self.current = None
        self.failed_retry = None
        self.panel = QWidget()
        row = QHBoxLayout(self.panel)
        row.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel()
        self.button = QPushButton('取消')
        self.button.clicked.connect(self.click)
        row.addWidget(self.label)
        row.addWidget(self.button)
        window.statusBar().addPermanentWidget(self.panel)
        self.panel.hide()
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        self.delivered.connect(self.receive)

    @property
    def w(self):
        return self._window()

    def begin(self, title, retry, *, batch=None):
        if self.current:
            self.cancel()
        self.failed_retry = None
        self.current = Job(self, title, retry, batch)
        self.timer.start()
        self.refresh()
        return self.current

    def bind(self, job, provider):
        if hasattr(provider, "__dict__"):
            provider.cancel_event = job.cancelled

    def refresh(self):
        job = self.current
        self.panel.setVisible(job is not None or self.failed_retry is not None)
        if job:
            self.label.setText(f'{job.title} · {int(time.monotonic() - job.started)} 秒')
            self.button.setText('取消')

    def receive(self, job, kind, payload):
        if self.current is job and not job.cancelled.is_set():
            self.w.events.put(("controlled_ai", (job, kind, payload)))

    def consume(self, job, kind, payload):
        if self.current is not job or job.cancelled.is_set() or self.w._closing:
            return False
        if kind.endswith('_progress'):
            if kind == 'creative_all_node_notes_progress':
                _, value, total, title = payload
                job.title = f'AI · {title} · {value}/{total}'
                self.refresh()
        elif kind.endswith('_done') or kind.endswith('_error'):
            self.current = None
            self.timer.stop()
            if kind.endswith('_error'):
                self.failed_retry = job.retry
                self.label.setText('生成未完成，原内容已保留')
                self.button.setText('重试')
            self.refresh()
        return True

    def click(self):
        if self.current:
            self.cancel()
        elif self.failed_retry:
            retry, self.failed_retry = self.failed_retry, None
            retry()

    def cancel(self):
        job = self.current
        if job is None:
            return
        job.cancelled.set()
        self.current = None
        self.timer.stop()
        self.panel.hide()
        self.w._set_creative_all_nodes_busy(False)
        self.w._sync_creative_node_panel()
        if job.batch:
            with job.lock:
                completed = list(job.completed)
            if completed:
                pid, node_id = job.batch
                self.w._handle_creative_all_node_notes_done(pid, node_id, completed, [], '')
        self.w.statusBar().showMessage('已停止本地等待；已完成建议保留，后续响应不会写入。服务端是否停止尚未确认。')
