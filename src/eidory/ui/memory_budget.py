"""Share a bounded image-cache budget between the wall and the active board."""
from weakref import ref
from PySide6.QtCore import QObject, QTimer


class ImageMemoryBudget(QObject):
    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self.budget = 240 * 1024 * 1024
        self.timer = QTimer(self)
        self.timer.setInterval(8000)
        self.timer.timeout.connect(self.trim)
        self.timer.start()
        window.center_result_stack.currentChanged.connect(self.schedule)
        self.idle = QTimer(self)
        self.idle.setSingleShot(True)
        self.idle.setInterval(2500)
        self.idle.timeout.connect(self.trim)

    def schedule(self, *_):
        self.idle.start()

    def trim(self):
        w = self._window()
        if w._closing:
            self.timer.stop()
            self.idle.stop()
            return
        grid, board = w.grid_view, w.project_board_view
        grid_budget = 96 * 1024 * 1024 if grid.isVisible() else 16 * 1024 * 1024
        grid._cache_byte_limit = grid_budget
        while grid._pixmap_cache and grid._pixmap_cache_total_bytes > grid_budget:
            key, _ = grid._pixmap_cache.popitem(last=False)
            grid._pixmap_cache_total_bytes -= grid._pixmap_cache_bytes.pop(key, 0)
        from eidory.ui.project_board import trim_pixmap_cache
        trim_pixmap_cache(24 * 1024 * 1024 if board.isVisible() else 0)
        board.trim_image_memory(self.budget - grid_budget - 24 * 1024 * 1024)
