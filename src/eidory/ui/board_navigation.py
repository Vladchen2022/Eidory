"""Lightweight overview and node jumps; never mutate saved board geometry."""
from weakref import ref
from PySide6.QtCore import QObject, QRectF, Qt, QEvent, QTimer
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QWidget, QPushButton, QMenu
from eidory.ui.theme_colors import theme_color


class BoardMap(QWidget):
    def __init__(self, board):
        super().__init__(board.viewport())
        self.board = board
        self.setFixedSize(160, 105)
        self.setAccessibleName('看板导航图')
        self.setToolTip('点击或拖动，定位画布')
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.bounds = QRectF()
        self.scale = 1.
        self.offset = None

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme_color('base'))
        painter.setPen(QPen(theme_color('border')))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        items = [item for item in self.board._image_items.values() if item.isVisible()]
        self.bounds = QRectF()
        for item in items:
            self.bounds = self.bounds.united(item.sceneBoundingRect())
        if self.bounds.isEmpty():
            return
        self.scale = min((self.width()-12)/self.bounds.width(), (self.height()-12)/self.bounds.height())
        painter.translate(6, 6)
        painter.scale(self.scale, self.scale)
        painter.translate(-self.bounds.left(), -self.bounds.top())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(theme_color('border'))
        for item in items:
            painter.drawRect(item.sceneBoundingRect())
        painter.setPen(QPen(theme_color('accent'), 1.5/self.scale))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.board.mapToScene(self.board.viewport().rect()).boundingRect())

    def mousePressEvent(self, event):
        self.move_to(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.move_to(event)

    def move_to(self, event):
        if self.bounds.isEmpty():
            return
        p = event.position()
        self.board.centerOn(self.bounds.left()+(p.x()-6)/self.scale, self.bounds.top()+(p.y()-6)/self.scale)
        self.board._view_mode = 'manual'
        self.update()


class BoardNavigation(QObject):
    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self.board = window.project_board_view
        self.map = BoardMap(self.board)
        self.map.hide()
        self.map_enabled = True
        self.primary_only = False
        self.button = QPushButton('定位 ▾')
        self.button.setAccessibleName('看板定位与主参考突出显示')
        self.menu = QMenu(self.button)
        self.button.setMenu(self.menu)
        self.menu.aboutToShow.connect(self.build_menu)
        window.workspace_chrome.board_tools.layout().insertWidget(2, self.button)
        self.board.viewport().installEventFilter(self)
        self.board.horizontalScrollBar().valueChanged.connect(lambda _: self.timer.start(0))
        self.board.verticalScrollBar().valueChanged.connect(lambda _: self.timer.start(0))
        self.board.layoutChanged.connect(self.refresh)
        self.board.sectionsChanged.connect(self.refresh)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.refresh)
        window.center_result_stack.currentChanged.connect(self.refresh)

    @property
    def w(self):
        return self._window()

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.Wheel):
            self.timer.start(0)
        return False

    def refresh(self, *_):
        self.map.move(max(0, self.board.viewport().width()-self.map.width()-14), max(0, self.board.viewport().height()-self.map.height()-14))
        self.map.setVisible(self.board.isVisible() and self.map_enabled and len(self.board._image_items) >= 12)
        self.map.raise_()
        self.map.update()
        self.button.setText("主参考突出 ▾" if self.primary_only else "定位 ▾")
        has_primary = any("主参考" in labels for labels in self.board._board_badges.values())
        for iid, item in self.board._image_items.items():
            primary = any('主参考' in label for label in self.board._board_badges.get(iid, []))
            item.setOpacity(1.0 if not self.primary_only or not has_primary or primary else .25)

    def build_menu(self):
        self.menu.clear()
        self.menu.addAction('适应全部', self.board.fit_all_images)
        action = self.menu.addAction('突出主参考')
        action.setCheckable(True)
        action.setChecked(self.primary_only)
        action.toggled.connect(self.set_primary)
        action = self.menu.addAction('显示导航图（12 张以上）')
        action.setCheckable(True)
        action.setChecked(self.map_enabled)
        action.toggled.connect(self.set_map)
        groups = {}
        for iid in self.board._image_items:
            for name in self.board._board_badges.get(iid, [])[:1]:
                groups.setdefault(name, []).append(iid)
        if groups:
            self.menu.addSeparator()
        for title, ids in groups.items():
            self.menu.addAction(f'{title} · {len(ids)} 张', lambda _=False, values=ids: self.focus(values))

    def focus(self, ids):
        items = [self.board._image_items[i] for i in ids if i in self.board._image_items]
        self.board._fit_selected_items(items)
        self.map.update()

    def set_primary(self, checked):
        self.primary_only = checked
        self.refresh()

    def set_map(self, checked):
        self.map_enabled = checked
        self.refresh()
