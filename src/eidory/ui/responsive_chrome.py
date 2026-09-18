"""Move secondary search options into an existing-row menu on small windows."""
from weakref import ref
from PySide6.QtCore import QObject, QEvent, QTimer
from PySide6.QtWidgets import QPushButton, QMenu, QSizePolicy, QLabel


class ResponsiveChrome(QObject):
    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self.more = QPushButton('显示选项 ▾')
        self.menu = QMenu(self.more)
        self.more.setMenu(self.menu)
        self.menu.aboutToShow.connect(self.build_menu)
        window.filter_panel.scope_row.addWidget(self.more)
        self.more.hide()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.refresh)
        window.installEventFilter(self)
        window.root_splitter.widget(1).installEventFilter(self)
        for label in (window.creative_workspace.context_label, window.workflow_upgrade.result_hint, window.creative_flow.search_context):
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.refresh()

    @property
    def w(self):
        return self._window()

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.timer.start(0)
        return False

    def refresh(self):
        w = self.w
        narrow = w.root_splitter.widget(1).width() < 850
        for widget in (w.saved_view_combo, w.sort_combo, w.sort_order_combo):
            widget.setVisible(not narrow)
        # The sort caption belongs to the same row, not the sidebar form.
        row = w.filter_panel.scope_row
        for i in range(row.count()):
            widget = row.itemAt(i).widget()
            if isinstance(widget, QLabel) and widget.text() == '排序':
                widget.setVisible(not narrow)
        self.more.setVisible(narrow)
        w.creative_project_combo.setMaximumWidth(210 if w.width() < 1250 else 340)
        w.workflow_upgrade.result_hint.setToolTip(w.workflow_upgrade.result_hint.text())
        w.creative_workspace.context_label.setToolTip(w.creative_workspace.context_label.text())
        w.thumbnail_size_slider.setMaximumWidth(100 if narrow else 180)
        w.thumbnail_size_label.setVisible(not narrow and w.grid_view.isVisible())

    def build_menu(self):
        self.menu.clear()
        self.submenus = []
        for title, combo in [('排序', self.w.sort_combo), ('顺序', self.w.sort_order_combo), ('筛选预设', self.w.saved_view_combo)]:
            menu = QMenu(title, self.menu)
            self.menu.addMenu(menu)
            self.submenus.append(menu)
            for i in range(combo.count()):
                action = menu.addAction(combo.itemText(i), lambda _=False, box=combo, index=i: box.setCurrentIndex(index))
                action.setCheckable(True)
                action.setChecked(i == combo.currentIndex())
                action.setEnabled(combo.isEnabled())
