"""Keep actions on one row, moving only those that do not fit into a menu."""
from PySide6.QtCore import QEvent, QSize, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLayout, QMenu, QPushButton, QSizePolicy, QWidget


class OverflowToolbar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 4)
        row.setSpacing(6)
        # Hidden actions must not impose their combined minimum width on the
        # window, otherwise it cannot shrink enough to trigger overflow.
        row.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        row.addStretch(1)
        self.more = QPushButton("更多 ▾", self)
        self.more.setAccessibleName("更多看板操作")
        self.menu = QMenu(self.more)
        self.more.setMenu(self.menu)
        row.addWidget(self.more)
        self.more.hide()
        self.overflow = []
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.refresh)
        self.menu.aboutToShow.connect(self.build_menu)

    def add_button(self, button):
        self.layout().insertWidget(self.layout().count() - 2, button)

    def minimumSizeHint(self):
        return QSize(0, self.sizeHint().height())

    def event(self, event):
        result = super().event(event)
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show,
                            QEvent.Type.LayoutRequest, QEvent.Type.FontChange,
                            QEvent.Type.StyleChange) and hasattr(self, "timer"):
            self.timer.start(0)
        return result

    def refresh(self):
        if not self.isVisible():
            return
        row = self.layout()
        buttons = [row.itemAt(i).widget() for i in range(row.count())
                   if isinstance(row.itemAt(i).widget(), QPushButton)
                   and row.itemAt(i).widget() is not self.more]
        widths = [max(b.minimumWidth(), min(b.maximumWidth(), b.sizeHint().width()))
                  for b in buttons]
        margins = row.contentsMargins()
        available = self.width() - margins.left() - margins.right()
        spacing = row.spacing()
        count = len(buttons)
        if sum(widths) + spacing * max(0, count - 1) > available:
            remaining = available - self.more.sizeHint().width()
            count = 0
            for width in widths:
                if width + spacing > remaining:
                    break
                remaining -= width + spacing
                count += 1
        self.overflow = buttons[count:]
        for index, button in enumerate(buttons):
            button.setVisible(index < count)
        self.more.setVisible(bool(self.overflow))

    def build_menu(self):
        self.menu.clear()
        for button in self.overflow:
            action = self.menu.addAction(button.text() or button.accessibleName())
            action.setEnabled(button.isEnabled())
            action.setToolTip(button.toolTip())
            if button.menu():
                action.setMenu(button.menu())
            else:
                action.setCheckable(button.isCheckable())
                action.setChecked(button.isChecked())
                action.triggered.connect(lambda _checked=False, b=button: b.click())
