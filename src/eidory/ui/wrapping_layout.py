"""A compact row that wraps at widget boundaries instead of squeezing controls."""
from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout


class WrappingLayout(QLayout):
    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self.items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item):
        self.items.append(item)

    def addWidget(self, widget):
        # macOS native buttons otherwise report an inset layout-item rectangle;
        # mixing icon and text buttons then produces different visible heights.
        widget.setAttribute(Qt.WidgetAttribute.WA_LayoutUsesWidgetRect, True)
        super().addWidget(widget)

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        return self.items[index] if 0 <= index < len(self.items) else None

    def takeAt(self, index):
        return self.items.pop(index) if 0 <= index < len(self.items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, width, 0), test=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, test=False)

    def sizeHint(self):
        return QSize(sum(i.sizeHint().width()+self.spacing() for i in self.items if not i.isEmpty()),
                     self.heightForWidth(1000))

    def minimumSize(self):
        size = QSize()
        for item in self.items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        return size

    def _arrange(self, rect, *, test):
        rows, row, width, height = [], [], 0, 0
        for item in self.items:
            if item.isEmpty():
                continue
            size = item.sizeHint().expandedTo(item.minimumSize())
            if row and width+size.width() > rect.width():
                rows.append((row, height))
                row, width, height = [], 0, 0
            row.append((item, width, size))
            width += size.width()+self.spacing()
            height = max(height, size.height())
        if row:
            rows.append((row, height))
        y = rect.y()
        for row, height in rows:
            if not test:
                for item, x, size in row:
                    item.setGeometry(QRect(QPoint(rect.x()+x, y+(height-size.height())//2), size))
            y += height+self.spacing()
        return max(0, y-rect.y()-self.spacing())
