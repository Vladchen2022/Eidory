from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem


class ReferenceEntriesTable(QTableWidget):
    """Row selection picks the detail target; checkboxes only pick search targets."""
    entriesChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self._rows = []
        self._locked = False
        self.setHorizontalHeaderLabels(["找图", "内容 · 用于语义搜索", "细节描述 · 不参与搜索"])
        self.setAccessibleName("参考条目表")
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setWordWrap(True)
        self.verticalHeader().hide()
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(0, 48)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.setColumnWidth(1, 210)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.itemChanged.connect(self._changed)

    def set_entries(self, rows, *, locked=False, detail_target=False):
        signature = (rows, locked, detail_target)
        if getattr(self, "_signature", None) == signature:
            return
        selected_id = (self.selected_entry() or {}).get("id")
        self._signature = deepcopy(signature)
        self._rows, self._locked = deepcopy(rows), locked
        self.blockSignals(True)
        self.setRowCount(len(rows))
        for i, row in enumerate(rows):
            checkbox = QTableWidgetItem()
            checkbox.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable |
                              (Qt.ItemFlag.NoItemFlags if detail_target else Qt.ItemFlag.ItemIsUserCheckable))
            checkbox.setCheckState(Qt.CheckState.Checked if row["enabled"] else Qt.CheckState.Unchecked)
            checkbox.setToolTip("勾选后参与找图；取消勾选不会删除内容，也不会排除含该物件的图片")
            self.setItem(i, 0, checkbox)
            for col, key in ((1, "name"), (2, "visual")):
                item = QTableWidgetItem(row[key])
                if locked or (detail_target and col == 1):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(row[key])
                self.setItem(i, col, item)
            if row["id"] == selected_id:
                self.selectRow(i)
        self.resizeRowsToContents()
        for i in range(self.rowCount()):
            self.setRowHeight(i, max(40, min(110, self.rowHeight(i))))
        self.setFixedHeight(min(330, max(76, self.horizontalHeader().height() + sum(self.rowHeight(i) for i in range(self.rowCount())) + 4)))
        self.blockSignals(False)

    def selected_entry(self):
        row = self.currentRow()
        return deepcopy(self._rows[row]) if 0 <= row < len(self._rows) else None

    def _changed(self, item):
        row, col = item.row(), item.column()
        if not 0 <= row < len(self._rows):
            return
        if col == 0:
            self._rows[row]["enabled"] = item.checkState() == Qt.CheckState.Checked
        else:
            value = item.text().strip()
            if col == 1 and not value:
                self.blockSignals(True)
                item.setText(self._rows[row]["name"])
                self.blockSignals(False)
                return
            self._rows[row]["name" if col == 1 else "visual"] = value
            self.blockSignals(True)
            item.setToolTip(value)
            self.blockSignals(False)
        self._signature = None
        self.entriesChanged.emit(deepcopy(self._rows))
