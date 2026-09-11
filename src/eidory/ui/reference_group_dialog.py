from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QLabel, QPushButton, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
)

from eidory.ui.justified_image_grid import JustifiedImageGridView


class ReferenceGroupDialog(QDialog):
    """Review names and membership before creating reference collections."""
    def __init__(self, store, pairs, error_message='', parent=None):
        super().__init__(parent)
        self.store, self.pairs = store, list(pairs)
        self.setWindowTitle('整理 AI 分组')
        self.resize(980, 650)
        layout = QVBoxLayout(self)
        label = QLabel('双击组名可改名。选择图片或整组后，可移到另一组；确认后才保存。')
        label.setWordWrap(True)
        layout.addWidget(label)
        if error_message:
            warning = QLabel('AI 命名失败，已使用临时名称；可以直接修改后保存。')
            layout.addWidget(warning)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['分组 / 图片', '张数'])
        self.tree.setColumnWidth(0, 250)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.preview = JustifiedImageGridView(thumbnail_size=150)
        split = QSplitter()
        split.addWidget(self.tree)
        split.addWidget(self.preview)
        split.setSizes([330, 610])
        layout.addWidget(split, 1)
        image_ids = [i for group, _ in pairs for i in group.image_ids]
        self.images = {im.id: im for im in store.images_by_ids(image_ids)}
        for index, (group, suggestion) in enumerate(pairs):
            item = QTreeWidgetItem([suggestion.name, str(len(group.image_ids))])
            item.setData(0, Qt.ItemDataRole.UserRole, index)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.tree.addTopLevelItem(item)
            for image_id in group.image_ids:
                image = self.images.get(image_id)
                child = QTreeWidgetItem([image.file_name if image else str(image_id), ''])
                child.setData(0, Qt.ItemDataRole.UserRole, image_id)
                item.addChild(child)
        self.tree.expandAll()
        row = QHBoxLayout()
        row.addWidget(QLabel('移入 / 合并到'))
        self.target = QComboBox()
        row.addWidget(self.target, 1)
        move = QPushButton('移动选中的图片或组')
        move.clicked.connect(self.move_selected)
        row.addWidget(move)
        layout.addLayout(row)
        self.tree.itemChanged.connect(self.refresh_targets)
        self.tree.itemSelectionChanged.connect(self.refresh_preview)
        self.refresh_targets()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))

    def refresh_targets(self, *_args):
        current = self.target.currentData()
        self.target.clear()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            self.target.addItem(item.text(0), i)
        index = self.target.findData(current)
        if index >= 0:
            self.target.setCurrentIndex(index)

    def selected_leaves(self):
        selected = []
        for item in self.tree.selectedItems():
            children = [item.child(i) for i in range(item.childCount())] if item.parent() is None else [item]
            for child in children:
                if child not in selected:
                    selected.append(child)
        return selected

    def refresh_preview(self):
        ids = [item.data(0, Qt.ItemDataRole.UserRole) for item in self.selected_leaves()]
        self.preview.set_images([self.images[i] for i in ids if i in self.images])

    def move_selected(self):
        index = self.target.currentData()
        if index is None:
            return
        target = self.tree.topLevelItem(index)
        self.tree.blockSignals(True)
        for child in self.selected_leaves():
            source = child.parent()
            if source is not target:
                source.takeChild(source.indexOfChild(child))
                target.addChild(child)
                source.setText(1, str(source.childCount()))
        target.setText(1, str(target.childCount()))
        self.tree.blockSignals(False)
        self.tree.setCurrentItem(target)
        self.refresh_preview()

    def result_pairs(self):
        pairs = []
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            ids = [item.child(i).data(0, Qt.ItemDataRole.UserRole) for i in range(item.childCount())]
            if not ids:
                continue
            group, suggestion = self.pairs[index]
            pairs.append((replace(group, image_ids=ids, representative_id=ids[0]),
                          replace(suggestion, name=item.text(0).strip() or f'参考组 {index+1}')))
        return pairs
