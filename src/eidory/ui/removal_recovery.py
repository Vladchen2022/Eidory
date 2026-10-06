"""One recoverable library history, including the ordinary Cmd+Z command."""
from weakref import ref
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout
from eidory.core.removal_archive import RemovalArchive


class RemovalRecovery:
    def __init__(self, window):
        self._window = ref(window)
        self.archive = RemovalArchive(window.store, window.paths.data_dir)
        latest = self.archive.records()
        if latest:
            window._last_removal_undo = latest[0]
            window.undo_removal_action.setEnabled(True)
        layout = window.workspace_chrome._dialogs[4].layout()
        button = QPushButton('回收区…')
        button.setToolTip('恢复本次或之前删除的图片、标签与项目关联；退出应用后仍保留')
        button.clicked.connect(self.open)
        layout.insertWidget(2, button)

    @property
    def w(self):
        return self._window()

    def open(self):
        dialog = QDialog(self.w)
        dialog.setWindowTitle('回收区 · 恢复图片及原有信息')
        dialog.resize(760, 560)
        layout = QVBoxLayout(dialog)
        info = QLabel('删除记录跨重启保留。恢复图片、标签、AI 信息和项目关联；原路径被占用时不覆盖。')
        info.setWordWrap(True)
        layout.addWidget(info)
        listing = QListWidget()
        listing.setAccessibleName('回收区删除记录')
        layout.addWidget(listing)
        detail = QLabel()
        detail.setWordWrap(True)
        layout.addWidget(detail)
        records = []
        def reload():
            records[:] = self.archive.records()
            listing.clear()
            for record in records:
                images = record.get('snapshot', {}).get('images', [])
                name = images[0]['file_name'] if images else ''
                item = QListWidgetItem(record['label'] + ' · ' + name + '\n' + record['created_at'])
                if images and images[0].get('thumbnail_path'):
                    item.setIcon(QIcon(images[0]['thumbnail_path']))
                listing.addItem(item)
            if records:
                listing.setCurrentRow(0)
            else:
                detail.setText('没有待恢复的删除记录。')
        def selected():
            i = listing.currentRow()
            if 0 <= i < len(records):
                record = records[i]
                images = record.get('snapshot', {}).get('images', [])
                detail.setText('\n'.join(row['file_name'] for row in images[:5]) + ('\n……' if len(images) > 5 else '') + f'\n共 {len(record["image_ids"])} 张，恢复到各自原位置。')
                restore.setEnabled(True)
            else:
                restore.setEnabled(False)
        def recover():
            i = listing.currentRow()
            if 0 <= i < len(records):
                self.w._last_removal_undo = records[i]
                self.w._undo_last_library_removal(from_archive=True)
                reload()
                selected()
        actions = QHBoxLayout()
        restore = QPushButton('恢复所选记录')
        restore.setProperty('variant', 'primary')
        restore.clicked.connect(recover)
        actions.addWidget(restore)
        close = QPushButton('关闭')
        close.clicked.connect(dialog.accept)
        actions.addWidget(close)
        layout.addLayout(actions)
        listing.currentRowChanged.connect(selected)
        reload()
        selected()
        dialog.exec()
