"""Two-pane comparison with linked zoom and decisions made beside the image."""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QCheckBox, QLabel, QWidget
from eidory.ui.image_preview_dialog import PreviewImageView
from eidory.ui.board_image_loader import BoardImageLoader


class ImageCompareDialog(QDialog):
    def __init__(self, images, *, parent=None):
        super().__init__(parent)
        self.w = parent
        self.images = images
        self.syncing = False
        self.generation = [0, 0]
        self.setWindowTitle('对比参考')
        self.resize(1160, 750)
        self.loader = BoardImageLoader(self)
        self.loader.loaded.connect(self.loaded)
        self.finished.connect(self.release)
        outer = QVBoxLayout(self)
        tools = QHBoxLayout()
        self.link = QCheckBox('同步缩放与移动')
        self.link.setChecked(True)
        tools.addWidget(self.link)
        fit = QPushButton('适应窗口')
        fit.clicked.connect(self.fit)
        tools.addWidget(fit)
        tools.addStretch()
        close = QPushButton('完成')
        close.clicked.connect(self.accept)
        tools.addWidget(close)
        outer.addLayout(tools)
        panels = QHBoxLayout()
        outer.addLayout(panels, 1)
        self.views, self.choices, self.owners, self.grade_buttons = [], [], [], []
        for index in range(2):
            panel = QWidget()
            col = QVBoxLayout(panel)
            col.addWidget(QLabel('基准图' if index == 0 else '候选图'))
            combo = QComboBox()
            combo.setAccessibleName('基准图' if index == 0 else '候选图')
            for image in images:
                combo.addItem(image.file_name, image.id)
            combo.setCurrentIndex(min(index, len(images)-1))
            col.addWidget(combo)
            view = PreviewImageView()
            view.setMinimumSize(200, 240)
            view.setAccessibleName('基准图预览' if index == 0 else '候选图预览')
            view.doubleClicked.connect(lambda i=index: self.w._open_image_preview(self.images[self.choices[i].currentIndex()]))
            view.zoomChanged.connect(lambda value, fit, i=index: self.sync_zoom(i, value, fit))
            for bar in (view.horizontalScrollBar(), view.verticalScrollBar()):
                bar.valueChanged.connect(lambda _, i=index: self.sync_pan(i))
            col.addWidget(view, 1)
            owners = QComboBox()
            owners.setAccessibleName(('基准图' if index == 0 else '候选图') + '的用途所属节点')
            col.addWidget(owners)
            row = QHBoxLayout()
            buttons = []
            for value, title in [('primary', '★ 主参考'), ('candidate', '备选'), ('unused', '暂不采用')]:
                button = QPushButton(title)
                button.setCheckable(True)
                button.toggled.connect(lambda checked, i=index, status=value: self.decide(i, status) if checked else self.refresh_grade(i))
                row.addWidget(button)
                buttons.append(button)
            col.addLayout(row)
            self.views.append(view); self.choices.append(combo); self.owners.append(owners); self.grade_buttons.append(buttons)
            combo.currentIndexChanged.connect(lambda _, i=index: self.load(i))
            owners.currentIndexChanged.connect(lambda _, i=index: self.refresh_grade(i))
            panels.addWidget(panel, 1)
        for i in range(2):
            self.load(i)

    def load(self, index):
        image = self.images[self.choices[index].currentIndex()]
        self.generation[index] += 1
        self.views[index].set_message('正在载入…')
        self.loader.request((index, self.generation[index]), [image.file_path, image.thumbnail_path or ''], 2048, urgent=True)
        combo = self.owners[index]
        combo.blockSignals(True)
        combo.clear()
        pid = self.w.current_creative_project_id
        with self.w.store.connect() as conn:
            rows = conn.execute('SELECT n.id,n.title FROM creative_nodes n JOIN creative_node_images l ON n.id=l.node_id WHERE n.project_id=? AND l.image_id=?', (pid, image.id)).fetchall()
        for row in rows:
            combo.addItem(row['title'], row['id'])
        node = self.w._current_creative_node()
        if not rows and node:
            combo.addItem('收集到「'+node.title+'」', node.id)
        combo.setVisible(combo.count() > 1)
        combo.blockSignals(False)
        self.refresh_grade(index)

    def refresh_grade(self, index):
        iid = self.choices[index].currentData()
        nid = self.owners[index].currentData()
        status = None
        linked = nid is not None and iid in self.w.store.creative_node_image_ids(nid)
        if linked:
            status = self.w.workflow_upgrade.state.decision(nid, iid)['status']
        for button, value in zip(self.grade_buttons[index], ('primary', 'candidate', 'unused')):
            title = {"primary": "★ 主参考", "candidate": "备选", "unused": "暂不采用"}[value]
            button.setText(title if linked or nid is None else "收集为" + title)
            button.setEnabled(nid is not None and (linked or value != "unused"))
            button.blockSignals(True)
            button.setChecked(status == value)
            button.blockSignals(False)
            button.setToolTip('直接保存到当前用途所属节点' if nid else '选择项目后可设置参考等级')

    def decide(self, index, status):
        iid, nid = self.choices[index].currentData(), self.owners[index].currentData()
        if nid is None:
            return
        with self.w.edit_history.record('对比选图与评级'):
            self.w.store.add_images_to_creative_node(nid, [iid])
            old = self.w.workflow_upgrade.state.decision(nid, iid)
            self.w.workflow_upgrade.state.set_decision(nid, iid, status, old['purpose'])
        self.refresh_grade(index)
        self.w.creative_flow.refresh()
        self.w.creative_flow.board_inspector.refresh_badges()

    def loaded(self, token, image, _path):
        index, generation = token
        if generation != self.generation[index]:
            return
        if image.isNull():
            self.views[index].set_message('无法读取图片')
            return
        source = self.images[self.choices[index].currentIndex()]
        self.views[index].set_pixmap(QPixmap.fromImage(image), original_width=source.width, original_height=source.height,
                                    fit_to_window=True, zoom_factor=1.)

    def sync_zoom(self, index, zoom, fit):
        if self.syncing or not self.link.isChecked():
            return
        self.syncing = True
        try:
            other = self.views[1-index]
            if fit:
                other.fit_image_to_window(emit=False)
            else:
                other.set_zoom_factor(zoom, emit=False)
        finally:
            self.syncing = False

    def sync_pan(self, index):
        if self.syncing or not self.link.isChecked() or len(self.views) < 2:
            return
        self.syncing = True
        try:
            ratio = self.views[index]._current_center_ratio()
            if ratio:
                self.views[1-index]._center_on_ratio(ratio)
        finally:
            self.syncing = False

    def fit(self):
        for view in self.views:
            view.fit_image_to_window()

    def done(self, result):
        self.loader.stop()
        super().done(result)

    def release(self):
        # Wait asynchronously for active readers before disposing their pool.
        # Closed comparison windows must not retain decoded image buffers.
        self.loader.stop()
        if self.loader.pool.activeThreadCount():
            QTimer.singleShot(25, self.release)
        else:
            self.deleteLater()

    def closeEvent(self, event):
        self.loader.stop()
        super().closeEvent(event)
