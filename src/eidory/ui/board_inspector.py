"""Edit selected board references in place, without a second image collection."""
from weakref import ref

from PySide6.QtCore import QObject, QSignalBlocker, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QTextEdit, QVBoxLayout, QWidget

from eidory.core.project_state import PURPOSES, REFERENCE_STATUSES


class BoardInspector(QObject):
    def __init__(self, window, parent_stack):
        super().__init__(window)
        self._window = ref(window)
        self.refreshing = False
        self._selection_key = None
        self._links = []
        self.page = QWidget()
        self.page.setObjectName('boardSelectionInspector')
        layout = QVBoxLayout(self.page)
        layout.setContentsMargins(8, 10, 8, 8)
        self.heading = self.label('看板 · 图片属性', layout)
        self.heading.setProperty('role', 'flowTitle')
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMaximumHeight(150)
        layout.addWidget(self.preview)
        self.info = self.label('在看板中选择图片，直接调整参考等级和用途。', layout)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.scope_label = self.label('用途所属节点', layout)
        self.scope = QComboBox()
        self.scope.setAccessibleName('看板参考的用途所属节点')
        layout.addWidget(self.scope)
        self.grade_label = self.label('参考等级', layout)
        self.grade = QComboBox()
        self.grade.setAccessibleName('看板所选图片的参考等级')
        self.grade.addItem('等级不同', None)
        for value, title in REFERENCE_STATUSES.items():
            self.grade.addItem(('★ ' if value == 'primary' else '') + title, value)
        layout.addWidget(self.grade)
        self.purpose_label = self.label('参考用途', layout)
        self.purpose = QComboBox()
        self.purpose.setAccessibleName('看板所选图片的参考用途')
        self.purpose.addItem('用途不同', None)
        for value in PURPOSES:
            self.purpose.addItem(value or '未指定用途', value)
        layout.addWidget(self.purpose)
        self.note_label = self.label('用途说明', layout)
        self.note = QTextEdit()
        self.note.setAcceptRichText(False)
        self.note.setAccessibleName('看板参考用途说明')
        self.note.setPlaceholderText('例如：只参考头冠造型，不参考配色。')
        self.note.setMaximumHeight(140)
        layout.addWidget(self.note)
        self.hint = self.label('', layout)
        self.hint.setProperty('role', 'muted')
        self.open_button = QPushButton('查看大图')
        self.open_button.clicked.connect(self.open_image)
        layout.addWidget(self.open_button)
        layout.addStretch(1)
        parent_stack.addWidget(self.page)

        self.actions = QWidget()
        self.actions.setObjectName('boardReferenceActions')
        self.actions.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(self.actions)
        row.setContentsMargins(0, 0, 0, 4)
        row.setSpacing(6)
        self.selection_label = QLabel('先在看板选图')
        row.addWidget(self.selection_label)
        self.quick_grades = {}
        for status, title in REFERENCE_STATUSES.items():
            b = QPushButton(('★ ' if status == 'primary' else '') + title)
            b.setAccessibleName('看板设为' + title)
            b.setCheckable(True)
            # macOS accessibility activates checkable buttons through toggled.
            b.toggled.connect(lambda _checked, value=status: self.save(status=value))
            row.addWidget(b)
            self.quick_grades[status] = b
        self.remove_button = QPushButton('移出看板')
        self.remove_button.setToolTip('移除所选图片在当前看板分支中的关联；保留图库源文件，可撤销。')
        self.remove_button.clicked.connect(lambda: self.w._remove_images_from_current_board(self.selected_ids()))
        row.addWidget(self.remove_button)
        row.addStretch(1)
        center_layout = window.center_result_stack.parentWidget().layout()
        center_layout.insertWidget(center_layout.indexOf(window.workspace_chrome.board_tools) + 1, self.actions)
        self.actions.hide()
        undo = window.creative_workspace.board_buttons[2]
        undo.clicked.disconnect()
        undo.clicked.connect(self.undo)
        undo.setToolTip('撤销看板布局编辑；没有布局编辑时，恢复上次移出的图片。')
        self.scope.currentIndexChanged.connect(self.scope_changed)
        self.grade.currentIndexChanged.connect(lambda: self.save(status=self.grade.currentData()))
        self.purpose.currentIndexChanged.connect(lambda: self.save(purpose=self.purpose.currentData()))
        self.note.textChanged.connect(self.save_note)
        window.project_board_view.selectionChanged.connect(self.refresh)
        window.project_board_view.sectionsChanged.connect(self.sections_changed)
        window.project_board_view.layoutChanged.connect(self.refresh)
        section_button = window.creative_workspace.board_buttons[1]
        section_button.clicked.disconnect()
        section_button.toggled.connect(lambda _: window.project_board_view.toggle_node_sections())
        self.sections_changed(bool(window.project_board_view._section_items))

    @property
    def w(self):
        return self._window()

    @staticmethod
    def label(text, layout):
        label = QLabel(text)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        layout.addWidget(label)
        return label

    def selected_ids(self):
        return self.w.project_board_view.selected_image_ids()

    def sections_changed(self, grouped):
        button = self.w.creative_workspace.board_buttons[1]
        button.setCheckable(True)
        with QSignalBlocker(button):
            button.setChecked(grouped)
        button.setText('取消分区 · 自由排列' if grouped else '按节点分区')
        button.setToolTip('仅移除分区框，保留图片当前位置；可撤销。' if grouped else '按节点整理图片；再次点击可取消分区。')

    def active_links(self):
        scope = self.scope.currentData()
        if scope is None:
            return []
        return [link for link in self._links if scope == 'all' or link['node_id'] == scope]

    def refresh(self, *_):
        if self.refreshing:
            return
        w = self.w
        if w.center_result_stack.currentWidget() is not w.project_board_view:
            return
        self.refreshing = True
        try:
            ids = self.selected_ids()
            self.selection_label.setText(f'已选 {len(ids)} 张' if ids else '先在看板选图')
            node = w.store.get_creative_node(w._current_board_node_id) if w._current_board_node_id else None
            self._links = w.store.creative_node_image_links_for_branch(node.id, ids) if node and ids else []
            selection_key = (w._current_board_node_id, w._current_board_temp_project_id, tuple(sorted(ids)),
                             tuple((r['node_id'], r['image_id']) for r in self._links))
            if selection_key != self._selection_key:
                self._selection_key = selection_key
                self.scope.clear()
                owners = sorted({r['node_id'] for r in self._links})
                ambiguous = any(sum(r['image_id'] == i for r in self._links) > 1 for i in ids)
                if ambiguous:
                    self.scope.addItem('请选择要调整的节点', None)
                if len(owners) > 1:
                    self.scope.addItem('当前看板内的全部用途', 'all')
                root_id = w.store.creative_root_node_id(node.project_id) if node else None
                root_path = w._creative_node_path_text(root_id) if root_id else ''
                for nid in owners:
                    full_path = w._creative_node_path_text(nid)
                    path = full_path.removeprefix(root_path + ' / ')
                    self.scope.addItem('项目根节点' if nid == root_id else path, nid)
                    self.scope.setItemData(self.scope.count() - 1, full_path, Qt.ItemDataRole.ToolTipRole)
                if node and node.id in owners and len(ids) == 1:
                    self.scope.setCurrentIndex(self.scope.findData(node.id))
                self.preview.clear()
                image = w._board_image_by_id.get(ids[0]) if len(ids) == 1 else None
                if image:
                    self.info.setText(f'{image.file_name}\n{image.width} × {image.height} · {image.file_size / 1024:.0f} KB\n{image.file_path}')
                    pixmap = QPixmap(image.thumbnail_path or '')
                    self.preview.setPixmap(pixmap.scaled(240, 140, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                else:
                    self.info.setText(f'已选 {len(ids)} 张，可批量设置参考等级和用途。' if ids else '在看板中选择图片，直接调整参考等级和用途。')
                self.preview.setVisible(image is not None)
            active = self.active_links()
            decisions = [w.workflow_upgrade.state.decision(r['node_id'], r['image_id']) for r in active]
            statuses = {d['status'] for d in decisions}
            purposes = {d['purpose'] for d in decisions}
            status = next(iter(statuses)) if len(statuses) == 1 else None
            purpose = next(iter(purposes)) if len(purposes) == 1 else None
            self.grade.setItemText(0, '等级不同' if active else '先选择图片与所属节点')
            self.purpose.setItemText(0, '用途不同' if active else '先选择图片与所属节点')
            self.grade.setCurrentIndex(self.grade.findData(status))
            self.purpose.setCurrentIndex(self.purpose.findData(purpose))
            self.scope.setEnabled(bool(self._links))
            self.grade.setEnabled(bool(active))
            self.purpose.setEnabled(bool(active))
            for value, b in self.quick_grades.items():
                b.setEnabled(bool(active))
                b.setChecked(value == status)
                b.setToolTip('直接设置所选图片的参考等级，保留看板视角。' if active else '先在看板选图；多处使用的图片需在右侧选择用途所属节点。')
            self.remove_button.setEnabled(bool(ids))
            self.open_button.setEnabled(len(ids) == 1)
            note_key = (active[0]['node_id'], active[0]['image_id']) if len(active) == 1 else None
            text = w.creative_workspace.store.reference_note(*note_key) if note_key else ''
            if self.note.toPlainText() != text:
                self.note.setPlainText(text)
            self.note.setEnabled(note_key is not None)
            if ids and not node:
                hint = '此看板不属于创作节点；参考等级按创作节点记录。'
            elif self._links and not active:
                hint = '图片用于多个节点，请先选择要修改哪一处。'
            elif len(active) > 1:
                hint = f'修改将应用到 {len(active)} 处参考关联；用途说明请单独选图编辑。'
            else:
                hint = '修改即时保存。红色边框与 ★ 标记表示主参考。' if active else ''
            self.hint.setText(hint)
        finally:
            self.refreshing = False

    def scope_changed(self, *_):
        if not self.refreshing:
            self.refresh()

    def save(self, *, status=None, purpose=None):
        if self.refreshing or (status is None and purpose is None):
            return
        active = self.active_links()
        if not active:
            return
        # One transaction preserves per-node purposes when only grade changes.
        with self.w.store.connect() as conn:
            for link in active:
                nid, iid = link['node_id'], link['image_id']
                old = self.w.workflow_upgrade.state.decision(nid, iid)
                conn.execute('''INSERT INTO creative_reference_decisions(node_id,image_id,status,purpose)
                    SELECT ?,?,?,? WHERE EXISTS(SELECT 1 FROM creative_node_images WHERE node_id=? AND image_id=?)
                    ON CONFLICT(node_id,image_id) DO UPDATE SET status=excluded.status,purpose=excluded.purpose''',
                    (nid, iid, status or old['status'], old['purpose'] if purpose is None else purpose, nid, iid))
        self.refresh_badges()
        self.refresh()
        self.w.statusBar().showMessage('已保存所选图片的参考等级与用途')

    def refresh_badges(self):
        w = self.w
        node = w.store.get_creative_node(w._current_board_node_id) if w._current_board_node_id else None
        if not node:
            return
        data = w.store.creative_node_branch_board_data(node.id)
        badges = data['badges']
        links = {(r['node_id'], r['image_id']) for r in w.store.creative_node_image_links_for_branch(node.id, list(badges))}
        for r in w.creative_workspace.store.reference_notes(node.project_id):
            if r['note'].strip() and (r['node_id'], r['image_id']) in links:
                badges.setdefault(r['image_id'], []).append(r['note'][:90])
        w.workflow_upgrade.decorate_references(node.project_id, badges, node.id)
        w.project_board_view.set_reference_badges(badges)
        w.creative_flow._tray_key = None

    def save_note(self):
        if self.refreshing:
            return
        links = self.active_links()
        if len(links) == 1:
            self.w.creative_workspace.store.save_reference_note(links[0]['node_id'], links[0]['image_id'], self.note.toPlainText())

    def open_image(self):
        ids = self.selected_ids()
        if len(ids) == 1:
            self.w._open_project_board_image_preview(ids[0])

    def undo(self):
        if not self.w.project_board_view.undo_layout():
            self.w._undo_last_board_removal()
