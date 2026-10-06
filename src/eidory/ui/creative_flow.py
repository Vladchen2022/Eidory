"""Task-focused presentation of the existing creative workflow.

No second project model: cards, search targets and the tray all use the same
stores and commands as the library, structured editor and board.
"""
from __future__ import annotations

from eidory.ui.edit_history import undoable

from copy import deepcopy
from functools import wraps
import json
from weakref import ref

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QPushButton, QSlider,
    QScrollArea, QSizePolicy, QStackedWidget, QTextEdit, QVBoxLayout, QWidget,
)

from eidory.core.creative_entries import decode_entries, encode_entries, composed_note, search_names
from eidory.core.creative_templates import CREATIVE_TEMPLATES
from eidory.core.search_targets import resolved_targets, target_signature
from eidory.ui.inline_idea import InlineIdeaEditor, change_html


def label(text, *, role=None):
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setWordWrap(True)
    widget.setMinimumWidth(0)
    widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
    if role:
        widget.setProperty('role', role)
    return widget


def button(text, callback, layout, *, primary=False):
    widget = QPushButton(text)
    widget.clicked.connect(callback)
    if primary:
        widget.setProperty('variant', 'primary')
    layout.addWidget(widget)
    return widget


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().hide()
            item.widget().deleteLater()
        elif item.layout():
            clear_layout(item.layout())


def batch_navigation_refresh(method):
    """Refresh the final page once after synchronous navigation signals settle."""
    @wraps(method)
    def navigate(self, *args, **kwargs):
        self._refresh_batch_depth = getattr(self, '_refresh_batch_depth', 0) + 1
        try:
            return method(self, *args, **kwargs)
        finally:
            self._refresh_batch_depth -= 1
            if not self._refresh_batch_depth:
                self.refresh()
    return navigate


class CreativeFlow(QObject):
    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self.refreshing = False
        self.routing = False
        self._cards_key = self._entries_key = self._targets_key = self._tray_key = None
        self._last_project = window.current_creative_project_id
        self._search_dirty_node = None
        self._result_node_id = None
        self._removed = None
        self._inline = None
        self._inline_key = None
        self._target_widgets = {}
        self._start_hides_inspector = False
        self._build_home()
        self._refine_editor()
        self._build_reference_panel()
        self._build_reference_grades()
        self._build_navigation()
        window.center_result_stack.currentChanged.connect(self.refresh)
        window.grid_view.selectionChanged.connect(self.refresh)
        window.creative_node_tree.currentItemChanged.connect(self.node_selected)
        self.refresh()

    @property
    def w(self):
        return self._window()

    @property
    def cw(self):
        return self.w.creative_workspace

    def _build_home(self):
        self.home = QScrollArea()
        self.home.setWidgetResizable(True)
        self.home.setObjectName('creativeHome')
        self.home_stack = QStackedWidget()
        self.home.setWidget(self.home_stack)
        self.w.center_result_stack.addWidget(self.home)
        self.search_empty = QWidget()
        empty = QVBoxLayout(self.search_empty)
        empty.setContentsMargins(28, 28, 28, 28)
        empty.setAlignment(Qt.AlignmentFlag.AlignTop)
        empty.addWidget(label('先选一个找图目标', role='flowTitle'))
        empty.addWidget(label('在上方添加想找的内容，或返回构思写下已经想到的内容。', role='muted'))
        self.w.center_result_stack.addWidget(self.search_empty)
        self.start = QWidget()
        layout = QVBoxLayout(self.start)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(label('你想创作什么？', role='flowTitle'))
        layout.addWidget(label('从一句主题开始，也可以先浏览图库找灵感。', role='muted'))
        self.start_brief = QTextEdit()
        self.start_brief.setAcceptRichText(False)
        self.start_brief.setAccessibleName('新项目的创作主题')
        self.start_brief.setPlaceholderText('例如：赛博朋克研发室，三位工程师围绕新款机器人进行改造。')
        self.start_brief.setFixedHeight(112)
        layout.addWidget(self.start_brief)
        row = QHBoxLayout()
        row.addWidget(label('创作类型'))
        self.start_template = QComboBox()
        self.start_template.setAccessibleName('新项目创作类型')
        for template in CREATIVE_TEMPLATES:
            self.start_template.addItem(template.label, template.id)
        row.addWidget(self.start_template, 1)
        layout.addLayout(row)
        self.start_extra_toggle = button('补充已经确定的内容 ▸', self.toggle_start_extra, layout)
        self.start_extra = QTextEdit()
        self.start_extra.setAcceptRichText(False)
        self.start_extra.setAccessibleName('新项目的补充信息')
        self.start_extra.setPlaceholderText('时代、地点、人物数量，或者不希望出现的内容。可以留空。')
        self.start_extra.setFixedHeight(100)
        self.start_extra.hide()
        layout.addWidget(self.start_extra)
        actions = QHBoxLayout()
        self.create_button = button('开始构思', self.create_project, actions, primary=True)
        self.create_button.setEnabled(False)
        self.start_brief.textChanged.connect(lambda: self.create_button.setEnabled(bool(self.start_brief.toPlainText().strip())))
        button('让 AI 提供主题', self.suggest_project, actions)
        actions.addStretch()
        layout.addLayout(actions)
        layout.addStretch(1)
        self.home_stack.addWidget(self.start)
        self.overview = QWidget()
        outer = QVBoxLayout(self.overview)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(14)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)
        heading = QHBoxLayout()
        self.project_title = label('', role='flowTitle')
        heading.addWidget(self.project_title, 1)
        outer.addLayout(heading)
        self.project_brief = label('', role='muted')
        outer.addWidget(self.project_brief)
        self.cards_widget = QWidget()
        self.cards_widget.setMinimumWidth(0)
        self.cards_layout = QGridLayout(self.cards_widget)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(12)
        outer.addWidget(self.cards_widget)
        self.overview.installEventFilter(self)
        self.home_stack.addWidget(self.overview)

    def toggle_start_extra(self):
        shown = self.start_extra.isHidden()
        self.start_extra.setVisible(shown)
        self.start_extra_toggle.setText('收起补充内容 ▾' if shown else '补充已经确定的内容 ▸')

    def create_project(self):
        brief = self.start_brief.toPlainText().strip()
        if not brief:
            return
        w = self.w
        w.creative_template_combo.setCurrentIndex(w.creative_template_combo.findData(self.start_template.currentData()))
        w._create_creative_project_from_seed(title=brief[:40], brief=brief, extra=self.start_extra.toPlainText().strip())
        self.show_overview()

    def resume_project(self, project_id):
        self.w._refresh_creative_projects(select_project_id=project_id)
        self.show_overview()

    def suggest_project(self):
        w = self.w
        w.creative_template_combo.setCurrentIndex(w.creative_template_combo.findData(self.start_template.currentData()))
        w.inspiration_brief_input.setPlainText(self.start_brief.toPlainText())
        w.inspiration_answers_input.setPlainText(self.start_extra.toPlainText())
        w.creative_ai_project_button.click()

    def browse_library(self):
        self.w.workspace_chrome.navigation_tabs.setCurrentIndex(0)
        self.w._show_gallery_view()
        self.w.right_tab_widget.setCurrentIndex(0)

    def _refine_editor(self):
        w, cw = self.w, self.cw
        self.editor_top = QWidget()
        row = QHBoxLayout(self.editor_top)
        row.setContentsMargins(0, 0, 0, 0)
        self.editor_title = label('', role='flowTitle')
        row.addWidget(self.editor_title, 1)
        button('内容选项', self.editor_options, row)
        cw.planning.layout().insertWidget(0, self.editor_top)
        node_layout = w.creative_node_note_input.parentWidget().layout()
        node_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # A stacked page fills its container. Consume spare height at the end,
        # instead of distributing it between the compact editor controls.
        node_layout.addStretch(1)
        for text in cw.planning.findChildren(QLabel):
            text.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.base_summary = label('')
        self.base_summary.setObjectName('creativeBaseSummary')
        node_layout.insertWidget(node_layout.indexOf(w.creative_node_note_input), self.base_summary)
        self.idea_cards = QWidget()
        self.idea_cards.setObjectName('creativeIdeaCards')
        self.idea_layout = QVBoxLayout(self.idea_cards)
        self.idea_layout.setContentsMargins(0, 0, 0, 0)
        self.idea_layout.setSpacing(8)
        cw.entries_panel.layout().insertWidget(1, self.idea_cards)
        self.pending_cards = QWidget()
        self.pending_layout = QVBoxLayout(self.pending_cards)
        self.pending_layout.setContentsMargins(0, 0, 0, 0)
        self.pending_layout.setSpacing(8)
        cw.proposal_panel.layout().insertWidget(1, self.pending_cards)
        self.prose_diff = label('')
        self.prose_diff.setTextFormat(Qt.TextFormat.RichText)
        self.prose_diff.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cw.proposal_panel.layout().insertWidget(2, self.prose_diff)
        self.edit_prose = QCheckBox('编辑建议文字')
        self.edit_prose.toggled.connect(cw.proposal_input.setVisible)
        cw.proposal_panel.layout().insertWidget(3, self.edit_prose)
        cw.proposal_input.textChanged.connect(self.refresh_prose_diff)
        self.node_caption = next((node_layout.itemAt(i).widget() for i in range(node_layout.count())
                                  if isinstance(node_layout.itemAt(i).widget(), QLabel)
                                  and node_layout.itemAt(i).widget().text() == '当前节点说明'), None)
        self._project_actions = w.creative_template_combo.parentWidget()
        # A single search-selection surface lives beside the image wall.
        w.creative_node_note_input.setAccessibleName('我的创作设定')
        w.generate_creative_children_button.setAccessibleName('AI 完善当前构思')
        w.generate_creative_details_button.setAccessibleName('AI 补充细节')

    def editor_options(self):
        dialog = QDialog(self.w)
        dialog.setWindowTitle('内容形式与恢复')
        layout = QVBoxLayout(dialog)
        combo = QComboBox()
        for i in range(self.cw.format_combo.count()):
            combo.addItem(self.cw.format_combo.itemText(i), self.cw.format_combo.itemData(i))
        combo.setCurrentIndex(self.cw.format_combo.currentIndex())
        combo.setEnabled(self.cw.format_combo.isEnabled())
        layout.addWidget(combo)
        button('切换内容形式', lambda: (self.cw.format_combo.setCurrentIndex(combo.currentIndex()), dialog.accept()), layout)
        restore = button('恢复上次采纳前的内容', lambda: (self.cw.restore_button.click(), dialog.accept()), layout)
        restore.setEnabled(self.cw.restore_button.isEnabled())
        button('关闭', dialog.reject, layout)
        dialog.exec()

    def _build_navigation(self):
        w, cw = self.w, self.cw
        self.overview_button = button('项目全貌', self.show_overview, cw.navigation.layout())
        # Keep the collection view command available in the tray, without a
        # fourth top-level stage competing with search and the board.
        cw.collected_button.hide()
        self.w.workspace_chrome.node_heading.setText('创作内容')
        w.ai_probe_mode_button.setText('探索灵感')
        w.creative_node_tree.setHeaderLabels(['创作内容', '参考'])

    def _build_reference_panel(self):
        self.reference_page = QStackedWidget()
        self.reference_collection = QWidget()
        self.reference_scroll = QScrollArea()
        self.reference_scroll.setWidgetResizable(True)
        self.reference_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.reference_scroll.setWidget(self.reference_collection)
        self.reference_page.addWidget(self.reference_scroll)
        layout = QVBoxLayout(self.reference_collection)
        layout.setContentsMargins(8, 10, 8, 8)
        self.reference_title = label('创作参考', role='flowTitle')
        layout.addWidget(self.reference_title)
        self.reference_context = label('', role='muted')
        layout.addWidget(self.reference_context)
        self.target_area = QWidget()
        targets = QVBoxLayout(self.target_area)
        targets.setContentsMargins(0, 8, 0, 6)
        targets.addWidget(label('这次找什么？', role='sectionTitle'))
        self.target_scroll = QScrollArea()
        self.target_scroll.setWidgetResizable(True)
        self.target_scroll.setMinimumHeight(100)
        self.target_scroll.setMaximumHeight(260)
        self.target_content = QWidget()
        self.target_layout = QVBoxLayout(self.target_content)
        self.target_layout.setContentsMargins(0, 0, 4, 0)
        self.target_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.target_scroll.setWidget(self.target_content)
        targets.addWidget(self.target_scroll)
        self.target_hint = label('各项分别找图，结果合并显示。', role='muted')
        targets.addWidget(self.target_hint)
        row = QHBoxLayout()
        self.target_actions = row
        self.target_run = button('找这些参考', self.run_targets, row, primary=True)
        button('＋ 找图目标', self.add_target, row)
        targets.addLayout(row)
        self.target_edit = QCheckBox('调整找图用语')
        self.target_edit.toggled.connect(self.show_target_queries)
        targets.addWidget(self.target_edit)
        # Search targets have one visible owner beside the results.
        tray_header = QHBoxLayout()
        self.tray_heading = label('已收集', role='sectionTitle')
        tray_header.addWidget(self.tray_heading, 1)
        button('展开整理', self.cw.show_collected, tray_header)
        layout.addLayout(tray_header)
        self.tray = QListWidget()
        self.tray.setAccessibleName('当前构思的参考托盘')
        self.tray.setViewMode(QListWidget.ViewMode.IconMode)
        self.tray.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.tray.setMovement(QListWidget.Movement.Static)
        self.tray.setIconSize(QSize(100, 76))
        self.tray.setGridSize(QSize(120, 118))
        self.tray.setWordWrap(True)
        self.tray.setMinimumHeight(140)
        self.tray.currentItemChanged.connect(lambda item, _old: self.tray_selected(item))
        self.tray.itemActivated.connect(lambda _item: self.edit_tray_reference())
        layout.addWidget(self.tray, 1)
        self.tray_note = label('图片收集后会出现在这里。', role='muted')
        layout.addWidget(self.tray_note)
        row = QHBoxLayout()
        self.use_button = button('等级与用途', self.edit_tray_reference, row)
        self.remove_button = button('移出此处', self.remove_reference, row)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.undo_remove = button('撤销移出', self.restore_reference, row)
        self.undo_remove.hide()
        layout.addLayout(row)
        self.reference_index = self.w.right_tab_widget.addTab(self.reference_page, '参考')
        from eidory.ui.board_inspector import BoardInspector
        self.board_inspector = BoardInspector(self.w, self.reference_page)
        self._on_board = False
        self.search_notice = QWidget()
        notice = QHBoxLayout(self.search_notice)
        notice.setContentsMargins(2, 4, 2, 4)
        self.search_context = label('')
        notice.addWidget(self.search_context, 1)
        self.search_refresh = button('更新找图结果', self.run_targets, notice, primary=True)
        # Keep the original idea in sight while choosing visual references.
        self.search_brief = label('', role='muted')
        self.search_brief.setAccessibleName('当前构思摘要')
        self.search_brief.setWordWrap(False)
        self.search_brief.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Maximum)
        self.search_brief.installEventFilter(self)
        notice.insertWidget(1, self.search_brief, 1)
        center = self.w.center_result_stack.parentWidget().layout()
        center.insertWidget(center.indexOf(self.w.center_result_stack), self.search_notice)
        center.insertWidget(center.indexOf(self.w.center_result_stack), self.target_area)
        self.target_scroll.setMinimumHeight(50)
        self.target_scroll.setMaximumHeight(180)
        self.target_edit.hide()
        self.target_hint.hide()
        self.target_overview = button('综合推荐', lambda: self.select_target_view(None), self.target_actions)
        self.target_overview.setCheckable(True)
        self.target_overview.clicked.disconnect()
        self.target_overview.toggled.connect(lambda checked: self.select_target_view(None) if checked else None)
        self.advanced_targets = button('组合筛选 ▸', self.toggle_advanced_targets, self.target_actions)
        self.advanced_targets.setCheckable(True)
        self.advanced_targets.clicked.disconnect()
        self.advanced_targets.toggled.connect(lambda _: self.toggle_advanced_targets())
        self.w.filter_panel.strengthChanged.connect(lambda _: self.refresh_target_controls())


    def select_target_view(self, target_id):
        row = next((r for r in self.targets() if r['id'] == target_id), None)
        combo = self.w.workflow_upgrade.result_mode
        index = combo.findData(row['query'] if row else None)
        if index >= 0:
            combo.setCurrentIndex(index)
        self.refresh_target_controls()

    def toggle_advanced_targets(self):
        self.advanced_targets.setText('收起组合筛选 ▾' if self.advanced_targets.isChecked() else '组合筛选 ▸')
        self.refresh_target_controls()

    def refresh_target_controls(self):
        w = self.w
        node = w._current_creative_node()
        active = bool(node and w.center_result_stack.currentWidget() in (w.grid_view, self.search_empty) and w.current_result_mode != 'creative_node')
        w.workflow_upgrade.results_bar.setVisible(w.current_result_mode == 'search_chain' and w.center_result_stack.currentWidget() is w.grid_view and not active)
        for box in w.filter_panel.section_boxes.values():
            box.setVisible(not active or self.advanced_targets.isChecked())
        if not active:
            return
        plan = w.filter_panel.draft
        strengths = {str(f.value): f.score_threshold for f in plan.positive_filters if f.kind == 'semantic'}
        current = w.workflow_upgrade.result_mode.currentData()
        self.target_overview.blockSignals(True)
        self.target_overview.setChecked(current is None)
        self.target_overview.blockSignals(False)
        for row in self.targets():
            widgets = self._target_widgets.get(row['id'])
            if not widgets:
                continue
            slider, caption, results = widgets
            value = strengths.get(row['query'], row.get('strength', 50))
            value = 50 if value is None else value
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
            caption.setText(str(value))
            slider.setEnabled(row['enabled'] and w.search_button.isEnabled())
            ids = getattr(w.workflow_upgrade, 'match_ids', {}).get(row['query'], set())
            evaluated = w.workflow_upgrade.result_mode.findData(row['query']) >= 0
            results.setText(str(len(ids)) + ' 张' if evaluated else '未检索')
            results.setEnabled(row['enabled'] and evaluated)
            results.blockSignals(True)
            results.setChecked(current == row['query'])
            results.blockSignals(False)

    def change_target_strength(self, target_id, value):
        row = next((r for r in self.targets() if r['id'] == target_id), None)
        if row is None:
            return
        panel = self.w.filter_panel
        matched = False
        for gi, group in enumerate(panel.draft.include):
            for fi, condition in enumerate(group.filters):
                if condition.kind == 'semantic' and condition.value == row['query']:
                    matched = True
                    panel._change_strength('include', gi, fi, value)
        rows = self.targets()
        for target in rows:
            if target['id'] == target_id:
                target['strength'] = value
        node = self.w._current_creative_node()
        self.w.workflow_upgrade.state.set_options(node.id, search_targets_json=json.dumps({'signature': target_signature(node, self.cw.store.entries(node.id)), 'rows': rows}, ensure_ascii=False))
        if not matched and hasattr(self.w, 'simple_search'):
            self.w.simple_search.schedule('targets')
        self.refresh_target_controls()

    def show_target_queries(self, shown):
        for field in self.target_content.findChildren(QLineEdit):
            field.setVisible(shown)

    def _build_reference_grades(self):
        ui = self.w.workflow_upgrade
        self.grade_bar = QWidget()
        row = QHBoxLayout(self.grade_bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.grade_buttons = []
        for index in range(ui.status.count()):
            b = QPushButton(ui.status.itemText(index))
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.toggled.connect(lambda checked, i=index: ui.status.setCurrentIndex(i) if checked and not self.refreshing else None)
            row.addWidget(b)
            self.grade_buttons.append(b)
        side = self.cw.reference_input.parentWidget().layout()
        side.insertWidget(side.indexOf(ui.reference_bar), self.grade_bar)
        ui.status.hide()

    @batch_navigation_refresh
    def show_planning(self):
        w = self.w
        w._save_current_board_layout_if_needed()
        w._set_board_focus_mode(False)
        w._set_board_window_pinned(False)
        w._set_board_toolbar_visible(False)
        node = w._current_creative_node()
        if not node or node.parent_id is None:
            w.center_result_stack.setCurrentWidget(self.home)
            self.home_stack.setCurrentWidget(self.overview if node else self.start)
        else:
            w.center_result_stack.setCurrentWidget(self.cw.planning_scroll)
        w.right_tab_widget.setCurrentIndex(self.reference_index)
        self.cw.refresh()
        w.statusBar().showMessage('构思 · ' + self.cw.context_label.text())

    @batch_navigation_refresh
    def show_overview(self):
        self.w.workflow_upgrade.flush()
        self.w._set_ai_workflow_mode('project')
        self.w._save_current_board_layout_if_needed()
        if self.w.current_creative_project_id:
            root = self.w.store.creative_root_node_id(self.w.current_creative_project_id)
            if root is not None:
                self.select_node(root)
        self.w._set_board_focus_mode(False)
        self.w._set_board_toolbar_visible(False)
        self.w.center_result_stack.setCurrentWidget(self.home)
        self.home_stack.setCurrentWidget(self.overview if self.w.current_creative_project_id else self.start)
        self.w.right_tab_widget.setCurrentIndex(self.reference_index)
        self.refresh()
        self.w.statusBar().showMessage('项目全貌 · ' + self.cw.context_label.text())

    def select_node(self, node_id):
        self.routing = True
        try:
            item = self.w._find_creative_node_tree_item(node_id)
            if item:
                self.w.creative_node_tree.setCurrentItem(item)
        finally:
            self.routing = False

    def edit_node(self, node_id):
        self.select_node(node_id)
        self.w._set_ai_workflow_mode('project')
        self.show_planning()

    def node_selected(self, *_args):
        if self.routing or self.refreshing:
            return
        keep_tree_focus = self.w.creative_node_tree.hasFocus()
        current = self.w.center_result_stack.currentWidget()
        if current is self.home or current is self.cw.planning_scroll:
            self.show_planning()
        elif current is self.w.grid_view and self.w.current_result_mode != 'creative_node':
            self.cw.show_candidates()
        elif current is self.w.grid_view:
            self.cw.show_collected()
        elif current is self.w.project_board_view:
            self.cw.show_board()
        elif current is self.search_empty:
            self.open_search()
        if keep_tree_focus:
            self.w.creative_node_tree.setFocus(Qt.FocusReason.OtherFocusReason)

    def open_search(self, node_id=None, target_id=None):
        w = self.w
        if node_id is not None:
            self.select_node(node_id)
        node = w._current_creative_node()
        if not node:
            self.browse_library()
            return
        w.workflow_upgrade.flush()
        if target_id is not None:
            rows = self.targets()
            for row in rows:
                row['enabled'] = row['id'] == target_id
            self.save_targets(rows)
        self.run_targets()
        w.right_tab_widget.setCurrentIndex(self.reference_index)

    def targets(self):
        node = self.w._current_creative_node()
        if not node:
            return []
        entries = self.cw.store.entries(node.id)
        options = self.w.workflow_upgrade.state.options(node.id)
        def hints(n):
            return {n.note}.union(*(hints(child) for child in n.children))
        template_hints = set().union(*(hints(t.root) for t in CREATIVE_TEMPLATES))
        if not entries and not options.get('search_targets_json') and node.note in template_hints:
            return []  # Template instructions are prompts for the author, not search terms.
        return resolved_targets(node, entries, options, self.w._creative_node_path_text(node.id))

    @undoable("修改找图目标")
    def save_targets(self, rows):
        node = self.w._current_creative_node()
        if not node:
            return
        self.w.workflow_upgrade.state.set_options(node.id, search_targets_json=json.dumps({
            'signature': target_signature(node, self.cw.store.entries(node.id)), 'rows': rows}, ensure_ascii=False))
        self.cw.invalidate_entry_search(node.id)
        self._search_dirty_node = node.id
        if hasattr(self.w, "simple_search"):
            self.w.simple_search.schedule("targets")

    def change_target(self, target_id, *, enabled=None, query=None):
        rows = self.targets()
        before = deepcopy(rows)
        for row in rows:
            if row['id'] == target_id:
                if enabled is not None:
                    row['enabled'] = enabled
                if query is not None:
                    row['query'] = query.strip()
        if rows == before:
            return
        self.save_targets(rows)
        self._targets_key = (self.w.current_creative_node_id, repr(rows))
        # Do not rebuild the editor under the caret; mark existing results as
        # stale and disable collection until the explicit search command.
        self.refresh_search_status()

    def run_targets(self):
        node = self.w._current_creative_node()
        if not node:
            return
        if not self.w.search_button.isEnabled():
            self._search_dirty_node = node.id
            self.refresh_search_status()
            if hasattr(self.w, 'simple_search'):
                self.w.simple_search.schedule('targets')
            self.w.statusBar().showMessage('正在找图，完成后自动更新到当前目标')
            return
        self.w.workflow_upgrade.flush()
        rows = self.targets()
        if not any(r['enabled'] for r in rows) or any(r['enabled'] and not r['query'].strip() for r in rows):
            self.w.center_result_stack.setCurrentWidget(self.search_empty)
            self.w.right_tab_widget.setCurrentIndex(self.reference_index)
            self.refresh()
            self.target_hint.setText('请至少勾选一项，并填写它的找图用语。')
            return
        self._search_dirty_node = None
        if hasattr(self.w, "simple_search"):
            self.w.simple_search.pending = None
            self.w.simple_search.timer.stop()
        preserve = self._result_node_id == node.id and self.w.current_result_mode == 'search_chain'
        self.cw.start_node_search(node, '', queries=list(dict.fromkeys(r['query'] for r in rows if r['enabled'])), preserve_filters=preserve)
        self._result_node_id = node.id
        self.w.right_tab_widget.setCurrentIndex(self.reference_index)
        self.refresh()

    def add_target(self):
        from PySide6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(self.w, '添加找图目标', '想找什么参考？')
        if ok and text.strip():
            from uuid import uuid4
            rows = self.targets()
            rows.append({'id': uuid4().hex, 'kind': '手动目标', 'name': text.strip(), 'query': text.strip(), 'enabled': True})
            self.save_targets(rows)
            self.refresh()

    def edit_context(self):
        project = self.w.store.get_creative_project(self.w.current_creative_project_id)
        if not project:
            self.show_planning()
            return
        dialog = QDialog(self.w)
        dialog.setWindowTitle('主题与补充设定')
        layout = QVBoxLayout(dialog)
        layout.addWidget(label('创作主题'))
        brief = QTextEdit(project.brief)
        brief.setAcceptRichText(False)
        layout.addWidget(brief)
        layout.addWidget(label('已经确定的背景与限制'))
        extra = QTextEdit(self.cw.store.project(project.id)['extra'])
        extra.setAcceptRichText(False)
        layout.addWidget(extra)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(640, 420)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.w._set_ai_workflow_mode('project')
            self.w.inspiration_brief_input.setPlainText(brief.toPlainText())
            self.w.inspiration_answers_input.setPlainText(extra.toPlainText())
            self.cw.save_context()
            self.refresh()

    def _idea_card(self, row, *, pending=False, locked=False, detail=False):
        node = self.w._current_creative_node()
        card = QWidget()
        card.setObjectName('creativeSuggestionCard' if pending else 'creativeIdeaCard')
        card.setProperty('ideaId', row['id'])
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        top = QHBoxLayout()
        top.addWidget(label(row['name'], role='flowSubtitle'), 1)
        if pending:
            edit = button('编辑建议', lambda: self.edit_entry(row['id'], pending=True), top)
            edit.setEnabled(not locked)
            action = button('采用细节' if detail else '采用这个想法', lambda: self.accept_one(row['id']), top, primary=True)
            action.setEnabled(not locked)
            skip = button('略过', lambda: self.skip_one(row['id']), top)
            skip.setEnabled(not locked)
        else:
            button('找它的参考', lambda: self.open_search(node.id, row['id']), top)
        layout.addLayout(top)
        if pending:
            proposal = decode_entries(self.cw.store.node(node.id)['proposal_entries_json']) or {}
            target = proposal.get('target')
            marker = label('替换此想法的细节' if target else '新增想法 · 不改已有内容', role='muted')
            layout.addWidget(marker)
            if target:
                diff = QLabel(change_html(target['visual'], row['visual']))
                diff.setTextFormat(Qt.TextFormat.RichText)
                diff.setWordWrap(True)
                layout.addWidget(diff)
            elif row['visual']:
                layout.addWidget(label(row['visual']))
        elif row['visual']:
            layout.addWidget(label(row['visual']))
        if not pending:
            actions = QHBoxLayout()
            edit = button('编辑', lambda: self.edit_entry(row['id']), actions)
            edit.setEnabled(not locked)
            details = button('AI 补充细节', lambda: self.entry_details(row['id']), actions)
            details.setEnabled(not locked and self.w.generate_creative_details_button.isEnabled())
            actions.addStretch()
            remove = button('删除想法', lambda: self.delete_entry(row['id']), actions)
            remove.setEnabled(not locked)
            layout.addLayout(actions)
        return card

    def delete_entry(self, row_id):
        for i, row in enumerate(self.cw.entries_table._rows):
            if row['id'] == row_id:
                self.cw.entries_table.selectRow(i)
                self.cw.delete_entry()
                break

    def entry_details(self, row_id):
        for i, row in enumerate(self.cw.entries_table._rows):
            if row['id'] == row_id:
                self.cw.entries_table.selectRow(i)
                self.w.generate_creative_details_button.click()
                break

    def edit_entry(self, row_id, *, pending=False):
        node = self.w._current_creative_node()
        if not node or self.cw.store.node(node.id)['locked']:
            return
        payload = decode_entries(self.cw.store.node(node.id)['proposal_entries_json']) if pending else self.cw.store.entries(node.id)
        row = next((r for r in (payload or {}).get('rows', []) if r['id'] == row_id), None)
        if not row:
            return
        self.begin_inline(node.id, row, pending=pending, fixed_name=bool(pending and payload.get('target')))

    def add_inline(self):
        node = self.w._current_creative_node()
        if node and not self.cw.store.node(node.id)['locked']:
            self.begin_inline(node.id, None)

    def begin_inline(self, node_id, row, *, pending=False, fixed_name=False):
        if self._inline:
            self._inline.finish()
            if self._inline:
                return
        editor = InlineIdeaEditor((row or {}).get('name', ''), (row or {}).get('visual', ''), fixed_name=fixed_name)
        self._inline, self._inline_key = editor, (node_id, (row or {}).get('id'), pending)
        target = self.pending_layout if pending else self.idea_layout
        card = next((target.itemAt(i).widget() for i in range(target.count())
                     if target.itemAt(i).widget() and target.itemAt(i).widget().property('ideaId') == (row or {}).get('id')), None) if row else None
        if card:
            for widget in card.findChildren(QWidget):
                widget.hide()
            card.layout().addWidget(editor)
        else:
            target.addWidget(editor)
        self.idea_cards.setVisible(True)
        self.cw.entries_panel.setVisible(True)
        editor.finished.connect(self.finish_inline)
        (editor.visual if fixed_name else editor.name).setFocus()

    @undoable("编辑想法")
    def finish_inline(self, values):
        editor, key = self._inline, self._inline_key
        if not editor:
            return
        node_id, row_id, pending = key
        self._inline = self._inline_key = None
        editor.hide()
        editor.deleteLater()
        node = self.w.store.get_creative_node(node_id)
        if values is not None and node and not self.cw.store.node(node_id)['locked']:
            state = self.cw.store.node(node_id)
            payload = decode_entries(state['proposal_entries_json']) if pending else self.cw.store.entries(node_id)
            payload = payload or {'base_note': node.note, 'base_query': node.search_query, 'rows': []}
            row = next((r for r in payload['rows'] if r['id'] == row_id), None)
            if row_id is None:
                from eidory.core.creative_entries import new_entry
                payload['rows'].append(new_entry(*values))
            elif row:
                row.update(name=values[0], visual=values[1])
            if pending:
                self.cw.store.save_node(node_id, proposal_entries_json=encode_entries(payload), proposal_note=composed_note(payload))
            elif row_id is None or row:
                self.cw.store.save_entries(node_id, payload, snapshot=True)
                self.cw.invalidate_entry_search(node_id)
            self.w.statusBar().showMessage('想法已保存 · 可撤销')
        self._entries_key = None
        if node_id == self.w.current_creative_node_id:
            self.cw.reload_node(node_id)
        self.refresh()

    @undoable("采纳想法")
    def accept_one(self, row_id):
        node = self.w._current_creative_node()
        state = self.cw.store.node(node.id)
        pending = decode_entries(state['proposal_entries_json'])
        if state['locked'] or not pending:
            return
        chosen = [r for r in pending['rows'] if r['id'] == row_id]
        if not chosen:
            return
        self.w._save_current_creative_node_editor_snapshot()
        one = {**pending, 'rows': chosen}
        self.cw.store.save_node(node.id, proposal_entries_json=encode_entries(one))
        if self.cw.store.accept_entries(node.id):
            rest = {**pending, 'rows': [r for r in pending['rows'] if r['id'] != row_id]}
            if rest['rows']:
                self.cw.store.save_node(node.id, proposal_entries_json=encode_entries(rest), proposal_note=state['proposal_note'], proposal_query=state['proposal_query'])
            self.cw.invalidate_entry_search(node.id)
            self.cw.reload_node(node.id)
            self.w.statusBar().showMessage('已采用这个想法；是否用于找图可在搜索目标中选择。')
        else:
            self.cw.store.save_node(node.id, proposal_entries_json=state['proposal_entries_json'])
            self.w.statusBar().showMessage('原内容已变化，建议未覆盖你的编辑。请重新检查建议。')

    def skip_one(self, row_id):
        node = self.w._current_creative_node()
        state = self.cw.store.node(node.id)
        pending = decode_entries(state['proposal_entries_json'])
        if not pending or state['locked']:
            return
        pending['rows'] = [r for r in pending['rows'] if r['id'] != row_id]
        if pending['rows']:
            self.cw.store.save_node(node.id, proposal_entries_json=encode_entries(pending))
            self.cw.reload_node(node.id)
        else:
            self.cw.discard_note()

    def tray_selected(self, item):
        self.use_button.setEnabled(item is not None)
        self.remove_button.setEnabled(item is not None)
        if item:
            self.tray_note.setText(item.toolTip())

    def edit_tray_reference(self):
        item = self.tray.currentItem()
        if item is None:
            return
        image_id = item.data(Qt.ItemDataRole.UserRole)
        owners = item.data(Qt.ItemDataRole.UserRole + 1) or []
        if self.w.current_creative_node_id not in [r['node_id'] for r in owners]:
            if len(owners) == 1:
                self.select_node(owners[0]['node_id'])
            elif owners:
                from PySide6.QtWidgets import QInputDialog
                names = [self.w._creative_node_path_text(r['node_id']) for r in owners]
                chosen, ok = QInputDialog.getItem(self.w, '选择参考用途', '这张图用于多处，选择要编辑的内容：', names, 0, False)
                if not ok:
                    return
                self.select_node(owners[names.index(chosen)]['node_id'])
        self.cw.show_collected()
        self.w.grid_view.select_image_ids([image_id])
        self.w.right_tab_widget.setCurrentIndex(1)

    @undoable("移出参考")
    def remove_reference(self):
        item, node = self.tray.currentItem(), self.w._current_creative_node()
        if item is None or node is None:
            return
        image_id = item.data(Qt.ItemDataRole.UserRole)
        # Parent trays aggregate descendants; don't guess which link to remove.
        if image_id not in self.w.store.creative_node_image_ids(node.id, include_descendants=False):
            self.tray_note.setText('这张图属于下级内容；选择对应内容后再移出。')
            return
        self._removed = (node.id, image_id, self.w.workflow_upgrade.state.decision(node.id, image_id), self.cw.store.reference_note(node.id, image_id))
        self.w.store.remove_images_from_creative_node(node.id, [image_id])
        self.w._refresh_after_creative_node_images_changed(node)
        self.refresh()

    def restore_reference(self):
        if not self._removed:
            return
        node_id, image_id, decision, note = self._removed
        node = self.w.store.get_creative_node(node_id)
        if node and node.project_id == self.w.current_creative_project_id:
            self.w.store.add_images_to_creative_node(node_id, [image_id])
            self.w.workflow_upgrade.state.set_decision(node_id, image_id, decision['status'], decision['purpose'])
            self.cw.store.save_reference_note(node_id, image_id, note)
            self._removed = None
            self.w._refresh_after_creative_node_images_changed(node)
            self.refresh()

    @batch_navigation_refresh
    def show_project_board(self):
        root = self.w.store.creative_root_node_id(self.w.current_creative_project_id) if self.w.current_creative_project_id else None
        if root:
            self.select_node(root)
            # Explicit project navigation must not inherit a temporary search
            # result set's rule that routes the generic board command to gallery.
            self.w._show_creative_board_for_node(root)
            self.w.right_tab_widget.setCurrentIndex(self.reference_index)

    def eventFilter(self, obj, event):
        if obj is getattr(self, 'search_brief', None) and event.type() in (QEvent.Type.Resize, QEvent.Type.FontChange):
            self._update_search_brief()
        if obj is self.overview and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self.refresh)
        return False

    def tray_memberships(self, node):
        if not node:
            return {}
        with self.w.store.connect() as conn:
            rows = conn.execute('''WITH RECURSIVE branch(id) AS (
                SELECT id FROM creative_nodes WHERE id=?
                UNION ALL SELECT n.id FROM creative_nodes n JOIN branch b ON n.parent_id=b.id
            ) SELECT i.image_id, i.node_id, n.title,
                COALESCE(d.status, 'candidate') AS status, COALESCE(d.purpose, '') AS purpose
            FROM creative_node_images i JOIN branch b ON b.id=i.node_id
            JOIN creative_nodes n ON n.id=i.node_id
            LEFT JOIN creative_reference_decisions d ON d.node_id=i.node_id AND d.image_id=i.image_id
            ORDER BY i.node_id, i.image_id''', (node.id,)).fetchall()
        result = {}
        for row in rows:
            result.setdefault(row['image_id'], []).append(dict(row))
        return result

    def refresh_cards(self):
        project = self.w.store.get_creative_project(self.w.current_creative_project_id)
        if not project:
            return
        nodes = self.w.store.list_creative_nodes(project.id)
        parents = {n.id: n.parent_id for n in nodes}
        def in_branch(nid, parent):
            seen = set()
            while nid in parents and nid not in seen:
                seen.add(nid)
                nid = parents[nid]
                if nid == parent:
                    return True
            return False
        root = next((n for n in nodes if n.parent_id is None), None)
        nodes = [n for n in nodes if root and n.parent_id == root.id]
        columns = max(1, min(3, (self.overview.width() - 40) // 270))
        states = [self.cw.store.node(n.id) for n in nodes]
        checks = self.w.workflow_upgrade.state.overview(project.id)
        key = (project, tuple(nodes), repr(states), repr(checks), columns)
        self.project_title.setText(project.title)
        self.project_brief.setText(project.brief)
        if key == self._cards_key:
            return
        self._cards_key = key
        clear_layout(self.cards_layout)
        for index, (node, state) in enumerate(zip(nodes, states)):
            card = QWidget()
            card.setObjectName('creativeOverviewCard')
            card.setMinimumWidth(0)
            layout = QVBoxLayout(card)
            layout.setContentsMargins(14, 12, 14, 12)
            heading = QHBoxLayout()
            heading.addWidget(label(node.title, role='flowSubtitle'), 1)
            pending = state['proposal_note'] or state['proposal_entries_json']
            heading.addWidget(label('已确定' if state['locked'] else '有待选建议' if pending else '可继续构思', role='muted'))
            layout.addLayout(heading)
            entries = decode_entries(state['entries_json'])
            text = ' · '.join(r['name'] for r in entries['rows']) if entries else node.note
            layout.addWidget(label(text[:108] + ('…' if len(text) > 108 else '')))
            ids = self.w.store.creative_node_image_ids(node.id, include_descendants=True)
            previews = QHBoxLayout()
            for image in self.w.store.images_by_ids(ids[:3]):
                preview = QLabel()
                preview.setFixedSize(48, 42)
                if image.thumbnail_path:
                    pixmap = QPixmap(image.thumbnail_path)
                    preview.setPixmap(pixmap.scaled(48, 42, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                previews.addWidget(preview)
            previews.addWidget(label(f'{len(ids)} 张参考' if ids else '尚未收集参考', role='muted'))
            previews.addStretch()
            layout.addLayout(previews)
            branch = [r for r in checks if r['id'] == node.id or in_branch(r['id'], node.id)]
            for record in branch:
                line = QHBoxLayout()
                caption = ('↳ ' + record['title'] + ' · ') if record['id'] != node.id else ''
                line.addWidget(label(caption + str(record['primary_count']) + ' 张主参考', role='muted'), 1)
                choice = QComboBox()
                choice.setAccessibleName('参考需要：' + record['title'])
                for title, value in [('需要参考', 'needed'), ('无需参考图', 'none'), ('参考已齐', 'resolved')]:
                    choice.addItem(title, value)
                choice.setCurrentIndex(choice.findData(record['requirement']))
                choice.currentIndexChanged.connect(lambda _=False, nid=record['id'], box=choice: self.set_requirement(nid, box.currentData()))
                line.addWidget(choice)
                layout.addLayout(line)
                if record['needs_review']:
                    review = QHBoxLayout()
                    review.addWidget(label(caption + '主题已改变，需复核', role='muted'), 1)
                    button('已复核', lambda _=False, nid=record['id']: self.review_node(nid), review)
                    layout.addLayout(review)
                elif record['proposal'] or record['proposal_entries']:
                    review = QHBoxLayout()
                    button('审阅' + ('「'+record['title']+'」' if record['id'] != node.id else '') + ' AI 建议', lambda _=False, nid=record['id']: self.edit_node(nid), review)
                    layout.addLayout(review)
            actions = QHBoxLayout()
            button('编辑构思', lambda _=False, nid=node.id: self.edit_node(nid), actions)
            button('找参考 →', lambda _=False, nid=node.id: self.open_search(nid), actions)
            actions.addStretch()
            layout.addLayout(actions)
            self.cards_layout.addWidget(card, index // columns, index % columns)

    @undoable("调整参考需要")
    def set_requirement(self, node_id, value):
        self.w.workflow_upgrade.state.set_options(node_id, reference_requirement=value)
        self._cards_key = None
        QTimer.singleShot(0, self.refresh)

    @undoable("复核项目设定")
    def review_node(self, node_id):
        self.w.workflow_upgrade.state.set_options(node_id, reviewed_context=self.w.workflow_upgrade.state.context_key(self.w.current_creative_project_id))
        self._cards_key = None
        self.refresh()

    def refresh_editor(self, node, state):
        w, cw = self.w, self.cw
        probe = w.ai_workflow_stack.currentIndex() == 1
        # Global project controls remain in the exploration page, but do not
        # compete with the selected creative content in the focused editor.
        for i in range(cw.planning.layout().count()):
            widget = cw.planning.layout().itemAt(i).widget()
            if widget and widget not in (w.ai_workflow_stack, self.editor_top):
                widget.setVisible(probe)
        w.ai_project_mode_button.setVisible(probe)
        w.ai_probe_mode_button.setVisible(probe)
        self._project_actions.hide()
        self.editor_top.setVisible(not probe)
        if self._inline and self._inline_key[0] != (node.id if node else None):
            self._inline.finish()
        self.editor_title.setText(node.title if node else '')
        cw.format_combo.hide()
        cw.restore_button.hide()
        # Hide the old format label, keeping the lock command visible.
        format_layout = cw.format_combo.parentWidget().layout()
        for l in cw.format_combo.parentWidget().findChildren(QLabel):
            if l.text() == '内容形式':
                l.hide()
        for widget in (w.creative_node_query_input, w.search_creative_node_button, w.generate_all_creative_nodes_button,
                       w.open_creative_copy_button, w.workflow_upgrade.target_button, w.workflow_upgrade.target_hint,
                       cw.entries_table, cw.entries_hint, cw.delete_entry_button, cw.proposal_entries_table, cw.proposal_query):
            widget.hide()
        for l in w.creative_node_note_input.parentWidget().findChildren(QLabel):
            if l.text() == '跳过已锁定和待采纳节点':
                l.hide()
        if self.node_caption:
            self.node_caption.setText('我的设定')
        cw.lock_button.setText('已确定，保护这部分构思')
        cw.add_entry_button.setText('＋ 添加想法')
        cw.organize_button.setText('AI 整理为独立想法')
        cw.original_button.hide()
        w.creative_node_note_input.setMaximumHeight(140)
        payload = decode_entries(state.get('entries_json', ''))
        pending = decode_entries(state.get('proposal_entries_json', ''))
        self.base_summary.setVisible(payload is not None)
        self.base_summary.setText((payload or {}).get('base_note', ''))
        self.idea_cards.setVisible(payload is not None)
        self.pending_cards.setVisible(pending is not None)
        cw.accept_button.setText('采用全部建议' if pending else '采用这段说明')
        cw.proposal_label.setText('AI 建议 · 尚未采用')
        self.refresh_prose_diff()
        key = (node.id if node else None, state.get('entries_json'), state.get('proposal_entries_json'), state.get('locked'), w.generate_creative_details_button.isEnabled())
        if key != self._entries_key and not self._inline:
            self._entries_key = key
            clear_layout(self.idea_layout)
            clear_layout(self.pending_layout)
            for row in (payload or {}).get('rows', []):
                self.idea_layout.addWidget(self._idea_card(row, locked=bool(state.get('locked'))))
            for row in (pending or {}).get('rows', []):
                self.pending_layout.addWidget(self._idea_card(row, pending=True, locked=bool(state.get('locked')), detail=bool(pending.get('target'))))

    def refresh_prose_diff(self):
        node = self.w._current_creative_node()
        state = self.cw.store.node(node.id) if node else {}
        prose = bool(state.get('proposal_note') and not decode_entries(state.get('proposal_entries_json', '')))
        self.prose_diff.setVisible(prose)
        self.edit_prose.setVisible(prose)
        self.cw.proposal_input.setVisible(prose and self.edit_prose.isChecked())
        self.cw.partial_button.setVisible(prose and self.edit_prose.isChecked())
        if prose:
            self.prose_diff.setText('替换当前说明 · 红色为删去，绿色为新增<br>' + change_html(node.note, self.cw.proposal_input.toPlainText()))

    def refresh_search_status(self):
        node = self.w._current_creative_node()
        dirty = bool(node and self._search_dirty_node == node.id)
        busy = not self.w.search_button.isEnabled()
        self.search_refresh.setVisible(dirty)
        self.target_run.setText('更新找图结果' if dirty else '找这些参考')
        self.target_run.setEnabled(self.w.search_button.isEnabled())
        self.search_refresh.setEnabled(self.w.search_button.isEnabled())
        self.target_hint.setText('目标已修改，更新结果后再收集。' if dirty else '各项分别找图，结果合并显示。')
        idea = ' '.join(node.note.split()) if node else ''
        self.search_brief.setToolTip(node.note if node else '')
        self._update_search_brief()
        self.search_brief.setVisible(bool(idea))
        self.search_context.setText(('目标已修改 · 仍显示上次结果' if dirty else '正在为「' + node.title + '」找参考') if node else '图库浏览')
        if busy and node:
            self.search_context.setText(f'正在为「{node.title}」更新结果…')
        elif node and self.w.current_result_mode == 'search_chain':
            target = self.w.workflow_upgrade.result_mode.currentData()
            if target and target != '__toolbar__':
                self.search_context.setText(f'仅看「{target}」 · {len(self.w.grid_view._images)} 张')
            else:
                self.search_context.setText(f'综合参考 · {len(self.w.grid_view._images)} 张')
        if dirty or busy:
            self.cw.collect_button.setEnabled(False)
        if hasattr(self.w, "simple_search"):
            self.w.simple_search.refresh()

    def _update_search_brief(self):
        idea = ' '.join(self.search_brief.toolTip().split())
        self.search_brief.setText(self.search_brief.fontMetrics().elidedText(
            '构思：' + idea if idea else '', Qt.TextElideMode.ElideRight,
            max(0, self.search_brief.contentsRect().width())))

    def refresh(self, *_args):
        if self.refreshing or getattr(self, '_refresh_batch_depth', 0):
            return
        self.refreshing = True
        try:
            w, cw = self.w, self.cw
            if hasattr(w, 'edit_history'):
                w.edit_history.refresh()
            node = w._current_creative_node()
            previews = [p for p in w._preview_dialogs if p.collect_title]
            cw.collect_button.setVisible(not previews)
            for preview in previews:
                w._refresh_preview_collection(preview, preview.current_image())
            on_start = not w.current_creative_project_id and w.center_result_stack.currentWidget() is self.home
            if on_start and not self._start_hides_inspector:
                self._start_hides_inspector = w._sidebar_visible[2]
                if self._start_hides_inspector:
                    w._set_sidebar_visible(2, False)
            elif not on_start and self._start_hides_inspector:
                self._start_hides_inspector = False
                w._set_sidebar_visible(2, True)
            state = cw.store.node(node.id) if node else {}
            for index, b in enumerate(self.grade_buttons):
                b.blockSignals(True)
                b.setChecked(index == w.workflow_upgrade.status.currentIndex())
                b.blockSignals(False)
                b.setEnabled(cw._reference_key is not None)
            is_gallery = w.center_result_stack.currentWidget() is w.grid_view
            is_board = w.center_result_stack.currentWidget() is w.project_board_view
            self.board_inspector.actions.setVisible(is_board)
            self.reference_page.setCurrentWidget(self.board_inspector.page if is_board else self.reference_scroll)
            w.right_tab_widget.setTabText(self.reference_index, '图片属性' if is_board else '参考')
            w.right_tab_widget.setTabVisible(1, not is_board)
            self.grade_bar.setVisible(not is_board)
            if is_board:
                self.board_inspector.refresh()
                if w.right_tab_widget.currentIndex() == 1 or (not self._on_board and w.current_creative_project_id):
                    w.right_tab_widget.setCurrentIndex(self.reference_index)
            self._on_board = is_board
            if not is_board:
                self.board_inspector.flush_note()
            is_search = (is_gallery and w.current_result_mode != 'creative_node') or w.center_result_stack.currentWidget() is self.search_empty
            self.overview_button.setEnabled(bool(w.current_creative_project_id))
            self.overview_button.setVisible(bool(node and node.parent_id is not None))
            cw.collected_button.hide()
            if w.center_result_stack.currentWidget() is self.search_empty or (is_gallery and w.current_result_mode == 'creative_node'):
                for i, b in enumerate(w.workspace_chrome.stage_buttons):
                    b.blockSignals(True)
                    b.setChecked(i == 1)
                    b.blockSignals(False)
            w.right_tab_widget.setTabVisible(self.reference_index, bool(w.current_creative_project_id))
            if w.center_result_stack.currentWidget() is self.home:
                self.home_stack.setCurrentWidget(self.overview if w.current_creative_project_id else self.start)
                if w.current_creative_project_id:
                    self.refresh_cards()
            if is_board:
                self.target_area.hide()
                self.search_notice.hide()
                for widget in (w.shuffle_results_button, w.save_result_set_button, w.thumbnail_size_label, w.thumbnail_size_slider):
                    widget.hide()
                return
            self.refresh_editor(node, state)
            self.target_area.setVisible(bool(node and is_search))
            self.search_notice.setVisible(bool(node and is_search))
            self.reference_title.setText(f'{node.title} · 参考' if node else '开始一个创作项目')
            if node and node.parent_id is None:
                self.reference_title.setText('项目参考')
            self.reference_context.setText(w._creative_node_path_text(node.id) if node else '填写主题后即可开始。')
            rows = self.targets() if node else []
            key = (node.id if node else None, repr(rows))
            if key != self._targets_key:
                self._targets_key = key
                clear_layout(self.target_layout)
                self._target_widgets = {}
                self.target_scroll.setFixedHeight(min(150, max(46, len(rows) * 46)))
                for row in rows:
                    card = QWidget()
                    card.setObjectName('searchTargetRow')
                    line = QHBoxLayout(card)
                    line.setContentsMargins(0, 2, 0, 2)
                    check = QCheckBox(row['name'])
                    check.setToolTip(row['name'])
                    check.setChecked(row['enabled'])
                    check.toggled.connect(lambda value, tid=row['id']: self.change_target(tid, enabled=value))
                    line.addWidget(check)
                    query = QLineEdit(row['query'])
                    query.setAccessibleName('找图用语：' + row['name'])
                    query.textEdited.connect(lambda value, tid=row['id']: self.change_target(tid, query=value))
                    line.addWidget(query, 1)
                    slider = QSlider(Qt.Orientation.Horizontal)
                    slider.setRange(0, 100)
                    slider.setFixedWidth(120)
                    slider.setAccessibleName('匹配强度：' + row['name'])
                    slider.setToolTip('左：宽松，右：严格；自动更新')
                    slider.valueChanged.connect(lambda value, tid=row['id']: self.change_target_strength(tid, value))
                    line.addWidget(slider)
                    strength = label('50')
                    strength.setFixedWidth(26)
                    line.addWidget(strength)
                    results = QPushButton('查看结果')
                    results.setCheckable(True)
                    results.toggled.connect(lambda checked, tid=row['id']: self.select_target_view(tid) if checked else None)
                    line.addWidget(results)
                    self.target_layout.addWidget(card)
                    self._target_widgets[row['id']] = (slider, strength, results)
                if not rows:
                    self.target_layout.addWidget(label('先写一条想法，也可以直接添加找图目标。', role='muted'))
            self.refresh_target_controls()
            ids = w.store.creative_node_image_ids(node.id, include_descendants=True) if node and not is_board else []
            memberships = self.tray_memberships(node) if not is_board else {}
            tray_key = (node.id if node else None, tuple(ids), repr(memberships))
            if tray_key != self._tray_key:
                self._tray_key = tray_key
                self.tray.clear()
                for im in w.store.images_by_ids(ids):
                    owners = memberships.get(im.id, [])
                    descriptions = [r['title'] + ' · ' + {'primary': '★ 主参考', 'candidate': '备选', 'unused': '暂不采用'}[r['status']] + (' · ' + r['purpose'] if r['purpose'] else '') for r in owners]
                    caption = descriptions[0] if len(descriptions) == 1 else f'已用于 {len(descriptions)} 处'
                    item = QListWidgetItem(QIcon(im.thumbnail_path) if im.thumbnail_path else QIcon(), caption)
                    item.setData(Qt.ItemDataRole.UserRole, im.id)
                    item.setData(Qt.ItemDataRole.UserRole + 1, owners)
                    item.setToolTip(im.file_name + '\n' + '\n'.join(descriptions))
                    self.tray.addItem(item)
                self.tray_note.setText('点击缩略图可调整等级、用途，或移出此处。' if ids else '收集后，缩略图会出现在这里。')
                self.tray_selected(None)
            self.tray_heading.setText(f'已收集 {len(ids)} 张')
            self.undo_remove.setVisible(bool(self._removed and w.store.get_creative_node(self._removed[0]) and w.store.get_creative_node(self._removed[0]).project_id == w.current_creative_project_id))
            if node:
                cw.collect_button.setText(f'收集到「{node.title}」')
                selected = w.grid_view.selected_image_ids()
                direct = set(w.store.creative_node_image_ids(node.id, include_descendants=False))
                if selected and all(i in direct for i in selected):
                    cw.collect_button.setText(f'已收集到「{node.title}」')
                    cw.collect_button.setEnabled(False)
            if is_search and cw.search_node_id == w.current_creative_node_id:
                self._result_node_id = cw.search_node_id
            self.refresh_search_status()
            if not is_gallery:
                for widget in (w.shuffle_results_button, w.save_result_set_button, w.thumbnail_size_label, w.thumbnail_size_slider):
                    widget.hide()
        finally:
            self.refreshing = False
