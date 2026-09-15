from __future__ import annotations

import json
from weakref import ref

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)

from eidory.ui.reference_entries import ReferenceEntriesTable
from eidory.core.creative_entries import decode_entries, encode_entries, new_entry, search_names, composed_note
from eidory.core.creative_workspace import CreativeWorkspaceStore, decode_state
from eidory.core.search_filters import FilterGroup, FilterPlan, SearchFilter


class CreativeWorkspace(QObject):
    """Coordinates project context without giving gallery controls a second owner."""
    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self.store = CreativeWorkspaceStore(window.store)
        self.loading = False
        self.project_id = None
        self.search_node_id = None
        self.search_states = {}
        self.restore_position = None
        self.probe_draft = None
        self._reference_key = None
        self._build()

    @property
    def w(self):
        return self._window()

    def _button(self, label, callback, layout):
        button = QPushButton(label)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _build(self):
        w = self.w
        tab_index = w.right_tab_widget.currentIndex()
        w.right_tab_widget.blockSignals(True)
        self.planning = w.right_tab_widget.widget(1)
        w.right_tab_widget.removeTab(1)
        self.planning_scroll = QScrollArea()
        self.planning_scroll.setWidgetResizable(True)
        self.planning_scroll.setWidget(self.planning)
        self.planning.layout().setAlignment(Qt.AlignmentFlag.AlignTop)
        w.center_result_stack.addWidget(self.planning_scroll)
        w.center_result_stack.parentWidget().setMinimumWidth(480)
        self.sidebar = QWidget()
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(6, 8, 6, 8)
        side.addWidget(QLabel("创作项目"))
        side.addWidget(w.creative_project_combo)
        w.creative_project_combo.show()
        self._button("新建创作项目", self.new_project, side)
        self._button("编辑构思与 AI 建议", self.show_project_planning, side)
        self._button("探索灵感方向", self.show_probe, side)
        w.creative_node_tree.setMinimumHeight(150)
        w.creative_node_tree.setMaximumHeight(16777215)
        side.addWidget(w.creative_node_tree, 1)
        w.creative_node_status_label.setMaximumHeight(64)
        side.addWidget(w.creative_node_status_label)
        self.find_button = self._button("为当前节点找参考", self.find, side)
        self.collect_button = self._button("加入当前节点", w._save_selection_to_current_creative_node, side)
        self._button("查看已收集图片", self.show_collected, side)
        side.addWidget(QLabel("本项目的参考说明"))
        self.reference_input = QTextEdit()
        self.reference_input.setPlaceholderText("选中已收集的图片，写下参考光线、构图或材质等用途。")
        self.reference_input.setMaximumHeight(110)
        self.reference_input.setAcceptRichText(False)
        self.reference_input.textChanged.connect(self.save_reference)
        side.addWidget(self.reference_input)
        self._button("导出创作参考包…", self.export_project, side)
        for label in self.sidebar.findChildren(QLabel, options=Qt.FindChildOption.FindDirectChildrenOnly):
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        w.right_tab_widget.insertTab(1, self.sidebar, "创作")
        w.right_tab_widget.setCurrentIndex(tab_index)
        w.right_tab_widget.blockSignals(False)

        bar = QWidget()
        bar.setObjectName("creativeWorkspaceNavigation")
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        self._button("构思", self.show_project_planning, row)
        self._button("找参考", self.show_candidates, row)
        self.collected_button = self._button("已收集", self.show_collected, row)
        self._button("看板", self.show_board, row)
        self.context_label = QLabel("图库浏览")
        self.context_label.setMinimumWidth(0)
        self.context_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self.context_label, 1)
        self.group_button = self._button("AI 分组…", w._group_selected_images_with_ai, row)
        self.group_button.setToolTip("选中至少 4 张已索引图片，预览并修改分组后保存")
        self.board_buttons = [
            self._button("便签", w.project_board_view.add_note, row),
            self._button("按节点分区", w.project_board_view.arrange_by_labels, row),
            self._button("撤销编辑", w.project_board_view.undo_layout, row),
            self._button("重做", w.project_board_view.redo_layout, row),
        ]
        for button in self.board_buttons:
            button.hide()
        w.center_result_stack.parentWidget().layout().insertWidget(0, bar)
        self.navigation = bar
        self.probe_tuning = QWidget()
        tuning = QHBoxLayout(self.probe_tuning)
        tuning.setContentsMargins(0, 0, 0, 0)
        tuning.addWidget(QLabel("灵感结果 · 更多"))
        tuning.addWidget(w.score_threshold_slider, 1)
        tuning.addWidget(QLabel("更相近"))
        tuning.addWidget(w.score_threshold_label)
        w.center_result_stack.parentWidget().layout().insertWidget(2, self.probe_tuning)
        self.probe_tuning.hide()
        w.project_board_view.setAccessibleName("创作参考看板")
        w.inspiration_brief_input.setFixedHeight(62)
        w.inspiration_answers_input.setFixedHeight(68)
        w.creative_node_note_input.setMinimumHeight(170)
        w.creative_node_note_input.setMaximumHeight(420)
        w.creative_node_note_input.setPlaceholderText("已经确定的创作细节；AI 的新建议会单独显示，采纳后才更新这里。")
        w.creative_ai_project_button.setText("给我一个新创作题目")
        w.creative_new_project_button.setText("按此主题新建项目")
        w.generate_all_creative_nodes_button.setText("批量补全项目节点")

        node_layout = w.creative_node_note_input.parentWidget().layout()
        node_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        node_layout.setSpacing(6)
        actions = QHBoxLayout()
        actions.addWidget(QLabel("内容形式"))
        self.format_combo = QComboBox()
        self.format_combo.setAccessibleName("节点内容形式")
        self.format_combo.addItem("整体说明", "text")
        self.format_combo.addItem("参考条目", "entries")
        self.format_combo.setToolTip("整体说明适合设定、氛围和构图；参考条目适合独立找图的目标。切换会保留原文，可恢复。")
        self.format_combo.currentIndexChanged.connect(self.change_content_format)
        actions.addWidget(self.format_combo)
        self.lock_button = QCheckBox("已确定，锁定此节点")
        self.lock_button.toggled.connect(self.toggle_lock)
        actions.addWidget(self.lock_button)
        self.restore_button = self._button("恢复上次采纳前的内容", self.restore_note, actions)
        node_layout.insertLayout(1, actions)
        self.entries_panel = QWidget()
        self.entries_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        entries_layout = QVBoxLayout(self.entries_panel)
        entries_layout.setContentsMargins(0, 4, 0, 4)
        entries_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.entries_hint = QLabel("仅勾选行用于找图，分别搜索并合并去重。原有说明作为 AI 约束，可整理为条目。点选一行后补充细节。")
        self.entries_hint.setWordWrap(True)
        self.entries_hint.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        entries_layout.addWidget(self.entries_hint)
        self.entries_table = ReferenceEntriesTable()
        entries_layout.addWidget(self.entries_table)
        entry_actions = QHBoxLayout()
        self.add_entry_button = self._button("＋ 手动添加条目", self.add_entry, entry_actions)
        self.delete_entry_button = self._button("删除选中行", self.delete_entry, entry_actions)
        self.organize_button = self._button("将原文整理为条目…", lambda: w._generate_creative_children_for_selected_node(completion_mode="organize"), entry_actions)
        self.edit_base_button = self._button("编辑原有说明", self.edit_base_note, entry_actions)
        self.original_button = self._button("查看完整说明", self.toggle_original, entry_actions)
        entry_actions.addStretch()
        entries_layout.addLayout(entry_actions)
        node_layout.insertWidget(node_layout.indexOf(w.creative_node_note_input) + 1, self.entries_panel)
        self.entries_table.entriesChanged.connect(self.save_entry_edits)
        self.proposal_panel = QWidget()
        self.proposal_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        proposals = QVBoxLayout(self.proposal_panel)
        proposals.setContentsMargins(0, 4, 0, 4)
        proposals.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.proposal_label = QLabel("AI 建议 · 可先修改文字，再采纳；也可选中其中一段追加到当前说明")
        self.proposal_label.setWordWrap(True)
        self.proposal_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        proposals.addWidget(self.proposal_label)
        self.proposal_entries_table = ReferenceEntriesTable()
        self.proposal_entries_table.setAccessibleName("AI 建议条目表")
        proposals.addWidget(self.proposal_entries_table)
        self.proposal_entries_table.entriesChanged.connect(self.save_pending_entries)
        self.proposal_input = QTextEdit()
        self.proposal_input.setAcceptRichText(False)
        self.proposal_input.setMaximumHeight(180)
        self.proposal_query = QLineEdit()
        self.proposal_query.setPlaceholderText("建议的找图语句")
        proposals.addWidget(self.proposal_input)
        proposals.addWidget(self.proposal_query)
        accept_row = QHBoxLayout()
        self.accept_button = self._button("采纳编辑后的建议", self.accept_note, accept_row)
        self.partial_button = self._button("追加选中段落", lambda: self.accept_note(partial=True), accept_row)
        self._button("放弃建议", self.discard_note, accept_row)
        proposals.addLayout(accept_row)
        node_layout.addWidget(self.proposal_panel)
        self.proposal_input.textChanged.connect(self.save_proposal_edits)
        self.proposal_query.textChanged.connect(self.save_proposal_edits)
        probe_layout = w.inspiration_term_list.parentWidget().layout()
        self._button("把选中的灵感方向加入当前创作项目", self.convert_probes, probe_layout)
        w.inspiration_brief_input.textChanged.connect(self.save_context)
        w.inspiration_answers_input.textChanged.connect(self.save_context)
        w.center_result_stack.currentChanged.connect(self.view_changed)
        w.grid_view.selectionChanged.connect(self.refresh_reference)
        w.project_board_view.selectionChanged.connect(self.refresh_reference)
        self.refresh()

    def save_context(self):
        w = self.w
        if self.loading or not self.project_id or w.ai_workflow_stack.currentIndex() != 0:
            return
        if w.store.get_creative_project(self.project_id):
            if hasattr(w, 'workflow_upgrade'):
                old_key = w.workflow_upgrade.state.context_key(self.project_id)
                for node in w.store.list_creative_nodes(self.project_id):
                    if not w.workflow_upgrade.state.options(node.id)['reviewed_context']:
                        w.workflow_upgrade.state.set_options(node.id, reviewed_context=old_key)
            w.store.update_creative_project_details(self.project_id, brief=w.inspiration_brief_input.toPlainText().strip())
            self.store.save_project(self.project_id, extra=w.inspiration_answers_input.toPlainText().strip())

    def before_project_load(self, project_id):
        if project_id != self.project_id:
            self.save_context()
            self.save_search()
            if project_id and not self.store.project(project_id)["scope_json"]:
                self.store.save_project(project_id, scope=self.w._current_view_payload())

    def load_project(self, project_id):
        if project_id == self.project_id:
            return
        self.project_id = project_id
        self.search_node_id = None
        self.probe_draft = None
        self.load_context()

    def load_context(self):
        w = self.w
        project = w.store.get_creative_project(self.project_id) if self.project_id else None
        self.loading = True
        try:
            w.inspiration_brief_input.setPlainText(project.brief if project else "")
            w.inspiration_answers_input.setPlainText(self.store.project(project.id)["extra"] if project else "")
        finally:
            self.loading = False

    def switch_ai_mode(self, mode):
        w = self.w
        previous = w.ai_workflow_stack.currentIndex()
        if (mode == "probe") == (previous == 1):
            return
        if mode == "probe":
            self.save_context()
            if self.probe_draft:
                self.loading = True
                w.inspiration_brief_input.setPlainText(self.probe_draft[0])
                w.inspiration_answers_input.setPlainText(self.probe_draft[1])
                self.loading = False
        else:
            self.probe_draft = (w.inspiration_brief_input.toPlainText(), w.inspiration_answers_input.toPlainText())
            if self.project_id:
                self.load_context()

    def ai_context(self, project):
        self.save_context()
        saved = self.w.store.get_creative_project(project.id)
        return (saved.brief or saved.title), self.store.project(project.id)["extra"]

    def refresh(self):
        w = self.w
        node = w._current_creative_node()
        state = self.store.node(node.id) if node else {}
        locked = bool(state.get("locked"))
        self.loading = True
        self.lock_button.setChecked(locked)
        self.lock_button.setEnabled(node is not None)
        self.proposal_input.setPlainText(state.get("proposal_note", ""))
        self.proposal_query.setText(state.get("proposal_query", ""))
        self.proposal_panel.setVisible(bool(state.get("proposal_note") or state.get("proposal_query")))
        self.proposal_panel.setEnabled(not locked)
        self.restore_button.setEnabled(bool(node and not locked and self.store.version_count(node.id)))
        self.loading = False
        self.refresh_entries(node, state, locked)
        structured = bool(state.get("entries_json"))
        w.creative_node_note_input.setReadOnly(locked or structured)
        w.creative_node_query_input.setReadOnly(locked or structured)
        if locked:
            w.generate_creative_children_button.setEnabled(False)
            w.generate_creative_details_button.setEnabled(False)
        label = f"加入『{node.title}』" if node else "先选择创作节点"
        self.collect_button.setText(label)
        w.save_selection_to_creative_node_button.setText(label)
        self.collect_button.setEnabled(bool(node and w._selected_grid_images()))
        self.find_button.setEnabled(node is not None)
        path = w._creative_node_path_text(node.id) if node else "图库浏览 · 可独立找图，也可新建创作项目"
        self.context_label.setText(path)
        self.context_label.setToolTip(path)
        count = len(w.store.creative_node_image_ids(node.id, include_descendants=True)) if node else 0
        self.collected_button.setText(f"已收集 {count}")
        if w.current_creative_project_id:
            for other in w.store.list_creative_nodes(w.current_creative_project_id):
                item = w._find_creative_node_tree_item(other.id)
                if item is not None:
                    info = self.store.node(other.id)
                    suffix = " · 已确定" if info["locked"] else (" · 待采纳" if info["proposal_note"] else "")
                    item.setText(0, other.title + suffix)
        self.refresh_project_references()
        self.refresh_reference()
        if hasattr(w, "workspace_chrome"):
            w.workspace_chrome.refresh()
        if hasattr(w, 'workflow_upgrade'):
            w.workflow_upgrade.refresh()

    def refresh_project_references(self):
        """Membership survives result rebuilds, independently of search badges."""
        w = self.w
        project_id = w.current_creative_project_id
        references = w.store.creative_node_image_badges(project_id, use_node_titles=True) if project_id else {}
        w.grid_view.set_project_references(references)

    def view_changed(self, _index=0):
        w = self.w
        is_gallery = w.center_result_stack.currentWidget() is w.grid_view
        w.filter_panel.setVisible(is_gallery)
        self.probe_tuning.setVisible(is_gallery and w.current_result_mode == "inspiration")
        self.group_button.setVisible(is_gallery)
        for button in self.board_buttons:
            button.setVisible(w.center_result_stack.currentWidget() is w.project_board_view)
        if not is_gallery:
            self.save_search()
        self.refresh_reference()

    def new_project(self):
        self.w._load_creative_project(None)
        self.w._set_ai_workflow_mode("project")
        self.load_context()
        self.show_planning()
        self.w.inspiration_brief_input.setFocus()

    def show_planning(self):
        w = self.w
        w._save_current_board_layout_if_needed()
        w._set_board_focus_mode(False)
        w._set_board_window_pinned(False)
        w._set_board_toolbar_visible(False)
        w.center_result_stack.setCurrentWidget(self.planning_scroll)
        w.right_tab_widget.setCurrentIndex(1)
        self.refresh()
        w.statusBar().showMessage("构思 · " + self.context_label.text())

    def show_project_planning(self):
        self.w._set_ai_workflow_mode("project")
        self.show_planning()

    def show_probe(self):
        self.w._set_ai_workflow_mode("probe")
        self.show_planning()

    def find(self):
        self.w._search_selected_creative_node()

    def start_node_search(self, node, query, *, queries=None):
        w = self.w
        self.save_search()
        scope = w._current_view_payload()
        # A node board is a collection of chosen references, not the library search scope.
        if w.current_result_mode == "creative_node":
            scope = decode_state(self.store.project(node.project_id)["scope_json"]) or scope
            if scope.get("filter_scope", {}).get("kind") == "snapshot":
                scope["filter_scope"] = {"kind": "sidebar"}
                scope["base_image_ids"] = None
                scope["base_label"] = None
        names = queries if queries is not None else [query]
        if not names:
            w.statusBar().showMessage("请至少勾选一个有名称的参考条目")
            return
        previous = FilterPlan.from_payload(scope.get("filter_plan", {}))
        strengths = {f.value: f.score_threshold for f in previous.positive_filters if f.kind == "semantic"}
        plan = FilterPlan((FilterGroup(tuple(SearchFilter("semantic", name, strengths.get(name, 50)) for name in names), match_mode="any"),)) if queries is not None else FilterPlan((FilterGroup((SearchFilter("semantic", query, 50),)),))
        scope.update(filter_plan=plan.to_payload(), search_filters=[], excluded_image_ids=[], excluded_collection_ids=[])
        self.store.save_project(node.project_id, scope=scope)
        self.search_node_id = node.id
        self.active_search_names = names if queries is not None else None
        w.current_result_mode = "search_chain"
        w._show_gallery_view()
        w._apply_view_payload(scope)
        w.right_tab_widget.setCurrentIndex(1)
        self.refresh()

    def save_search(self):
        w = self.w
        if not self.search_node_id or w.current_result_mode != "search_chain":
            return
        if not w.search_button.isEnabled() or w.filter_panel.error:
            return
        state = {"entry_names": getattr(self, "active_search_names", None), "view": w._current_view_payload(), "scroll": w.grid_view.verticalScrollBar().value(),
                 "selected": w.grid_view.selected_image_ids()}
        self.search_states[self.search_node_id] = state
        self.store.save_node(self.search_node_id, search_json=json.dumps(state, ensure_ascii=False))
        node = w.store.get_creative_node(self.search_node_id)
        if node:
            self.store.save_project(node.project_id, scope=state["view"])

    def show_candidates(self):
        w = self.w
        node = w._current_creative_node()
        if node is None:
            w._show_gallery_view()
            return
        state = self.search_states.get(node.id) or decode_state(self.store.node(node.id)["search_json"])
        payload = self.store.entries(node.id)
        names = w.workflow_upgrade.queries(node) if hasattr(w, 'workflow_upgrade') else search_names(payload) if payload else None
        if payload is not None and state.get("entry_names") != names:
            state = {}
        if state:
            self.active_search_names = state.get("entry_names")
            self.search_node_id = node.id
            self.restore_position = (state.get("scroll", 0), state.get("selected", []))
            w.current_result_mode = "search_chain"
            w._show_gallery_view()
            w._apply_view_payload(state["view"])
        else:
            self.find()

    def search_finished(self):
        w = self.w
        self.refresh_project_references()
        if self.search_node_id and self.search_node_id == w.current_creative_node_id:
            badges = w.store.creative_node_image_badges(w.current_creative_project_id)
            w.grid_view.set_images(w.current_chain_filtered_images, badges_by_image_id=badges)
        if hasattr(w, 'workflow_upgrade'):
            w.workflow_upgrade.search_finished()
        if self.restore_position is not None:
            scroll, selected = self.restore_position
            self.restore_position = None
            if selected:
                w.grid_view.select_image_ids(selected)
            w.grid_view.verticalScrollBar().setValue(scroll)
        self.refresh_reference()

    def show_collected(self):
        node = self.w._current_creative_node()
        if node:
            self.save_search()
            self.w._clear_result_management_state()
            self.w._show_creative_node_saved_images(node.id)
            if hasattr(self.w, "workflow_upgrade"):
                self.w.workflow_upgrade.display_collected()
            self.w.right_tab_widget.setCurrentIndex(1)
            self.refresh()

    def show_board(self):
        self.save_search()
        self.w._show_current_project_board()

    def toggle_lock(self, checked):
        node = self.w._current_creative_node()
        if not self.loading and node:
            self.w._save_current_creative_node_editor_snapshot()
            self.store.save_node(node.id, locked=int(checked))
            if checked and hasattr(self.w, 'workflow_upgrade'):
                self.w.workflow_upgrade.state.set_options(node.id, reviewed_context=self.w.workflow_upgrade.state.context_key(node.project_id))
            self.w._sync_creative_node_panel()
            self.w.statusBar().showMessage("节点已锁定" if checked else "节点已解锁")

    def content_format(self, node):
        if not node:
            return "text"
        return self.store.content_format(node.id, node.title, self.w._creative_node_path_text(node.id))

    def change_content_format(self):
        node = self.w._current_creative_node()
        if self.loading or not node:
            return
        state = self.store.node(node.id)
        if state["locked"] or state["proposal_note"] or state["proposal_entries_json"] or self.w._creative_node_note_in_progress or self.w._creative_all_nodes_in_progress:
            return
        target = self.format_combo.currentData()
        if target == self.content_format(node):
            return
        self.w._save_current_creative_node_editor_snapshot()
        node = self.w.store.get_creative_node(node.id)
        # Explicit conversion preserves the full text, snapshots row IDs/choices,
        # and can be undone through the existing restore action.
        if self.store.accept(node.id, note=node.note, query=node.search_query,
                             entries_json="" if target == "text" else None, content_format=target):
            self.invalidate_entry_search(node.id)
            self.reload_node(node.id)
            self.w.statusBar().showMessage("内容形式已切换，原文已保留；可恢复切换前的内容")

    def refresh_entries(self, node, state, locked):
        payload = decode_entries(state.get("entries_json", ""))
        pending = decode_entries(state.get("proposal_entries_json", ""))
        content_format = self.content_format(node)
        self.format_combo.blockSignals(True)
        self.format_combo.setCurrentIndex(self.format_combo.findData(content_format))
        self.format_combo.blockSignals(False)
        self.format_combo.setEnabled(bool(node and not locked and not pending and not state.get("proposal_note")
            and not self.w._creative_node_note_in_progress and not self.w._creative_all_nodes_in_progress))
        self.entries_panel.setVisible(node is not None and content_format == "entries")
        self.entries_table.setVisible(payload is not None)
        self.entries_hint.setVisible(payload is not None)
        self.entries_table.set_entries((payload or {}).get("rows", []), locked=locked)
        self.delete_entry_button.setVisible(payload is not None)
        for button in (self.add_entry_button, self.delete_entry_button, self.organize_button):
            button.setEnabled(bool(node and not locked))
        busy = self.w._creative_node_note_in_progress or self.w._creative_all_nodes_in_progress
        original = payload.get("base_note", "") if payload is not None else (node.note if node else "")
        self.organize_button.setEnabled(bool(node and original.strip() and not locked and not pending and not busy and not state.get("proposal_note")))
        self.organize_button.setVisible(True)
        self.original_button.setVisible(payload is not None)
        self.edit_base_button.setVisible(payload is not None)
        self.edit_base_button.setEnabled(bool(node and not locked))
        if getattr(self, "_entries_node", None) != (node.id if node else None):
            self._entries_node = node.id if node else None
            self._original_visible = False
        self.w.creative_node_note_input.setVisible(payload is None or getattr(self, "_original_visible", False))
        self.w.creative_node_note_input.setMinimumHeight(100)
        self.w.creative_node_note_input.setMaximumHeight(170)
        self.w.creative_node_query_input.setVisible(payload is None)
        self.w.search_creative_node_button.setText("按勾选条目找参考" if payload is not None else "用此语句找参考")
        self.proposal_entries_table.setVisible(pending is not None)
        self.proposal_entries_table.set_entries((pending or {}).get("rows", []), locked=locked, detail_target=bool(pending and pending.get("target")))
        self.proposal_input.setVisible(pending is None)
        self.proposal_query.setVisible(pending is None)
        self.partial_button.setVisible(pending is None)
        self.accept_button.setText("采纳这些条目" if pending else "采纳编辑后的建议")
        self.proposal_label.setText("AI 建议 · 名称和描述可修改；采纳后保存所有行，只有勾选行参与找图。" if pending else "AI 建议 · 可先修改文字，再采纳")
        if pending and pending.get("target"):
            self.proposal_label.setText("所选条目的细节建议 · 采纳后仅更新这一行的描述，搜索内容与勾选状态保持不变。")
            self.accept_button.setText("采纳细节")
        if pending:
            self.proposal_panel.show()
        if state.get("proposal_note") or pending:
            self.w.generate_creative_children_button.setEnabled(False)
            self.w.generate_creative_details_button.setEnabled(False)

    def edit_base_note(self):
        node = self.w._current_creative_node()
        if not node or self.store.node(node.id)["locked"]:
            return
        payload = self.store.entries(node.id)
        if payload is None:
            return
        dialog = QDialog(self.w)
        dialog.setWindowTitle("编辑原有说明")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("原有说明是 AI 必须遵守的内容。找图目标由表格中的勾选行决定。"))
        editor = QTextEdit(payload.get("base_note", ""))
        editor.setAcceptRichText(False)
        layout.addWidget(editor)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(640, 300)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            payload["base_note"] = editor.toPlainText()
            if self.store.save_entries(node.id, payload, snapshot=True):
                self.reload_node(node.id)

    def toggle_original(self):
        self._original_visible = not getattr(self, "_original_visible", False)
        self.w.creative_node_note_input.setVisible(self._original_visible)
        self.original_button.setText("收起完整说明" if self._original_visible else "查看完整说明")

    def selected_entry_target(self):
        return self.entries_table.selected_entry()

    def invalidate_entry_search(self, node_id):
        self.search_states.pop(node_id, None)
        self.store.save_node(node_id, search_json="")
        if self.search_node_id == node_id:
            self.search_node_id = None

    def save_entry_edits(self, rows):
        node = self.w._current_creative_node()
        if self.loading or node is None:
            return
        payload = self.store.entries(node.id)
        if payload is None:
            return
        old_names = search_names(payload)
        payload["rows"] = rows
        if self.store.save_entries(node.id, payload):
            if old_names != search_names(payload):
                self.invalidate_entry_search(node.id)
            # Synchronize compatibility fields without rebuilding the active table/editor.
            updated = self.w.store.get_creative_node(node.id)
            self.w._set_creative_node_editor_draft(node.id, note=updated.note, search_query=updated.search_query)
            for widget, value, setter in ((self.w.creative_node_note_input, updated.note, "setPlainText"),
                                          (self.w.creative_node_query_input, updated.search_query, "setText")):
                widget.blockSignals(True)
                getattr(widget, setter)(value)
                widget.blockSignals(False)
            self.w.statusBar().showMessage(f"已保存 · 勾选 {len(search_names(payload))} 项；点“找参考”按当前勾选搜索")

    def save_pending_entries(self, rows):
        node = self.w._current_creative_node()
        if node is None or self.loading or self.store.node(node.id)["locked"]:
            return
        payload = decode_entries(self.store.node(node.id)["proposal_entries_json"])
        if payload:
            payload["rows"] = rows
            self.store.save_node(node.id, proposal_entries_json=encode_entries(payload))

    def add_entry(self):
        node = self.w._current_creative_node()
        if not node or self.store.node(node.id)["locked"]:
            return
        dialog = QDialog(self.w)
        dialog.setWindowTitle("添加参考条目")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("搜索内容（含必要限定，例如：废土拼装汽车）"))
        name = QLineEdit()
        name.setAccessibleName("参考条目搜索内容")
        layout.addWidget(name)
        layout.addWidget(QLabel("细节描述（可选，不自动参与搜索）"))
        visual = QTextEdit()
        visual.setAccessibleName("参考条目细节描述")
        layout.addWidget(visual)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        name.textChanged.connect(lambda value: buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(value.strip())))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.w._save_current_creative_node_editor_snapshot()
        node = self.w.store.get_creative_node(node.id)
        payload = self.store.entries(node.id) or {"base_note": node.note, "base_query": node.search_query, "rows": []}
        payload["rows"].append(new_entry(name.text(), visual.toPlainText()))
        if self.store.save_entries(node.id, payload, snapshot=True):
            self.invalidate_entry_search(node.id)
            self.reload_node(node.id)

    def delete_entry(self):
        node = self.w._current_creative_node()
        row = self.selected_entry_target()
        if node and row and not self.store.node(node.id)["locked"]:
            payload = self.store.entries(node.id)
            payload["rows"] = [r for r in payload["rows"] if r["id"] != row["id"]]
            if self.store.save_entries(node.id, payload, snapshot=True):
                self.invalidate_entry_search(node.id)
                self.reload_node(node.id)
                self.w.statusBar().showMessage("条目已删除，可用“恢复”撤销")

    def save_proposal_edits(self):
        node = self.w._current_creative_node()
        if not self.loading and node:
            self.store.save_node(node.id, proposal_note=self.proposal_input.toPlainText(), proposal_query=self.proposal_query.text())

    def accept_note(self, _checked=False, *, partial=False):
        w = self.w
        node = w._current_creative_node()
        if node is None:
            return
        if self.store.node(node.id)["proposal_entries_json"]:
            w._save_current_creative_node_editor_snapshot()
            if self.store.accept_entries(node.id):
                self.invalidate_entry_search(node.id)
                self.reload_node(node.id)
                w.statusBar().showMessage("条目已采纳；仅勾选内容参与找图，可恢复采纳前的内容")
            else:
                w.statusBar().showMessage("无法采纳：目标条目已更改、节点已锁定或存在空名称，请检查建议")
            return
        if partial:
            text = self.proposal_input.textCursor().selectedText().replace("\u2029", "\n").strip()
            if not text:
                w.statusBar().showMessage("先在 AI 建议中选中要追加的段落")
                return
            note = w.creative_node_note_input.toPlainText().rstrip() + "\n" + text
            query = w.creative_node_query_input.text()
        else:
            note, query = self.proposal_input.toPlainText(), self.proposal_query.text()
        w._save_current_creative_node_editor_snapshot()
        entries = self.store.entries(node.id)
        if entries is not None:
            # Older copy-writing providers may still return prose; preserve row identity and choices.
            entries["base_note"] = note.strip()
            note, query = composed_note(entries), " ".join(search_names(entries))[:240]
        if self.store.accept(node.id, note=note.strip(), query=query.strip(), keep_proposal=partial,
                             entries_json=encode_entries(entries) if entries is not None else None):
            if partial:
                remaining = self.proposal_input.toPlainText().replace(text, "", 1).strip()
                self.store.save_node(node.id, proposal_note=remaining)
            self.reload_node(node.id)
            w.statusBar().showMessage("已采纳建议，可恢复采纳前的内容")

    def reload_node(self, node_id):
        w = self.w
        node = w.store.get_creative_node(node_id)
        if node:
            w._set_creative_node_editor_draft(node_id, note=node.note, search_query=node.search_query)
        w._sync_creative_node_panel()

    def discard_note(self):
        node = self.w._current_creative_node()
        if node:
            self.store.save_node(node.id, proposal_note="", proposal_query="", proposal_entries_json="")
            self.w._sync_creative_node_panel()

    def restore_note(self):
        node = self.w._current_creative_node()
        if node and self.store.restore(node.id):
            self.invalidate_entry_search(node.id)
            self.reload_node(node.id)
            self.w.statusBar().showMessage("已恢复上次采纳前的内容")

    def refresh_reference(self, *_args):
        w = self.w
        node = w._current_creative_node()
        is_board = w.center_result_stack.currentWidget() is w.project_board_view
        selected = w.store.images_by_ids(w.project_board_view.selected_image_ids()) if is_board else w._selected_grid_images()
        self.collect_button.setEnabled(bool(node and selected and not is_board))
        self.group_button.setEnabled(len(selected) >= 4 and not is_board)
        key = None
        if node and len(selected) == 1 and selected[0].id in w.store.creative_node_image_ids(node.id):
            key = (node.id, selected[0].id)
        if key != self._reference_key:
            self._reference_key = key
            self.reference_input.blockSignals(True)
            self.reference_input.setPlainText(self.store.reference_note(*key) if key else "")
            self.reference_input.blockSignals(False)
        self.reference_input.setEnabled(key is not None)
        if hasattr(w, 'workflow_upgrade'):
            w.workflow_upgrade.refresh()

    def save_reference(self):
        if self._reference_key:
            self.store.save_reference_note(*self._reference_key, self.reference_input.toPlainText())

    def convert_probes(self):
        w = self.w
        terms = w._selected_inspiration_terms()
        node = w._current_creative_node()
        if not terms or node is None:
            w.statusBar().showMessage("先选择创作项目和要加入的灵感方向")
            return
        self.store.save_project(node.project_id, probe={"history_id": w.current_inspiration_project_id, "titles": [term.title for term in terms]})
        first = None
        for term in terms:
            new_id = w.store.create_creative_node(project_id=node.project_id, parent_id=node.id, title=term.title,
                                                 note=term.reason or term.title, search_query=term.query)
            first = first or new_id
            images = [im.id for im in w._selected_grid_images() if any(match.query == term.query for match in w.current_inspiration_matches.get(im.id, []))]
            if images:
                w.store.add_images_to_creative_node(new_id, images, intent_label=term.title, intent_query=term.query)
        w._set_ai_workflow_mode("project")
        w._refresh_creative_projects(select_project_id=node.project_id)
        w._load_creative_project(node.project_id, select_node_id=first)
        self.show_planning()
        w.statusBar().showMessage(f"已加入 {len(terms)} 个灵感节点，并带入选中的匹配图片")

    def export_project(self):
        w = self.w
        if not w.current_creative_project_id:
            w.statusBar().showMessage("先选择创作项目")
            return
        folder = QFileDialog.getExistingDirectory(w, "选择创作参考包保存位置")
        if folder:
            from eidory.core.creative_export import export_creative_project
            self.save_context()
            w._save_current_creative_node_editor_snapshot()
            w._save_current_board_layout_if_needed()
            try:
                path = export_creative_project(w.store, w.current_creative_project_id, folder, render_board=self.render_board)
                QMessageBox.information(w, "创作参考包已导出", f"已保存到：\n{path}\n打开 index.html 可查看设定、参考说明和看板。")
            except Exception as exc:
                QMessageBox.warning(w, "导出失败", str(exc))

    def render_board(self, node_id, path):
        from eidory.ui.project_board import ProjectBoardView
        data = self.w.store.creative_node_branch_board_data(node_id)
        board = ProjectBoardView()
        if hasattr(self.w, 'workflow_upgrade'):
            self.w.workflow_upgrade.decorate_references(data['node'].project_id, data['badges'])
        board.set_images(data['images'], title=data['node'].title,
                         layout_payload=decode_state(self.w.store.get_creative_node_board_layout(node_id)),
                         badges_by_image_id=data['badges'])
        board.render_to_png(path)
        board.deleteLater()
