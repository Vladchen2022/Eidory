from __future__ import annotations

import json
from dataclasses import replace
from eidory.ui.edit_history import undoable

from weakref import ref

from PySide6.QtCore import QObject, Qt, Signal, QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QListWidget, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget, QMenu, QProgressDialog)

from eidory.core.project_state import ProjectStateStore, REFERENCE_STATUSES, PURPOSES, capture_project
from eidory.core.search_targets import resolved_targets, target_signature
from eidory.core.filter_engine import threshold_matches


class WorkflowUpgrade(QObject):
    jobFinished = Signal(str, object, object)
    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self.state = ProjectStateStore(window.store)
        self.refreshing = False
        self._toolbar_sort_requested = False
        self.jobs = {}
        self.jobFinished.connect(self.finish_job)
        self._build()
        QTimer.singleShot(1600, self.backup_automatically)

    @property
    def w(self):
        return self._window()

    def button(self, text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _build(self):
        w, cw = self.w, self.w.creative_workspace
        tools = QPushButton('项目工具 ▾')
        tools.setAccessibleName('项目工具')
        menu = QMenu(tools)
        self.project_actions = []
        for label, callback, needs_project in [
                ('主题与补充设定…', lambda: w.creative_flow.edit_context(), True),
                ('方案版本…', self.show_versions, True),
                ('导出创作参考包…', cw.export_project, True),
                ('导出可编辑项目包…', self.export_package, True),
                ('导入可编辑项目包…', self.import_package, False)]:
            action = menu.addAction(label, callback)
            if needs_project:
                self.project_actions.append(action)
        menu.addSeparator()
        self.batch_action = menu.addAction('批量补全未确定的内容', w.generate_all_creative_nodes_button.click)
        self.copy_action = menu.addAction('项目文案', w.open_creative_copy_button.click)
        menu.addAction('探索灵感方向', cw.show_probe)
        menu.aboutToShow.connect(lambda: self.batch_action.setEnabled(w.generate_all_creative_nodes_button.isEnabled()))
        menu.aboutToShow.connect(lambda: self.copy_action.setEnabled(w.open_creative_copy_button.isEnabled()))
        self.tools_menu = menu
        tools.setMenu(menu)
        cw.navigation.layout().addWidget(tools)
        self.tools = tools
        settings = w.workspace_chrome._dialogs[4].layout()
        utilities = QHBoxLayout()
        self.button('功能准备检查…', self.show_readiness, utilities)
        self.backup_hint = QLabel('自动备份：正在检查（每天一次，保留最近 7 份）')
        self.backup_hint.setWordWrap(True)
        utilities.addWidget(self.backup_hint, 1)
        self.button('查看自动备份', self.open_backups, utilities)
        settings.insertLayout(2, utilities)
        row = QHBoxLayout()
        self.target_button = self.button('检查与编辑搜索目标…', self.edit_targets, row)
        self.target_hint = QLabel('核心目标与条目分别找图，细节不自动加入搜索')
        self.target_hint.setWordWrap(True)
        self.target_hint.setProperty('role', 'muted')
        row.addWidget(self.target_hint, 1)
        node_layout = w.creative_node_note_input.parentWidget().layout()
        node_layout.insertLayout(node_layout.indexOf(cw.entries_panel) + 1, row)
        cw.entries_panel.setObjectName('confirmedEntriesPanel')
        cw.proposal_panel.setObjectName('pendingSuggestionPanel')
        self.reference_bar = QWidget()
        choices = QHBoxLayout(self.reference_bar)
        choices.setContentsMargins(0, 2, 0, 2)
        self.status = QComboBox()
        self.status.setAccessibleName('参考等级')
        for value, label in REFERENCE_STATUSES.items():
            self.status.addItem(label, value)
        self.purpose = QComboBox()
        self.purpose.setAccessibleName('参考用途类型')
        for value in PURPOSES:
            self.purpose.addItem(value or '选择用途', value)
        choices.addWidget(self.status)
        choices.addWidget(self.purpose)
        side = cw.reference_input.parentWidget().layout()
        side.insertWidget(side.indexOf(cw.reference_input), self.reference_bar)
        self.status.currentIndexChanged.connect(self.save_decision)
        self.purpose.currentIndexChanged.connect(self.save_decision)
        self.results_bar = QWidget()
        results = QHBoxLayout(self.results_bar)
        results.setContentsMargins(0, 2, 0, 2)
        results.addWidget(QLabel('候选视图'))
        self.result_mode = QComboBox()
        self.result_mode.setAccessibleName('综合推荐或按条目查看')
        self.result_mode.addItem('综合推荐', None)
        self.result_mode.currentIndexChanged.connect(self.result_mode_changed)
        results.addWidget(self.result_mode)
        self.result_hint = QLabel('分别检索，合并去重；标记表示命中的搜索目标')
        self.result_hint.setProperty('role', 'muted')
        results.addWidget(self.result_hint, 1)
        w.center_result_stack.parentWidget().layout().insertWidget(
            w.center_result_stack.parentWidget().layout().indexOf(w.center_result_stack), self.results_bar)
        self.results_bar.hide()
        self.collection_bar = QWidget()
        collected = QHBoxLayout(self.collection_bar)
        collected.setContentsMargins(0, 2, 0, 2)
        collected.addWidget(QLabel('参考等级'))
        self.collection_mode = QComboBox()
        self.collection_mode.setAccessibleName('按参考等级查看已收集图片')
        self.collection_mode.addItem('全部已收集', '')
        for key, value in REFERENCE_STATUSES.items():
            self.collection_mode.addItem(value, key)
        self.collection_mode.currentIndexChanged.connect(self.display_collected)
        collected.addWidget(self.collection_mode)
        self.collection_hint = QLabel('选中图片，在右侧“用途”设置等级与用途。')
        self.collection_hint.setWordWrap(True)
        collected.addWidget(self.collection_hint, 1)
        w.center_result_stack.parentWidget().layout().insertWidget(
            w.center_result_stack.parentWidget().layout().indexOf(w.center_result_stack), self.collection_bar)
        self.collection_bar.hide()
        w.grid_view.selectionChanged.connect(self.refresh)
        w.project_board_view.selectionChanged.connect(self.refresh)
        w.center_result_stack.currentChanged.connect(self.refresh)

    def flush(self):
        if hasattr(self.w, 'creative_flow'):
            if self.w.creative_flow._inline:
                self.w.creative_flow._inline.finish()
            self.w.creative_flow.board_inspector.flush_note()
        self.w.creative_workspace.save_context()
        self.w._save_current_creative_node_editor_snapshot()
        self.w._save_current_board_layout_if_needed()

    def refresh(self, *_args):
        if self.refreshing:
            return
        self.refreshing = True
        try:
            cw = self.w.creative_workspace
            node = self.w._current_creative_node()
            for action in self.project_actions:
                action.setEnabled(self.w.current_creative_project_id is not None)
            self.target_button.setEnabled(node is not None)
            key = cw._reference_key
            self.reference_bar.setEnabled(key is not None)
            d = self.state.decision(*key) if key else {'status': 'candidate', 'purpose': ''}
            self.status.setCurrentIndex(self.status.findData(d['status']))
            self.purpose.setCurrentIndex(self.purpose.findData(d['purpose']))
            self.collection_bar.setVisible(self.w.current_result_mode == 'creative_node' and self.w.center_result_stack.currentWidget() is self.w.grid_view)
            self.results_bar.setVisible(self.w.current_result_mode == 'search_chain' and self.w.center_result_stack.currentWidget() is self.w.grid_view)
        finally:
            self.refreshing = False
        if hasattr(self.w, 'creative_flow'):
            self.w.creative_flow.refresh()

    def queries(self, node):
        entries = self.w.creative_workspace.store.entries(node.id)
        rows = resolved_targets(node, entries, self.state.options(node.id), self.w._creative_node_path_text(node.id))
        return list(dict.fromkeys(r['query'].strip() for r in rows if r['enabled'] and r['query'].strip()))

    def edit_targets(self):
        node = self.w._current_creative_node()
        if not node:
            return
        self.flush()
        node = self.w.store.get_creative_node(node.id)
        entries = self.w.creative_workspace.store.entries(node.id)
        rows = resolved_targets(node, entries, self.state.options(node.id), self.w._creative_node_path_text(node.id))
        dialog = QDialog(self.w)
        dialog.setWindowTitle('搜索目标 · '+node.title)
        layout = QVBoxLayout(dialog)
        hint = QLabel('保留核心对象，也可只找某个局部。修改这里不会改写创作说明。\n例如：人物名可改为“科幻工程师工作服”；找动作可改为“多人协作维修”。细节描述不自动参与。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        table = QTableWidget(len(rows), 4)
        table.setAccessibleName('搜索目标预览表')
        table.setHorizontalHeaderLabels(['搜索', '来源', '创作内容', '实际找图语句'])
        for i, row in enumerate(rows):
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if row['enabled'] else Qt.CheckState.Unchecked)
            table.setItem(i, 0, item)
            for col, value in [(1, row['kind']), (2, row['name']), (3, row['query'])]:
                item = QTableWidgetItem(value)
                if col != 3:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(i, col, item)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        table.setColumnWidth(0, 50)
        table.setColumnWidth(1, 90)
        table.setColumnWidth(2, 180)
        layout.addWidget(table)
        actions = QHBoxLayout()
        def add():
            text, ok = QInputDialog.getText(dialog, '添加搜索目标', '要找什么参考？')
            if ok and text.strip():
                from uuid import uuid4
                rows.append({'id': uuid4().hex, 'kind': '手动目标', 'name': text.strip(), 'query': text.strip(), 'enabled': True})
                i = table.rowCount()
                table.insertRow(i)
                item = QTableWidgetItem()
                item.setCheckState(Qt.CheckState.Checked)
                table.setItem(i, 0, item)
                for col, value in enumerate(['手动目标', text.strip(), text.strip()], 1):
                    table.setItem(i, col, QTableWidgetItem(value))
        self.button('＋ 搜索目标', add, actions)
        actions.addStretch()
        self.button('取消', dialog.reject, actions)
        def save(run=False):
            for i, row in enumerate(rows):
                row['enabled'] = table.item(i, 0).checkState() == Qt.CheckState.Checked
                row['query'] = table.item(i, 3).text().strip()
            if any(r['enabled'] and not r['query'] for r in rows):
                QMessageBox.warning(dialog, '搜索内容为空', '请填写已勾选目标的找图语句。')
                return
            self.state.set_options(node.id, search_targets_json=json.dumps({'signature': target_signature(node, entries), 'rows': rows}, ensure_ascii=False))
            self.w.creative_workspace.invalidate_entry_search(node.id)
            dialog.accept()
            if run:
                self.w.creative_workspace.start_node_search(node, '', queries=self.queries(node))
        self.button('保存', save, actions)
        self.button('保存并找图', lambda: save(True), actions)
        layout.addLayout(actions)
        dialog.resize(900, 440)
        dialog.exec()

    def search_finished(self):
        result = self.w.current_chain_result
        if not result:
            return
        current = self.result_mode.currentData()
        self.result_mode.blockSignals(True)
        self.result_mode.clear()
        self.result_mode.addItem('综合推荐', None)
        self.result_mode.addItem('按工具栏排序', '__toolbar__')
        allowed = {im.id for im in self.w.current_chain_filtered_images}
        self.match_badges = {}
        self.match_ids = {}
        for f in result.evaluated_plan.positive_filters if result.evaluated_plan else ():
            if f.kind != 'semantic':
                continue
            ids = {im.id for im in threshold_matches(f, result.filter_matches.get(f, ()))} & allowed
            key = str(f.value)
            self.match_ids[key] = ids
            self.result_mode.addItem(f'{key[:32]} · {len(ids)} 张', key)
            for image_id in ids:
                self.match_badges.setdefault(image_id, []).append('命中：'+key[:20])
        # Folder/tag/size filters have no recommendation rank to choose from.
        if not self.match_ids:
            current = '__toolbar__'
        self.result_mode.setCurrentIndex(max(0, self.result_mode.findData(current)))
        self.result_mode.blockSignals(False)
        self.display_results()
        self.refresh()

    def request_toolbar_sort(self):
        self._toolbar_sort_requested = True
        if hasattr(self.w, 'simple_search'):
            # A pending preference result must not override a newer sort action.
            self.w.simple_search.serial += 1
        if self.result_mode.currentData() is None:
            previous = self.result_mode.blockSignals(True)
            self.result_mode.setCurrentIndex(max(0, self.result_mode.findData('__toolbar__')))
            self.result_mode.blockSignals(previous)

    def request_recommendation_sort(self):
        self._toolbar_sort_requested = False
        if self.result_mode.currentData() == '__toolbar__':
            previous = self.result_mode.blockSignals(True)
            self.result_mode.setCurrentIndex(0)
            self.result_mode.blockSignals(previous)

    def result_mode_changed(self, *_args):
        mode = self.result_mode.currentData()
        if mode is None:
            self.request_recommendation_sort()
        elif mode == '__toolbar__':
            self.request_toolbar_sort()
        self.display_results()

    def display_results(self, *_args):
        if self.w.current_result_mode != 'search_chain':
            return
        selected = self.w.grid_view.selected_image_ids()
        images = self.w.current_chain_filtered_images
        target = self.result_mode.currentData()
        toolbar_order = (self._toolbar_sort_requested or target == '__toolbar__'
                         or bool(self.w.manual_result_order_ids))
        scores = {}
        if target and target != '__toolbar__':
            images = [im for im in images if im.id in self.match_ids.get(target, set())]
            scores = {im.id: im.score for f, matches in self.w.current_chain_result.filter_matches.items() if str(f.value) == target for im in matches}
        if toolbar_order:
            if scores and self.w.current_sort_key == 'score':
                images = [replace(im, score=scores.get(im.id)) for im in images]
            images = self.w._sort_images(images)
        elif target:
            images = sorted(images, key=lambda im: (-(scores.get(im.id) or 0), im.id))
        else:
            ranks = {im.id: i for i, im in enumerate(self.w.current_chain_result.ranked_scope_images or images)}
            images = sorted(images, key=lambda im: ranks.get(im.id, len(ranks)))
        if not toolbar_order and hasattr(self.w, "simple_search"):
            images = self.w.simple_search.sort(images)
        badges = self.w.store.creative_node_image_badges(self.w.current_creative_project_id) if self.w.current_creative_project_id else {}
        node = self.w._current_creative_node()
        self.decorate_references(self.w.current_creative_project_id, badges, node.id if node else None)
        for image_id, reasons in getattr(self, 'match_badges', {}).items():
            badges.setdefault(image_id, []).extend(reasons)
        self.w.grid_view.set_images(images, badges_by_image_id=badges)
        self.w.grid_view.select_image_ids(selected)
        self.result_hint.setText(f'当前视图 {len(images)} 张 · 分别检索，合并去重；命中标记不代表识别概率')
        if hasattr(self.w, 'creative_flow'):
            self.w.creative_flow.refresh_search_status()

    @undoable("调整参考等级与用途")
    def save_decision(self, *_args):
        key = self.w.creative_workspace._reference_key
        if self.refreshing or not key:
            return
        self.state.set_decision(*key, self.status.currentData(), self.purpose.currentData())
        if self.w.center_result_stack.currentWidget() is self.w.project_board_view:
            self.w.creative_flow.board_inspector.refresh_badges()
            self.w.creative_flow.board_inspector.refresh()
        elif self.w.current_result_mode == 'creative_node':
            self.display_collected()
        self.w.statusBar().showMessage('已保存参考等级与用途')

    def display_collected(self, *_args):
        node = self.w._current_creative_node()
        if node is None or self.w.current_result_mode != 'creative_node':
            return
        data = self.w.store.creative_node_branch_board_data(node.id)
        selected = self.w.grid_view.selected_image_ids()
        badges = data['badges']
        self.decorate_references(node.project_id, badges, node.id)
        mode = self.collection_mode.currentData()
        images = self.w._sort_images(data['images'])
        if mode:
            def status(image):
                labels = badges.get(image.id, [])
                return 'primary' if '主参考' in labels else 'unused' if '暂不采用' in labels else 'candidate'
            images = [im for im in images if status(im) == mode]
        self.collection_hint.setText(f'当前显示 {len(images)}/{len(data["images"])} 张 · 暂不采用仍保留，可随时找回。')
        self.w.grid_view.set_images(images, badges_by_image_id=badges)
        self.w.grid_view.select_image_ids(selected)
        self.refresh()

    def decorate_references(self, project_id, badges, node_id=None):
        # A reference decision belongs to a node-image link, not the whole project.
        # Parent views aggregate only their branch; any active reference keeps a
        # shared image active even when another node has rejected it.
        with self.w.store.connect() as conn:
            rows = conn.execute("""
                WITH RECURSIVE scope(id) AS (
                    SELECT id FROM creative_nodes
                    WHERE project_id=? AND (? IS NULL OR id=?)
                    UNION
                    SELECT n.id FROM creative_nodes n JOIN scope s ON n.parent_id=s.id
                    WHERE n.project_id=?
                )
                SELECT l.image_id, COALESCE(d.status, 'candidate') AS status,
                       COALESCE(d.purpose, '') AS purpose,
                       COALESCE(NULLIF(TRIM(l.intent_label), ''), n.title) AS node_label
                FROM scope s JOIN creative_node_images l ON l.node_id=s.id
                JOIN creative_nodes n ON n.id=s.id
                LEFT JOIN creative_reference_decisions d
                  ON d.node_id=l.node_id AND d.image_id=l.image_id
            """, (project_id, node_id, node_id, project_id)).fetchall()
        decisions = {}
        priority = {'unused': 0, 'candidate': 1, 'primary': 2}
        for row in rows:
            image_id = row['image_id']
            if image_id in badges:
                decision = decisions.setdefault(image_id, {'status': 'unused', 'purposes': [], 'node_label': row['node_label']})
                if priority.get(row['status'], 1) > priority[decision['status']]:
                    decision['status'] = row['status']
                    decision['node_label'] = row['node_label']
                if row['purpose'] and row['purpose'] not in decision['purposes']:
                    decision['purposes'].append(row['purpose'])
        for image_id, decision in decisions.items():
            # A shared image goes in the section that actually uses it, rather
            # than an earlier node that marked it unused.
            label = decision['node_label']
            if label in badges[image_id]:
                badges[image_id].remove(label)
                badges[image_id].insert(0, label)
            badges[image_id].append(REFERENCE_STATUSES.get(decision['status'], '备选'))
            badges[image_id].extend(decision['purposes'])

    def show_overview(self):
        pid = self.w.current_creative_project_id
        if not pid:
            self.w.statusBar().showMessage('先选择创作项目')
            return
        self.flush()
        dialog = QDialog(self.w)
        dialog.setWindowTitle('项目检查 · '+self.w.store.get_creative_project(pid).title)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel('先处理待采纳与需复核内容，再选择主参考；不需要参考图的节点可明确标记。'))
        rows = self.state.overview(pid)
        table = QTableWidget(len(rows), 6)
        table.setAccessibleName('项目总览节点清单')
        table.setHorizontalHeaderLabels(['节点', '设定', '参考图', '主参考', '参考需要', '下一步'])
        for i, r in enumerate(rows):
            pending = bool(r['proposal'] or r['proposal_entries'])
            next_step = '复核主题变化' if r['needs_review'] else '审阅 AI 建议' if pending else '无需找图' if r['requirement']=='none' else '参考已齐' if r['requirement']=='resolved' else '确认参考是否已齐' if r['primary_count'] else '选择主参考' if r['image_count'] else '寻找参考'
            for j, value in [(0, r['title']), (1, '已确定' if r['locked'] else '可编辑'), (2, str(r['image_count'])), (3, str(r['primary_count'])), (5, next_step)]:
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(i, j, item)
            choice = QComboBox()
            for label, key in [('需要参考', 'needed'), ('无需参考图', 'none'), ('参考已齐', 'resolved')]:
                choice.addItem(label, key)
            choice.setCurrentIndex(choice.findData(r['requirement']))
            def change_need(_, n=r['id'], box=choice, row=i, record=r):
                value = box.currentData()
                self.state.set_options(n, reference_requirement=value)
                if not record['needs_review'] and not (record['proposal'] or record['proposal_entries']):
                    table.item(row, 5).setText('无需找图' if value == 'none' else '参考已齐' if value == 'resolved' else '确认参考是否已齐' if record['primary_count'] else '选择主参考' if record['image_count'] else '寻找参考')
            choice.currentIndexChanged.connect(change_need)
            table.setCellWidget(i, 4, choice)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)
        actions = QHBoxLayout()
        def open_node():
            i = table.currentRow()
            if i >= 0:
                self.w._load_creative_project(pid, select_node_id=rows[i]['id'])
                self.w.creative_workspace.show_planning()
                dialog.accept()
        def review():
            i = table.currentRow()
            if i >= 0:
                self.state.set_options(rows[i]['id'], reviewed_context=self.state.context_key(pid))
                rows[i]['needs_review'] = False
                table.item(i, 5).setText('已复核')
        self.button('已复核所选节点', review, actions)
        self.button('进入所选节点', open_node, actions)
        self.button('关闭', dialog.accept, actions)
        layout.addLayout(actions)
        dialog.resize(940, 480)
        dialog.exec()

    def show_versions(self):
        if self.w._creative_all_nodes_in_progress or self.w._creative_node_note_in_progress:
            QMessageBox.information(self.w, 'AI 正在生成', '请等待当前 AI 任务完成后再保存或恢复方案，避免生成结果落入另一套方案。')
            return
        pid = self.w.current_creative_project_id
        if not pid:
            self.w.statusBar().showMessage('先选择创作项目')
            return
        self.flush()
        dialog = QDialog(self.w)
        dialog.setWindowTitle('方案版本 · 文字、勾选、参考图和布局一起保存')
        layout = QVBoxLayout(dialog)
        listing = QListWidget()
        listing.setAccessibleName('项目方案列表')
        empty_hint = QLabel('还没有保存的方案。点击“保存当前方案”留下当前构思和看板。')
        empty_hint.setWordWrap(True)
        layout.addWidget(empty_hint)
        from eidory.ui.project_diff import ProjectDiff
        preview = ProjectDiff(self.w.store)
        listing.setMaximumHeight(150)
        layout.addWidget(listing)
        layout.addWidget(preview, 1)
        versions = []
        def reload():
            versions[:] = self.state.versions(pid)
            listing.clear()
            listing.addItems([v['title']+' · '+v['created_at'] for v in versions])
            empty_hint.setVisible(not versions)
            listing.setVisible(bool(versions))
            preview.setVisible(bool(versions))
            restore_button.setEnabled(bool(versions))
            if versions:
                listing.setCurrentRow(0)
        def show():
            i = listing.currentRow()
            restore_button.setEnabled(0 <= i < len(versions))
            if i < 0:
                return
            state = self.state.version(pid, versions[i]['id'])
            preview.display(state, capture_project(self.w.store, pid))
        listing.currentRowChanged.connect(show)
        actions = QHBoxLayout()
        def save():
            name, ok = QInputDialog.getText(dialog, '保存当前方案', '方案名称')
            if ok and name.strip():
                self.state.snapshot(pid, name)
                reload()
        def restore():
            i = listing.currentRow()
            if i < 0:
                return
            if QMessageBox.question(dialog, '恢复整套方案', '文字、勾选、参考图和布局将一起恢复；当前方案会先自动保存，可再切回来。') != QMessageBox.StandardButton.Yes:
                return
            self.w._clear_creative_node_editor_state()
            self.w.creative_workspace.search_node_id = None
            self.w._current_board_node_id = None
            self.w._current_board_image_ids = ()
            self.w._mark_current_board_layout_clean()
            self.state.restore(pid, versions[i]['id'])
            self.w.creative_workspace.search_states.clear()
            self.w.creative_workspace.project_id = None
            self.w.creative_workspace._reference_key = None
            self.w._load_creative_project(pid, save_current_editor=False)
            self.w.creative_workspace.show_planning()
            dialog.accept()
        self.button('保存当前方案', save, actions)
        restore_button = self.button('恢复所选方案', restore, actions)
        self.button('关闭', dialog.accept, actions)
        layout.addLayout(actions)
        reload()
        dialog.resize(850, 580)
        dialog.exec()

    def export_package(self):
        pid = self.w.current_creative_project_id
        if not pid:
            return
        self.flush()
        mode, ok = QInputDialog.getItem(self.w, '导出可编辑项目包', '素材方式', ['连同素材打包（可迁移）', '只保存原路径链接（原文件需保持可访问）'], 0, False)
        if not ok:
            return
        folder = QFileDialog.getExistingDirectory(self.w, '选择项目包保存位置')
        if folder:
            from eidory.core.project_package import export_project_package
            self.run_job('正在导出项目与素材…',
                lambda: export_project_package(self.w.store, pid, folder, include_files=mode.startswith('连同')),
                lambda target: QMessageBox.information(self.w, '项目包已保存', str(target)+'\n用“导入可编辑项目包”继续编辑。'))

    def import_package(self):
        path, _ = QFileDialog.getOpenFileName(self.w, '选择可编辑项目包', '', 'Eidory 项目 (project.eidory.json)')
        if not path:
            return
        from eidory.core.project_package import inspect_project_package, import_project_package
        def preview(result):
            _, doc, _, missing = result
            if QMessageBox.question(self.w, '导入预览', f"新建项目：{doc['state']['project']['title']}\n{len(doc['state']['nodes'])} 个节点，{len(doc['media'])} 张图片，{len(missing)} 个文件缺失。\n不会覆盖已有项目；素材复制到应用数据目录。") != QMessageBox.StandardButton.Yes:
                return
            def loaded(result):
                pid, missing = result
                self.w._refresh_creative_projects(select_project_id=pid)
                self.w._load_creative_project(pid)
                self.w.creative_workspace.show_planning()
                self.w.statusBar().showMessage(f'已导入可编辑项目 · {len(missing)} 个文件缺失')
            self.run_job('正在导入项目与素材…', lambda: import_project_package(self.w.store, path, self.w.paths.data_dir), loaded, modal=False)
        self.run_job('正在校验项目包…', lambda: inspect_project_package(path), preview)

    def run_job(self, label, work, done, *, silent=False, modal=True):
        from uuid import uuid4
        key = uuid4().hex
        progress = None
        if not silent:
            progress = QProgressDialog(label, '', 0, 0, self.w)
            progress.setWindowTitle('Eidory')
            progress.setCancelButton(None)
            progress.setWindowModality(Qt.WindowModality.ApplicationModal if modal else Qt.WindowModality.NonModal)
            progress.show()
        self.jobs[key] = (done, progress, silent)
        def run():
            try:
                result = work()
                self.jobFinished.emit(key, result, None)
            except Exception as exc:
                self.jobFinished.emit(key, None, str(exc))
        self.w._start_background_task(run, on_rejected=lambda: self.finish_job(key, None, '后台任务暂不可用，请稍后重试'))

    def finish_job(self, key, result, error):
        job = self.jobs.pop(key, None)
        if not job:
            return
        done, progress, silent = job
        if progress:
            progress.close()
            progress.deleteLater()
        if error:
            if silent:
                self.backup_hint.setText('自动备份未完成：' + error)
                self.w.statusBar().showMessage('自动备份未完成，请到设置查看')
            else:
                QMessageBox.warning(self.w, '操作未完成', error)
        else:
            done(result)

    def backup_automatically(self):
        from eidory.core.automatic_backup import automatic_backup
        self.run_job('', lambda: automatic_backup(self.w.paths.database_path, self.w.paths.data_dir / 'backups' / 'automatic'),
                     lambda p: self.backup_hint.setText('自动备份：'+p.name+' · 保留最近 7 份'), silent=True)

    def open_backups(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        folder = self.w.paths.data_dir / 'backups' / 'automatic'
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def show_readiness(self):
        dialog = QDialog(self.w)
        dialog.setWindowTitle('功能准备检查')
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel('图库浏览、语义索引和 AI 补全分别检查；一种服务不可用不影响其他已就绪功能。'))
        output = QTextEdit()
        output.setReadOnly(True)
        output.setAccessibleName('功能准备检查结果')
        output.setPlainText('点击“检查准备情况”检查本地图库、索引与模型服务。\n“测试 AI 生成”会向当前配置的模型发出一次小请求。')
        layout.addWidget(output)
        actions = QHBoxLayout()
        service = self.w._llm_service_key()
        endpoint, model, key = self.w._llm_endpoint(service), self.w._llm_model(service), self.w._llm_api_key(service)
        def check(generation=False):
            from eidory.core.readiness import library_readiness, check_model_service
            def work():
                report = library_readiness(self.w.store)
                ok, message = check_model_service(endpoint, model, key, generate=generation)
                report.append(('AI 实际生成' if generation else 'AI 模型服务', ok, message))
                return report
            self.run_job('正在测试 AI 生成…' if generation else '正在检查…', work,
                         lambda report: output.setPlainText('\n\n'.join(('就绪' if ok else '需处理')+' · '+name+'\n'+message for name, ok, message in report)))
        self.button('检查准备情况', lambda: check(), actions)
        self.button('测试 AI 生成', lambda: check(True), actions)
        self.button('关闭', dialog.accept, actions)
        layout.addLayout(actions)
        dialog.resize(780, 500)
        dialog.exec()
