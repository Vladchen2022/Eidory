"""One automatic search update path and reversible, query-local visual feedback."""
from dataclasses import replace
from weakref import ref
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QPushButton, QMenu
from eidory.core.search_filters import FilterPlan
from eidory.core.filter_engine import threshold_matches, matching_group_ids


class SimpleSearch(QObject):
    ranked = Signal(object, object)

    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self.pending = None
        self.serial = 0
        self.feedback = {}
        self.scores = {}
        self.context = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(450)
        self.timer.timeout.connect(self.update)
        self.ranked.connect(self.accept_ranking)
        panel = window.filter_panel
        panel.draftChanged.connect(lambda: self.schedule('filters'))
        panel.strengthChanged.connect(lambda _: self.schedule('strength'))
        panel.clear_button.setToolTip('清除后自动更新图片墙')
        panel.apply_button.setText('更新结果')
        panel.applyRequested.disconnect()
        panel.applyRequested.connect(lambda _: self.update(force=True))
        self.rescue = QPushButton('没有结果？')
        self.rescue.clicked.connect(self.show_rescue)
        panel.result_row.insertWidget(1, self.rescue)
        self.rescue.hide()
        self.reset_feedback = QPushButton('恢复原排序')
        self.reset_feedback.clicked.connect(self.clear_feedback)
        window.workflow_upgrade.results_bar.layout().addWidget(self.reset_feedback)
        self.reset_feedback.hide()

    @property
    def w(self):
        return self._window()

    def schedule(self, source):
        # A target edit owns the next query; slider flushes must not replace it.
        if self.pending != 'targets':
            self.pending = source
        if source == 'targets':
            self.target_node = self.w.current_creative_node_id
        self.timer.start()
        self.refresh()

    def update(self, *, force=False):
        w, panel = self.w, self.w.filter_panel
        source = self.pending
        self.timer.stop()
        if getattr(w, '_closing', False):
            return
        if not w.search_button.isEnabled():
            self.timer.start()
            return
        self.pending = None
        if source == 'targets':
            if self.target_node != w.current_creative_node_id:
                return
            w.creative_flow.run_targets()
        elif panel.draft != panel.applied or panel.clear_context_on_apply or force or panel.error:
            w.filter_workflow.apply(panel.draft)
        self.refresh()

    def signature(self):
        w = self.w
        return (w.current_creative_project_id, w.current_creative_node_id,
                repr(w.filter_workflow.serialized_scope()), w.current_chain_result.evaluated_plan)

    def refresh(self):
        w, p = self.w, self.w.filter_panel
        gallery = w.center_result_stack.currentWidget() is w.grid_view
        # The same rule applies to text, checkboxes and sliders: edits update
        # automatically. Keep a manual retry only when useful.
        p.apply_button.setVisible(bool(p.error))
        if not p._busy and not p.error:
            p.status_label.setText('正在准备更新…' if self.pending else '自动更新')
        flow = w.creative_flow
        flow.refresh_target_controls()
        flow.search_refresh.hide()
        missing_targets = any(row['enabled'] and w.workflow_upgrade.result_mode.findData(row['query']) < 0
                              for row in flow.targets())
        flow.target_run.setVisible(w.center_result_stack.currentWidget() is flow.search_empty
                                  or (gallery and missing_targets and not p._busy and not self.pending))
        if self.pending == 'targets':
            flow.target_hint.setText('编辑后自动更新结果')
        self.rescue.setVisible(gallery and w.current_result_mode == 'search_chain'
                               and not p._busy and not p.error and not w.grid_view.images())
        if self.context is not None and self.context != self.signature():
            self.serial += 1
            self.feedback.clear()
            self.scores.clear()
            self.context = None
        self.reset_feedback.setVisible(bool(self.feedback) and gallery)

    def clear_feedback(self):
        if not self.feedback:
            return
        self.serial += 1
        self.feedback.clear()
        self.scores.clear()
        self.w.workflow_upgrade.display_results()
        self.refresh()

    def sort(self, images):
        if not self.scores or self.context != self.signature():
            return images
        size = max(1, len(images) - 1)
        # Original target rank remains the main signal. Feedback never widens
        # the scope, changes conditions, or deletes images.
        ranked = sorted(enumerate(images), key=lambda p: (p[0] / size - .65 * self.scores.get(p[1].id, 0), p[0]))
        return [image for _, image in ranked]

    def prefer(self, image_id, direction):
        w = self.w
        if w.current_result_mode != 'search_chain' or not w.search_button.isEnabled():
            return
        self.context = self.signature()
        before = (dict(self.feedback), dict(self.scores), self.context)
        proposed = {**self.feedback, image_id: direction}
        ids = {im.id for im in w.current_chain_filtered_images}
        self.serial += 1
        token = self.serial
        provider = w.embedding_provider
        def run():
            try:
                values = []
                for iid, sign in proposed.items():
                    vector = w.store.embedding_vector_for_image(iid, model_name=provider.model_name,
                        model_revision=provider.model_revision, embedding_dim=provider.dim)
                    if vector is None:
                        if iid == image_id:
                            self.ranked.emit(token, {'error': '此图片尚未建立语义索引，请先完成索引。'})
                            return
                        continue
                    values.append(sign * vector / max(float(np.linalg.norm(vector)), 1e-9))
                vector = np.sum(values, axis=0)
                scores = dict(w.vector_index.search(vector, len(ids), allowed_image_ids=ids))
                self.ranked.emit(token, {'feedback': proposed, 'scores': scores, 'before': before})
            except Exception as exc:
                self.ranked.emit(token, {'error': str(exc)})
        w.statusBar().showMessage('正在按图片偏好调整排序…')
        w._start_background_task(run)

    def accept_ranking(self, token, payload):
        if token != self.serial or self.context != self.signature():
            return
        if 'error' in payload:
            self.w.statusBar().showMessage(payload['error'])
            return
        self.feedback, self.scores = payload['feedback'], payload['scores']
        self.w.workflow_upgrade.request_recommendation_sort()
        if hasattr(self.w, 'edit_history'):
            after = (dict(self.feedback), dict(self.scores), self.context)
            self.w.edit_history.push('调整图片偏好', lambda: self.restore_feedback(payload['before']),
                                     lambda: self.restore_feedback(after))
        self.w.workflow_upgrade.display_results()
        self.w.statusBar().showMessage('已调整当前结果排序；可以撤销或恢复原排序')
        self.refresh()

    def restore_feedback(self, state):
        if state[2] != self.signature():
            return False
        self.serial += 1
        self.feedback, self.scores, self.context = dict(state[0]), dict(state[1]), state[2]
        self.w.workflow_upgrade.request_recommendation_sort()
        self.w.workflow_upgrade.display_results()
        self.refresh()
        return True

    def rescue_options(self):
        w = self.w
        result, plan = w.current_chain_result, w.filter_workflow.plan()
        matches = {f: {im.id for im in threshold_matches(f, raw)} for f, raw in result.filter_matches.items()}
        base = {im.id for im in result.ranked_scope_images or ()}
        def count(candidate):
            groups = [g for g in candidate.include if g.filters]
            keep = set.union(*(matching_group_ids(g, matches) for g in groups)) if groups else base
            exclude = set().union(*(matching_group_ids(g, matches) for g in candidate.exclude))
            return len(keep - exclude - w.result_excluded_image_ids - w.result_excluded_collection_image_ids)
        options = []
        for section in ('include', 'exclude'):
            for gi, group in enumerate(getattr(plan, section)):
                for fi, condition in enumerate(group.filters):
                    groups = list(getattr(plan, section))
                    groups[gi] = replace(group, filters=tuple(f for j, f in enumerate(group.filters) if j != fi))
                    candidate = replace(plan, **{section: tuple(groups)})
                    n = count(candidate)
                    options.append((n, f'移除「{w._filter_label(condition)}」 · 可保留 {n} 张', candidate))
        return sorted(options, key=lambda entry: -entry[0])

    def show_rescue(self):
        w = self.w
        menu = QMenu(self.rescue)
        for count, title, candidate in self.rescue_options()[:5]:
            menu.addAction(title, lambda _=False, p=candidate: w.filter_workflow.apply(p))
        if w.filter_workflow.scope_label() != '全部图库':
            menu.addAction('扩大到全部图库', w.filter_workflow.select_all_scope)
        if 'semantic' in {f.kind for f in w.filter_workflow.plan().positive_filters}:
            menu.addAction('检查并补建语义索引…', w.filter_workflow.open_index_panel)
        menu.addAction('清除条件，重新找图', w.filter_panel.clear_draft)
        menu.exec(self.rescue.mapToGlobal(self.rescue.rect().bottomLeft()))
