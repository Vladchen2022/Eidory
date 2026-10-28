from __future__ import annotations

from dataclasses import replace
from weakref import ref, WeakMethod

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QMenu, QMessageBox, QVBoxLayout

from eidory.core.search_filters import FilterGroup, FilterPlan, SearchFilter, SCORED_FILTER_KINDS
from eidory.core.filter_engine import refresh_annotation_matches, rethreshold_result, same_filter_criteria
from eidory.ui.filter_panel import FilterPanel
from eidory.ui.theme_colors import set_theme_rich_text
from eidory.ui.justified_image_grid import JustifiedImageGridView


def _weak_callback(method):
    reference = WeakMethod(method)

    def call(*args, **kwargs):
        target = reference()
        return target(*args, **kwargs) if target is not None else None

    return call


class FilterWorkflow(QObject):
    """Adapts the staged rule editor to the existing gallery/project lifecycle."""

    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self.include_groups = ()
        self.exclude_groups = ()
        self.scope = {"kind": "sidebar"}
        self.history = []
        self.restoring = False
        self.panel = FilterPanel(
            label_for=_weak_callback(window._filter_label),
            pick_values=_weak_callback(self.pick_values),
        )
        self.panel.applyRequested.connect(self.apply)
        self.panel.scopeRequested.connect(self.choose_scope)
        self.panel.undoRequested.connect(self.undo)
        self.panel.inspectExcludedRequested.connect(self.inspect_excluded)
        self.panel.strengthChanged.connect(self.adjust_strength)
        self.panel.strengthFinished.connect(self.finish_strength)
        self.panel.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.panel.result_label.linkActivated.connect(self.open_index_panel)
        self._pending_strength = None
        self._strength_revision = None
        self._strength_history_saved = False
        self._strength_timer = QTimer(self)
        self._strength_timer.setSingleShot(True)
        self._strength_timer.setInterval(40)
        self._strength_timer.timeout.connect(self.flush_strength)
        self._strength_idle_timer = QTimer(self)
        self._strength_idle_timer.setSingleShot(True)
        self._strength_idle_timer.setInterval(350)
        self._strength_idle_timer.timeout.connect(self._finish_idle_strength)
        self._annotation_dirty = False
        self._annotation_timer = QTimer(self)
        self._annotation_timer.setSingleShot(True)
        self._annotation_timer.setInterval(80)
        self._annotation_timer.timeout.connect(self._refresh_annotation_results)

    def annotation_membership_changed(self, _image_id):
        plan = self.plan()
        if any(f.kind == "annotation" for f in (*plan.positive_filters, *plan.negative_filters)):
            self._annotation_dirty = True
            self._annotation_timer.start()

    def refresh_annotations_in_result(self, result):
        if not self._annotation_dirty:
            return result
        plan = result.evaluated_plan
        if plan is not None and result.ranked_scope_images is not None:
            if any(f.kind == "annotation" for f in (*plan.positive_filters, *plan.negative_filters)):
                result = refresh_annotation_matches(result, self.window.store.image_ids_with_preview_annotations())
            self._annotation_dirty = False
        return result

    def _refresh_annotation_results(self):
        w = self.window
        if w is None or w._closing or not self._annotation_dirty:
            return
        plan = self.plan()
        if not any(f.kind == "annotation" for f in (*plan.positive_filters, *plan.negative_filters)):
            self._annotation_dirty = False
            return
        if not w.search_button.isEnabled() or self.panel.error:
            # The pending result is refreshed before being displayed, including
            # annotation changes made while an expensive search was running.
            return
        result = w.current_chain_result
        if (w.current_result_mode == "search_chain" and result.evaluated_plan == plan
                and result.ranked_scope_images is not None):
            w._handle_search_chain_done(filters=plan.positive_filters, result=result)
        else:
            w._execute_search_chain(operation_mode="recompute")

    @property
    def window(self):
        return self._window()

    def plan(self) -> FilterPlan:
        w = self.window
        if tuple(f for g in self.include_groups for f in g.filters) != tuple(w.search_filters):
            self.include_groups = (FilterGroup(tuple(w.search_filters)),) if w.search_filters else ()
        if tuple(f for g in self.exclude_groups for f in g.filters) != tuple(w.result_exclusion_filters):
            # Old independent exclusions are migrated as OR, never silently AND.
            self.exclude_groups = (FilterGroup(tuple(w.result_exclusion_filters), "any"),) if w.result_exclusion_filters else ()
        return FilterPlan(self.include_groups, self.exclude_groups)

    def set_plan(self, plan):
        self.cancel_strength()
        w = self.window
        self.include_groups, self.exclude_groups = plan.include, plan.exclude
        w.search_filters = list(plan.positive_filters)
        w.result_exclusion_filters = list(plan.negative_filters)
        w.result_exclusion_filter_matches.clear()
        w.active_filter_index = w._last_score_filter_index()
        w._sync_legacy_search_state_from_filters()
        self.panel.set_plan(plan, force=True)

    def _can_adjust_strength(self, plan):
        w = self.window
        result = w.current_chain_result
        self.panel.result_label.setProperty("eidoryThemeRichText", None)
        return (w.current_result_mode == "search_chain" and w.search_button.isEnabled()
                and not self.panel.error and not self.panel.clear_context_on_apply
                and result.ranked_scope_images is not None and result.evaluated_plan == self.plan()
                and same_filter_criteria(result.evaluated_plan, plan))

    def adjust_strength(self, plan):
        if not self._can_adjust_strength(plan):
            return
        if not self._strength_history_saved:
            self.remember()
            self._strength_history_saved = True
        self._pending_strength = plan
        self._strength_revision = self.window.semantic_search_revision
        # Throttle, rather than debounce: the wall changes while the mouse is
        # still moving. The release always flushes the final value immediately.
        if not self._strength_timer.isActive():
            self._strength_timer.start()
        self._strength_idle_timer.start()

    def flush_strength(self):
        self._strength_timer.stop()
        plan, self._pending_strength = self._pending_strength, None
        w = self.window
        if (plan is None or self._strength_revision != w.semantic_search_revision
                or not self._can_adjust_strength(plan)):
            return
        result = rethreshold_result(w.current_chain_result, plan)
        self.include_groups, self.exclude_groups = plan.include, plan.exclude
        w.search_filters = list(plan.positive_filters)
        w.result_exclusion_filters = list(plan.negative_filters)
        w.active_filter_index = w._last_score_filter_index()
        self.panel.accept_live_plan(plan)
        w._handle_search_chain_done(filters=plan.positive_filters, result=result)

    def _finish_idle_strength(self):
        if any(slider.isSliderDown() for slider in self.panel.strength_sliders.values()):
            return
        self.finish_strength()

    def finish_strength(self):
        self.flush_strength()
        self._strength_idle_timer.stop()
        self._strength_history_saved = False

    def cancel_strength(self):
        self._strength_timer.stop()
        self._strength_idle_timer.stop()
        self._pending_strength = None
        self._strength_history_saved = False

    def remember(self):
        if not self.restoring:
            self.history.append(self.window._current_view_payload())
            self.history = self.history[-30:]
            self.panel.undo_button.setEnabled(True)

    def capture_external_scope(self):
        w = self.window
        if self.scope.get("kind") != "sidebar":
            return
        if w.current_result_mode in {"temp_project", "creative_node", "inspiration", "duplicate_group"}:
            images = {
                "temp_project": w.current_temp_project_images,
                "creative_node": w.current_creative_node_images,
                "inspiration": w.current_inspiration_images,
                "duplicate_group": w.current_duplicate_images,
            }[w.current_result_mode]
            label = f"当前结果（固定 {len(images)} 张）"
            if w.current_result_mode == "temp_project":
                project = w.store.get_temporary_project(w.current_temp_project_id)
                if project is not None:
                    label = f"基于{w._temporary_project_label(project)}：{project.name}"
            elif w.current_result_mode == "creative_node":
                node = w.store.get_creative_node(w.current_creative_node_id)
                if node is not None:
                    label = f"基于创作节点：{node.title}"
            self.scope = {"kind": "snapshot", "image_ids": [im.id for im in images], "label": label}

    def apply(self, plan: FilterPlan, *, remember=True):
        w = self.window
        if remember:
            self.remember()
        self.capture_external_scope()
        w.current_result_mode = "search_chain"
        if self.panel.clear_context_on_apply:
            w._clear_result_management_state()
            previous = w.status_filter_combo.blockSignals(True)
            w._set_combo_to_data(w.status_filter_combo, "all")
            w.status_filter_combo.blockSignals(previous)
            w._save_status_filter()
        self.set_plan(plan)
        w._show_gallery_view()
        w._clear_manual_result_order()
        w._execute_search_chain(operation_mode="recompute")

    def append(self, condition: SearchFilter, *, exclude=False):
        plan = self.plan()
        if condition.kind in SCORED_FILTER_KINDS and condition.score_threshold is None:
            condition = replace(condition, score_threshold=50)
        if exclude:
            # A direct context-menu action is one independent exclusion batch.
            plan = replace(plan, exclude=(*plan.exclude, FilterGroup((condition,))))
        else:
            groups = list(plan.include) or [FilterGroup()]
            groups = [replace(group, filters=(*group.filters, condition)) for group in groups]
            plan = replace(plan, include=tuple(groups))
        self.apply(plan)

    def base_context(self):
        kind = self.scope.get("kind")
        if kind == "snapshot":
            return set(self.scope.get("image_ids", [])), str(self.scope.get("label", "固定结果范围"))
        if kind == "collections":
            ids = self.window._image_ids_for_collections(self.scope.get("collection_ids", []))
            return ids, self.scope_label()
        return None

    def follow_sidebar(self):
        self.cancel_strength()
        self.scope = {"kind": "sidebar"}
        if self.window.current_result_mode in {"temp_project", "creative_node", "inspiration", "duplicate_group"}:
            self.window.current_result_mode = "search_chain"
        # A sidebar change supersedes a running query on the previous scope.
        self.window.semantic_search_revision += 1
        self.window.search_button.setEnabled(True)
        self.window.current_chain_base_image_ids = None
        self.window.current_chain_base_label = None

    def scope_label(self):
        w = self.window
        if self.scope.get("kind") == "snapshot":
            return str(self.scope.get("label", "固定结果范围"))
        if self.scope.get("kind") == "collections":
            names = [w._collection_path_text(cid) for cid in self.scope.get("collection_ids", [])]
            return " / ".join(names) or "已移除的文件夹（空范围）"
        if w.current_chain_base_image_ids is not None and w.current_result_mode == "search_chain":
            return w.current_chain_base_label or f"固定范围 {len(w.current_chain_base_image_ids)} 张"
        prefix = w._selected_folder_path_prefix()
        if prefix:
            return "磁盘来源：" + prefix
        virtual = w._selected_virtual_filter()
        if virtual:
            return w._virtual_filter_label(virtual)
        cid = w._selected_collection_id()
        return w._collection_path_text(cid) if cid is not None else "全部图库"

    def serialized_scope(self):
        scope = dict(self.scope)
        cid = self.window._selected_collection_id()
        if scope.get("kind") == "sidebar" and cid is not None and not self.window._selected_folder_path_prefix() and not self.window._selected_virtual_filter():
            # Keep a deleted preset folder empty instead of falling back to all.
            return {"kind": "collections", "collection_ids": [cid]}
        return scope

    def sync(self):
        w = self.window
        if hasattr(w, "creative_workspace"):
            w.creative_workspace.probe_tuning.setVisible(w.current_result_mode == "inspiration" and w.center_result_stack.currentWidget() is w.grid_view)
        if w.current_result_mode in {"temp_project", "creative_node", "inspiration", "duplicate_group"}:
            self.scope = {"kind": "sidebar"}
            self.capture_external_scope()
        plan = self.plan()
        self.panel.set_plan(plan)
        label = self.scope_label()
        self.panel.scope_button.setText(label + " ▾")
        self.panel.scope_button.setToolTip(label)
        self.panel.set_busy(not w.search_button.isEnabled())
        self.panel.set_live_available(self._can_adjust_strength(plan))
        unfinished = self.panel._busy or bool(self.panel.error)
        w.sort_combo.setEnabled(not unfinished)
        w.sort_order_combo.setEnabled(not unfinished)
        if unfinished:
            w.save_result_set_button.setEnabled(False)
        actions = [(label, callback) for label, callback in w._context_filter_actions() if not label.startswith(("文件夹：", "聚类："))]
        actions.extend(w._result_management_filter_actions())
        omitted = len(w._search_excluded_collection_ids()) + len(w._search_excluded_folder_prefixes())
        if omitted:
            actions.append((f"有 {omitted} 个文件夹停用搜索", self.show_disabled_folders))
        self.panel.set_context_actions(actions)
        count = len(w.grid_view.images())
        excluded = w._result_exclusion_filter_image_ids() | w.result_excluded_image_ids | w.result_excluded_collection_image_ids
        available_excluded = [im for im in w.current_chain_images if im.id in excluded]
        self.panel.inspect_button.setVisible(bool(available_excluded))
        self.panel.inspect_button.setText(f"查看排除 · {len(available_excluded)}")
        result = w.current_chain_result
        if w.current_result_mode == "search_chain" and result.filters_applied:
            details = f"范围 {result.scope_count} 张  ·  保留命中 {len(result.images)} 张  ·  排除 {len(available_excluded)} 张  ·  显示 {count} 张"
            self.panel.result_label.setText(f"显示 {count} 张 · 排除 {len(available_excluded)} · 范围 {result.scope_count}")
            if self.panel._busy or self.panel.error or result.evaluated_plan != plan:
                self.panel.result_label.setText("上次结果 · " + self.panel.result_label.text())
            coverage = []
            kinds = {f.kind for f in (*plan.positive_filters, *plan.negative_filters)}
            for kind, name in (("semantic", "语义"), ("similar", "相似图")):
                indexed = getattr(result, kind + "_searchable_count")
                if kind in kinds and indexed < result.scope_count:
                    coverage.append(f"{name}索引覆盖 {indexed}/{result.scope_count} 张")
            if coverage:
                details += "\n" + "；".join(coverage) + "，未索引图片不命中这些条件"
                set_theme_rich_text(self.panel.result_label, self.panel.result_label.text() + ' · <a href="index" style="color:@link@">有图片尚未索引，查看处理</a>')
            self.panel.result_label.setToolTip(details)
        else:
            self.panel.result_label.setText(f"图片墙 · 已加载 {count} 张")
            self.panel.result_label.setToolTip(self.panel.result_label.text())

        if hasattr(w, "simple_search"):
            w.simple_search.refresh()

    def choose_scope(self):
        self.finish_strength()
        menu = QMenu(self.panel)
        menu.addAction("全部图库", self.select_all_scope)
        menu.addAction("选择文件夹…", self.select_collection_scope)
        action = menu.addAction("将当前结果固定为基础范围", self.freeze_scope)
        action.setEnabled(bool(self.window.grid_view.images()))
        menu.exec(self.panel.scope_button.mapToGlobal(self.panel.scope_button.rect().bottomLeft()))

    def open_index_panel(self, _link="index"):
        self.window.right_tab_widget.setCurrentIndex(3)
        self.window.statusBar().showMessage(self.panel.result_label.toolTip() + "；可启动索引或重试失败项目")

    def select_all_scope(self):
        w = self.window
        self.remember()
        # Preserve the rule draft while changing its explicitly displayed scope.
        draft = self.panel.draft
        w.collection_tree.blockSignals(True)
        w.collection_tree.setCurrentItem(w.collection_tree.topLevelItem(0))
        w.collection_tree.blockSignals(False)
        w.folder_tree.blockSignals(True)
        w.folder_tree.setCurrentItem(w.folder_tree.topLevelItem(0))
        w.folder_tree.blockSignals(False)
        w.current_virtual_filter = None
        self.follow_sidebar()
        self.apply(draft, remember=False)

    def reset_library_scope(self):
        """Leave frozen/project scopes without issuing an intermediate query."""
        w = self.window
        for tree in (w.collection_tree, w.folder_tree):
            previous = tree.blockSignals(True)
            tree.setCurrentItem(tree.topLevelItem(0))
            tree.blockSignals(previous)
        w.current_virtual_filter = None
        w._clear_tag_virtual_filter_selection()
        w._clear_ai_vision_virtual_filter_selection()
        previous = w.status_filter_combo.blockSignals(True)
        w._set_combo_to_data(w.status_filter_combo, "all")
        w.status_filter_combo.blockSignals(previous)
        w._save_status_filter()
        self.follow_sidebar()

    def select_collection_scope(self):
        w = self.window
        ids = w._select_collections_for_search_dialog(reverse_mode=False, context_image=w._selected_grid_image())
        if not ids:
            return
        self.remember()
        draft = self.panel.draft
        self.scope = {"kind": "collections", "collection_ids": list(ids)}
        w.collection_tree.blockSignals(True)
        w.collection_tree.clearSelection()
        w.collection_tree.setCurrentItem(None)
        if len(ids) == 1:
            def select(item):
                if item.data(0, Qt.ItemDataRole.UserRole) == ids[0]:
                    w.collection_tree.setCurrentItem(item)
                for i in range(item.childCount()):
                    select(item.child(i))
            for i in range(w.collection_tree.topLevelItemCount()):
                select(w.collection_tree.topLevelItem(i))
        w.collection_tree.blockSignals(False)
        w.current_virtual_filter = None
        previous = w.folder_tree.blockSignals(True)
        w.folder_tree.setCurrentItem(w.folder_tree.topLevelItem(0))
        w.folder_tree.blockSignals(previous)
        self.apply(draft, remember=False)

    def freeze_scope(self):
        self.remember()
        images = self.window.grid_view.images()
        self.scope = {"kind": "snapshot", "image_ids": [im.id for im in images], "label": f"固定结果集 · {len(images)} 张"}
        self.window._clear_result_management_state()
        self.apply(FilterPlan(), remember=False)

    def undo(self):
        if not self.history:
            return
        payload = self.history.pop()
        self.restoring = True
        try:
            self.window._apply_view_payload(payload)
        finally:
            self.restoring = False
        self.panel.undo_button.setEnabled(bool(self.history))

    def pick_values(self, kind, current):
        w = self.window
        if kind == "collection":
            ids = w._select_collections_for_search_dialog(reverse_mode=False, context_image=w._selected_grid_image())
            return SearchFilter(kind, w._collection_filter_value(ids)) if ids else None
        if kind == "tag":
            result = w._select_tags_for_search_dialog(reverse_mode=False)
            return SearchFilter(kind, w._tag_filter_value(*result)) if result and result[0] else None
        if kind == "similar":
            image = w._selected_grid_image()
            reason = w._similar_image_blocking_reason(image, check_vector=True)
            if reason:
                QMessageBox.information(self.panel, "相似图条件", reason)
                return None
            return SearchFilter("similar", image.id, 50)
        return None

    def inspect_excluded(self):
        w = self.window
        excluded = w._result_exclusion_filter_image_ids() | w.result_excluded_image_ids | w.result_excluded_collection_image_ids
        images = [im for im in w.current_chain_images if im.id in excluded]
        dialog = QDialog(w)
        dialog.setWindowTitle(f"筛选排除项 · {len(images)} 张")
        dialog.resize(820, 600)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("这些图片仅从当前结果中隐藏，文件和图库记录保留。"))
        grid = JustifiedImageGridView(thumbnail_size=140)
        grid.set_images(images)
        layout.addWidget(grid, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def show_disabled_folders(self):
        w = self.window
        names = [w._collection_path_text(cid) for cid in w._search_excluded_collection_ids()]
        names.extend(w._search_excluded_folder_prefixes())
        QMessageBox.information(self.panel, "停用搜索的文件夹", "\n".join(names) + "\n\n可在左侧对应文件夹的右键菜单中重新启用搜索。")
