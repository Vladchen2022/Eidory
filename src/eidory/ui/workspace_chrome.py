"""Workspace presentation. Existing controllers keep ownership of user data."""
from __future__ import annotations

from weakref import ref

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QHBoxLayout, QLabel, QMenu, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from eidory.ui.theme import apply_theme
from eidory.ui.theme_colors import THEMES, get_theme


class WorkspaceChrome(QObject):
    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self._last_project = window.current_creative_project_id
        self._collection_history = []
        self._dialogs = {}
        self._last_inspector_tab = 0
        self._build_navigation()
        self._build_header()
        self._build_selection_bar()
        self._build_board_tools()
        self._build_global_panels()
        self._refine_planning()
        window.root_splitter.splitterMoved.connect(self.sync_sidebars)
        window.center_result_stack.currentChanged.connect(self.refresh)
        window.grid_view.selectionChanged.connect(self.refresh_selection)
        window.project_board_view.selectionChanged.connect(self.refresh_selection)
        self.retranslate()
        # Reparented controls may already have a cached stylesheet before roles
        # are assigned. Polish them once so first launch matches density changes.
        from PySide6.QtWidgets import QStyle
        self._polished_styles = []
        for widget in window.findChildren(QWidget):
            if widget.property("variant") or widget.property("role") or widget is window.creative_workspace.proposal_panel:
                style = widget.style()
                if isinstance(style, QStyle):
                    self._polished_styles.append(style)
                    style.unpolish(widget)
                    style.polish(widget)
        self.refresh()
        QTimer.singleShot(0, self.sync_sidebars)

    @property
    def w(self):
        return self._window()

    def _button(self, text, callback, layout, *, primary=False):
        button = QPushButton(text)
        button.clicked.connect(callback)
        if primary:
            button.setProperty("variant", "primary")
        layout.addWidget(button)
        return button

    def _build_navigation(self):
        w = self.w
        cw = w.creative_workspace
        panel = w.root_splitter.widget(0)
        panel.setObjectName("navigationSidebar")
        layout = panel.layout()
        # Keep legacy adapters alive; their existing signals drive the menus.
        self.adapters = QWidget(w)
        self.adapters.hide()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(self.adapters)
        self.navigation_tabs = QTabWidget()
        self.navigation_tabs.setObjectName("navigationTabs")
        layout.addWidget(self.navigation_tabs)
        library = QWidget()
        lib = QVBoxLayout(library)
        lib.setContentsMargins(0, 8, 0, 0)
        actions = QHBoxLayout()
        lib.addLayout(actions)
        w.add_collection_button.setText("＋ 文件夹")
        actions.addWidget(w.add_collection_button)
        w.add_collection_button.show()
        imports = QPushButton("导入 ▾")
        menu = QMenu(imports)
        menu.addAction("导入图片…", w.add_folder_button.click)
        menu.addAction("导入文件夹…", w.import_folder_tree_button.click)
        imports.setMenu(menu)
        actions.addWidget(imports)
        lib.addWidget(w.collection_tree, 1)
        w.collection_tree.show()
        self.navigation_tabs.addTab(library, "图库")

        projects = QWidget()
        project_layout = QVBoxLayout(projects)
        project_layout.setContentsMargins(0, 8, 0, 0)
        self.new_project_button = self._button("＋ 新建创作项目", cw.new_project, project_layout)
        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        current = QWidget()
        current_layout = QVBoxLayout(current)
        current_layout.setContentsMargins(0, 0, 0, 0)
        self.node_heading = QLabel("当前项目节点")
        self.node_heading.setProperty("role", "sectionTitle")
        current_layout.addWidget(self.node_heading)
        w.creative_node_tree.setMinimumHeight(120)
        w.creative_node_tree.setMaximumHeight(16777215)
        current_layout.addWidget(w.creative_node_tree, 1)
        w.creative_node_tree.show()
        w.creative_node_status_label.setWordWrap(True)
        w.creative_node_status_label.setMaximumHeight(72)
        w.creative_node_status_label.setProperty("role", "muted")
        current_layout.addWidget(w.creative_node_status_label)
        w.creative_node_status_label.show()
        self.project_node_panel = current
        split.addWidget(current)
        saved = QWidget()
        saved_layout = QVBoxLayout(saved)
        saved_layout.setContentsMargins(0, 8, 0, 0)
        heading = QLabel("项目与参考集")
        heading.setProperty("role", "sectionTitle")
        saved_layout.addWidget(heading)
        w.temp_project_list.setMinimumHeight(130)
        saved_layout.addWidget(w.temp_project_list, 1)
        w.temp_project_list.show()
        split.addWidget(saved)
        split.setSizes([400, 220])
        project_layout.addWidget(split, 1)
        self.explore_button = self._button("探索灵感方向", cw.show_probe, project_layout)
        self.navigation_tabs.addTab(projects, "项目")
        self.navigation_tabs.setCurrentIndex(w._setting_int("ui.navigation_tab", 0, 0, 1))
        self.navigation_tabs.currentChanged.connect(lambda i: w.store.set_setting("ui.navigation_tab", str(i)))

        # The right side now concerns the current reference, not project navigation.
        side = cw.sidebar.layout()
        for i in range(side.count() - 1, -1, -1):
            widget = side.itemAt(i).widget()
            if widget and widget is not cw.reference_input:
                side.removeWidget(widget)
                widget.setParent(self.adapters)
                widget.hide()
        self.reference_heading = QLabel("当前节点的参考用途")
        self.reference_heading.setProperty("role", "sectionTitle")
        side.insertWidget(0, self.reference_heading)
        self.reference_membership = QLabel("选择一张图片，查看它在项目中的用途。")
        self.reference_membership.setWordWrap(True)
        side.insertWidget(1, self.reference_membership)
        cw.reference_input.setMinimumHeight(140)
        cw.reference_input.setMaximumHeight(280)
        side.addStretch(1)
        self.export_button = self._button("导出创作参考包…", cw.export_project, side)
        w.right_tab_widget.setTabText(1, "用途")

    def _build_header(self):
        w = self.w
        self.header = QWidget()
        self.header.setObjectName("workspaceHeader")
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(self.header)
        row.setContentsMargins(12, 6, 12, 6)
        row.setSpacing(8)
        brand = QLabel("Eidory")
        brand.setObjectName("workspaceBrand")
        row.addWidget(brand)
        self.left_button = self._button("导航", lambda: self.toggle_sidebar(0), row)
        self.left_button.setCheckable(True)
        self.exit_focus_button = self._button("退出专注", lambda: w._set_board_focus_mode(False), row, primary=True)
        self.exit_focus_button.hide()
        w.creative_project_combo.setMaximumWidth(330)
        w.creative_project_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        w.creative_project_combo.setMinimumContentsLength(12)
        row.addWidget(w.creative_project_combo)
        w.creative_project_combo.show()
        row.addStretch(1)
        self.tasks_button = self._button("后台任务", lambda: self.open_global_panel(3), row)
        self.settings_button = self._button("设置", lambda: self.open_global_panel(4), row)
        self.right_button = self._button("信息", lambda: self.toggle_sidebar(2), row)
        self.right_button.setCheckable(True)
        w.central_layout.insertWidget(0, self.header)
        w.central_layout.setStretch(w.central_layout.indexOf(w.root_splitter), 1)
        w.root_splitter.widget(2).setObjectName("referenceSidebar")
        w.root_splitter.setHandleWidth(6)

    def _build_selection_bar(self):
        w = self.w
        cw = w.creative_workspace
        self.selection_bar = QWidget()
        self.selection_bar.setObjectName("selectionActions")
        self.selection_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(self.selection_bar)
        row.setContentsMargins(0, 4, 0, 4)
        self.selection_label = QLabel("选择图片后加入当前节点")
        self.selection_label.setMinimumWidth(0)
        self.selection_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.selection_label.setProperty("role", "muted")
        row.addWidget(self.selection_label, 1)
        row.addWidget(cw.group_button)
        cw.group_button.show()
        self.undo_collect_button = self._button("撤销加入", self.undo_collection, row)
        row.addWidget(cw.collect_button)
        cw.collect_button.setProperty("variant", "primary")
        cw.collect_button.show()
        layout = w.center_result_stack.parentWidget().layout()
        layout.insertWidget(layout.indexOf(w.center_result_stack) + 1, self.selection_bar)
        w.show_gallery_button.hide()
        w.show_project_board_button.hide()
        # One navigation strip owns the four stages.
        self.stage_buttons = [cw.navigation.layout().itemAt(i).widget() for i in range(4)]
        for button, callback in zip(self.stage_buttons, (cw.show_project_planning, cw.show_candidates, cw.show_collected, cw.show_board)):
            button.clicked.disconnect()
            button.setCheckable(True)
            button.toggled.connect(lambda checked, action=callback: action() if checked else self.refresh())
            button.setProperty("variant", "navigation")
        cw.context_label.setProperty("role", "muted")

    def _build_board_tools(self):
        from eidory.ui.overflow_toolbar import OverflowToolbar

        w = self.w
        cw = w.creative_workspace
        self.board_tools = OverflowToolbar()
        self.board_tools.setObjectName("boardTools")
        for button in (w.board_fit_all_button, *cw.board_buttons):
            self.board_tools.add_button(button)
        self.board_section_lock_button = QPushButton("锁定／解锁所选分区")
        self.board_section_lock_button.clicked.connect(w.project_board_view.toggle_section_lock)
        for button in (
            w.board_pin_button, w.board_hide_selected_button,
            w.board_flip_button, w.board_grayscale_button,
            w.board_edit_local_display_button, w.board_toggle_local_display_button,
            self.board_section_lock_button,
        ):
            self.board_tools.add_button(button)
        w.center_result_stack.parentWidget().layout().insertWidget(1, self.board_tools)

    def _build_global_panels(self):
        w = self.w
        tabs = w.right_tab_widget
        # Preserve stable tab indexes used by the existing adapters.
        for index, title in ((4, "设置"), (3, "后台任务与索引")):
            page = tabs.widget(index)
            tabs.removeTab(index)
            tabs.insertTab(index, QWidget(), title)
            tabs.setTabVisible(index, False)
            dialog = QDialog(w)
            dialog.setWindowTitle(title)
            dialog.resize(760, 780)
            layout = QVBoxLayout(dialog)
            if index == 4:
                appearance = QHBoxLayout()
                appearance.addWidget(QLabel("主题配色"))
                self.theme_combo = QComboBox()
                self.theme_combo.setAccessibleName("主题配色")
                for theme in THEMES:
                    self.theme_combo.addItem(theme.name, theme.id)
                self.theme_combo.setCurrentIndex(self.theme_combo.findData(get_theme(w.store.get_setting("ui.theme")).id))
                self.theme_combo.currentIndexChanged.connect(self.change_theme)
                appearance.addWidget(self.theme_combo)
                hint = QLabel("立即生效，自动保存")
                hint.setProperty("role", "muted")
                appearance.addWidget(hint)
                appearance.addStretch(1)
                layout.addLayout(appearance)
                display = QHBoxLayout()
                display.addWidget(QLabel("界面密度"))
                self.density_combo = QComboBox()
                self.density_combo.addItem("紧凑", "compact")
                self.density_combo.addItem("舒展", "comfortable")
                self.density_combo.setCurrentIndex(1 if w.store.get_setting("ui.density") == "comfortable" else 0)
                self.density_combo.currentIndexChanged.connect(self.change_density)
                display.addWidget(self.density_combo)
                display.addWidget(QLabel("文字大小"))
                self.font_size = QSpinBox()
                self.font_size.setAccessibleName("界面文字大小")
                self.font_size.setRange(11, 20)
                self.font_size.setSuffix(" pt")
                self.font_size.setValue(w._setting_int("ui.font_size", 13, minimum=11, maximum=20))
                self.font_size.valueChanged.connect(self.change_font_size)
                display.addWidget(self.font_size)
                display.addStretch(1)
                self.restore_layout_button = self._button("恢复默认布局", w._restore_workspace_layout, display)
                self.restore_layout_button.setToolTip("展开左右侧栏；保留当前项目、筛选、选中图片与看板内容")
                layout.addLayout(display)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            page.setMinimumWidth(0)
            scroll.setWidget(page)
            layout.addWidget(scroll)
            self._button("返回工作台", dialog.hide, layout)
            self._dialogs[index] = dialog
        tabs.tabBar().setMinimumWidth(tabs.tabBar().minimum_width_for_tab_count(3))
        if tabs.currentIndex() >= 3:
            tabs.setCurrentIndex(0)
        tabs.currentChanged.connect(self._inspector_changed)
        for results in (w.ai_vision_virtual_filter_list, w.ai_character_vision_virtual_filter_list):
            results.itemClicked.connect(self._show_task_results)
            results.itemActivated.connect(self._show_task_results)

    def _show_task_results(self, *_args):
        self._dialogs[3].hide()
        self.w._show_gallery_view()

    def retranslate(self):
        english = self.w.current_language == "en"
        self.w.right_tab_widget.setTabText(1, "Usage" if english else "用途")
        self.w.add_collection_button.setText("＋ Folder" if english else "＋ 文件夹")
        self.w.search_creative_node_button.setText("Find References" if english else "用此语句找参考")
        for button, chinese, translated in (
            (self.w.board_pin_button, "窗口置顶", "Keep on Top"),
            (self.w.board_hide_selected_button, "专注看板", "Focus Board"),
            (self.w.board_flip_button, "左右翻转", "Flip Horizontally"),
            (self.w.board_grayscale_button, "黑白显示", "Grayscale"),
            (self.w.board_edit_local_display_button, "编辑局部显示", "Edit Local View"),
            (self.w.board_toggle_local_display_button, "局部／全图切换", "Local / Full"),
            (self.board_section_lock_button, "锁定／解锁所选分区", "Lock / Unlock Sections"),
        ):
            # Former icon-only controls now have readable labels. Let the font
            # and theme determine their size, including after language changes.
            button.setMinimumSize(0, 0)
            button.setMaximumSize(16777215, 16777215)
            button.setProperty("eidoryBoardIcon", None)
            button.setIcon(QIcon())
            button.setText(translated if english else chinese)
            button.setAccessibleName(button.text())
        shortcut = self.w.board_focus_shortcut.key().toString(QKeySequence.SequenceFormat.NativeText)
        self.w.board_hide_selected_button.setToolTip(
            ("Focus board" if english else "专注看板") + f"（{shortcut}）")
        self.board_tools.more.setText("More ▾" if english else "更多 ▾")
        self.w.board_grayscale_button.setToolTip("有选图时只切换所选图片；未选图时统一切换全部图片的黑白显示")
        if hasattr(self.w, 'board_shortcuts'):
            self.w.board_shortcuts.refresh_labels()
        self.board_tools.timer.start(0)

    def _inspector_changed(self, index):
        if index in (3, 4):
            self.open_global_panel(index)
            self.w.right_tab_widget.setCurrentIndex(self._last_inspector_tab)
        else:
            self._last_inspector_tab = index

    def open_global_panel(self, index):
        dialog = self._dialogs.get(index)
        if dialog:
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()

    def change_font_size(self, value):
        self.w.store.set_setting("ui.font_size", str(value))
        apply_theme(QApplication.instance(), self.theme_combo.currentData(), density=self.density_combo.currentData(), font_size=value)

    def change_density(self):
        density = self.density_combo.currentData()
        self.w.store.set_setting("ui.density", density)
        apply_theme(QApplication.instance(), self.theme_combo.currentData(), density=density)

    def change_theme(self):
        theme_id = self.theme_combo.currentData()
        self.w.store.set_setting("ui.theme", theme_id)
        apply_theme(QApplication.instance(), theme_id, density=self.density_combo.currentData())

    def _refine_planning(self):
        w = self.w
        cw = w.creative_workspace
        cw.planning.setObjectName("planningWorkspace")
        cw.planning.layout().setContentsMargins(20, 14, 20, 18)
        cw.planning.layout().setSpacing(12)
        cw.proposal_panel.setObjectName("aiProposalPanel")
        for button in cw.proposal_panel.findChildren(QPushButton):
            if button.text().startswith("采纳"):
                button.setProperty("variant", "primary")
        w.generate_creative_children_button.setProperty("variant", "primary")
        w.search_creative_node_button.setText("用此语句找参考")
        w.save_selection_to_creative_node_button.hide()
        w.inspiration_brief_input.setFixedHeight(70)
        w.inspiration_answers_input.setFixedHeight(80)
        # These actions were stacked for a narrow inspector. In the workspace,
        # keep their natural width and the project template on one compact row.
        project_layout = w.ai_workflow_stack.widget(0).layout()
        project_actions = QWidget()
        actions = QHBoxLayout(project_actions)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addWidget(w.creative_template_label)
        w.creative_template_combo.setMaximumWidth(240)
        actions.addWidget(w.creative_template_combo)
        actions.addWidget(w.creative_new_project_button)
        actions.addWidget(w.creative_ai_project_button)
        actions.addStretch(1)
        for i in range(project_layout.count() - 1, -1, -1):
            item = project_layout.itemAt(i)
            if item.spacerItem() or (item.layout() and not item.layout().count()):
                project_layout.takeAt(i)
        project_layout.insertWidget(0, project_actions)
        project_layout.setStretch(project_layout.indexOf(w.creative_content_tabs), 0)
        project_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        node_layout = w.creative_node_note_input.parentWidget().layout()
        for i in range(node_layout.count()):
            layout = node_layout.itemAt(i).layout()
            if isinstance(layout, QHBoxLayout) and any(isinstance(layout.itemAt(j).widget(), QPushButton) for j in range(layout.count())):
                layout.addStretch(1)
        w.creative_node_note_input.setMinimumHeight(200)
        w.open_creative_copy_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

    def toggle_sidebar(self, index):
        w = self.w
        show = w._board_focus_mode or not w._sidebar_visible[index]
        w._set_sidebar_visible(index, show)

    def sync_sidebars(self, *_args):
        w = self.w
        english = w.current_language == "en"
        self.exit_focus_button.setVisible(w._board_focus_mode)
        self.exit_focus_button.setText("Exit Focus" if english else "退出专注")
        for index, button, label in ((0, self.left_button, "左"), (2, self.right_button, "右")):
            visible = not w._board_focus_mode and w._sidebar_visible[index]
            button.setChecked(visible)
            action = ("收起" if visible else "显示") + label + "侧栏"
            if english:
                action = ("Hide " if visible else "Show ") + ("Navigation" if index == 0 else "Inspector")
            button.setText(action)
            button.setAccessibleName(action)
            button.setToolTip(action + ("; sidebars reopen on launch" if english else "；仅影响本次使用，重启后自动展开"))

    def record_collection(self, node, image_ids):
        if image_ids:
            self._collection_history.append((node.project_id, node.id, node.title, list(image_ids)))
            self._collection_history = self._collection_history[-20:]
            self.refresh_selection()

    def undo_collection(self):
        w = self.w
        if not self._collection_history or self._collection_history[-1][0] != w.current_creative_project_id:
            return
        _, node_id, title, ids = self._collection_history.pop()
        node = w.store.get_creative_node(node_id)
        if node:
            w.store.remove_images_from_creative_node(node_id, ids)
            w._refresh_creative_projects(select_project_id=node.project_id)
            w._refresh_after_creative_node_images_changed(node)
            if w.current_result_mode == "creative_node":
                w._show_creative_node_saved_images(w.current_creative_node_id)
            w.statusBar().showMessage(f"已撤销加入『{title}』的 {len(ids)} 张图片")
        self.refresh_selection()

    def refresh_selection(self, *_args):
        w = self.w
        images = w._current_selection_images()
        node = w._current_creative_node()
        count = len(images)
        memberships = w.grid_view._project_references
        used = sum(image.id in memberships for image in images)
        text = f"已选 {count} 张" if count else "选择图片后加入当前节点"
        if used:
            text += f" · {used} 张已用于本项目"
        self.selection_label.setText(text)
        self.selection_label.setToolTip(text)
        if node:
            self.reference_heading.setText(f"『{node.title}』的参考用途")
        else:
            self.reference_heading.setText("参考用途")
        if len(images) == 1:
            names = memberships.get(images[0].id, [])
            self.reference_membership.setText(images[0].file_name + "\n\n" + ("本项目已选 · " + "、".join(names) if names else "尚未加入本项目节点"))
        else:
            self.reference_membership.setText("选择一张已收集图片，记录它提供的构图、材质或光线参考。")
        can_undo = bool(self._collection_history and self._collection_history[-1][0] == w.current_creative_project_id)
        self.undo_collect_button.setEnabled(can_undo)
        self.undo_collect_button.setVisible(can_undo)
        if can_undo:
            self.undo_collect_button.setToolTip("撤销最近加入『" + self._collection_history[-1][2] + "』的图片")

    def refresh(self, *_args):
        w = self.w
        cw = w.creative_workspace
        project_id = w.current_creative_project_id
        if project_id != self._last_project:
            self._last_project = project_id
            if project_id:
                self.navigation_tabs.setCurrentIndex(1)
        self.project_node_panel.setVisible(bool(project_id))
        if w.creative_project_combo.count() and w.creative_project_combo.itemData(0) is None:
            w.creative_project_combo.setItemText(0, "独立图库")
        self._sync_temporary_context()
        is_gallery = w.center_result_stack.currentWidget() is w.grid_view
        is_board = w.center_result_stack.currentWidget() is w.project_board_view
        phase = 3 if is_board else 0 if not is_gallery else 2 if w.current_result_mode == "creative_node" else 1
        for index, button in enumerate(self.stage_buttons):
            previous = button.blockSignals(True)
            button.setChecked(index == phase)
            button.blockSignals(previous)
        self.stage_buttons[2].setEnabled(bool(project_id))
        can_show_board = (
            w.current_search_result_set_project_id is None
            and (bool(project_id) or (
                w.current_temp_project_id is not None
                and w.current_result_mode == "temp_project"
                and w._current_temporary_project_kind() != "search"
            ))
        )
        self.stage_buttons[3].setEnabled(can_show_board)
        self.stage_buttons[3].setToolTip("" if can_show_board else "先选择创作项目或参考集，再使用看板")
        self.selection_bar.setVisible(is_gallery)
        self.board_tools.setVisible(is_board and not w._board_focus_mode)
        w.show_gallery_button.hide()
        w.show_project_board_button.hide()
        for widget in (w.shuffle_results_button, w.save_result_set_button, w.thumbnail_size_label, w.thumbnail_size_slider):
            widget.setVisible(is_gallery)
        self.export_button.setEnabled(bool(project_id))
        self.refresh_selection()
        self.sync_sidebars()
        if hasattr(w, 'creative_flow'):
            w.creative_flow.refresh()

    def _sync_temporary_context(self):
        # A reference set has its own board. Never label that context as the
        # independent library, whose explicit selection exits all boards.
        w = self.w
        combo = w.creative_project_combo
        project = None
        if w.current_creative_project_id is None and w.current_temp_project_id is not None:
            project = w.store.get_temporary_project(w.current_temp_project_id)
        key = f"temporary:{project.id}" if project else None
        previous = combo.blockSignals(True)
        try:
            for index in range(combo.count() - 1, 0, -1):
                data = combo.itemData(index)
                if isinstance(data, str) and data.startswith("temporary:") and data != key:
                    combo.removeItem(index)
            if project:
                index = combo.findData(key)
                if index < 0:
                    combo.addItem(f"{w._temporary_project_label(project)}：{project.name}", key)
                    index = combo.count() - 1
                combo.setCurrentIndex(index)
            elif w.current_creative_project_id is None:
                combo.setCurrentIndex(0)
        finally:
            combo.blockSignals(previous)
