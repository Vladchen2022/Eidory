"""Workspace presentation. Existing controllers keep ownership of user data."""
from __future__ import annotations

from weakref import ref

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QHBoxLayout, QLabel, QMenu, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from eidory.ui.theme import apply_theme
from eidory.ui.theme_colors import THEMES, get_theme, theme_color


def _sidebar_arrow_icon(point_left):
    pixmap = QPixmap(36, 36)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(theme_color("text"), 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    tip, shoulder = (4, 8) if point_left else (14, 10)
    painter.drawLine(4, 9, 14, 9)
    painter.drawLine(tip, 9, shoulder, 5)
    painter.drawLine(tip, 9, shoulder, 13)
    painter.end()
    return QIcon(pixmap)


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
        current = QWidget()
        current_layout = QVBoxLayout(current)
        current_layout.setContentsMargins(0, 0, 0, 0)
        self.node_heading = QLabel("创作内容")
        self.node_heading.setProperty("role", "sectionTitle")
        current_layout.addWidget(self.node_heading)
        w.creative_node_tree.setMinimumHeight(120)
        w.creative_node_tree.setMaximumHeight(16777215)
        current_layout.addWidget(w.creative_node_tree, 1)
        w.creative_node_tree.show()
        # The stage strip already names the active content and reference count.
        w.creative_node_status_label.hide()
        self.project_node_panel = current
        project_layout.addWidget(current, 1)
        self.project_empty = QLabel("尚未选择项目")
        self.project_empty.setProperty("role", "muted")
        self.project_empty.setAlignment(Qt.AlignmentFlag.AlignTop)
        project_layout.addWidget(self.project_empty, 1)
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
        self._sidebar_icon_theme = None
        self.left_button = self._button("", lambda: self.toggle_sidebar(0), row)
        self.left_button.setCheckable(True)
        self.left_button.setIconSize(QSize(18, 18))
        self.exit_focus_button = self._button("退出专注", lambda: w._set_board_focus_mode(False), row, primary=True)
        self.exit_focus_button.hide()
        # Keep the combo as the existing project-state adapter, not a second
        # user-facing selector beside the saved-project list.
        w.creative_project_combo.setParent(self.adapters)
        w.creative_project_combo.hide()
        self.project_button = self._button("独立图库 ▾", lambda: self.open_project_picker(), row)
        self.project_button.setAccessibleName("选择项目或独立图库")
        self.project_button.setMaximumWidth(330)
        self.project_board_button = self._button("项目看板", self.toggle_context_board, row)
        self.project_board_button.hide()
        self.project_picker = QDialog(w)
        self.project_picker.setWindowTitle("选择项目或参考集")
        self.project_picker.setWindowModality(Qt.WindowModality.WindowModal)
        picker_layout = QVBoxLayout(self.project_picker)
        picker_actions = QHBoxLayout()
        self.library_button = self._button("独立图库", self.open_library, picker_actions)
        self.new_project_button = self._button("＋ 新建项目", self.choose_project_type, picker_actions)
        self.import_project_button = self._button("导入项目包…", lambda: w.workflow_upgrade.import_package(), picker_actions)
        picker_layout.addLayout(picker_actions)
        picker_layout.addWidget(w.temp_project_list, 1)
        w.temp_project_list.setMinimumHeight(180)
        w.temp_project_list.show()
        w.temp_project_list.itemSelectionChanged.disconnect(w._load_selected_temporary_project)
        w.temp_project_list.itemSelectionChanged.connect(self._project_selection_changed)
        # Loading a project rebuilds (and deletes) its QListWidgetItem objects.
        # Use one itemClicked receiver: a second Python receiver would make
        # PySide wrap the already-freed signal argument and crash on macOS.
        w.temp_project_list.itemClicked.disconnect(w._handle_project_sidebar_item_clicked)
        w.temp_project_list.itemClicked.connect(self._activate_project)
        w.temp_project_list.itemActivated.connect(self._activate_project)
        w.temp_project_list.installEventFilter(self)
        self.library_button.setAutoDefault(False)
        self.new_project_button.setAutoDefault(False)
        self.import_project_button.setAutoDefault(False)
        self.project_type_picker = QDialog(self.project_picker)
        self.project_type_picker.setWindowTitle("选择项目类型")
        type_layout = QVBoxLayout(self.project_type_picker)
        type_layout.setSpacing(10)
        self.create_node_project_button = self._button("创作节点项目", self.new_project, type_layout)
        self.node_project_hint = QLabel("按节点构思、找参考，最后整理成看板。")
        self.node_project_hint.setWordWrap(True)
        type_layout.addWidget(self.node_project_hint)
        self.create_inspiration_project_button = self._button("语义探针项目", self.new_inspiration_project, type_layout)
        self.inspiration_project_hint = QLabel("用一句话快速探索多个灵感方向，再从图库找图。")
        self.inspiration_project_hint.setWordWrap(True)
        type_layout.addWidget(self.inspiration_project_hint)
        self.cancel_project_type_button = self._button("取消", self.project_type_picker.reject, type_layout)
        for button in (self.create_node_project_button, self.create_inspiration_project_button, self.cancel_project_type_button):
            button.setAutoDefault(False)
        row.addStretch(1)
        self.tasks_button = self._button("后台任务", lambda: self.open_global_panel(3), row)
        self.settings_button = self._button("设置", lambda: self.open_global_panel(4), row)
        self.right_button = self._button("", lambda: self.toggle_sidebar(2), row)
        self.right_button.setCheckable(True)
        self.right_button.setIconSize(QSize(18, 18))
        w.central_layout.insertWidget(0, self.header)
        w.central_layout.setStretch(w.central_layout.indexOf(w.root_splitter), 1)
        w.root_splitter.widget(2).setObjectName("referenceSidebar")
        w.root_splitter.setHandleWidth(6)

    def open_project_picker(self):
        screen = self.project_button.screen().availableGeometry()
        self.project_picker.resize(min(480, screen.width() - 16), min(470, screen.height() - 40))
        if not any(self.w.project_sidebar_expanded_sections.values()):
            self.w._set_project_sidebar_expanded_section("creative")
            self.w._refresh_project_sidebar()
        self.project_picker.show()
        self.project_picker.raise_()
        self.project_picker.activateWindow()
        self.w.temp_project_list.setFocus()

    def eventFilter(self, watched, event):
        if (watched is self.w.temp_project_list and self.project_picker.isVisible()
                and event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress)
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)):
            event.accept()
            if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                item = self.w.temp_project_list.currentItem()
                if item:
                    self._activate_project(item)
            return True
        return False

    def _project_selection_changed(self):
        # Arrow keys browse the chooser without loading every intermediate
        # project's board. Mouse click / Enter explicitly activate a row.
        if not self.project_picker.isVisible():
            self.w._load_selected_temporary_project()

    def _activate_project(self, item):
        # Switching workspaces can clear or rebuild the picker selection.
        is_project = item.data(Qt.ItemDataRole.UserRole) is not None
        self.w._handle_project_sidebar_item_clicked(item)
        if is_project:
            self.project_picker.hide()

    def open_library(self):
        self.project_picker.hide()
        self.w._load_creative_project(None)
        self.navigation_tabs.setCurrentIndex(0)

    def toggle_context_board(self):
        if self.w.current_creative_project_id:
            self.w.creative_flow.toggle_project_board()
        else:
            self.w._toggle_semantic_workspace_board()

    def choose_project_type(self):
        self.project_type_picker.resize(min(420, self.project_button.screen().availableGeometry().width() - 16), 0)
        self.project_type_picker.open()

    def new_project(self):
        self.project_type_picker.accept()
        self.project_picker.hide()
        self.w.creative_workspace.new_project()

    def new_inspiration_project(self):
        self.project_type_picker.accept()
        self.project_picker.hide()
        self.w._start_new_inspiration()

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
        for index, (button, callback) in enumerate(zip(self.stage_buttons, (cw.show_project_planning, cw.show_candidates, cw.show_collected, cw.show_board))):
            button.clicked.disconnect()
            button.setCheckable(True)
            button.toggled.connect(lambda checked, action=callback, stage=index:
                action() if checked or (stage == 1 and self.w.current_result_mode == "creative_node") else self.refresh())
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
                w.settings_sections["appearance"].insertLayout(0, appearance)
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
                w.settings_sections["appearance"].insertLayout(1, display)
            if index == 4:
                layout.addWidget(page, 1)
                page.show()  # Removing the former inspector tab hides its page.
            else:
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
        self.new_project_button.setText("＋ New Project" if english else "＋ 新建项目")
        self.import_project_button.setText("Import Project…" if english else "导入项目包…")
        self.project_type_picker.setWindowTitle("Choose Project Type" if english else "选择项目类型")
        self.create_node_project_button.setText("Creative Node Project" if english else "创作节点项目")
        self.create_inspiration_project_button.setText("Semantic Probe Project" if english else "语义探针项目")
        self.cancel_project_type_button.setText("Cancel" if english else "取消")
        self.node_project_hint.setText("Develop ideas by node, collect references, and organize a board." if english else "按节点构思、找参考，最后整理成看板。")
        self.inspiration_project_hint.setText("Explore several directions from a short description and find images in the library." if english else "用一句话快速探索多个灵感方向，再从图库找图。")
        self.w.creative_workspace.back_to_inspiration.setText("← Back to Explore Ideas" if english else "← 返回探索灵感")
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
        self.sync_sidebars()

    def _inspector_changed(self, index):
        if index in (3, 4):
            self.open_global_panel(index)
            self.w.right_tab_widget.setCurrentIndex(self._last_inspector_tab)
        else:
            self._last_inspector_tab = index

    def open_global_panel(self, index):
        if index == 3 and hasattr(self.w, "task_center"):
            self.w.task_center.open()
            return
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
        project_layout = w.creative_editor.layout()
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
        theme_id = get_theme().id
        if theme_id != self._sidebar_icon_theme:
            self._sidebar_icon_theme = theme_id
            self._sidebar_arrows = {direction: _sidebar_arrow_icon(direction) for direction in (False, True)}
        self.exit_focus_button.setVisible(w._board_focus_mode)
        self.exit_focus_button.setText("Exit Focus" if english else "退出专注")
        for index, button, label in ((0, self.left_button, "左"), (2, self.right_button, "右")):
            visible = not w._board_focus_mode and w._sidebar_visible[index]
            button.setChecked(visible)
            action = ("收起" if visible else "显示") + label + "侧栏"
            if english:
                action = ("Hide " if visible else "Show ") + ("Navigation" if index == 0 else "Inspector")
            icon = self._sidebar_arrows[(index == 0) == visible]
            if button.icon().cacheKey() != icon.cacheKey():
                button.setIcon(icon)
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
        exploring = cw.is_exploring_inspiration()
        if hasattr(w, 'image_upscaler'):
            w.image_upscaler.result_button.hide()
        if hasattr(w, 'preview_ai'):
            w.preview_ai.entry.hide()
        if project_id != self._last_project:
            self._last_project = project_id
            if project_id:
                self.navigation_tabs.setCurrentIndex(1)
        self.project_node_panel.setVisible(bool(project_id))
        self.project_empty.setVisible(not bool(project_id))
        if w.creative_project_combo.count() and w.creative_project_combo.itemData(0) is None:
            w.creative_project_combo.setItemText(0, "独立图库")
        self._sync_temporary_context()
        title = w.creative_project_combo.currentText()
        if project_id:
            title = title.rsplit(" (", 1)[0].removeprefix("⬆ ")
        elif exploring:
            title = "Semantic Probe Project" if w.current_language == "en" else "语义探针项目"
        self.project_button.setText(self.project_button.fontMetrics().elidedText(title, Qt.TextElideMode.ElideRight, 280) + " ▾")
        self.project_button.setToolTip(title)
        w.creative_node_status_label.hide()
        is_gallery = w.center_result_stack.currentWidget() is w.grid_view
        is_board = w.center_result_stack.currentWidget() is w.project_board_view
        semantic_id = w.semantic_workspace_id
        linked_board = bool(is_board and semantic_id and w._current_board_temp_project_id == semantic_id and
                            (w._temporary_project_state_payload(semantic_id) or {}).get("inspiration_id"))
        semantic = w.store.get_temporary_project(semantic_id) if semantic_id and (exploring or linked_board) else None
        if semantic:
            title = semantic.name
            self.project_button.setText(self.project_button.fontMetrics().elidedText(title, Qt.TextElideMode.ElideRight, 280) + " ▾")
            self.project_button.setToolTip(title)
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
        for index, button in enumerate(self.stage_buttons):
            button.setVisible(not project_id and not linked_board and index in (1, 3))
        self.project_board_button.setVisible(bool(project_id or semantic))
        if semantic:
            self.project_board_button.setText("← 返回探索灵感" if linked_board else f"已收集 · {semantic.image_count} 张")
        if project_id:
            previous = w.creative_flow._board_return if hasattr(w, 'creative_flow') else None
            from_search = bool(previous and previous[0] == project_id and previous[2])
            self.project_board_button.setText(('← 返回搜图' if from_search else '← 返回构思') if is_board else '项目看板')
            cw.context_label.setText("看板" if is_board else "找参考" if is_gallery else "构思")
        cw.navigation.setVisible(not exploring and not linked_board)
        self.selection_bar.setVisible(is_gallery and not exploring)
        self.board_tools.setVisible(is_board and not w._board_focus_mode)
        w.show_gallery_button.hide()
        w.show_project_board_button.hide()
        for widget in (w.shuffle_results_button, w.save_result_set_button, w.thumbnail_size_label, w.thumbnail_size_slider):
            widget.setVisible(is_gallery)

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
