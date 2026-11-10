from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
import pytest

from test_filter_workflow import app, library, window, apply
from test_creative_workspace import project
from eidory.core.search_filters import FilterGroup, FilterPlan, SearchFilter
from eidory.ui.main_window import MainWindow, LEFT_SIDEBAR_WIDTH, RIGHT_SIDEBAR_WIDTH


def test_navigation_preserves_project_filters_and_selection(window, app):
    w = window
    pid, node = project(w)
    plan = FilterPlan(include=(FilterGroup((SearchFilter('keyword', 'plane'),)),))
    apply(w, plan)
    w.grid_view.select_image_ids([w.grid_view.images()[0].id])
    before = w.grid_view.selected_image_ids()
    chrome = w.workspace_chrome
    assert w.root_splitter.widget(0).isAncestorOf(w.creative_node_tree)
    assert chrome.header.isAncestorOf(chrome.project_button)
    assert w.creative_project_combo.isHidden()
    for index in (0, 1, 0):
        chrome.navigation_tabs.setCurrentIndex(index)
        app.processEvents()
        assert w.current_creative_project_id == pid
        assert w.current_creative_node_id == node.id
        assert w.filter_workflow.plan() == plan
        assert w.grid_view.selected_image_ids() == before
    w.creative_workspace.refresh()
    assert chrome.navigation_tabs.currentIndex() == 0


def test_sidebars_can_be_restored_from_header_in_focus_mode(window, app):
    w = window
    chrome = w.workspace_chrome
    original = w.root_splitter.sizes()
    left_arrow = chrome.left_button.icon().cacheKey()
    right_arrow = chrome.right_button.icon().cacheKey()
    assert left_arrow != right_arrow
    for index, button, expanded_icon, collapsed_icon in (
        (0, chrome.left_button, left_arrow, right_arrow),
        (2, chrome.right_button, right_arrow, left_arrow),
    ):
        assert not button.text() and not button.icon().isNull()
        button.click()
        app.processEvents()
        assert w.root_splitter.sizes()[index] == 0
        assert button.isVisible()
        assert not button.isChecked()
        assert not button.text()
        assert button.icon().cacheKey() == collapsed_icon
        assert button.accessibleName() == ("显示左侧栏" if index == 0 else "显示右侧栏")
        button.click()
        app.processEvents()
        assert w.root_splitter.sizes()[index] == original[index]
        assert button.icon().cacheKey() == expanded_icon
    w._set_board_focus_mode(True)
    assert w.root_splitter.sizes()[0] == 0
    assert chrome.left_button.isVisible()
    chrome.left_button.click()
    app.processEvents()
    assert not w._board_focus_mode
    assert w.root_splitter.sizes()[0] == original[0]


@pytest.mark.parametrize("saved", ["0,1200,386", "0,1500,0", "20,900,30"])
def test_startup_recovers_legacy_collapsed_sidebars(library, app, saved):
    paths, store, _ = library
    store.set_setting("ui.root_splitter_sizes", saved)
    w = MainWindow(paths=paths, store=store)
    try:
        w.resize(1500, 900)
        w.show()
        app.processEvents()
        assert w.root_splitter.sizes()[0] == LEFT_SIDEBAR_WIDTH
        assert w.root_splitter.sizes()[2] == RIGHT_SIDEBAR_WIDTH
        assert w.collection_tree.isVisible()
    finally:
        w.close()


@pytest.mark.parametrize("mode", ["left", "both", "focus", "board_focus"])
def test_restart_after_temporary_hiding_restores_sidebars(window, app, mode):
    w = window
    if mode in ("left", "both"):
        w.workspace_chrome.left_button.click()
        if mode == "both":
            w.workspace_chrome.right_button.click()
    else:
        if mode == "board_focus":
            w.center_result_stack.setCurrentWidget(w.project_board_view)
        w._set_board_focus_mode(True)
    app.processEvents()
    assert w.root_splitter.sizes()[0] == 0
    w.close()
    saved = [int(value) for value in w.store.get_setting("ui.root_splitter_sizes").split(",")]
    assert saved[0] == LEFT_SIDEBAR_WIDTH
    assert saved[2] == RIGHT_SIDEBAR_WIDTH
    reopened = MainWindow(paths=w.paths, store=w.store)
    try:
        reopened.resize(1500, 900)
        reopened.show()
        app.processEvents()
        assert reopened.collection_tree.isVisible()
        assert reopened.root_splitter.sizes()[0] == LEFT_SIDEBAR_WIDTH
        assert reopened.root_splitter.sizes()[2] == RIGHT_SIDEBAR_WIDTH
        assert not reopened._board_focus_mode
    finally:
        reopened.close()


def test_tab_never_hides_sidebars_while_navigating_or_editing(window, app):
    w = window
    w.right_tab_widget.setCurrentIndex(2)
    for target in (w.grid_view, w.tag_search_input, w.tag_list):
        target.setFocus()
        QTest.keyClick(target, Qt.Key.Key_Tab)
        app.processEvents()
        assert not w._board_focus_mode
        assert w.root_splitter.sizes()[0] == LEFT_SIDEBAR_WIDTH
        assert w.root_splitter.sizes()[2] == RIGHT_SIDEBAR_WIDTH
    w.grid_view.setFocus()
    QTest.keyClick(w.grid_view, Qt.Key.Key_F, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    app.processEvents()
    assert w._board_focus_mode
    assert w.workspace_chrome.exit_focus_button.isVisible()
    w.workspace_chrome.exit_focus_button.click()
    app.processEvents()
    assert not w._board_focus_mode
    assert w.collection_tree.isVisible()


def test_resize_and_splitter_drag_cannot_hide_navigation(window, app):
    w = window
    for width in (1000, 2300, 1200, 1500):
        w.resize(width, 900)
        app.processEvents()
        w.root_splitter.moveSplitter(0, 1)
        w.root_splitter.moveSplitter(w.root_splitter.width(), 2)
        app.processEvents()
        assert w.root_splitter.sizes()[0] >= 220
        assert w.root_splitter.sizes()[2] >= 260
        assert w.collection_tree.isVisible()
        assert w.workspace_chrome.left_button.isChecked()


def test_restore_layout_preserves_filter_and_selection(window, app):
    w = window
    plan = FilterPlan(include=(FilterGroup((SearchFilter('keyword', 'plane'),)),))
    apply(w, plan)
    w.grid_view.select_image_ids([w.grid_view.images()[0].id])
    before = w.grid_view.selected_image_ids()
    w.workspace_chrome.left_button.click()
    w._set_board_focus_mode(True)
    w.workspace_chrome.restore_layout_button.click()
    app.processEvents()
    assert not w._board_focus_mode
    assert w.root_splitter.sizes()[0] == LEFT_SIDEBAR_WIDTH
    assert w.root_splitter.sizes()[2] == RIGHT_SIDEBAR_WIDTH
    assert w.filter_workflow.plan() == plan
    assert w.grid_view.selected_image_ids() == before


def test_global_panels_reuse_controls_without_replacing_results(window, app):
    w = window
    chrome = w.workspace_chrome
    plan = FilterPlan(include=(FilterGroup((SearchFilter('keyword', 'plane'),)),))
    apply(w, plan)
    results = [image.id for image in w.grid_view.images()]
    for index, button in ((3, chrome.tasks_button), (4, chrome.settings_button)):
        button.click()
        app.processEvents()
        dialog = chrome._dialogs[index]
        assert dialog.isVisible()
        assert not dialog.isModal()
        assert not w.right_tab_widget.isTabVisible(index)
        if index == 3:
            assert dialog.isAncestorOf(w.ai_vision_rule_tree)
        else:
            chrome.density_combo.setCurrentIndex(1)
            assert w.store.get_setting('ui.density') == 'comfortable'
            chrome.density_combo.setCurrentIndex(0)
        dialog.close()
        assert w.filter_workflow.plan() == plan
        assert [image.id for image in w.grid_view.images()] == results


def test_collection_undo_keeps_existing_and_other_node_references(window, app):
    w = window
    pid, node = project(w)
    other = w.store.list_creative_nodes(pid)[2]
    images = w.store.list_images(limit=None)
    first, new = images[:2]
    w.store.add_images_to_creative_node(node.id, [first.id])
    w.store.add_images_to_creative_node(other.id, [new.id])
    w.grid_view.set_images(images)
    w.grid_view.select_image_ids([first.id, new.id])
    w._save_selection_to_current_creative_node()
    assert set(w.store.creative_node_image_ids(node.id)) == {first.id, new.id}
    assert w.workspace_chrome.undo_collect_button.isEnabled()
    w.workspace_chrome.undo_collect_button.click()
    assert w.store.creative_node_image_ids(node.id) == [first.id]
    assert w.store.creative_node_image_ids(other.id) == [new.id]
    assert not w.workspace_chrome.undo_collect_button.isEnabled()
    assert new.id in w.grid_view._project_references
    assert node.title not in w.grid_view._project_references[new.id]


def test_collection_undo_does_not_act_on_a_different_project(window):
    w = window
    pid, node = project(w)
    image = w.store.list_images(limit=None)[0]
    w.grid_view.select_image_ids([image.id])
    w._save_selection_to_current_creative_node()
    project(w, '第二个项目')
    assert not w.workspace_chrome.undo_collect_button.isEnabled()
    w.workspace_chrome.undo_collection()
    assert w.store.creative_node_image_ids(node.id) == [image.id]
    w._load_creative_project(pid, select_node_id=node.id)
    w.workspace_chrome.undo_collect_button.click()
    assert w.store.creative_node_image_ids(node.id) == []


def test_stage_navigation_reflects_current_work_and_recovers_from_board(window, app):
    w = window
    pid, node = project(w)
    w.creative_node_note_input.setPlainText('沙漠古城')
    w.workflow_upgrade.flush()
    image = w.store.list_images(limit=None)[0]
    w.store.add_images_to_creative_node(node.id, [image.id])
    chrome = w.workspace_chrome
    w.creative_flow.show_planning()
    w.creative_flow.document_search.click()
    w._poll_events()
    chrome.project_board_button.click()
    assert chrome.board_tools.isVisible()
    w._set_board_focus_mode(True)
    assert chrome.project_board_button.isVisible()
    chrome.project_board_button.click()
    app.processEvents()
    assert not w._board_focus_mode
    assert w.center_result_stack.currentWidget() is w.grid_view
    assert chrome.selection_bar.isVisible()


def test_header_stays_compact_when_window_grows(window, app):
    w = window
    for width, height in ((1400, 900), (2300, 1200)):
        w.resize(width, height)
        app.processEvents()
        assert w.workspace_chrome.header.height() < 64
        assert w.root_splitter.height() > w.height() * .8


def test_index_filter_returns_to_visible_results(window, app):
    w = window
    w.workspace_chrome.open_global_panel(3)
    app.processEvents()
    choices = w.ai_vision_virtual_filter_list
    item = choices.item(0)
    assert item is not None
    QTest.mouseClick(choices.viewport(), Qt.MouseButton.LeftButton, pos=choices.visualItemRect(item).center())
    app.processEvents()
    assert not w.workspace_chrome._dialogs[3].isVisible()
    assert w.center_result_stack.currentWidget() is w.grid_view
    assert w.current_virtual_filter == 'un_ai_scene_tagged'


def test_accessibility_toggle_changes_stage_content(window, app):
    w = window
    _, node = project(w)
    w.creative_node_note_input.setPlainText('沙漠古城')
    w.workflow_upgrade.flush()
    image = w.store.list_images(limit=None)[0]
    w.store.add_images_to_creative_node(node.id, [image.id])
    for stage in (3, 0, 2, 1):
        # macOS accessibility sends a check-state change without clicked().
        w.workspace_chrome.stage_buttons[stage].setChecked(True)
        w._poll_events()
        app.processEvents()
        assert sum(b.isChecked() for b in w.workspace_chrome.stage_buttons) == 1
        if stage == 3:
            assert w.center_result_stack.currentWidget() is w.project_board_view
        elif stage in (1, 2):
            assert w.center_result_stack.currentWidget() is w.grid_view
        else:
            assert w.center_result_stack.currentWidget() is not w.grid_view
            assert w.center_result_stack.currentWidget() is not w.project_board_view


def test_board_toolbar_overflows_without_losing_actions_or_board_state(window, library, app):
    from test_board_inspector import board_project

    w = window
    board_project(w, library[2])
    board = w.project_board_view
    iid = library[2][0].id
    board.select_image_id(iid)
    toolbar = w.workspace_chrome.board_tools
    layout = board.layout_payload()
    w.resize(2400, 900)
    QTest.qWait(50)
    assert not toolbar.more.isVisible()
    assert w.board_grayscale_button.isVisible()
    assert w.workspace_chrome.board_section_lock_button.isVisible()

    w.resize(1050, 900)
    QTest.qWait(50)
    assert toolbar.more.isVisible()
    assert not w.board_grayscale_button.isVisible()
    toolbar.build_menu()
    transform = board.transform()
    action = next(a for a in toolbar.menu.actions() if a.text() == '黑白显示 (B)')
    action.trigger()
    assert board.layout_payload()['items'][str(iid)]['grayscale'] is True
    assert board.selected_image_ids() == [iid]
    assert board.transform() == transform
    local = next(a for a in toolbar.menu.actions() if a.text() == '局部／全图切换 (V)')
    assert not local.isEnabled()
    from unittest.mock import patch
    with patch.object(w, '_set_native_window_pinned', return_value=True):
        # Native accessibility can toggle a button without sending clicked().
        w.board_pin_button.setChecked(True)
        assert w._board_window_pinned
        toolbar.build_menu()
        pin = next(a for a in toolbar.menu.actions() if a.text() == '窗口置顶 (P)')
        assert pin.isChecked()
        pin.trigger()
        assert not w._board_window_pinned
        assert not w.board_pin_button.isChecked()
    w.resize(2400, 900)
    QTest.qWait(50)
    assert not toolbar.more.isVisible()
    w.board_grayscale_button.click()
    assert board.layout_payload() == layout


def test_board_toolbar_adapts_to_font_and_stage_changes_without_clipping(window, library, app):
    from test_board_inspector import board_project
    from PySide6.QtWidgets import QPushButton

    w = window
    board_project(w, library[2])
    toolbar = w.workspace_chrome.board_tools
    for width, points in ((2400, 13), (1250, 20), (1050, 20), (1800, 16), (2400, 13)):
        w.resize(width, 900)
        for b in toolbar.findChildren(QPushButton):
            font = b.font()
            font.setPointSize(points)
            b.setFont(font)
        w.creative_flow.show_planning()
        w.creative_flow.show_project_board()
        QTest.qWait(50)
        toolbar.build_menu()
        visible = [b for b in toolbar.findChildren(QPushButton) if b.isVisible()]
        visible.sort(key=lambda b: b.x())
        for b in visible:
            assert b.width() >= min(b.maximumWidth(), b.sizeHint().width())
            assert b.geometry().right() < toolbar.width()
        assert all(a.geometry().right() < b.x() for a, b in zip(visible, visible[1:]))
        assert len(toolbar.menu.actions()) == len(toolbar.overflow)
        assert all(not b.isVisible() for b in toolbar.overflow)
