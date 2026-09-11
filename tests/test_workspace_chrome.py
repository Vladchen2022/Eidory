from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from test_filter_workflow import app, library, window, apply
from test_creative_workspace import project
from eidory.core.search_filters import FilterGroup, FilterPlan, SearchFilter


def test_navigation_preserves_project_filters_and_selection(window, app):
    w = window
    pid, node = project(w)
    plan = FilterPlan(include=(FilterGroup((SearchFilter('keyword', 'plane'),)),))
    apply(w, plan)
    w.grid_view.select_image_ids([w.grid_view.images()[0].id])
    before = w.grid_view.selected_image_ids()
    chrome = w.workspace_chrome
    assert w.root_splitter.widget(0).isAncestorOf(w.creative_node_tree)
    assert chrome.header.isAncestorOf(w.creative_project_combo)
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
    for index, button in ((0, chrome.left_button), (2, chrome.right_button)):
        button.click()
        app.processEvents()
        assert w.root_splitter.sizes()[index] == 0
        assert button.isVisible()
        assert not button.isChecked()
        button.click()
        app.processEvents()
        assert w.root_splitter.sizes()[index] == original[index]
    w._set_board_focus_mode(True)
    assert w.root_splitter.sizes()[0] == 0
    assert chrome.left_button.isVisible()
    chrome.left_button.click()
    app.processEvents()
    assert not w._board_focus_mode
    assert w.root_splitter.sizes()[0] == original[0]


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
    image = w.store.list_images(limit=None)[0]
    w.store.add_images_to_creative_node(node.id, [image.id])
    chrome = w.workspace_chrome
    for index in (0, 1, 2, 3):
        chrome.stage_buttons[index].click()
        w._poll_events()
        app.processEvents()
        assert chrome.stage_buttons[index].isChecked()
        assert sum(button.isChecked() for button in chrome.stage_buttons) == 1
    assert chrome.board_tools.isVisible()
    w._set_board_focus_mode(True)
    assert chrome.stage_buttons[1].isVisible()
    chrome.stage_buttons[1].click()
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
