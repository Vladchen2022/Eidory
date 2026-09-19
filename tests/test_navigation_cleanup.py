from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QPushButton

from test_filter_workflow import app, library, window
from test_creative_flow import make_project
from eidory.ui.main_window import PROJECT_LIST_ID_ROLE, PROJECT_LIST_KIND_ROLE


def project_item(w, pid):
    return next(w.temp_project_list.item(i) for i in range(w.temp_project_list.count())
                if w.temp_project_list.item(i).data(PROJECT_LIST_KIND_ROLE) == 'creative'
                and w.temp_project_list.item(i).data(PROJECT_LIST_ID_ROLE) == pid)


def test_picker_keyboard_requires_activation_and_escape_keeps_context(window, app):
    w = window
    pid, node, _ = make_project(w)
    chrome = w.workspace_chrome
    chrome.open_library()
    chrome.open_project_picker()
    w.temp_project_list.setCurrentItem(project_item(w, pid))
    assert w.current_creative_project_id is None
    QTest.keyClick(w.temp_project_list, Qt.Key.Key_Escape)
    assert not chrome.project_picker.isVisible()
    assert w.current_creative_project_id is None
    chrome.open_project_picker()
    w.temp_project_list.setCurrentItem(project_item(w, pid))
    QTest.keyClick(w.temp_project_list, Qt.Key.Key_Return)
    assert w.current_creative_project_id == pid
    assert not chrome.project_picker.isVisible()
    assert w.center_result_stack.currentWidget() is w.project_board_view
    chrome.open_project_picker()
    chrome.library_button.click()
    assert w.current_creative_project_id is None
    assert w.current_creative_node_id is None
    assert w.center_result_stack.currentWidget() is w.grid_view
    assert not chrome.stage_buttons[3].isEnabled()


def test_remaining_stage_controls_complete_reference_workflow(window, app):
    w = window
    _, node, _ = make_project(w)
    flow, chrome = w.creative_flow, w.workspace_chrome
    flow.show_planning()
    # The focused editor has one whole-node search entry, in the stage strip.
    visible = [b.text() for b in w.findChildren(QPushButton) if b.isVisible()]
    assert visible.count('找参考') == 1
    assert '找参考 →' not in visible
    assert '‹ 项目全貌' not in visible
    assert visible.count('项目全貌') == 1
    chrome.stage_buttons[1].click()
    w._poll_events()
    assert w.center_result_stack.currentWidget() is w.grid_view
    assert flow.search_brief.toolTip() == node.note
    assert flow.search_brief.text().startswith("构思：")
    assert w.store.get_creative_node(node.id).note == node.note
    images = w.grid_view.images()
    w.grid_view.select_image_ids([images[0].id])
    w.creative_workspace.collect_button.click()
    chrome.stage_buttons[3].click()
    assert w.center_result_stack.currentWidget() is w.project_board_view
    assert images[0].id in w.project_board_view._image_items
    chrome.stage_buttons[0].click()
    assert w.center_result_stack.currentWidget() is w.creative_workspace.planning_scroll
    assert w.current_creative_node_id == node.id


def test_project_commands_have_one_visible_home_and_new_project_still_works(window, app):
    w = window
    make_project(w)
    flow, chrome = w.creative_flow, w.workspace_chrome
    flow.show_overview()
    labels = [a.text() for a in w.workflow_upgrade.tools.menu().actions()]
    for name in ('主题与补充设定…', '项目检查…', '方案版本…', '导出创作参考包…', '探索灵感方向'):
        assert labels.count(name) == 1
    assert '功能准备检查…' not in labels  # Lives in Settings only.
    visible = [b.text() for b in w.findChildren(QPushButton) if b.isVisible()]
    for name in ('保存与恢复方案', '导出创作参考包', '导出创作参考包…', '整理项目看板', '项目看板 →'):
        assert name not in visible
    chrome.open_project_picker()
    chrome.new_project_button.click()
    assert not chrome.project_picker.isVisible()
    assert w.current_creative_project_id is None
    assert flow.home_stack.currentWidget() is flow.start
    flow.start_brief.setPlainText('入口验收：海边的机器人维修间')
    flow.create_button.click()
    assert w.current_creative_project_id is not None
    assert flow.home_stack.currentWidget() is flow.overview


def test_picker_mouse_click_can_rebuild_project_rows_safely(window, app):
    w = window
    pid, _, _ = make_project(w)
    chrome = w.workspace_chrome
    for _ in range(3):
        chrome.open_library()
        chrome.open_project_picker()
        app.processEvents()
        target = project_item(w, pid)
        w.temp_project_list.scrollToItem(target)
        QTest.mouseClick(w.temp_project_list.viewport(), Qt.MouseButton.LeftButton,
                         pos=w.temp_project_list.visualItemRect(target).center())
        app.processEvents()
        assert not chrome.project_picker.isVisible()
        assert w.current_creative_project_id == pid
        assert w.center_result_stack.currentWidget() is w.project_board_view
