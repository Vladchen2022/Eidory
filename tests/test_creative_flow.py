from copy import deepcopy

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QLineEdit

from test_filter_workflow import app, library, window, apply
from eidory.core.creative_templates import CREATIVE_TEMPLATES
from eidory.core.creative_entries import decode_entries
from eidory.core.search_filters import FilterGroup, FilterPlan, SearchFilter


def make_project(w):
    pid = w._create_creative_project_from_seed(title='交互验收', brief='三位工程师改造机器人', extra='无武器')
    nodes = w.store.list_creative_nodes(pid)
    node = next(n for n in nodes if n.title == '物件')
    w._load_creative_project(pid, select_node_id=node.id)
    w.creative_node_note_input.setPlainText('白色机器人，打开膝关节，不能携带武器。')
    w.workflow_upgrade.flush()
    return pid, w.store.get_creative_node(node.id), nodes


@pytest.mark.parametrize('template', CREATIVE_TEMPLATES, ids=lambda t: t.id)
def test_start_preserves_template_and_never_searches_its_instructions(window, template):
    w, f = window, window.creative_flow
    f.start_brief.setPlainText('测试主题')
    f.start_extra.setPlainText('恰好三人')
    f.start_template.setCurrentIndex(f.start_template.findData(template.id))
    f.create_project()
    assert w.center_result_stack.currentWidget() is f.home
    pid = w.current_creative_project_id
    assert w.creative_workspace.store.project(pid)['extra'] == '恰好三人'
    for n in w.store.list_creative_nodes(pid):
        if n.parent_id is not None:
            f.select_node(n.id)
            assert f.targets() == []
    f.open_search()
    assert w.center_result_stack.currentWidget() is f.search_empty
    assert not w.search_service.scopes


def test_partial_adoption_and_skip_preserve_original_and_remaining_ideas(window):
    w, f = window, window.creative_flow
    pid, node, _ = make_project(w)
    store = w.creative_workspace.store
    store.stage_entries(node.id, [{'name':'检测仪','visual':'手持式'}, {'name':'机械臂','visual':'固定在台面'}, {'name':'电缆架','visual':'可移动'}])
    pending = decode_entries(store.node(node.id)['proposal_entries_json'])
    f.accept_one(pending['rows'][0]['id'])
    assert store.entries(node.id)['base_note'] == node.note
    assert [r['name'] for r in store.entries(node.id)['rows']] == ['检测仪']
    assert len(decode_entries(store.node(node.id)['proposal_entries_json'])['rows']) == 2
    f.skip_one(pending['rows'][1]['id'])
    assert [r['name'] for r in decode_entries(store.node(node.id)['proposal_entries_json'])['rows']] == ['电缆架']
    f.accept_one(pending['rows'][2]['id'])
    assert [r['name'] for r in store.entries(node.id)['rows']] == ['检测仪','电缆架']
    assert not store.node(node.id)['proposal_entries_json']
    store.restore(node.id)
    assert [r['name'] for r in store.entries(node.id)['rows']] == ['检测仪']


def test_search_changes_do_not_edit_ideas_and_keep_advanced_filters(window):
    w, f = window, window.creative_flow
    _, node, _ = make_project(w)
    rows = f.targets()
    f.change_target(rows[0]['id'], query='机器人')
    f.run_targets(); w._poll_events()
    old = w.filter_workflow.plan()
    advanced = FilterPlan((FilterGroup(old.include[0].filters + (SearchFilter('orientation','landscape'),), 'all'),), (FilterGroup((SearchFilter('keyword','car'),)),))
    apply(w, advanced)
    w.grid_view.select_image_ids([w.grid_view.images()[0].id])
    f.change_target(rows[0]['id'], query='机械关节')
    assert not w.creative_workspace.collect_button.isEnabled()
    assert '上次结果' in f.search_context.text()
    assert w.store.get_creative_node(node.id).note == node.note
    f.run_targets(); w._poll_events()
    after = w.filter_workflow.plan()
    assert after.exclude == advanced.exclude
    assert SearchFilter('orientation','landscape') in after.positive_filters
    assert any(c.value == '机械关节' for c in after.positive_filters)
    assert not any(c.value == '机器人' for c in after.positive_filters)


def test_switching_content_changes_collected_images_and_board(window, library):
    w, f = window, window.creative_flow
    pid, a, nodes = make_project(w)
    b = next(n for n in nodes if n.title == '地点')
    ids = [im.id for im in library[2][:2]]
    w.store.add_images_to_creative_node(a.id, [ids[0]])
    w.store.add_images_to_creative_node(b.id, [ids[1]])
    w.creative_workspace.show_collected()
    w.creative_node_tree.setCurrentItem(w._find_creative_node_tree_item(b.id))
    assert [im.id for im in w.grid_view.images()] == [ids[1]]
    w.creative_workspace.show_board()
    w.creative_node_tree.setCurrentItem(w._find_creative_node_tree_item(a.id))
    assert set(w.project_board_view.layout_payload()['items']) == {str(ids[0])}


def test_tray_remove_undo_keeps_grade_purpose_and_note(window, library):
    w, f = window, window.creative_flow
    _, node, _ = make_project(w)
    image = library[2][0]
    w.store.add_images_to_creative_node(node.id, [image.id])
    w.workflow_upgrade.state.set_decision(node.id, image.id, 'primary', '造型')
    w.creative_workspace.store.save_reference_note(node.id, image.id, '只参考关节')
    f.refresh()
    f.tray.setCurrentRow(0); f.remove_reference()
    assert not w.store.creative_node_image_ids(node.id)
    assert w.creative_project_combo.currentText().endswith('/0)')
    f.restore_reference()
    assert w.store.creative_node_image_ids(node.id) == [image.id]
    assert w.creative_project_combo.currentText().endswith('/1)')
    assert w.workflow_upgrade.state.decision(node.id,image.id)['status'] == 'primary'
    assert w.creative_workspace.store.reference_note(node.id,image.id) == '只参考关节'


def test_start_sidebar_is_restored_and_active_filter_boxes_cannot_collapse(window):
    w, f = window, window.creative_flow
    f.show_planning()
    assert not w._sidebar_visible[2] and w._sidebar_visible[0]
    f.browse_library()
    assert w._sidebar_visible[2] and w._sidebar_visible[0]
    panel = w.filter_panel
    assert not panel.section_boxes['include'].isVisible()
    panel.expand_button.click()
    assert panel.section_boxes['include'].isVisible()
    panel.put_condition('exclude',0,SearchFilter('keyword','car'))
    assert not panel.expand_button.isVisible()
    assert all(b.isVisible() for b in panel.section_boxes.values())


def test_editor_does_not_stretch_short_labels_to_fill_window(window):
    w, f = window, window.creative_flow
    make_project(w)
    f.show_planning()
    QTest.qWait(30)
    assert f.node_caption.height() < 40
    assert w.creative_node_note_input.height() <= 140
    assert not w.thumbnail_size_slider.isVisible()


def test_unstructured_core_uses_user_note_instead_of_template_query(window):
    w = window
    _, node, _ = make_project(w)
    assert w.creative_flow.targets()[0]['query'] == node.note


def test_busy_search_cannot_collect_old_results(window, library):
    w = window
    _, node, _ = make_project(w)
    w.grid_view.set_images(library[2])
    w.grid_view.select_image_ids([library[2][0].id])
    w.search_button.setEnabled(False)
    w.creative_flow.refresh()
    assert not w.creative_workspace.collect_button.isEnabled()
    w._save_selection_to_current_creative_node()
    assert not w.store.creative_node_image_ids(node.id)
    w.search_button.setEnabled(True)


def test_search_query_is_saved_without_requiring_focus_change(window):
    w, f = window, window.creative_flow
    make_project(w)
    f.open_search(); w._poll_events()
    field = f.target_content.findChild(QLineEdit)
    field.setText('机器人关节')
    field.textEdited.emit(field.text())
    assert f.targets()[0]['query'] == '机器人关节'
    assert f._search_dirty_node == w.current_creative_node_id
    f.refresh()
    assert field is f.target_content.findChild(QLineEdit)
    f.run_targets(); w._poll_events()
    assert any(c.value == '机器人关节' for c in w.filter_workflow.plan().positive_filters)


def test_project_picker_resumes_existing_content(window, app):
    from eidory.ui.main_window import PROJECT_LIST_ID_ROLE, PROJECT_LIST_KIND_ROLE
    w, f = window, window.creative_flow
    pid, node, _ = make_project(w)
    chrome = w.workspace_chrome
    chrome.open_library()
    chrome.open_project_picker()
    item = next(w.temp_project_list.item(i) for i in range(w.temp_project_list.count())
                if w.temp_project_list.item(i).data(PROJECT_LIST_KIND_ROLE) == 'creative'
                and w.temp_project_list.item(i).data(PROJECT_LIST_ID_ROLE) == pid)
    w.temp_project_list.setCurrentItem(item)
    assert w.current_creative_project_id is None
    chrome._activate_project(item)
    assert not chrome.project_picker.isVisible()
    assert w.current_creative_project_id == pid
    assert w.creative_project_combo.currentData() == pid
    assert w.current_creative_node_id == node.id
    assert w.center_result_stack.currentWidget() is w.creative_workspace.planning_scroll
    assert w.store.get_creative_node(node.id).note == node.note


def test_new_project_clears_draft_and_project_context(window):
    w, f = window, window.creative_flow
    make_project(w)
    f.start_brief.setPlainText('上个项目')
    f.start_extra.setPlainText('旧约束')
    w.creative_workspace.new_project()
    assert w.current_creative_project_id is None
    assert w.creative_project_combo.currentData() is None
    assert not f.start_brief.toPlainText()
    assert not f.start_extra.toPlainText()


def test_reference_grade_buttons_save_to_selected_node(window, library):
    w, f = window, window.creative_flow
    _, node, _ = make_project(w)
    image = library[2][0]
    w.store.add_images_to_creative_node(node.id, [image.id])
    w.creative_workspace.show_collected()
    w.grid_view.select_image_ids([image.id])
    w.creative_workspace.refresh_reference()
    f.refresh()
    primary = w.workflow_upgrade.status.findData('primary')
    f.grade_buttons[primary].setChecked(True)
    assert w.workflow_upgrade.state.decision(node.id, image.id)['status'] == 'primary'
    assert f.grade_buttons[primary].isChecked()
    assert sum(b.isChecked() for b in f.grade_buttons) == 1


def test_project_tray_preserves_child_grade_and_opens_its_owner(window, library):
    w, f = window, window.creative_flow
    pid, node, _ = make_project(w)
    image = library[2][0]
    w.store.add_images_to_creative_node(node.id, [image.id])
    w.workflow_upgrade.state.set_decision(node.id, image.id, 'primary', '造型')
    f.select_node(w.store.creative_root_node_id(pid))
    f.refresh()
    f.tray.setCurrentRow(0)
    assert '主参考' in f.tray.currentItem().text()
    assert node.title in f.tray.currentItem().text()
    f.edit_tray_reference()
    assert w.current_creative_node_id == node.id
    assert w.creative_workspace._reference_key == (node.id, image.id)


def test_overview_uses_whole_project_context(window):
    w, f = window, window.creative_flow
    pid, node, _ = make_project(w)
    f.show_overview()
    assert w.current_creative_node_id == w.store.creative_root_node_id(pid)
    assert w.center_result_stack.currentWidget() is f.home
    assert w.store.get_creative_node(node.id).note == node.note


def test_open_board_cannot_override_a_later_navigation(window):
    w, f = window, window.creative_flow
    pid, node, _ = make_project(w)
    w._load_creative_project(pid, select_node_id=node.id, show_board=True)
    assert w.center_result_stack.currentWidget() is w.project_board_view
    f.show_planning()
    QTest.qWait(30)
    assert w.center_result_stack.currentWidget() is w.creative_workspace.planning_scroll


def test_keyboard_node_navigation_keeps_focus_on_tree(window):
    w, f = window, window.creative_flow
    make_project(w)
    w.creative_workspace.show_board()
    w.creative_node_tree.setFocus()
    QTest.keyClick(w.creative_node_tree, Qt.Key.Key_Up)
    assert w.creative_node_tree.hasFocus()
    assert w.center_result_stack.currentWidget() is w.project_board_view
    first = w.current_creative_node_id
    QTest.keyClick(w.creative_node_tree, Qt.Key.Key_Up)
    assert w.current_creative_node_id != first
    assert w.creative_node_tree.hasFocus()
