"""Exercise public UI entry points against real library/project state."""
from unittest.mock import patch

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QFileDialog, QMenu

from test_filter_workflow import app, library, window, apply
from test_creative_workspace import project
from eidory.core.search_filters import FilterGroup, FilterPlan, SearchFilter
from eidory.ui.main_window import COLLECTION_VIRTUAL_FILTER_ROLE


def tag_library(w, images):
    w.store.set_image_tags(images[0].id, ['验收标签'])
    tag = next(t for t in w.store.list_tags() if t.tag_name == '验收标签')
    w._refresh_tags()
    return tag


@pytest.mark.parametrize('source', ['library', 'board', 'collected', 'frozen_search'])
def test_tag_context_search_applies_visible_rule_from_every_stage(window, library, source):
    w = window
    images = library[2]
    tag = tag_library(w, images)
    if source != 'library':
        _, node = project(w)
        w.store.add_images_to_creative_node(node.id, [images[1].id])
        if source == 'board':
            w.creative_workspace.show_board()
        elif source == 'collected':
            w.creative_workspace.show_collected()
        else:
            w.filter_workflow.scope = {'kind': 'snapshot', 'image_ids': [images[1].id], 'label': '旧范围'}
            apply(w, FilterPlan((FilterGroup((SearchFilter('keyword', 'no-match'),)),)))
    item = next(w.tag_list.item(i) for i in range(w.tag_list.count())
                if w.tag_list.item(i).data(Qt.ItemDataRole.UserRole) == tag.id)
    class SearchMenu(QMenu):
        def exec(self, *_):
            return self.actions()[0]

    with patch('eidory.ui.main_window.QMenu', SearchMenu):
        w._show_tag_context_menu(w.tag_list.visualItemRect(item).center())
    w._poll_events()
    assert w.center_result_stack.currentWidget() is w.grid_view
    assert [im.id for im in w.grid_view.images()] == [images[0].id]
    assert w.filter_workflow.plan().positive_filters == (SearchFilter('tag', f'any:{tag.id}'),)
    assert w.filter_panel.applied == w.filter_workflow.plan()
    assert w.filter_workflow.scope_label() == '全部图库'
    assert w.workspace_chrome.stage_buttons[1].isChecked()


@pytest.mark.parametrize('stage', ['board', 'collected', 'planning', 'search'])
def test_independent_library_leaves_all_project_context(window, library, stage):
    w = window
    pid, node = project(w)
    iid = library[2][0].id
    w.store.add_images_to_creative_node(node.id, [iid])
    if stage == 'board':
        w.creative_workspace.show_board()
        w.project_board_view._image_items[iid].setPos(QPointF(812, 463))
        w.project_board_view.select_image_id(iid)
        w.creative_flow.board_inspector.note.setPlainText('切换前未到防抖时间的用途')
    elif stage == 'collected':
        w.creative_workspace.show_collected()
    elif stage == 'planning':
        w.creative_flow.show_planning()
    else:
        w.creative_workspace.find()
        # Keep the completion event queued until after leaving the project.
    previous_revision = w.semantic_search_revision
    w.creative_project_combo.setCurrentIndex(0)
    w._poll_events()
    assert w.semantic_search_revision > previous_revision
    assert w.current_creative_project_id is None
    assert w.current_creative_node_id is None
    assert w.current_temp_project_id is None
    assert w.current_search_result_set_project_id is None
    assert w.creative_workspace.search_node_id is None
    assert w.center_result_stack.currentWidget() is w.grid_view
    assert w.current_result_mode == 'library'
    assert {im.id for im in w.grid_view.images()} == {im.id for im in library[2]}
    assert w.filter_workflow.plan() == FilterPlan()
    assert not w.workspace_chrome.stage_buttons[3].isEnabled()
    assert sum(b.isChecked() for b in w.workspace_chrome.stage_buttons) == 1
    assert w.workspace_chrome.stage_buttons[1].isChecked()
    w.workspace_chrome.stage_buttons[3].click()
    assert w.center_result_stack.currentWidget() is w.grid_view
    assert w.store.creative_node_image_ids(node.id) == [iid]
    if stage == 'board':
        assert w.creative_workspace.store.reference_note(node.id, iid) == '切换前未到防抖时间的用途'
        w._load_creative_project(pid, select_node_id=node.id, show_board=True)
        assert w.project_board_view._image_items[iid].pos() == QPointF(812, 463)


def test_untagged_navigation_leaves_frozen_collected_scope(window, library):
    w = window
    tag_library(w, library[2])
    _, node = project(w)
    w.store.add_images_to_creative_node(node.id, [library[2][0].id])
    w.creative_workspace.show_collected()
    w.creative_workspace.show_board()
    item = next(w.tag_list.item(i) for i in range(w.tag_list.count())
                if w.tag_list.item(i).data(COLLECTION_VIRTUAL_FILTER_ROLE) == 'untagged')
    w.tag_list.setCurrentItem(item)
    w._poll_events()
    assert w.center_result_stack.currentWidget() is w.grid_view
    assert {im.id for im in w.grid_view.images()} == {im.id for im in library[2][1:]}


def test_repeated_stage_click_still_has_exactly_one_active_stage(window):
    w = window
    for index in [1, 1, 0, 0, 1]:
        w.workspace_chrome.stage_buttons[index].click()
        assert sum(b.isChecked() for b in w.workspace_chrome.stage_buttons) == 1
        assert w.workspace_chrome.stage_buttons[index].isChecked()


def test_index_result_link_opens_actual_tasks_panel(window):
    w = window
    w.filter_workflow.open_index_panel()
    assert w.workspace_chrome._dialogs[3].isVisible()
    w.workspace_chrome._dialogs[3].hide()


@pytest.mark.parametrize('kind', ['quick', 'semantic', 'search'])
def test_reference_set_has_explicit_context_and_can_exit_to_library(window, library, kind):
    w = window
    tid = w.store.create_temporary_project('导航验收参考集', [library[2][0].id], kind=kind)
    w._show_temporary_project_board(tid)
    w.workspace_chrome.refresh()
    assert w.creative_project_combo.currentText() != '独立图库'
    assert w.workspace_chrome.stage_buttons[3].isEnabled() == (kind != 'search')
    w.creative_project_combo.setCurrentIndex(0)
    assert w.current_temp_project_id is None
    assert w.center_result_stack.currentWidget() is w.grid_view
    assert len(w.grid_view.images()) == len(library[2])
    assert not w.workspace_chrome.stage_buttons[3].isEnabled()


def test_project_tools_show_only_inside_a_project_and_import_stays_available(window):
    w = window
    chrome, tools = w.workspace_chrome, w.workflow_upgrade.tools
    assert not tools.isVisible()
    assert all(not action.isEnabled() for action in w.workflow_upgrade.project_actions)
    chrome.open_project_picker()
    assert chrome.import_project_button.isVisible()
    with patch.object(QFileDialog, 'getOpenFileName', return_value=('', '')) as choose_file:
        chrome.import_project_button.click()
    assert choose_file.call_args.args[0] is chrome.project_picker
    assert chrome.project_picker.isVisible()
    chrome.project_picker.hide()
    project(w)
    assert tools.isVisible()
    assert '导出创作参考包…' in [a.text() for a in tools.menu().actions()]
    assert not any('导入' in a.text() for a in tools.menu().actions())
    assert all(action.isEnabled() for action in w.workflow_upgrade.project_actions)
    chrome.open_library()
    assert not tools.isVisible()
    assert all(not action.isEnabled() for action in w.workflow_upgrade.project_actions)


def test_empty_tag_search_shows_empty_wall_instead_of_old_results(window):
    w = window
    tid = w.store.create_tag('没有图片')
    w._refresh_tags()
    w._search_tag_from_management(tid, '没有图片')
    w._poll_events()
    assert w.grid_view.images() == []
    assert '0' in w.filter_panel.result_label.text()
    assert not w.filter_panel.error


def test_returning_to_creative_board_ignores_previous_saved_search_context(window, library):
    w = window
    pid, node = project(w)
    w.store.add_images_to_creative_node(node.id, [library[2][0].id])
    tid = w.store.create_temporary_project('搜索结果集', [library[2][1].id], kind='search')
    w._load_temporary_project(tid)
    w._load_creative_project(pid, select_node_id=node.id, show_board=True)
    assert w.current_search_result_set_project_id is None
    assert w.workspace_chrome.stage_buttons[3].isEnabled()
    w.creative_flow.show_planning()
    w.workspace_chrome.stage_buttons[3].click()
    assert w.center_result_stack.currentWidget() is w.project_board_view


@pytest.mark.parametrize('entry', ['collection', 'folder', 'untagged', 'ai_index'])
def test_sidebar_filter_navigation_exits_board_with_active_search(window, library, entry):
    w = window
    _, node = project(w)
    w.store.add_images_to_creative_node(node.id, [library[2][0].id])
    apply(w, FilterPlan((FilterGroup((SearchFilter('keyword', 'plane'),)),)))
    w.creative_workspace.show_board()
    if entry == 'collection':
        w._on_collection_selection_changed()
    elif entry == 'folder':
        w._on_folder_selection_changed()
    elif entry == 'untagged':
        item = next(w.tag_list.item(i) for i in range(w.tag_list.count())
                    if w.tag_list.item(i).data(COLLECTION_VIRTUAL_FILTER_ROLE) == 'untagged')
        w.tag_list.setCurrentItem(item)
    else:
        w.ai_vision_virtual_filter_list.setCurrentRow(0)
    w._poll_events()
    assert w.center_result_stack.currentWidget() is w.grid_view
    assert {im.file_name for im in w.grid_view.images()} == {'red-plane.png', 'blue-plane.png'}
