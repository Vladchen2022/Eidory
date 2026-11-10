from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox, QPushButton
from types import SimpleNamespace
from unittest.mock import patch

from test_filter_workflow import app, library, window
from test_creative_flow import make_project
from eidory.ui.main_window import PROJECT_LIST_ID_ROLE, PROJECT_LIST_KIND_ROLE
from eidory.core.llm_provider import InspirationTerm


def test_task_center_excludes_detached_results_and_can_discard_them(window):
    w = window
    repo = w.reference_generation.repo
    center = w.task_center

    def candidate(job_id, node):
        path = repo.root / 'candidates' / str(node.project_id) / (job_id + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(job_id.encode())
        repo.save({'id': job_id, 'state': 'ready', 'project_id': node.project_id,
                   'node_id': node.id, 'project_title': '交互验收', 'node_title': node.title,
                   'file_path': str(path), 'created_at': '2026-11-10T10:00:00+00:00'})
        return path

    deleted_pid, deleted_project_node, _ = make_project(w)
    removed_project_file = candidate('deleted-project', deleted_project_node)
    _, deleted_node, nodes = make_project(w)
    removed_node_file = candidate('deleted-node', deleted_node)
    intact_file = candidate('intact-node', next(n for n in nodes if n.title == '地点'))
    center.open(2)
    center.kind.setCurrentIndex(1)
    assert center.ai_table.rowCount() == 3

    with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
        w._delete_selected_creative_node()
        assert center.counts['references'][0] == 2
        w._delete_creative_project(deleted_pid)

    assert center.tabs.tabText(2) == 'AI 生图与编辑 · 待审 1'
    assert set(center.visible_jobs) == {'intact-node'}
    assert '见「全部状态」' in center.ai_summary.text()
    center.kind.setCurrentIndex(0)
    assert center.visible_jobs['deleted-project']['detached_reason'] == '原项目已删除'
    assert center.visible_jobs['deleted-node']['detached_reason'] == '原节点已删除'
    row = next(r for r in range(center.ai_table.rowCount())
               if center.ai_table.item(r, 0).data(Qt.ItemDataRole.UserRole) == 'deleted-project')
    center.ai_table.selectRow(row)
    assert not center.review.isEnabled() and center.discard.isEnabled()
    center.discard.click()
    discarded = repo.root / 'discarded' / str(deleted_pid) / removed_project_file.name
    assert repo.get('deleted-project')['state'] == 'discarded'
    assert not removed_project_file.exists() and discarded.read_bytes() == b'deleted-project'
    assert intact_file.exists() and removed_node_file.exists()
    assert center.discard.text() == '撤销放弃'
    center.discard.click()
    assert removed_project_file.read_bytes() == b'deleted-project'
    assert repo.get('deleted-project')['state'] == 'ready'
    assert center.counts['references'][0] == 1
    center.dialog.hide()


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
    assert w.current_creative_node_id == node.id
    assert w.center_result_stack.currentWidget() is w.creative_workspace.planning_scroll
    chrome.open_project_picker()
    chrome.library_button.click()
    assert w.current_creative_project_id is None
    assert w.current_creative_node_id is None
    assert w.center_result_stack.currentWidget() is w.grid_view
    assert not chrome.stage_buttons[3].isEnabled()


def test_document_search_and_project_board_have_one_visible_entry(window, app):
    w = window
    _, node, _ = make_project(w)
    flow, chrome = w.creative_flow, w.workspace_chrome
    flow.show_planning()
    visible = [b.text() for b in w.findChildren(QPushButton) if b.isVisible()]
    assert visible.count('用这段构思找参考') == 1
    assert visible.count('项目看板') == 1
    assert not any(button.isVisible() for button in chrome.stage_buttons)
    assert w.generate_creative_details_button.isVisible()
    flow.change_target('core', query='旧找图关键词')
    flow.document_search.click()
    w._poll_events()
    assert w.center_result_stack.currentWidget() is w.grid_view
    assert [condition.value for condition in w.filter_workflow.plan().positive_filters
            if condition.kind == 'semantic'] == [node.note]
    assert not flow.target_area.isVisible()
    assert not w.filter_panel.isVisible()
    assert flow.search_brief.toolTip() == node.note
    assert flow.search_brief.text().startswith("构思：")
    assert w.store.get_creative_node(node.id).note == node.note
    images = w.grid_view.images()
    w.grid_view.select_image_ids([images[0].id])
    w.creative_workspace.collect_button.click()
    chrome.project_board_button.click()
    assert w.center_result_stack.currentWidget() is w.project_board_view
    assert images[0].id in w.project_board_view._image_items
    assert w._current_board_node_id == w.store.creative_root_node_id(node.project_id)
    assert w.project_board_view.layout_payload()['sections']
    assert chrome.project_board_button.text() == '← 返回搜图'
    chrome.project_board_button.click()
    assert w.center_result_stack.currentWidget() is w.grid_view
    flow.search_back.click()
    assert w.center_result_stack.currentWidget() is w.creative_workspace.planning_scroll
    assert w.current_creative_node_id == node.id


def test_project_commands_have_one_visible_home_and_new_project_still_works(window, app):
    w = window
    make_project(w)
    flow, chrome = w.creative_flow, w.workspace_chrome
    flow.show_overview()
    labels = [a.text() for a in w.workflow_upgrade.tools.menu().actions()]
    for name in ('主题与补充设定…', '方案版本…', '导出创作参考包…'):
        assert labels.count(name) == 1
    assert '功能准备检查…' not in labels  # Lives in Settings only.
    visible = [b.text() for b in w.findChildren(QPushButton) if b.isVisible()]
    for name in ('批量补全项目节点', '项目文案'):
        assert visible.count(name) == 1
        assert name not in labels
    assert not any('探索灵感' in name for name in visible + labels)
    assert not w.image_upscaler.result_button.isVisible()
    assert not w.preview_ai.entry.isVisible()
    w.open_creative_copy_button.click()
    assert w.center_result_stack.currentWidget() is w.creative_workspace.planning_scroll
    assert w.creative_content_tabs.currentIndex() == 1
    chrome.open_library()
    assert not w.image_upscaler.result_button.isVisible()
    assert not w.preview_ai.entry.isVisible()
    assert chrome.tasks_button.isVisible()
    for name in ('保存与恢复方案', '导出创作参考包', '导出创作参考包…', '整理项目看板', '项目看板 →'):
        assert name not in visible
    chrome.open_project_picker()
    chrome.new_project_button.click()
    assert chrome.project_type_picker.isVisible()
    chrome.create_node_project_button.click()
    assert not chrome.project_picker.isVisible()
    assert w.current_creative_project_id is None
    assert flow.home_stack.currentWidget() is flow.start
    flow.start_brief.setPlainText('入口验收：海边的机器人维修间')
    flow.create_button.click()
    assert w.current_creative_project_id is not None
    assert flow.home_stack.currentWidget() is flow.overview


def test_semantic_project_is_a_separate_new_project_route(window):
    w = window
    pid, node, _ = make_project(w)
    chrome, cw = w.workspace_chrome, w.creative_workspace
    w.creative_flow.show_planning()
    w.creative_node_note_input.setPlainText('先保存这条手写设定')
    w.inspiration_brief_input.setPlainText('上一次探针草稿')

    chrome.open_project_picker()
    chrome.new_project_button.click()
    chrome.cancel_project_type_button.click()
    assert chrome.project_picker.isVisible()
    assert w.current_creative_project_id == pid
    chrome.new_project_button.click()
    chrome.create_inspiration_project_button.click()

    assert not chrome.project_picker.isVisible()
    assert w.current_creative_project_id is None and w.current_creative_node_id is None
    assert w.center_result_stack.currentWidget() is cw.inspiration_scroll
    assert w.inspiration_brief_input.toPlainText() == ''
    assert chrome.project_button.text().startswith('语义探针项目')
    assert not w.creative_node_tree.isVisible() and not cw.navigation.isVisible()
    assert not w.open_creative_copy_button.isVisible()
    assert w.store.get_creative_node(node.id).note == '先保存这条手写设定'

    w.inspiration_brief_input.setPlainText('低光环境中的玻璃温室')
    terms = [InspirationTerm(title='温室玻璃', query='温室玻璃', reason='寻找透光结构')]
    w._show_inspiration_proposal(SimpleNamespace(terms=terms, questions=[], model_name='fixture'))
    images = w.store.list_images(limit=None)
    with patch.object(w.search_service, 'semantic_search', return_value=SimpleNamespace(images=images)):
        w.search_inspiration_button.click()
        w._poll_events()
    draft_id = w.inspiration_draft_project_id
    assert draft_id and w.store.get_inspiration_project(draft_id).brief == '低光环境中的玻璃温室'
    assert w.center_result_stack.currentWidget() is w.grid_view and w.current_result_mode == 'inspiration'
    assert cw.back_to_inspiration.isVisible() and w.save_temp_project_button.isVisible()
    assert not chrome.selection_bar.isVisible()
    semantic_id = w.semantic_workspace_id
    assert w.store.semantic_workspace_links()[draft_id] == semantic_id
    visible = w.grid_view.images()
    assert visible
    w.grid_view.select_image_ids([visible[0].id])
    w.save_temp_project_button.click()
    assert w.store.temporary_project_image_ids(semantic_id) == [visible[0].id]
    cw.back_to_inspiration.click()
    assert w.center_result_stack.currentWidget() is cw.inspiration_scroll
    assert w.inspiration_brief_input.toPlainText() == '低光环境中的玻璃温室'
    assert w.inspiration_draft_project_id == draft_id
    chrome.project_board_button.click()
    assert w.center_result_stack.currentWidget() is w.project_board_view
    assert w._current_board_temp_project_id == semantic_id
    chrome.project_board_button.click()
    assert w.center_result_stack.currentWidget() is cw.inspiration_scroll
    chrome.open_project_picker()
    w._expand_project_sidebar_section("creative")
    w._refresh_project_sidebar()
    chrome._activate_project(project_item(w, pid))
    assert w.current_creative_project_id == pid
    assert w.current_creative_node_id == node.id
    assert w.center_result_stack.currentWidget() is cw.planning_scroll
    assert not cw.inspiration_scroll.isVisible()
    assert w.inspiration_draft_project_id == draft_id
    chrome.open_project_picker()
    w._expand_project_sidebar_section("temporary")
    w._refresh_project_sidebar()
    saved = next(w.temp_project_list.item(i) for i in range(w.temp_project_list.count())
                 if w.temp_project_list.item(i).data(PROJECT_LIST_KIND_ROLE) == "temporary"
                 and w.temp_project_list.item(i).data(PROJECT_LIST_ID_ROLE) == semantic_id)
    chrome._activate_project(saved)
    assert not chrome.project_picker.isVisible()
    assert w.current_creative_project_id is None
    assert w.inspiration_draft_project_id == draft_id
    assert w.center_result_stack.currentWidget() is cw.inspiration_scroll


def test_reopen_project_resumes_search_and_image_selection(window):
    w = window
    pid, node, _ = make_project(w)
    w.creative_flow.show_planning()
    w.creative_flow.document_search.click()
    w._poll_events()
    image = w.grid_view.images()[0]
    w.grid_view.select_image_ids([image.id])
    w.workspace_chrome.open_library()
    w.creative_workspace.open_project(pid)
    w._poll_events()
    assert w.current_creative_node_id == node.id
    assert w.center_result_stack.currentWidget() is w.grid_view
    assert w.grid_view.selected_image_ids() == [image.id]


def test_task_center_uses_one_window_and_settings_are_split(window):
    w = window
    center = w.task_center
    w.workspace_chrome.open_global_panel(3)
    w.image_upscaler.open_tasks()
    assert center.tabs.currentIndex() == 1
    assert center.upscale_placeholder.isAncestorOf(w.image_upscaler.tasks)
    center.tabs.setCurrentIndex(2)
    assert center.ai_table.isVisible()
    assert w.settings_categories.count() == 4
    backup = w.settings_sections["backup"].parentWidget()
    assert backup.isAncestorOf(w.workspace_migration_button)
    assert backup.isAncestorOf(w.backup_database_button)
    center.dialog.hide()
    w.workspace_chrome.open_global_panel(4)
    assert w.settings_categories.isVisible()
    w.settings_categories.setCurrentIndex(3)
    assert w.workspace_migration_button.isVisible()
    w.workspace_chrome._dialogs[4].hide()


def test_picker_mouse_click_can_rebuild_project_rows_safely(window, app):
    w = window
    pid, node, _ = make_project(w)
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
        assert w.current_creative_node_id == node.id
        assert w.center_result_stack.currentWidget() is w.creative_workspace.planning_scroll
