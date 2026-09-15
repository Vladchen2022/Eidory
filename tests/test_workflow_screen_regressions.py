from dataclasses import replace
import json
from types import SimpleNamespace

from test_filter_workflow import app, library, window, apply
from eidory.core.search_filters import FilterPlan, FilterGroup, SearchFilter


def test_toolbar_sort_keeps_target_subset_badges_and_selection(window, library):
    w = window
    images = library[2]
    def search(text, **kwargs):
        matches = [replace(im, score=.9 if im.file_name.startswith(text) else .1) for im in images]
        return SimpleNamespace(images=matches, searchable_count=4, indexed_count=4, candidate_limit=4)
    w.search_service.semantic_search = search
    apply(w, FilterPlan((FilterGroup((SearchFilter('semantic', 'red', 50), SearchFilter('semantic', 'blue', 50)), 'any'),)))
    ui = w.workflow_upgrade
    ui.result_mode.setCurrentIndex(ui.result_mode.findData('red'))
    expected = {im.id for im in images if im.file_name.startswith('red')}
    assert {im.id for im in w.grid_view.images()} == expected
    selected = next(iter(expected))
    w.grid_view.select_image_ids([selected])
    w.sort_order_combo.setCurrentIndex(1)
    w._on_sort_changed()
    assert ui.result_mode.currentData() == 'red'
    assert {im.id for im in w.grid_view.images()} == expected
    assert w.grid_view.selected_image_ids() == [selected]
    assert all('命中：red' in w.grid_view._badges_by_image_id[i] for i in expected)
    ui.result_mode.setCurrentIndex(ui.result_mode.findData('__toolbar__'))
    assert len(w.grid_view.images()) == 4


def test_shared_reference_grade_is_scoped_in_collection_and_board(window, library):
    w = window
    pid = w._create_creative_project_from_seed(title='跨节点验收', brief='研发室', extra='')
    root, a, b = w.store.list_creative_nodes(pid)[:3]
    image = library[2][0]
    for node in (a, b):
        w.store.add_images_to_creative_node(node.id, [image.id])
    state = w.workflow_upgrade.state
    state.set_decision(a.id, image.id, 'primary', '造型')
    state.set_decision(b.id, image.id, 'unused', '光线')
    w.creative_workspace.store.save_reference_note(a.id, image.id, '仅用于人物造型')
    w.creative_workspace.store.save_reference_note(b.id, image.id, '仅用于空间光线')
    w._load_creative_project(pid, select_node_id=b.id)
    w.creative_workspace.show_collected()
    ui = w.workflow_upgrade
    ui.collection_mode.setCurrentIndex(ui.collection_mode.findData('unused'))
    assert [im.id for im in w.grid_view.images()] == [image.id]
    badges = w.grid_view._badges_by_image_id[image.id]
    assert '暂不采用' in badges and '光线' in badges
    assert '主参考' not in badges and '造型' not in badges
    w._on_sort_changed()
    assert [im.id for im in w.grid_view.images()] == [image.id]
    w._show_creative_board_for_node(b.id, force=True)
    assert '主参考' not in w.project_board_view._board_badges[image.id]
    assert '仅用于人物造型' not in w.project_board_view._board_badges[image.id]
    assert '仅用于空间光线' in w.project_board_view._board_badges[image.id]
    w._load_creative_project(pid, select_node_id=a.id)
    w.creative_workspace.show_collected()
    ui.collection_mode.setCurrentIndex(ui.collection_mode.findData('primary'))
    assert [im.id for im in w.grid_view.images()] == [image.id]
    assert '暂不采用' not in w.grid_view._badges_by_image_id[image.id]
    # At the common ancestor, any primary reference takes precedence.
    badges = {image.id: [b.title, a.title]}
    ui.decorate_references(pid, badges, root.id)
    assert '主参考' in badges[image.id] and '暂不采用' not in badges[image.id]
    assert badges[image.id][0] == a.title


def test_returning_from_collected_to_cached_search_selects_only_find_stage(window):
    w = window
    pid = w._create_creative_project_from_seed(title='页面状态', brief='实验室', extra='')
    node = w.store.list_creative_nodes(pid)[1]
    w._load_creative_project(pid, select_node_id=node.id)
    w.creative_node_query_input.setText('red')
    w._search_selected_creative_node()
    w._poll_events()
    w.creative_workspace.show_collected()
    assert w.workspace_chrome.stage_buttons[2].isChecked()
    w.creative_workspace.show_candidates()
    w._poll_events()
    assert [b.isChecked() for b in w.workspace_chrome.stage_buttons] == [False, True, False, False]


def test_new_board_images_are_saved_without_dragging_existing_layout(window, library):
    w = window
    pid = w._create_creative_project_from_seed(title='新增参考布局', brief='研发室', extra='')
    root, node = w.store.list_creative_nodes(pid)[:2]
    first, second = library[2][:2]
    w.store.add_images_to_creative_node(node.id, [first.id])
    w._show_creative_board_for_node(root.id, force=True)
    saved = json.loads(w.store.get_creative_node_board_layout(root.id))
    saved['items'][str(first.id)].update(x=700.0, y=500.0)
    w.store.save_creative_node_board_layout(root.id, json.dumps(saved))
    w.store.add_images_to_creative_node(node.id, [second.id])
    w._show_creative_board_for_node(root.id, force=True)
    after = json.loads(w.store.get_creative_node_board_layout(root.id))
    assert set(after['items']) == {str(first.id), str(second.id)}
    assert after['items'][str(first.id)] == saved['items'][str(first.id)]
    assert after == w.project_board_view.layout_payload()


def test_settings_retranslation_preserves_node_role_and_lock(window):
    w = window
    pid = w._create_creative_project_from_seed(title='按钮职责', brief='实验室', extra='')
    node = next(n for n in w.store.list_creative_nodes(pid) if n.title == '地点')
    w._load_creative_project(pid, select_node_id=node.id)
    w._apply_runtime_language_settings()
    assert w.generate_creative_children_button.text() == '补充空间组成'
    w.creative_workspace.store.save_node(node.id, locked=1)
    w._apply_runtime_language_settings()
    assert not w.generate_creative_children_button.isEnabled()
