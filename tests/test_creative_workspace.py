from types import SimpleNamespace
from unittest.mock import patch

from test_filter_workflow import app, library, window, apply
from eidory.core.search_filters import FilterGroup, FilterPlan, SearchFilter
from eidory.core.creative_workspace import CreativeWorkspaceStore
from eidory.core.creative_export import export_creative_project


def test_gallery_keyboard_selection_and_range(window):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    grid = window.grid_view
    grid.set_images(window.store.list_images(limit=None))
    QTest.keyClick(grid, Qt.Key.Key_Home)
    assert len(grid.selected_images()) == 1
    first = grid.current_image().id
    QTest.keyClick(grid, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    assert len(grid.selected_images()) == 2
    assert first in grid.selected_image_ids()
    QTest.keyClick(grid, Qt.Key.Key_End)
    assert len(grid.selected_images()) == 1
    assert grid.current_image() == grid.images()[-1]


def test_probe_threshold_is_visible_only_for_probe_results(window, app):
    w = window
    w.show()
    w.current_result_mode = 'inspiration'
    w._show_gallery_view()
    w.filter_workflow.sync()
    app.processEvents()
    assert w.score_threshold_slider.isVisible()
    w.creative_workspace.show_planning()
    assert not w.score_threshold_slider.isVisible()
    w.current_result_mode = 'search_chain'
    w._show_gallery_view()
    w.filter_workflow.sync()
    assert not w.score_threshold_slider.isVisible()


def test_project_copy_stages_nodes_and_respects_lock(window):
    w = window
    pid, node = project(w)
    original = w.store.get_creative_node(node.id).note
    result = SimpleNamespace(copy_text='故事的创作文案', nodes=[SimpleNamespace(title=node.title, note='古城中的木质商铺在日落时点起油灯。', search_query='古城 木质商铺 油灯')])
    w._handle_creative_project_copy_done(pid, node.id, False, result, 'fixture')
    assert w.store.get_creative_node(node.id).note == original
    assert w.creative_workspace.store.node(node.id)['proposal_note']
    w.creative_workspace.store.save_node(node.id, locked=1, proposal_note='', proposal_query='')
    w._handle_creative_project_copy_done(pid, node.id, False, result, 'fixture')
    assert not w.creative_workspace.store.node(node.id)['proposal_note']


def project(w, title='沙漠古城', extra='只能使用暖色'):
    pid = w._create_creative_project_from_seed(title=title, brief=title, extra=extra)
    node = w.store.list_creative_nodes(pid)[1]
    w._load_creative_project(pid, select_node_id=node.id)
    return pid, node


def test_project_context_is_isolated_and_persistent(window):
    w = window
    a, node = project(w)
    project(w, '极地空间站', '只能使用冷色')
    w._load_creative_project(a, select_node_id=node.id)
    assert w.creative_brief_input.toPlainText() == '沙漠古城'
    assert w._creative_node_ai_context(w.store.get_creative_project(a)) == ('沙漠古城', '只能使用暖色')
    w.creative_extra_input.setPlainText('夜晚，暖色')
    w.creative_workspace.show_inspiration()
    w.inspiration_brief_input.setPlainText('自由探索海底')
    assert w._creative_node_ai_context(w.store.get_creative_project(a)) == ('沙漠古城', '夜晚，暖色')
    w._load_creative_project(a, select_node_id=node.id)
    assert w.creative_brief_input.toPlainText() == '沙漠古城'
    assert w.inspiration_brief_input.toPlainText() == '自由探索海底'
    assert CreativeWorkspaceStore(w.store).project(a)['extra'] == '夜晚，暖色'


def test_node_search_uses_visible_scope_and_live_filter(window):
    w = window
    pid, node = project(w)
    images = w.store.list_images(limit=None)[:2]
    allowed = {im.id for im in images}
    w.current_result_mode = 'search_chain'
    w.filter_workflow.scope = {'kind': 'snapshot', 'image_ids': sorted(allowed), 'label': '明确指定的范围'}
    w.creative_node_query_input.setText('飞机')
    w._search_selected_creative_node()
    w._poll_events()
    assert w.current_result_mode == 'search_chain'
    assert all(scope == allowed for scope in w.search_service.scopes)
    assert w.filter_workflow.plan().include[0].filters[0] == SearchFilter('semantic', '飞机', 50)
    assert w.filter_panel._live_available


def test_ai_suggestion_requires_adoption_and_can_restore(window):
    w = window
    pid, node = project(w)
    original = w.store.get_creative_node(node.id)
    suggestion = SimpleNamespace(note='沙漠中高耸的石砌城门，暖色夕阳下留有深色裂缝。', search_query='暖色 沙漠 石砌城门')
    w._handle_creative_node_note_done(node.id, '', suggestion, 'fixture')
    assert w.store.get_creative_node(node.id).note == original.note
    assert '石砌' in w.creative_workspace.store.node(node.id)['proposal_note']
    w.creative_workspace.accept_note()
    assert '石砌' in w.store.get_creative_node(node.id).note
    w.creative_workspace.restore_note()
    assert w.store.get_creative_node(node.id).note == original.note
    w.creative_workspace.toggle_lock(True)
    with patch.object(w, '_make_llm_provider') as provider:
        w._generate_creative_children_for_selected_node()
        provider.assert_not_called()


def test_reference_notes_are_project_specific_and_exported(window, tmp_path):
    w = window
    pid, node = project(w)
    image = w.store.list_images(limit=None)[0]
    w.store.add_images_to_creative_node(node.id, [image.id])
    w.creative_workspace.store.save_reference_note(node.id, image.id, '只参考光线 <script>')
    other, other_node = project(w, '空间站', '冷色')
    w.store.add_images_to_creative_node(other_node.id, [image.id])
    assert not w.creative_workspace.store.reference_note(other_node.id, image.id)
    target = export_creative_project(w.store, pid, tmp_path)
    markup = (target / 'index.html').read_text()
    assert '只参考光线 &lt;script&gt;' in markup
    assert '<script>' not in markup
    assert len(list((target / 'images').iterdir())) == 1
    assert image.note == w.store.get_image(image.id).note


def test_candidate_filters_survive_collected_and_board_roundtrip(window):
    w = window
    pid, node = project(w)
    w.creative_node_query_input.setText('飞机')
    w._search_selected_creative_node()
    w._poll_events()
    plan = w.filter_workflow.plan()
    names = [im.file_name for im in w.grid_view.images()]
    selected = [im.id for im in w.grid_view.images()[:2]]
    w.grid_view.select_image_ids(selected)
    w._save_selection_to_current_creative_node()
    w.creative_workspace.show_collected()
    assert set(im.id for im in w.grid_view.images()) == set(selected)
    w.creative_workspace.show_board()
    w.creative_workspace.show_candidates()
    w._poll_events()
    assert w.filter_workflow.plan() == plan
    assert [im.file_name for im in w.grid_view.images()] == names
    assert w.grid_view.selected_image_ids() == selected


def test_node_search_marks_project_references_through_live_filter_and_refresh(window):
    w = window
    pid, node_a = project(w)
    images = {im.file_name: im for im in w.store.list_images(limit=None)}
    saved = images['red-plane.png']
    unsaved = images['blue-plane.png']
    w.grid_view.set_images(list(images.values()))
    w.grid_view.select_image_ids([saved.id])
    w._save_selection_to_current_creative_node()
    node_b = w.store.list_creative_nodes(pid)[2]
    w._load_creative_project(pid, select_node_id=node_b.id)
    w.creative_node_query_input.setText('飞机')
    w._search_selected_creative_node()
    w._poll_events()
    assert saved.id in {im.id for im in w.grid_view.images()}
    assert w.grid_view._project_references == {saved.id: [node_a.title]}
    assert '本项目已选 · 所属节点：' + node_a.title in w.grid_view.image_tooltip(saved)
    w.grid_view.select_image_ids([unsaved.id])
    assert '本项目已选' not in w.grid_view.image_tooltip(unsaved)

    slider = w.filter_panel.strength_sliders['include', 0, 0]
    for strength, count in [(95, 1), (0, 4)]:
        slider.setValue(strength)
        w.filter_workflow.finish_strength()
        assert len(w.grid_view.images()) == count
        assert w.grid_view._project_references == {saved.id: [node_a.title]}
    w._on_sort_changed()
    assert w.grid_view._project_references == {saved.id: [node_a.title]}


def test_project_reference_membership_updates_on_reuse_remove_rename_and_switch(window):
    w = window
    pid, node_a = project(w)
    node_b = w.store.list_creative_nodes(pid)[2]
    image = w.store.list_images(limit=None)[0]
    for node in [node_a, node_b]:
        w._load_creative_project(pid, select_node_id=node.id)
        w.grid_view.set_images([image])
        w.grid_view.select_image_ids([image.id])
        w._save_selection_to_current_creative_node()
    assert set(w.grid_view._project_references[image.id]) == {node_a.title, node_b.title}
    root = w.store.creative_root_node_id(pid)
    assert w.store.creative_node_image_ids(root, include_descendants=True) == [image.id]

    w.store.update_creative_node(node_a.id, title='改名后的节点')
    w._load_creative_project(pid, select_node_id=node_a.id)
    assert '改名后的节点' in w.grid_view.image_tooltip(image)
    assert node_a.title not in w.grid_view._project_references[image.id]
    # Keep the original intent label for existing consumers of that API.
    assert node_a.title in w.store.creative_node_image_badges(pid)[image.id]
    w.creative_workspace.show_collected()
    w.grid_view.select_image_ids([image.id])
    w._remove_selection_from_current_creative_node()
    assert w.grid_view._project_references == {image.id: [node_b.title]}

    project(w, '另一个创作项目')
    assert w.grid_view._project_references == {}
    w._load_creative_project(pid, select_node_id=node_b.id)
    assert w.grid_view._project_references == {image.id: [node_b.title]}
    w._load_creative_project(None)
    assert w.grid_view._project_references == {}
    w._load_creative_project(pid, select_node_id=node_b.id)
    w.creative_workspace.show_collected()
    w.grid_view.select_image_ids([image.id])
    w._remove_selection_from_current_creative_node()
    assert w.grid_view._project_references == {}


def test_board_edit_undo_redo_and_text_survive_save(window, tmp_path):
    w = window
    pid, node = project(w)
    images = w.store.list_images(limit=None)[:2]
    w.store.add_images_to_creative_node(node.id, [im.id for im in images])
    w._show_current_creative_board()
    board = w.project_board_view
    initial = board.layout_payload()
    item = board._image_items[images[0].id]
    item.setPos(item.x() + 140, item.y() + 50)
    moved = board.layout_payload()
    assert moved != initial
    assert board.undo_layout()
    assert board.layout_payload() == initial
    assert board.redo_layout()
    assert board.layout_payload() == moved
    board._add_text('参考暖色光线', 80, 60, heading=True)
    board._schedule_layout_changed()
    payload = board.layout_payload()
    w._save_current_board_layout_if_needed(force=True)
    w._show_gallery_view()
    w._show_current_creative_board()
    assert board.layout_payload()['notes'] == payload['notes']
    destination = export_creative_project(w.store, pid, tmp_path, render_board=w.creative_workspace.render_board)
    assert (destination / 'boards' / f'{node.id}.png').is_file()


def test_locked_node_rejects_delayed_suggestion(window):
    w = window
    pid, node = project(w)
    original = w.store.get_creative_node(node.id).note
    w.creative_workspace.toggle_lock(True)
    w._handle_creative_node_note_done(node.id, '', SimpleNamespace(note='新的暖色城门结构和粗糙石材。', search_query='城门 石材'), 'fixture')
    assert w.store.get_creative_node(node.id).note == original
    assert not w.creative_workspace.store.node(node.id)['proposal_note']
