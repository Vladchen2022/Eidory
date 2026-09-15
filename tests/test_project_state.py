import json

import pytest
from PIL import Image

from eidory.core.metadata_store import MetadataStore
from eidory.core.project_state import ProjectStateStore, capture_project
from eidory.core.project_package import export_project_package, import_project_package
from eidory.core.scanner import ImageScanner
from eidory.core.thumbnailer import Thumbnailer


@pytest.fixture
def project(tmp_path):
    store = MetadataStore(tmp_path / 'source.sqlite3')
    store.initialize()
    folder = tmp_path / 'images'
    folder.mkdir()
    Image.new('RGB', (80, 60), 'blue').save(folder / 'robot.png')
    scan = ImageScanner(store, Thumbnailer(tmp_path / 'thumbs')).scan_folder(str(folder))
    pid = store.create_creative_project(title='研发室', brief='三位工程师')
    node = store.create_creative_node(project_id=pid, parent_id=store.list_creative_nodes(pid)[0].id, title='人物', note='保留三个人', search_query='工程师')
    store.add_images_to_creative_node(node, scan.image_ids)
    return store, pid, node, scan.image_ids[0]


def test_snapshot_restores_whole_project_and_keeps_recovery_version(project):
    store, pid, node, image = project
    state = ProjectStateStore(store)
    state.set_decision(node, image, 'primary', '造型')
    state.set_options(node, reference_requirement='resolved', search_targets_json='[{"query":"工程师"}]')
    from eidory.core.creative_workspace import CreativeWorkspaceStore
    workflow = CreativeWorkspaceStore(store)
    workflow.save_node(node, locked=1, proposal_note='待采纳文本', entries_json='{"rows":[]}')
    store.save_creative_node_board_layout(node, '{"items":{},"notes":[{"text":"方案一"}]}')
    version = state.snapshot(pid, '方案一')
    store.update_creative_project_details(pid, brief='四个人')
    state.set_decision(node, image, 'unused', '动作')
    state.restore(pid, version)
    assert store.get_creative_project(pid).brief == '三位工程师'
    assert state.decision(node, image)['status'] == 'primary'
    assert workflow.node(node)['locked'] == 1
    assert workflow.node(node)['proposal_note'] == '待采纳文本'
    assert len(state.versions(pid)) == 2
    state.restore(pid, state.versions(pid)[0]['id'])
    assert store.get_creative_project(pid).brief == '四个人'


def test_editable_package_roundtrip_with_id_remap(project, tmp_path):
    store, pid, node, image = project
    state = ProjectStateStore(store)
    state.set_decision(node, image, 'primary', '造型')
    from eidory.core.creative_workspace import CreativeWorkspaceStore
    CreativeWorkspaceStore(store).save_node(node, locked=1, entries_json='{"rows":[{"id":"r1","name":"主管","visual":"白衣","enabled":false}]}', proposal_note='待采纳')
    store.save_creative_node_board_layout(node, json.dumps({'items': {str(image): {'x': 50}}, 'sections': [{'image_ids': [image]}]}))
    package = export_project_package(store, pid, tmp_path)
    target = MetadataStore(tmp_path / 'target/db.sqlite3')
    target.initialize()
    # Occupy original IDs to prove that no foreign reference survives the import.
    target.create_creative_project(title='另一个项目')
    imported, missing = import_project_package(target, package, tmp_path / 'target')
    assert not missing
    new_node = next(n.id for n in target.list_creative_nodes(imported) if n.title == '人物')
    assert new_node != node
    new_image = target.creative_node_image_ids(new_node)[0]
    assert target.get_image(new_image).file_path.startswith(str(tmp_path / 'target/imported-projects'))
    assert ProjectStateStore(target).decision(new_node, new_image)['status'] == 'primary'
    assert CreativeWorkspaceStore(target).node(new_node)['proposal_note'] == '待采纳'
    assert json.loads(CreativeWorkspaceStore(target).node(new_node)['entries_json'])['rows'][0]['enabled'] is False
    layout = json.loads(target.get_creative_node_board_layout(new_node))
    assert str(new_image) in layout['items']
    assert layout['sections'][0]['image_ids'] == [new_image]


def test_corrupt_package_is_rejected_before_registering_project(project, tmp_path):
    store, pid, _, _ = project
    package = export_project_package(store, pid, tmp_path)
    next((package / 'media').iterdir()).write_bytes(b'broken')
    count = len(store.list_creative_projects())
    with pytest.raises(ValueError, match='校验'):
        import_project_package(store, package, tmp_path / 'data')
    assert len(store.list_creative_projects()) == count


def test_context_change_marks_review_without_editing_locked_content(project):
    store, pid, node, _ = project
    state = ProjectStateStore(store)
    state.set_options(node, reviewed_context=state.context_key(pid), reference_requirement='none')
    store.update_creative_project_details(pid, brief='夜间的研发室')
    row = next(r for r in state.overview(pid) if r['id'] == node)
    assert row['needs_review'] == 1 and row['requirement'] == 'none'
    assert store.get_creative_node(node).note == '保留三个人'


def test_package_preserves_named_versions(project, tmp_path):
    store, pid, node, image = project
    service = ProjectStateStore(store)
    service.snapshot(pid, '白天方案')
    store.update_creative_project_details(pid, brief='夜晚')
    package = export_project_package(store, pid, tmp_path)
    new_id, _ = import_project_package(store, package, tmp_path / 'destination')
    versions = service.versions(new_id)
    assert len(versions) == 1
    service.restore(new_id, versions[0]['id'])
    assert store.get_creative_project(new_id).brief == '三位工程师'
    assert store.get_creative_project(pid).brief == '夜晚'


def test_restore_remaps_nodes_when_old_ids_belong_to_another_project(project):
    store, pid, node, image = project
    state = ProjectStateStore(store)
    state.set_decision(node, image, 'primary', '造型')
    version = state.snapshot(pid, '保存后节点被移动')
    other = store.create_creative_project(title='其他项目')
    with store.connect() as c:
        c.execute('UPDATE creative_nodes SET project_id=?, parent_id=NULL WHERE id=?', (other, node))
    state.restore(pid, version)
    restored = next(n for n in store.list_creative_nodes(pid) if n.title == '人物')
    assert restored.id != node
    assert state.decision(restored.id, image)['status'] == 'primary'
    assert store.get_creative_node(node).project_id == other
    assert restored.parent_id in {n.id for n in store.list_creative_nodes(pid)}
