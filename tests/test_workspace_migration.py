import gzip
import json
import os
from pathlib import Path
import shutil
import sqlite3
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from eidory.config import AppPaths
from eidory.core.metadata_store import MetadataStore
from eidory.core.scanner import ImageScanner
from eidory.core.thumbnailer import Thumbnailer
from eidory.core.upscale_workflow import UpscaleWorkflow
from eidory.core.vector_index import VectorIndex
from eidory.core.workspace_package import (MANIFEST, MigrationCancelled, _json_write,
    database_counts, export_workspace, inspect_package, prepare_export, restore_workspace, sha256)
from eidory.core import workspace_selection as selection


@pytest.fixture
def workspace(tmp_path):
    data = tmp_path/'old'/'data'
    paths = AppPaths(data, data/'thumbnails', data/'eidory.sqlite3', data/'logs')
    paths.ensure()
    store = MetadataStore(paths.database_path)
    store.initialize()
    photos = tmp_path/'old'/'Pictures'/'图库'
    photos.mkdir(parents=True)
    for name, color in [('A', 'red'), ('B', 'blue')]:
        Image.new('RGB', (60, 40), color).save(photos/(name+'.png'))
    ImageScanner(store, Thumbnailer(paths.thumbnail_dir)).scan_folder(str(photos))
    with store.connect() as c:
        images = [dict(r) for r in c.execute('SELECT * FROM images ORDER BY id')]
        for i, image in enumerate(images):
            vec = np.zeros(512, dtype=np.float32); vec[i] = 1
            c.execute("INSERT INTO embeddings(image_id,model_name,model_revision,embedding_dim,vector_blob,status,created_at,updated_at) VALUES(?, 'jinaai/jina-clip-v2','main',512,?,'ready','now','now')", (image['id'], vec.tobytes()))
            c.execute("UPDATE images SET embedding_status='ready',note='原来的笔记' WHERE id=?", (image['id'],))
            c.execute("INSERT INTO ai_vision_tags(image_id,analysis_type,provider_name,model_name,prompt_version,status,notes,source_modified_time_ns,created_at,updated_at) VALUES(?,'scene','local','test','v1','ready','保留标签与细节',?,'now','now')", (image['id'], image['modified_time_ns']))
            c.execute("INSERT INTO image_preview_annotations VALUES(?,?,?,?)", (image['id'], '{"strokes":[]}', image['file_size'], image['modified_time_ns']))
        c.execute("INSERT INTO creative_projects(id,title,created_at,updated_at) VALUES(7,'迁移项目','now','now')")
        c.execute("INSERT INTO creative_nodes(id,project_id,title,created_at,updated_at) VALUES(8,7,'节点','now','now')")
        c.execute("INSERT INTO creative_node_images(node_id,image_id,created_at) VALUES(8,?,'now')", (images[0]['id'],))
        c.execute("INSERT INTO creative_project_context(project_id,scope_json) VALUES(7,?)", (json.dumps({'folder_paths':[str(photos)]}),))
    store.set_setting('ui.file_watch_enabled', '1')
    store.set_setting('llm.openai.api_key', 'SENTINEL-SECRET-MUST-NOT-LEAK')
    store.set_setting('search.excluded_folder_prefixes', json.dumps([str(photos)]))
    return paths, store, images, photos


def exported(workspace, tmp_path, **options):
    paths, *_ = workspace
    package = tmp_path/'package'
    manifest = export_workspace(paths, package, include_model=False, **options)
    return package, manifest


def table(path, name, columns='*'):
    with sqlite3.connect(path) as c:
        return c.execute(f'SELECT {columns} FROM {name} ORDER BY rowid').fetchall()


def test_full_round_trip_preserves_ids_vectors_labels_projects_and_source(workspace, tmp_path):
    paths, store, images, photos = workspace
    original = sha256(paths.database_path)
    package, manifest = exported(workspace, tmp_path)
    data, report = restore_workspace(package, tmp_path/'new-computer')
    db = data/'eidory.sqlite3'
    for name in ('embeddings', 'creative_projects', 'creative_nodes', 'creative_node_images', 'ai_vision_tags', 'image_preview_annotations'):
        assert table(paths.database_path, name) == table(db, name)
    restored = MetadataStore(db)
    for image in images:
        result = restored.get_image(image['id'])
        assert result.file_path.startswith(str(tmp_path/'new-computer'))
        assert Path(result.file_path).parent.name == photos.name
        assert Path(result.file_path).read_bytes() == Path(image['file_path']).read_bytes()
        assert Path(result.thumbnail_path).exists()
        assert result.embedding_status == 'ready'
    assert report['counts']['ready_vectors'] == 2
    assert sha256(paths.database_path) == original
    assert restored.get_setting('ui.file_watch_enabled') == '0'
    assert restored.get_setting('upscale.queue_paused') == '1'
    assert table(db, 'creative_project_context', 'scope_json')[0][0].find(str(photos)) == -1
    assert str(photos) not in restored.get_setting('search.excluded_folder_prefixes')
    index = VectorIndex(restored, model_name='jinaai/jina-clip-v2', model_revision='main', embedding_dim=512)
    index.reload()
    query = np.zeros(512, dtype=np.float32); query[0] = 1
    assert index.search(query, 2)[0] == (images[0]['id'], 1.0)


def test_keys_are_excluded_including_sqlite_free_pages(workspace, tmp_path):
    package, _ = exported(workspace, tmp_path)
    assert b'SENTINEL-SECRET' not in (package/'database.sqlite3').read_bytes()
    assert b'SENTINEL-SECRET' not in (package/MANIFEST).read_bytes()
    assert workspace[1].get_setting('llm.openai.api_key') == 'SENTINEL-SECRET-MUST-NOT-LEAK'


def test_keys_can_be_explicitly_included(workspace, tmp_path):
    package, _ = exported(workspace, tmp_path, include_secrets=True)
    assert MetadataStore(package/'database.sqlite3').get_setting('llm.openai.api_key') == 'SENTINEL-SECRET-MUST-NOT-LEAK'


def test_link_mode_accepts_identical_bytes_with_new_timestamps(workspace, tmp_path):
    paths, _, images, photos = workspace
    package, manifest = exported(workspace, tmp_path, include_media=False)
    linked = tmp_path/'new-images'; shutil.copytree(photos, linked)
    for p in linked.iterdir():
        os.utime(p, ns=(1111111111000000000, 1111111111000000000))
    data, _ = restore_workspace(package, tmp_path/'restored', root_mapping={manifest['roots'][0]['id']:str(linked)})
    db = data/'eidory.sqlite3'
    assert table(db, 'embeddings') == table(paths.database_path, 'embeddings')
    assert table(db, 'ai_vision_tags', 'notes,status') == [('保留标签与细节','ready')]*2
    assert table(db, 'ai_vision_tags', 'source_modified_time_ns') == [(1111111111000000000,)]*2
    assert table(db, 'image_preview_annotations', 'modified_time_ns') == [(1111111111000000000,)]*2
    assert not (tmp_path/'restored/media').exists()


def test_link_mode_rejects_same_size_and_timestamp_with_different_contents(workspace, tmp_path):
    _, _, _, photos = workspace
    package, manifest = exported(workspace, tmp_path, include_media=False)
    linked = tmp_path/'new-images'; shutil.copytree(photos, linked)
    p = linked/'A.png'; stat = p.stat(); raw = bytearray(p.read_bytes()); raw[-1] ^= 1; p.write_bytes(raw)
    os.utime(p, ns=(stat.st_atime_ns,stat.st_mtime_ns))
    target = tmp_path/'restored'
    with pytest.raises(ValueError, match='内容与旧电脑不一致'):
        restore_workspace(package, target, root_mapping={manifest['roots'][0]['id']:str(linked)})
    assert not target.exists()


def test_export_pause_and_resume_uses_same_database_snapshot(workspace, tmp_path):
    paths, store, *_ = workspace
    package = tmp_path/'package'
    seen = []
    with pytest.raises(MigrationCancelled):
        export_workspace(paths, package, include_model=False, progress=lambda *a:seen.append(a), cancel=lambda:bool(seen))
    before = sha256(package/'database.sqlite3')
    store.set_setting('after_snapshot', 'this must not migrate')
    manifest = export_workspace(paths, package, include_model=False)
    assert manifest['complete'] and sha256(package/'database.sqlite3') == before
    assert MetadataStore(package/'database.sqlite3').get_setting('after_snapshot') is None


def test_partial_package_cannot_be_imported(workspace, tmp_path):
    package = tmp_path/'package'
    prepare_export(workspace[0], package, include_model=False)
    with pytest.raises(ValueError, match='尚未导出完成'):
        restore_workspace(package, tmp_path/'restore')


def test_changed_source_blocks_resume_without_overwriting_snapshot(workspace, tmp_path):
    package = tmp_path/'package'
    prepare_export(workspace[0], package, include_model=False)
    (workspace[3]/'A.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='发生变化'):
        export_workspace(workspace[0], package, include_model=False)
    assert not json.loads((package/MANIFEST).read_text())['complete']


def test_restore_pause_and_resume(workspace, tmp_path):
    package, _ = exported(workspace, tmp_path)
    target = tmp_path/'restored'; seen=[]
    with pytest.raises(MigrationCancelled):
        restore_workspace(package, target, progress=lambda *a:seen.append(a), cancel=lambda:bool(seen))
    assert not target.exists() and (tmp_path/'restored.incomplete').exists()
    data, _ = restore_workspace(package, target)
    assert database_counts(data/'eidory.sqlite3')['images'] == 2


def test_corrupt_package_file_never_activates(workspace, tmp_path):
    package, manifest = exported(workspace, tmp_path)
    entry = manifest['files'][0]; p=package/entry['path'];raw=bytearray(p.read_bytes());raw[-1]^=1;p.write_bytes(raw)
    with pytest.raises(ValueError, match='校验未通过'):
        restore_workspace(package, tmp_path/'restored')
    assert not (tmp_path/'restored').exists()


@pytest.mark.parametrize('malicious', ['../escape', '/tmp/escape', 'data/../../escape'])
def test_manifest_path_traversal_rejected(workspace, tmp_path, malicious):
    package, manifest = exported(workspace, tmp_path)
    manifest['files'][0]['path'] = malicious
    _json_write(package/MANIFEST, manifest)
    with pytest.raises(ValueError):
        restore_workspace(package, tmp_path/'restored')


def test_symlink_escape_rejected(workspace, tmp_path):
    package, manifest = exported(workspace, tmp_path)
    file=package/manifest['files'][0]['path']; file.unlink(); file.symlink_to(workspace[3]/'A.png')
    with pytest.raises(ValueError, match='符号链接'):
        inspect_package(package)


def test_missing_image_retains_vector_and_links(workspace, tmp_path):
    paths, _, images, photos = workspace
    (photos/'A.png').unlink()
    package, manifest = exported(workspace, tmp_path)
    assert manifest['missing'] == 1
    data, _ = restore_workspace(package, tmp_path/'restored')
    assert table(data/'eidory.sqlite3', 'embeddings') == table(paths.database_path, 'embeddings')
    assert table(data/'eidory.sqlite3', 'creative_node_images') == table(paths.database_path, 'creative_node_images')
    assert MetadataStore(data/'eidory.sqlite3').get_image(images[0]['id']).is_missing


def test_candidates_backups_and_old_server_tasks_migrate(workspace, tmp_path):
    paths, store, images, _ = workspace
    candidate=paths.data_dir/'image-upscale/candidates/0/test.png';candidate.parent.mkdir(parents=True);candidate.write_bytes(b'candidate')
    backup=paths.data_dir/'image-upscale/sources/test/original.png';backup.parent.mkdir(parents=True);backup.write_bytes(b'original')
    jobs, _ = UpscaleWorkflow(store, paths).enqueue([i['id'] for i in images],2,'http://127.0.0.1:8188')
    jobs[0].update(state='ready', file_path=str(candidate),original_path=str(backup),source_path=images[0]['file_path'])
    jobs[1].update(state='running', prompt_id='old-device-id')
    for job in jobs:
        UpscaleWorkflow(store,paths).save(job)
    package, _ = exported(workspace, tmp_path)
    data, _ = restore_workspace(package,tmp_path/'restored')
    result={r[0]['id']:r[0] for r in [(json.loads(row[0]),) for row in table(data/'eidory.sqlite3','image_upscale_jobs','payload')]}
    assert result[jobs[0]['id']]['state']=='ready'
    assert Path(result[jobs[0]['id']]['file_path']).read_bytes()==b'candidate'
    assert Path(result[jobs[0]['id']]['original_path']).read_bytes()==b'original'
    assert result[jobs[1]['id']]['state']=='cancelled'
    assert result[jobs[1]['id']]['migration_previous_state']=='running'
    assert [result[j['id']]['batch_id'] for j in jobs] == [j['batch_id'] for j in jobs]


def test_compressed_recovery_archive_paths_are_rebound(workspace,tmp_path):
    paths, _, images, _ = workspace
    root=paths.data_dir/'recovery-records';root.mkdir()
    payload={'snapshot':{'images':images},'backup_dir':str(paths.data_dir/'undo-removal/test')}
    (root/'test.json.gz').write_bytes(gzip.compress(json.dumps(payload).encode()))
    package,_=exported(workspace,tmp_path)
    data,_=restore_workspace(package,tmp_path/'restored')
    restored=json.loads(gzip.decompress((data/'recovery-records/test.json.gz').read_bytes()))
    assert restored['backup_dir']==str(data/'undo-removal/test')
    assert restored['snapshot']['images'][0]['file_path'].startswith(str(tmp_path/'restored/media'))


def test_no_overwrite_of_existing_destination(workspace,tmp_path):
    package,_=exported(workspace,tmp_path)
    target=tmp_path/'existing';target.mkdir();(target/'keep.txt').write_text('keep')
    with pytest.raises(ValueError,match='不会覆盖'):
        restore_workspace(package,target)
    assert (target/'keep.txt').read_text()=='keep'


def test_database_failure_keeps_old_and_can_retry(workspace,tmp_path):
    paths, *_=workspace
    package,_=exported(workspace,tmp_path)
    old=sha256(paths.database_path)
    with patch('eidory.core.workspace_package._rewrite_database',side_effect=OSError('disk test')):
        with pytest.raises(OSError):restore_workspace(package,tmp_path/'restored')
    assert sha256(paths.database_path)==old and not (tmp_path/'restored').exists()
    data,_=restore_workspace(package,tmp_path/'restored')
    assert data.exists()


def test_workspace_switch_and_fallback_are_nondestructive(workspace,tmp_path,monkeypatch):
    paths,*_=workspace
    monkeypatch.setenv('EIDORY_DATA_DIR',str(paths.data_dir))
    package,_=exported(workspace,tmp_path)
    data,_=restore_workspace(package,tmp_path/'restored')
    selection.select_workspace(data,paths.data_dir)
    assert AppPaths.default().data_dir==data
    assert selection.selection()['history']==[str(paths.data_dir)]
    data.rename(data.with_name('unplugged-data'))
    assert AppPaths.default().data_dir==paths.data_dir
    assert '回退' in selection.startup_warning


def test_disk_full_detected_before_copy(workspace,tmp_path):
    package,_=exported(workspace,tmp_path)
    with patch('eidory.core.workspace_package.shutil.disk_usage',return_value=shutil._ntuple_diskusage(100,99,1)):
        with pytest.raises(ValueError,match='空间不足'):
            restore_workspace(package,tmp_path/'restored')
    assert not (tmp_path/'restored').exists()


def test_manifest_cannot_omit_images_from_verification(workspace,tmp_path):
    package,manifest=exported(workspace,tmp_path)
    manifest['images'].pop()
    _json_write(package/MANIFEST,manifest)
    with pytest.raises(ValueError,match='清单与数据库不一致'):
        restore_workspace(package,tmp_path/'restored')


def test_full_package_works_after_original_roots_disappear(workspace,tmp_path):
    paths,_,_,photos=workspace
    package,_=exported(workspace,tmp_path)
    photos.rename(photos.with_name('old-disk-offline'))
    paths.data_dir.rename(paths.data_dir.with_name('old-data-offline'))
    data,_=restore_workspace(package,tmp_path/'restored')
    for path, in table(data/'eidory.sqlite3','images','file_path'):
        assert Path(path).is_file()


def test_removed_image_with_no_live_folder_migrates_recovery(workspace,tmp_path):
    paths,*_=workspace
    old_file=tmp_path/'separate-drive'/'removed.png';old_file.parent.mkdir();old_file.write_bytes(b'recoverable original')
    root=paths.data_dir/'recovery-records';root.mkdir()
    payload={'image_ids':[999], 'snapshot':{'images':[{'id':999,'file_path':str(old_file)}]}}
    (root/'removed.json.gz').write_bytes(gzip.compress(json.dumps(payload).encode()))
    package,_=exported(workspace,tmp_path)
    old_file.unlink()
    data,_=restore_workspace(package,tmp_path/'restored')
    record=json.loads(gzip.decompress((data/'recovery-records/removed.json.gz').read_bytes()))
    assert Path(record['snapshot']['images'][0]['file_path']).read_bytes()==b'recoverable original'


def test_switching_back_clears_only_managed_model_environment(tmp_path,monkeypatch):
    from eidory.core.portable_model import configure_portable_model
    monkeypatch.delenv('EIDORY_PREVIOUS_HF_ENV',raising=False)
    monkeypatch.setenv('HF_HOME',str(tmp_path/'original-cache'))
    for key in ('HF_HUB_CACHE','HF_MODULES_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):
        monkeypatch.delenv(key,raising=False)
    data=tmp_path/'portable';(data/'models/huggingface').mkdir(parents=True)
    (data/'models/huggingface/model-manifest.json').write_text('{"model_name":"jinaai/jina-clip-v2"}')
    configure_portable_model(data)
    assert os.environ['HF_HUB_OFFLINE']=='1'
    configure_portable_model(tmp_path/'plain')
    assert os.environ['HF_HOME']==str(tmp_path/'original-cache')
    assert 'HF_HUB_OFFLINE' not in os.environ


def test_preexisting_orphan_is_preserved_and_reported_without_deleting_history(workspace,tmp_path):
    paths,*_=workspace
    with sqlite3.connect(paths.database_path) as conn:
        conn.execute("INSERT INTO creative_nodes(id,project_id,title,created_at,updated_at) VALUES(99,999,'旧孤立节点','now','now')")
    package,manifest=exported(workspace,tmp_path)
    assert len(manifest['existing_relation_issues'])==1
    data,report=restore_workspace(package,tmp_path/'restored')
    assert report['existing_relation_issues']==manifest['existing_relation_issues']
    assert table(paths.database_path,'creative_nodes')==table(data/'eidory.sqlite3','creative_nodes')


def test_new_orphan_during_restore_is_rejected(workspace,tmp_path):
    from eidory.core.workspace_package import _rewrite_database
    package,_=exported(workspace,tmp_path)
    def inject(database,*args):
        with sqlite3.connect(database) as conn:
            conn.execute("INSERT INTO creative_nodes(id,project_id,title,created_at,updated_at) VALUES(99,999,'bad','now','now')")
        return _rewrite_database(database,*args)
    with patch('eidory.core.workspace_package._rewrite_database',side_effect=inject):
        with pytest.raises(ValueError,match='关联关系校验失败'):
            restore_workspace(package,tmp_path/'restored')
    assert not (tmp_path/'restored').exists()


def test_partial_restore_cannot_be_selected_as_a_live_workspace(workspace,tmp_path):
    data=tmp_path/'restore.incomplete/data';data.mkdir(parents=True)
    shutil.copy2(workspace[0].database_path,data/'eidory.sqlite3')
    with pytest.raises(ValueError,match='暂存目录'):
        selection.validate_workspace(data)
