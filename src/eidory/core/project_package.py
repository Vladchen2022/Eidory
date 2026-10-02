"""Portable, editable projects. Import validates before writing and never trusts IDs."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from uuid import uuid4

from eidory.core.project_state import capture_project, install_state, validate_state, remap_project_state
from eidory.core.time_utils import utc_now_iso


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_project_package(store, project_id, destination, *, include_files=True):
    with store.connect() as conn:
        conn.execute("BEGIN")
        state = capture_project(store, project_id, conn=conn)
        versions = [dict(r) for r in conn.execute('SELECT title,created_at,payload_json FROM creative_project_snapshots WHERE project_id=? ORDER BY id', (project_id,))]
    history = [dict(title=v['title'], created_at=v['created_at'], state=json.loads(v['payload_json'])) for v in versions]
    target = Path(destination) / ('Eidory-project-' + uuid4().hex[:8])
    target.mkdir(parents=True)
    ids = {r['image_id'] for snapshot in [state] + [v['state'] for v in history] for r in snapshot['tables']['creative_node_images']}
    media = []
    try:
        for image in store.images_by_ids(sorted(ids)):
            source = Path(image.file_path)
            record = {'id': image.id, 'name': image.file_name, 'source': str(source),
                      'width': image.width, 'height': image.height, 'duration_ms': image.duration_ms,
                      'file_size': image.file_size, 'sha256': file_digest(source) if source.is_file() else '',
                      'missing': not source.is_file()}
            if include_files and source.is_file():
                relative = Path('media') / f'{image.id}{source.suffix.lower()}'
                (target / 'media').mkdir(exist_ok=True)
                shutil.copy2(source, target / relative)
                record['relative'] = relative.as_posix()
            media.append(record)
        if {r['id'] for r in media} != ids:
            raise ValueError('项目关联了已移除的图片，无法完整导出')
        document = {'format': 'eidory-project', 'version': 1, 'state': state, 'media': media, 'history': history}
        (target / 'project.eidory.json').write_text(json.dumps(document, ensure_ascii=False, indent=2))
        return target
    except Exception:
        shutil.rmtree(target)
        raise


def inspect_project_package(path):
    path = Path(path)
    if path.is_dir():
        path /= 'project.eidory.json'
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError('项目清单过大')
    doc = json.loads(path.read_text())
    if doc.get('format') != 'eidory-project' or doc.get('version') != 1:
        raise ValueError('请选择可编辑项目包的 project.eidory.json')
    validate_state(doc.get('state'))
    history = doc.get('history', [])
    if not isinstance(history, list) or len(history) > 10000:
        raise ValueError('项目历史格式无效')
    for version in history:
        validate_state(version.get('state'))
    ids = {r['image_id'] for snapshot in [doc['state']] + [v['state'] for v in history] for r in snapshot['tables'].get('creative_node_images', [])}
    if not isinstance(doc.get('media'), list) or len(doc['media']) != len(ids) or {r['id'] for r in doc['media']} != ids:
        raise ValueError('项目包的图片清单不完整或重复')
    prepared, missing = [], []
    for record in doc['media']:
        if record.get('relative'):
            source = (path.parent / record['relative']).resolve()
            if not source.is_relative_to(path.parent.resolve()):
                raise ValueError('项目包图片路径越界')
        else:
            source = Path(record['source']).expanduser()
        if source.is_file():
            if record.get('sha256') and file_digest(source) != record['sha256']:
                raise ValueError('图片校验不一致：' + record['name'])
        else:
            missing.append(record['name'])
        prepared.append((record, source))
    return path, doc, prepared, missing


def import_project_package(store, path, data_dir):
    _, doc, prepared, missing = inspect_project_package(path)
    owned = Path(data_dir) / 'imported-projects' / uuid4().hex
    owned.mkdir(parents=True)
    try:
        image_map = {}
        staged = []
        missing = []
        # File I/O must finish before acquiring SQLite's single writer lock.
        for record, source in prepared:
            destination = owned / f'{record["id"]}{Path(record["name"]).suffix.lower()}'
            available = source.is_file()
            if available:
                shutil.copy2(source, destination)
                if record.get('sha256') and file_digest(destination) != record['sha256']:
                    raise ValueError('复制期间素材已改变：' + record['name'])
            else:
                missing.append(record['name'])
            staged.append((record, destination, destination.stat() if available else None))
        with store.connect() as c:
            now = utc_now_iso()
            folder = c.execute("INSERT INTO folders(folder_path,import_mode,added_at,is_active) VALUES (?,'indexed',?,1)", (str(owned), now)).lastrowid
            for record, destination, st in staged:
                image_id, _ = store._upsert_image_on_connection(c, now=now, folder_id=folder,
                    file_path=str(destination), file_size=st.st_size if st else record.get('file_size', 0),
                    width=record.get('width'), height=record.get('height'), created_time_ns=None,
                    modified_time_ns=st.st_mtime_ns if st else 0, duration_ms=record.get('duration_ms'))
                if st is None:
                    c.execute('UPDATE images SET is_missing=1 WHERE id=?', (image_id,))
                image_map[record['id']] = image_id
            node_map = {}
            project_id = install_state(c, doc['state'], image_map=image_map, mappings=node_map)
            historic_ids = {n['id'] for v in doc.get('history', []) for n in v['state']['nodes']} - node_map.keys()
            highest = c.execute('SELECT COALESCE(MAX(id),0) FROM creative_nodes').fetchone()[0]
            node_map.update({old: highest+i+1 for i, old in enumerate(sorted(historic_ids))})
            for version in doc.get('history', []):
                remapped = remap_project_state(version['state'], project_id, node_map, image_map)
                c.execute('INSERT INTO creative_project_snapshots(project_id,title,created_at,payload_json) VALUES (?,?,?,?)',
                          (project_id, version['title'], version['created_at'], json.dumps(remapped, ensure_ascii=False)))
        return project_id, missing
    except Exception:
        shutil.rmtree(owned)
        raise
