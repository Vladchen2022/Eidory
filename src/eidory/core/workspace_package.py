"""Resumable, verified workspace copies. Never re-index imported image records.

The package and restored workspace are directories, so interrupted multi-GB
copies can resume. A restore becomes visible only after its database and every
available media file pass validation. The old workspace is never overwritten.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import gzip
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import uuid

from eidory.core.database_restore import prepare_database
from eidory.core.portable_model import cached_model_files
from eidory import __version__


FORMAT = 'eidory-workspace'
VERSION = 1
MANIFEST = 'workspace.eidory.json'
SKIP_DATA = {'logs', 'backups', 'app-backups', 'models', 'workspace-location.json',
             'workspace.json', 'migration-report.json'}
PATH_KEYS = {'path', 'paths', 'source', 'backup', 'stage', 'root', 'roots',
             'folder', 'folders', 'queued_source', 'thumbnail', 'original', 'destination', 'backup_dir', 'backup_files'}
ACTIVE_JOBS = {'submitting', 'unknown', 'queued', 'running', 'downloading', 'offline', 'paused'}


class MigrationCancelled(Exception):
    pass


def _check_cancel(cancel):
    if cancel and cancel():
        raise MigrationCancelled('已暂停；重新选择同一个目录可以继续。')


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def sha256(path, cancel=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(4 * 1024 * 1024):
            _check_cancel(cancel)
            h.update(block)
    return h.hexdigest()


def _contained(root, relative):
    """Reject traversal and symlink escapes, including destination parents."""
    relative = PurePosixPath(relative)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts or '\\' in str(relative):
        raise ValueError('迁移包包含无效的相对路径')
    result = Path(root).joinpath(*relative.parts)
    current = result
    while current != Path(root):
        if current.is_symlink():
            raise ValueError(f'迁移路径不能是符号链接：{relative}')
        current = current.parent
    if not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('迁移路径超出所选目录')
    return result


def _signature(path):
    s = Path(path).stat()
    return s.st_size, s.st_mtime_ns


def _copy_verified(source, destination, *, expected=None, signature=None, progress=None, cancel=None):
    source, destination = Path(source), Path(destination)
    _check_cancel(cancel)
    before = _signature(source)
    if signature is not None and before != tuple(signature):
        raise ValueError(f'快照后文件发生变化，请重新导出：{source.name}')
    if destination.is_file() and expected and sha256(destination, cancel) == expected:
        if progress:
            progress(before[0], source.name)
        return expected
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name('.' + destination.name + '.part')
    h = hashlib.sha256()
    try:
        with source.open('rb') as src, temporary.open('wb') as dst:
            while block := src.read(4 * 1024 * 1024):
                _check_cancel(cancel)
                dst.write(block)
                h.update(block)
                if progress:
                    progress(len(block), source.name)
            dst.flush()
            os.fsync(dst.fileno())
        digest = h.hexdigest()
        if _signature(source) != before or (expected and digest != expected):
            raise ValueError(f'内容校验未通过，文件可能发生变化：{source.name}')
        shutil.copystat(source, temporary)
        # Read back copied bytes; a successful write alone is not verification.
        if sha256(temporary, cancel) != digest:
            raise ValueError(f'写入后校验失败：{destination.name}')
        os.replace(temporary, destination)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _db(path, readonly=False):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + ('?mode=ro' if readonly else ''), uri=True)
    conn.row_factory = sqlite3.Row
    return closing(conn)


def database_counts(path):
    with _db(path, True) as conn:
        if conn.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('数据库完整性检查未通过')
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts = {t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in
                  ('images', 'embeddings', 'collections', 'creative_projects', 'creative_nodes',
                   'ai_vision_tags', 'ai_vision_profile_results', 'image_upscale_jobs', 'image_replacements') if t in tables}
        counts['ready_vectors'] = conn.execute("SELECT COUNT(*) FROM embeddings WHERE status='ready' AND vector_blob IS NOT NULL").fetchone()[0]
        return counts


def _media_roots(folders, images, data_dir):
    candidates = {Path(f) for f in folders}
    for image in images:
        p = Path(image['file_path'])
        if not any(p.is_relative_to(r) for r in candidates):
            candidates.add(p.parent)
    # Group nested library folders beneath their common Pictures/volume child.
    # Only referenced images are copied; unrelated files in that parent are not.
    roots = set()
    for root in candidates:
        if root.is_relative_to(data_dir):
            continue
        for name in ('Pictures', 'Documents', 'Desktop'):
            if name in root.parts:
                i = root.parts.index(name)
                if len(root.parts) > i + 1:
                    root = Path(*root.parts[:i+2])
                break
        roots.add(root)
    minimal = []
    for root in sorted(roots, key=lambda p: (len(p.parts), str(p))):
        if not any(root.is_relative_to(parent) for parent in minimal):
            minimal.append(root)
    return [dict(id=f'r{i+1:03}', source=str(root), name=root.name,
                 directory=f'r{i+1:03}/' + (root.name or '图库')) for i, root in enumerate(minimal)]


def _root_directory(root):
    return root.get('directory', root['id'])


def _new_manifest(paths, package, include_media, include_model, include_secrets):
    db = package / 'database.sqlite3'
    with _db(paths.database_path, True) as src, _db(db) as dst:
        src.backup(dst)
        if not include_secrets:
            keys = [r[0] for r in dst.execute('SELECT key FROM app_settings')
                    if any(s in r[0].lower() for s in ('api_key', 'password', 'secret', 'access_token'))]
            dst.executemany('DELETE FROM app_settings WHERE key=?', [(k,) for k in keys])
            dst.commit()
            # Remove deleted plaintext values from SQLite's free pages too.
            dst.execute('VACUUM')
    # Removed images may no longer have a live folder row, but their recovery
    # records must still resolve on the new computer.
    archived_images = []
    for file in (paths.data_dir/'recovery-records').glob('*.json.gz'):
        record = json.loads(gzip.decompress(file.read_bytes()))
        removed = set(record.get('image_ids', []))
        archived_images.extend(i for i in record.get('snapshot', {}).get('images', []) if i.get('id') in removed)
    with _db(db, True) as conn:
        existing_relation_issues = [list(r) for r in conn.execute('PRAGMA foreign_key_check')]
        images = [dict(r) for r in conn.execute('SELECT id,file_path,file_size,modified_time_ns,is_missing FROM images ORDER BY id')]
        roots = _media_roots([r[0] for r in conn.execute('SELECT folder_path FROM folders')], images + archived_images, paths.data_dir)
        active = sum(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE json_extract(payload,'$.state') IN ('submitting','unknown','queued','running','downloading','offline')").fetchone()[0]
                     for table in ('image_upscale_jobs', 'generated_reference_jobs'))
        if conn.execute("SELECT 1 FROM image_replacements WHERE state='prepared' LIMIT 1").fetchone():
            raise ValueError('存在尚未完成的原图替换，请重新启动软件完成恢复后再导出。')
    entries, folded = {}, set()

    def add(path, relative, included=True):
        size, mtime = _signature(path)
        entry = dict(path=relative, source=str(path), size=size, mtime_ns=mtime, included=included)
        if relative.casefold() in folded and relative not in entries:
            raise ValueError(f'文件名仅大小写不同，无法安全迁移到默认 Mac 磁盘：{path.name}')
        entries[relative] = entry
        folded.add(relative.casefold())
        return relative

    for image in images:
        source = Path(image['file_path'])
        if source.is_relative_to(paths.data_dir):
            relative = 'data/' + source.relative_to(paths.data_dir).as_posix()
            included = True
        else:
            root = next(r for r in roots if source.is_relative_to(Path(r['source'])))
            relative = 'media/' + _root_directory(root) + '/' + source.relative_to(root['source']).as_posix()
            included = include_media
        image['entry'] = relative
        image['missing'] = not source.is_file()
        if image['missing']:
            continue
        if _signature(source) != (image['file_size'], image['modified_time_ns']):
            raise ValueError(f'图库记录与原文件不一致，请先扫描更新再导出：{source.name}')
        add(source, relative, included)
    live_paths = {i['file_path'] for i in images}
    for image in archived_images:
        source = Path(image['file_path'])
        if str(source) in live_paths or not source.is_file() or source.is_relative_to(paths.data_dir):
            continue
        root = next(r for r in roots if source.is_relative_to(r['source']))
        add(source, 'media/' + _root_directory(root) + '/' + source.relative_to(root['source']).as_posix(), include_media)
    # Thumbnails, pending AI candidates, undo/replacement media and imported
    # projects remain part of the workspace in both modes. Historical database
    # backups and logs are not runtime dependencies and are intentionally omitted.
    for directory, dirs, files in os.walk(paths.data_dir, followlinks=False):
        current = Path(directory)
        if current == paths.data_dir:
            dirs[:] = [d for d in dirs if d not in SKIP_DATA]
        for d in dirs:
            if (current/d).is_symlink():
                raise ValueError(f'数据目录中含符号链接，请先整理后导出：{current/d}')
        for name in sorted(files):
            source = current / name
            rel = source.relative_to(paths.data_dir)
            if source == paths.database_path or str(source) in {str(paths.database_path)+'-wal', str(paths.database_path)+'-shm'}:
                continue
            if rel.parts[0] in SKIP_DATA or (len(rel.parts) == 1 and (name.startswith('eidory.sqlite3') or name.startswith('.'))):
                continue
            if source.is_symlink():
                raise ValueError(f'数据目录中含符号链接：{source}')
            add(source, 'data/' + rel.as_posix())
    model = None
    if include_model:
        model_files, model = cached_model_files(paths.data_dir)
        for source, relative in model_files:
            add(source, 'data/' + relative)
    return dict(format=FORMAT, version=VERSION, app_version=__version__, id=str(uuid.uuid4()), complete=False,
                created_at=datetime.now(timezone.utc).isoformat(), source_data_dir=str(paths.data_dir),
                include_media=include_media, include_model=include_model, include_secrets=include_secrets,
                counts=database_counts(db), roots=roots, images=images, files=list(entries.values()),
                model=model, missing=sum(i['missing'] for i in images), active_jobs=active,
                existing_relation_issues=existing_relation_issues)


def prepare_export(paths, destination, *, include_media=True, include_model=True, include_secrets=False):
    package = Path(destination).expanduser().absolute()
    if package.resolve().is_relative_to(paths.data_dir.resolve()):
        raise ValueError('请把迁移包放在当前工作区以外，避免递归复制。')
    if package.is_symlink():
        raise ValueError('迁移包目录不能是符号链接')
    package.mkdir(parents=True, exist_ok=True)
    manifest_path = package / MANIFEST
    if manifest_path.exists():
        manifest = inspect_package(package, allow_partial=True)
        if manifest['complete']:
            raise ValueError('此目录已经是完整迁移包，请选择新的空目录。')
        if manifest['source_data_dir'] != str(paths.data_dir) or any(manifest[k] != v for k, v in
            [('include_media', include_media), ('include_model', include_model), ('include_secrets', include_secrets)]):
            raise ValueError('继续导出时，源工作区和导出选项必须与上次相同。')
    else:
        if any(package.iterdir()):
            raise ValueError('请选择空目录，或上次未完成的迁移包目录。')
        try:
            manifest = _new_manifest(paths, package, include_media, include_model, include_secrets)
            manifest['database_sha256'] = sha256(package/'database.sqlite3')
            _json_write(manifest_path, manifest)
        except Exception:
            (package/'database.sqlite3').unlink(missing_ok=True)
            raise
    return manifest


def export_workspace(paths, destination, *, include_media=True, include_model=True, include_secrets=False,
                     progress=None, cancel=None):
    manifest = prepare_export(paths, destination, include_media=include_media, include_model=include_model,
                              include_secrets=include_secrets)
    package = Path(destination).expanduser().absolute()
    manifest_path = package/MANIFEST
    total = sum(f['size'] for f in manifest['files'])
    remaining = sum(f['size'] for f in manifest['files'] if f['included'] and not (package/f['path']).exists())
    if shutil.disk_usage(package).free < remaining + 64*1024*1024:
        raise ValueError('目标磁盘剩余空间不足')
    done = 0

    def advance(amount, label):
        nonlocal done
        done += amount
        if progress:
            progress(done, total, label)

    journal_path = _contained(package, '.export-progress.sqlite3')
    # A compact journal avoids rewriting a 30,000-image manifest per file.
    with closing(sqlite3.connect(journal_path)) as journal:
        journal.execute('CREATE TABLE IF NOT EXISTS copied(path TEXT PRIMARY KEY, digest TEXT NOT NULL)')
        previous = dict(journal.execute('SELECT path,digest FROM copied'))
        for index, entry in enumerate(manifest['files']):
            _check_cancel(cancel)
            source = Path(entry['source'])
            signature = (entry['size'], entry['mtime_ns'])
            if _signature(source) != signature:
                raise ValueError(f'快照后文件发生变化，请重新导出：{source.name}')
            expected = previous.get(entry['path'])
            if entry['included']:
                entry['sha256'] = _copy_verified(source, _contained(package, entry['path']), expected=expected,
                                                   signature=signature, progress=advance, cancel=cancel)
            else:
                digest = sha256(source, cancel)
                if _signature(source) != signature or (expected and digest != expected):
                    raise ValueError(f'校验时原图发生变化：{source.name}')
                entry['sha256'] = digest
                advance(entry['size'], source.name)
            journal.execute('INSERT OR REPLACE INTO copied VALUES(?,?)', (entry['path'], entry['sha256']))
            if index % 32 == 31:
                journal.commit()
        journal.commit()
    if sha256(package/'database.sqlite3', cancel) != manifest['database_sha256']:
        raise ValueError('迁移包数据库校验未通过')
    manifest['complete'] = True
    _json_write(manifest_path, manifest)
    journal_path.unlink(missing_ok=True)
    (package/'迁移说明.txt').write_text(
        '在新电脑安装 Eidory，打开设置 → 工作区迁移 → 从迁移包恢复。\n'
        '选择本目录；恢复完成后切换工作区并重新打开软件。原图、向量、标签和项目关联无需重建。\n'
        '请保留完整目录结构。迁移包不是双向同步；后续修改不会自动同步到另一台电脑。\n'
        '外部 AI 服务需在新电脑另行配置。API Key ' + ('已包含（明文，请自行保管）。' if include_secrets else '未包含，需要重新填写。'), encoding='utf-8')
    return manifest


def inspect_package(package, *, allow_partial=False):
    package = Path(package).resolve()
    manifest = json.loads((package/MANIFEST).read_text(encoding='utf-8'))
    if manifest.get('format') != FORMAT or manifest.get('version') != VERSION:
        raise ValueError('不支持的工作区迁移包格式或版本')
    try:
        if tuple(int(p) for p in manifest['app_version'].split('.')) > tuple(int(p) for p in __version__.split('.')):
            raise ValueError('迁移包来自更新版本的 Eidory，请先升级软件再恢复。')
    except (KeyError, TypeError):
        raise ValueError('迁移包缺少应用版本') from None
    if not allow_partial and not manifest.get('complete'):
        raise ValueError('迁移包尚未导出完成，请在旧电脑继续导出。')
    if not isinstance(manifest.get('files'), list) or not isinstance(manifest.get('images'), list):
        raise ValueError('迁移清单不完整')
    seen = set()
    for entry in manifest['files']:
        rel = entry['path']
        _contained(package, rel)
        if not rel.startswith(('data/', 'media/')) or rel.casefold() in seen or entry['size'] < 0:
            raise ValueError('迁移清单包含重复或无效文件')
        seen.add(rel.casefold())
        if manifest.get('complete') and (len(entry.get('sha256', '')) != 64 or any(c not in '0123456789abcdef' for c in entry['sha256'])):
            raise ValueError('迁移清单缺少文件校验码')
    for root in manifest['roots']:
        if not root['id'].isalnum() or not Path(root['source']).is_absolute():
            raise ValueError('原图库根目录记录无效')
        directory = PurePosixPath(_root_directory(root))
        _contained(package, 'media/' + str(directory))
        if directory.parts[0] != root['id'] or len(directory.parts) > 2:
            raise ValueError('迁移根目录结构无效')
    if len({r['id'] for r in manifest['roots']}) != len(manifest['roots']):
        raise ValueError('根目录编号重复')
    for image in manifest['images']:
        _contained(package, image['entry'])
        if not image['missing'] and image['entry'].casefold() not in seen:
            raise ValueError('图片没有对应的文件校验记录')
    if sha256(_contained(package, 'database.sqlite3')) != manifest['database_sha256']:
        raise ValueError('迁移包数据库校验未通过')
    with _db(package/'database.sqlite3', True) as conn:
        actual_issues = [list(r) for r in conn.execute('PRAGMA foreign_key_check')]
        if 'existing_relation_issues' in manifest and manifest['existing_relation_issues'] != actual_issues:
            raise ValueError('迁移清单的历史关联异常记录与数据库不一致')
        manifest['existing_relation_issues'] = actual_issues
        actual = {r['id']: dict(r) for r in conn.execute('SELECT id,file_path,file_size,modified_time_ns FROM images')}
        if len(manifest['images']) != len(actual) or len({i['id'] for i in manifest['images']}) != len(actual):
            raise ValueError('图片校验清单与数据库不一致')
        files = {f['path']:f for f in manifest['files']}
        data_dir = Path(manifest['source_data_dir'])
        if not data_dir.is_absolute():
            raise ValueError('源工作区路径无效')
        for image in manifest['images']:
            if any(image.get(k) != v for k,v in actual.get(image['id'], {}).items()) or image['id'] not in actual:
                raise ValueError('图片校验清单与数据库不一致')
            source = Path(image['file_path'])
            if source.is_relative_to(data_dir):
                expected = 'data/' + source.relative_to(data_dir).as_posix()
            else:
                roots = [r for r in manifest['roots'] if source.is_relative_to(r['source'])]
                if len(roots) != 1:
                    raise ValueError('图片对应的根目录不明确')
                expected = 'media/' + _root_directory(roots[0]) + '/' + source.relative_to(roots[0]['source']).as_posix()
            if image['entry'] != expected or (not image['missing'] and files[expected]['size'] != image['file_size']):
                raise ValueError('图片与文件校验记录不匹配')
    return manifest


def _map_path(value, mappings):
    if not isinstance(value, str) or not value.startswith('/'):
        return value
    path = Path(value)
    for before, after in mappings:
        if path.is_relative_to(before):
            return str(after / path.relative_to(before))
    return value


def _rewrite_json(value, mappings, key=''):
    if isinstance(value, dict):
        return {_map_path(k, mappings): _rewrite_json(v, mappings, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite_json(v, mappings, key) for v in value]
    if key in PATH_KEYS or key.endswith(('_path', '_paths', '_root', '_roots')):
        return _map_path(value, mappings)
    return value


def _rewrite_database(database, manifest, mappings, image_stats, final_data):
    with _db(database) as conn:
        conn.execute('PRAGMA foreign_keys=ON')
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for table in tables:
            columns = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
            paths = [c for c in columns if c.endswith('_path')]
            jsons = [c for c in columns if c.endswith('_json') or c == 'payload']
            if table == 'app_settings':
                paths, jsons = ['value'], []
            for row in list(conn.execute(f'SELECT rowid AS _rowid_,* FROM "{table}"')) if paths or jsons else []:
                updates = {}
                for column in paths:
                    value = row[column]
                    mapped = _map_path(value, mappings)
                    if table == 'app_settings' and isinstance(value, str):
                        try:
                            parsed = json.loads(value)
                            context = 'paths' if any(s in row['key'] for s in ('folder', 'path', 'root')) else ''
                            rewritten = _rewrite_json(parsed, mappings, context)
                            if rewritten != parsed:
                                mapped = json.dumps(rewritten, ensure_ascii=False)
                        except (ValueError, TypeError):
                            pass
                    if mapped != value:
                        updates[column] = mapped
                for column in jsons:
                    value = row[column]
                    if not value:
                        continue
                    try:
                        original = json.loads(value)
                    except (ValueError, TypeError):
                        continue
                    mapped = _rewrite_json(original, mappings)
                    if table in ('image_upscale_jobs', 'generated_reference_jobs'):
                        # Server prompt IDs belong to the old computer. Never poll
                        # or replay them in the restored workspace.
                        if mapped.get('state') in ACTIVE_JOBS:
                            mapped.update(migration_previous_state=mapped['state'], state='cancelled',
                                          error='迁移时中止跟踪旧设备任务；可在本机手动重试。')
                        if 'queued_source' in mapped and mapped.get('image_id') in image_stats:
                            old = mapped['queued_source']
                            stat = image_stats[mapped['image_id']]
                            mapped['queued_source'] = [old[0], stat[0], stat[1]]
                    if mapped != original:
                        updates[column] = json.dumps(mapped, ensure_ascii=False)
                if updates:
                    conn.execute(f'UPDATE "{table}" SET ' + ','.join(f'"{c}"=?' for c in updates) + ' WHERE rowid=?',
                                 (*updates.values(), row['_rowid_']))
        for image in manifest['images']:
            iid = image['id']
            if iid not in image_stats:
                conn.execute('UPDATE images SET is_missing=1 WHERE id=?', (iid,))
                continue
            size, mtime = image_stats[iid]
            old_mtime = image['modified_time_ns']
            conn.execute('UPDATE images SET modified_time_ns=?,file_size=?,is_missing=0 WHERE id=?', (mtime, size, iid))
            # Timestamp rebinding only follows a successful SHA-256 comparison.
            # Existing stale labels remain stale; completed AI content is untouched.
            conn.execute('UPDATE image_hashes SET modified_time_ns=? WHERE image_id=? AND modified_time_ns=?', (mtime, iid, old_mtime))
            conn.execute('UPDATE image_preview_annotations SET modified_time_ns=? WHERE image_id=? AND modified_time_ns=? AND file_size=?', (mtime, iid, old_mtime, size))
            for table in ('ai_vision_tags', 'ai_vision_profile_results'):
                if table in tables:
                    conn.execute(f'UPDATE {table} SET source_modified_time_ns=? WHERE image_id=? AND source_modified_time_ns=?', (mtime, iid, old_mtime))
        now = datetime.now(timezone.utc).isoformat()
        for key, value in [('ui.file_watch_enabled', '0'), ('upscale.queue_paused', '1'),
                           ('migration.restored_at', now), ('migration.review_services', '1')]:
            conn.execute('INSERT INTO app_settings(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at', (key, value, now))
        issues = {tuple(r) for r in conn.execute('PRAGMA foreign_key_check')}
        if issues != {tuple(r) for r in manifest.get('existing_relation_issues', [])}:
            raise ValueError('恢复后关联关系校验失败')
        conn.commit()
        if conn.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('恢复后数据库完整性检查失败')
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')


def restore_workspace(package, destination, *, root_mapping=None, progress=None, cancel=None):
    package = Path(package).resolve()
    manifest = inspect_package(package)
    final = Path(destination).expanduser().absolute()
    if final.exists() or final.is_symlink():
        raise ValueError('恢复目标必须是尚不存在的新文件夹；不会覆盖现有工作区。')
    if final.resolve().is_relative_to(package):
        raise ValueError('请把恢复工作区放在迁移包之外')
    partial = final.with_name(final.name + '.incomplete')
    if partial.is_symlink():
        raise ValueError('恢复暂存目录不能是符号链接')
    partial.mkdir(parents=True, exist_ok=True)
    identity = dict(package_id=manifest['id'], manifest_sha256=sha256(package/MANIFEST), destination=str(final))
    journal = partial/'restore-state.json'
    if journal.exists():
        if json.loads(journal.read_text()) != identity:
            raise ValueError('暂存目录属于另一次恢复，请选择新的目标位置。')
    elif any(partial.iterdir()):
        raise ValueError('暂存目录非空，未改动其中内容。')
    else:
        _json_write(journal, identity)
    roots = {}
    for root in manifest['roots']:
        if manifest['include_media']:
            roots[root['id']] = final/'media'/_root_directory(root)
        else:
            mapped = (root_mapping or {}).get(root['id'])
            if not mapped or not Path(mapped).is_dir():
                raise ValueError(f'请选择原图库在新电脑上的位置：{root["source"]}')
            roots[root['id']] = Path(mapped).expanduser().resolve()
    entries = {f['path']: f for f in manifest['files']}
    total = sum(f['size'] for f in entries.values())
    required = sum(f['size'] for f in entries.values() if f['included'] and not (partial/f['path']).exists())
    if shutil.disk_usage(partial).free < required + 2*(package/'database.sqlite3').stat().st_size + 64*1024*1024:
        raise ValueError('恢复位置空间不足')
    done = 0

    def advance(amount, label):
        nonlocal done
        done += amount
        if progress:
            progress(done, total, label)

    stats = {}
    for entry in entries.values():
        _check_cancel(cancel)
        if entry['included']:
            source = _contained(package, entry['path'])
            if source.stat().st_size != entry['size']:
                raise ValueError(f'迁移包文件大小不一致：{source.name}')
            _copy_verified(source, _contained(partial, entry['path']), expected=entry['sha256'], progress=advance, cancel=cancel)
        else:
            parts = PurePosixPath(entry['path']).parts
            if len(parts) < 3 or parts[0] != 'media' or parts[1] not in roots:
                raise ValueError('图片关联记录无效')
            root = next(r for r in manifest['roots'] if r['id'] == parts[1])
            relative = PurePosixPath(entry['path']).relative_to('media/' + _root_directory(root))
            source = _contained(roots[parts[1]], str(relative))
            before = _signature(source)
            if before[0] != entry['size'] or sha256(source, cancel) != entry['sha256'] or _signature(source) != before:
                raise ValueError(f'关联图片内容与旧电脑不一致，未切换工作区：{source.name}')
            stats[entry['path']] = before
            advance(entry['size'], source.name)
    mappings = [(Path(manifest['source_data_dir']), final/'data')]
    mappings += [(Path(r['source']), roots[r['id']]) for r in manifest['roots']]
    mappings.sort(key=lambda p: len(p[0].parts), reverse=True)
    image_stats = {}
    for image in manifest['images']:
        if image['missing']:
            continue
        entry = entries[image['entry']]
        image_stats[image['id']] = (_signature(_contained(partial, entry['path'])) if entry['included'] else stats[entry['path']])
    data = partial/'data'
    data.mkdir(exist_ok=True)
    prepared = data/'eidory.sqlite3'
    # Prepare anew on each retry. The immutable source DB is never edited.
    for suffix in ('', '-wal', '-shm'):
        Path(str(prepared)+suffix).unlink(missing_ok=True)
    prepare_database(package/'database.sqlite3', prepared, existing_foreign_key_issues=manifest['existing_relation_issues'])
    _rewrite_database(prepared, manifest, mappings, image_stats, final/'data')
    if manifest['model']:
        _json_write(data/'models/huggingface/model-manifest.json', manifest['model'])
    # Recovery archives have their own structured path references.
    for directory in ('recovery-records', 'undo-removal'):
        for file in (data/directory).rglob('*.json') if (data/directory).exists() else []:
            _json_write(file, _rewrite_json(json.loads(file.read_text()), mappings))
        for file in (data/directory).rglob('*.json.gz') if (data/directory).exists() else []:
            payload = json.loads(gzip.decompress(file.read_bytes()))
            file.write_bytes(gzip.compress(json.dumps(_rewrite_json(payload, mappings), ensure_ascii=False).encode()))
    counts = database_counts(prepared)
    if any(counts.get(k) != v for k, v in manifest['counts'].items()):
        raise ValueError('恢复前后记录数量不一致，未启用此工作区')
    report = dict(package_id=manifest['id'], created_at=manifest['created_at'], restored_at=datetime.now(timezone.utc).isoformat(),
                  counts=counts, missing=manifest['missing'], include_model=manifest['include_model'],
                  api_keys_included=manifest['include_secrets'], roots={k:str(v) for k,v in roots.items()},
                  auto_tasks_paused=True, external_services_need_setup=True)
    report['existing_relation_issues'] = manifest['existing_relation_issues']
    _json_write(data/'migration-report.json', report)
    _json_write(data/'workspace.json', dict(format=FORMAT, version=VERSION, package_id=manifest['id'], complete=True))
    # Create empty/missing-only roots so the original folder hierarchy survives.
    if manifest['include_media']:
        for root in manifest['roots']:
            (partial/'media'/_root_directory(root)).mkdir(parents=True, exist_ok=True)
    _check_cancel(cancel)
    if final.exists():
        raise ValueError('恢复目标刚刚被其他程序创建，未覆盖。')
    partial.rename(final)
    return final/'data', report
