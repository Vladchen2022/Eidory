"""An atomic workspace selector; switching never deletes the previous library."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
from contextlib import closing

from eidory.core.workspace_package import _json_write

startup_warning = ''


def base_directory():
    override = os.environ.get('EIDORY_DATA_DIR', '').strip()
    return Path(override).expanduser().absolute() if override else Path.home()/'Library/Application Support/Eidory'


def validate_workspace(data_dir):
    path = Path(data_dir).expanduser().resolve()
    if path.name.endswith('.incomplete') or path.parent.name.endswith('.incomplete'):
        raise ValueError('这个工作区尚在恢复暂存目录中，请先完成恢复。')
    db = path/'eidory.sqlite3'
    with closing(sqlite3.connect(db.as_uri()+'?mode=ro', uri=True)) as conn:
        if conn.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('工作区数据库完整性检查失败')
        for table in ('images', 'folders', 'embeddings', 'app_settings'):
            conn.execute(f'SELECT 1 FROM {table} LIMIT 1')
    return path


def selection():
    file = base_directory()/'workspace-location.json'
    return json.loads(file.read_text()) if file.exists() else dict(current=str(base_directory()), history=[])


def active_directory():
    global startup_warning
    startup_warning = ''
    base = base_directory()
    if not (base/'workspace-location.json').exists():
        return base
    try:
        state = selection()
    except (OSError, ValueError):
        startup_warning = '工作区切换记录无法读取，已回到默认工作区。'
        return base
    candidates = [state.get('current', ''), *state.get('history', []), str(base)]
    for index, candidate in enumerate(dict.fromkeys(candidates)):
        if not candidate:
            continue
        try:
            path = validate_workspace(candidate)
        except (OSError, ValueError, sqlite3.Error):
            continue
        if index:
            startup_warning = '选定工作区暂不可用，已回退到上一次可用工作区。请检查外置磁盘连接。'
        return path
    raise ValueError('已登记的工作区均不可用。请连接原磁盘，或检查工作区数据库。')


def select_workspace(data_dir, previous):
    target = validate_workspace(data_dir)
    state = selection()
    history = list(dict.fromkeys([str(Path(previous).resolve()), *state.get('history', [])]))
    history = [p for p in history if p != str(target)]
    _json_write(base_directory()/'workspace-location.json', dict(current=str(target), history=history))
    return target
