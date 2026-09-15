"""Online SQLite snapshots, separate from explicit user backups."""
from datetime import datetime
from pathlib import Path
import sqlite3
from uuid import uuid4


def database_summary(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        if conn.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('数据库完整性检查未通过')
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {table: conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] if table in tables else 0
                for table in ('images', 'creative_projects', 'creative_nodes')}
    finally:
        conn.close()


def automatic_backup(database_path, backup_dir, *, day=None, keep=7):
    day = day or datetime.now().strftime('%Y%m%d')
    root = Path(backup_dir); root.mkdir(parents=True, exist_ok=True)
    target = root / f'auto-{day}.sqlite3'
    if target.exists():
        database_summary(target)
        return target
    temporary = root / ('.pending-' + uuid4().hex + '.sqlite3')
    try:
        source = sqlite3.connect(Path(database_path).resolve().as_uri() + '?mode=ro', uri=True, timeout=30)
        try:
            destination = sqlite3.connect(temporary)
            try:
                source.backup(destination, pages=256, sleep=.01)
            finally:
                destination.close()
        finally:
            source.close()
        database_summary(temporary)
        temporary.replace(target)
        for old in sorted(root.glob('auto-????????.sqlite3'), reverse=True)[max(1, keep):]:
            old.unlink()
        return target
    finally:
        temporary.unlink(missing_ok=True)
