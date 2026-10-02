"""Validate and migrate a private backup before switching the live database."""
from contextlib import closing
from pathlib import Path
import os
import sqlite3

from eidory.core.metadata_store import MetadataStore


def prepare_database(source, destination):
    source, destination = Path(source), Path(destination)
    with closing(sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)) as src:
        if src.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('数据库完整性检查未通过')
        for table, required in {'folders': {'id', 'folder_path'},
                                'images': {'id', 'folder_id', 'file_path'},
                                'app_settings': {'key', 'value'}}.items():
            columns = {row[1] for row in src.execute(f'PRAGMA table_info({table})')}
            if not required <= columns:
                raise ValueError(f'数据库结构不兼容：{table}')
        with closing(sqlite3.connect(destination)) as dst:
            src.backup(dst)
    staged = MetadataStore(destination)
    staged.initialize()
    with staged.connect() as conn:
        schema = Path(__file__).resolve().parent.parent / 'db' / 'schema.sql'
        with closing(sqlite3.connect(':memory:')) as expected:
            expected.executescript(schema.read_text())
            tables = [row[0] for row in expected.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            for table in tables:
                required = {row[1] for row in expected.execute(f'PRAGMA table_info({table})')}
                actual = {row['name'] for row in conn.execute(f'PRAGMA table_info({table})')}
                if not required <= actual:
                    raise ValueError(f'数据库缺少必要字段：{table}')
        if conn.execute('PRAGMA foreign_key_check').fetchone():
            raise ValueError('数据库存在无效关联')
        if conn.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('迁移后的数据库完整性检查未通过')
    # Close all connections and fold the WAL into the staged main file.
    with closing(sqlite3.connect(destination)) as conn:
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')


def install_database(store, prepared, rollback):
    """Caller has stopped all live readers/writers and made a rollback snapshot."""
    target = store.database_path
    for suffix in ('-wal', '-shm'):
        Path(str(target) + suffix).unlink(missing_ok=True)
    os.replace(prepared, target)
    try:
        store.initialize()
    except Exception:
        for suffix in ('-wal', '-shm'):
            Path(str(target) + suffix).unlink(missing_ok=True)
        # Copy, preserving the user's recovery backup even if switching fails.
        import shutil
        recovery = target.with_name('.rollback-' + target.name)
        shutil.copy2(rollback, recovery)
        os.replace(recovery, target)
        raise
