"""Durable deletion records. Restoring never overwrites a conflicting source."""
from __future__ import annotations
import base64
import gzip
import json
import os
from pathlib import Path
import uuid
from eidory.core.time_utils import utc_now_iso


def encode_value(value):
    if isinstance(value, bytes):
        return {'__eidory_bytes__': base64.b64encode(value).decode('ascii')}
    raise TypeError(type(value).__name__)


def decode_value(value):
    if set(value) == {'__eidory_bytes__'}:
        return base64.b64decode(value['__eidory_bytes__'], validate=True)
    return value


class RemovalArchive:
    def __init__(self, store, data_dir):
        self.store = store
        self.directory = Path(data_dir) / 'recovery-records'

    def identity(self):
        token = self.store.get_setting('library.recovery_identity')
        if not token:
            token = uuid.uuid4().hex
            self.store.set_setting('library.recovery_identity', token)
        return token

    def save(self, payload):
        payload.setdefault('archive_id', uuid.uuid4().hex)
        payload.setdefault('created_at', utc_now_iso())
        payload.setdefault('identity', self.identity())
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / (payload['archive_id'] + '.json.gz')
        temporary = destination.with_suffix('.pending')
        raw = gzip.compress(json.dumps(payload, ensure_ascii=False, default=encode_value).encode())
        try:
            with temporary.open('wb') as file:
                file.write(raw)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        metadata = {k: payload.get(k) for k in ('archive_id', 'created_at', 'identity', 'label', 'context', 'image_ids', 'backup_dir')}
        removed_ids = set(payload.get('image_ids', []))
        metadata['snapshot'] = {'images': [image for image in payload.get('snapshot', {}).get('images', []) if image['id'] in removed_ids]}
        index = destination.with_suffix('.meta.json')
        staging = index.with_suffix('.pending')
        try:
            with staging.open('w', encoding='utf-8') as file:
                json.dump(metadata, file, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(staging, index)
        finally:
            staging.unlink(missing_ok=True)
        return payload

    def load(self, record):
        key = record.get('archive_id', '')
        if not key or Path(key).name != key:
            raise ValueError('回收记录无效')
        payload = json.loads(gzip.decompress((self.directory / (key + '.json.gz')).read_bytes()), object_hook=decode_value)
        if payload.get('identity') != self.store.get_setting('library.recovery_identity'):
            raise ValueError('此记录属于其他数据库')
        return payload

    def records(self):
        result = []
        token = self.store.get_setting('library.recovery_identity')
        for path in self.directory.glob('*.json.gz'):
            try:
                index = path.with_suffix('.meta.json')
                record = json.loads(index.read_text()) if index.is_file() else json.loads(gzip.decompress(path.read_bytes()), object_hook=decode_value)
                if record.get('identity') == token and record.get('image_ids'):
                    record['archive_size'] = path.stat().st_size
                    result.append(record)
            except (OSError, ValueError, TypeError):
                continue
        return sorted(result, key=lambda r: r['created_at'], reverse=True)

    def remove(self, payload):
        key = payload.get('archive_id', '')
        if key and Path(key).name == key:
            path = self.directory / (key + '.json.gz')
            path.unlink(missing_ok=True)
            path.with_suffix('.meta.json').unlink(missing_ok=True)
