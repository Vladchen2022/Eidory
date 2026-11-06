"""Review decisions are bound to file identities, never to a group number."""
import hashlib
import json
from eidory.core.time_utils import utc_now_iso


def fingerprint(group):
    values = [(m.image.id, m.file_sha256, m.image.file_size, m.image.modified_time_ns) for m in group.members]
    return hashlib.sha256(json.dumps([group.kind, sorted(values)], ensure_ascii=False).encode()).hexdigest()


class DuplicateReview:
    def __init__(self, store):
        self.store = store

    def decisions(self):
        with self.store.connect(readonly=True) as conn:
            return dict(conn.execute('SELECT fingerprint,decision FROM duplicate_reviews'))

    def mark(self, group, decision):
        if decision not in {'keep', 'different', 'later', 'pending'}:
            raise ValueError('无效的审核状态')
        with self.store.connect() as conn:
            conn.execute('INSERT INTO duplicate_reviews VALUES(?,?,?) ON CONFLICT(fingerprint) DO UPDATE SET decision=excluded.decision,updated_at=excluded.updated_at',
                         (fingerprint(group), decision, utc_now_iso()))

    def cursor(self):
        try:
            return json.loads(self.store.get_setting('duplicate.review_cursor', 'null'))
        except ValueError:
            return None

    def save_cursor(self, group, candidate):
        self.store.set_setting('duplicate.review_cursor', json.dumps([fingerprint(group), candidate]))
