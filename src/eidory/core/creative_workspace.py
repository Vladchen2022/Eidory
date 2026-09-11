"""Persistent workflow state, separate from library-wide image metadata."""
from __future__ import annotations

import json

from eidory.core.time_utils import utc_now_iso


class CreativeWorkspaceStore:
    def __init__(self, store):
        self.store = store

    def project(self, project_id):
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM creative_project_context WHERE project_id = ?", (project_id,)).fetchone()
        return dict(row) if row else {"extra": "", "scope_json": "", "probe_json": ""}

    def save_project(self, project_id, *, extra=None, scope=None, probe=None):
        if self.store.get_creative_project(project_id) is None:
            return
        current = self.project(project_id)
        with self.store.connect() as conn:
            conn.execute("""INSERT INTO creative_project_context(project_id, extra, scope_json, probe_json)
                VALUES (?, ?, ?, ?) ON CONFLICT(project_id) DO UPDATE SET
                extra=excluded.extra, scope_json=excluded.scope_json, probe_json=excluded.probe_json""",
                (project_id, current["extra"] if extra is None else extra,
                 current["scope_json"] if scope is None else json.dumps(scope, ensure_ascii=False),
                 current["probe_json"] if probe is None else json.dumps(probe, ensure_ascii=False)))

    def node(self, node_id):
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM creative_node_workflow WHERE node_id = ?", (node_id,)).fetchone()
        return dict(row) if row else {"locked": 0, "proposal_note": "", "proposal_query": "", "search_json": ""}

    def save_node(self, node_id, **changes):
        allowed = {"locked", "proposal_note", "proposal_query", "search_json"}
        if not changes or set(changes) - allowed or self.store.get_creative_node(node_id) is None:
            return
        with self.store.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO creative_node_workflow(node_id) VALUES (?)", (node_id,))
            conn.execute("UPDATE creative_node_workflow SET " + ", ".join(f"{key} = ?" for key in changes) + " WHERE node_id = ?",
                         (*changes.values(), node_id))

    def accept(self, node_id, *, note, query, keep_proposal=False):
        """Save previous accepted content and the replacement in one transaction."""
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM creative_nodes WHERE id = ?", (node_id,)).fetchone()
            state = conn.execute("SELECT locked FROM creative_node_workflow WHERE node_id = ?", (node_id,)).fetchone()
            if row is None or (state and state["locked"]):
                return False
            conn.execute("INSERT INTO creative_node_versions(node_id, note, search_query, created_at) VALUES (?, ?, ?, ?)",
                         (node_id, row["note"], row["search_query"], utc_now_iso()))
            conn.execute("UPDATE creative_nodes SET note = ?, search_query = ?, updated_at = ? WHERE id = ?",
                         (note, query, utc_now_iso(), node_id))
            conn.execute("UPDATE creative_projects SET updated_at = ? WHERE id = ?", (utc_now_iso(), row["project_id"]))
            if not keep_proposal:
                conn.execute("UPDATE creative_node_workflow SET proposal_note = '', proposal_query = '' WHERE node_id = ?", (node_id,))
        return True

    def restore(self, node_id):
        with self.store.connect() as conn:
            state = conn.execute("SELECT locked FROM creative_node_workflow WHERE node_id = ?", (node_id,)).fetchone()
            if state and state["locked"]:
                return False
            row = conn.execute("SELECT * FROM creative_node_versions WHERE node_id = ? ORDER BY id DESC LIMIT 1", (node_id,)).fetchone()
            if row is None:
                return False
            conn.execute("UPDATE creative_nodes SET note = ?, search_query = ?, updated_at = ? WHERE id = ?",
                         (row["note"], row["search_query"], utc_now_iso(), node_id))
            conn.execute("DELETE FROM creative_node_versions WHERE id = ?", (row["id"],))
        return True

    def version_count(self, node_id):
        with self.store.connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM creative_node_versions WHERE node_id = ?", (node_id,)).fetchone()[0]

    def reference_note(self, node_id, image_id):
        with self.store.connect() as conn:
            row = conn.execute("SELECT note FROM creative_reference_notes WHERE node_id = ? AND image_id = ?", (node_id, image_id)).fetchone()
        return str(row[0]) if row else ""

    def save_reference_note(self, node_id, image_id, note):
        with self.store.connect() as conn:
            exists = conn.execute("SELECT 1 FROM creative_node_images WHERE node_id = ? AND image_id = ?", (node_id, image_id)).fetchone()
            if exists:
                conn.execute("""INSERT INTO creative_reference_notes(node_id, image_id, note) VALUES (?, ?, ?)
                    ON CONFLICT(node_id, image_id) DO UPDATE SET note = excluded.note""", (node_id, image_id, note))

    def reference_notes(self, project_id):
        with self.store.connect() as conn:
            return [dict(row) for row in conn.execute("""SELECT r.*, n.title FROM creative_reference_notes r
                JOIN creative_nodes n ON n.id = r.node_id WHERE n.project_id = ?""", (project_id,)).fetchall()]


def decode_state(raw, default=None):
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else (default or {})
    except (ValueError, TypeError):
        return default or {}
