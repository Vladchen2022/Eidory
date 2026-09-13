"""Persistent workflow state, separate from library-wide image metadata."""
from __future__ import annotations

import json

from eidory.core.creative_entries import decode_entries, encode_entries, composed_note, search_names, name_key, new_entry
from eidory.core.creative_templates import creative_node_default_format
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
        return dict(row) if row else {"locked": 0, "proposal_note": "", "proposal_query": "", "search_json": "", "entries_json": "", "proposal_entries_json": "", "content_format": ""}

    def save_node(self, node_id, **changes):
        allowed = {"locked", "proposal_note", "proposal_query", "search_json", "entries_json", "proposal_entries_json", "content_format"}
        if not changes or set(changes) - allowed or self.store.get_creative_node(node_id) is None:
            return
        with self.store.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO creative_node_workflow(node_id) VALUES (?)", (node_id,))
            conn.execute("UPDATE creative_node_workflow SET " + ", ".join(f"{key} = ?" for key in changes) + " WHERE node_id = ?",
                         (*changes.values(), node_id))

    def content_format(self, node_id, node_title, node_path=""):
        state = self.node(node_id)
        if state["content_format"] in {"text", "entries"}:
            return state["content_format"]
        # Existing rows remain usable until the user explicitly changes format.
        if state["entries_json"]:
            return "entries"
        return creative_node_default_format(node_title, node_path)

    def accept(self, node_id, *, note, query, keep_proposal=False, entries_json=None, content_format=None):
        """Save previous accepted content and the replacement in one transaction."""
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM creative_nodes WHERE id = ?", (node_id,)).fetchone()
            state = conn.execute("SELECT * FROM creative_node_workflow WHERE node_id = ?", (node_id,)).fetchone()
            if row is None or (state and state["locked"]):
                return False
            conn.execute("INSERT INTO creative_node_versions(node_id, note, search_query, entries_json, content_format, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                         (node_id, row["note"], row["search_query"], state["entries_json"] if state else "", state["content_format"] if state else "", utc_now_iso()))
            conn.execute("UPDATE creative_nodes SET note = ?, search_query = ?, updated_at = ? WHERE id = ?",
                         (note, query, utc_now_iso(), node_id))
            conn.execute("UPDATE creative_projects SET updated_at = ? WHERE id = ?", (utc_now_iso(), row["project_id"]))
            conn.execute("INSERT OR IGNORE INTO creative_node_workflow(node_id) VALUES (?)", (node_id,))
            if content_format is not None:
                conn.execute("UPDATE creative_node_workflow SET content_format = ?, search_json = '' WHERE node_id = ?", (content_format, node_id))
            if entries_json is not None:
                conn.execute("UPDATE creative_node_workflow SET entries_json = ?, search_json = '' WHERE node_id = ?", (entries_json, node_id))
            if not keep_proposal:
                conn.execute("UPDATE creative_node_workflow SET proposal_note = '', proposal_query = '', proposal_entries_json = '' WHERE node_id = ?", (node_id,))
        return True

    def restore(self, node_id):
        with self.store.connect() as conn:
            state = conn.execute("SELECT * FROM creative_node_workflow WHERE node_id = ?", (node_id,)).fetchone()
            if state and state["locked"]:
                return False
            row = conn.execute("SELECT * FROM creative_node_versions WHERE node_id = ? ORDER BY id DESC LIMIT 1", (node_id,)).fetchone()
            if row is None:
                return False
            conn.execute("UPDATE creative_nodes SET note = ?, search_query = ?, updated_at = ? WHERE id = ?",
                         (row["note"], row["search_query"], utc_now_iso(), node_id))
            conn.execute("INSERT OR IGNORE INTO creative_node_workflow(node_id) VALUES (?)", (node_id,))
            conn.execute("UPDATE creative_node_workflow SET entries_json = ?, content_format = ?, search_json = '' WHERE node_id = ?", (row["entries_json"], row["content_format"], node_id))
            conn.execute("DELETE FROM creative_node_versions WHERE id = ?", (row["id"],))
        return True

    def entries(self, node_id):
        return decode_entries(self.node(node_id)["entries_json"])

    def stage_entries(self, node_id, additions, *, mode="append", target=None):
        state = self.node(node_id)
        if state["locked"] or state["proposal_note"] or state["proposal_entries_json"]:
            return False
        current = self.entries(node_id)
        known = {name_key(row["name"]) for row in (current or {}).get("rows", [])}
        rows = []
        for idea in additions:
            key = name_key(idea["name"])
            if key and (target or key not in known):
                rows.append(new_entry(target["name"] if target else idea["name"], idea["visual"],
                                      enabled=target["enabled"] if target else True))
                known.add(key)
        if target:
            detail = "\n".join(row["visual"] for row in rows)
            if target["visual"].strip() and target["visual"].strip() not in detail:
                detail = target["visual"].rstrip() + "\n" + detail
            rows = [new_entry(target["name"], detail, enabled=target["enabled"])] if rows else []
        if not rows:
            return False
        proposal = {"rows": rows, "mode": mode, "target": target}
        self.save_node(node_id, proposal_entries_json=encode_entries(proposal),
                       proposal_note=composed_note(proposal), proposal_query=" ".join(search_names(proposal))[:240])
        return True

    def accept_entries(self, node_id):
        state = self.node(node_id)
        proposal = decode_entries(state["proposal_entries_json"])
        node = self.store.get_creative_node(node_id)
        if not proposal or not node or state["locked"]:
            return False
        payload = self.entries(node_id) or {"base_note": node.note, "base_query": node.search_query, "rows": []}
        if proposal.get("target"):
            target = proposal["target"]
            row = next((r for r in payload["rows"] if r["id"] == target["id"]), None)
            # Changes made during generation must never be overwritten silently.
            if row is None or any(row[k] != target[k] for k in ("name", "visual")):
                return False
            row["visual"] = proposal["rows"][0]["visual"]
        else:
            known = {name_key(row["name"]) for row in payload["rows"]}
            for row in proposal["rows"]:
                if not row["name"].strip():
                    return False
                if name_key(row["name"]) not in known:
                    payload["rows"].append(row)
                    known.add(name_key(row["name"]))
        return self.accept(node_id, note=composed_note(payload), query=" ".join(search_names(payload))[:240],
                           entries_json=encode_entries(payload), content_format="entries")

    def save_entries(self, node_id, payload, *, snapshot=False):
        old = self.entries(node_id)
        if self.node(node_id)["locked"]:
            # A locked brief still permits selecting its reference search targets.
            def content(value):
                return {**value, "rows": [{k: v for k, v in r.items() if k != "enabled"} for r in value["rows"]]}
            if old is None or content(old) != content(payload):
                return False
        elif snapshot:
            return self.accept(node_id, note=composed_note(payload), query=" ".join(search_names(payload))[:240],
                               keep_proposal=True, entries_json=encode_entries(payload))
        with self.store.connect() as conn:
            conn.execute("UPDATE creative_nodes SET note = ?, search_query = ?, updated_at = ? WHERE id = ?",
                         (composed_note(payload), " ".join(search_names(payload))[:240], utc_now_iso(), node_id))
            conn.execute("INSERT OR IGNORE INTO creative_node_workflow(node_id) VALUES (?)", (node_id,))
            conn.execute("UPDATE creative_node_workflow SET entries_json = ?, search_json = '' WHERE node_id = ?", (encode_entries(payload), node_id))
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
