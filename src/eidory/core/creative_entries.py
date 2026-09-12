"""Structured reference ideas. Selection controls search, never the creative brief."""
from __future__ import annotations

import json
import re
from uuid import uuid4


def name_key(name):
    return re.sub(r"[\W_]+", "", name).casefold()


def new_entry(name, visual="", *, enabled=True):
    return {"id": uuid4().hex, "name": str(name).strip(), "visual": str(visual).strip(), "enabled": bool(enabled)}


def encode_entries(payload):
    return json.dumps(payload, ensure_ascii=False)


def decode_entries(raw):
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
            return None
        if any(not isinstance(row, dict) or not isinstance(row.get("id"), str)
               or not isinstance(row.get("name"), str) or not isinstance(row.get("visual"), str)
               or not isinstance(row.get("enabled"), bool) for row in payload["rows"]):
            return None
        return payload
    except (ValueError, TypeError):
        return None


def search_names(payload):
    return list(dict.fromkeys(row["name"].strip() for row in payload["rows"] if row["enabled"] and row["name"].strip()))


def composed_note(payload):
    parts = [payload.get("base_note", "").strip()]
    parts.extend(f'{row["name"]}: {row["visual"]}'.rstrip(": ") for row in payload["rows"] if row["name"].strip())
    return "\n".join(part for part in parts if part)
