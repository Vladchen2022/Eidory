import json
from hashlib import sha256

from eidory.core.creative_entries import search_names
from eidory.core.creative_templates import creative_node_template, template_search_query, CREATIVE_TEMPLATES


def target_signature(node, entries):
    data = (node.search_query, node.note) if entries is None else (entries.get('base_query', ''), entries.get('base_note', ''), [(r['id'], r['name'], r['enabled']) for r in entries['rows']])
    return sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()


def initial_targets(node, entries, path=''):
    if entries is None:
        return [{'id': 'core', 'name': node.title, 'query': node.search_query or node.note, 'enabled': True, 'kind': '核心目标'}]
    rows = [{'id': r['id'], 'name': r['name'], 'query': r['name'], 'enabled': r['enabled'], 'kind': '参考条目'} for r in entries['rows']]
    base = entries.get('base_query', '').strip() or entries.get('base_note', '').strip().split('\n')[0]
    template = creative_node_template(node.title, path)
    hints = [n.note for t in CREATIVE_TEMPLATES for n in t.root.children]
    if any(hint and hint in base for hint in hints) or (template and base == template_search_query(template.title, template.note)):
        base = ''
    if base and base not in search_names(entries):
        # An explicit all-unchecked list means no search. Never silently broaden it.
        rows.insert(0, {'id': 'core', 'name': '原有核心内容', 'query': base[:180], 'enabled': bool(search_names(entries)), 'kind': '核心目标'})
    return rows


def resolved_targets(node, entries, options, path=''):
    fresh = initial_targets(node, entries, path)
    try:
        saved = json.loads(options.get('search_targets_json', ''))
    except (ValueError, TypeError):
        return fresh
    if saved.get('signature') == target_signature(node, entries):
        return saved['rows']
    # Keep a user's wording, but respect newly changed checkbox state and new rows.
    by_id = {r['id']: r for r in saved.get('rows', [])}
    for row in fresh:
        previous = by_id.get(row['id'])
        if previous and previous['name'] == row['name']:
            row['query'] = previous['query']
    fresh.extend(r for r in saved.get("rows", []) if r.get("kind") == "手动目标")
    return fresh
