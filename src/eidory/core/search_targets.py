import json
from hashlib import sha256

from eidory.core.creative_entries import search_names
from eidory.core.creative_templates import creative_node_template, template_search_query, CREATIVE_TEMPLATES


def target_signature(node, entries):
    data = (node.search_query, node.note) if entries is None else (entries.get('base_query', ''), entries.get('base_note', ''), [(r['id'], r['name'], r['enabled']) for r in entries['rows']])
    return sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()


def initial_targets(node, entries, path=''):
    template = creative_node_template(node.title, path)
    hints = [n.note for t in CREATIVE_TEMPLATES for n in t.root.children]
    def placeholder(value):
        return any(hint and hint in value for hint in hints) or (template and value == template_search_query(template.title, template.note))
    if entries is None:
        query = node.search_query or node.note
        if placeholder(query) and node.note and not placeholder(node.note):
            query = node.note[:180]
        return [{'id': 'core', 'name': node.title, 'query': query, 'enabled': True, '_source_enabled': True, 'kind': '核心目标'}]
    rows = [{'id': r['id'], 'name': r['name'], 'query': r['name'], 'enabled': r['enabled'], '_source_enabled': r['enabled'], 'kind': '参考条目'} for r in entries['rows']]
    base = entries.get('base_query', '').strip()
    if not base or placeholder(base):
        # Editing the note does not overwrite the default template query. Fall
        # back to the user's note instead of discarding their core subject.
        base = entries.get('base_note', '').strip().split('\n')[0]
        if placeholder(base):
            base = ''
    if base and base not in search_names(entries):
        enabled = bool(search_names(entries))
        rows.insert(0, {'id': 'core', 'name': '原有核心内容', 'query': base[:180], 'enabled': enabled, '_source_enabled': enabled, 'kind': '核心目标'})
    return rows


def resolved_targets(node, entries, options, path=''):
    fresh = initial_targets(node, entries, path)
    try:
        saved = json.loads(options.get('search_targets_json', ''))
    except (ValueError, TypeError):
        return fresh
    if not isinstance(saved, dict) or not isinstance(saved.get('rows'), list):
        return fresh
    if any(not isinstance(r, dict) or not all(isinstance(r.get(k), str) for k in ('id', 'name', 'query', 'kind')) or not isinstance(r.get('enabled'), bool) for r in saved['rows']):
        return fresh
    if saved.get('signature') == target_signature(node, entries):
        return saved['rows']
    by_id = {r['id']: r for r in saved['rows']}
    for row in fresh:
        previous = by_id.get(row['id'])
        if previous and previous['name'] == row['name']:
            row['query'] = previous['query']
            # Preserve independent search choices unless this source row's
            # checkbox actually changed. Unrelated edits must not re-enable it.
            if row['_source_enabled'] == previous.get('_source_enabled', previous['enabled']):
                row['enabled'] = previous['enabled']
    fresh.extend(r for r in saved['rows'] if r['kind'] == '手动目标')
    return fresh
