"""Complete project state and decisions, independent of widget lifetime."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from eidory.core.time_utils import utc_now_iso

NODE_TABLES = (
    'creative_node_workflow', 'creative_node_options', 'creative_node_versions',
    'creative_node_images', 'creative_reference_notes', 'creative_reference_decisions',
    'creative_node_board_layouts',
)
PROJECT_TABLES = ('creative_project_context', 'creative_board_layouts')
REFERENCE_STATUSES = {'candidate': '备选', 'primary': '主参考', 'unused': '暂不采用'}
PURPOSES = ('', '空间', '造型', '动作', '材质', '光线', '构图')


def capture_project(store, project_id, *, conn=None):
    if conn is None:
        with store.connect() as connection:
            connection.execute("BEGIN")
            return capture_project(store, project_id, conn=connection)
    row = conn.execute('SELECT * FROM creative_projects WHERE id=?', (project_id,)).fetchone()
    if row is None:
        raise ValueError('项目不存在')
    nodes = [dict(r) for r in conn.execute('SELECT * FROM creative_nodes WHERE project_id=? ORDER BY id', (project_id,))]
    tables = {}
    for table in NODE_TABLES:
        tables[table] = [dict(r) for r in conn.execute(
            f'SELECT * FROM {table} WHERE node_id IN (SELECT id FROM creative_nodes WHERE project_id=?)', (project_id,))]
    for table in PROJECT_TABLES:
        tables[table] = [dict(r) for r in conn.execute(f'SELECT * FROM {table} WHERE project_id=?', (project_id,))]
    return {'version': 1, 'project': dict(row), 'nodes': nodes, 'tables': tables}


def _insert(conn, table, row):
    # Table names come only from the fixed schema above; reject unknown columns.
    columns = {r['name'] for r in conn.execute(f'PRAGMA table_info({table})')}
    if not row or set(row) - columns:
        raise ValueError(f'项目包字段不兼容：{table}')
    keys = list(row)
    return conn.execute(f'INSERT INTO {table} ({",".join(keys)}) VALUES ({",".join("?" for _ in keys)})',
                        tuple(row[k] for k in keys)).lastrowid


def validate_state(payload):
    if not isinstance(payload, dict) or payload.get('version') != 1:
        raise ValueError('不支持的项目状态版本')
    if not isinstance(payload.get('project'), dict) or not isinstance(payload.get('nodes'), list):
        raise ValueError('项目状态缺少节点或项目信息')
    nodes = payload['nodes']
    if not nodes or len(nodes) > 10000:
        raise ValueError('项目节点数量无效')
    ids = {n['id'] for n in nodes}
    if len(ids) != len(nodes) or any(not isinstance(i, int) or i <= 0 for i in ids):
        raise ValueError('项目节点 ID 重复或无效')
    parents = {n['id']: n.get('parent_id') for n in nodes}
    for nid in ids:
        seen = set()
        while nid is not None:
            if nid not in parents or nid in seen:
                raise ValueError('项目节点层级无效')
            seen.add(nid)
            nid = parents[nid]
    tables = payload.get('tables', {})
    if not isinstance(tables, dict) or set(tables) - set(NODE_TABLES + PROJECT_TABLES):
        raise ValueError('项目状态包含未知数据表')
    for table, rows in tables.items():
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise ValueError('项目状态行格式无效')
        if table in NODE_TABLES and any(r.get('node_id') not in ids for r in rows):
            raise ValueError('项目状态引用了不存在的节点')
    return payload


def remap_layout(raw, image_map):
    if not raw:
        return raw
    layout = json.loads(raw)
    layout['items'] = {str(image_map[int(k)]): v for k, v in layout.get('items', {}).items() if int(k) in image_map}
    for section in layout.get('sections', []):
        section['image_ids'] = [image_map[i] for i in section.get('image_ids', []) if i in image_map]
    return json.dumps(layout, ensure_ascii=False)


def remap_project_state(payload, project_id, node_map, image_map=None):
    state = deepcopy(payload)
    state['project']['id'] = project_id
    for node in state['nodes']:
        node['id'] = node_map[node['id']]
        node['parent_id'] = node_map.get(node.get('parent_id'))
        node['project_id'] = project_id
    for table, rows in state['tables'].items():
        for row in rows:
            if 'node_id' in row:
                row['node_id'] = node_map[row['node_id']]
            if 'project_id' in row:
                row['project_id'] = project_id
            if image_map is not None:
                if 'image_id' in row:
                    row['image_id'] = image_map[row['image_id']]
                if table in ('creative_board_layouts', 'creative_node_board_layouts'):
                    row['payload_json'] = remap_layout(row['payload_json'], image_map)
                if table == 'creative_node_workflow':
                    row['search_json'] = ''
                if table == 'creative_project_context':
                    row['scope_json'] = ''
    return state


def install_state(conn, payload, *, project_id=None, image_map=None, mappings=None):
    """Caller owns the transaction. Restore in place or import with fresh IDs."""
    validate_state(payload)
    state = deepcopy(payload)
    project = state['project']
    importing = project_id is None
    if importing:
        project.pop('id', None)
        project['created_at'] = project['updated_at'] = utc_now_iso()
        project_id = _insert(conn, 'creative_projects', project)
    else:
        # Preserve IDs for saved search context, board sections and other node references.
        ids = {n['id'] for n in state['nodes']}
        if any(r['id'] in ids and r['project_id'] != project_id for r in conn.execute('SELECT id,project_id FROM creative_nodes')):
            highest = conn.execute('SELECT COALESCE(MAX(id),0) FROM creative_nodes').fetchone()[0]
            payload = remap_project_state(payload, project_id, {old: highest+i+1 for i, old in enumerate(sorted(ids))})
            state = deepcopy(payload)
            project = state['project']
        conn.execute('DELETE FROM creative_nodes WHERE project_id=?', (project_id,))
        for table in PROJECT_TABLES:
            conn.execute(f'DELETE FROM {table} WHERE project_id=?', (project_id,))
        fields = {k: v for k, v in project.items() if k != 'id'}
        fields['updated_at'] = utc_now_iso()
        columns = {r['name'] for r in conn.execute('PRAGMA table_info(creative_projects)')}
        if set(fields) - columns:
            raise ValueError('项目字段不兼容')
        conn.execute('UPDATE creative_projects SET '+','.join(f'{k}=?' for k in fields)+' WHERE id=?', (*fields.values(), project_id))
    node_map = {}
    for node in state['nodes']:
        old_id = node['id']
        node['project_id'] = project_id
        node['parent_id'] = None
        if importing:
            node.pop('id')
        node_map[old_id] = _insert(conn, 'creative_nodes', node)
    for node in payload['nodes']:
        conn.execute('UPDATE creative_nodes SET parent_id=? WHERE id=?',
                     (node_map.get(node.get('parent_id')), node_map[node['id']]))
    for table in NODE_TABLES + PROJECT_TABLES:
        for row in state['tables'].get(table, []):
            if 'project_id' in row:
                row['project_id'] = project_id
            if 'node_id' in row:
                row['node_id'] = node_map[row['node_id']]
            if table == 'creative_node_versions':
                row.pop('id', None)
            if image_map is not None:
                if 'image_id' in row:
                    row['image_id'] = image_map[row['image_id']]
                if 'payload_json' in row:
                    row['payload_json'] = remap_layout(row['payload_json'], image_map)
                # Search filters and scope refer to a different library. The independent
                # search-target definitions remain; clear transient library-specific filters.
                if table == 'creative_node_workflow':
                    row['search_json'] = ''
                if table == 'creative_project_context':
                    row['scope_json'] = ''
            _insert(conn, table, row)
    if mappings is not None:
        mappings.update(node_map)
    return project_id


class ProjectStateStore:
    def __init__(self, store):
        self.store = store

    def options(self, node_id):
        with self.store.connect() as c:
            r = c.execute('SELECT * FROM creative_node_options WHERE node_id=?', (node_id,)).fetchone()
        return dict(r) if r else {'search_targets_json': '', 'reference_requirement': 'needed', 'reviewed_context': ''}

    def set_options(self, node_id, **changes):
        if set(changes) - {'search_targets_json', 'reference_requirement', 'reviewed_context'}:
            raise ValueError('未知的节点选项')
        if 'reference_requirement' in changes and changes['reference_requirement'] not in {'needed', 'none', 'resolved'}:
            raise ValueError('未知的参考状态')
        with self.store.connect() as c:
            c.execute('INSERT OR IGNORE INTO creative_node_options(node_id) VALUES (?)', (node_id,))
            if changes:
                c.execute('UPDATE creative_node_options SET '+','.join(f'{k}=?' for k in changes)+' WHERE node_id=?', (*changes.values(), node_id))

    def decision(self, node_id, image_id):
        with self.store.connect() as c:
            r = c.execute('SELECT * FROM creative_reference_decisions WHERE node_id=? AND image_id=?', (node_id, image_id)).fetchone()
        return dict(r) if r else {'status': 'candidate', 'purpose': ''}

    def set_decision(self, node_id, image_id, status, purpose):
        if status not in REFERENCE_STATUSES or purpose not in PURPOSES:
            raise ValueError('未知的参考等级或用途')
        with self.store.connect() as c:
            c.execute('INSERT INTO creative_reference_decisions(node_id,image_id,status,purpose) VALUES (?,?,?,?) '
                      'ON CONFLICT(node_id,image_id) DO UPDATE SET status=excluded.status,purpose=excluded.purpose',
                      (node_id, image_id, status, purpose))

    def context_key(self, project_id):
        with self.store.connect() as c:
            r = c.execute('SELECT p.brief,COALESCE(x.extra,\'\') FROM creative_projects p LEFT JOIN creative_project_context x ON x.project_id=p.id WHERE p.id=?', (project_id,)).fetchone()
        return hashlib.sha256(json.dumps(tuple(r) if r else [], ensure_ascii=False).encode()).hexdigest()

    def snapshot(self, project_id, title):
        with self.store.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            payload = capture_project(self.store, project_id, conn=c)
            return c.execute('INSERT INTO creative_project_snapshots(project_id,title,created_at,payload_json) VALUES (?,?,?,?)',
                             (project_id, title.strip()[:120] or '未命名方案', utc_now_iso(), json.dumps(payload, ensure_ascii=False))).lastrowid

    def versions(self, project_id):
        with self.store.connect() as c:
            return [dict(r) for r in c.execute('SELECT id,title,created_at FROM creative_project_snapshots WHERE project_id=? ORDER BY id DESC', (project_id,))]

    def version(self, project_id, version_id):
        with self.store.connect() as c:
            r = c.execute('SELECT payload_json FROM creative_project_snapshots WHERE project_id=? AND id=?', (project_id, version_id)).fetchone()
        if not r:
            raise ValueError('方案不存在')
        return json.loads(r[0])

    def restore(self, project_id, version_id):
        payload = self.version(project_id, version_id)
        with self.store.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            before = capture_project(self.store, project_id, conn=c)
            c.execute('INSERT INTO creative_project_snapshots(project_id,title,created_at,payload_json) VALUES (?,?,?,?)',
                      (project_id, '恢复前自动保存', utc_now_iso(), json.dumps(before, ensure_ascii=False)))
            install_state(c, payload, project_id=project_id)

    def overview(self, project_id):
        key = self.context_key(project_id)
        with self.store.connect() as c:
            return [dict(r) for r in c.execute('''SELECT n.id,n.title,n.parent_id,
                COALESCE(w.locked,0) locked,COALESCE(w.proposal_note,'') proposal,
                COALESCE(w.proposal_entries_json,'') proposal_entries,
                COALESCE(o.reference_requirement,'needed') requirement,
                CASE WHEN COALESCE(o.reviewed_context,'') NOT IN ('',?) THEN 1 ELSE 0 END needs_review,
                (SELECT COUNT(*) FROM creative_node_images i WHERE i.node_id=n.id) image_count,
                (SELECT COUNT(*) FROM creative_reference_decisions d WHERE d.node_id=n.id AND d.status='primary') primary_count
                FROM creative_nodes n LEFT JOIN creative_node_workflow w ON w.node_id=n.id
                LEFT JOIN creative_node_options o ON o.node_id=n.id WHERE n.project_id=? ORDER BY n.sort_order,n.id''', (key, project_id))]


def reference_badge_text(badges):
    for status in ('主参考', '暂不采用', '备选'):
        if status in badges:
            return ' · '.join([status] + [p for p in PURPOSES if p and p in badges])
    if not badges:
        return ''
    return badges[0] + (f' +{len(badges)-1}' if len(badges)>1 else '')
