"""Local ComfyUI jobs and durable candidates, separate from indexed library images."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import time
from urllib.parse import urlparse
import uuid

from PIL import Image
import requests

from eidory.core.creative_entries import decode_entries
from eidory.core.creative_workspace import CreativeWorkspaceStore
from eidory.core.creative_templates import CREATIVE_TEMPLATES
from eidory.core.thumbnailer import Thumbnailer
from eidory.core.time_utils import utc_now_iso


ACTIVE_STATES = {'submitting', 'unknown', 'queued', 'running', 'downloading', 'offline'}
STATE_LABELS = {
    'submitting': '正在提交', 'unknown': '连接中断，正在核对原任务',
    'queued': '等待 ComfyUI', 'running': '正在生成', 'downloading': '正在保存候选',
    'ready': '待采用', 'adopted': '已采用并保存', 'discarded': '已丢弃，可撤销',
    'offline': '服务离线，等待重连', 'paused': '已停止等待，服务端状态未确认',
    'cancelled': '已取消', 'failed': '生成失败', 'missing': '服务中未找到原任务',
}


def connection_message(exc):
    if isinstance(exc, requests.ConnectionError):
        return '无法连接本机 ComfyUI。请启动 ComfyUI，并检查服务地址和端口。'
    if isinstance(exc, requests.Timeout):
        return 'ComfyUI 响应超时。请检查服务是否繁忙；已有任务不会重复提交。'
    return str(exc)


def _template_notes():
    notes = set()
    def walk(node):
        notes.add(node.note.strip())
        for child in node.children:
            walk(child)
    for template in CREATIVE_TEMPLATES:
        walk(template.root)
    return notes


TEMPLATE_NOTES = _template_notes()


def local_url(value: str) -> str:
    value = value.strip().rstrip('/')
    parsed = urlparse(value)
    if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise ValueError('请输入本机 ComfyUI 地址，例如 http://127.0.0.1:8188')
    return value


def prepare_template(graph: dict) -> dict:
    """Bind one text-to-image output; never submit unrelated branches/output nodes."""
    if not isinstance(graph, dict) or not graph or 'nodes' in graph:
        raise ValueError('需要 ComfyUI 的 API 格式 JSON，请使用「导出（API）」保存工作流。')
    if not all(isinstance(n, dict) and isinstance(n.get('inputs'), dict) and n.get('class_type') for n in graph.values()):
        raise ValueError('工作流不是有效的 API 格式。')
    outputs = [k for k, n in graph.items() if n['class_type'] in {'SaveImage', 'PreviewImage'}]
    if len(outputs) != 1:
        raise ValueError('首版支持一个图片输出节点的文生图工作流。')
    output = outputs[0]
    reachable = set()
    def visit(key):
        if key in reachable:
            return
        if key not in graph:
            raise ValueError('工作流包含不存在的节点连接。')
        reachable.add(key)
        for value in graph[key]['inputs'].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                visit(value[0])
    visit(output)
    graph = {k: deepcopy(v) for k, v in graph.items() if k in reachable}
    samplers = [k for k, n in graph.items() if n['class_type'] in {'KSampler', 'SamplerCustomAdvanced'}]
    if len(samplers) != 1:
        raise ValueError('需要一个 KSampler 或受支持的高级采样节点。')
    sampler = samplers[0]
    try:
        seed_node, seed_field = sampler, 'seed'
        advanced = graph[sampler]['class_type'] == 'SamplerCustomAdvanced'
        if advanced:
            guider = graph[sampler]['inputs']['guider'][0]
            noise = graph[sampler]['inputs']['noise'][0]
            sigmas = graph[sampler]['inputs']['sigmas'][0]
            sampler_select = graph[sampler]['inputs']['sampler'][0]
            assert graph[guider]['class_type'] == 'BasicGuider'
            assert graph[noise]['class_type'] == 'RandomNoise'
            assert graph[sigmas]['class_type'] in {'ViggleTurboSigmas', 'ManualSigmas'}
            assert graph[sampler_select]['class_type'] == 'KSamplerSelect'
            positive = graph[guider]['inputs']['conditioning'][0]
            seed_node, seed_field = noise, 'noise_seed'
        else:
            positive = graph[sampler]['inputs']['positive'][0]
        latent = graph[sampler]['inputs']['latent_image'][0]
        if advanced and graph[sigmas]['class_type'] == 'ViggleTurboSigmas':
            assert graph[sigmas]['inputs']['latent'] == graph[sampler]['inputs']['latent_image']
        encoder_type = graph[positive]['class_type']
        field = {'TextEncodeQwenImage21': 'prompt', 'CLIPTextEncode': 'text'}[encoder_type]
        assert graph[latent]['class_type'] in {'EmptyLatentImage', 'EmptySD3LatentImage'}
        assert not any(n['class_type'] == 'LoadImage' for n in graph.values())
    except (KeyError, IndexError, TypeError, AssertionError):
        raise ValueError('请选择直接连接文字编码和空白画布的文生图工作流；高级采样支持 BasicGuider、RandomNoise 与 ViggleTurboSigmas。') from None
    # PreviewImage is a transport cache, not a permanent ComfyUI output. The
    # original file and its sampling parameters are never modified.
    graph[output] = {'class_type': 'PreviewImage', 'inputs': {'images': graph[output]['inputs']['images']}}
    graph[positive]['inputs'][field] = ''
    graph[latent]['inputs']['batch_size'] = 1
    return {'graph': graph, 'prompt_node': positive, 'prompt_field': field,
            'sampler_node': sampler, 'seed_node': seed_node, 'seed_field': seed_field,
            'size_multiple': 32 if advanced else 8, 'latent_node': latent, 'output_node': output}


def sampling_description(template):
    graph = template['graph']
    sampler = graph[template['sampler_node']]
    values = sampler['inputs']
    if sampler['class_type'] == 'SamplerCustomAdvanced':
        schedule = graph[values['sigmas'][0]]
        sigmas = schedule['inputs'].get('nodes', schedule['inputs'].get('sigmas', ''))
        steps = len([v for v in sigmas.split(',') if v.strip()]) - (schedule['class_type'] == 'ManualSigmas')
        method = graph[values['sampler'][0]]['inputs']['sampler_name']
        return f'{steps} 步 · {method} · Turbo 专用调度 · 无额外 CFG · 每次新种子'
    return f"{values['steps']} 步 · {values['sampler_name']} / {values['scheduler']} · CFG {values['cfg']} · 每次新种子"


def build_prompt(store, node_id: int) -> str:
    node = store.get_creative_node(node_id)
    if node is None:
        raise ValueError('请先选择创作节点。')
    project = store.get_creative_project(node.project_id)
    nodes = store.list_creative_nodes(node.project_id)
    by_id = {n.id: n for n in nodes}
    chain, cursor, visited = [], node, set()
    while cursor and cursor.id not in visited:
        visited.add(cursor.id)
        chain.append(cursor.title)
        cursor = by_id.get(cursor.parent_id)
    parts = [f'项目主题：{project.brief or project.title}', f'本次只生成：{" / ".join(reversed(chain[1:-1])) + " / " if len(chain) > 2 else ""}{node.title}']
    for n in nodes:
        if n.title == '世界观' and n.id != node.id and n.note.strip() and n.note.strip() not in TEMPLATE_NOTES:
            parts.append('世界观：' + n.note.strip())
    workspace = CreativeWorkspaceStore(store)
    extra = workspace.project(node.project_id).get('extra', '').strip()
    if extra:
        parts.append('已确定的项目约束：' + extra)
    entries = decode_entries(workspace.node(node.id).get('entries_json', ''))
    note = entries.get('base_note', '') if entries else node.note
    if note.strip() and note.strip() not in TEMPLATE_NOTES:
        parts.append('当前设定：' + note.strip())
    for row in entries['rows'] if entries else []:
        if row.get('enabled', True):
            parts.append(row['name'] + ('：' + row.get('visual', '') if row.get('visual') else ''))
    parts.append('请生成可用于绘画构思的清晰视觉参考，重点表现当前对象，不添加其他节点的独立方案。')
    return '\n\n'.join(parts)


class ComfyClient:
    def __init__(self, base_url: str):
        self.url = local_url(base_url)
        self.session = requests.Session()
        self.session.trust_env = False

    def request(self, method, route, **kwargs):
        response = self.session.request(method, self.url + route, timeout=(3, 12), **kwargs)
        if response.status_code >= 400:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text[:1000]
            raise ValueError(f'ComfyUI 返回 {response.status_code}：{detail}')
        return response.json()

    def validate(self, template):
        self.request('GET', '/system_stats')
        schemas = {}
        for node in template['graph'].values():
            kind = node['class_type']
            if kind not in schemas:
                schemas[kind] = self.request('GET', '/object_info/' + kind).get(kind)
            schema = schemas[kind]
            if not schema:
                raise ValueError('ComfyUI 缺少节点：' + kind)
            for field, rule in schema.get('input', {}).get('required', {}).items():
                if kind in {'LoadImage', 'LoadImageMask'} and field == 'image':
                    # The upload route supports subfolders which aren't enumerated
                    # by the UI combo. /prompt validates the actual uploaded path.
                    continue
                value = node['inputs'].get(field)
                if isinstance(rule[0], list) and value is not None and not isinstance(value, list) and value not in rule[0]:
                    raise ValueError(f'{kind} 的 {field} 不可用：{value}')
        return '连接成功，工作流节点与模型可用'

    def download(self, descriptor, target: Path):
        response = self.session.get(self.url + '/view', params=descriptor, timeout=(3, 30), stream=True)
        response.raise_for_status()
        temporary = target.with_suffix('.part')
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            size = 0
            with temporary.open('wb') as out:
                for chunk in response.iter_content(256 * 1024):
                    size += len(chunk)
                    if size > 100 * 1024 * 1024:
                        raise ValueError('生成结果过大，超过 100 MB。')
                    out.write(chunk)
            with Image.open(temporary) as image:
                image.verify()
            os.replace(temporary, target)
        finally:
            response.close()
            temporary.unlink(missing_ok=True)


class GeneratedReferences:
    def __init__(self, store, paths):
        self.store = store
        self.paths = paths
        self.root = paths.data_dir / 'generated-references'

    def all(self, project_id=None):
        query = "SELECT json_remove(payload, '$.workflow') FROM generated_reference_jobs"
        args = ()
        if project_id is not None:
            query += ' WHERE project_id=?'
            args = (project_id,)
        with self.store.connect(readonly=True) as conn:
            return [json.loads(row[0]) for row in conn.execute(query + ' ORDER BY created_at DESC, id DESC', args)]

    def references_for_review(self):
        """Keep detached results recoverable, with their live destination status."""
        with self.store.connect(readonly=True) as conn:
            rows = conn.execute("""
                SELECT json_remove(j.payload, '$.workflow') AS payload,
                       CASE WHEN p.id IS NULL THEN '原项目已删除'
                            WHEN n.id IS NULL THEN '原节点已删除'
                            ELSE '' END AS detached_reason
                FROM generated_reference_jobs j
                LEFT JOIN creative_projects p ON p.id = j.project_id
                LEFT JOIN creative_nodes n ON n.project_id = j.project_id
                    AND n.id = json_extract(j.payload, '$.node_id')
                WHERE COALESCE(json_extract(j.payload, '$.origin'), '') <> 'preview'
                ORDER BY j.created_at DESC, j.id DESC
            """).fetchall()
        return [dict(json.loads(row['payload']), detached_reason=row['detached_reason']) for row in rows]

    def active(self):
        with self.store.connect(readonly=True) as conn:
            row = conn.execute("SELECT json_remove(payload, '$.workflow') FROM generated_reference_jobs WHERE json_extract(payload, '$.state') IN ('submitting','unknown','queued','running','downloading','offline') ORDER BY created_at LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None

    def get(self, job_id):
        with self.store.connect(readonly=True) as conn:
            row = conn.execute('SELECT payload FROM generated_reference_jobs WHERE id=?', (job_id,)).fetchone()
        if not row:
            raise ValueError('生成记录不存在。')
        return json.loads(row[0])

    def save(self, job, conn=None):
        values = (job['id'], job['project_id'], job['created_at'], json.dumps(job, ensure_ascii=False))
        sql = '''INSERT INTO generated_reference_jobs(id,project_id,created_at,payload) VALUES(?,?,?,?)
                 ON CONFLICT(id) DO UPDATE SET payload=excluded.payload'''
        if conn is not None:
            conn.execute(sql, values)
        else:
            with self.store.connect() as conn:
                conn.execute(sql, values)

    def create(self, node_id, prompt, template, url, *, width=None, height=None):
        if self.active():
            raise ValueError('请等待当前生成完成，或先取消当前任务。')
        with self.store.connect(readonly=True) as conn:
            if conn.execute("SELECT 1 FROM image_upscale_jobs WHERE json_extract(payload,'$.state') IN ('submitting','unknown','queued','running','downloading','offline') LIMIT 1").fetchone():
                raise ValueError('请等待当前 AI 放大完成，或先取消放大任务。')
        node = self.store.get_creative_node(node_id)
        if not node or not prompt.strip():
            raise ValueError('需要有效的创作节点和图片描述。')
        project = self.store.get_creative_project(node.project_id)
        graph = deepcopy(template['graph'])
        graph[template['prompt_node']]['inputs'][template['prompt_field']] = prompt.strip()
        seed = secrets.randbelow(2**53)
        graph[template.get('seed_node', template['sampler_node'])]['inputs'][template.get('seed_field', 'seed')] = seed
        multiple = template.get('size_multiple', 8)
        for field, value in [('width', width), ('height', height)]:
            value = graph[template['latent_node']]['inputs'][field] if value is None else value
            if value < 256 or value > 4096 or value % multiple:
                raise ValueError(f'图片尺寸需为 256–4096 之间的 {multiple} 的倍数。')
            graph[template['latent_node']]['inputs'][field] = value
        key = str(uuid.uuid4())
        job = dict(id=key, prompt_id=key, project_id=node.project_id, node_id=node.id,
                   project_title=project.title, node_title=node.title, prompt=prompt.strip(),
                   seed=seed, workflow=graph, output_node=template['output_node'], url=local_url(url),
                   workflow_id=template.get('workflow_id', 'custom'), workflow_name=template.get('workflow_name', ''),
                   created_at=utc_now_iso(), submitted_at=time.time(), state='submitting', error='',
                   client_id='eidory-' + key)
        self.save(job)  # Durable identity exists before the HTTP request.
        return job

    def submit(self, job_id):
        job = self.get(job_id)
        if job['state'] != 'submitting':
            return job
        client = ComfyClient(job['url'])
        try:
            client.validate({'graph': job['workflow']})
        except (requests.RequestException, ValueError) as exc:
            job.update(state='failed', error=connection_message(exc))
            self.save(job)
            return job
        try:
            result = client.request('POST', '/prompt', json={
                'prompt_id': job['prompt_id'], 'client_id': job['client_id'],
                'prompt': job['workflow'], 'extra_data': {'eidory_job_id': job['id']}})
            job.update(prompt_id=result['prompt_id'], state='queued', error='')
        except requests.RequestException:
            job.update(state='unknown', error='未确认提交结果，正在核对原任务；不会自动重复提交。')
        except ValueError as exc:
            job.update(state='failed', error=str(exc))
        self.save(job)
        return job

    def reconcile(self, job_id):
        job = self.get(job_id)
        if job['state'] not in ACTIVE_STATES:
            return job
        if job.get('next_check_at', 0) > time.time():
            return job
        client = ComfyClient(job['url'])
        try:
            history = client.request('GET', '/history/' + job['prompt_id'])
            result = history.get(job['prompt_id'])
            if result:
                descriptors = result.get('outputs', {}).get(job['output_node'], {}).get('images', [])
                status = result.get('status', {})
                if status.get('status_str') == 'error':
                    interrupted = any(m[0] == 'execution_interrupted' for m in status.get('messages', []))
                    job.update(state='cancelled' if interrupted else 'failed', error='' if interrupted else str(status.get('messages', []))[-1800:])
                elif descriptors:
                    if len(descriptors) != 1:
                        raise ValueError('工作流返回了多张图片；请将输出批量设为 1。')
                    job['state'] = 'downloading'
                    self.save(job)
                    target = self.root / 'candidates' / str(job['project_id']) / (job['id'] + '.png')
                    client.download(descriptors[0], target)
                    thumb = target.with_suffix('.webp')
                    with Image.open(target) as image:
                        job['width'], job['height'] = image.size
                        image.thumbnail((480, 480))
                        image.convert('RGB').save(thumb, 'WEBP', quality=85)
                    job.update(state='ready', file_path=str(target), thumbnail_path=str(thumb), error='')
                elif status.get('completed'):
                    job.update(state='failed', error='任务已结束，但没有图片输出。')
            else:
                queue = client.request('GET', '/queue')
                # Also recover a response lost against servers that minted the id.
                matches = [(state, item) for key, state in [('queue_running', 'running'), ('queue_pending', 'queued')]
                           for item in queue.get(key, [])
                           if item[1] == job['prompt_id'] or (len(item) > 3 and item[3].get('eidory_job_id') == job['id'])]
                if matches:
                    job.update(state=matches[0][0], prompt_id=matches[0][1][1], error='')
                    job.pop('not_found_since', None)
                else:
                    job.setdefault('not_found_since', time.time())
                    if time.time() - job['not_found_since'] > 45:
                        job.update(state='missing', error='ComfyUI 未保留此任务。未重复提交；请检查服务后再生成。')
            if job['state'] == 'offline':
                job.update(state='unknown', error='服务已连接，正在核对原任务。')
            job.pop('next_check_at', None)
            job.pop('connection_failures', None)
        except requests.RequestException:
            self._mark_offline(job)
        except (ValueError, OSError) as exc:
            # Preserve downloadable results for another attempt, including disk full.
            job['error'] = str(exc)
        self.save(job)
        return job

    def cancel(self, job_id):
        job = self.get(job_id)
        if job['state'] not in ACTIVE_STATES:
            return job
        if job['state'] == 'offline':
            job.update(state='paused', error='已停止本地等待；服务端是否停止尚未确认，可继续核对原任务。')
            self.save(job)
            return job
        # This local server's atomic per-job endpoint avoids a global interrupt
        # racing with a browser-owned job. No global-interrupt fallback.
        client = ComfyClient(job['url'])
        try:
            result = client.request('POST', '/api/jobs/' + job['prompt_id'] + '/cancel', json={})
        except requests.RequestException:
            self._mark_offline(job)
            self.save(job)
            return job
        if result.get('cancelled'):
            job.update(state='cancelled', error='')
            self.save(job)
        else:
            job = self.reconcile(job_id)
            if job['state'] in ACTIVE_STATES:
                job['error'] = '服务尚未确认取消，继续核对任务。'
                self.save(job)
        return job

    @staticmethod
    def _mark_offline(job):
        failures = job.get('connection_failures', 0) + 1
        job.update(state='offline', connection_failures=failures,
                   next_check_at=time.time() + min(60, 2 ** min(failures, 6)),
                   error='ComfyUI 离线，保留原任务并自动重连；可停止等待。')

    def resume(self, job_id):
        job = self.get(job_id)
        if job['state'] != 'paused':
            return job
        if self.active():
            raise ValueError('请先等待当前任务结束。')
        with self.store.connect(readonly=True) as conn:
            for table in ('generated_reference_jobs', 'image_upscale_jobs'):
                if conn.execute(f"SELECT 1 FROM {table} WHERE json_extract(payload,'$.state') IN ('submitting','unknown','queued','running','downloading','offline') LIMIT 1").fetchone():
                    raise ValueError('请先等待当前生图或放大任务结束。')
        job.update(state='unknown', error='正在核对原任务，不重复提交。')
        job.pop('next_check_at', None)
        self.save(job)
        return self.reconcile(job_id)

    def discard(self, job_id, *, undo=False):
        job = self.get(job_id)
        expected = 'discarded' if undo else 'ready'
        if job['state'] != expected:
            raise ValueError('只有未采用的候选可以丢弃；已采用图片不受影响。')
        base = self.root / 'candidates' / str(job['project_id'])
        trash = self.root / 'discarded' / str(job['project_id'])
        source, dest = (trash, base) if undo else (base, trash)
        dest.mkdir(parents=True, exist_ok=True)
        for suffix in ('.png', '.webp'):
            src = source / (job['id'] + suffix)
            dst = dest / src.name
            if src.exists():
                os.replace(src, dst)
        job.update(state='ready' if undo else 'discarded', file_path=str(dest / (job['id'] + '.png')),
                   thumbnail_path=str(dest / (job['id'] + '.webp')))
        self.save(job)
        return job

    def adopt(self, job_id, collection_id, directory):
        job = self.get(job_id)
        if job['state'] == 'adopted':
            return job
        if job['state'] != 'ready':
            raise ValueError('只能采用已经生成的候选。')
        node = self.store.get_creative_node(job['node_id'])
        if not node or node.project_id != job['project_id']:
            raise ValueError('原节点已被删除，候选仍保留。请恢复原项目方案后再采用。')
        source = Path(job['file_path'])
        directory = Path(directory).expanduser().resolve()
        if directory == self.root or self.root in directory.parents:
            raise ValueError('永久保存位置不能是生成候选目录。')
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / ('AI-' + job['id'] + '.png')
        digest = lambda p: hashlib.sha256(p.read_bytes()).digest()
        if target.exists():
            if digest(source) != digest(target):
                raise ValueError('保存位置存在不同内容的同名图片，未覆盖。')
        else:
            temp = target.with_suffix('.part')
            try:
                shutil.copyfile(source, temp)
                os.replace(temp, target)
            finally:
                temp.unlink(missing_ok=True)
        # A sidecar preserves exact reproduction settings without filling the UI.
        sidecar = target.with_suffix('.json')
        sidecar.write_text(json.dumps({k: job[k] for k in ('prompt', 'seed', 'workflow', 'created_at', 'project_title', 'node_title')}, ensure_ascii=False, indent=2), encoding='utf-8')
        folder_id = self.store.add_folder(str(directory))
        stat = target.stat()
        now = utc_now_iso()
        with Image.open(target) as image:
            width, height = image.size
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            current = json.loads(conn.execute('SELECT payload FROM generated_reference_jobs WHERE id=?', (job_id,)).fetchone()[0])
            if current['state'] == 'adopted':
                return current
            if current['state'] != 'ready':
                raise ValueError('候选状态已改变，请重新选择。')
            if not conn.execute('SELECT id FROM creative_nodes WHERE id=? AND project_id=?', (job['node_id'], job['project_id'])).fetchone():
                raise ValueError('原节点已被删除，未修改图库关联。')
            image_id, _ = self.store._upsert_image_on_connection(conn, now=now, folder_id=folder_id,
                file_path=str(target), file_size=stat.st_size, width=width, height=height,
                created_time_ns=getattr(stat, 'st_birthtime_ns', None), modified_time_ns=stat.st_mtime_ns, duration_ms=None)
            conn.execute('INSERT OR IGNORE INTO image_collections(image_id,collection_id,created_at) VALUES(?,?,?)', (image_id, collection_id, now))
            conn.execute('''INSERT OR IGNORE INTO creative_node_images(node_id,image_id,sort_order,created_at,intent_label,intent_query)
                VALUES(?,?,(SELECT COALESCE(MAX(sort_order),-1)+1 FROM creative_node_images WHERE node_id=?),?,?,?)''',
                (job['node_id'], image_id, job['node_id'], now, job['node_title'], job['node_title']))
            conn.execute('UPDATE creative_projects SET updated_at=? WHERE id=?', (now, job['project_id']))
            conn.execute('UPDATE images SET note=? WHERE id=?', (f'AI 生成参考 · ComfyUI\n种子：{job["seed"]}\n提示词：{job["prompt"]}\n工作流记录：{sidecar}', image_id))
            job.update(state='adopted', image_id=image_id, file_path=str(target), collection_id=collection_id, error='')
            self.save(job, conn)
        try:
            thumbnail = Thumbnailer(self.paths.thumbnail_dir).generate(image_id, str(target))
            self.store.update_thumbnail(image_id, str(thumbnail), 'ready')
            job['thumbnail_path'] = str(thumbnail)
            self.save(job)
        except Exception:
            pass  # The library's ordinary pending-thumbnail repair can retry.
        source.unlink(missing_ok=True)
        return job
