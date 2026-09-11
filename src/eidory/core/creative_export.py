from __future__ import annotations

import html
import json
import re
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from eidory.core.creative_workspace import CreativeWorkspaceStore, decode_state
from eidory.core.media_types import is_supported_video


def export_creative_project(store, project_id: int, destination: str | Path, *, render_board=None) -> Path:
    project = store.get_creative_project(project_id)
    if project is None:
        raise ValueError("创作项目不存在")
    name = re.sub(r'[^\w\u4e00-\u9fff .-]', '_', project.title).strip(' .')[:70] or '创作参考包'
    target = Path(destination) / f"{name}-{uuid.uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=False)
    (target / 'images').mkdir()
    workspace = CreativeWorkspaceStore(store)
    nodes = store.list_creative_nodes(project_id)
    references = workspace.reference_notes(project_id)
    notes = {(r['node_id'], r['image_id']): r['note'] for r in references}
    image_ids = set()
    node_images = {}
    for node in nodes:
        node_images[node.id] = store.creative_node_image_ids(node.id)
        image_ids.update(node_images[node.id])
    paths = {}
    missing = []
    for image in store.images_by_ids(sorted(image_ids)):
        source = Path(image.file_path)
        if not source.is_file():
            missing.append(image.file_path)
            continue
        relative = f"images/{image.id}{source.suffix.lower()}"
        shutil.copy2(source, target / relative)
        paths[image.id] = (relative, is_supported_video(image.file_path))
    esc = lambda value: html.escape(str(value), quote=True)

    def media(image_id, style=''):
        if image_id not in paths:
            return '<span>源文件缺失</span>'
        path, video = paths[image_id]
        if video:
            return f'<video controls src="{esc(path)}" style="{style}"></video>'
        return f'<img loading="lazy" src="{esc(path)}" style="{style}" alt="参考图 {image_id}">'

    sections = []
    layouts = {}
    for node in nodes:
        layout = decode_state(store.get_creative_node_board_layout(node.id))
        layouts[str(node.id)] = layout
        cards = ''.join(f'<figure>{media(i)}<figcaption>{esc(notes.get((node.id, i), ""))}</figcaption></figure>' for i in node_images[node.id])
        board = ''
        items = layout.get('items', {})
        visible = [(int(i), v) for i, v in items.items() if str(i).isdigit() and isinstance(v, dict) and v.get('visible', True)]
        if visible:
            left = min(float(v.get('x', 0)) for _, v in visible)
            top = min(float(v.get('y', 0)) for _, v in visible)
            width = max(float(v.get('x', 0)) + float(v.get('width', 200)) for _, v in visible) - left
            height = max(float(v.get('y', 0)) + float(v.get('height', 160)) for _, v in visible) - top
            scale = min(1.0, 1100 / max(1, width))
            placed = []
            for image_id, item in visible:
                x, y = (float(item.get('x', 0)) - left) * scale, (float(item.get('y', 0)) - top) * scale
                iw, ih = float(item.get('width', 200)) * scale, float(item.get('height', 160)) * scale
                style = f'position:absolute;left:{x}px;top:{y}px;width:{iw}px;height:{ih}px;object-fit:contain;'
                if item.get('flipped'):
                    style += 'transform:scaleX(-1);'
                if item.get('grayscale'):
                    style += 'filter:grayscale(1);'
                placed.append(media(image_id, style))
            text_items = ''.join(f'<div class="board-note" style="left:{(float(t.get("x",0))-left)*scale}px;top:{(float(t.get("y",0))-top)*scale}px">{esc(t.get("text", ""))}</div>' for t in layout.get('notes', []) if isinstance(t, dict))
            board = f'<details><summary>查看看板布局</summary><div class="board" style="height:{max(100, height*scale+40)}px">{"".join(placed)}{text_items}</div><p class="muted">原图与布局均已保存；局部裁切显示参数见 project.json。</p></details>'
        if render_board and store.creative_node_image_ids(node.id, include_descendants=True):
            (target / 'boards').mkdir(exist_ok=True)
            render_board(node.id, target / 'boards' / f'{node.id}.png')
            board = f'<details open><summary>看板</summary><img style="max-width:100%" src="boards/{node.id}.png" alt="{esc(node.title)}看板"></details>'
        locked = ' · 已确定' if workspace.node(node.id)['locked'] else ''
        sections.append(f'<section id="node-{node.id}"><h2>{esc(node.title)}{locked}</h2><p class="text">{esc(node.note)}</p><p class="muted">找图语句：{esc(node.search_query)}</p><div class="cards">{cards}</div>{board}</section>')
    context = workspace.project(project_id)
    document = {'version': 1, 'project': asdict(project), 'extra': context['extra'], 'nodes': [asdict(n) for n in nodes],
                'node_images': node_images, 'reference_notes': references, 'board_layouts': layouts, 'missing_files': missing}
    (target / 'project.json').write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
    markup = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(project.title)} · 创作参考包</title><style>
body{{font:16px/1.7 system-ui,sans-serif;margin:auto;padding:36px;max-width:1160px;background:#f4f3ef;color:#262d34}}
h1,h2{{line-height:1.3}}section{{background:white;padding:24px;margin:24px 0;border-radius:12px}}.text{{white-space:pre-wrap}}
.muted{{color:#636d76;font-size:13px}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}figure{{margin:0;width:240px}}figure img,figure video{{width:100%;height:180px;object-fit:contain;background:#e9e9e9}}figcaption{{white-space:pre-wrap}}
.board{{position:relative;overflow:auto;background:#202830;margin-top:16px}}.board-note{{position:absolute;white-space:pre-wrap;color:#fff2be;padding:8px;background:#51442f}}
@media print{{body{{padding:0}}section{{break-inside:avoid}}}}
</style><h1>{esc(project.title)}</h1><p class="text">{esc(project.brief)}</p><p class="text">{esc(context['extra'])}</p>
<p class="muted">{len(nodes)} 个创作节点 · {len(paths)} 个参考文件 · {len(missing)} 个源文件缺失</p>
<p class="text">{esc(project.copy_text)}</p>{''.join(sections)}</html>'''
    (target / 'index.html').write_text(markup, encoding='utf-8')
    return target
