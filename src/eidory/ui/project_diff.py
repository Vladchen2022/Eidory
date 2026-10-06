"""A compact, read-only comparison; the saved project state remains authoritative."""
from html import escape
import json
from pathlib import Path
from PySide6.QtCore import QRectF, Qt, QUrl
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QScrollArea, QTextEdit, QVBoxLayout, QWidget
from eidory.ui.inline_idea import change_html
from eidory.core.creative_entries import decode_entries


def stable(row):
    return {key: value for key, value in row.items() if key not in {'created_at', 'updated_at'}}


class LayoutMiniature(QWidget):
    def __init__(self, payload, images, parent=None):
        super().__init__(parent)
        self.payload = payload
        self.images = images
        self.setMinimumHeight(180)
        self.setAccessibleName('看板布局差异缩略图')

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#e9e5dc'))
        items = [(key, row) for key, row in self.payload.get('items', {}).items() if row.get('visible', True)]
        rects = [QRectF(row.get('x', 0), row.get('y', 0), max(1, row.get('width', 1)), max(1, row.get('height', 1))) for _, row in items]
        if not rects:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, '空看板')
            return
        bounds = QRectF(rects[0])
        for rect in rects[1:]:
            bounds = bounds.united(rect)
        scale = min((self.width()-16)/max(1, bounds.width()), (self.height()-16)/max(1, bounds.height()))
        painter.translate(8, 8)
        painter.scale(scale, scale)
        painter.translate(-bounds.x(), -bounds.y())
        for index, ((key, row), rect) in enumerate(zip(items, rects)):
            pixmap = self.images.get(int(key)) if index < 80 else None
            if pixmap and not pixmap.isNull():
                painter.drawPixmap(rect, pixmap, QRectF(pixmap.rect()))
            else:
                painter.fillRect(rect, QColor('#bcb2a1'))
            painter.setPen(QColor('#695b45'))
            painter.drawRect(rect)


class ProjectDiff(QScrollArea):
    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self.setWidgetResizable(True)
        self.setAccessibleName('方案差异预览')

    def display(self, saved, current):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        changes = 0
        def text(title, before, after):
            nonlocal changes
            if before == after:
                return
            changes += 1
            label = QLabel('<b>' + escape(title) + '</b><br>' + change_html(str(before or ''), str(after or '')))
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(label)
        layout.addWidget(QLabel('与当前方案比较 · 红色划线为所选版本，绿色为当前内容'))
        for key, caption in [('title', '项目名称'), ('brief', '创作主题')]:
            text(caption, saved['project'].get(key), current['project'].get(key))
        old = {row['id']: row for row in saved['nodes']}
        now = {row['id']: row for row in current['nodes']}
        for nid in sorted(old.keys() | now.keys()):
            a, b = old.get(nid, {}), now.get(nid, {})
            text('节点名称', a.get('title'), b.get('title'))
            text((b or a).get('title', '') + ' · 说明', a.get('note'), b.get('note'))
            if changes >= 100:
                layout.addWidget(QLabel('差异较多，其余文字可展开完整方案查看。'))
                break
        old_work = {r['node_id']: r for r in saved['tables'].get('creative_node_workflow', [])}
        now_work = {r['node_id']: r for r in current['tables'].get('creative_node_workflow', [])}
        for nid in sorted(old_work.keys() | now_work.keys()):
            a, b = old_work.get(nid, {}), now_work.get(nid, {})
            title = (now.get(nid) or old.get(nid) or {}).get('title', '')
            for key, caption in [('entries_json', '想法'), ('proposal_entries_json', '待采纳想法')]:
                ap, bp = decode_entries(a.get(key, '')) or {}, decode_entries(b.get(key, '')) or {}
                ar = {r['id']: r for r in ap.get('rows', [])}
                br = {r['id']: r for r in bp.get('rows', [])}
                for rid in sorted(ar.keys() | br.keys()):
                    x, y = ar.get(rid, {}), br.get(rid, {})
                    text(title + ' · ' + caption, x.get('name'), y.get('name'))
                    text((y or x).get('name', '') + ' · 细节', x.get('visual'), y.get('visual'))
                    if x and y and x.get('enabled') != y.get('enabled'):
                        text(title + ' · 是否找图', '用于找图' if x.get('enabled') else '不用于找图', '用于找图' if y.get('enabled') else '不用于找图')
            text(title + ' · 待采纳说明', a.get('proposal_note'), b.get('proposal_note'))
        old_refs = {(r['node_id'], r['image_id']) for r in saved['tables'].get('creative_node_images', [])}
        now_refs = {(r['node_id'], r['image_id']) for r in current['tables'].get('creative_node_images', [])}
        for caption, keys in [('当前新增参考', now_refs-old_refs), ('当前移出参考', old_refs-now_refs)]:
            if not keys:
                continue
            changes += 1
            layout.addWidget(QLabel(caption + ' · ' + str(len(keys)) + ' 处关联'))
            row = QHBoxLayout()
            for nid, iid in sorted(keys)[:12]:
                image = self.store.get_image(iid)
                tile = QWidget()
                tile_layout = QVBoxLayout(tile)
                icon = QLabel()
                if image and image.thumbnail_path:
                    pixmap = QPixmap(str(image.thumbnail_path)).scaled(95, 70, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    icon.setPixmap(pixmap)
                tile_layout.addWidget(icon)
                label = QLabel((now.get(nid) or old.get(nid) or {}).get('title', '') + '\n' + (image.file_name if image else f'图片 #{iid}'))
                label.setWordWrap(True)
                tile_layout.addWidget(label)
                row.addWidget(tile)
            layout.addLayout(row)
        for table, caption in [('creative_reference_decisions', '参考等级与用途'), ('creative_reference_notes', '用途说明'), ('creative_node_options', '搜索目标与参考需要'), ('creative_project_context', '补充信息与工作上下文')]:
            a = [stable(r) for r in saved['tables'].get(table, [])]
            b = [stable(r) for r in current['tables'].get(table, [])]
            if a != b:
                changes += 1
                layout.addWidget(QLabel(caption + '有变化 · 恢复时一并恢复'))
        for table in ('creative_board_layouts', 'creative_node_board_layouts'):
            key = 'node_id' if table == 'creative_node_board_layouts' else 'project_id'
            a = {r[key]: r for r in saved['tables'].get(table, [])}
            b = {r[key]: r for r in current['tables'].get(table, [])}
            for identity in sorted(a.keys() | b.keys()):
                ap = json.loads(a.get(identity, {}).get('payload_json') or '{}')
                bp = json.loads(b.get(identity, {}).get('payload_json') or '{}')
                if ap == bp:
                    continue
                changes += 1
                caption = '项目看板' if key == 'project_id' else (now.get(identity) or old.get(identity) or {}).get('title', '节点') + '看板'
                layout.addWidget(QLabel(caption + ' · 布局 / 图片显示有变化'))
                ids = {int(i) for p in (ap, bp) for i in p.get('items', {})}
                images = {}
                for iid in sorted(ids)[:80]:
                    image = self.store.get_image(iid)
                    if image and image.thumbnail_path:
                        images[iid] = QPixmap(str(image.thumbnail_path)).scaled(160, 160, Qt.AspectRatioMode.KeepAspectRatio)
                row = QHBoxLayout()
                for name, payload in [('所选版本', ap), ('当前方案', bp)]:
                    tile = QWidget()
                    tl = QVBoxLayout(tile)
                    tl.addWidget(QLabel(name))
                    tl.addWidget(LayoutMiniature(payload, images))
                    row.addWidget(tile)
                layout.addLayout(row)
        if not changes:
            layout.addWidget(QLabel('没有内容差异。'))
        full = QCheckBox('展开所选版本完整文字')
        complete = QTextEdit()
        complete.setReadOnly(True)
        complete.setPlainText(saved['project']['brief'] + '\n\n' + '\n\n'.join(n['title']+'\n'+n['note'] for n in saved['nodes']))
        complete.setMinimumHeight(260)
        complete.hide()
        full.toggled.connect(complete.setVisible)
        layout.addWidget(full)
        layout.addWidget(complete)
        old_page = self.takeWidget()
        if old_page:
            old_page.deleteLater()
        self.setWidget(page)
