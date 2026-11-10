"""Embedded preview pages for local Qwen generation and masked image editing."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from weakref import ref

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal, QSignalBlocker
from PySide6.QtGui import QColor, QImage, QPainter, QShortcut, QKeySequence
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSpinBox,
    QSplitter, QTabWidget, QVBoxLayout, QWidget)

from eidory.core.comfy_reference import ACTIVE_STATES, STATE_LABELS, connection_message
from eidory.core.edit_tone import tone_options, TonePreview
from eidory.core.image_loader import open_local_image, srgb_pixels
from eidory.core.image_upscale import ImageReplacement, digest
from eidory.core.preview_ai import PreviewAIJobs, raster_mask
from eidory.ui.image_upscale import WipeComparison
from eidory.ui.wrapping_layout import WrappingLayout
from eidory.ui.tone_preview import LatestTonePreview

LABELS = dict(STATE_LABELS, applied='已替换原图，可恢复', restored='已恢复原图')


def read_image(path):
    with open_local_image(path) as original:
        image = srgb_pixels(original).convert('RGBA')
        data = image.tobytes()
        return QImage(data, image.width, image.height, image.width*4, QImage.Format.Format_RGBA8888).copy()


def same_source(job, current):
    """Reuse normalized pixels only after proving the current source is unchanged."""
    if not job or current != job.get('source_path') or not job.get('original_path'):
        return False
    try:
        current_stat, backup_stat = Path(current).stat(), Path(job['original_path']).stat()
        return ((current_stat.st_size, current_stat.st_mtime_ns) == (backup_stat.st_size, backup_stat.st_mtime_ns)
                and digest(current) == job.get('source_digest'))
    except OSError:
        return False


class MaskCanvas(WipeComparison):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.strokes, self.redo_strokes = [], []
        self.diameter, self.erase, self.show_mask = .08, False, True
        self.stroke = None
        self.space = False
        self.overlay = QImage()
        self.hover = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName('遮罩画布：红色为修改区域；按住空格拖动，滚轮缩放')

    def image_rect(self):
        w, h = self.before.width()*self.scale, self.before.height()*self.scale
        return QRectF((self.width()-w)/2+self.pan.x(), (self.height()-h)/2+self.pan.y(), w, h)

    def point(self, position, *, clamp=False):
        rect = self.image_rect()
        if self.before.isNull() or rect.width() <= 0 or (not clamp and not rect.contains(position)):
            return None
        return [max(0., min(1., (position.x()-rect.x())/rect.width())),
                max(0., min(1., (position.y()-rect.y())/rect.height()))]

    def rebuild(self):
        if not self.before.isNull():
            size = self.before.size().scaled(1400, 1400, Qt.AspectRatioMode.KeepAspectRatio)
            mask = raster_mask(self.strokes, (size.width(), size.height()))
            data = mask.tobytes()
            alpha = QImage(data, mask.width, mask.height, mask.width, QImage.Format.Format_Grayscale8).copy()
            self.overlay = QImage(mask.width, mask.height, QImage.Format.Format_ARGB32)
            self.overlay.fill(QColor(255, 55, 55, 105))
            self.overlay.setAlphaChannel(alpha)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.show_mask and not self.overlay.isNull():
            painter = QPainter(self)
            painter.setOpacity(.4)
            painter.drawImage(self.image_rect(), self.overlay)
            painter.end()
        if self.hover is not None and not self.space and not self.before.isNull():
            painter = QPainter(self)
            painter.setClipRect(self.image_rect())
            painter.setPen(QColor('white'))
            radius = self.diameter*min(self.before.width(), self.before.height())*self.scale/2
            painter.drawEllipse(self.hover, radius, radius)

    def mousePressEvent(self, event):
        point = self.point(event.position())
        if not self.space and event.button() == Qt.MouseButton.LeftButton and point:
            self.setFocus()
            if len(self.strokes) >= 4000 or sum(len(s['points']) for s in self.strokes) >= 99_000:
                return
            self.stroke = dict(points=[point], diameter=self.diameter, erase=self.erase)
            self.strokes.append(self.stroke)
            self.redo_strokes.clear()
            self.rebuild()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.hover = event.position()
        if self.stroke:
            point = self.point(event.position(), clamp=True)
            if point and point != self.stroke['points'][-1] and len(self.stroke['points']) < 1000:
                self.stroke['points'].append(point)
                self.rebuild()
        else:
            super().mouseMoveEvent(event)
        self.setCursor(Qt.CursorShape.OpenHandCursor if self.space else Qt.CursorShape.CrossCursor)
        self.update()

    def leaveEvent(self, event):
        self.hover = None
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.stroke:
            self.stroke = None
            self.changed.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space = True
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space = False
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self.space = False
        super().focusOutEvent(event)

    def undo(self):
        if self.strokes:
            self.redo_strokes.append(self.strokes.pop())
            self.rebuild()
            self.changed.emit()

    def redo(self):
        if self.redo_strokes:
            self.strokes.append(self.redo_strokes.pop())
            self.rebuild()
            self.changed.emit()

    def clear(self):
        self.strokes.clear()
        self.redo_strokes.clear()
        self.rebuild()
        self.changed.emit()


class PreviewAIPanel(QDialog):
    def __init__(self, controller, image, mode, parent=None, embedded=False):
        super().__init__(parent, Qt.WindowType.Widget if embedded else Qt.WindowType.Window)
        self.c, self.image, self.mode = controller, image, mode
        self.image_id = image.id if image else None
        self.job_id, self.load_key = None, None
        self.tone_dirty = False
        self.tone_cache = None
        self.tone_cache_job = None
        self.tone_ready = False
        self.tone_timer = QTimer(self)
        self.tone_timer.setSingleShot(True)
        self.tone_timer.setInterval(180)
        self.tone_timer.timeout.connect(lambda: self.c.adjust_tone(self))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(('AI 编辑 · '+image.file_name) if mode=='edit' and image else 'AI 生图')
        self.resize(1100, 780)
        root = QVBoxLayout(self)
        header = QHBoxLayout()
        self.back = QPushButton('← 返回预览' if embedded else '关闭')
        self.back.setToolTip('返回预览；任务在后台继续，候选结果和草稿保留（Esc）')
        self.back.clicked.connect(self.close)
        header.addWidget(self.back)
        header.addWidget(QLabel('千问官方 Turbo · 8 步 · 本机处理'))
        header.addStretch()
        self.history = QComboBox()
        self.history.setMinimumWidth(240)
        self.history.setMaximumWidth(420)
        self.history.setAccessibleName('生图编辑任务与历史结果')
        self.history.currentIndexChanged.connect(self.select_job)
        header.addWidget(self.history)
        root.addLayout(header)
        self.result_summary = QLabel()
        self.result_summary.setTextFormat(Qt.TextFormat.PlainText)
        self.result_summary.setWordWrap(True)
        self.result_summary.setMaximumHeight(60)
        root.addWidget(self.result_summary)
        splitter = QSplitter()
        sidebar = QWidget()
        sidebar.setMinimumWidth(260)
        sidebar.setMaximumWidth(390)
        form = QVBoxLayout(sidebar)
        form.setContentsMargins(0, 4, 12, 4)
        title = QLabel('修改要求（新任务）' if mode == 'edit' else '画面描述（新任务）')
        form.addWidget(title)
        self.prompt = QPlainTextEdit()
        self.prompt.setAccessibleName(title.text())
        self.prompt.setPlaceholderText('例如：去掉涂选区域的文字，补全原来的背景。' if mode == 'edit' else '描述主体、环境、构图、光线和画面风格。')
        self.prompt.setMinimumHeight(140)
        form.addWidget(self.prompt, 1)
        self.canvas = MaskCanvas()
        self.comparison = WipeComparison()
        self.comparison.after_label = 'AI 编辑结果'
        self.comparison.empty_text = '生成完成后，在这里审核结果。' if mode=='generate' else '编辑完成后，在这里对比结果。'
        self.tabs = QTabWidget()
        if mode == 'edit':
            self.tabs.addTab(self.canvas, '涂选区域')
            result_page = QWidget()
            result_layout = QVBoxLayout(result_page)
            result_layout.setContentsMargins(0, 0, 0, 0)
            result_layout.addWidget(self.comparison, 1)
            self.comparison.setMinimumHeight(200)
            self.tone_group = QGroupBox('结果明度 / 纯度 · 仅调整涂选区域，不重跑 AI')
            tone_form = QFormLayout(self.tone_group)
            tone_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            tone_form.setVerticalSpacing(4)
            tone_form.setContentsMargins(8, 6, 8, 6)
            self.tone_match = QCheckBox('自动匹配原图亮度')
            self.tone_match.toggled.connect(self.tone_changed)
            self.tone_sliders, self.tone_spins = {}, {}
            for key, label, low, high, value in [('strength', '匹配强度', 0, 100, 100),
                    ('lightness', '明度', -50, 50, 0), ('saturation', '纯度（饱和度）', 0, 200, 100)]:
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setMinimumWidth(160)
                spin = QSpinBox()
                slider.setRange(low, high)
                spin.setRange(low, high)
                slider.setValue(value)
                slider.setTracking(True)
                slider.sliderPressed.connect(self.tone_drag_started)
                slider.sliderReleased.connect(self.tone_drag_finished)
                spin.setValue(value)
                if key != 'lightness':
                    spin.setSuffix('%')
                spin.setMinimumWidth(78)
                slider.setAccessibleName(label)
                spin.setAccessibleName(label+'数值')
                slider.valueChanged.connect(spin.setValue)
                spin.valueChanged.connect(slider.setValue)
                spin.valueChanged.connect(self.tone_changed)
                if key == 'strength':
                    row_layout.addWidget(QLabel('强度'))
                row_layout.addWidget(slider, 1)
                row_layout.addWidget(spin)
                self.tone_sliders[key], self.tone_spins[key] = slider, spin
                tone_form.addRow(self.tone_match if key == 'strength' else label, row)
            self.tone_match.setToolTip('从未涂选背景估计明度偏差；不会把修改后的物体颜色强制匹配回原图。')
            self.tone_sliders['lightness'].setToolTip('0 不额外调整；向右变亮，向左变暗。')
            self.tone_sliders['saturation'].setToolTip('100% 保留原始纯度，0% 去色；不会改动未涂选区域。')
            tone_actions = QHBoxLayout()
            self.tone_reset = QPushButton('还原 AI 原始色调')
            self.tone_reset.clicked.connect(self.reset_tone)
            self.tone_reset.setToolTip('关闭自动匹配，明度归零、纯度恢复 100%；不丢弃 AI 编辑内容。')
            self.tone_apply = QPushButton('应用调整')
            self.tone_apply.clicked.connect(lambda: self.c.adjust_tone(self))
            tone_actions.addWidget(self.tone_reset)
            tone_actions.addWidget(self.tone_apply)
            tone_form.addRow(tone_actions)
            self.tone_info = QLabel()
            self.tone_info.setWordWrap(True)
            self.tone_info.setTextFormat(Qt.TextFormat.PlainText)
            self.tone_info.setMaximumHeight(36)
            tone_form.addRow(self.tone_info)
            result_layout.addWidget(self.tone_group)
            self.tabs.addTab(result_page, '结果对比')
            hint = QLabel('红色区域允许修改；未涂选区域保留原图。\n滚轮缩放 · 按住空格拖动图片')
            hint.setWordWrap(True)
            form.addWidget(hint)
            tools = QHBoxLayout()
            self.eraser = QCheckBox('橡皮擦')
            self.eraser.toggled.connect(lambda v: setattr(self.canvas, 'erase', v))
            tools.addWidget(QLabel('画笔'))
            tools.addWidget(self.eraser)
            self.show_mask = QCheckBox('显示遮罩')
            self.show_mask.setChecked(True)
            self.show_mask.toggled.connect(self.toggle_mask)
            tools.addWidget(self.show_mask)
            form.addLayout(tools)
            self.brush = QSlider(Qt.Orientation.Horizontal)
            self.brush.setRange(1, 50)
            self.brush.setValue(8)
            self.brush.setAccessibleName('画笔大小，占原图短边的百分比')
            self.brush.valueChanged.connect(lambda v: setattr(self.canvas, 'diameter', v/100))
            form.addWidget(self.brush)
            undo_row = QHBoxLayout()
            self.undo = QPushButton('撤销')
            self.redo = QPushButton('重做')
            clear = QPushButton('清空')
            for button, fn in [(self.undo, self.canvas.undo), (self.redo, self.canvas.redo), (clear, self.canvas.clear)]:
                button.clicked.connect(fn)
                undo_row.addWidget(button)
            form.addLayout(undo_row)
            self.feather = QSpinBox()
            self.feather.setRange(0, 64)
            self.feather.setValue(8)
            self.feather.setSuffix(' px · 向内')
            detail = QFormLayout()
            detail.addRow('边缘羽化', self.feather)
            form.addLayout(detail)
            self.canvas.changed.connect(self.mask_changed)
            self.feather.valueChanged.connect(self.queue_draft)
        else:
            self.comparison.before_label = '生成结果'
            self.tabs.addTab(self.comparison, '生成结果')
            self.width_box, self.height_box = QSpinBox(), QSpinBox()
            sizes = QFormLayout()
            for label, box in [('宽度', self.width_box), ('高度', self.height_box)]:
                box.setRange(256, 2048)
                box.setSingleStep(32)
                box.setValue(1024)
                sizes.addRow(label, box)
            form.addLayout(sizes)
        self.start = QPushButton('开始编辑' if mode == 'edit' else '开始生图')
        self.start.clicked.connect(controller.generate)
        self.cancel = QPushButton('取消任务')
        self.cancel.clicked.connect(controller.cancel)
        form.addWidget(self.start)
        form.addWidget(self.cancel)
        splitter.addWidget(sidebar)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)
        footer = WrappingLayout()
        fit, actual = QPushButton('适应窗口'), QPushButton('100%')
        fit.clicked.connect(lambda: self.active_canvas().fit())
        actual.clicked.connect(lambda: self.active_canvas().actual())
        footer.addWidget(fit)
        footer.addWidget(actual)
        self.replace = QPushButton('替换原图')
        self.replace.setVisible(mode == 'edit')
        self.replace.clicked.connect(controller.replace)
        self.save = QPushButton('另存副本到图库' if mode == 'edit' else '保存到图库')
        self.save.clicked.connect(controller.adopt)
        self.discard = QPushButton('放弃结果')
        self.discard.clicked.connect(controller.discard)
        self.restore = QPushButton('恢复替换前原图')
        self.restore.setVisible(mode == 'edit')
        self.restore.clicked.connect(controller.restore)
        from eidory.ui.review_actions import describe_review_actions
        describe_review_actions(self.save, self.replace, self.discard, copy_is_generated=mode == 'generate')
        for button in (self.save, self.replace, self.discard, self.restore):
            footer.addWidget(button)
        root.addLayout(footer)
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(6)
        self.progress.setRange(0, 0)
        progress_policy = self.progress.sizePolicy()
        progress_policy.setRetainSizeWhenHidden(True)
        self.progress.setSizePolicy(progress_policy)
        root.addWidget(self.progress)
        self.status = QLabel('涂选修改区域后输入要求，再开始编辑。' if mode=='edit' else '输入文字生成新图片，审核后保存到图库。')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setMaximumHeight(80)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.status)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
        self.draft_timer = QTimer(self)
        self.draft_timer.setSingleShot(True)
        self.draft_timer.setInterval(500)
        self.draft_timer.timeout.connect(self.save_draft)
        self.load_draft()
        self.prompt.textChanged.connect(self.queue_draft)
        shortcut = QShortcut(QKeySequence('Ctrl+Return'), self)
        shortcut.activated.connect(controller.generate)
        if mode == 'edit':
            for key, fn in [('Undo', self.canvas.undo), ('Redo', self.canvas.redo)]:
                shortcut = QShortcut(QKeySequence(getattr(QKeySequence.StandardKey, key)), self.canvas)
                shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
                shortcut.activated.connect(fn)
        self.refresh()

    def active_canvas(self):
        return self.canvas if self.mode == 'edit' and self.tabs.currentIndex()==0 else self.comparison

    def toggle_mask(self, value):
        self.canvas.show_mask = value
        self.canvas.update()

    def mask_changed(self):
        self.undo.setEnabled(bool(self.canvas.strokes))
        self.redo.setEnabled(bool(self.canvas.redo_strokes))
        self.queue_draft()

    def draft_path(self):
        return self.c.repo.root/'drafts'/f'{self.mode}-{self.image_id if self.mode=="edit" else "text"}.json'

    def queue_draft(self, *_):
        self.draft_timer.start()

    def save_draft(self):
        draft = {'prompt': self.prompt.toPlainText()}
        if self.mode == 'edit':
            draft.update(strokes=self.canvas.strokes, feather=self.feather.value(),
                         signature=[self.image.file_size, self.image.modified_time_ns] if self.image else None)
        try:
            path = self.draft_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            stage = path.with_suffix('.part')
            stage.write_text(json.dumps(draft, ensure_ascii=False), encoding='utf-8')
            stage.replace(path)
        except OSError as exc:
            self.status.setText('草稿保存失败：'+str(exc))

    def load_draft(self):
        try:
            path = self.draft_path()
            if not path.exists() or path.stat().st_size > 8*1024*1024:
                return
            draft = json.loads(path.read_text(encoding='utf-8'))
            self.prompt.setPlainText(draft.get('prompt', ''))
            if self.mode=='edit' and self.image and draft.get('signature') == [self.image.file_size, self.image.modified_time_ns]:
                raster_mask(draft['strokes'], (10, 10))  # Validate untrusted persisted coordinates.
                self.canvas.strokes = draft['strokes']
                self.feather.setValue(draft.get('feather', 8))
        except (ValueError, KeyError, TypeError, OSError):
            pass

    def select_job(self, *_):
        if self.tone_dirty:
            self.c.adjust_tone(self, force=True)
        self.c.live_preview.invalidate()
        self.tone_cache = None
        self.tone_cache_job = None
        self.job_id = self.history.currentData()
        self.load_key = None
        job = self.job()
        if job:
            self.status.setText(job.get('error') or LABELS.get(job['state'], job['state']))
        self.c.load_images()
        self.refresh_controls()

    def job(self):
        return self.c.repo.get(self.job_id) if self.job_id else None

    def refresh(self):
        jobs = self.c.repo.all(self.image_id if self.mode=='edit' else None, self.mode)
        with QSignalBlocker(self.history):
            self.history.clear()
            self.history.addItem('当前草稿 / 新任务', None)
            for job in jobs:
                timestamp = datetime.fromisoformat(job['created_at']).astimezone().strftime('%m-%d %H:%M:%S')
                self.history.addItem(f"{timestamp} · {LABELS.get(job['state'], job['state'])} · {job['prompt'][:20]}", job['id'])
            self.history.setCurrentIndex(max(0, self.history.findData(self.job_id)))
        self.refresh_controls()

    def refresh_controls(self):
        active, job = self.c.repo.active(), self.job()
        busy = self.c.working or self.c.w._database_maintenance_active
        ready = bool(job and job['state']=='ready' and (job['mode']=='generate' or job.get('composited')))
        unsettled = bool(self.tone_dirty or (job and job.get('tone_pending')))
        self.result_summary.setText(('当前结果：'+job['prompt'][:180]) if job else '')
        self.result_summary.setToolTip(job['prompt'] if job else '')
        self.result_summary.setVisible(bool(job))
        shared = self.c.w.reference_generation.repo.active() or self.c.w.image_upscaler.repo.active()
        self.start.setEnabled(not busy and not shared and not unsettled)
        self.start.setText('继续核对原任务' if job and job['state']=='paused' else ('开始编辑' if self.mode=='edit' else '开始生图'))
        self.cancel.setEnabled(bool(active) and not busy)
        self.cancel.setText('停止等待' if active and active['state']=='offline' else '取消任务')
        self.history.setEnabled(not busy and not unsettled)
        self.save.setEnabled(ready and not busy and not unsettled)
        self.replace.setEnabled(ready and not busy and not unsettled and self.mode=='edit')
        self.discard.setEnabled(bool(job and job['state'] in {'ready', 'discarded'}) and not busy and not unsettled)
        self.discard.setText('撤销放弃' if job and job['state']=='discarded' else '放弃结果')
        self.restore.setEnabled(bool(self.image_id and self.c.replacements.latest(self.image_id)) and not busy and not unsettled)
        self.progress.setVisible(bool(active or busy))
        if active and self.job_id == active['id']:
            self.status.setText(active.get('error') or LABELS.get(active['state'], active['state']))
        if self.mode=='edit':
            self.undo.setEnabled(bool(self.canvas.strokes))
            self.redo.setEnabled(bool(self.canvas.redo_strokes))
            self.sync_tone(job)
            self.tone_sliders['strength'].setEnabled(self.tone_match.isChecked())
            self.tone_spins['strength'].setEnabled(self.tone_match.isChecked())
            raw_available = bool(job and job.get('file_path') and Path(job['file_path']).with_suffix('.raw.png').is_file())
            self.tone_ready = ready and raw_available
            if not ready and self.tone_cache is not None:
                self.c.live_preview.invalidate()
                self.tone_cache = None
                self.tone_cache_job = None
            self.tone_group.setVisible(bool(job))
            tone_running_here = bool(self.c.tone_rendering and job and self.c.tone_rendering[0] == job['id'])
            self.tone_group.setEnabled(self.tone_ready and self.tone_cache is not None and (not self.c.working or tone_running_here)
                                       and not self.c.w._database_maintenance_active)
            self.tone_group.setToolTip(('仅待审核结果可调整；已放弃的结果请先撤销放弃。' if not ready else
                '原始 AI 结果缓存缺失，无法无损重调。' if not raw_available else
                '正在准备实时预览…' if self.tone_cache is None else ''))
            self.tone_apply.setEnabled(unsettled)

    def tone_values(self):
        return dict(match=self.tone_match.isChecked(), **{key: spin.value() for key, spin in self.tone_spins.items()})

    def sync_tone(self, job):
        if not job or self.tone_dirty:
            return
        options = tone_options((job.get('tone_pending') or {}).get('options', job.get('tone')))
        widgets = [self.tone_match, *self.tone_spins.values(), *self.tone_sliders.values()]
        blockers = [QSignalBlocker(widget) for widget in widgets]
        self.tone_match.setChecked(options['match'])
        for key in self.tone_spins:
            self.tone_spins[key].setValue(options[key])
            self.tone_sliders[key].setValue(options[key])
        del blockers
        report = job.get('tone_report') or {}
        message = report.get('reason') or (f"自动匹配明度补偿 {report.get('offset', 0)*options['strength']/100:+.1f}" if options['match'] else '保留 AI 原始明度，可手动微调。')
        self.tone_info.setText('正在应用最新调整…' if job.get('tone_pending') else message)

    def tone_changed(self, *_):
        if self.mode != 'edit' or not self.job_id or not self.tone_ready:
            return
        self.tone_dirty = True
        # This hot path performs no SQL, file reads or full-resolution work.
        for button in (self.save, self.replace, self.discard, self.restore, self.start, self.history):
            button.setEnabled(False)
        self.tone_apply.setEnabled(True)
        self.tone_sliders['strength'].setEnabled(self.tone_match.isChecked())
        self.tone_spins['strength'].setEnabled(self.tone_match.isChecked())
        self.c.preview_tone(self)
        if self.tone_sliding():
            self.tone_timer.stop()
        else:
            self.tone_timer.start()

    def tone_sliding(self):
        return self.mode == 'edit' and any(s.isSliderDown() for s in self.tone_sliders.values())

    def tone_drag_started(self):
        self.tone_timer.stop()

    def tone_drag_finished(self):
        if self.tone_dirty:
            self.tone_timer.start(0)

    def reset_tone(self):
        widgets = [self.tone_match, *self.tone_spins.values(), *self.tone_sliders.values()]
        blockers = [QSignalBlocker(widget) for widget in widgets]
        self.tone_match.setChecked(False)
        for key, value in [('strength', 100), ('lightness', 0), ('saturation', 100)]:
            self.tone_spins[key].setValue(value)
            self.tone_sliders[key].setValue(value)
        del blockers
        self.tone_changed()

    def closeEvent(self, event):
        self.tone_timer.stop()
        if self.tone_dirty and not self.c.adjust_tone(self, force=True):
            event.ignore()
            return
        self.draft_timer.stop()
        self.save_draft()
        super().closeEvent(event)


class PreviewAI(QObject):
    completed = Signal(object, object)

    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self.repo = PreviewAIJobs(window.store, window.paths)
        self.replacements = ImageReplacement(window.store, window.paths)
        self.panel, self.working = None, False
        self.tone_rendering = None
        self.tone_failed = {}
        self.live_preview = LatestTonePreview(lambda worker: not self.w._closing and
            self.w._start_background_task(worker, name='tone-preview'), self)
        self.live_preview.ready.connect(self.preview_ready)
        self.live_preview.failed.connect(self.preview_failed)
        self.completed.connect(self.finished)
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.poll)
        self.timer.start()
        self.entry = QPushButton('AI 生图 / 编辑')
        self.entry.clicked.connect(self.open_latest)
        window.statusBar().addPermanentWidget(self.entry)
        self.refresh()

    @property
    def w(self):
        return self._window()

    def open(self, image, mode, *, preview=None, job_id=None, resume=True):
        if self.w._database_maintenance_active:
            return
        if preview and not preview.annotation_controls.flush(interactive=True):
            return
        if self.panel:
            if not self.panel.close():
                return
        self.panel = panel = PreviewAIPanel(self, image, mode, preview or self.w, embedded=bool(preview))
        panel.finished.connect(lambda _: self.closed(panel))
        if job_id is None and resume:
            jobs = self.repo.all(panel.image_id if mode=='edit' else None, mode)
            latest = next((j for j in jobs if j['state'] in ACTIVE_STATES or j['state']=='ready'), None)
            job_id = latest['id'] if latest else None
        panel.job_id = job_id
        panel.refresh()
        if job_id:
            job = self.repo.get(job_id)
            panel.status.setText(job.get('error') or LABELS.get(job['state'], job['state']))
        if preview:
            preview.show_upscale_panel(panel)
        panel.show()
        panel.raise_()
        self.load_images()

    def closed(self, panel):
        if self.panel is panel:
            self.live_preview.invalidate()
            panel.comparison.after_preview = QImage()
            panel.comparison.before_preview = QImage()
            panel.tone_cache = None
            panel.tone_cache_job = None
            self.panel = None
            QTimer.singleShot(0, self.drain_tone)

    def open_latest(self):
        jobs = self.repo.all()
        job = next((j for j in jobs if j['state'] in ACTIVE_STATES or j['state']=='ready'), jobs[0] if jobs else None)
        image = self.w.store.get_image(job['image_id']) if job and job.get('image_id') else self.w.selected_image
        self.open(image, job['mode'] if job else 'generate', job_id=job['id'] if job else None)

    def run(self, operation, callback):
        if self.working or self.w._closing or self.w._database_maintenance_active:
            return False
        self.working = True
        self.w.reference_generation.hold_indexing(True)
        self.refresh()
        def worker():
            try:
                result = operation()
            except Exception as exc:
                logging.getLogger('eidory.runtime').exception('Preview AI operation failed')
                result = exc
            self.completed.emit(callback, result)
        if not self.w._start_background_task(worker, name='preview-ai'):
            self.working = False
            self.w.reference_generation.hold_indexing(False)
            return False
        return True

    def finished(self, callback, result):
        rendering, self.tone_rendering = self.tone_rendering, None
        self.working = False
        if self.w._closing:
            return
        if isinstance(result, Exception):
            if rendering:
                self.tone_failed[rendering[0]] = rendering[1]
            if self.panel:
                self.panel.status.setText(connection_message(result))
            self.w.statusBar().showMessage('AI 生图 / 编辑：'+connection_message(result))
        else:
            try:
                callback(result)
            except Exception as exc:
                logging.getLogger('eidory.runtime').exception('Preview AI result handling failed')
                if self.panel:
                    self.panel.status.setText(connection_message(exc))
        self.refresh()
        self.w.reference_generation.hold_indexing(bool(self.w.reference_generation.active()))
        QTimer.singleShot(0, self.drain_tone)

    def refresh(self):
        jobs = self.repo.all()
        if hasattr(self.w, 'task_center'):
            self.w.task_center.update_counts('ai', jobs)
        count = sum(j['state']=='ready' for j in jobs)
        self.entry.setText(f'AI 生图 / 编辑 · {count} 张待审核' if count else 'AI 生图 / 编辑')
        if self.panel:
            self.panel.refresh()

    def load_images(self):
        panel = self.panel
        if not panel or self.working:
            return
        job = panel.job()
        key = (panel.image_id, panel.job_id, job['state'] if job else None, job.get('render_id') if job else None)
        if panel.load_key == key:
            return
        panel.load_key = key
        current = panel.image.file_path if panel.image and panel.mode=='edit' else None
        source = (job.get('input_path') if job and panel.mode=='edit' else None) or current
        result = job.get('file_path') if job and job['state'] in {'ready','applied','adopted','restored'} else None
        cached = panel.tone_cache if panel.tone_cache_job == panel.job_id else None
        prepare = bool(job and job['mode']=='edit' and job['state']=='ready' and job.get('composited')
                       and Path(job['file_path']).with_suffix('.raw.png').is_file())
        def load():
            before = read_image(source) if source else QImage()
            cache = (cached or TonePreview.from_job(job)) if prepare else None
            preview_source = QImage()
            if cache is not None:
                pixels = cache.original
                data = pixels.tobytes()
                preview_source = QImage(data, pixels.width, pixels.height, pixels.width*4, QImage.Format.Format_RGBA8888).copy()
            return (read_image(current) if current and current != source and Path(current).is_file() and not same_source(job, current) else before,
                    before, read_image(result) if result else QImage(),
                    cache, preview_source)
        def done(images):
            if self.panel is not panel or panel.load_key != key:
                return
            current_image, before, after, cache, preview_source = images
            if panel.mode=='edit':
                panel.tone_cache = cache
                panel.tone_cache_job = panel.job_id if cache is not None else None
                panel.canvas.set_image('before', current_image)
                panel.canvas.rebuild()
                panel.comparison.set_image('before', before)
                panel.comparison.set_image('after', after)
                panel.comparison.before_preview = preview_source
                if not after.isNull():
                    panel.tabs.setCurrentIndex(1)
                if panel.tone_dirty:
                    self.preview_tone(panel)
            else:
                panel.comparison.set_image('before', after)
                panel.comparison.set_image('after', QImage())
        self.run(load, done)

    def generate(self):
        panel = self.panel
        if not panel or self.working:
            return
        selected = panel.job()
        if selected and selected['state']=='paused':
            self.run(lambda: self.repo.resume(selected['id']), self.updated)
            return
        prompt, mode, image_id = panel.prompt.toPlainText(), panel.mode, panel.image_id
        drawing = json.loads(json.dumps(panel.canvas.strokes))
        feather = panel.feather.value() if mode=='edit' else 0
        width, height = (panel.width_box.value(), panel.height_box.value()) if mode=='generate' else (1024, 1024)
        url = self.w.store.get_setting('comfy.url', 'http://127.0.0.1:8188')
        panel.save_draft()
        def submit():
            job = self.repo.create(image_id, mode, prompt, url, drawing=drawing, feather=feather, width=width, height=height)
            return self.repo.submit(job['id'])
        self.run(submit, self.updated)

    def updated(self, job):
        panel = self.panel
        if panel and panel.mode == job['mode'] and (panel.mode=='generate' or panel.image_id==job.get('image_id')):
            panel.job_id = job['id']
            panel.status.setText(job.get('error') or LABELS.get(job['state'], job['state']))
            panel.refresh()
            self.load_images()
        if job['state']=='ready':
            self.w.statusBar().showMessage('AI '+('编辑' if job['mode']=='edit' else '生图')+'已完成，结果待审核。', 15000)

    def poll(self):
        if self.working or self.w._closing or self.w._database_maintenance_active:
            return
        if self.drain_tone():
            return
        job = self.repo.active()
        # Finish a local composite after a crash during download/finalization.
        if not job:
            job = next((j for j in self.repo.all() if j['state']=='ready' and j['mode']=='edit' and not j.get('composited')), None)
        if job:
            self.run(lambda: self.repo.reconcile(job['id']), self.updated)
        else:
            self.load_images()

    def preview_tone(self, panel):
        if panel.tone_cache is not None and self.panel is panel:
            self.live_preview.request((id(panel), panel.job_id), panel.tone_cache, panel.tone_values())

    def preview_ready(self, request, image):
        owner, options = request
        panel = self.panel
        if (panel is not None and owner == (id(panel), panel.job_id) and panel.tone_cache is not None
                and (panel.tone_dirty or panel.tone_sliding() or options == panel.tone_values())):
            panel.comparison.set_tone_preview(image)
            panel.tone_info.setText('实时预览；松开滑块后保存完整分辨率结果。')

    def preview_failed(self, request, error):
        panel = self.panel
        if panel is not None and request[0] == (id(panel), panel.job_id):
            panel.status.setText('实时预览失败：'+str(error))

    def adjust_tone(self, panel, *, force=False):
        if panel.mode != 'edit' or not panel.job_id:
            return True
        if panel.tone_sliding() and not force:
            panel.tone_timer.stop()
            return True
        panel.tone_timer.stop()
        try:
            self.repo.queue_tone(panel.job_id, panel.tone_values())
        except (OSError, ValueError) as exc:
            panel.status.setText(str(exc))
            return False
        panel.tone_dirty = False
        panel.refresh_controls()
        self.drain_tone()
        return True

    def drain_tone(self):
        if self.working or self.w._closing or self.w._database_maintenance_active:
            return False
        job = next((j for j in self.repo.all() if j['state']=='ready' and j.get('tone_pending')
                    and self.tone_failed.get(j['id']) != j['tone_pending']['id']), None)
        if not job:
            return False
        panel = self.panel
        if panel is not None and panel.job_id == job['id'] and panel.tone_sliding():
            return False
        analysis = panel.tone_cache.analysis if panel is not None and panel.job_id == job['id'] and panel.tone_cache is not None else None
        self.tone_rendering = (job['id'], job['tone_pending']['id'])
        def render():
            result = self.repo.render_tone(job['id'], analysis)
            return result, read_image(result['file_path'])
        def done(result):
            rendered, pixels = result
            panel = self.panel
            if panel and panel.job_id == rendered['id']:
                panel.load_key = (panel.image_id, panel.job_id, rendered['state'], rendered.get('render_id'))
                # An older disk save must never overwrite a newer live frame.
                if (not panel.tone_dirty and not panel.tone_sliding() and not rendered.get('tone_pending')
                        and panel.tone_values() == tone_options(rendered.get('tone'))):
                    self.live_preview.invalidate()
                    panel.comparison.set_image('after', pixels)
                    panel.status.setText('完整分辨率调整已保存，可替换或另存。')
        started = self.run(render, done)
        if not started:
            self.tone_rendering = None
        return started

    def cancel(self):
        job = self.repo.active()
        if job:
            self.run(lambda: self.repo.cancel(job['id']), self.updated)

    def discard(self):
        panel = self.panel
        job = panel.job() if panel else None
        if job:
            self.run(lambda: self.repo.discard(job['id'], undo=job['state']=='discarded'), self.updated)

    def replace(self):
        job = self.panel.job() if self.panel else None
        if job:
            self.run(lambda: self.repo.replace(job['id']), self.media_changed)

    def restore(self):
        image_id = self.panel.image_id if self.panel else None
        if image_id:
            self.run(lambda: self.replacements.restore(image_id), self.media_changed)

    def media_changed(self, image):
        self.w.vector_index.invalidate()
        self.w.image_upscaler.media_changed(image)
        self.w._refresh_embedding_stats()
        if self.panel:
            if self.panel.image_id == image.id:
                self.panel.image = image
                self.panel.canvas.clear()
                self.panel.load_key = None
            self.panel.status.setText('图片已更新。手动标签和项目关联保留；恢复原图可还原原文件与索引。')
        self.refresh()

    def adopt(self):
        panel = self.panel
        job = panel.job() if panel else None
        if not job or self.working:
            return
        default = str(Path.home()/'Pictures'/'Eidory Imports'/'AI 生成')
        directory = QFileDialog.getExistingDirectory(panel, '选择永久保存文件夹（保存后自动加入图库）', self.w.store.get_setting('preview_ai.directory', default))
        if not directory:
            return
        self.w.store.set_setting('preview_ai.directory', directory)
        def done(image):
            self.w.image_upscaler.refresh_exported_library()
            if self.panel:
                self.panel.status.setText('已保存并加入图库：'+image.file_path)
            self.w.statusBar().showMessage('AI 图片已保存到图库：'+image.file_name, 12000)
        self.run(lambda: self.repo.adopt(job['id'], directory), done)

    def close(self):
        self.timer.stop()
        self.live_preview.close()
        if self.panel:
            self.panel.close()
