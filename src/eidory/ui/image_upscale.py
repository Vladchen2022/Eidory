"""Single preview entry, background SeedVR jobs, and a shared-coordinate wipe comparison."""
from __future__ import annotations

import time
import logging
import traceback
import hashlib

from pathlib import Path
from dataclasses import replace
from weakref import ref

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QSignalBlocker, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QLabel, QMessageBox, QProgressBar, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from eidory.core.comfy_reference import ACTIVE_STATES, STATE_LABELS, connection_message
from eidory.core.image_upscale import ImageReplacement, UpscaleJobs, output_size, processing_input_size, source_info
from eidory.core.upscale_workflow import UpscaleWorkflow
from eidory.ui.board_image_loader import BoardImageLoader
from eidory.ui.wrapping_layout import WrappingLayout

LABELS = dict(STATE_LABELS, waiting='已加入本机队列', exported='已保存为新图片，原图保留', applied='已替换原图，原有信息保留', restored='已恢复原图', discarded='已丢弃结果，原图未改变；可撤销丢弃')


class WipeComparison(QWidget):
    """Both images use the output's coordinate system, including at 100%."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.before, self.after = QImage(), QImage()
        self.after_preview = QImage()
        self.before_preview = QImage()
        self.scale = 1.0
        self.pan = QPointF()
        self.divider = .5
        self.fit_mode = True
        self.drag = None
        self.before_label = '原图'
        self.after_label = 'SeedVR2 7B'
        self.empty_text = '正在读取原图…'
        self.setMinimumSize(350, 260)
        self.setAccessibleName('原图与放大结果对比；滚轮缩放，拖动画面移动，拖动中线对比')
        self.setMouseTracking(True)

    def set_image(self, side, image):
        setattr(self, side, image)
        if side == 'after':
            self.after_preview = QImage()
        elif side == 'before':
            self.before_preview = QImage()
        if self.fit_mode:
            self.fit()
        self.update()

    def set_tone_preview(self, image):
        # The display layer does not change native dimensions, zoom or pan.
        self.after_preview = image
        self.update()

    def dimensions(self):
        return self.after.size() if not self.after.isNull() else self.before.size()

    def fit(self):
        size = self.dimensions()
        if size.width() > 0 and size.height() > 0:
            self.scale = min((self.width()-24)/size.width(), (self.height()-24)/size.height())
        self.fit_mode = True
        self.pan = QPointF()
        self.update()

    def actual(self):
        self.scale, self.pan, self.fit_mode = 1., QPointF(), False
        self.update()

    def resizeEvent(self, event):
        if self.fit_mode:
            self.fit()
        super().resizeEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#232323'))
        if self.before.isNull():
            p.setPen(QColor('white'))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)
            return
        size = self.dimensions()
        w, h = size.width()*self.scale, size.height()*self.scale
        rect = QRectF((self.width()-w)/2 + self.pan.x(), (self.height()-h)/2 + self.pan.y(), w, h)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        before = self.before_preview if not self.after_preview.isNull() and not self.before_preview.isNull() else self.before
        p.drawImage(rect, before)
        x = int(self.width()*self.divider)
        if not self.after.isNull():
            p.save()
            p.setClipRect(x, 0, self.width()-x, self.height())
            p.drawImage(rect, self.after_preview if not self.after_preview.isNull() else self.after)
            p.restore()
            p.setPen(QPen(QColor('white'), 2))
            p.drawLine(x, 0, x, self.height())
            p.setBrush(QColor('#ffffff'))
            p.drawEllipse(QPointF(x, self.height()/2), 8, 8)
        p.setPen(QColor('white'))
        p.fillRect(8, 8, 58, 26, QColor(0, 0, 0, 150))
        p.drawText(18, 26, self.before_label)
        if not self.after.isNull():
            p.fillRect(self.width()-135, 8, 127, 26, QColor(0, 0, 0, 150))
            p.drawText(self.width()-125, 26, self.after_label)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if not delta or self.before.isNull():
            return
        old = self.scale
        self.scale = max(.02, min(12., old * (1.15 ** (delta/120))))
        anchor = event.position() - QPointF(self.width()/2, self.height()/2)
        self.pan = anchor - (anchor-self.pan)*(self.scale/old)
        self.fit_mode = False
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag = 'divider' if not self.after.isNull() and abs(event.position().x()-self.width()*self.divider) < 14 else 'pan'
            self.previous = event.position()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.drag == 'divider':
            self.divider = max(0., min(1., event.position().x()/self.width()))
        elif self.drag == 'pan':
            self.pan += event.position()-self.previous
            self.previous = event.position()
            self.fit_mode = False
        else:
            on_line = not self.after.isNull() and abs(event.position().x()-self.width()*self.divider) < 14
            self.setCursor(Qt.CursorShape.SplitHCursor if on_line else Qt.CursorShape.OpenHandCursor)
        self.update()

    def mouseReleaseEvent(self, event):
        self.drag = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.fit()


class UpscalePanel(QWidget):
    finished = Signal(int)

    def __init__(self, controller, image, *, preview=None):
        super().__init__(
            preview if preview is not None else controller.w,
            Qt.WindowType.Widget if preview is not None else Qt.WindowType.Window,
        )
        self.preview = preview
        self._released = False
        self.c, self.image_id = controller, image.id
        self.info = None
        self.inspection_pending = True
        self.job = None
        self.selected_job_id = None
        self.load_key = None
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle('AI 放大 · ' + image.file_name)
        self.resize(1080, 780)
        if preview is None:
            self.setMinimumSize(760, 540)
        outer = QVBoxLayout(self)
        header = WrappingLayout()
        self.back = QPushButton('← 返回预览')
        self.back.setToolTip('返回原来的图片预览；已提交的任务继续在后台运行（Esc）')
        self.back.clicked.connect(self.close)
        self.back.setVisible(preview is not None)
        header.addWidget(self.back)
        header.addWidget(QLabel('SeedVR2 7B · 本机放大'))
        self.mode = QComboBox()
        self.mode.addItems(['2 倍', '4 倍', '指定长边'])
        self.mode.setAccessibleName('放大方式')
        header.addWidget(self.mode)
        self.edge = QSpinBox()
        self.edge.setRange(128, 4096)
        self.edge.setValue(2048)
        self.edge.setSingleStep(64)
        self.edge.setSuffix(' 像素')
        self.edge.setAccessibleName('输出长边')
        header.addWidget(self.edge)
        self.reduce_input_button = QPushButton('缩小输入至 2048')
        self.reduce_input_button.setToolTip('只缩小送入 AI 的临时副本，保留硬盘原图；可撤销')
        self.reduce_input_button.clicked.connect(self.toggle_input_reduction)
        header.addWidget(self.reduce_input_button)
        self.start = QPushButton('开始放大')
        self.start.setProperty('variant', 'primary')
        self.start.clicked.connect(controller.generate)
        header.addWidget(self.start)
        self.cancel = QPushButton('取消任务')
        self.cancel.clicked.connect(controller.cancel)
        header.addWidget(self.cancel)
        outer.addLayout(header)
        self.history = QComboBox()
        self.history.setAccessibleName('该图片的放大历史')
        self.history.currentIndexChanged.connect(self.select_history)
        outer.addWidget(self.history)
        self.dimensions = QLabel('正在读取图片…')
        outer.addWidget(self.dimensions)
        self.view = WipeComparison()
        outer.addWidget(self.view, 1)
        row = WrappingLayout()
        fit = QPushButton('适应窗口')
        fit.clicked.connect(self.view.fit)
        actual = QPushButton('结果 100%')
        actual.setToolTip('原图按相同画面尺度显示，便于比较同一处细节')
        actual.clicked.connect(self.view.actual)
        row.addWidget(fit)
        row.addWidget(actual)
        hint = QLabel('滚轮缩放 · 拖动画面移动 · 拖动中线对比')
        hint.setWordWrap(True)
        hint.setMaximumWidth(285)
        row.addWidget(hint)
        self.restore = QPushButton('恢复原图')
        self.restore.clicked.connect(controller.restore)
        self.discard = QPushButton('丢弃结果')
        self.discard.clicked.connect(controller.discard)
        self.replace = QPushButton('替换原图')
        self.replace.clicked.connect(controller.replace)
        self.export = QPushButton('保存为新图片…')
        self.export.clicked.connect(controller.export_copy)
        from eidory.ui.review_actions import describe_review_actions
        describe_review_actions(self.export, self.replace, self.discard)
        for action in (self.export, self.replace, self.discard, self.restore):
            row.addWidget(action)
        tasks = QPushButton('任务与结果')
        tasks.clicked.connect(controller.open_tasks)
        row.addWidget(tasks)
        outer.addLayout(row)
        self.status = QLabel('结果暂存。另存可选择文件位置并加入图库；替换会更新原图所在位置。')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        outer.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumHeight(8)
        self.progress.setTextVisible(False)
        outer.addWidget(self.progress)
        self.dimensions.setWordWrap(True)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setFixedHeight(32)
        self.loader = BoardImageLoader(self)
        self.loader.loaded.connect(self.loaded)
        self.finished.connect(self.release)
        self.mode.currentIndexChanged.connect(self.parameters_changed)
        self.edge.valueChanged.connect(self.parameters_changed)
        self.parameters_changed()
        self.refresh()

    def select_history(self, _):
        self.selected_job_id = self.history.currentData()
        self.load_key = None
        self.refresh()
        if self.job:
            self.status.setText(self.job.get('error') or LABELS.get(self.job['state'], self.job['state']))

    def loaded(self, token, image, path):
        if token[0] == self.load_key:
            if image.isNull():
                self.status.setText('图片读取失败，请检查源文件是否仍存在。')
            else:
                self.view.set_image(token[1], image)

    def load_images(self, before, after=None):
        key = (before, after)
        if key == self.load_key:
            return
        self.load_key = key
        self.loader.reset()
        self.view.before, self.view.after = QImage(), QImage()
        self.view.fit_mode = True
        self.loader.request((key, 'before'), [before], 4096)
        if after:
            self.loader.request((key, 'after'), [after], 4096)
        self.view.update()

    def locked_job(self):
        return self.job if self.job and self.job['state'] in ACTIVE_STATES | {'ready', 'paused'} and self.job.get('source_info') else None

    def reduced_input(self):
        job = self.locked_job()
        return bool(job.get('input_reduced')) if job else self.image_id in self.c.input_reductions

    def processing_info(self):
        job = self.locked_job()
        if job:
            return job.get('input_info', job['source_info'])
        if self.info:
            w, h = processing_input_size(self.info['width'], self.info['height'], reduce_input=self.reduced_input())
            return dict(width=w, height=h)
        return None

    def factor(self):
        info = self.processing_info()
        return (2, 4)[self.mode.currentIndex()] if self.mode.currentIndex() < 2 else self.edge.value()/max(info['width'], info['height'])

    def toggle_input_reduction(self):
        if not self.info or self.locked_job() or self.c.working or self.c.w._database_maintenance_active:
            return
        if self.image_id in self.c.input_reductions:
            mode, edge = self.c.input_reductions.pop(self.image_id)
            message = '已撤销缩小输入，恢复使用原图；硬盘原图未改动。'
        else:
            if max(self.info['width'], self.info['height']) <= 2048:
                return
            self.c.input_reductions[self.image_id] = (self.mode.currentIndex(), self.edge.value())
            mode, edge = 0, 4096
            message = '已启用缩小 AI 输入，硬盘原图未改动；请对比结果后再决定是否替换。'
        with QSignalBlocker(self.mode), QSignalBlocker(self.edge):
            self.mode.setCurrentIndex(mode)
            self.edge.setValue(edge)
        self.parameters_changed()
        self.status.setText(message)

    def parameters_changed(self, *_):
        self.edge.setVisible(self.mode.currentIndex() == 2)
        self.refresh()

    def update_dimensions(self):
        self.valid = False
        info = self.processing_info()
        if self.info and info:
            original = self.locked_job()['source_info'] if self.locked_job() else self.info
            try:
                w, h = output_size(info['width'], info['height'], self.factor())
                if self.reduced_input():
                    text = f"原图 {original['width']} × {original['height']} · 输入 {info['width']} × {info['height']} → {w} × {h}"
                    if max(original['width'], original['height']) > max(w, h):
                        text += ' · 结果分辨率低于原图'
                else:
                    text = f"{info['width']} × {info['height']} → {w} × {h} · 输出长边最多 4096"
                self.dimensions.setText(text)
                self.valid = True
            except ValueError as exc:
                self.dimensions.setText(f"原图 {self.info['width']} × {self.info['height']} · {exc}")
            for index, factor in ((0, 2), (1, 4)):
                self.mode.model().item(index).setEnabled(max(info['width'], info['height'])*factor <= 4096)

    def refresh(self):
        active = self.c.repo.active()
        busy = self.c.working or self.c.w._database_maintenance_active
        jobs = self.c.repo.all(self.image_id)
        job = next((j for j in jobs if j['id'] == self.selected_job_id), jobs[0] if jobs else None)
        with QSignalBlocker(self.history):
            self.history.clear()
            for j in jobs:
                self.history.addItem(f"{j['created_at'][:19].replace('T', ' ')} · {j.get('factor', 0):g}× · {LABELS.get(j['state'], j['state'])}", j['id'])
            self.history.setCurrentIndex(self.history.findData(job['id']) if job else -1)
        self.history.setVisible(bool(jobs))
        self.job = job
        # Reopening a running/completed candidate must show its submitted settings.
        if job and job['state'] in ACTIVE_STATES | {'ready', 'paused'} and job.get('source_info'):
            if job.get('input_reduced'):
                self.c.input_reductions.setdefault(self.image_id, (0, 2048))
            factor = job['factor']
            with QSignalBlocker(self.mode), QSignalBlocker(self.edge):
                self.mode.setCurrentIndex(0 if factor == 2 else 1 if factor == 4 else 2)
                self.edge.setValue(max(job['target_width'], job['target_height']))
            self.edge.setVisible(self.mode.currentIndex() == 2)
        self.update_dimensions()
        self.start.setText('再次放大' if job and job['state'] == 'applied' else '开始放大')
        if job and job['state'] == 'paused':
            self.start.setText('继续核对原任务')
        if job and job['state'] == 'applied' and self.info:
            try:
                info = self.processing_info()
                w, h = output_size(info['width'], info['height'], self.factor())
                if not self.reduced_input():
                    self.dimensions.setText(f"再次放大：{self.info['width']} × {self.info['height']} → {w} × {h} · 下方保留上次对比")
            except ValueError:
                pass
        ready = bool(job and job['state'] == 'ready')
        same_active = bool(active and active['image_id'] == self.image_id)
        preview = getattr(self.c.w, 'preview_ai', None)
        other_generation = bool(self.c.w.reference_generation.repo.active() or self.c.w.reference_generation.working or (preview and preview.working))
        pending = any(j['state'] in ACTIVE_STATES | {'waiting'} or
                      (j['state'] == 'paused' and (not job or j['id'] != job['id'])) for j in jobs)
        self.start.setEnabled(self.valid and not busy and not active and not other_generation and not ready and not pending)
        self.start.setToolTip('请先替换或丢弃当前结果' if ready else '')
        self.mode.setEnabled(not busy and not self.locked_job())
        self.edge.setEnabled(self.mode.isEnabled())
        self.reduce_input_button.setVisible(bool(self.info and max(self.info['width'], self.info['height']) > 2048))
        self.reduce_input_button.setText('撤销缩小输入' if self.reduced_input() else '缩小输入至 2048')
        self.reduce_input_button.setEnabled(bool(self.info) and not busy and not self.locked_job())
        self.reduce_input_button.setToolTip('当前任务的输入已固定，请先完成、取消或丢弃结果' if self.locked_job()
                                           else '只缩小送入 AI 的临时副本，保留硬盘原图；可撤销')
        self.cancel.setVisible(same_active)
        self.cancel.setEnabled(same_active and not busy)
        self.cancel.setText('停止等待' if active and active['state'] == 'offline' else '取消任务')
        self.progress.setVisible(busy or same_active)
        self.replace.setEnabled(ready and not busy)
        self.export.setEnabled(bool(job and job['state'] in {'ready', 'applied', 'exported'} and not job.get('purged_at')) and not busy)
        self.discard.setEnabled(bool(job and job['state'] in {'ready', 'discarded'} and not job.get('purged_at')) and not busy and not active)
        self.discard.setText('撤销放弃' if job and job['state'] == 'discarded' else '放弃结果')
        self.restore.setVisible(bool(self.c.replacements.latest(self.image_id)))
        self.restore.setEnabled(not busy and not active and not ready)
        if job and job.get('original_path') and Path(job['original_path']).is_file():
            after = job.get('file_path') if job['state'] in {'ready', 'applied', 'exported'} else None
            before = job.get('comparison_path') or (job['original_path'] if job.get('input_reduced') else job.get('input_path', job['original_path']))
            self.load_images(before, after)
        elif self.info:
            image = self.c.w.store.get_image(self.image_id)
            if image:
                self.load_images(image.file_path)

    def release(self, *_):
        if self._released:
            return
        self._released = True
        self.loader.stop()
        self.view.before, self.view.after = QImage(), QImage()
        if self.c.panel is self:
            self.c.panel = None

    def closeEvent(self, event):
        self.release()
        super().closeEvent(event)
        self.finished.emit(0)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            event.accept()
        else:
            super().keyPressEvent(event)


class ImageUpscaler(QObject):
    completed = Signal(object, object)

    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self.repo = UpscaleWorkflow(window.store, window.paths)
        self.replacements = ImageReplacement(window.store, window.paths)
        self.panel = None
        self.tasks = None
        self.batch_review = None
        self.batch_notice = None
        self.notified_job_id = None
        self.result_button = QPushButton('AI 任务与结果')
        self.result_button.clicked.connect(self.open_tasks)
        window.statusBar().addPermanentWidget(self.result_button)
        # Remember the pre-reduction parameters while moving between preview
        # pages. No source file or image metadata changes until explicit adoption.
        self.input_reductions = {}
        self.working = False
        self.completed.connect(self.finished)
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.poll)
        self.timer.start()

    @property
    def w(self):
        return self._window()

    def open(self, image, *, preview=None):
        if not image or self.w._closing:
            return
        if preview is not None and not preview.annotation_controls.flush(interactive=True):
            return
        if self.panel and (self.panel.image_id != image.id or self.panel.preview is not preview):
            self.panel.close()
        if not self.panel:
            self.panel = UpscalePanel(self, image, preview=preview)
            if preview is not None:
                preview.show_upscale_panel(self.panel)
        self.panel.show()
        if preview is None:
            self.panel.raise_()
            self.panel.activateWindow()
        else:
            preview._focus_active_page()
        self.inspect()

    def open_tasks(self):
        if hasattr(self.w, "task_center"):
            self.w.task_center.open(1)
            self.tasks.refresh()
            return
        if self.tasks is None:
            from eidory.ui.upscale_tasks import UpscaleTasks
            self.tasks = UpscaleTasks(self)
        self.tasks.refresh()
        self.tasks.show()
        self.tasks.raise_()
        self.tasks.activateWindow()

    def open_selection(self, images):
        if not images or self.working or self.w._closing or self.w._database_maintenance_active:
            return
        self.open_tasks()
        self.tasks.enqueue(list(images))

    def open_batch(self, batch_id):
        if not self.repo.batch(batch_id) or self.w._closing:
            return
        if self.batch_review and self.batch_review.batch_id != batch_id:
            self.batch_review.close()
        if not self.batch_review:
            from eidory.ui.upscale_batch_review import UpscaleBatchReview
            self.batch_review = UpscaleBatchReview(self, batch_id)
        self.batch_review.refresh()
        self.batch_review.show()
        self.batch_review.raise_()
        self.batch_review.activateWindow()

    def notify_batches(self):
        jobs = self.repo.all()
        ready = sum(j['state'] == 'ready' for j in jobs)
        if hasattr(self.w, 'task_center'):
            self.w.task_center.update_counts('upscale', jobs)
        self.result_button.setText(f'AI 任务与结果 · 待审 {ready} 张' if ready else 'AI 任务与结果')
        if self.batch_notice or self.w._closing:
            return
        for batch in self.repo.batches():
            if not batch['complete'] or not (batch['ready'] or batch['failed']):
                continue
            # Retrying a failed slot creates a new completion to notify. Review
            # decisions alone do not trigger repeated completion pop-ups.
            signature = hashlib.sha256('|'.join(j['id'] for j in batch['jobs']).encode()).hexdigest()
            key = 'upscale.batch_notified.' + batch['id']
            if self.w.store.get_setting(key) == signature:
                continue
            self.w.store.set_setting(key, signature)
            if self.batch_review and self.batch_review.batch_id == batch['id'] and self.batch_review.isVisible():
                continue
            message = (f"本批次 {batch['total']} 张已处理结束：{batch['ready']} 张待审核、"
                       f"{batch['failed']} 张失败、{batch['cancelled']} 张取消。\n"
                       '原图尚未自动替换。请逐张对比后决定采用或放弃；也可以稍后审核。')
            self.w.statusBar().showMessage(message.replace('\n', ' '))
            notice = QMessageBox(QMessageBox.Icon.Information, 'AI 批量放大已完成', message, parent=self.w)
            notice.setWindowModality(Qt.WindowModality.NonModal)
            review = notice.addButton('审核本批次', QMessageBox.ButtonRole.AcceptRole)
            notice.addButton('稍后审核', QMessageBox.ButtonRole.RejectRole)
            review.clicked.connect(lambda _=False, key=batch['id']: QTimer.singleShot(0, lambda: self.open_batch(key)))
            notice.finished.connect(lambda *_: setattr(self, 'batch_notice', None))
            self.batch_notice = notice
            notice.show()
            QApplication.alert(self.w, 5000)
            break

    def open_job(self, job_id):
        job = self.repo.get(job_id)
        image = self.w.store.get_image(job['image_id'])
        if not image:
            if self.tasks:
                self.tasks.message.setText('原图已移出图库；任务记录保留，可复制诊断信息。')
            return
        preview = next((p for p in self.w._preview_dialogs if p.current_image() and p.current_image().id == image.id), None)
        if preview is None:
            before = set(self.w._preview_dialogs)
            self.w._open_image_preview(image)
            preview = next(iter(set(self.w._preview_dialogs) - before), None)
        self.open(image, preview=preview)
        if self.panel:
            self.panel.selected_job_id = job_id
            self.panel.load_key = None
            self.panel.refresh()
        if preview:
            preview.show()
            preview.raise_()
            preview.activateWindow()

    def export_copy(self):
        panel = self.panel
        if not panel or not panel.job or self.working:
            return
        job = panel.job
        source = Path(job['source_path'])
        default = source.with_name(f"{source.stem}-AI-{job['factor']:g}x-{job['id'][:8]}.png")
        owner = panel.preview or panel
        path, _ = QFileDialog.getSaveFileName(owner, '保存 AI 结果为新图片', str(default), 'PNG 图片 (*.png)')
        # Cocoa's native save panel can reactivate the main application window
        # when dismissed. Return focus to its owning comparison, even on cancel.
        if owner.isVisible():
            owner.raise_()
            owner.activateWindow()
        if not path:
            return
        if not Path(path).suffix:
            path += '.png'
        def done(image):
            if owner.isVisible():
                # Updating the gallery can change native window focus. Defer
                # that refresh until this preview closes, keeping comparison
                # and zoom intact while the user examines the saved result.
                if not getattr(owner, '_upscale_copy_refresh_pending', False):
                    owner._upscale_copy_refresh_pending = True
                    owner.finished.connect(self.refresh_exported_library)
            else:
                self.refresh_exported_library()
            if self.panel:
                self.panel.status.setText('已保存新图片并加入图库；原图保持不变。')
            self.w.statusBar().showMessage('已保存 AI 副本：' + image.file_name)
        self.run(lambda: self.repo.export_copy(job['id'], path), done, stage='另存新图片', job_id=job['id'])

    def refresh_exported_library(self, *_):
        if self.w._closing:
            return
        self.w._refresh_folders()
        self.w._refresh_collections()
        self.w._refresh_tags()
        if self.w.current_result_mode == 'library':
            self.w._reload_images()

    def inspect(self):
        panel = self.panel
        if not panel or not panel.inspection_pending or self.working:
            return
        image = self.w.store.get_image(panel.image_id)
        if not image:
            panel.status.setText('原图已移出图库。')
            return
        panel.inspection_pending = False
        def done(info):
            if self.panel is panel:
                panel.info = info
                panel.parameters_changed()
                panel.refresh()
                job = panel.job
                if job:
                    panel.status.setText(job.get('error') or LABELS.get(job['state'], job['state']))
        self.run(lambda: source_info(image.file_path), done)

    def run(self, operation, callback, *, stage=None, job_id=None):
        if self.working or self.w._closing or self.w._database_maintenance_active:
            return
        self.working = True
        self.w.reference_generation.hold_indexing(True)
        if self.panel:
            self.panel.refresh()
        if self.batch_review:
            self.batch_review.refresh()
        def worker():
            try:
                result = operation()
            except Exception as exc:
                logging.getLogger('eidory.runtime').exception('AI upscale operation failed')
                if job_id and stage:
                    diagnostic = traceback.format_exc()
                    try:
                        job = self.repo.get(job_id)
                        job.update(error=f'{stage}失败：{connection_message(exc)}', stage=stage, diagnostic=diagnostic)
                        self.repo.save(job)
                    except Exception:
                        logging.getLogger('eidory.runtime').exception('Could not persist upscale diagnostic')
                result = exc
            self.completed.emit(callback, result)
        if not self.w._start_background_task(worker, name='seedvr2-upscale'):
            self.working = False
            self.w.reference_generation.hold_indexing(bool(self.w.reference_generation.active()))

    def finished(self, callback, result):
        self.working = False
        if self.w._closing:
            return
        if isinstance(result, Exception):
            message = connection_message(result)
            if self.panel:
                self.panel.status.setText(message)
            if self.batch_review:
                self.batch_review.status.setText(message + '；本张尚未完成决定，可以重试或稍后处理。')
            self.w.statusBar().showMessage('AI 放大：' + message)
        else:
            try:
                callback(result)
            except Exception as exc:
                logging.getLogger('eidory.runtime').exception('AI upscale result handling failed')
                if self.panel:
                    self.panel.status.setText(connection_message(exc))
                self.w.statusBar().showMessage('AI 放大：' + connection_message(exc))
        if self.panel:
            self.panel.refresh()
        self.w.reference_generation.hold_indexing(bool(self.w.reference_generation.active()))
        if self.tasks:
            self.tasks.refresh()
        if self.batch_review:
            self.batch_review.refresh()
        self.notify_batches()
        self.inspect()

    def generate(self):
        panel = self.panel
        if not panel or not panel.valid or self.working:
            return
        if panel.job and panel.job['state'] == 'paused':
            self.run(lambda: self.repo.resume(panel.job['id']), self.updated)
            return
        image_id, factor, reduce_input = panel.image_id, panel.factor(), panel.reduced_input()
        url = self.w.store.get_setting('comfy.url', 'http://127.0.0.1:8188')
        panel.selected_job_id = None
        panel.status.setText('正在准备原图并连接本机 ComfyUI…')
        def submit():
            job = self.repo.create(image_id, factor, url, reduce_input=reduce_input)
            return self.repo.submit(job['id'])
        self.run(submit, self.updated)

    def updated(self, job):
        if not job:
            return
        if self.panel and self.panel.image_id == job['image_id']:
            self.panel.status.setText(job.get('error') or LABELS.get(job['state'], job['state']))
        if job['state'] == 'ready':
            self.notified_job_id = job['id']
            self.w.statusBar().showMessage('AI 放大完成，在「后台任务 → AI 放大」审核结果。')
            if hasattr(self, 'result_button'):
                self.result_button.setText('AI 结果待确认')

    def poll(self):
        if self.w._closing or self.w._database_maintenance_active:
            return
        if self.working:
            if self.tasks and self.tasks.isVisible():
                self.tasks.refresh()
            return
        self.notify_batches()
        job = self.repo.active()
        if job:
            if job.get('next_check_at', 0) > time.time():
                return
            self.run(lambda: self.repo.reconcile(job['id']), self.updated)
        elif self.repo.waiting() and self.w.store.get_setting('upscale.queue_paused', '0') != '1' and not self.w.reference_generation.repo.active() and not (getattr(self.w, 'preview_ai', None) and self.w.preview_ai.working):
            self.run(self.repo.start_next, self.updated)
        elif self.panel:
            self.inspect()

    def cancel(self):
        job = self.repo.active()
        if job:
            self.run(lambda: self.repo.cancel(job['id']), self.updated)

    def discard(self):
        if self.panel and self.panel.job:
            job_id = self.panel.job['id']
            undo = self.panel.job['state'] == 'discarded'
            self.run(lambda: self.repo.discard(job_id, undo=undo), self.updated)

    def replace(self):
        if self.panel and self.panel.job:
            job = self.repo.get(self.panel.job['id'])
            self.run(lambda: self.replacements.replace(job), self.media_changed, stage='替换原图', job_id=job['id'])

    def restore(self):
        if self.panel:
            image_id = self.panel.image_id
            self.run(lambda: self.replacements.restore(image_id), self.media_changed)

    def media_changed(self, image):
        if not image:
            return
        self.input_reductions.pop(image.id, None)
        w = self.w
        # Refresh records in place: no search rerun, project switching or board rearrangement.
        def updated(items):
            return [replace(image, score=x.score) if x.id == image.id else x for x in items]
        for name in ('current_semantic_images', 'current_semantic_filtered_images', 'current_color_images',
                     'current_color_filtered_images', 'current_inspiration_images', 'current_inspiration_filtered_images',
                     'current_temp_project_images', 'current_creative_node_images', 'current_creative_node_filtered_images',
                     'current_chain_images', 'current_chain_filtered_images', 'current_duplicate_images'):
            setattr(w, name, updated(getattr(w, name)))
        w.grid_view.refresh_image(image)
        if image.id in w._board_image_by_id:
            w._board_image_by_id[image.id] = image
            w.project_board_view.refresh_media_image(image)
            w.creative_flow.board_inspector.refresh_preview(image.id)
        for preview in list(w._preview_dialogs):
            preview.images = [image if x.id == image.id else x for x in preview.images]
            if preview.current_image() and preview.current_image().id == image.id:
                preview._clear_preview_pixmap_cache()
                preview._refresh()
        if w.selected_image and w.selected_image.id == image.id:
            w.selected_image = image
            w._show_image_details(image)
        w._invalidate_near_duplicate_hash_cache()
        if self.panel and self.panel.image_id == image.id:
            self.panel.info = source_info(image.file_path)
            # Offer a valid next operation after adopting a custom size, rather than
            # displaying an error because that target now equals the new original.
            with QSignalBlocker(self.panel.mode), QSignalBlocker(self.panel.edge):
                edge = max(image.width, image.height)
                self.panel.mode.setCurrentIndex(0 if edge <= 2048 else 2)
                self.panel.edge.setValue(min(4096, edge * 2))
            self.panel.parameters_changed()
            message = '操作完成，标签、AI 语义与项目引用保持不变。'
            record = self.replacements.latest(image.id)
            if record and record.get('metadata_omissions'):
                message += f" 已跳过 {len(record['metadata_omissions'])} 项无效拍摄参数；恢复原图可还原原始文件。"
            self.panel.status.setText(message)
            self.panel.load_key = None
        w.statusBar().showMessage('图片已更新，原有信息保留。')

    def close(self):
        self.timer.stop()
        if self.batch_notice:
            self.batch_notice.close()
        if self.batch_review:
            self.batch_review.close()
        if self.panel:
            self.panel.close()
        if self.tasks:
            self.tasks.close()
