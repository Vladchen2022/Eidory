"""Single preview entry, background SeedVR jobs, and a shared-coordinate wipe comparison."""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from weakref import ref

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QSignalBlocker, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from eidory.core.comfy_reference import ACTIVE_STATES, STATE_LABELS, connection_message
from eidory.core.image_upscale import ImageReplacement, UpscaleJobs, output_size, source_info
from eidory.ui.board_image_loader import BoardImageLoader

LABELS = dict(STATE_LABELS, applied='已替换原图，原有信息保留', restored='已恢复原图', discarded='已丢弃结果，原图未改变；可撤销丢弃')


class WipeComparison(QWidget):
    """Both images use the output's coordinate system, including at 100%."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.before, self.after = QImage(), QImage()
        self.scale = 1.0
        self.pan = QPointF()
        self.divider = .5
        self.fit_mode = True
        self.drag = None
        self.setMinimumSize(350, 260)
        self.setAccessibleName('原图与放大结果对比；滚轮缩放，拖动画面移动，拖动中线对比')
        self.setMouseTracking(True)

    def set_image(self, side, image):
        setattr(self, side, image)
        if self.fit_mode:
            self.fit()
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
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, '正在读取原图…')
            return
        size = self.dimensions()
        w, h = size.width()*self.scale, size.height()*self.scale
        rect = QRectF((self.width()-w)/2 + self.pan.x(), (self.height()-h)/2 + self.pan.y(), w, h)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(rect, self.before)
        x = int(self.width()*self.divider)
        if not self.after.isNull():
            p.save()
            p.setClipRect(x, 0, self.width()-x, self.height())
            p.drawImage(rect, self.after)
            p.restore()
            p.setPen(QPen(QColor('white'), 2))
            p.drawLine(x, 0, x, self.height())
            p.setBrush(QColor('#ffffff'))
            p.drawEllipse(QPointF(x, self.height()/2), 8, 8)
        p.setPen(QColor('white'))
        p.fillRect(8, 8, 58, 26, QColor(0, 0, 0, 150))
        p.drawText(18, 26, '原图')
        if not self.after.isNull():
            p.fillRect(self.width()-135, 8, 127, 26, QColor(0, 0, 0, 150))
            p.drawText(self.width()-125, 26, 'SeedVR2 7B')

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


class UpscalePanel(QDialog):
    def __init__(self, controller, image):
        super().__init__(controller.w)
        self.c, self.image_id = controller, image.id
        self.info = None
        self.inspection_pending = True
        self.job = None
        self.load_key = None
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle('AI 放大 · ' + image.file_name)
        self.resize(1080, 780)
        self.setMinimumSize(760, 540)
        outer = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel('SeedVR2 7B · 本机放大'), 1)
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
        self.start = QPushButton('开始放大')
        self.start.setProperty('variant', 'primary')
        self.start.clicked.connect(controller.generate)
        header.addWidget(self.start)
        self.cancel = QPushButton('取消任务')
        self.cancel.clicked.connect(controller.cancel)
        header.addWidget(self.cancel)
        outer.addLayout(header)
        self.dimensions = QLabel('正在读取图片…')
        outer.addWidget(self.dimensions)
        self.view = WipeComparison()
        outer.addWidget(self.view, 1)
        row = QHBoxLayout()
        fit = QPushButton('适应窗口')
        fit.clicked.connect(self.view.fit)
        actual = QPushButton('结果 100%')
        actual.setToolTip('原图按相同画面尺度显示，便于比较同一处细节')
        actual.clicked.connect(self.view.actual)
        row.addWidget(fit)
        row.addWidget(actual)
        row.addWidget(QLabel('滚轮缩放 · 拖动画面移动 · 拖动中线对比'), 1)
        self.restore = QPushButton('恢复原图')
        self.restore.clicked.connect(controller.restore)
        row.addWidget(self.restore)
        self.discard = QPushButton('丢弃结果')
        self.discard.clicked.connect(controller.discard)
        row.addWidget(self.discard)
        self.replace = QPushButton('替换原图')
        self.replace.setProperty('variant', 'primary')
        self.replace.clicked.connect(controller.replace)
        row.addWidget(self.replace)
        outer.addLayout(row)
        self.status = QLabel('结果临时保存；对比后再决定是否替换。')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        outer.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumHeight(8)
        self.progress.setTextVisible(False)
        outer.addWidget(self.progress)
        self.loader = BoardImageLoader(self)
        self.loader.loaded.connect(self.loaded)
        self.finished.connect(self.release)
        self.mode.currentIndexChanged.connect(self.parameters_changed)
        self.edge.valueChanged.connect(self.parameters_changed)
        self.parameters_changed()
        self.refresh()

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

    def factor(self):
        return (2, 4)[self.mode.currentIndex()] if self.mode.currentIndex() < 2 else self.edge.value()/max(self.info['width'], self.info['height'])

    def parameters_changed(self, *_):
        self.edge.setVisible(self.mode.currentIndex() == 2)
        self.valid = False
        if self.info:
            try:
                w, h = output_size(self.info['width'], self.info['height'], self.factor())
                self.dimensions.setText(f"{self.info['width']} × {self.info['height']} → {w} × {h} · 输出长边最多 4096")
                self.valid = True
            except ValueError as exc:
                self.dimensions.setText(f"原图 {self.info['width']} × {self.info['height']} · {exc}")
            for index, factor in ((0, 2), (1, 4)):
                self.mode.model().item(index).setEnabled(max(self.info['width'], self.info['height'])*factor <= 4096)
        self.refresh()

    def refresh(self):
        active = self.c.repo.active()
        busy = self.c.working or self.c.w._database_maintenance_active
        jobs = self.c.repo.all(self.image_id)
        job = jobs[0] if jobs else None
        self.job = job
        # Reopening a running/completed candidate must show its submitted settings.
        if job and job['state'] in ACTIVE_STATES | {'ready'}:
            factor = job['factor']
            with QSignalBlocker(self.mode), QSignalBlocker(self.edge):
                self.mode.setCurrentIndex(0 if factor == 2 else 1 if factor == 4 else 2)
                self.edge.setValue(max(job['target_width'], job['target_height']))
            self.edge.setVisible(self.mode.currentIndex() == 2)
            original = job['source_info']
            self.dimensions.setText(f"{original['width']} × {original['height']} → {job['target_width']} × {job['target_height']} · 输出长边最多 4096")
        self.start.setText('再次放大' if job and job['state'] == 'applied' else '开始放大')
        if job and job['state'] == 'applied' and self.info:
            try:
                w, h = output_size(self.info['width'], self.info['height'], self.factor())
                self.dimensions.setText(f"再次放大：{self.info['width']} × {self.info['height']} → {w} × {h} · 下方保留上次对比")
            except ValueError:
                pass
        ready = bool(job and job['state'] == 'ready')
        same_active = bool(active and active['image_id'] == self.image_id)
        other_generation = bool(self.c.w.reference_generation.active() or self.c.w.reference_generation.working)
        self.start.setEnabled(self.valid and not busy and not active and not other_generation and not ready)
        self.start.setToolTip('请先替换或丢弃当前结果' if ready else '')
        self.mode.setEnabled(not busy and not same_active and not ready)
        self.edge.setEnabled(self.mode.isEnabled())
        self.cancel.setVisible(same_active)
        self.cancel.setEnabled(same_active and not busy)
        self.progress.setVisible(busy or same_active)
        self.replace.setEnabled(ready and not busy)
        self.discard.setEnabled(bool(job and job['state'] in {'ready', 'discarded'}) and not busy and not active)
        self.discard.setText('撤销丢弃' if job and job['state'] == 'discarded' else '丢弃结果')
        self.restore.setVisible(bool(self.c.replacements.latest(self.image_id)))
        self.restore.setEnabled(not busy and not active and not ready)
        if job and job.get('original_path') and Path(job['original_path']).is_file():
            after = job.get('file_path') if job['state'] in {'ready', 'applied'} else None
            self.load_images(job.get('input_path', job['original_path']), after)
        elif self.info:
            image = self.c.w.store.get_image(self.image_id)
            if image:
                self.load_images(image.file_path)

    def release(self, *_):
        self.loader.stop()
        self.view.before, self.view.after = QImage(), QImage()
        if self.c.panel is self:
            self.c.panel = None

    def closeEvent(self, event):
        self.release()
        super().closeEvent(event)


class ImageUpscaler(QObject):
    completed = Signal(object, object)

    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self.repo = UpscaleJobs(window.store, window.paths)
        self.replacements = ImageReplacement(window.store, window.paths)
        self.panel = None
        self.working = False
        self.completed.connect(self.finished)
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.poll)
        self.timer.start()

    @property
    def w(self):
        return self._window()

    def open(self, image):
        if not image or self.w._closing:
            return
        if self.panel and self.panel.image_id != image.id:
            self.panel.close()
        if not self.panel:
            self.panel = UpscalePanel(self, image)
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()
        self.inspect()

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

    def run(self, operation, callback):
        if self.working or self.w._closing or self.w._database_maintenance_active:
            return
        self.working = True
        self.w.reference_generation.hold_indexing(True)
        if self.panel:
            self.panel.refresh()
        def worker():
            try:
                result = operation()
            except Exception as exc:
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
            self.w.statusBar().showMessage('AI 放大：' + message)
        else:
            try:
                callback(result)
            except Exception as exc:
                if self.panel:
                    self.panel.status.setText(connection_message(exc))
                self.w.statusBar().showMessage('AI 放大：' + connection_message(exc))
        if self.panel:
            self.panel.refresh()
        self.w.reference_generation.hold_indexing(bool(self.w.reference_generation.active()))
        self.inspect()

    def generate(self):
        panel = self.panel
        if not panel or not panel.valid or self.working:
            return
        image_id, factor = panel.image_id, panel.factor()
        url = self.w.store.get_setting('comfy.url', 'http://127.0.0.1:8188')
        panel.status.setText('正在准备原图并连接本机 ComfyUI…')
        def submit():
            job = self.repo.create(image_id, factor, url)
            return self.repo.submit(job['id'])
        self.run(submit, self.updated)

    def updated(self, job):
        if self.panel and self.panel.image_id == job['image_id']:
            self.panel.status.setText(job.get('error') or LABELS.get(job['state'], job['state']))
        if job['state'] == 'ready':
            self.w.statusBar().showMessage('AI 放大完成，打开原图的「AI 放大」查看并对比。')

    def poll(self):
        if self.working or self.w._closing or self.w._database_maintenance_active:
            return
        job = self.repo.active()
        if job:
            self.run(lambda: self.repo.reconcile(job['id']), self.updated)
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
            self.run(lambda: self.replacements.replace(job), self.media_changed)

    def restore(self):
        if self.panel:
            image_id = self.panel.image_id
            self.run(lambda: self.replacements.restore(image_id), self.media_changed)

    def media_changed(self, image):
        if not image:
            return
        w = self.w
        # Refresh records in place: no search rerun, project switching or board rearrangement.
        def updated(items):
            return [replace(image, score=x.score) if x.id == image.id else x for x in items]
        for name in ('current_semantic_images', 'current_semantic_filtered_images', 'current_color_images',
                     'current_color_filtered_images', 'current_inspiration_images', 'current_inspiration_filtered_images',
                     'current_temp_project_images', 'current_creative_node_images', 'current_creative_node_filtered_images',
                     'current_chain_images', 'current_chain_filtered_images', 'current_duplicate_images'):
            setattr(w, name, updated(getattr(w, name)))
        w.grid_view.set_images(updated(w.grid_view.images()), badges_by_image_id=w.grid_view._badges_by_image_id)
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
            self.panel.status.setText('操作完成，标签、AI 语义与项目引用保持不变。')
            self.panel.load_key = None
        w.statusBar().showMessage('图片已更新，原有信息保留。')

    def close(self):
        self.timer.stop()
        if self.panel:
            self.panel.close()
