"""One comparison window for explicit, resumable decisions within a batch."""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QDialog, QFileDialog, QLabel, QPushButton, QVBoxLayout

from eidory.ui.board_image_loader import BoardImageLoader
from eidory.ui.image_upscale import WipeComparison, LABELS
from eidory.ui.wrapping_layout import WrappingLayout


class UpscaleBatchReview(QDialog):
    def __init__(self, controller, batch_id):
        super().__init__(controller.w)
        self.c, self.batch_id = controller, batch_id
        self.job_id = None
        self.batch = None
        self.load_key = None
        self.generation = 0
        self.loaded_sides = set()
        self.closed = False
        self.saved_copy = False
        self.setWindowTitle('AI 放大 · 批次审核')
        self.resize(1120, 820)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.details)
        self.view = WipeComparison()
        layout.addWidget(self.view, 1)
        tools = WrappingLayout()
        for text, callback in [('适应窗口', self.view.fit), ('结果 100%', self.view.actual)]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            tools.addWidget(button)
        tools.addWidget(QLabel('滚轮缩放 · 拖动画面移动 · 拖动中线对比'))
        layout.addLayout(tools)
        actions = WrappingLayout()
        self.adopt = QPushButton('采用并替换原图')
        self.adopt.clicked.connect(lambda: self.decide('adopt'))
        self.copy = QPushButton('采用并另存为副本…')
        self.copy.clicked.connect(self.export_copy)
        self.discard = QPushButton('放弃结果，保留原图')
        self.discard.clicked.connect(lambda: self.decide('discard'))
        self.undo = QPushButton('撤销本张决定')
        self.undo.clicked.connect(self.undo_decision)
        from eidory.ui.review_actions import describe_review_actions
        describe_review_actions(self.copy, self.adopt, self.discard)
        for button in (self.copy, self.adopt, self.discard, self.undo):
            actions.addWidget(button)
        layout.addLayout(actions)
        navigation = WrappingLayout()
        self.previous = QPushButton('← 上一张')
        self.previous.clicked.connect(lambda: self.move(-1))
        self.next = QPushButton('下一张 →')
        self.next.clicked.connect(lambda: self.move(1))
        self.later = QPushButton('稍后处理，查看下一待审张')
        self.later.clicked.connect(self.next_pending)
        close = QPushButton('关闭，稍后继续审核')
        close.clicked.connect(self.close)
        for button in (self.previous, self.next, self.later, close):
            navigation.addWidget(button)
        layout.addLayout(navigation)
        self.status = QLabel('每张结果都由你决定；采用或放弃成功后进入下一待审张。关闭窗口不会丢失待审结果。')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
        self.loader = BoardImageLoader(self)
        self.loader.loaded.connect(self.loaded)
        self.refresh()

    def current(self):
        return next((j for j in self.batch['jobs'] if j['id'] == self.job_id), None) if self.batch else None

    def refresh(self):
        if self.closed:
            return
        self.batch = self.c.repo.batch(self.batch_id)
        if not self.batch:
            return
        jobs = self.batch['jobs']
        if not self.current():
            self.job_id = next((j['id'] for j in jobs if j['state'] == 'ready'), jobs[0]['id'])
        job = self.current()
        index = next(i for i, j in enumerate(jobs) if j['id'] == self.job_id)
        b = self.batch
        self.summary.setText(f"本批次第 {index+1} / {b['total']} 张 · 已审核 {b['reviewed']} · 待审核 {b['ready']}"
                             f" · 处理中 {b['pending']} · 失败 {b['failed']} · 取消 {b['cancelled']}")
        source = Path(job.get('source_path', ''))
        dimensions = ''
        if info := job.get('source_info'):
            dimensions = f" · {info['width']} × {info['height']} → {job['target_width']} × {job['target_height']}"
        self.details.setText(f"{job.get('file_name', source.name)}{dimensions}\n原图文件夹：{source.parent}\n"
                             f"状态：{LABELS.get(job['state'], job['state'])}" +
                             (f"\n{job['adjustment_reason']}" if job.get('adjustment_reason') else '') +
                             (f"\n{job['error']}" if job.get('error') else ''))
        before = job.get('comparison_path') or job.get('input_path') or job.get('original_path') or str(source)
        after = job.get('file_path') if job['state'] in {'ready', 'applied', 'exported', 'discarded'} and not job.get('purged_at') else None
        key = (job['id'], before, after)
        if self.load_key != key:
            self.load_key = key
            self.generation += 1
            self.loader.reset()
            self.loaded_sides.clear()
            self.view.before, self.view.after = QImage(), QImage()
            self.view.fit()
            for side, path in [('before', before), ('after', after)]:
                if path:
                    self.loader.request((self.generation, side), [path], 0)
        busy = self.c.working or self.c.w._database_maintenance_active
        ready = job['state'] == 'ready' and self.loaded_sides == {'before', 'after'} and not busy
        self.adopt.setEnabled(ready)
        self.copy.setEnabled(ready)
        self.discard.setEnabled(ready)
        self.undo.setEnabled(not busy and not job.get('purged_at') and
                             (job['state'] == 'discarded' or (job['state'] == 'applied' and
                              (self.c.replacements.latest(job['image_id']) or {}).get('job_id') == job['id'])))
        self.previous.setEnabled(not busy and index > 0)
        self.next.setEnabled(not busy and index < len(jobs)-1)
        self.later.setEnabled(not busy and any(j['state'] == 'ready' and j['id'] != self.job_id for j in jobs))

    def loaded(self, token, image, path):
        generation, side = token
        if self.closed or generation != self.generation:
            return
        if image.isNull():
            self.status.setText('图片读取失败，尚不能采用或放弃。请从任务中心检查文件与诊断信息。')
        else:
            self.loaded_sides.add(side)
            self.view.set_image(side, image)
        self.refresh()

    def move(self, delta):
        if self.c.working:
            return
        jobs = self.batch['jobs']
        index = next(i for i, j in enumerate(jobs) if j['id'] == self.job_id) + delta
        if 0 <= index < len(jobs):
            self.job_id = jobs[index]['id']
            self.status.setText('本张尚未做出新的决定。')
            self.refresh()

    def next_pending(self):
        if self.c.working or self.closed:
            return
        self.refresh()
        jobs = self.batch['jobs']
        index = next(i for i, j in enumerate(jobs) if j['id'] == self.job_id)
        remaining = jobs[index+1:] + jobs[:index]
        target = next((j for j in remaining if j['state'] == 'ready'), None)
        if target:
            self.job_id = target['id']
            self.refresh()
        elif not self.batch['ready']:
            text = '本批次已生成的结果全部审核完成。'
            if self.batch['pending']:
                text += ' 其余图片仍在处理，完成后会再次提示。'
            if self.batch['failed']:
                text += f" 另有 {self.batch['failed']} 张失败，可在任务中心重试。"
            self.status.setText(text)

    def decide(self, decision):
        self.refresh()
        button = self.adopt if decision == 'adopt' else self.discard
        if not button.isEnabled():
            return
        job = self.c.repo.get(self.job_id)
        if decision == 'adopt':
            for preview in self.c.w._preview_dialogs:
                image = preview.current_image()
                if image and image.id == job['image_id'] and not preview.annotation_controls.flush(interactive=True):
                    return
        def done(result):
            if decision == 'adopt':
                self.c.media_changed(result)
            if not self.closed:
                self.status.setText('已采用本张结果。' if decision == 'adopt' else '已放弃本张结果，原图保持不变。')
                self.next_pending()
                self.raise_()
                self.activateWindow()
        operation = (lambda: self.c.replacements.replace(job)) if decision == 'adopt' else (lambda: self.c.repo.discard(job['id']))
        self.c.run(operation, done, stage='批次采用' if decision == 'adopt' else '批次放弃', job_id=job['id'])

    def export_copy(self):
        self.refresh()
        if not self.copy.isEnabled():
            return
        job = self.c.repo.get(self.job_id)
        source = Path(job['source_path'])
        default = source.with_name(f"{source.stem}-AI-{job['factor']:g}x-{job['id'][:8]}.png")
        path, _ = QFileDialog.getSaveFileName(self, '采用并另存 AI 副本', str(default), 'PNG 图片 (*.png)')
        if self.isVisible():
            self.raise_()
            self.activateWindow()
        if not path:
            return
        if not Path(path).suffix:
            path += '.png'
        def done(image):
            self.saved_copy = True
            if self.closed:
                self.c.refresh_exported_library()
            else:
                self.status.setText('已另存副本并加入图库，原图保持不变。')
                self.next_pending()
        self.c.run(lambda: self.c.repo.export_copy(job['id'], path), done, stage='批次另存副本', job_id=job['id'])

    def undo_decision(self):
        self.refresh()
        if not self.undo.isEnabled():
            return
        job = self.c.repo.get(self.job_id)
        def operation():
            if job['state'] == 'discarded':
                return self.c.repo.discard(job['id'], undo=True)
            image = self.c.replacements.restore(job['image_id'])
            restored = self.c.repo.get(job['id'])
            restored.update(state='ready', error='')
            self.c.repo.save(restored)
            return image
        def done(value):
            if job['state'] == 'applied':
                self.c.media_changed(value)
            if not self.closed:
                self.status.setText('已撤销本张决定，恢复待审核。')
                self.refresh()
        self.c.run(operation, done, stage='撤销审核决定', job_id=job['id'])

    def done(self, result):
        if self.closed:
            super().done(result)
            return
        self.closed = True
        self.loader.stop()
        if self.saved_copy:
            self.c.refresh_exported_library()
        if self.c.batch_review is self:
            self.c.batch_review = None
        super().done(result)
