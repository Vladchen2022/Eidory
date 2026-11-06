"""One place to enqueue images, review results and inspect failures."""
from pathlib import Path
import json
from datetime import datetime

from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout)
from eidory.core.comfy_reference import ACTIVE_STATES, STATE_LABELS
from eidory.core.upscale_workflow import batch_size_plan
from eidory.ui.wrapping_layout import WrappingLayout

LABELS = dict(STATE_LABELS, waiting='本机排队', applied='已替换', restored='已恢复', exported='已另存为副本')


class BatchUpscaleDialog(QDialog):
    def __init__(self, images, parent=None):
        super().__init__(parent)
        self.images = list(images)
        self.setWindowTitle(f'AI 批量放大 · {len(images)} 张图片')
        self.resize(1020, 560)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel('首选放大倍数'))
        self.mode = QComboBox()
        self.mode.setAccessibleName('批量放大首选倍数')
        self.mode.addItem('2 倍', 2)
        self.mode.addItem('4 倍', 4)
        row.addWidget(self.mode)
        row.addStretch()
        layout.addLayout(row)
        note = QLabel('每张图优先使用所选倍数；4 倍超限时改为 2 倍，2 倍仍超限时先缩小临时输入至长边 2048，再放大 2 倍。\n'
                      '输出长边最多 4096。缩小输入后的输出可能比原图小，具体尺寸见下表。')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 6)
        self.table.setAccessibleName('逐张放大方案')
        self.table.setHorizontalHeaderLabels(['图片', '原图尺寸', 'AI 输入尺寸', '实际倍数', '输出尺寸', '自动调整说明'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 220)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        review = QLabel('这些图片会组成同一个批次依次处理。完成后提示你逐张对比，决定采用或放弃；审核前保留原图。')
        review.setWordWrap(True)
        layout.addWidget(review)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText('加入队列')
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.mode.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def factor(self):
        return self.mode.currentData()

    def refresh(self, *_):
        counts = dict(four=0, two=0, reduced=0, invalid=0)
        self.table.setRowCount(len(self.images))
        for row, image in enumerate(self.images):
            try:
                if image.is_missing:
                    raise ValueError('原图缺失，请先重新定位')
                plan = batch_size_plan(image.width, image.height, self.factor())
                counts['four' if plan['factor'] == 4 else 'two'] += 1
                counts['reduced'] += plan['reduce_input']
                sizes = plan['size_plan']
                dimension = lambda key: ' × '.join(map(str, sizes[key]))
                cells = [image.file_name, dimension('source'), dimension('input'), f"{plan['factor']:g} 倍",
                         dimension('output'), plan['adjustment_reason'] or '使用首选倍数']
            except (ValueError, TypeError) as exc:
                counts['invalid'] += 1
                cells = [image.file_name, '—', '—', '—', '—', str(exc)]
            for col, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setToolTip(image.file_path if col == 0 else value)
                self.table.setItem(row, col, item)
        self.table.resizeRowsToContents()
        self.summary.setText(f"4 倍：{counts['four']} 张 · 2 倍：{counts['two']} 张"
                             f"（其中先缩小输入：{counts['reduced']} 张） · 图片缺失或尺寸无效：{counts['invalid']} 张")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(counts['invalid'] < len(self.images))


class UpscaleTasks(QDialog):
    def __init__(self, controller):
        super().__init__(controller.w)
        self.c = controller
        self.setWindowTitle('AI 放大 · 任务与结果')
        self.resize(1050, 650)
        layout = QVBoxLayout(self)
        row = WrappingLayout()
        self.filter = QComboBox()
        self.filter.addItems(['待确认结果', '排队与运行', '失败与中断', '全部历史'])
        self.filter.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.filter)
        self.batch_filter = QComboBox()
        self.batch_filter.setAccessibleName('放大批次')
        self.batch_filter.addItem('全部批次', None)
        self.batch_filter.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.batch_filter)
        self.review = QPushButton('审核所选批次')
        self.review.clicked.connect(self.review_batch)
        row.addWidget(self.review)
        self.add = QPushButton('放大图库中所选图片…')
        self.add.clicked.connect(self.enqueue)
        row.addWidget(self.add)
        self.pause = QPushButton()
        self.pause.clicked.connect(self.pause_queue)
        row.addWidget(self.pause)
        storage = QPushButton('存储占用与清理…')
        storage.clicked.connect(self.storage)
        row.addWidget(storage)
        layout.addLayout(row)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['图片', '状态', '倍数', '时间', '批次', '说明'])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.actions)
        self.table.cellDoubleClicked.connect(lambda *_: self.open_result())
        layout.addWidget(self.table, 1)
        actions = WrappingLayout()
        self.buttons = {}
        for key, label, callback in [('open', '打开图片 / 对比', self.open_result),
                ('up', '排队上移', lambda: self.move(-1)), ('down', '排队下移', lambda: self.move(1)),
                ('cancel', '取消任务', self.cancel), ('retry', '重试失败任务', self.retry),
                ('resume', '继续核对原任务', self.resume), ('diagnostic', '复制诊断信息', self.diagnostic)]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            self.buttons[key] = button
            actions.addWidget(button)
        layout.addLayout(actions)
        self.message = QLabel('每次多选提交为一个批次。完成后会提醒；点击「审核所选批次」逐张决定采用或放弃。')
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.message)
        self.refresh()

    def selected(self):
        item = self.table.item(self.table.currentRow(), 0)
        return self.c.repo.get(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def refresh(self, *_):
        previous = self.selected()
        selected_batch = self.batch_filter.currentData()
        with QSignalBlocker(self.batch_filter):
            self.batch_filter.clear()
            self.batch_filter.addItem('全部批次', None)
            for batch in self.c.repo.batches():
                stamp = datetime.fromisoformat(batch['created_at']).astimezone().strftime('%m-%d %H:%M')
                self.batch_filter.addItem(f"{stamp} · {batch['total']} 张 · 待审 {batch['ready']} · {batch['id'][:4]}", batch['id'])
            self.batch_filter.setCurrentIndex(max(0, self.batch_filter.findData(selected_batch)))
        jobs = self.c.repo.all()
        waiting = self.c.repo.waiting()
        if selected_batch:
            jobs = [j for j in jobs if j.get('batch_id') == selected_batch]
            waiting = [j for j in waiting if j.get('batch_id') == selected_batch]
        counts = {state: sum(j['state'] == state for j in jobs) for state in ('ready', 'waiting', 'failed')}
        self.summary.setText(f"待确认 {counts['ready']} · 排队 {counts['waiting']} · 失败 {counts['failed']} · 共 {len(jobs)} 次")
        paused = self.c.w.store.get_setting('upscale.queue_paused', '0') == '1'
        self.pause.setText('继续队列' if paused else '暂停队列')
        selected_filter = self.filter.currentIndex()
        if selected_filter == 1:
            jobs = [j for j in jobs if j['state'] in ACTIVE_STATES | {'paused'}] + waiting
        elif selected_filter == 0:
            jobs = [j for j in jobs if j['state'] == 'ready']
        elif selected_filter == 2:
            jobs = [j for j in jobs if j['state'] in {'failed', 'missing', 'paused', 'offline'}]
        with QSignalBlocker(self.table):
            self.table.clearSelection()
            self.table.setCurrentCell(-1, -1)
            self.table.setRowCount(len(jobs))
            for row, job in enumerate(jobs):
                name = job.get('file_name') or Path(job.get('source_path', '')).name or str(job['image_id'])
                state = LABELS.get(job['state'], job['state'])
                if job.get('purged_at'):
                    state += ' · 文件已清理'
                note = '\n'.join(filter(None, [job.get('error') or (job.get('stage', '') if job['state'] in ACTIVE_STATES else ''),
                                               job.get('adjustment_reason')]))
                factor = f"{job.get('factor', 0):g}×"
                if job.get('requested_factor', job.get('factor')) != job.get('factor'):
                    factor = f"{job['requested_factor']:g}× → {factor}"
                batch = (f"{job['batch_id'][:4]} · 第 {job['batch_index']+1} 张" if job.get('batch_id') else '单张 / 旧任务')
                stamp = datetime.fromisoformat(job['created_at']).astimezone().strftime('%Y-%m-%d %H:%M:%S')
                for col, text in enumerate([name, state, factor, stamp, batch, note]):
                    item = QTableWidgetItem(text)
                    item.setToolTip(job.get('source_path', '') if col == 0 else text)
                    item.setData(Qt.ItemDataRole.UserRole, job['id'])
                    self.table.setItem(row, col, item)
                if previous and previous['id'] == job['id']:
                    self.table.selectRow(row)
        self.actions()

    def actions(self):
        job = self.selected()
        idle = not self.c.working and not self.c.w._database_maintenance_active
        for key, button in self.buttons.items():
            enabled = bool(job)
            if key in {'up', 'down'}:
                enabled = bool(job and job['state'] == 'waiting' and idle)
            if key == 'cancel':
                enabled = bool(job and job['state'] in ACTIVE_STATES | {'waiting'} and idle)
            if key == 'retry':
                enabled = bool(job and job['state'] in {'failed', 'cancelled'} and idle)
            if key == 'resume':
                enabled = bool(job and job['state'] == 'paused' and idle and not self.c.repo.active())
            button.setEnabled(enabled)
        self.add.setEnabled(idle)
        self.review.setEnabled(bool(self.batch_filter.currentData() or (job and job.get('batch_id'))))

    def review_batch(self):
        job = self.selected()
        batch_id = self.batch_filter.currentData() or (job.get('batch_id') if job else None)
        if batch_id:
            self.c.open_batch(batch_id)

    def open_result(self):
        job = self.selected()
        if job:
            self.c.open_job(job['id'])

    def move(self, delta):
        if job := self.selected():
            self.c.repo.move(job['id'], delta)
            self.refresh()

    def cancel(self):
        if job := self.selected():
            self.c.run(lambda: self.c.repo.cancel(job['id']), self.c.updated)

    def retry(self):
        if job := self.selected():
            self.c.run(lambda: self.c.repo.retry(job['id']), self.c.updated)

    def resume(self):
        if job := self.selected():
            self.c.run(lambda: self.c.repo.resume(job['id']), self.c.updated)

    def diagnostic(self):
        if job := self.selected():
            data = {key: job.get(key) for key in ('id', 'source_path', 'state', 'stage', 'factor', 'requested_factor',
                                                'reduce_input', 'size_plan', 'adjustment_reason', 'created_at', 'error', 'diagnostic')}
            QApplication.clipboard().setText(json.dumps(data, ensure_ascii=False, indent=2))
            self.message.setText('诊断信息已复制，可直接用于排查。')

    def pause_queue(self):
        paused = self.c.w.store.get_setting('upscale.queue_paused', '0') == '1'
        self.c.w.store.set_setting('upscale.queue_paused', '0' if paused else '1')
        self.refresh()
        self.message.setText('队列已暂停；已提交的任务继续完成。' if not paused else '队列已恢复。')

    def enqueue(self, images=None):
        if images is None or isinstance(images, bool):
            images = self.c.w.grid_view.selected_images()
        if not images:
            self.message.setText('请先在图片墙选择需要放大的图片，可多选。')
            return
        dialog = BatchUpscaleDialog(images, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        ids, factor = [i.id for i in images], dialog.factor()
        url = self.c.w.store.get_setting('comfy.url', 'http://127.0.0.1:8188')
        names = {i.id: i.file_name for i in images}
        def done(result):
            accepted, skipped = result
            self.filter.setCurrentIndex(1)
            self.refresh()
            if accepted:
                self.batch_filter.setCurrentIndex(self.batch_filter.findData(accepted[0]['batch_id']))
            four = sum(j['factor'] == 4 for j in accepted)
            reduced = sum(j['reduce_input'] for j in accepted)
            self.message.setText(f'已加入 {len(accepted)} 张：4 倍 {four} 张，2 倍 {len(accepted)-four} 张'
                                 f'（其中先缩小输入 {reduced} 张）。未入队 {len(skipped)} 张。' +
                                 ('\n' + '\n'.join(f"{names.get(i, i)}：{reason}" for i, reason in skipped) if skipped else ''))
        self.c.run(lambda: self.c.repo.enqueue(ids, factor, url), done)

    def storage(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('AI 放大 · 存储占用与清理')
        layout = QVBoxLayout(dialog)
        days = QSpinBox()
        days.setRange(0, 3650)
        days.setValue(30)
        days.setPrefix('保留最近 ')
        days.setSuffix(' 天；仅清理已丢弃 / 取消 / 失败项')
        layout.addWidget(days)
        label = QLabel('输入副本、候选和原图恢复备份分别统计。\n运行中的任务、待确认结果和原图恢复备份始终保留。')
        label.setWordWrap(True)
        layout.addWidget(label)
        files = QTableWidget(0, 3)
        files.setHorizontalHeaderLabels(['将清理的任务', '临时文件数', '占用'])
        files.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        files.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(files)
        scan = QPushButton('计算占用与可清理空间')
        clean = QPushButton('清理列出的文件…')
        clean.setEnabled(False)
        layout.addWidget(scan)
        layout.addWidget(clean)
        report = {}
        def show(value):
            report.clear()
            report.update(value)
            t = value['totals']
            label.setText(f"输入副本 {t['sources']/2**30:.2f} GiB · 候选 {t['candidates']/2**30:.2f} GiB\n"
                          f"原图恢复备份 {t['replacements']/2**30:.2f} GiB（保留）\n"
                          f"可清理 {len(value['eligible'])} 项，{value['bytes']/2**20:.1f} MiB。清理后不能撤销这些项的丢弃。")
            files.setRowCount(len(value['eligible']))
            for row, entry in enumerate(value['eligible']):
                job = self.c.repo.get(entry['id'])
                for col, text in enumerate([job.get('file_name') or Path(job.get('source_path', '')).name or job['id'],
                                            str(len(entry['files'])), f"{entry['bytes']/2**20:.1f} MiB"]):
                    item = QTableWidgetItem(text)
                    item.setToolTip('\n'.join(entry['files']))
                    files.setItem(row, col, item)
            clean.setEnabled(bool(value['eligible']))
        def calculate():
            retention = days.value()
            clean.setEnabled(False)
            self.c.run(lambda: self.c.repo.storage_report(retention), show)
        def purge():
            if QMessageBox.question(dialog, '确认清理', f"清理这 {len(report['eligible'])} 项临时文件？\n这些项的丢弃将不能撤销；原图恢复备份保留。") != QMessageBox.StandardButton.Yes:
                return
            snapshot = dict(report)
            clean.setEnabled(False)
            self.c.run(lambda: self.c.repo.cleanup(snapshot), lambda count: label.setText(f'已释放 {count/2**20:.1f} MiB。原图及恢复备份保留。'))
        scan.clicked.connect(calculate)
        clean.clicked.connect(purge)
        days.valueChanged.connect(lambda _: clean.setEnabled(False))
        dialog.resize(720, 460)
        dialog.exec()
