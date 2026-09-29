"""One search-stage entry to local generation, with durable project-scoped results."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from weakref import ref

from PySide6.QtCore import QObject, QTimer, Qt, Signal, QSize, QUrl
from PySide6.QtGui import QIcon, QImageReader, QPixmap
from PySide6.QtWebSockets import QWebSocket
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from eidory.core.comfy_reference import (
    ACTIVE_STATES, STATE_LABELS, ComfyClient, GeneratedReferences, build_prompt, local_url, prepare_template, connection_message, sampling_description,
)
from eidory.core.comfy_workflows import ReferenceWorkflows


def text_label(text=''):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    return label


class GenerationPanel(QDialog):
    def __init__(self, controller, node):
        super().__init__(controller.w)
        self.c = controller
        self.node_id = node.id
        self.project_id = node.project_id
        self.node_title = node.title
        self.setWindowTitle('生成参考 · ' + node.title)
        self.setModal(False)
        self.resize(850, 760)
        self.setMinimumSize(600, 600)
        outer = QVBoxLayout(self)
        project = controller.w.store.get_creative_project(node.project_id)
        self.heading = text_label(project.title + ' / ' + node.title)
        self.heading.setProperty('role', 'flowTitle')
        outer.addWidget(self.heading)
        workflow_row = QHBoxLayout()
        workflow_row.addWidget(text_label('工作流'))
        self.workflow = QComboBox()
        self.workflow.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.workflow.setAccessibleName('生成参考工作流')
        self.history_template = None
        self.sync_workflows()
        self.workflow.currentIndexChanged.connect(self.change_workflow)
        workflow_row.addWidget(self.workflow, 1)
        outer.addLayout(workflow_row)
        self.prompt = QTextEdit()
        self.prompt.setAcceptRichText(False)
        self.prompt.setAccessibleName('生成参考的图片描述')
        self.prompt.setPlaceholderText('描述这次要生成的参考图片。')
        self.prompt.setPlainText(build_prompt(controller.w.store, node.id))
        self.prompt.setFixedHeight(140)
        outer.addWidget(self.prompt)
        row = QHBoxLayout()
        self.generate = QPushButton('生成 1 张')
        self.generate.setProperty('variant', 'primary')
        self.generate.clicked.connect(controller.generate)
        row.addWidget(self.generate)
        self.cancel = QPushButton('取消生成')
        self.cancel.clicked.connect(controller.cancel)
        row.addWidget(self.cancel)
        self.settings_toggle = QPushButton('生成设置 ▸')
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.toggled.connect(self.toggle_settings)
        row.addWidget(self.settings_toggle)
        row.addStretch()
        self.connection = text_label('本机 ComfyUI')
        row.addWidget(self.connection)
        outer.addLayout(row)
        self.settings = QWidget()
        form = QFormLayout(self.settings)
        form.setContentsMargins(0, 0, 0, 0)
        self.url = QLineEdit(controller.w.store.get_setting('comfy.url', 'http://127.0.0.1:8188'))
        self.url.setAccessibleName('本机 ComfyUI 地址')
        form.addRow('服务地址', self.url)
        binding = QHBoxLayout()
        self.import_button = QPushButton('导入自定义 API 工作流…')
        self.import_button.clicked.connect(controller.import_workflow)
        binding.addWidget(self.import_button)
        self.check = QPushButton('检查连接')
        self.check.clicked.connect(controller.check_connection)
        binding.addWidget(self.check)
        form.addRow(binding)
        dimensions = QHBoxLayout()
        self.width, self.height = QSpinBox(), QSpinBox()
        for spin in (self.width, self.height):
            spin.setRange(256, 4096)
            spin.setSingleStep(64)
        self.width.setAccessibleName('生成图片宽度')
        self.height.setAccessibleName('生成图片高度')
        dimensions.addWidget(self.width)
        dimensions.addWidget(text_label('×'))
        dimensions.addWidget(self.height)
        dimensions.addStretch()
        form.addRow('尺寸', dimensions)
        self.sampling = text_label('保留工作流的模型、步数和采样参数，每次使用新种子。')
        form.addRow(self.sampling)
        self.destination = QPushButton('设置采用后的保存位置…')
        self.destination.clicked.connect(lambda: controller.choose_destination(self.project_id))
        form.addRow('正式保存', self.destination)
        outer.addWidget(self.settings)
        self.settings.hide()
        self.status = text_label('候选不会进入普通图库；采用时保存并加入原节点。')
        outer.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(12)
        self.progress.setTextVisible(False)
        self.progress.hide()
        outer.addWidget(self.progress)
        heading = QHBoxLayout()
        heading.addWidget(text_label('生成候选'), 1)
        self.show_all = QComboBox()
        self.show_all.addItems(['当前节点', '本项目全部节点'])
        self.show_all.setAccessibleName('生成候选范围')
        self.show_all.currentIndexChanged.connect(lambda: self.refresh_candidates())
        heading.addWidget(self.show_all)
        outer.addLayout(heading)
        self.candidates = QListWidget()
        self.candidates.setAccessibleName('生成的参考图片')
        self.candidates.setViewMode(QListWidget.ViewMode.IconMode)
        self.candidates.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.candidates.setMovement(QListWidget.Movement.Static)
        self.candidates.setIconSize(QSize(185, 170))
        self.candidates.setGridSize(QSize(225, 218))
        self.candidates.setSpacing(6)
        self.candidates.currentItemChanged.connect(self.selection_changed)
        self.candidates.itemDoubleClicked.connect(lambda _: controller.preview())
        outer.addWidget(self.candidates, 1)
        self.details = text_label('生成结果会保留在这里，关闭窗口不会丢失。')
        outer.addWidget(self.details)
        actions = QHBoxLayout()
        self.adopt = QPushButton('采用并保存')
        self.adopt.setProperty('variant', 'primary')
        self.adopt.clicked.connect(controller.adopt)
        actions.addWidget(self.adopt)
        self.preview_button = QPushButton('放大查看')
        self.preview_button.clicked.connect(controller.preview)
        actions.addWidget(self.preview_button)
        self.regenerate = QPushButton('再生成')
        self.regenerate.clicked.connect(controller.regenerate)
        actions.addWidget(self.regenerate)
        self.discard = QPushButton('丢弃')
        self.discard.clicked.connect(controller.discard)
        actions.addWidget(self.discard)
        actions.addStretch()
        done = QPushButton('返回找参考')
        done.clicked.connect(self.close)
        actions.addWidget(done)
        outer.addLayout(actions)
        # Enter while choosing a workflow/editing dimensions must not submit
        # an expensive generation or adopt a candidate via QDialog defaults.
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)
        self._list_key = None
        self.apply_template()
        self.settings_toggle.setChecked(not bool(controller.template()))
        self.refresh_candidates()

    def toggle_settings(self, checked):
        self.settings.setVisible(checked)
        self.settings_toggle.setText('生成设置 ▾' if checked else '生成设置 ▸')

    def sync_workflows(self):
        self.workflow.blockSignals(True)
        self.workflow.clear()
        for key, name in self.c.workflows.options():
            self.workflow.addItem(name, key)
        self.workflow.setCurrentIndex(self.workflow.findData(self.c.workflows.selected()))
        self.workflow.blockSignals(False)
        self.history_template = None

    def change_workflow(self, _index):
        if self.workflow.currentData() != 'history':
            try:
                self.c.workflows.select(self.workflow.currentData())
            except (ValueError, KeyError, TypeError) as exc:
                self.status.setText('工作流不可用：' + str(exc))
                self.generate.setEnabled(False)
                return
        self.apply_template()
        self.connection.setText('本机 ComfyUI')
        self.status.setText('已切换工作流，尺寸使用该工作流的默认值；图片描述保留。')
        self.c.refresh()

    def apply_template(self):
        template = self.c.template()
        if template:
            latent = template['graph'][template['latent_node']]['inputs']
            self.width.setValue(latent['width'])
            self.height.setValue(latent['height'])
            for spin in (self.width, self.height):
                spin.setSingleStep(template.get('size_multiple', 8))
            self.sampling.setText(sampling_description(template))

    def selected_job(self):
        item = self.candidates.currentItem()
        return self.c.repo.get(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def refresh_candidates(self, select_id=None):
        jobs = self.c.repo.all(self.project_id)
        jobs = [j for j in jobs if self.show_all.currentIndex() or j['node_id'] == self.node_id]
        key = [(j['id'], j['state'], j.get('thumbnail_path')) for j in jobs]
        if key == self._list_key:
            self.selection_changed()
            return
        selected = self.selected_job()
        selected_id = select_id or (selected['id'] if selected else None)
        self._list_key = key
        self.candidates.blockSignals(True)
        self.candidates.clear()
        for job in jobs:
            created = datetime.fromisoformat(job['created_at']).astimezone().strftime('%m-%d %H:%M')
            item = QListWidgetItem(QIcon(job.get('thumbnail_path', '')), f"{job['node_title']} · {STATE_LABELS[job['state']]}\n{created}")
            item.setData(Qt.ItemDataRole.UserRole, job['id'])
            item.setToolTip(job['prompt'])
            self.candidates.addItem(item)
            if job['id'] == selected_id:
                self.candidates.setCurrentItem(item)
        if self.candidates.currentRow() < 0 and self.candidates.count():
            self.candidates.setCurrentRow(0)
        self.candidates.blockSignals(False)
        self.selection_changed()

    def selection_changed(self, *_):
        job = self.selected_job()
        busy = self.c.working
        has_image = bool(job and job.get('file_path') and Path(job['file_path']).is_file())
        self.adopt.setEnabled(bool(job and job['state'] == 'ready' and has_image and not busy))
        self.adopt.setText('采用并保存到「' + job['node_title'] + '」' if job else '采用并保存')
        self.preview_button.setEnabled(has_image)
        self.regenerate.setEnabled(bool(job and not self.c.active() and not busy))
        self.discard.setEnabled(bool(job and job['state'] in {'ready', 'discarded'} and not busy))
        self.discard.setText('撤销丢弃' if job and job['state'] == 'discarded' else '丢弃')
        if job:
            workflow = job.get('workflow_name')
            self.details.setText(job.get('error') or (workflow + ' · ' if workflow else '') + f"{STATE_LABELS[job['state']]} · {job.get('width', '—')} × {job.get('height', '—')} · 种子 {job['seed']}\n" +
                                 ('保存位置：' + job['file_path'] if job['state'] == 'adopted' else '临时生成／' + job['project_title']))
        else:
            self.details.setText('这里还没有生成结果。生成后可以预览、采用或丢弃。')


class ReferenceGeneration(QObject):
    completed = Signal(object, object)

    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self.repo = GeneratedReferences(window.store, window.paths)
        self.workflows = ReferenceWorkflows(window.store)
        self.panel = None
        self.working = False
        self.held_workers = []
        self.completed.connect(self.finish_work)
        self.entry = QPushButton('生成参考')
        self.entry.setToolTip('使用本机 ComfyUI 生成候选参考，采用后加入当前节点')
        self.entry.clicked.connect(self.open)
        window.creative_workspace.navigation.layout().insertWidget(4, self.entry)
        window.center_result_stack.currentChanged.connect(self.refresh)
        window.creative_node_tree.currentItemChanged.connect(self.refresh)
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.poll)
        self.timer.start()
        self.socket = QWebSocket(parent=self)
        self.socket.textMessageReceived.connect(self.on_progress)
        self.socket_job = None
        self.progress_values = {}
        self.refresh()

    @property
    def w(self):
        return self._window()

    def template(self):
        try:
            if self.panel and self.panel.workflow.currentData() == 'history':
                return self.panel.history_template
            profile = self.panel.workflow.currentData() if self.panel else None
            return self.workflows.template(profile)
        except (ValueError, KeyError, TypeError):
            return None

    def active(self):
        return self.repo.active()

    def refresh(self, *_):
        w = self.w
        self.entry.setVisible(w.center_result_stack.currentWidget() in (w.grid_view, w.creative_flow.search_empty))
        self.entry.setEnabled(bool(w._current_creative_node()) and not w._database_maintenance_active)
        job = self.active()
        self.entry.setText('生成参考 · 进行中' if job else '生成参考')
        if self.panel:
            upscaler = getattr(w, 'image_upscaler', None)
            upscale_busy = bool(upscaler and (upscaler.working or upscaler.repo.active()))
            self.panel.generate.setEnabled(not self.working and not job and not upscale_busy and bool(self.template()))
            self.panel.cancel.setEnabled(bool(job) and not self.working)
            self.panel.settings.setEnabled(not self.working and not job)
            self.panel.workflow.setEnabled(not self.working and not job)
            self.panel.refresh_candidates()
            if job:
                progress = self.progress_values.get(job['id'])
                suffix = f' {progress[0]}/{progress[1]}' if progress and job['state'] == 'running' else ''
                self.panel.status.setText(f"{job['project_title']} / {job['node_title']} · {job.get('error') or STATE_LABELS[job['state']]}{suffix}")
            self.panel.progress.setVisible(bool(job))
        self.hold_indexing(bool(job))

    def open(self):
        node = self.w._current_creative_node()
        if not node:
            return
        self.w.creative_workspace.save_context()
        self.w._save_pending_note()
        if self.panel and (self.panel.node_id != node.id):
            self.panel.close()
            self.panel.deleteLater()
            self.panel = None
        if not self.panel:
            self.panel = GenerationPanel(self, node)
        self.panel.show()
        self.panel.raise_()
        self.refresh()

    def run(self, operation, callback):
        if self.working or self.w._closing:
            return
        self.working = True
        self.refresh()
        def worker():
            try:
                result = operation()
            except Exception as exc:
                result = exc
            self.completed.emit(callback, result)
        if not self.w._start_background_task(worker, name='comfy-reference'):
            self.working = False
            self.refresh()

    def finish_work(self, callback, result):
        self.working = False
        if self.w._closing:
            return
        if isinstance(result, Exception):
            if self.panel:
                self.panel.status.setText(connection_message(result))
                if self.panel.connection.text() == '检查中…':
                    self.panel.connection.setText('未连接')
            self.w.statusBar().showMessage('生成参考：' + connection_message(result))
        else:
            callback(result)
        self.refresh()

    def import_workflow(self):
        path, _ = QFileDialog.getOpenFileName(self.panel, '选择 ComfyUI API 工作流', '', 'JSON (*.json)')
        if not path:
            return
        try:
            if Path(path).stat().st_size > 4 * 1024 * 1024:
                raise ValueError('工作流 JSON 过大。')
            self.workflows.import_custom(json.loads(Path(path).read_text(encoding='utf-8')), Path(path).name)
            self.panel.sync_workflows()
            self.panel.apply_template()
            self.panel.status.setText('工作流已绑定。模型和采样参数保留，输出作为临时候选接收。')
            self.refresh()
        except (ValueError, OSError) as exc:
            self.panel.status.setText(str(exc))

    def check_connection(self):
        try:
            url = local_url(self.panel.url.text())
            template = self.template()
            if not template:
                raise ValueError('请先选择 API 格式的文生图工作流。')
            self.w.store.set_setting('comfy.url', url)
        except ValueError as exc:
            self.panel.status.setText(str(exc))
            return
        self.panel.connection.setText('检查中…')
        def done(message):
            self.panel.connection.setText('已连接')
            self.panel.status.setText(message)
            self.w.statusBar().showMessage('生成参考：' + message, 8000)
        self.run(lambda: ComfyClient(url).validate(template), done)

    def generate(self, _checked=False, *, template=None):
        if self.active() or self.working:
            return
        try:
            template = template or self.template()
            if not template:
                raise ValueError('请先绑定 API 工作流。')
            url = local_url(self.panel.url.text())
            self.w.store.set_setting('comfy.url', url)
            job = self.repo.create(self.panel.node_id, self.panel.prompt.toPlainText(), template, url,
                                   width=self.panel.width.value(), height=self.panel.height.value())
        except ValueError as exc:
            self.panel.status.setText(str(exc))
            return
        self.connect_progress(job)
        self.panel.progress.setRange(0, 0)
        self.panel.refresh_candidates(select_id=job['id'])
        self.run(lambda: self.repo.submit(job['id']), lambda j: self.job_updated(j))

    def job_updated(self, job):
        if self.panel:
            self.panel.status.setText(job.get('error') or STATE_LABELS[job['state']])
            self.panel.refresh_candidates()
        if job['state'] == 'ready':
            self.w.statusBar().showMessage('参考已生成：' + job['node_title'] + '，待采用')

    def connect_progress(self, job):
        if self.socket_job != job['id'] or self.socket.state().name == 'UnconnectedState':
            self.socket.close()
            self.socket_job = job['id']
            self.socket.open(QUrl(job['url'].replace('http://', 'ws://', 1) + '/ws?clientId=' + job['client_id']))

    def on_progress(self, raw):
        try:
            event = json.loads(raw)
            job = self.active()
            data = event.get('data', {})
            if not job or data.get('prompt_id') != job['prompt_id'] or not self.panel:
                return
            if event.get('type') == 'progress':
                value, maximum = int(data['value']), int(data['max'])
                self.progress_values[job['id']] = (value, maximum)
                self.panel.progress.setRange(0, max(1, maximum))
                self.panel.progress.setValue(value)
                self.panel.status.setText(f"{job['node_title']} · 生成中 {value}/{maximum}")
        except (ValueError, KeyError, TypeError):
            pass

    def poll(self):
        if self.w._closing or self.working or self.w._database_maintenance_active:
            return
        job = self.active()
        if job:
            self.connect_progress(job)
            self.run(lambda: self.repo.reconcile(job['id']), self.job_updated)
        else:
            self.socket.close()
            self.hold_indexing(False)

    def cancel(self):
        job = self.active()
        if job:
            self.run(lambda: self.repo.cancel(job['id']), self.job_updated)

    def regenerate(self):
        job = self.panel.selected_job()
        if not job:
            return
        # Explicitly keep the result's original node, even in the project-wide list.
        node = self.w.store.get_creative_node(job['node_id'])
        if not node:
            self.panel.status.setText('原节点已删除，无法继续生成。')
            return
        if node.id != self.panel.node_id:
            self.panel.status.setText('请切换到「' + node.title + '」后继续生成，避免混入其他节点。')
            return
        self.panel.prompt.setPlainText(job['prompt'])
        template = prepare_template(job['workflow'])
        template.update(workflow_id=job.get('workflow_id', 'original'),
                        workflow_name=job.get('workflow_name') or '原结果工作流')
        # Reuse the immutable result snapshot, not today's profile/settings.
        self.panel.sync_workflows()
        self.panel.history_template = template
        self.panel.workflow.addItem(template['workflow_name'] + '（原结果参数）', 'history')
        self.panel.workflow.setCurrentIndex(self.panel.workflow.findData('history'))
        self.generate(template=template)

    def preview(self):
        job = self.panel.selected_job()
        if not job or not job.get('file_path'):
            return
        dialog = QDialog(self.panel)
        dialog.setWindowTitle('生成参考 · ' + job['node_title'])
        dialog.resize(880, 820)
        outer = QVBoxLayout(dialog)
        from eidory.ui.image_preview_dialog import PreviewImageView
        view = PreviewImageView(dialog)
        reader = QImageReader(job['file_path'])
        if not reader.canRead():
            self.panel.status.setText('无法读取这张图片：' + reader.errorString())
            dialog.deleteLater()
            return
        size = reader.size()
        original = QSize(size)
        if max(size.width(), size.height()) > 2400:
            size.scale(QSize(2400, 2400), Qt.AspectRatioMode.KeepAspectRatio)
            reader.setScaledSize(size)
        view.set_pixmap(QPixmap.fromImage(reader.read()), original_width=original.width(), original_height=original.height(), fit_to_window=True, zoom_factor=1.0)
        outer.addWidget(view)
        row = QHBoxLayout()
        fit = QPushButton('适应窗口')
        fit.clicked.connect(lambda: view.fit_image_to_window())
        row.addWidget(fit)
        actual = QPushButton('实际大小')
        actual.clicked.connect(lambda: view.set_actual_size())
        row.addWidget(actual)
        row.addStretch()
        close = QPushButton('关闭')
        close.clicked.connect(dialog.accept)
        row.addWidget(close)
        outer.addLayout(row)
        dialog.exec()
        dialog.deleteLater()

    def choose_destination(self, project_id):
        project = self.w.store.get_creative_project(project_id)
        if not project:
            return None
        dialog = QDialog(self.panel)
        dialog.setWindowTitle('采用后的保存位置')
        dialog.resize(600, 190)
        layout = QVBoxLayout(dialog)
        layout.addWidget(text_label('保存到图库文件夹；此项目以后沿用这里。'))
        choice = QComboBox()
        choice.setAccessibleName('生成参考永久保存的图库文件夹')
        choice.addItem('AI生成／' + project.title, None)
        for col in self.w.store.list_collections():
            choice.addItem(self.w._collection_path_text(col.id), col.id)
        layout.addWidget(choice)
        location = text_label()
        def describe():
            cid = choice.currentData()
            path = self.w._collection_import_directory(cid) if cid else Path.home() / 'Pictures' / 'Eidory Imports' / 'AI生成' / self.w._safe_path_component(project.title)
            location.setText(str(path))
        choice.currentIndexChanged.connect(describe)
        describe()
        layout.addWidget(location)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText('使用此位置')
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        cid = choice.currentData()
        if cid is None:
            cid = self.w.store.ensure_collection_path(['AI生成', project.title])
        self.w.store.set_setting('comfy.destination.' + str(project_id), str(cid))
        return cid

    def adopt(self):
        job = self.panel.selected_job()
        if not job:
            return
        raw = self.w.store.get_setting('comfy.destination.' + str(job['project_id']))
        cid = int(raw) if raw and raw.isdigit() else None
        if cid not in {c.id for c in self.w.store.list_collections()}:
            cid = self.choose_destination(job['project_id'])
        if cid is None:
            return
        directory = self.w._collection_import_directory(cid)
        def adopted(result):
            w = self.w
            w._refresh_folders()
            w._refresh_collections()
            w._refresh_tags()
            w._refresh_file_watcher()
            node = w.store.get_creative_node(result['node_id'])
            if node:
                w._refresh_after_creative_node_images_changed(node)
            self.job_updated(result)
            self.panel.status.setText('已永久保存，并加入「' + result['node_title'] + '」。可返回看板整理。')
        self.run(lambda: self.repo.adopt(job['id'], cid, directory), adopted)

    def discard(self):
        job = self.panel.selected_job()
        if job:
            self.run(lambda: self.repo.discard(job['id'], undo=job['state'] == 'discarded'), self.job_updated)

    def hold_indexing(self, hold):
        upscaler = getattr(self.w, 'image_upscaler', None)
        hold = bool(hold or (upscaler and (upscaler.working or upscaler.repo.active())))
        for worker in (self.w.embedding_worker, self.w.ai_vision_worker, self.w.ai_character_vision_worker):
            if worker is not None and worker.is_alive():
                worker.set_generation_hold(hold)

    def close(self):
        self.timer.stop()
        self.socket.close()
        if self.panel:
            self.panel.close()
