"""Migration wizard with bounded UI updates and cancellable file copying."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading
import time

from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QTabWidget, QVBoxLayout, QWidget)

from eidory.core.workspace_package import (MigrationCancelled, database_counts, export_workspace,
    inspect_package, prepare_export, restore_workspace)
from eidory.core.workspace_selection import selection, validate_workspace


def _size(value):
    return f'{value / 1024**3:.2f} GB' if value >= 1024**3 else f'{value / 1024**2:.1f} MB'


class _MigrationThread(QThread):
    progressed = Signal(int, str)

    def __init__(self, task, parent):
        super().__init__(parent)
        self.task = task
        self.cancelled = threading.Event()
        self.result = None
        self.error = None
        self.last_update = 0.0

    def report(self, done, total, label):
        now = time.monotonic()
        if now - self.last_update >= .1 or done == total:
            self.last_update = now
            self.progressed.emit(int(done*1000/max(1, total)), f'{label}\n已处理 {_size(done)} / {_size(total)}')

    def run(self):
        try:
            self.result = self.task(self.report, self.cancelled.is_set)
        except Exception as exc:
            self.error = exc


class MigrationProgress(QDialog):
    def __init__(self, parent, title, task):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.resize(560, 200)
        layout = QVBoxLayout(self)
        self.label = QLabel('正在检查文件与工作区…')
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        layout.addWidget(self.bar)
        self.cancel_button = QPushButton('暂停')
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        self.worker = _MigrationThread(task, self)
        self.worker.progressed.connect(self.update_progress)
        self.worker.finished.connect(self.accept)

    def update_progress(self, value, label):
        self.bar.setRange(0, 1000)
        self.bar.setValue(value)
        self.label.setText(label)

    def reject(self):
        self.worker.cancelled.set()
        self.cancel_button.setEnabled(False)
        self.label.setText('正在安全暂停；已复制并校验的文件会保留…')

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.reject()
            event.ignore()
        else:
            event.accept()

    def run(self):
        self.worker.start()
        self.exec()
        self.worker.wait()
        if self.worker.error:
            raise self.worker.error
        return self.worker.result


class WorkspaceMigrationDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.w = window
        self.setWindowTitle('工作区迁移 · 换电脑与回退')
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.resize(790, 610)
        layout = QVBoxLayout(self)
        self.intro = QLabel('迁移图片、已有向量、AI 标签、收藏、标注、项目和待审核结果。\n'
                            '恢复到独立工作区，核对后切换；旧工作区保留，可随时切回。')
        self.intro.setWordWrap(True)
        layout.addWidget(self.intro)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._build_export()
        self._build_restore()
        self._build_switch()
        close = QPushButton('关闭')
        close.clicked.connect(self.accept)
        layout.addWidget(close)

    def _path_row(self, label, default='', *, callback=None):
        row = QHBoxLayout()
        field = QLineEdit(str(default))
        field.setAccessibleName(label)
        button = QPushButton('选择…')
        def browse():
            selected = QFileDialog.getExistingDirectory(self, label, field.text() or str(Path.home()))
            if selected:
                field.setText(selected)
                if callback:
                    callback()
        button.clicked.connect(browse)
        row.addWidget(field, 1)
        row.addWidget(button)
        return row, field

    def _build_export(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel('旧电脑：制作迁移包'))
        self.export_mode = QComboBox()
        self.export_mode.addItem('完整迁移包：图片 + 工作区（推荐）', True)
        self.export_mode.addItem('关联现有图片：工作区 + 图片校验清单', False)
        layout.addWidget(self.export_mode)
        self.include_model = QCheckBox('携带当前检索模型，供新电脑离线语义搜索')
        self.include_model.setChecked(True)
        layout.addWidget(self.include_model)
        self.include_secrets = QCheckBox('同时携带 API Key（以明文保存在迁移包中）')
        layout.addWidget(self.include_secrets)
        help_text = QLabel('完整包只复制图库引用的原图，保留文件夹层级。关联模式适合已复制图片或使用外置磁盘。\n'
                           '两种模式都会携带缩略图、待审核结果和回收备份。外部 AI 服务与其模型需在新电脑另行配置。')
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        default = Path.home()/'Desktop'/('Eidory迁移-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
        row, self.export_path = self._path_row('迁移包目录（新目录、空目录或上次未完成的目录）', default)
        layout.addLayout(row)
        self.export_button = QPushButton('检查并导出 / 继续导出')
        self.export_button.setProperty('variant', 'primary')
        self.export_button.clicked.connect(self.export)
        layout.addWidget(self.export_button)
        self.export_status = QLabel('检查后会显示总大小、缺失图片和已有任务；确认后开始复制。')
        self.export_status.setWordWrap(True)
        self.export_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.export_status)
        layout.addStretch()
        self.tabs.addTab(page, '导出迁移包')

    def _build_restore(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel('新电脑：恢复工作环境'))
        row, self.package_path = self._path_row('选择迁移包目录')
        layout.addLayout(row)
        inspect = QPushButton('读取迁移包信息')
        inspect.clicked.connect(self.inspect)
        layout.addWidget(inspect)
        self.restore_summary = QLabel('先读取迁移包，再选择恢复位置。')
        self.restore_summary.setWordWrap(True)
        layout.addWidget(self.restore_summary)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.mapping_widget = QWidget()
        self.mapping_form = QFormLayout(self.mapping_widget)
        scroll.setWidget(self.mapping_widget)
        layout.addWidget(scroll, 1)
        row, self.restore_path = self._path_row('恢复位置的父文件夹', Path.home()/'Pictures')
        layout.addLayout(row)
        self.restore_name = QLineEdit('Eidory工作区-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.restore_name.setAccessibleName('新工作区文件夹名称')
        layout.addWidget(self.restore_name)
        self.restore_button = QPushButton('校验并恢复 / 继续恢复')
        self.restore_button.setProperty('variant', 'primary')
        self.restore_button.clicked.connect(self.restore)
        self.restore_button.setEnabled(False)
        layout.addWidget(self.restore_button)
        self.manifest = None
        self.mapping_fields = {}
        self.package_path.textChanged.connect(self.invalidate_package)
        self.tabs.addTab(page, '从迁移包恢复')

    def invalidate_package(self):
        self.manifest = None
        self.restore_button.setEnabled(False)

    def _build_switch(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        info = QLabel('当前工作区\n'+str(self.w.paths.data_dir)+'\n\n切换会保存当前工作并重新打开软件。各工作区独立，修改不会相互同步。')
        info.setWordWrap(True)
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(info)
        self.workspaces = QComboBox()
        self.workspaces.setAccessibleName('可回退的工作区')
        for path in dict.fromkeys([str(self.w.paths.data_dir), *selection().get('history', [])]):
            self.workspaces.addItem(path, path)
        layout.addWidget(self.workspaces)
        switch = QPushButton('切换到所选工作区并重启')
        switch.clicked.connect(lambda: self.switch(self.workspaces.currentData()))
        layout.addWidget(switch)
        choose = QPushButton('选择其他已有工作区…')
        def browse():
            path = QFileDialog.getExistingDirectory(self, '选择工作区（或其中的 data 文件夹）')
            if path:
                path = Path(path)
                self.switch(path/'data' if (path/'data/eidory.sqlite3').is_file() else path)
        choose.clicked.connect(browse)
        layout.addWidget(choose)
        layout.addStretch()
        self.tabs.addTab(page, '切换 / 回退')

    def _task(self, title, task):
        return MigrationProgress(self, title, task).run()

    def _error(self, exc):
        if isinstance(exc, MigrationCancelled):
            QMessageBox.information(self, '迁移已暂停', str(exc))
        else:
            QMessageBox.warning(self, '迁移未完成', str(exc)+'\n当前工作区未被替换。')

    def _flush(self):
        for preview in list(getattr(self.w, '_preview_dialogs', ())):
            if not preview.annotation_controls.flush(interactive=True):
                return False
        self.w._save_pending_note()
        self.w._save_current_board_layout_if_needed()
        if self.w.creative_flow._inline:
            self.w.creative_flow._inline.finish()
        self.w.creative_workspace.save_context()
        self.w.creative_workspace.save_search()
        return True

    def export(self):
        if not self.export_path.text().strip() or not self._flush():
            return
        options = dict(include_media=self.export_mode.currentData(), include_model=self.include_model.isChecked(),
                       include_secrets=self.include_secrets.isChecked())
        target = Path(self.export_path.text()).expanduser()
        polling = self.w.poll_timer.isActive()
        try:
            with self.w._database_maintenance('工作区导出', restart_index_workers=True):
                self.w.poll_timer.stop()
                self.w._backup_database_to_default_location()
                manifest = self._task('检查迁移内容', lambda p,c: prepare_export(self.w.paths, target, **options))
                amount = sum(f['size'] for f in manifest['files'] if f['included']) + (target/'database.sqlite3').stat().st_size
                c = manifest['counts']
                message = (f'快照时间：{manifest["created_at"][:19]}\n图片 {c["images"]:,} 张；已有向量 {c["ready_vectors"]:,} 条；项目 {c["creative_projects"]} 个。\n'
                           f'迁移包约 {_size(amount)}。缺失原图 {manifest["missing"]} 张，保留其记录。\n'
                           f'正在跟踪的 AI 任务 {manifest["active_jobs"]} 个，恢复后不会自动继续。\n\n'
                           f'导出到：{target}\n开始复制并校验？')
                if manifest.get('existing_relation_issues'):
                    message += f'\n\n源数据库已有 {len(manifest["existing_relation_issues"])} 条历史关联异常，将原样保留并记入恢复报告。'
                if QMessageBox.question(self, '确认迁移内容', message) != QMessageBox.StandardButton.Yes:
                    return
                result = self._task('正在导出迁移包', lambda p,c: export_workspace(self.w.paths, target, progress=p, cancel=c, **options))
                self.export_status.setText(f'导出完成：{target}\n复制整个文件夹到新电脑，然后在“从迁移包恢复”中打开。\n缺失原图 {result["missing"]} 张。')
        except Exception as exc:
            self._error(exc)
        finally:
            if polling:
                self.w.poll_timer.start()

    def inspect(self):
        try:
            self.manifest = self._task('读取迁移包', lambda p,c: inspect_package(self.package_path.text()))
            while self.mapping_form.rowCount():
                self.mapping_form.removeRow(0)
            self.mapping_fields.clear()
            m = self.manifest
            c = m['counts']
            self.restore_summary.setText(f'导出时间：{m["created_at"][:19]}\n'
                f'图片 {c["images"]:,} 张 · 向量 {c["ready_vectors"]:,} 条 · 项目 {c["creative_projects"]} 个 · 缺失 {m["missing"]} 张\n'
                + ('含原图；自动关联新位置。' if m['include_media'] else '请为下面的原图库选择新位置；将逐张校验内容。')
                + (' 已携带检索模型。' if m['include_model'] else ' 未携带检索模型，首次语义搜索需要下载模型。'))
            if m.get('existing_relation_issues'):
                self.restore_summary.setText(self.restore_summary.text() + f'\n源数据库已有 {len(m["existing_relation_issues"])} 条历史关联异常；原样保留，不新增异常。')
            if not m['include_media']:
                for root in m['roots']:
                    row, field = self._path_row(root['source'], root['source'] if Path(root['source']).is_dir() else '')
                    label = QLabel(root['source'])
                    label.setWordWrap(True)
                    self.mapping_form.addRow(label, row)
                    self.mapping_fields[root['id']] = field
            self.restore_button.setEnabled(True)
        except Exception as exc:
            self.invalidate_package()
            self._error(exc)

    def restore(self):
        if not self.manifest:
            return
        name = self.restore_name.text().strip()
        if not name or name in ('.', '..') or Path(name).name != name:
            self._error(ValueError('请输入单个文件夹名称'))
            return
        target = Path(self.restore_path.text()).expanduser()/name
        try:
            mapping = {k:v.text() for k,v in self.mapping_fields.items()}
            data, report = self._task('正在校验并恢复工作区', lambda p,c: restore_workspace(
                self.package_path.text(), target, root_mapping=mapping, progress=p, cancel=c))
            self.workspaces.addItem(str(data), str(data))
            self.workspaces.setCurrentIndex(self.workspaces.count()-1)
            self.restore_summary.setText(f'恢复完成：{target}\n已有向量和标签已保留；缺失原图 {report["missing"]} 张。旧工作区保留。')
            box = QMessageBox(self)
            box.setWindowTitle('工作区已恢复')
            box.setText('文件内容校验通过，数据库没有新增关联错误。\n切换后保留已有向量、标签、项目及待审核结果。\n'
                        '自动扫描与 AI 排队已暂停。请先核对外部 AI 服务设置；API Key '+('已携带。' if report['api_keys_included'] else '需重新填写。'))
            if report.get('existing_relation_issues'):
                box.setText(box.text() + f'\n\n源数据库原有 {len(report["existing_relation_issues"])} 条历史关联异常，已保留并写入 migration-report.json。')
            now = box.addButton('切换并重启', QMessageBox.ButtonRole.AcceptRole)
            box.addButton('稍后切换', QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() == now:
                self.switch(data, confirmed=True)
        except Exception as exc:
            self._error(exc)

    def switch(self, data_dir, *, confirmed=False):
        try:
            target = validate_workspace(data_dir)
            if target == self.w.paths.data_dir.resolve():
                QMessageBox.information(self, '工作区', '已经在使用这个工作区。')
                return
            if not confirmed and QMessageBox.question(self, '切换工作区', f'保存当前工作并重启到：\n{target}\n\n当前工作区保留，可随时切回。') != QMessageBox.StandardButton.Yes:
                return
            if not self._flush():
                return
            app = QApplication.instance()
            app._eidory_next_workspace = str(target)
            self.accept()
            # MainWindow owns graceful shutdown; main() commits the selector
            # only after all old workers have exited.
            self.w.close()
            if not self.w._closing:
                app._eidory_next_workspace = None
        except Exception as exc:
            self._error(exc)
