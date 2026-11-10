"""One workspace entry for background work and image results."""
from pathlib import Path
from sqlite3 import Error as DatabaseError

from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import (QComboBox, QHeaderView, QHBoxLayout, QLabel, QPushButton,
                               QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from eidory.ui.preview_ai import LABELS
from eidory.core.comfy_reference import STATE_LABELS


class TaskCenter:
    def __init__(self, window):
        self.w = window
        chrome = window.workspace_chrome
        self.dialog = chrome._dialogs[3]
        self.dialog.setWindowTitle("后台任务")
        self.dialog.resize(1020, 730)
        layout = self.dialog.layout()
        index_page = layout.takeAt(0).widget()
        self.tabs = QTabWidget(self.dialog)
        layout.insertWidget(0, self.tabs, 1)
        self.tabs.addTab(index_page, "索引与识别")
        self.upscale_placeholder = QWidget()
        self.tabs.addTab(self.upscale_placeholder, "AI 放大")
        self.ai_page = QWidget()
        content = QVBoxLayout(self.ai_page)
        actions = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItems(["全部状态", "待审核", "运行中", "失败与中断"])
        self.kind.currentIndexChanged.connect(self.refresh_ai)
        actions.addWidget(self.kind)
        actions.addStretch()
        new = QPushButton("新建生图")
        new.clicked.connect(lambda: window.preview_ai.open(None, "generate", resume=False))
        actions.addWidget(new)
        content.addLayout(actions)
        self.ai_summary = QLabel()
        content.addWidget(self.ai_summary)
        self.ai_table = QTableWidget(0, 4)
        self.ai_table.setHorizontalHeaderLabels(["类型", "任务", "状态", "说明"])
        self.ai_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.ai_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.ai_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ai_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.ai_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.ai_table.cellDoubleClicked.connect(lambda *_: self.open_result())
        content.addWidget(self.ai_table, 1)
        self.review = QPushButton("查看所选任务")
        self.review.clicked.connect(self.open_result)
        self.discard = QPushButton("放弃结果")
        self.discard.setToolTip("从待审核移出，保留任务历史；可在「全部状态」里撤销。")
        self.discard.clicked.connect(self.discard_result)
        self.ai_table.itemSelectionChanged.connect(self.update_selection)
        decisions = QHBoxLayout()
        decisions.addWidget(self.review)
        decisions.addWidget(self.discard)
        content.addLayout(decisions)
        self.feedback = QLabel()
        self.feedback.setWordWrap(True)
        content.addWidget(self.feedback)
        self.update_selection()
        self.tabs.addTab(self.ai_page, "AI 生图与编辑")
        self.tabs.currentChanged.connect(self.tab_changed)
        self.counts = {}
        self.references = []
        window.image_upscaler.result_button.hide()
        window.preview_ai.entry.hide()
        self.update_counts("upscale", window.image_upscaler.repo.all())
        self.refresh_references()

    def tab_changed(self, index):
        if index == 1:
            self.ensure_upscale()
        elif index == 2:
            self.refresh_references()

    def ensure_upscale(self):
        if self.w.image_upscaler.tasks is not None:
            self.w.image_upscaler.tasks.show()
            return
        from eidory.ui.upscale_tasks import UpscaleTasks
        controller = self.w.image_upscaler
        controller.tasks = UpscaleTasks(controller)
        controller.tasks.setParent(self.upscale_placeholder, Qt.WindowType.Widget)
        layout = QVBoxLayout(self.upscale_placeholder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(controller.tasks)
        controller.tasks.show()

    def open(self, index=None):
        if index is not None and index != self.tabs.currentIndex():
            self.tabs.setCurrentIndex(index)
        else:
            self.tab_changed(self.tabs.currentIndex())
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def update_counts(self, source, jobs):
        ready = sum(j["state"] == "ready" for j in jobs)
        failed = sum(j["state"] in {"failed", "offline", "missing"} for j in jobs)
        self.counts[source] = (ready, failed)
        tab = 1 if source == "upscale" else 2
        title = "AI 放大" if tab == 1 else "AI 生图与编辑"
        if source == "ai":
            other_ready, other_failed = self.counts.get("references", (0, 0))
            ready += other_ready
            failed += other_failed
        parts = [title] + ([f"待审 {ready}"] if ready else []) + ([f"异常 {failed}"] if failed else [])
        self.tabs.setTabText(tab, " · ".join(parts))
        self.update_header()
        if source == "ai" and self.dialog.isVisible() and self.tabs.currentIndex() == 2:
            self.refresh_ai(jobs=jobs)

    def refresh_references(self):
        self.references = self.w.reference_generation.repo.references_for_review()
        self.counts["references"] = (sum(self.is_pending(j) for j in self.references),
                                     sum(j["state"] in {"failed", "offline", "missing"} for j in self.references))
        jobs = self.w.preview_ai.repo.all()
        self.update_counts("ai", jobs)
        if not self.dialog.isVisible() and self.tabs.currentIndex() == 2:
            self.refresh_ai(jobs=jobs)

    def update_index_count(self, failed):
        self.counts["index"] = (0, failed)
        self.tabs.setTabText(0, "索引与识别" + (f" · 异常 {failed}" if failed else ""))
        self.update_header()

    def update_header(self):
        ready = sum(v[0] for v in self.counts.values())
        failed = sum(v[1] for v in self.counts.values())
        label = "后台任务" + (f" · 待审 {ready}" if ready else "") + (f" · 异常 {failed}" if failed else "")
        self.w.workspace_chrome.tasks_button.setText(label)

    def refresh_ai(self, *_args, jobs=None):
        if jobs is None:
            jobs = self.w.preview_ai.repo.all()
        jobs = sorted([*jobs, *self.references], key=lambda j: j.get("created_at", ""), reverse=True)
        current = self.ai_table.item(self.ai_table.currentRow(), 0)
        selected = current.data(Qt.ItemDataRole.UserRole) if current else None
        mode = self.kind.currentIndex()
        failures = {"failed", "offline", "missing", "paused", "interrupted"}
        running = {"preparing", "submitting", "queued", "running", "downloading", "waiting"}
        counts = (sum(self.is_pending(j) for j in jobs), sum(j["state"] in failures for j in jobs))
        detached = sum(j["state"] == "ready" and bool(j.get("detached_reason")) for j in jobs)
        history_hint = f" · 原项目或节点已删除 {detached} 条（见「全部状态」）" if detached and mode == 1 else ""
        self.ai_summary.setText(f"待审核 {counts[0]} · 失败与中断 {counts[1]} · 共 {len(jobs)} 条{history_hint}")
        jobs = [j for j in jobs if not mode or (self.is_pending(j) if mode == 1 else
                                               j["state"] in (running if mode == 2 else failures))]
        self.visible_jobs = {job["id"]: job for job in jobs}
        with QSignalBlocker(self.ai_table):
            self.ai_table.clearSelection()
            self.ai_table.setRowCount(len(jobs))
            self.ai_table.setCurrentCell(-1, -1)
            for row, job in enumerate(jobs):
                reference = job.get("origin") != "preview"
                label = "项目参考" if reference else "局部编辑" if job.get("mode") == "edit" else "生图"
                name = (f"{job['project_title']} / {job['node_title']}" if reference else
                        Path(job.get("source_path") or "").name or (job.get("prompt") or "图片任务")[:100])
                states = STATE_LABELS if reference else LABELS
                detached = job.get("detached_reason") or ""
                status = "已失去归属" if detached and job["state"] == "ready" else states.get(job["state"], job["state"])
                for col, text in enumerate((label, name, status, detached or job.get("error") or "")):
                    item = QTableWidgetItem(text)
                    item.setToolTip(text)
                    item.setData(Qt.ItemDataRole.UserRole, job["id"])
                    self.ai_table.setItem(row, col, item)
                if job["id"] == selected:
                    self.ai_table.selectRow(row)
        self.update_selection()

    @staticmethod
    def is_pending(job):
        return job["state"] == "ready" and not job.get("detached_reason")

    def selected_job(self):
        item = self.ai_table.item(self.ai_table.currentRow(), 0)
        return getattr(self, "visible_jobs", {}).get(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def update_selection(self):
        job = self.selected_job()
        self.review.setEnabled(bool(job and not job.get("detached_reason")))
        self.discard.setText("撤销放弃" if job and job["state"] == "discarded" else "放弃结果")
        self.discard.setEnabled(bool(job and job["state"] in {"ready", "discarded"}))

    def discard_result(self):
        job = self.selected_job()
        if not job or job["state"] not in {"ready", "discarded"}:
            return
        reference = job.get("origin") != "preview"
        controller = self.w.reference_generation if reference else self.w.preview_ai
        if controller.working or self.w._database_maintenance_active or self.w._closing:
            self.feedback.setText("当前正在处理任务，请稍后再试。")
            return
        undo = job["state"] == "discarded"
        try:
            controller.repo.discard(job["id"], undo=undo)
        except (DatabaseError, OSError, ValueError) as exc:
            self.feedback.setText(f"未能更新结果：{exc}")
            self.refresh_references()
            return
        self.refresh_references()
        controller.refresh()
        self.feedback.setText("已撤销放弃。" if undo else "已放弃结果。可在「全部状态」中撤销。")

    def open_result(self):
        job = self.selected_job()
        if not job or job.get("detached_reason"):
            return
        if job.get("origin") != "preview":
            node = self.w.store.get_creative_node(job["node_id"])
            if node is None or node.project_id != job["project_id"]:
                self.refresh_references()
                return
            self.w.reference_generation.open_node(node, job_id=job["id"])
        else:
            image = self.w.store.get_image(job["image_id"]) if job.get("image_id") else None
            self.w.preview_ai.open(image, job["mode"], job_id=job["id"])
