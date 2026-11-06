"""Review duplicate groups with each action attached to its image."""
from __future__ import annotations

import sys
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from eidory.core.duplicate_review import DuplicateReview, fingerprint
from eidory.ui.wrapping_layout import WrappingLayout
from eidory.core.duplicate_detection import DuplicateGroup, DuplicateMember
from eidory.ui.board_image_loader import BoardImageLoader
from eidory.ui.image_preview_dialog import PreviewImageView
from eidory.ui.theme_colors import set_theme_style


GroupKey = tuple[str, tuple[int, ...]]


class DuplicateCompareDialog(QDialog):
    removeRequested = Signal(int)
    deleteRequested = Signal(int)
    undoRequested = Signal()

    def __init__(self, groups: list[DuplicateGroup], *, start_index: int = 0,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("重复图片 · 并列对比")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(1180, 820)
        self.setMinimumSize(800, 600)
        self._group_order: dict[GroupKey, int] = {}
        self._exact_owner_by_image: dict[int, GroupKey] = {}
        self.groups = self._ordered_groups(groups)
        self.group_index = max(0, min(start_index, len(groups) - 1))
        self._navigation_notice = ""
        self.candidate_index = 0
        self.selected_image_id: int | None = None
        self.cards: dict[int, QFrame] = {}
        self.views: dict[int, PreviewImageView] = {}
        self.select_buttons: dict[int, QPushButton] = {}
        self.meta_labels: dict[int, dict[str, QLabel]] = {}
        self._align_timer = QTimer(self)
        self._align_timer.setSingleShot(True)
        self._align_timer.timeout.connect(self._align_metadata_rows)
        self._generation = 0
        self._busy = False
        owner = parent
        while owner is not None and not hasattr(owner, 'store'):
            owner = owner.parent()
        self.review = DuplicateReview(owner.store) if owner is not None else None
        self.decisions = self.review.decisions() if self.review else {}
        self.resume_cursor = self.review.cursor() if self.review else None
        self._syncing = False
        self._full_resolution = set()
        self._actual_requested = False
        self._remember_cursor = False
        self.loader = BoardImageLoader(self)
        self.loader.loaded.connect(self._loaded)
        self.finished.connect(self._release)

        root = QVBoxLayout(self)
        root.setSpacing(8)
        row = QHBoxLayout()
        self.group_label = QLabel()
        self.group_label.setProperty("role", "flowTitle")
        row.addWidget(self.group_label)
        row.addStretch()
        self.fit_button = QPushButton("适应窗口")
        self.fit_button.clicked.connect(self._fit_images)
        self.link_mode = QComboBox()
        self.link_mode.addItems(['联动画面比例', '联动原始像素', '独立查看'])
        row.addWidget(self.link_mode)
        self.actual_button = QPushButton('原图 100%')
        self.actual_button.clicked.connect(self.actual_images)
        row.addWidget(self.actual_button)
        row.addWidget(self.fit_button)
        root.addLayout(row)
        self.reason_label = QLabel()
        self.reason_label.setWordWrap(True)
        self.reason_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self.reason_label)
        self.order_label = QLabel()
        self.order_label.setWordWrap(True)
        self.order_label.setProperty("role", "muted")
        root.addWidget(self.order_label)
        pair_row = QHBoxLayout()
        self.pair_label = QLabel()
        self.previous_candidate_button = QPushButton("↑ 本组上一份")
        self.next_candidate_button = QPushButton("↓ 本组下一份")
        self.previous_candidate_button.clicked.connect(lambda: self.change_candidate(-1))
        self.next_candidate_button.clicked.connect(lambda: self.change_candidate(1))
        pair_row.addWidget(self.pair_label)
        pair_row.addStretch()
        pair_row.addWidget(self.previous_candidate_button)
        pair_row.addWidget(self.next_candidate_button)
        root.addLayout(pair_row)
        review_row = WrappingLayout()
        for label, state in [('保留本组', 'keep'), ('不是重复', 'different'), ('稍后处理', 'later'), ('恢复待审', 'pending')]:
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, state=state: self.mark_review(state))
            review_row.addWidget(button)
        self.next_pending = QPushButton('下一待审组')
        self.next_pending.clicked.connect(self.next_unreviewed)
        review_row.addWidget(self.next_pending)
        self.resume_button = QPushButton('继续上次审核')
        self.resume_button.setEnabled(bool(self.resume_cursor))
        self.resume_button.clicked.connect(self.resume_review)
        review_row.addWidget(self.resume_button)
        root.addLayout(review_row)
        self.review_label = QLabel()
        root.addWidget(self.review_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        root.addWidget(self.scroll, 1)
        self.selection_label = QLabel()
        self.selection_label.setWordWrap(True)
        self.selection_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self.selection_label)
        modifier = "⌘" if sys.platform == "darwin" else "Ctrl+"
        help_label = QLabel(
            f"点击图片选中后：Delete 删除源文件 · {modifier}Delete 只从图库移除 · "
            f"← / → 切换组 · ↑ / ↓ 切换本组副本 · {modifier}Z 撤销\n"
            "删除会先保存回收备份，再移到废纸篓；移除保留硬盘上的源文件。"
        )
        help_label.setWordWrap(True)
        help_label.setProperty("role", "muted")
        root.addWidget(help_label)
        footer = QHBoxLayout()
        self.previous_button = QPushButton("← 上一组")
        self.next_button = QPushButton("下一组 →")
        self.undo_button = QPushButton("撤销上次删除 / 移除")
        close_button = QPushButton("关闭")
        self.previous_button.clicked.connect(lambda: self.change_group(-1))
        self.next_button.clicked.connect(lambda: self.change_group(1))
        self.undo_button.clicked.connect(self.undoRequested.emit)
        close_button.clicked.connect(self.reject)
        footer.addWidget(self.previous_button)
        footer.addWidget(self.next_button)
        footer.addStretch()
        footer.addWidget(self.undo_button)
        footer.addWidget(close_button)
        root.addLayout(footer)

        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setAutoRepeat(False)
        self._shortcut("Left", lambda: self.change_group(-1))
        self._shortcut("Right", lambda: self.change_group(1))
        self._shortcut("Up", lambda: self.change_candidate(-1))
        self._shortcut("Down", lambda: self.change_candidate(1))
        for key in ("Delete", "Backspace"):
            self._shortcut(key, lambda: self.request_action("delete"))
            self._shortcut("Ctrl+" + key, lambda: self.request_action("remove"))
        self._shortcut(QKeySequence.StandardKey.Undo, self.undoRequested.emit)
        self._render_group()
        self._remember_cursor = True

    def _shortcut(self, key, callback) -> None:
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        shortcut.setAutoRepeat(False)
        shortcut.activated.connect(callback)

    def mark_review(self, state):
        if not self.groups or self._busy:
            return
        group = self.groups[self.group_index]
        if self.review:
            self.review.mark(group, state)
            self.review.save_cursor(group, self.candidate_index)
        self.decisions[fingerprint(group)] = state
        self.update_review_label()

    def update_review_label(self):
        reviewed = sum(self.decisions.get(fingerprint(g)) in {'keep', 'different'} for g in self.groups)
        state = self.decisions.get(fingerprint(self.groups[self.group_index]), 'pending') if self.groups else 'pending'
        labels = {'keep': '已决定保留本组', 'different': '已标记为不是重复', 'later': '稍后处理', 'pending': '待审核'}
        self.review_label.setText(f"已审核 {reviewed} 组 · 待审核 {len(self.groups)-reviewed} 组 · 当前：{labels[state]}")

    def resume_review(self):
        if self.resume_cursor:
            for index, group in enumerate(self.groups):
                if fingerprint(group) == self.resume_cursor[0]:
                    self.group_index, self.candidate_index = index, self.resume_cursor[1]
                    self._render_group()
                    return
            self.review_label.setText('上次图片组已改变或已处理，请从当前待审组继续。')

    def next_unreviewed(self):
        order = list(range(self.group_index + 1, len(self.groups))) + list(range(self.group_index))
        for i in order:
            if self.decisions.get(fingerprint(self.groups[i])) not in {'keep', 'different'}:
                self.group_index, self.candidate_index = i, 0
                self._render_group()
                return
        self.review_label.setText('没有其他待审核组。左右方向键仍按原顺序查看全部组。')

    def actual_images(self):
        self._actual_requested = True
        self._generation += 1
        self.loader.reset()
        self.link_mode.setCurrentIndex(1)
        for member in self._visible_members():
            iid = member.image.id
            if iid not in self._full_resolution:
                self.loader.request((self._generation, iid), [member.image.file_path], 0, urgent=True)
            self.views[iid].set_actual_size()
        self.review_label.setText('正在按需读取原始像素；左右画面使用相同像素比例。')

    def sync_view(self, source_id):
        if self._syncing or self.link_mode.currentIndex() == 2:
            return
        source = self.views.get(source_id)
        if not source or not source._has_pixmap:
            return
        self._syncing = True
        try:
            center = source._current_center_ratio()
            for iid, target in self.views.items():
                if iid == source_id or not target._has_pixmap:
                    continue
                ratio = source._original_width / target._original_width if self.link_mode.currentIndex() == 0 else 1
                target.set_zoom_factor(source.zoom_factor * ratio, emit=False)
                target._center_on_ratio(center)
        finally:
            self._syncing = False

    def change_group(self, delta: int) -> None:
        target = self.group_index + delta
        if not self._busy and 0 <= target < len(self.groups):
            self.group_index = target
            self.candidate_index = 0
            self._navigation_notice = ""
            self._render_group()

    def change_candidate(self, delta: int) -> None:
        if self._busy or not self.groups:
            return
        target = self.candidate_index + delta
        if 0 <= target < len(self.groups[self.group_index].members) - 1:
            self.candidate_index = target
            self._navigation_notice = ""
            self._render_group()

    def _group_key(self, group: DuplicateGroup) -> GroupKey:
        ids = tuple(sorted(m.image.id for m in group.members))
        if group.kind == "exact":
            owner = None
            for iid in ids:
                known = self._exact_owner_by_image.get(iid)
                if known is None:
                    continue
                if owner is None:
                    owner = known
                elif known is not owner and known != owner:
                    return group.kind, ids
            if owner is not None:
                return owner
        return group.kind, ids

    def _ordered_groups(self, groups: list[DuplicateGroup]) -> list[DuplicateGroup]:
        """Keep one review order for the detection session, including undo."""
        for group in groups:
            key = self._group_key(group)
            if key not in self._group_order:
                self._group_order[key] = len(self._group_order) + 1
            if group.kind == "exact":
                for member in group.members:
                    self._exact_owner_by_image[member.image.id] = key
        return sorted(groups, key=lambda group: self._group_order[self._group_key(group)])

    def _visible_members(self) -> tuple[DuplicateMember, ...]:
        if not self.groups:
            return ()
        group = self.groups[self.group_index]
        if group.kind == "near" and len(group.members) > 2:
            # Old connected components do not establish any pairwise relation.
            # Do not present unrelated members as a deletion decision.
            return ()
        if len(group.members) < 2:
            return ()
        self.candidate_index = min(self.candidate_index, len(group.members) - 2)
        return group.members[0], group.members[self.candidate_index + 1]

    def set_groups(self, groups: list[DuplicateGroup]) -> None:
        previous_keys = [self._group_key(group) for group in self.groups]
        previous_index = self.group_index
        current_key = previous_keys[previous_index] if previous_keys else None
        visible = self._visible_members()
        candidate_id = visible[1].image.id if len(visible) == 2 else None
        self.groups = self._ordered_groups(groups)
        positions = {self._group_key(group): index for index, group in enumerate(self.groups)}
        self._navigation_notice = ""
        current_survives = current_key in positions
        if current_survives:
            self.group_index = positions[current_key]
            if len(previous_keys) != len(self.groups):
                self._navigation_notice = "结果已更新，当前图片组保持不变"
        else:
            # A shared image does not identify a near pair. When (A, B) is
            # resolved, proceed to the next surviving queue entry, never to a
            # distant (A, C) just because A remains in the library.
            following = next((key for key in previous_keys[previous_index + 1:] if key in positions), None)
            preceding = next((key for key in reversed(previous_keys[:previous_index]) if key in positions), None)
            if following is not None:
                self.group_index = positions[following]
                self._navigation_notice = "当前组已处理，已显示原顺序中的下一组"
            elif preceding is not None:
                self.group_index = positions[preceding]
                self._navigation_notice = "当前组已处理，已显示最后一个待处理组"
            else:
                self.group_index = min(previous_index, max(0, len(self.groups) - 1))
        self.candidate_index = 0
        if self.groups and current_survives:
            for index, member in enumerate(self.groups[self.group_index].members[1:]):
                if member.image.id == candidate_id:
                    self.candidate_index = index
                    break
        self._render_group()

    def _render_group(self) -> None:
        self._generation += 1
        self.loader.reset()
        self._full_resolution.clear()
        self._actual_requested = False
        self.selected_image_id = None
        self.cards.clear()
        self.views.clear()
        self.select_buttons.clear()
        self.meta_labels.clear()
        old = self.scroll.takeWidget()
        if old is not None:
            old.hide()
            old.deleteLater()
        content = QWidget()
        panels = QHBoxLayout(content)
        panels.setContentsMargins(0, 0, 0, 0)
        panels.setSpacing(12)
        self.scroll.setWidget(content)
        self.previous_button.setEnabled(bool(self.groups) and self.group_index > 0)
        self.next_button.setEnabled(self.group_index + 1 < len(self.groups))
        self.fit_button.setEnabled(bool(self.groups))
        self.previous_candidate_button.setEnabled(False)
        self.next_candidate_button.setEnabled(False)
        self.pair_label.setText("")
        if not self.groups:
            self.group_label.setText("本次检测的重复组已处理完")
            self.order_label.setText(f"本次检测共 {len(self._group_order)} 组，当前待处理 0 组")
            self.reason_label.setText("可以撤销上次操作，或关闭窗口后重新检测。")
            panels.addWidget(QLabel("没有剩余的重复候选组。"), alignment=Qt.AlignmentFlag.AlignCenter)
        else:
            group = self.groups[self.group_index]
            kind = "完全重复" if group.kind == "exact" else "近似重复"
            self.group_label.setText(
                f"待处理第 {self.group_index + 1} / {len(self.groups)} 组 · {kind} · {len(group.members)} 张"
            )
            original_number = self._group_order[self._group_key(group)]
            total = len(self._group_order)
            self.order_label.setText(
                f"本次检测编号 #{original_number} / {total} · 已解决 {max(0, total - len(self.groups))} 组 · "
                + (self._navigation_notice or "检测顺序固定，先完全重复再近似候选")
            )
            self.reason_label.setText(group.reason)
            visible = self._visible_members()
            if not visible:
                self.reason_label.setText(
                    "这是旧版合并的近似结果，无法证明组内图片互相重复。请关闭窗口后重新检测。"
                )
                panels.addWidget(QLabel("旧版近似结果需要重新检测，当前不提供删除操作。"))
                self.fit_button.setEnabled(False)
            elif len(group.members) > 2:
                self.pair_label.setText(
                    f"左侧为本组基准图 · 右侧副本 {self.candidate_index + 1} / {len(group.members) - 1} · 每次对比 2 张"
                )
                self.previous_candidate_button.setEnabled(self.candidate_index > 0)
                self.next_candidate_button.setEnabled(self.candidate_index < len(group.members) - 2)
            else:
                self.pair_label.setText("每次对比 2 张图片，操作按钮只作用于所在列")
            for member in visible:
                panels.addWidget(self._make_card(member), 1)
        self.update_review_label()
        if self.review and self.groups and self._remember_cursor:
            self.review.save_cursor(self.groups[self.group_index], self.candidate_index)
        self._update_selection()
        self.scroll.horizontalScrollBar().setValue(0)
        self._align_timer.start(0)

    @staticmethod
    def _text_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        label.setMinimumWidth(0)
        label.setToolTip(text)
        return label

    def _make_card(self, member: DuplicateMember) -> QFrame:
        image = member.image
        card = QFrame()
        card.setObjectName("duplicateCard")
        card.setMinimumWidth(350)
        col = QVBoxLayout(card)
        col.setSpacing(6)
        select = QPushButton("选择此图")
        select.setCheckable(True)
        select.setAutoDefault(False)
        select.setAccessibleName(f"选择图片：{image.file_name}；{member.folder_label}")
        select.toggled.connect(lambda _checked: self.select_image(image.id))
        col.addWidget(select)
        name = self._text_label(image.file_name)
        name.setMinimumHeight(24)
        col.addWidget(name)
        self.meta_labels[image.id] = {"文件名": name}
        view = PreviewImageView(minimum_zoom_factor=0.001)
        view.setMinimumSize(300, 180)
        view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        view.setAccessibleName(f"对比预览：{image.file_name}")
        view.set_message("正在载入…")
        col.addWidget(view, 1)
        form = QFormLayout()
        form.setVerticalSpacing(5)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        fields = [
            ("所属文件夹", member.folder_label or "未归类"),
            ("尺寸 / 大小", f"{image.width or '—'} × {image.height or '—'} · {self._size_text(image.file_size)}"),
            ("修改时间", image.modified_at or "未知"),
            ("完整路径", image.file_path),
        ]
        for title, value in fields:
            label = self._text_label(value)
            label.setObjectName("duplicate" + title)
            self.meta_labels[image.id][title] = label
            if title == "所属文件夹":
                label.setMinimumHeight(36)
            elif title == "完整路径":
                label.setMinimumHeight(54)
            form.addRow(title, label)
        col.addLayout(form)
        actions = QHBoxLayout()
        remove = QPushButton("从图库移除")
        remove.setObjectName("removeDuplicate")
        remove.setToolTip("只移除图库索引，保留硬盘源文件")
        delete = QPushButton("删除源文件")
        delete.setObjectName("deleteDuplicate")
        delete.setToolTip("保存回收备份后，将源文件移到废纸篓")
        for button, mode in ((remove, "remove"), (delete, "delete")):
            button.setAutoDefault(False)
            button.clicked.connect(lambda _checked=False, m=mode, iid=image.id: self.request_action(m, iid))
            actions.addWidget(button)
        col.addLayout(actions)
        self.cards[image.id] = card
        self.views[image.id] = view
        view.zoomChanged.connect(lambda *_: self.sync_view(image.id))
        view.horizontalScrollBar().valueChanged.connect(lambda *_: self.sync_view(image.id))
        view.verticalScrollBar().valueChanged.connect(lambda *_: self.sync_view(image.id))
        self.select_buttons[image.id] = select
        # Selection follows clicks on the image and its nearby metadata.
        for widget in [card, *card.findChildren(QWidget)]:
            widget.setProperty("duplicateImageId", image.id)
            widget.installEventFilter(self)
        self.loader.request((self._generation, image.id),
                            [image.file_path, image.thumbnail_path or ""], 2048)
        return card

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_align_timer"):
            self._align_timer.start(0)

    def _align_metadata_rows(self) -> None:
        if not self.meta_labels:
            return
        minimums = {"文件名": 24, "所属文件夹": 36, "完整路径": 54}
        for title in next(iter(self.meta_labels.values())):
            labels = [fields[title] for fields in self.meta_labels.values()]
            height = max(minimums.get(title, 0), *(
                max(label.fontMetrics().height(), label.heightForWidth(max(1, label.width())))
                for label in labels
            ))
            for label in labels:
                label.setFixedHeight(height)

    @staticmethod
    def _size_text(size: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:,.1f} {unit}" if unit != "B" else f"{size:,} B"
            size /= 1024
        return "—"

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            image_id = watched.property("duplicateImageId")
            if image_id in self.cards:
                self.select_image(image_id)
        return super().eventFilter(watched, event)

    def select_image(self, image_id: int) -> None:
        if image_id in self.cards and not self._busy:
            self.selected_image_id = image_id
            self._update_selection()

    def _update_selection(self) -> None:
        for image_id, card in self.cards.items():
            selected = image_id == self.selected_image_id
            border = "@link@" if selected else "@border@"
            set_theme_style(card, "QFrame#duplicateCard { background: @base@; border: 2px solid "
                            + border + "; border-radius: 8px; }")
            button = self.select_buttons[image_id]
            was_blocked = button.blockSignals(True)
            button.setChecked(selected)
            button.blockSignals(was_blocked)
            button.setText("已选中 · 快捷键作用于此图" if selected else "选择此图")
        if self.selected_image_id is None:
            self.selection_label.setText("尚未选中图片。点击要处理的图片，或使用该图片下方的按钮。")
            self.selection_label.setToolTip("")
        else:
            group = self.groups[self.group_index]
            image = next(m.image for m in group.members if m.image.id == self.selected_image_id)
            self.selection_label.setText(f"当前操作对象：{image.file_name}")
            self.selection_label.setToolTip(image.file_path)

    def request_action(self, mode: str, image_id: int | None = None) -> None:
        target = image_id if image_id is not None else self.selected_image_id
        if self._busy or target not in self.cards:
            return
        self.select_image(target)
        self._busy = True
        try:
            if mode == "remove":
                self.removeRequested.emit(target)
            elif mode == "delete":
                self.deleteRequested.emit(target)
        finally:
            self._busy = False
        # No new target is chosen after an action, including a failed action.
        self.selected_image_id = None
        self._update_selection()

    def _loaded(self, token, image, path) -> None:
        generation, image_id = token
        if generation != self._generation or image_id not in self.views:
            return
        view = self.views[image_id]
        if image.isNull():
            view.set_message("无法读取图片；仍可查看路径及文件信息")
            return
        member = next(m for m in self.groups[self.group_index].members if m.image.id == image_id)
        self._syncing = True
        try:
            view.set_pixmap(QPixmap.fromImage(image), original_width=member.image.width,
                            original_height=member.image.height, fit_to_window=not self._actual_requested, zoom_factor=1.)
        finally:
            self._syncing = False
        if self._actual_requested:
            self._full_resolution.add(image_id)
            self.review_label.setText('原图 100% · 联动原始像素；拖动画面可比较同一位置。')
        if path != member.image.file_path:
            view.setToolTip("源文件无法读取，当前显示缓存缩略图")

    def _fit_images(self) -> None:
        self._actual_requested = False
        self._syncing = True
        try:
            for view in self.views.values():
                view.fit_image_to_window()
        finally:
            self._syncing = False

    def done(self, result: int) -> None:
        self._align_timer.stop()
        self.loader.stop()
        super().done(result)

    def _release(self) -> None:
        self.loader.stop()
        if self.loader.pool.activeThreadCount():
            QTimer.singleShot(25, self._release)
        else:
            self.deleteLater()
