from __future__ import annotations

from dataclasses import replace
from collections.abc import Callable

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor
from eidory.ui.theme_colors import set_theme_style
from PySide6.QtWidgets import (
    QColorDialog, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from eidory.core.ai_vision import (
    AI_CHARACTER_VISION_FIELD_VALUES, AI_VISION_FIELD_VALUES,
    AI_VISION_LIGHTING_VALUES, ai_vision_label,
)
from eidory.core.filter_engine import same_filter_criteria
from eidory.core.search_filters import (
    FilterGroup, FilterPlan, SearchFilter, SCORED_FILTER_KINDS,
    file_type_filter_label, orientation_filter_label, size_filter_label,
)


class ConditionNameButton(QPushButton):
    """Keep a stable preferred width while eliding long names in narrow columns."""

    def __init__(self, text):
        super().__init__(text)
        self.full_text = text

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(self.fontMetrics().horizontalAdvance(self.full_text) + 12)
        return hint

    def minimumSizeHint(self):
        return QSize(24, self.sizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setText(self.fontMetrics().elidedText(self.full_text, Qt.TextElideMode.ElideRight, max(0, self.width() - 10)))


class FilterEditorDialog(QDialog):
    def __init__(self, *, condition: SearchFilter | None, label_for: Callable,
                 pick_values: Callable, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑筛选条件" if condition else "添加筛选条件")
        self.resize(440, 280)
        self.label_for = label_for
        self.pick_values = pick_values
        self._picked: SearchFilter | None = None
        self.result_filter: SearchFilter | None = None
        self.color = (255, 0, 0)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.kind_combo = QComboBox()
        for label, kind in [
            ("画面内容（语义）", "semantic"), ("文字信息（关键词）", "keyword"),
            ("画面颜色", "color"), ("相似图", "similar"), ("文件夹", "collection"),
            ("用户标签", "tag"), ("文件类型", "file_type"), ("方向", "orientation"),
            ("尺寸", "size"), ("状态", "status"), ("标记", "annotation"), ("AI 场景标签", "ai_scene"),
            ("AI 角色标签", "ai_character"),
        ]:
            self.kind_combo.addItem(label, kind)
        form.addRow("条件类型", self.kind_combo)
        self.text_input = QLineEdit()
        self.text_input.setMaxLength(500)
        self.text_input.setPlaceholderText("例如：飞机")
        form.addRow(self.text_input)
        self.value_combo = QComboBox()
        form.addRow(self.value_combo)
        self.pick_button = QPushButton("选择")
        form.addRow(self.pick_button)
        self.strength_box = QWidget()
        strength_layout = QVBoxLayout(self.strength_box)
        strength_layout.setContentsMargins(0, 0, 0, 0)
        self.strength_label = QLabel()
        strength_layout.addWidget(self.strength_label)
        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("宽松"))
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(50)
        self.strength_slider.setAccessibleName("当前条件匹配强度")
        strength_row.addWidget(self.strength_slider, 1)
        strength_row.addWidget(QLabel("严格"))
        strength_layout.addLayout(strength_row)
        self.strength_box.setToolTip("每个条件独立设置。根据基础范围内的分数计算相对强度，不是识别概率；其他条件不会改变本项标准。")
        form.addRow(self.strength_box)
        layout.addLayout(form)
        self.help_label = QLabel()
        self.help_label.setWordWrap(True)
        layout.addWidget(self.help_label)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setProperty("role", "warning")
        layout.addWidget(self.error_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定条件")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.kind_combo.currentIndexChanged.connect(self._change_kind)
        self.pick_button.clicked.connect(self._pick)
        self.strength_slider.valueChanged.connect(self._update_strength)
        if condition is not None:
            kind = condition.kind
            if kind == "ai_vision":
                kind = "ai_character" if str(condition.value).startswith("character_") else "ai_scene"
            self.kind_combo.setCurrentIndex(max(0, self.kind_combo.findData(kind)))
        self._change_kind()
        if condition is not None:
            self.text_input.setText(str(condition.value))
            self.value_combo.setCurrentIndex(max(0, self.value_combo.findData(condition.value)))
            self.strength_slider.setValue(condition.score_threshold if condition.score_threshold is not None else 50)
            if condition.kind == "color":
                self.color = condition.value
                self._update_color()
            elif condition.kind in {"collection", "tag", "similar"}:
                self._picked = condition
                self.pick_button.setText(self.label_for(condition))
        self._update_strength()

    def _update_strength(self):
        value = self.strength_slider.value()
        self.strength_label.setText(f"本条件匹配强度：{value} / 100")

    def _change_kind(self):
        kind = self.kind_combo.currentData()
        self._picked = None
        self.error_label.clear()
        self.text_input.setVisible(kind in {"keyword", "semantic"})
        self.pick_button.setVisible(kind in {"color", "collection", "tag", "similar"})
        self.pick_button.setText("使用图片墙中选中的图片" if kind == "similar" else "选择")
        self.strength_box.setVisible(kind in SCORED_FILTER_KINDS)
        self.value_combo.clear()
        options = []
        if kind == "file_type":
            options = [(file_type_filter_label(v), v) for v in ["media:image", "media:video", *("ext:"+e for e in (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"))]]
        elif kind == "orientation":
            options = [(orientation_filter_label(v), v) for v in ["landscape", "portrait", "square"]]
        elif kind == "size":
            options = [(size_filter_label(v), v) for v in ["large", "small"]]
        elif kind == "status":
            options = [("收藏", "favorite"), ("未索引", "unindexed"), ("源文件丢失", "missing"), ("文件可用", "available")]
        elif kind == "annotation":
            options = [("含有标记", "present")]
        elif kind in {"ai_scene", "ai_character"}:
            fields = dict(AI_CHARACTER_VISION_FIELD_VALUES if kind == "ai_character" else AI_VISION_FIELD_VALUES)
            if kind == "ai_scene":
                fields["lighting"] = AI_VISION_LIGHTING_VALUES
            options = [(ai_vision_label(field, value), f"{field}:{value}") for field, values in fields.items() for value in values]
        for label, value in options:
            self.value_combo.addItem(label, value)
        self.value_combo.setVisible(bool(options))
        self.help_label.setText({
            "keyword": "查找文件名、路径、备注和用户标签中的文字。", "semantic": "按画面内容匹配；需要本地语义索引。",
            "color": "匹配整张图片的颜色倾向，不指定某个物体的颜色。", "similar": "先在图片墙中选中一张已索引图片。",
            "ai_scene": "匹配已生成的 AI 场景标签；没有该标签的图片不命中。", "ai_character": "匹配已生成的 AI 角色标签；没有该标签的图片不命中。",
            "annotation": "查找在预览窗口里保存了画笔标记的图片。隐藏标记仍会命中；清空最后一笔后不再命中。",
        }.get(kind, "确定后自动更新图片墙。"))
        if kind == "color":
            self._update_color()

    def _update_color(self):
        value = QColor(*self.color)
        self.pick_button.setText("选择颜色  " + value.name().upper())

    def _pick(self):
        kind = self.kind_combo.currentData()
        if kind == "color":
            color = QColorDialog.getColor(QColor(*self.color), self, "选择画面颜色")
            if color.isValid():
                self.color = (color.red(), color.green(), color.blue())
                self._update_color()
        else:
            selected = self.pick_values(kind, self._picked)
            if selected is not None:
                self._picked = selected
                self.pick_button.setText(self.label_for(selected))

    def accept(self):
        kind = str(self.kind_combo.currentData())
        if kind in {"semantic", "keyword"}:
            value = self.text_input.text().strip()
            if not value:
                self.error_label.setText("请填写条件内容。")
                return
        elif kind == "color":
            value = self.color
        elif kind in {"collection", "tag", "similar"}:
            if self._picked is None:
                self.error_label.setText("请先选择条件内容。")
                return
            value = self._picked.value
        else:
            value = self.value_combo.currentData()
            if value is None:
                return
            if kind.startswith("ai_"):
                kind = "ai_vision"
        self.result_filter = SearchFilter(kind, value, self.strength_slider.value() if kind in SCORED_FILTER_KINDS else None)
        super().accept()


class FilterPanel(QWidget):
    draftChanged = Signal()
    applyRequested = Signal(object)
    scopeRequested = Signal()
    undoRequested = Signal()
    inspectExcludedRequested = Signal()
    strengthChanged = Signal(object)
    strengthFinished = Signal()

    def __init__(self, *, label_for: Callable, pick_values: Callable, parent=None):
        super().__init__(parent)
        self.label_for = label_for
        self.pick_values = pick_values
        self.draft = FilterPlan()
        self.applied = FilterPlan()
        self._busy = False
        self._live_available = False
        self.clear_context_on_apply = False
        self.error = ""
        self.strength_sliders: dict[tuple[str, int, int], QSlider] = {}
        self.section_boxes: dict[str, QWidget] = {}
        self.section_headers: dict[str, QHBoxLayout] = {}
        self.section_scrolls: dict[str, QScrollArea] = {}
        self.section_layouts: dict[str, QVBoxLayout] = {}
        self.condition_rows: dict[tuple[str, int, int], QWidget] = {}
        self._context_actions = []
        set_theme_style(self, """
            FilterPanel QPushButton { padding: 3px 7px; min-height: 20px; }
            FilterPanel QComboBox { padding: 3px 20px 3px 6px; min-height: 20px; }
            QWidget#includeFilterBox { background: @raised@; border: 1px solid @border@; border-radius: 6px; }
            QWidget#excludeFilterBox { background: @raised@; border: 1px solid @border@; border-radius: 6px; }
            QWidget#filterSectionHeader, QWidget#filterSectionContent { background: transparent; }
            QWidget#includeFilterBox QLabel, QWidget#excludeFilterBox QLabel { background: transparent; }
            QWidget#filterCondition { background: transparent; border: none; border-radius: 4px; }
            QWidget#filterCondition QPushButton { border: none; background: transparent; padding: 2px 3px; }
            QWidget#filterCondition QPushButton:hover { background: @hover@; }
            QPushButton#filterConditionName { text-align: left; }
        """)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        self.scope_row = QHBoxLayout()
        self.scope_row.setSpacing(6)
        self.scope_row.addWidget(QLabel("范围"))
        self.scope_button = QPushButton("全部图库 ▾")
        self.scope_button.setMinimumWidth(100)
        self.scope_button.setMaximumWidth(280)
        self.scope_button.clicked.connect(self.scopeRequested)
        self.scope_row.addWidget(self.scope_button)
        self._show_empty_conditions = False
        self.expand_button = QPushButton('＋ 筛选条件')
        self.expand_button.clicked.connect(self._toggle_empty_conditions)
        self.scope_row.addWidget(self.expand_button)
        self.scope_row.addStretch(1)
        outer.addLayout(self.scope_row)

        columns = QHBoxLayout()
        columns.setSpacing(10)
        columns.setAlignment(Qt.AlignmentFlag.AlignTop)
        for section in ("include", "exclude"):
            box = QWidget()
            box.setObjectName(section + "FilterBox")
            box.setAccessibleName("保留条件区域" if section == "include" else "排除条件区域")
            box.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            box.setMinimumWidth(0)
            layout = QVBoxLayout(box)
            layout.setContentsMargins(8, 6, 8, 6)
            layout.setSpacing(4)
            header = QWidget()
            header.setObjectName("filterSectionHeader")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(4)
            layout.addWidget(header)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setStyleSheet("QScrollArea { border: 0; background: transparent; }")
            scroll.viewport().setStyleSheet("background: transparent;")
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            content = QWidget()
            content.setObjectName("filterSectionContent")
            rows = QVBoxLayout(content)
            rows.setContentsMargins(0, 0, 0, 0)
            rows.setSpacing(4)
            rows.setAlignment(Qt.AlignmentFlag.AlignTop)
            scroll.setWidget(content)
            layout.addWidget(scroll)
            columns.addWidget(box, 1, Qt.AlignmentFlag.AlignTop)
            self.section_boxes[section] = box
            self.section_headers[section] = header_layout
            self.section_scrolls[section] = scroll
            self.section_layouts[section] = rows
        outer.addLayout(columns)
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._reflow_conditions)

        self.result_row = QHBoxLayout()
        self.result_row.setSpacing(6)
        self.result_label = QLabel()
        self.result_label.setMinimumWidth(110)
        self.result_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.result_row.addWidget(self.result_label, 1)
        self.context_button = QPushButton("范围限制")
        self.context_button.clicked.connect(self._show_context_actions)
        self.context_button.hide()
        self.result_row.addWidget(self.context_button)
        self.inspect_button = QPushButton("查看排除项")
        self.inspect_button.clicked.connect(self.inspectExcludedRequested)
        self.inspect_button.hide()
        self.result_row.addWidget(self.inspect_button)
        self.status_label = QLabel()
        self.status_label.setMaximumWidth(175)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.result_row.addWidget(self.status_label)
        self.undo_button = QPushButton("撤销")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undoRequested)
        self.result_row.addWidget(self.undo_button)
        self.clear_button = QPushButton("清除")
        self.clear_button.setToolTip("清除筛选条件，应用后恢复基础范围")
        self.clear_button.clicked.connect(self.clear_draft)
        self.result_row.addWidget(self.clear_button)
        self.apply_button = QPushButton("应用筛选")
        self.apply_button.setObjectName("filterApplyButton")
        self.apply_button.setProperty("variant", "primary")
        self.apply_button.clicked.connect(self._apply_draft)
        self.result_row.addWidget(self.apply_button)
        outer.addLayout(self.result_row)
        self._render()

    def _apply_draft(self):
        self.strengthFinished.emit()
        self.applyRequested.emit(self.draft)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_timer.start(0)

    def _reflow_conditions(self):
        # Each box fits its own rows. Only unusually long rule lists scroll,
        # keeping at least five conditions visible even in a short window.
        maximum = max(166, min(360, int(self.window().height() * .35) - 48))
        for section, layout in self.section_layouts.items():
            layout.invalidate()
            height = layout.sizeHint().height()
            self.section_scrolls[section].setFixedHeight(min(maximum, height))

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()

    def set_plan(self, plan: FilterPlan, *, force=False):
        if force or plan != self.applied:
            self.applied = plan
            self.draft = plan
            self.clear_context_on_apply = False
            self.error = ""
            self._render()

    def accept_live_plan(self, plan):
        # Never rebuild widgets under an active mouse drag.
        self.applied = plan
        self._update_status()

    def set_live_available(self, available):
        self._live_available = available
        self._update_status()

    def set_busy(self, busy: bool):
        self._busy = busy
        self.apply_button.setEnabled(not busy)
        self.scope_button.setEnabled(not busy)
        for slider in self.strength_sliders.values():
            slider.setEnabled(not busy)
        self._update_status()

    def _update_status(self):
        dirty = self.draft != self.applied or self.clear_context_on_apply
        live = self._live_available and same_filter_criteria(self.applied, self.draft) and not self.clear_context_on_apply
        text = ("正在筛选…" if self._busy else self.error or
                ("实时调整中…" if dirty and live else "修改未应用" if dirty else "拖动滑块即更新" if self.strength_sliders and self._live_available else
                 "条件已应用" if self.applied.has_conditions else "添加条件开始筛选"))
        self.status_label.setText(text)
        self.status_label.setToolTip(text)
        for (section, _, _), slider in self.strength_sliders.items():
            effect = "宽松：排除更多图片；严格：排除更少图片。" if section == "exclude" else "宽松：保留更多图片；严格：保留更少图片。"
            slider.setToolTip(effect + ("拖动立即更新图片墙，无需应用。" if live else "条件会自动更新；完成后可拖动实时调整。") + "\n数值是相对匹配强度，不是识别概率。")

    def clear_draft(self):
        self.strengthFinished.emit()
        self.draft = FilterPlan()
        self.clear_context_on_apply = True
        self.error = ""
        self._render()
        self.draftChanged.emit()

    def set_context_actions(self, actions):
        self._context_actions = actions
        self.context_button.setText(f"范围限制 · {len(actions)}")
        self.context_button.setToolTip("\n".join(label for label, _ in actions))
        self.context_button.setVisible(bool(actions))

    def _show_context_actions(self):
        menu = QMenu(self.context_button)
        for label, callback in self._context_actions:
            menu.addAction(label, callback)
        menu.exec(self.context_button.mapToGlobal(self.context_button.rect().bottomLeft()))

    def _render(self):
        self.strength_sliders.clear()
        self.condition_rows.clear()
        for section in ("include", "exclude"):
            self._clear_layout(self.section_headers[section])
            self._clear_layout(self.section_layouts[section])
            for index, group in enumerate(getattr(self.draft, section) or (FilterGroup(),)):
                self._render_group(section, index, group)
        self._reflow_conditions()
        self._layout_timer.start(0)
        self._update_status()
        self._sync_empty_conditions()

    def _toggle_empty_conditions(self):
        self._show_empty_conditions = not self._show_empty_conditions
        self._sync_empty_conditions()

    def _sync_empty_conditions(self):
        active = any(group.filters for plan in (self.draft, self.applied) for section in ('include', 'exclude') for group in getattr(plan, section))
        for box in self.section_boxes.values():
            box.setVisible(active or self._show_empty_conditions)
        self.expand_button.setVisible(not active)
        self.expand_button.setText('收起空筛选' if self._show_empty_conditions else '＋ 筛选条件')

    def _render_group(self, section, index, group):
        rows = self.section_layouts[section]
        if index == 0:
            header = self.section_headers[section]
        else:
            header_widget = QWidget()
            header_widget.setObjectName("filterSectionHeader")
            header = QHBoxLayout(header_widget)
            header.setContentsMargins(0, 4, 0, 0)
            header.setSpacing(4)
            rows.addWidget(header_widget)
        title = ("保留" if section == "include" else "排除") + (f" {index+1}" if index else "")
        title_label = QLabel(title)
        title_label.setProperty("role", "filterExclude" if section == "exclude" else "filterInclude")
        title_label.setToolTip("从保留结果中移除命中的图片" if section == "exclude" else "留下命中条件的图片")
        header.addWidget(title_label)
        mode = QComboBox()
        mode.addItem("全部满足", "all")
        mode.addItem("任一满足", "any")
        mode.setCurrentIndex(mode.findData(group.match_mode))
        mode.setAccessibleName(f"{title}组匹配方式")
        mode.setToolTip("全部：同时满足本组所有条件。任一：满足其中一个就命中。不同组的命中图片会合并。")
        mode.currentIndexChanged.connect(lambda value, s=section, i=index, combo=mode: self.set_group_mode(s, i, combo.currentData()))
        header.addWidget(mode)
        header.addStretch(1)
        add = QPushButton("＋ 条件")
        add.setAccessibleName("添加" + title + "条件")
        add.clicked.connect(lambda checked=False, s=section, i=index: self.edit_condition(s, i))
        header.addWidget(add)
        if index == 0:
            add_group = QPushButton("更多")
            add_group.setAccessibleName("新增" + title + "组")
            add_group.setToolTip("增加一组独立排除条件，各组分别排除" if section == "exclude" else "增加一组保留条件，合并各组命中结果")
            menu = QMenu(add_group)
            menu.addAction("再排除一批图片" if section == "exclude" else "或加一组条件", lambda s=section: self.add_group(s))
            add_group.setMenu(menu)
            header.addWidget(add_group)
        remove = QPushButton("×")
        remove.setFixedWidth(22)
        remove.setAccessibleName("删除" + title + "组")
        remove.setToolTip("删除本组条件")
        remove.clicked.connect(lambda checked=False, s=section, i=index: self.remove_group(s, i))
        remove.setVisible(bool(group.filters) or len(self._groups(section)) > 1)
        header.addWidget(remove)
        for ci, condition in enumerate(group.filters):
            row_widget = QWidget()
            row_widget.setObjectName("filterCondition")
            row_widget.setFixedHeight(30)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(4, 0, 2, 0)
            row_layout.setSpacing(4)
            if condition.kind == "color":
                swatch = QLabel()
                swatch.setFixedSize(12, 12)
                swatch.setStyleSheet(f"background: {QColor(*condition.value).name()}; border-radius: 2px;")
                row_layout.addWidget(swatch)
            text = self.label_for(condition)
            edit = ConditionNameButton(text)
            edit.setObjectName("filterConditionName")
            edit.setMinimumWidth(24)
            edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            edit.setToolTip(text + "\n点击编辑此条件")
            edit.clicked.connect(lambda checked=False, s=section, i=index, c=ci: self.edit_condition(s, i, c))
            row_layout.addWidget(edit, 3)
            if condition.kind in SCORED_FILTER_KINDS:
                row_layout.addWidget(QLabel("宽松"))
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0, 100)
                slider.setMinimumWidth(80)
                slider.setMaximumWidth(220)
                slider.setValue(condition.score_threshold or 0)
                slider.setAccessibleName(f"{title} · {text} · 匹配强度")
                slider.setProperty("filterSection", section)
                self.strength_sliders[section, index, ci] = slider
                row_layout.addWidget(slider, 2)
                row_layout.addWidget(QLabel("严格"))
                number = QLabel(str(slider.value()))
                number.setFixedWidth(24)
                number.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                row_layout.addWidget(number)
                slider.valueChanged.connect(lambda value, label=number: label.setText(str(value)))
                slider.valueChanged.connect(lambda value, s=section, i=index, c=ci: self._change_strength(s, i, c, value))
                slider.sliderReleased.connect(self.strengthFinished)
                slider.setEnabled(not self._busy)
            close = QPushButton("×")
            close.setFixedWidth(20)
            close.setAccessibleName("移除" + text)
            close.clicked.connect(lambda checked=False, s=section, i=index, c=ci: self.remove_condition(s, i, c))
            row_layout.addWidget(close)
            self.condition_rows[section, index, ci] = row_widget
            rows.addWidget(row_widget)
        if not group.filters:
            label = QLabel("未设条件，保留范围内全部图片" if section == "include" else "未设条件，不排除图片")
            label.setProperty("role", "muted")
            label.setFixedHeight(24)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            rows.addWidget(label)

    def _change_strength(self, section, index, condition_index, value):
        groups = self._groups(section)
        filters = list(groups[index].filters)
        filters[condition_index] = replace(filters[condition_index], score_threshold=value)
        groups[index] = replace(groups[index], filters=tuple(filters))
        self.draft = replace(self.draft, **{section: tuple(groups)})
        self._update_status()
        self.strengthChanged.emit(self.draft)

    def _groups(self, section):
        groups = list(getattr(self.draft, section))
        return groups or [FilterGroup()]

    def _set_groups(self, section, groups):
        self.strengthFinished.emit()
        self.draft = replace(self.draft, **{section: tuple(groups)})
        self.error = ""
        self._render()
        self.draftChanged.emit()

    def add_group(self, section):
        groups = self._groups(section)
        groups.append(FilterGroup())
        self._set_groups(section, groups)

    def remove_group(self, section, index):
        groups = self._groups(section)
        groups.pop(index)
        self._set_groups(section, groups)

    def set_group_mode(self, section, index, mode):
        groups = self._groups(section)
        groups[index] = replace(groups[index], match_mode=mode)
        self._set_groups(section, groups)

    def remove_condition(self, section, index, condition_index):
        groups = self._groups(section)
        conditions = list(groups[index].filters)
        conditions.pop(condition_index)
        groups[index] = replace(groups[index], filters=tuple(conditions))
        self._set_groups(section, groups)

    def edit_condition(self, section, index, condition_index=None):
        self.strengthFinished.emit()
        groups = self._groups(section)
        condition = groups[index].filters[condition_index] if condition_index is not None else None
        dialog = FilterEditorDialog(condition=condition, label_for=self.label_for, pick_values=self.pick_values, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.result_filter is None:
            return
        self.put_condition(section, index, dialog.result_filter, condition_index)

    def put_condition(self, section, index, condition, condition_index=None):
        groups = self._groups(section)
        conditions = list(groups[index].filters)
        if condition_index is None:
            conditions.append(condition)
        else:
            conditions[condition_index] = condition
        groups[index] = replace(groups[index], filters=tuple(conditions))
        self._set_groups(section, groups)
