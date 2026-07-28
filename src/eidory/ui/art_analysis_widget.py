from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from eidory.core.art_analysis import ArtAnalysisResult, DistributionEntry, PaletteEntry


class ArtAnalysisDonut(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._palette: tuple[PaletteEntry, ...] = ()
        self.setMinimumHeight(132)

    def set_palette(self, palette: tuple[PaletteEntry, ...]) -> None:
        self._palette = tuple(palette)
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        side = min(self.width() - 18, self.height() - 10)
        rect = QRectF(
            (self.width() - side) / 2,
            (self.height() - side) / 2,
            side,
            side,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        start_angle = 90 * 16
        for entry in self._palette:
            span = -int(round(entry.proportion * 360.0 * 16.0))
            painter.setBrush(QColor(*entry.rgb))
            painter.drawPie(rect, start_angle, span)
            start_angle += span

        inner_margin = side * 0.29
        painter.setBrush(QColor("#2b3038"))
        painter.drawEllipse(rect.adjusted(inner_margin, inner_margin, -inner_margin, -inner_margin))
        painter.setPen(QPen(QColor("#e5eaf2")))
        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignCenter,
            f"{len(self._palette)} 色" if self._palette else "—",
        )


class ArtAnalysisSpatialGrid(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._values: tuple[float, ...] = (0.0,) * 9
        self._color = QColor("#4f8ef7")
        self._marker: tuple[float, float] | None = None
        self.setMinimumHeight(126)

    def set_distribution(
        self,
        values: tuple[float, ...],
        *,
        color: QColor,
        marker: tuple[float, float] | None = None,
    ) -> None:
        self._values = tuple(values[:9]) + (0.0,) * max(0, 9 - len(values))
        self._color = QColor(color)
        self._marker = marker
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bounds = QRectF(1.0, 1.0, max(1.0, self.width() - 2.0), max(1.0, self.height() - 2.0))
        cell_width = bounds.width() / 3.0
        cell_height = bounds.height() / 3.0
        maximum = max(self._values, default=0.0)
        for index, value in enumerate(self._values[:9]):
            row, column = divmod(index, 3)
            rect = QRectF(
                bounds.left() + column * cell_width,
                bounds.top() + row * cell_height,
                cell_width,
                cell_height,
            ).adjusted(2.0, 2.0, -2.0, -2.0)
            intensity = 0.12 if maximum <= 0 else 0.14 + 0.72 * value / maximum
            fill = QColor(self._color)
            fill.setAlphaF(min(1.0, intensity))
            painter.setPen(QPen(QColor("#596270"), 1.0))
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, 4.0, 4.0)
            painter.setPen(QPen(QColor("#e5eaf2")))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{value * 100:.0f}%")

        if self._marker is not None:
            marker_x = bounds.left() + float(self._marker[0]) * bounds.width()
            marker_y = bounds.top() + float(self._marker[1]) * bounds.height()
            painter.setPen(QPen(QColor("#ffffff"), 2.0))
            painter.setBrush(QColor("#4f8ef7"))
            painter.drawEllipse(QRectF(marker_x - 6.0, marker_y - 6.0, 12.0, 12.0))


class ArtAnalysisWidget(QWidget):
    viewRequested = Signal(str, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result: ArtAnalysisResult | None = None
        self._selected_palette_index = 0
        self.palette_buttons: list[QPushButton] = []
        self.lightness_bars: list[QProgressBar] = []
        self.saturation_bars: list[QProgressBar] = []
        self.temperature_bars: list[QProgressBar] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self.status_label = QLabel("选择“美术分析”后计算当前图片")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color:#aeb7c4;padding:20px 6px;")
        root.addWidget(self.status_label)

        self.content = QWidget()
        self.content.setObjectName("artAnalysisContent")
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(7)
        root.addWidget(self.content)
        self.content.hide()

        self.tabs = QTabWidget()
        self.tabs.setObjectName("artAnalysisTabs")
        self.tabs.setDocumentMode(True)
        content_layout.addWidget(self.tabs)

        self.color_tab = QWidget()
        self.lightness_tab = QWidget()
        self.tone_tab = QWidget()
        self.composition_tab = QWidget()
        self.tabs.addTab(self.color_tab, "色彩")
        self.tabs.addTab(self.lightness_tab, "明度")
        self.tabs.addTab(self.tone_tab, "色调")
        self.tabs.addTab(self.composition_tab, "构图")

        self._build_color_tab()
        self._build_lightness_tab()
        self._build_tone_tab()
        self._build_composition_tab()

        self.sample_label = QLabel()
        self.sample_label.setStyleSheet("color:#929cab;font-size:11px;")
        self.sample_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.sample_label)

    def _build_color_tab(self) -> None:
        layout = QVBoxLayout(self.color_tab)
        layout.setContentsMargins(2, 8, 2, 2)
        layout.setSpacing(8)
        layout.addWidget(self._section_title("主要色板 · 点击色标定位画面区域"))
        self.palette_grid = QGridLayout()
        self.palette_grid.setContentsMargins(0, 0, 0, 0)
        self.palette_grid.setHorizontalSpacing(6)
        self.palette_grid.setVerticalSpacing(6)
        layout.addLayout(self.palette_grid)

        actions = QHBoxLayout()
        self.original_view_button = QPushButton("原图")
        self.palette_view_button = QPushButton("颜色概括")
        self.copy_palette_button = QPushButton("复制色板")
        self.export_palette_button = QPushButton("导出 PNG")
        self.original_view_button.clicked.connect(lambda: self.viewRequested.emit("original", -1))
        self.palette_view_button.clicked.connect(lambda: self.viewRequested.emit("palette", -1))
        self.copy_palette_button.clicked.connect(self._copy_palette)
        self.export_palette_button.clicked.connect(self._export_palette)
        for button in (
            self.original_view_button,
            self.palette_view_button,
            self.copy_palette_button,
            self.export_palette_button,
        ):
            actions.addWidget(button)
        layout.addLayout(actions)

        self.donut = ArtAnalysisDonut()
        layout.addWidget(self.donut)
        self.harmony_label = QLabel()
        self.harmony_label.setWordWrap(True)
        self.harmony_label.setStyleSheet(
            "background:#343c47;border:1px solid #4c5663;border-radius:6px;padding:8px;"
        )
        layout.addWidget(self.harmony_label)
        layout.addWidget(self._section_title("所选颜色在画面中的位置"))
        self.color_spatial_grid = ArtAnalysisSpatialGrid()
        layout.addWidget(self.color_spatial_grid)
        layout.addStretch(1)

    def _build_lightness_tab(self) -> None:
        layout = QVBoxLayout(self.lightness_tab)
        layout.setContentsMargins(2, 8, 2, 2)
        layout.setSpacing(8)
        layout.addWidget(self._section_title("明度分布 · L*"))
        self.lightness_rows = QVBoxLayout()
        self.lightness_rows.setSpacing(6)
        layout.addLayout(self.lightness_rows)
        layout.addWidget(self._section_title("明暗与对比"))
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary_label.setStyleSheet(
            "background:#343c47;border:1px solid #4c5663;border-radius:6px;"
            "padding:9px;line-height:1.35;"
        )
        layout.addWidget(self.summary_label)
        layout.addStretch(1)

    def _build_tone_tab(self) -> None:
        layout = QVBoxLayout(self.tone_tab)
        layout.setContentsMargins(2, 8, 2, 2)
        layout.setSpacing(8)
        layout.addWidget(self._section_title("饱和度结构"))
        self.saturation_rows = QVBoxLayout()
        self.saturation_rows.setSpacing(6)
        layout.addLayout(self.saturation_rows)
        layout.addWidget(self._section_title("冷暖比例"))
        self.temperature_rows = QVBoxLayout()
        self.temperature_rows.setSpacing(6)
        layout.addLayout(self.temperature_rows)
        self.tone_summary_label = QLabel(
            "冷暖判断会忽略低饱和与近中性色像素，结果用于观察画面倾向，不改变图片。"
        )
        self.tone_summary_label.setWordWrap(True)
        self.tone_summary_label.setStyleSheet("color:#aeb7c4;padding-top:6px;")
        layout.addWidget(self.tone_summary_label)
        layout.addStretch(1)

    def _build_composition_tab(self) -> None:
        layout = QVBoxLayout(self.composition_tab)
        layout.setContentsMargins(2, 8, 2, 2)
        layout.setSpacing(8)
        self.composition_summary_label = QLabel()
        self.composition_summary_label.setWordWrap(True)
        self.composition_summary_label.setStyleSheet(
            "background:#343c47;border:1px solid #4c5663;border-radius:6px;padding:8px;"
        )
        layout.addWidget(self.composition_summary_label)
        layout.addWidget(self._section_title("细节在画面中的分布 · 圆点为视觉重心线索"))
        self.detail_spatial_grid = ArtAnalysisSpatialGrid()
        layout.addWidget(self.detail_spatial_grid)
        actions = QHBoxLayout()
        self.detail_view_button = QPushButton("细节热图")
        self.restore_view_button = QPushButton("恢复原图")
        self.detail_view_button.clicked.connect(lambda: self.viewRequested.emit("detail", -1))
        self.restore_view_button.clicked.connect(lambda: self.viewRequested.emit("original", -1))
        actions.addWidget(self.detail_view_button)
        actions.addWidget(self.restore_view_button)
        layout.addLayout(actions)
        self.composition_note_label = QLabel(
            "视觉重心由局部边缘、饱和度与明暗差异估算；它是构图线索，不是主体识别。"
        )
        self.composition_note_label.setWordWrap(True)
        self.composition_note_label.setStyleSheet("color:#aeb7c4;")
        layout.addWidget(self.composition_note_label)
        layout.addStretch(1)

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight:600;")
        return label

    def set_loading(self, file_name: str) -> None:
        self.content.hide()
        self.status_label.setText(f"正在分析\n{file_name}")
        self.status_label.show()

    def set_error(self, message: str) -> None:
        self.content.hide()
        self.status_label.setText(f"无法分析当前图片\n{message}")
        self.status_label.show()

    def set_result(self, result: ArtAnalysisResult) -> None:
        self._result = result
        self._selected_palette_index = 0
        self._clear_layout(self.palette_grid)
        self.palette_buttons.clear()
        for index, entry in enumerate(result.palette):
            button = QPushButton(f"{entry.hex_code}   {entry.proportion * 100:.1f}%")
            button.setAccessibleName(f"定位颜色 {entry.hex_code}")
            button.setToolTip(f"在原图中高亮 {entry.hex_code} 对应区域")
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.setFixedHeight(34)
            foreground = "#15181d" if self._relative_luminance(entry.rgb) > 0.52 else "#ffffff"
            button.setStyleSheet(
                f"QPushButton{{background:{entry.hex_code};color:{foreground};"
                "border:1px solid #596270;border-radius:6px;font-weight:600;}"
                "QPushButton:hover{border:2px solid #ffffff;}"
                "QPushButton:checked{border:3px solid #4f8ef7;}"
            )
            button.clicked.connect(
                lambda _checked=False, palette_index=index: self._select_palette(palette_index)
            )
            self.palette_grid.addWidget(button, index // 2, index % 2)
            self.palette_buttons.append(button)

        self.donut.set_palette(result.palette)
        self.lightness_bars = self._set_distribution_rows(
            self.lightness_rows,
            result.lightness_distribution,
            colors=("#171a20", "#333945", "#59616e", "#858d99", "#b9c0c8", "#eef2f5"),
        )
        self.saturation_bars = self._set_distribution_rows(
            self.saturation_rows,
            result.saturation_distribution,
            colors=("#66717f", "#4d7fb4", "#2f8be8"),
        )
        self.temperature_bars = self._set_distribution_rows(
            self.temperature_rows,
            result.temperature_distribution,
            colors=("#df795c", "#5c91d8", "#8d929a"),
        )
        self.summary_label.setText(
            f"平均明度：{result.average_lightness:.1f} / 100\n"
            f"明度跨度：{result.tonal_range:.1f}\n"
            f"对比倾向：{result.contrast_label}（σ {result.lightness_stddev:.1f}）\n"
            f"暗部剪切：{result.shadow_clipping * 100:.1f}%\n"
            f"高光剪切：{result.highlight_clipping * 100:.1f}%"
        )
        self.harmony_label.setText(
            f"配色关系：{result.harmony_label}\n{result.harmony_description}"
        )
        self.composition_summary_label.setText(
            f"细节密度：{result.detail_density * 100:.1f}%（{result.detail_label}）\n"
            f"视觉重心：横向 {result.visual_center_x * 100:.0f}% · "
            f"纵向 {result.visual_center_y * 100:.0f}%\n"
            f"左右相似度：{result.horizontal_similarity * 100:.1f}%"
        )
        self.detail_spatial_grid.set_distribution(
            result.detail_distribution,
            color=QColor("#f08a5d"),
            marker=(result.visual_center_x, result.visual_center_y),
        )
        if result.palette:
            first = result.palette[0]
            self.color_spatial_grid.set_distribution(
                first.spatial_distribution,
                color=QColor(*first.rgb),
            )
        self.sample_label.setText(
            f"本地采样 {result.sampled_width} × {result.sampled_height}，"
            f"{result.sampled_pixels:,} 个可见像素"
        )
        self.status_label.hide()
        self.content.show()

    def _select_palette(self, palette_index: int) -> None:
        if self._result is None or not (0 <= palette_index < len(self._result.palette)):
            return
        self._selected_palette_index = palette_index
        for index, button in enumerate(self.palette_buttons):
            button.blockSignals(True)
            button.setChecked(index == palette_index)
            button.blockSignals(False)
        entry = self._result.palette[palette_index]
        self.color_spatial_grid.set_distribution(
            entry.spatial_distribution,
            color=QColor(*entry.rgb),
        )
        self.viewRequested.emit("highlight", palette_index)

    def _copy_palette(self) -> None:
        if self._result is None:
            return
        text = "\n".join(
            f"{entry.hex_code}\t{entry.proportion * 100:.1f}%"
            for entry in self._result.palette
        )
        QApplication.clipboard().setText(text)

    def _export_palette(self) -> None:
        if self._result is None or not self._result.palette:
            return
        output_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出色板",
            str(Path.home() / "eidory-palette.png"),
            "PNG Image (*.png)",
        )
        if not output_path:
            return
        target = Path(output_path)
        if target.suffix.lower() != ".png":
            target = target.with_suffix(".png")
        image = QImage(720, 150, QImage.Format.Format_ARGB32)
        image.fill(QColor("#f7f8fa"))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        x = 18.0
        available = image.width() - 36.0
        for index, entry in enumerate(self._result.palette):
            width = available * entry.proportion
            if index == len(self._result.palette) - 1:
                width = image.width() - 18.0 - x
            rect = QRectF(x, 18.0, max(1.0, width), 84.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(*entry.rgb))
            painter.drawRect(rect)
            x += width
        painter.setPen(QPen(QColor("#20242b")))
        legend = "   ".join(
            f"{entry.hex_code} {entry.proportion * 100:.1f}%"
            for entry in self._result.palette
        )
        painter.drawText(QRectF(18.0, 108.0, available, 28.0), Qt.AlignmentFlag.AlignCenter, legend)
        painter.end()
        if not image.save(str(target), "PNG"):
            QMessageBox.warning(self, "Eidory", "色板导出失败。")

    def _set_distribution_rows(
        self,
        layout: QVBoxLayout,
        entries: Iterable[DistributionEntry],
        *,
        colors: tuple[str, ...],
    ) -> list[QProgressBar]:
        self._clear_layout(layout)
        bars: list[QProgressBar] = []
        for entry, color in zip(entries, colors, strict=False):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            name = QLabel(entry.label)
            name.setFixedWidth(48)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(int(round(entry.proportion * 1000)))
            bar.setTextVisible(False)
            bar.setFixedHeight(9)
            bar.setStyleSheet(
                "QProgressBar{background:#414955;border:none;border-radius:4px;}"
                f"QProgressBar::chunk{{background:{color};border-radius:4px;}}"
            )
            value = QLabel(f"{entry.proportion * 100:.1f}%")
            value.setFixedWidth(42)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row_layout.addWidget(name)
            row_layout.addWidget(bar, 1)
            row_layout.addWidget(value)
            layout.addWidget(row)
            bars.append(bar)
        return bars

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    @staticmethod
    def _relative_luminance(rgb: tuple[int, int, int]) -> float:
        channels = []
        for channel in rgb:
            value = channel / 255.0
            channels.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
