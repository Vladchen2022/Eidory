from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from eidory.core.art_analysis import ArtAnalysisResult, DistributionEntry, PaletteEntry


class ArtAnalysisDonut(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._palette: tuple[PaletteEntry, ...] = ()
        self.setMinimumHeight(174)

    def set_palette(self, palette: tuple[PaletteEntry, ...]) -> None:
        self._palette = tuple(palette)
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        side = min(self.width() - 20, self.height() - 12)
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

        inner_margin = side * 0.28
        painter.setBrush(QColor("#2b3038"))
        painter.drawEllipse(rect.adjusted(inner_margin, inner_margin, -inner_margin, -inner_margin))
        painter.setPen(QPen(QColor("#e5eaf2")))
        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignCenter,
            f"{len(self._palette)} 色" if self._palette else "—",
        )


class ArtAnalysisWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.palette_buttons: list[QPushButton] = []
        self.lightness_bars: list[QProgressBar] = []
        self.saturation_bars: list[QProgressBar] = []
        self.temperature_bars: list[QProgressBar] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        self.status_label = QLabel("选择“美术分析”后计算当前图片")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color:#aeb7c4;padding:20px 6px;")
        root.addWidget(self.status_label)

        self.content = QWidget()
        self.content.setObjectName("artAnalysisContent")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(12)
        root.addWidget(self.content)
        self.content.hide()

        self.palette_title = self._section_title("主要色板 · 近似占比")
        self.content_layout.addWidget(self.palette_title)
        self.palette_grid = QGridLayout()
        self.palette_grid.setContentsMargins(0, 0, 0, 0)
        self.palette_grid.setHorizontalSpacing(6)
        self.palette_grid.setVerticalSpacing(6)
        self.content_layout.addLayout(self.palette_grid)

        self.donut_title = self._section_title("颜色占比")
        self.content_layout.addWidget(self.donut_title)
        self.donut = ArtAnalysisDonut()
        self.content_layout.addWidget(self.donut)

        self.lightness_title = self._section_title("明度分布 · L*")
        self.content_layout.addWidget(self.lightness_title)
        self.lightness_rows = QVBoxLayout()
        self.lightness_rows.setSpacing(6)
        self.content_layout.addLayout(self.lightness_rows)

        self.saturation_title = self._section_title("饱和度结构")
        self.content_layout.addWidget(self.saturation_title)
        self.saturation_rows = QVBoxLayout()
        self.saturation_rows.setSpacing(6)
        self.content_layout.addLayout(self.saturation_rows)

        self.temperature_title = self._section_title("冷暖比例")
        self.content_layout.addWidget(self.temperature_title)
        self.temperature_rows = QVBoxLayout()
        self.temperature_rows.setSpacing(6)
        self.content_layout.addLayout(self.temperature_rows)

        self.summary_title = self._section_title("明暗与对比")
        self.content_layout.addWidget(self.summary_title)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary_label.setStyleSheet(
            "background:#343c47;border:1px solid #4c5663;border-radius:6px;"
            "padding:9px;line-height:1.35;"
        )
        self.content_layout.addWidget(self.summary_label)
        self.sample_label = QLabel()
        self.sample_label.setStyleSheet("color:#929cab;font-size:11px;")
        self.content_layout.addWidget(self.sample_label)

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
        self._clear_layout(self.palette_grid)
        self.palette_buttons.clear()
        for index, entry in enumerate(result.palette):
            button = QPushButton(f"{entry.hex_code}   {entry.proportion * 100:.1f}%")
            button.setAccessibleName(f"复制颜色 {entry.hex_code}")
            button.setToolTip(f"点击复制 {entry.hex_code}")
            button.setFixedHeight(38)
            foreground = "#15181d" if self._relative_luminance(entry.rgb) > 0.52 else "#ffffff"
            button.setStyleSheet(
                f"QPushButton{{background:{entry.hex_code};color:{foreground};"
                "border:1px solid #596270;border-radius:6px;font-weight:600;}"
                "QPushButton:hover{border:2px solid #ffffff;}"
            )
            button.clicked.connect(
                lambda _checked=False, color=entry.hex_code: QApplication.clipboard().setText(color)
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
        self.sample_label.setText(
            f"本地采样 {result.sampled_width} × {result.sampled_height}，"
            f"{result.sampled_pixels:,} 个可见像素"
        )
        self.status_label.hide()
        self.content.show()

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
