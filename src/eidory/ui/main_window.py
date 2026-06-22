from __future__ import annotations

import os
import queue
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import html as html_lib
import json
import mimetypes
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np
from PySide6.QtCore import QFile, QBuffer, QFileSystemWatcher, QIODevice, QRect, QSize, QStringListModel, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QDesktopServices,
    QIcon,
    QImage,
    QImageReader,
    QKeySequence,
    QPainter,
    QPixmap,
    QPen,
    QShortcut,
    QTextOption,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QTabBar,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QSizePolicy,
    QWidget,
)

from eidory.config import AppPaths
from eidory.core.ai_vision import (
    AI_VISION_FIELD_VALUES,
    AI_VISION_LIGHTING_VALUES,
    AI_VISION_PROMPT_VERSION,
    AIVisionProvider,
    ai_vision_label,
)
from eidory.core.creative_templates import (
    CREATIVE_TEMPLATES,
    CreativeTemplateNode,
    creative_template_by_id,
    template_search_query,
)
from eidory.core.duplicate_detection import (
    DuplicateGroup,
    ImageDHashRecord,
    NearDuplicateCandidate,
    build_image_dhash_records,
    find_duplicate_groups,
    find_near_duplicate_candidates,
)
from eidory.core.ai_vision_worker import AIVisionProgress, AIVisionWorker
from eidory.core.embedding_provider import JinaClipV2Provider
from eidory.core.embedding_worker import EmbeddingProgress, EmbeddingWorker
from eidory.core.exporter import (
    ExportResult,
    export_images_to_directory,
    export_library_to_directory,
)
from eidory.core.inspiration import (
    InspirationMatch,
    InspirationTerm,
    mix_inspiration_search_results,
)
from eidory.core.llm_provider import LMStudioProvider, LLMProviderError, SearchPlanFilter
from eidory.core.media_types import (
    SUPPORTED_MEDIA_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    is_supported_image,
    is_supported_media,
    is_supported_video,
)
from eidory.core.media_tools import find_media_tool
from eidory.core.metadata_store import MetadataStore
from eidory.core.reference_grouping import ReferenceGroup, cluster_reference_vectors
from eidory.core.scanner import ImageScanner, ScanResult
from eidory.core.search_filters import (
    SCORED_FILTER_KINDS,
    SearchChainResult,
    SearchFilter,
    ai_vision_filter_parts,
    file_type_filter_label,
    filter_label,
    format_color_hex,
    format_filter_chain,
    last_score_filter_kind,
    orientation_filter_label,
    search_filter_from_payload,
    search_filter_to_payload,
    size_filter_label,
)
from eidory.core.search_service import SearchService
from eidory.core.thumbnailer import Thumbnailer
from eidory.core.vector_index import VectorIndex
from eidory.models import CreativeNodeItem, ImageItem
from eidory.ui.accessibility import disable_qt_accessibility
from eidory.ui.collection_tree import CollectionTreeWidget
from eidory.ui.image_preview_dialog import ImagePreviewDialog
from eidory.ui.justified_image_grid import JustifiedImageGridView
from eidory.ui.near_duplicate_dialog import NearDuplicateDecision, NearDuplicateDialog
from eidory.ui.project_board import ProjectBoardView


LLM_SERVICE_OPTIONS = [
    ("lm_studio", "LM Studio"),
    ("openai", "OpenAI API"),
    ("deepseek", "DeepSeek API"),
    ("ollama", "Ollama"),
    ("openai_compatible", "OpenAI-compatible"),
]

DEFAULT_LLM_ENDPOINTS = {
    "lm_studio": "http://localhost:1234/v1",
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "ollama": "http://localhost:11434/v1",
    "openai_compatible": "http://localhost:1234/v1",
}

COLLECTION_VIRTUAL_FILTER_ROLE = Qt.ItemDataRole.UserRole + 3
PROJECT_LIST_KIND_ROLE = Qt.ItemDataRole.UserRole + 10
PROJECT_LIST_ID_ROLE = Qt.ItemDataRole.UserRole + 11
PROJECT_LIST_NAME_ROLE = Qt.ItemDataRole.UserRole + 12
PROJECT_LIST_COLOR_ROLE = Qt.ItemDataRole.UserRole + 13
PROJECT_LIST_COUNT_ROLE = Qt.ItemDataRole.UserRole + 14
PROJECT_LIST_PINNED_ROLE = Qt.ItemDataRole.UserRole + 15
PROJECT_LIST_SECTION_ID_ROLE = Qt.ItemDataRole.UserRole + 16
PROJECT_LIST_SECTION_EXPANDED_ROLE = Qt.ItemDataRole.UserRole + 17
PROJECT_LIST_SECTION_KIND = "section"
VIRTUAL_COLLECTION_FILTERS = (
    ("untagged", "未标签", "没有用户手动标签的图片/视频。"),
    ("un_ai_tagged", "未AI标签", "还没有可用 AI 场景视觉标签的图片。"),
    ("uncategorized", "未分类", "没有放入任何 Eidory 文件夹的图片/视频。"),
)


class EqualWidthTabBar(QTabBar):
    BUTTON_MATCH_HEIGHT = 26
    BOTTOM_GAP = 8
    MIN_TAB_WIDTH = 72

    @classmethod
    def minimum_width_for_tab_count(cls, count: int) -> int:
        return max(1, count) * cls.MIN_TAB_WIDTH

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.updateGeometry()

    def tabSizeHint(self, index: int) -> QSize:
        size = super().tabSizeHint(index)
        count = max(1, self.count())
        parent_width = self.parentWidget().width() if self.parentWidget() is not None else 0
        available_width = max(self.width(), parent_width)
        if available_width > 0:
            size.setWidth(max(self.MIN_TAB_WIDTH, available_width // count))
        else:
            size.setWidth(max(self.MIN_TAB_WIDTH, size.width()))
        size.setHeight(self.BUTTON_MATCH_HEIGHT + self.BOTTOM_GAP)
        return size


class ProjectListItemDelegate(QStyledItemDelegate):
    SECTION_TOP_GAP = 7
    SECTION_HEIGHT = 23

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        kind = index.data(PROJECT_LIST_KIND_ROLE)
        if kind == PROJECT_LIST_SECTION_KIND:
            painter.save()
            rect = option.rect.adjusted(2, self.SECTION_TOP_GAP, -2, -1)
            painter.fillRect(rect, QColor("#343b44"))
            painter.setPen(QColor("#d8dee9"))
            expanded = bool(index.data(PROJECT_LIST_SECTION_EXPANDED_ROLE))
            section_text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
            text = option.fontMetrics.elidedText(
                f"{'▾' if expanded else '▸'} {section_text}",
                Qt.TextElideMode.ElideRight,
                max(1, rect.width() - 12),
            )
            painter.drawText(
                rect.adjusted(6, 0, -6, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                text,
            )
            painter.restore()
            return
        if kind not in {"temporary", "quick", "creative"}:
            super().paint(painter, option, index)
            return

        painter.save()
        rect = option.rect.adjusted(2, 1, -2, -1)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        foreground = QColor("#edf3fb")
        if selected:
            painter.fillRect(rect, option.palette.highlight().color())
            foreground = option.palette.highlightedText().color()
        else:
            text_brush = index.data(Qt.ItemDataRole.ForegroundRole)
            if isinstance(text_brush, QBrush) and text_brush.color().isValid():
                foreground = text_brush.color()

        painter.setPen(foreground)

        count_value = index.data(PROJECT_LIST_COUNT_ROLE)
        count_text = "" if count_value is None else str(count_value)
        count_width = max(34, option.fontMetrics.horizontalAdvance(count_text) + 8)
        count_rect = QRect(rect.right() - count_width, rect.top(), count_width, rect.height())
        name_width = max(8, rect.width() - count_width - 14)
        name_rect = QRect(rect.left() + 6, rect.top(), name_width, rect.height())

        prefix = "◇ "
        pinned = "⬆ " if bool(index.data(PROJECT_LIST_PINNED_ROLE)) else ""
        name = str(index.data(PROJECT_LIST_NAME_ROLE) or "")
        label = option.fontMetrics.elidedText(
            f"{prefix}{pinned}{name}",
            Qt.TextElideMode.ElideMiddle,
            max(1, name_rect.width()),
        )
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
        painter.drawText(count_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, count_text)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        size = super().sizeHint(option, index)
        if index.data(PROJECT_LIST_KIND_ROLE) == PROJECT_LIST_SECTION_KIND:
            size.setHeight(self.SECTION_TOP_GAP + self.SECTION_HEIGHT)
        else:
            size.setHeight(max(20, size.height()))
        return size


TOOL_BUTTON_MIN_WIDTH = EqualWidthTabBar.MIN_TAB_WIDTH
BOARD_ICON_BUTTON_SIZE = QSize(32, 26)
BOARD_ICON_SIZE = QSize(18, 18)
LEFT_SIDEBAR_WIDTH = 300
RIGHT_SIDEBAR_WIDTH = 386
SIDEBAR_COLLAPSE_THRESHOLD = 48
SIDEBAR_COUNT_COLUMN_WIDTH = 68


def _board_control_icon(kind: str) -> QIcon:
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    line_color = QColor("#e4e9f1")
    fill_color = QColor("#e4e9f1")
    muted_color = QColor("#8792a0")
    painter.setPen(QPen(line_color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.setBrush(Qt.BrushStyle.NoBrush)

    if kind == "pin":
        painter.drawEllipse(9, 3, 6, 6)
        painter.drawLine(12, 9, 12, 18)
        painter.drawLine(8, 12, 16, 12)
        painter.drawLine(12, 18, 9, 21)
    elif kind == "focus":
        painter.drawRoundedRect(4, 5, 16, 14, 2, 2)
        painter.drawLine(8, 5, 8, 19)
        painter.drawLine(16, 5, 16, 19)
        painter.setPen(QPen(line_color, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(5, 20, 19, 4)
    elif kind == "fit":
        painter.drawLine(5, 9, 5, 5)
        painter.drawLine(5, 5, 9, 5)
        painter.drawLine(15, 5, 19, 5)
        painter.drawLine(19, 5, 19, 9)
        painter.drawLine(19, 15, 19, 19)
        painter.drawLine(19, 19, 15, 19)
        painter.drawLine(9, 19, 5, 19)
        painter.drawLine(5, 19, 5, 15)
        painter.setPen(QPen(muted_color, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawRect(8, 8, 8, 8)
    elif kind == "flip":
        painter.drawLine(12, 4, 12, 20)
        painter.drawLine(5, 7, 10, 12)
        painter.drawLine(5, 17, 10, 12)
        painter.drawLine(19, 7, 14, 12)
        painter.drawLine(19, 17, 14, 12)
    elif kind == "grayscale":
        painter.setBrush(fill_color)
        painter.drawPie(5, 5, 14, 14, 90 * 16, 180 * 16)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(5, 5, 14, 14)
        painter.drawLine(12, 5, 12, 19)

    painter.end()
    return QIcon(pixmap)


class MissingFilesDialog(QDialog):
    def __init__(
        self,
        *,
        images: list[ImageItem],
        folder_labels: dict[int, str],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("缺失文件修复")
        self.resize(980, 520)
        layout = QVBoxLayout(self)
        intro = QLabel("这些文件的索引还在，但源文件已经不在原路径。可以单个重新定位，也可以按旧目录批量重映射。")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["文件名", "原路径", "所在文件夹", "尺寸", "大小"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.setColumnWidth(0, 180)
        self.table.setColumnWidth(1, 430)
        self.table.setColumnWidth(2, 210)
        self.table.setColumnWidth(3, 90)
        self.table.setColumnWidth(4, 90)
        layout.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.relink_button = QPushButton("重新指定选中文件")
        self.remap_button = QPushButton("按目录批量重定位")
        self.remove_button = QPushButton("从图库移除选中")
        self.close_button = QPushButton("关闭")
        row.addWidget(self.relink_button)
        row.addWidget(self.remap_button)
        row.addWidget(self.remove_button)
        row.addStretch(1)
        row.addWidget(self.close_button)
        layout.addLayout(row)
        self.close_button.clicked.connect(self.reject)
        self.set_images(images, folder_labels)

    def set_images(self, images: list[ImageItem], folder_labels: dict[int, str]) -> None:
        self.table.setRowCount(0)
        for image in images:
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [
                image.file_name,
                image.file_path,
                folder_labels.get(image.id, "-"),
                self._dimension_text(image),
                self._size_text(image.file_size),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, image.id)
                self.table.setItem(row, column, item)
        has_rows = bool(images)
        self.relink_button.setEnabled(has_rows)
        self.remap_button.setEnabled(has_rows)
        self.remove_button.setEnabled(has_rows)
        if has_rows:
            self.table.selectRow(0)

    def selected_image_ids(self) -> list[int]:
        ids: list[int] = []
        for item in self.table.selectedItems():
            image_id = item.data(Qt.ItemDataRole.UserRole)
            if image_id is not None:
                ids.append(int(image_id))
        return sorted(set(ids))

    def current_image_id(self) -> int | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return int(value) if value is not None else None

    @staticmethod
    def _dimension_text(image: ImageItem) -> str:
        if image.width and image.height:
            return f"{image.width} x {image.height}"
        return "-"

    @staticmethod
    def _size_text(file_size: int) -> str:
        if file_size >= 1024 * 1024:
            return f"{file_size / (1024 * 1024):.1f} MB"
        if file_size >= 1024:
            return f"{file_size / 1024:.1f} KB"
        return f"{file_size} B"


class DuplicateResultsDialog(QDialog):
    def __init__(self, groups: list[DuplicateGroup], *, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("重复 / 近重复图片")
        self.resize(1040, 620)
        self.selected_group_image_ids: list[int] = []
        layout = QVBoxLayout(self)
        intro = QLabel("完全重复是文件内容相同；近重复是同图不同尺寸、轻微裁切或同构图变体的候选。不要默认删除，先看所在文件夹。")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["项目", "所在文件夹", "尺寸", "大小", "路径"])
        self.tree.setColumnWidth(0, 260)
        self.tree.setColumnWidth(1, 260)
        self.tree.setColumnWidth(2, 100)
        self.tree.setColumnWidth(3, 90)
        self.tree.setColumnWidth(4, 420)
        layout.addWidget(self.tree, 1)

        for index, group in enumerate(groups, start=1):
            title = "完全重复" if group.kind == "exact" else "近重复"
            parent_item = QTreeWidgetItem([
                f"{title} #{index}：{len(group.members)} 张",
                group.reason,
                "",
                "",
                "",
            ])
            parent_item.setData(0, Qt.ItemDataRole.UserRole, [member.image.id for member in group.members])
            self.tree.addTopLevelItem(parent_item)
            for member in group.members:
                image = member.image
                child = QTreeWidgetItem([
                    image.file_name,
                    member.folder_label or "-",
                    MissingFilesDialog._dimension_text(image),
                    MissingFilesDialog._size_text(image.file_size),
                    image.file_path,
                ])
                child.setData(0, Qt.ItemDataRole.UserRole, [image.id])
                parent_item.addChild(child)
            parent_item.setExpanded(index <= 8)

        row = QHBoxLayout()
        self.load_group_button = QPushButton("载入选中组")
        close_button = QPushButton("关闭")
        row.addWidget(self.load_group_button)
        row.addStretch(1)
        row.addWidget(close_button)
        layout.addLayout(row)
        self.load_group_button.clicked.connect(self._accept_selected_group)
        close_button.clicked.connect(self.reject)

    def _accept_selected_group(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            return
        value = item.data(0, Qt.ItemDataRole.UserRole)
        if not value:
            return
        self.selected_group_image_ids = [int(image_id) for image_id in value]
        self.accept()


def _load_scaled_qt_pixmap(image_path: str, max_width: int, max_height: int) -> QPixmap:
    if max_width <= 0 or max_height <= 0:
        return QPixmap()
    path = Path(image_path)
    if not path.exists():
        return QPixmap()
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    source_size = reader.size()
    if source_size.isValid() and (
        source_size.width() > max_width or source_size.height() > max_height
    ):
        scaled_size = QSize(source_size.width(), source_size.height())
        scaled_size.scale(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio)
        reader.setScaledSize(scaled_size)
    image = reader.read()
    return QPixmap.fromImage(image) if not image.isNull() else QPixmap()


class ImageCompareDialog(QDialog):
    def __init__(self, images: list[ImageItem], *, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("对比查看")
        self.resize(1280, 760)
        layout = QVBoxLayout(self)
        hint = QLabel("多图对比用于最终判断。需要精细缩放时，双击单张图进入快速预览。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QHBoxLayout(content)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(10)
        for image in images[:6]:
            column = QVBoxLayout()
            preview = QLabel()
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setMinimumSize(260, 380)
            preview.setStyleSheet("background: #20262e; border: 1px solid #47515d;")
            pixmap = QPixmap()
            if not image.is_missing and Path(image.file_path).exists() and not is_supported_video(image.file_path):
                pixmap = _load_scaled_qt_pixmap(image.file_path, 1040, 1040)
            if not pixmap.isNull():
                preview.setPixmap(
                    pixmap.scaled(
                        520,
                        520,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                preview.setText("无法预览")
            info = QLabel(
                f"{image.file_name}\n"
                f"{MissingFilesDialog._dimension_text(image)} / {MissingFilesDialog._size_text(image.file_size)}\n"
                f"{image.file_path}"
            )
            info.setWordWrap(True)
            column.addWidget(preview, 1)
            column.addWidget(info)
            grid.addLayout(column, 1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class MainWindow(QMainWindow):
    def __init__(self, *, paths: AppPaths, store: MetadataStore):
        super().__init__()
        disable_qt_accessibility()
        self.paths = paths
        self.store = store
        self.initial_thumbnail_size = self._setting_int("ui.thumbnail_size", 180, 96, 320)
        self.initial_score_threshold = self._setting_int("ui.score_threshold", 0, 0, 100)
        self.initial_status_filter = self._setting_choice(
            "ui.status_filter",
            "all",
            {"all", "favorite", "unindexed", "missing"},
        )
        self.initial_tag_sort = self._setting_choice(
            "ui.tag_sort",
            "name",
            {"name", "count_desc", "count_asc"},
        )
        self.initial_tag_match_mode = self._setting_choice(
            "ui.tag_match_mode",
            "all",
            {"all", "any"},
        )
        self._pending_tag_restore_ids: set[int] | None = set(
            self._setting_int_csv("ui.selected_tag_ids")
        )
        self.thumbnailer = Thumbnailer(paths.thumbnail_dir)
        self.scanner = ImageScanner(store, self.thumbnailer)
        self.embedding_provider = JinaClipV2Provider()
        self.vector_index = VectorIndex(
            store,
            model_name=self.embedding_provider.model_name,
            model_revision=self.embedding_provider.model_revision,
            embedding_dim=self.embedding_provider.dim,
        )
        self.search_service = SearchService(
            store=store,
            embedding_provider=self.embedding_provider,
            vector_index=self.vector_index,
        )
        self.embedding_worker: EmbeddingWorker | None = None
        self.ai_vision_worker: AIVisionWorker | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.current_offset = 0
        self.page_size = 500
        self.current_sort_key = self._setting_choice(
            "ui.sort_key",
            "default",
            {
                "default",
                "score",
                "imported",
                "modified",
                "name",
                "file_size",
                "width",
                "height",
                "pixels",
                "duration",
            },
        )
        self.current_sort_desc = self._setting_choice("ui.sort_order", "desc", {"asc", "desc"}) != "asc"
        self.current_keyword_query: str | None = None
        self.current_semantic_query: str | None = None
        self.current_result_mode = "library"
        self.current_semantic_images: list[ImageItem] = []
        self.current_semantic_filtered_images: list[ImageItem] = []
        self.current_semantic_searchable_count = 0
        self.current_semantic_candidate_limit = 0
        self.current_similar_searchable_count = 0
        self.current_similar_candidate_limit = 0
        self.current_color_rgb = self._setting_color("ui.search_color", (255, 0, 0))
        self.current_color_images: list[ImageItem] = []
        self.current_color_filtered_images: list[ImageItem] = []
        self.current_color_searchable_count = 0
        self.current_color_indexed_count = 0
        self.current_color_candidate_limit = 0
        self.current_search_scope_count: int | None = None
        self.current_inspiration_project_id: int | None = None
        self.current_inspiration_terms: list[InspirationTerm] = []
        self.current_inspiration_raw_term_results: list[tuple[InspirationTerm, list[ImageItem]]] = []
        self.current_inspiration_images: list[ImageItem] = []
        self.current_inspiration_filtered_images: list[ImageItem] = []
        self.current_inspiration_matches: dict[int, list[InspirationMatch]] = {}
        self.current_inspiration_plan_filters: list[SearchPlanFilter] = []
        self.inspiration_proposal_terms: list[InspirationTerm] = []
        self.inspiration_plan_filters: list[SearchPlanFilter] = []
        self.inspiration_questions: list[str] = []
        self.inspiration_model_name = ""
        self.current_temp_project_id: int | None = None
        self.current_temp_project_images: list[ImageItem] = []
        self.current_temp_project_badges: dict[int, list[str]] = {}
        self.project_sidebar_expanded_sections: dict[str, bool] = {
            "creative": False,
            "temporary": False,
            "quick": False,
        }
        self.current_virtual_filter: str | None = None
        self.current_creative_project_id: int | None = None
        self.current_creative_node_id: int | None = None
        self._current_board_node_id: int | None = None
        self._current_board_temp_project_id: int | None = None
        self._current_board_image_ids: tuple[int, ...] = ()
        self.current_creative_node_images: list[ImageItem] = []
        self.current_creative_node_filtered_images: list[ImageItem] = []
        self.current_creative_node_searchable_count = 0
        self.current_creative_node_candidate_limit = 0
        self.current_creative_node_badges: dict[int, list[str]] = {}
        self._refreshing_creative_projects = False
        self._creative_node_undo_stack: list[dict[str, object]] = []
        self._board_removal_undo_stack: list[dict[str, object]] = []
        self._pending_board_import_node_id: int | None = None
        self._board_focus_mode = False
        self._board_focus_previous_splitter_sizes: list[int] | None = None
        self._board_focus_widget_visibility: dict[QWidget, bool] = {}
        self._board_window_pinned = False
        self.search_filters: list[SearchFilter] = []
        self.active_filter_index: int | None = None
        self.current_chain_images: list[ImageItem] = []
        self.current_chain_filtered_images: list[ImageItem] = []
        self.current_chain_result = SearchChainResult(images=[])
        self.current_chain_base_image_ids: set[int] | None = None
        self.current_chain_base_label: str | None = None
        self.current_chain_operation_mode = "replace"
        self.current_duplicate_images: list[ImageItem] = []
        self.manual_result_order_ids: list[int] | None = None
        self.result_excluded_image_ids: set[int] = set()
        self.result_excluded_collection_ids: set[int] = set()
        self.result_excluded_collection_image_ids: set[int] = set()
        self.result_exclusion_filters: list[SearchFilter] = []
        self.result_exclusion_filter_matches: dict[SearchFilter, list[ImageItem]] = {}
        self.semantic_search_revision = 0
        self.selected_image: ImageItem | None = None
        self._applying_view_payload = False
        self.current_language = self._setting_choice("ui.language", "zh", {"zh", "en"})
        self.error_log_messages: list[str] = []
        self.operation_history_messages: list[str] = []
        self._last_removal_undo: dict[str, object] | None = None
        self._macos_titlebar_applied = False
        self._database_maintenance_active = False
        self._background_threads: set[threading.Thread] = set()
        self._background_threads_lock = threading.Lock()
        self._near_duplicate_hash_records_cache: list[ImageDHashRecord] | None = None
        self._near_duplicate_hash_records_cache_count: int | None = None
        self._near_duplicate_job_counter = 0
        self._near_duplicate_callbacks: dict[
            int,
            tuple[Callable[[list[str], set[str], list[int], int], None], list[str]],
        ] = {}
        self.file_watch_enabled = self._setting_choice("ui.file_watch_enabled", "1", {"0", "1"}) == "1"
        self.file_watcher = QFileSystemWatcher(self)
        self._watch_path_roots: dict[str, str] = {}
        self._pending_watch_scan_roots: set[str] = set()
        self._watch_scan_running = False
        self._maintenance_controls_enabled: bool | None = None
        self._import_controls_enabled: bool | None = None
        self._export_controls_enabled: bool | None = None
        self.watch_scan_timer = QTimer(self)
        self.watch_scan_timer.setSingleShot(True)
        self.watch_scan_timer.timeout.connect(self._run_pending_watch_scans)

        self.setWindowTitle("Eidory")
        self._configure_native_titlebar()
        self.setStatusBar(QStatusBar())
        self._build_ui()
        self._configure_accessibility_labels()
        disable_qt_accessibility(self._record_error)
        self._apply_runtime_language_settings()
        self._connect_signals()
        startup_removed_roots, startup_removed_images = self._cleanup_missing_active_scan_roots()
        self._refresh_folders()
        self._refresh_collections()
        self.store.seed_default_ai_vision_collection_rules()
        self._refresh_temporary_projects()
        self._refresh_creative_projects()
        self._refresh_tags()
        self._refresh_saved_views()
        self._reload_images()
        self._refresh_search_operation_controls()
        self._refresh_embedding_stats()
        self._refresh_ai_vision_stats()
        self._refresh_file_watcher()
        if startup_removed_images:
            message = (
                f"启动清理：移除 {startup_removed_roots} 个已不存在导入目录，"
                f"{startup_removed_images} 张失效索引"
            )
            self._record_operation_history(message)
            self.statusBar().showMessage(message)
        QTimer.singleShot(1200, self._run_startup_self_check)

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_events)
        self.poll_timer.start(250)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_macos_titlebar()

    def _configure_native_titlebar(self) -> None:
        if sys.platform != "darwin":
            return
        self.setWindowFlag(Qt.WindowType.NoTitleBarBackgroundHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        if hasattr(self, "setUnifiedTitleAndToolBarOnMac"):
            self.setUnifiedTitleAndToolBarOnMac(True)
        QTimer.singleShot(0, self._apply_macos_titlebar)
        QTimer.singleShot(250, self._apply_macos_titlebar)

    def _apply_macos_titlebar(self) -> None:
        if self._macos_titlebar_applied or sys.platform != "darwin":
            return
        app = QApplication.instance()
        if app is None or app.platformName().lower() != "cocoa":
            return
        try:
            if not self._make_macos_titlebar_transparent():
                return
        except Exception as exc:
            self._record_error(f"macOS 标题栏颜色设置失败：{exc}")
            return
        self._macos_titlebar_applied = True
        if hasattr(self, "central_layout"):
            self.central_layout.setContentsMargins(0, 0, 0, 0)

    def _make_macos_titlebar_transparent(self) -> bool:
        import ctypes
        import ctypes.util

        objc_path = ctypes.util.find_library("objc")
        if not objc_path:
            return False
        objc = ctypes.cdll.LoadLibrary(objc_path)
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        msg_send = objc.objc_msgSend

        def cls(name: str) -> int:
            return int(objc.objc_getClass(name.encode("utf-8")) or 0)

        def sel(name: str) -> int:
            return int(objc.sel_registerName(name.encode("utf-8")) or 0)

        def send_id(receiver: int, selector: str, *args) -> int:
            if not receiver:
                return 0
            msg_send.restype = ctypes.c_void_p
            return int(msg_send(ctypes.c_void_p(receiver), ctypes.c_void_p(sel(selector)), *args) or 0)

        def send_ulong(receiver: int, selector: str) -> int:
            if not receiver:
                return 0
            msg_send.restype = ctypes.c_ulong
            return int(msg_send(ctypes.c_void_p(receiver), ctypes.c_void_p(sel(selector))) or 0)

        def send_void_bool(receiver: int, selector: str, value: bool) -> None:
            if receiver:
                msg_send.restype = None
                msg_send(ctypes.c_void_p(receiver), ctypes.c_void_p(sel(selector)), ctypes.c_bool(value))

        def send_void_ulong(receiver: int, selector: str, value: int) -> None:
            if receiver:
                msg_send.restype = None
                msg_send(ctypes.c_void_p(receiver), ctypes.c_void_p(sel(selector)), ctypes.c_ulong(value))

        def send_void_long(receiver: int, selector: str, value: int) -> None:
            if receiver:
                msg_send.restype = None
                msg_send(ctypes.c_void_p(receiver), ctypes.c_void_p(sel(selector)), ctypes.c_long(value))

        def send_void_id(receiver: int, selector: str, value: int) -> None:
            if receiver and value:
                msg_send.restype = None
                msg_send(ctypes.c_void_p(receiver), ctypes.c_void_p(sel(selector)), ctypes.c_void_p(value))

        native_view = int(self.winId())
        window = send_id(native_view, "window")
        if not window:
            return False

        full_size_content_view_mask = 1 << 15
        style_mask = send_ulong(window, "styleMask")
        send_void_ulong(window, "setStyleMask:", style_mask | full_size_content_view_mask)
        send_void_bool(window, "setTitlebarAppearsTransparent:", True)
        send_void_long(window, "setTitleVisibility:", 0)
        # A fully draggable background steals drag gestures from controls that live
        # under the transparent titlebar, including text selection and sliders.
        send_void_bool(window, "setMovableByWindowBackground:", False)

        ns_string = cls("NSString")
        ns_appearance = cls("NSAppearance")
        if ns_string and ns_appearance:
            appearance_name = send_id(
                ns_string,
                "stringWithUTF8String:",
                ctypes.c_char_p(b"NSAppearanceNameDarkAqua"),
            )
            appearance = send_id(
                ns_appearance,
                "appearanceNamed:",
                ctypes.c_void_p(appearance_name),
            )
            send_void_id(window, "setAppearance:", appearance)
        return True

    def closeEvent(self, event) -> None:
        self._save_current_board_layout_if_needed()
        self._clear_last_removal_undo(cleanup_backups=True)
        self._database_maintenance_active = True
        self._stop_file_watcher_for_maintenance()
        self._stop_index_workers_for_maintenance(timeout_seconds=2.0)
        self._wait_for_background_tasks(timeout_seconds=2.0)
        if getattr(self, "video_player", None) is not None:
            self.video_player.stop()
        if hasattr(self, "root_splitter"):
            self.store.set_setting(
                "ui.root_splitter_sizes",
                ",".join(str(size) for size in self._root_splitter_sizes_for_settings()),
            )
        size = self.size()
        self.store.set_setting("ui.window_width", str(size.width()))
        self.store.set_setting("ui.window_height", str(size.height()))
        super().closeEvent(event)

    def _start_background_task(
        self,
        target: Callable[[], None],
        *,
        name: str = "background",
        on_rejected: Callable[[], None] | None = None,
    ) -> bool:
        if self._database_maintenance_active:
            message = "数据库维护中，暂不启动新的后台任务"
            self._record_error(message)
            self.statusBar().showMessage(message)
            if on_rejected is not None:
                on_rejected()
            return False

        def run_and_unregister() -> None:
            thread = threading.current_thread()
            try:
                target()
            finally:
                with self._background_threads_lock:
                    self._background_threads.discard(thread)

        thread = threading.Thread(target=run_and_unregister, daemon=True, name=f"Eidory-{name}")
        with self._background_threads_lock:
            self._background_threads.add(thread)
        thread.start()
        return True

    def _wait_for_background_tasks(self, *, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while True:
            with self._background_threads_lock:
                threads = [thread for thread in self._background_threads if thread.is_alive()]
            if not threads:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._record_error(f"数据库维护等待后台任务超时：仍有 {len(threads)} 个任务未结束")
                return False
            for thread in threads:
                thread.join(timeout=min(0.25, remaining))

    def _stop_index_workers_for_maintenance(self, *, timeout_seconds: float) -> bool:
        workers = [
            ("语义索引", self.embedding_worker),
            ("AI 视觉识别", self.ai_vision_worker),
        ]
        for _label, worker in workers:
            if worker is not None and worker.is_alive():
                worker.stop()
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        for label, worker in workers:
            if worker is None or not worker.is_alive():
                continue
            remaining = max(0.0, deadline - time.monotonic())
            worker.join(timeout=remaining)
            if worker.is_alive():
                self._record_error(f"数据库维护等待{label} worker 停止超时")
                return False
        return True

    @contextmanager
    def _database_maintenance(
        self,
        label: str,
        *,
        restart_index_workers: bool = False,
    ) -> Iterator[None]:
        previous_state = self._database_maintenance_active
        restart_embedding = (
            restart_index_workers
            and self.embedding_worker is not None
            and self.embedding_worker.is_alive()
        )
        restart_ai_vision = (
            restart_index_workers
            and self.ai_vision_worker is not None
            and self.ai_vision_worker.is_alive()
        )
        workers_stopped = False
        self._database_maintenance_active = True
        self._stop_file_watcher_for_maintenance()
        try:
            if not self._stop_index_workers_for_maintenance(timeout_seconds=5.0):
                raise RuntimeError(f"{label}已中止：索引 worker 未能停止")
            workers_stopped = True
            if not self._wait_for_background_tasks(timeout_seconds=10.0):
                raise RuntimeError(f"{label}已中止：后台任务未能停止")
            with MetadataStore._connection_lock:
                yield
        finally:
            self._database_maintenance_active = previous_state
            if not previous_state:
                self._refresh_file_watcher()
                if workers_stopped:
                    self._restart_index_workers_after_maintenance(
                        restart_embedding=restart_embedding,
                        restart_ai_vision=restart_ai_vision,
                    )

    def _stop_file_watcher_for_maintenance(self) -> None:
        if not hasattr(self, "file_watcher"):
            return
        if hasattr(self, "watch_scan_timer"):
            self.watch_scan_timer.stop()
        self._pending_watch_scan_roots.clear()
        watched = list(self.file_watcher.directories()) + list(self.file_watcher.files())
        if watched:
            self.file_watcher.removePaths(watched)

    def _restart_index_workers_after_maintenance(
        self,
        *,
        restart_embedding: bool,
        restart_ai_vision: bool,
    ) -> None:
        if restart_embedding:
            self._start_embedding()
        if restart_ai_vision:
            self._start_ai_vision()

    def _build_ui(self) -> None:
        self.central_shell = QWidget()
        self.central_layout = QVBoxLayout(self.central_shell)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(0)
        self.root_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.root_splitter.addWidget(self._build_sidebar())
        self.root_splitter.addWidget(self._build_library_panel())
        self.root_splitter.addWidget(self._build_detail_panel())
        self.root_splitter.setCollapsible(0, True)
        self.root_splitter.setCollapsible(1, False)
        self.root_splitter.setCollapsible(2, True)
        self.root_splitter.setStretchFactor(0, 0)
        self.root_splitter.setStretchFactor(1, 1)
        self.root_splitter.setStretchFactor(2, 0)
        self._enforcing_sidebar_widths = False
        self.root_splitter.splitterMoved.connect(self._enforce_fixed_sidebar_widths)
        self.root_splitter.setSizes(self._initial_root_splitter_sizes())
        QTimer.singleShot(0, self._enforce_fixed_sidebar_widths)
        self.central_layout.addWidget(self.root_splitter)
        self.setCentralWidget(self.central_shell)

    def _initial_root_splitter_sizes(self) -> list[int]:
        saved = self._setting_int_list("ui.root_splitter_sizes", [LEFT_SIDEBAR_WIDTH, 944, RIGHT_SIDEBAR_WIDTH], 3)
        left = 0 if saved[0] <= SIDEBAR_COLLAPSE_THRESHOLD else LEFT_SIDEBAR_WIDTH
        right = 0 if saved[2] <= SIDEBAR_COLLAPSE_THRESHOLD else RIGHT_SIDEBAR_WIDTH
        center = max(640, saved[1])
        return [left, center, right]

    def _root_splitter_sizes_for_settings(self) -> list[int]:
        sizes = self.root_splitter.sizes()
        if len(sizes) != 3:
            return [LEFT_SIDEBAR_WIDTH, 944, RIGHT_SIDEBAR_WIDTH]
        left = 0 if sizes[0] <= SIDEBAR_COLLAPSE_THRESHOLD else LEFT_SIDEBAR_WIDTH
        right = 0 if sizes[2] <= SIDEBAR_COLLAPSE_THRESHOLD else RIGHT_SIDEBAR_WIDTH
        center = max(1, sum(sizes) - left - right)
        return [left, center, right]

    def _enforce_fixed_sidebar_widths(self, _pos: int | None = None, _index: int | None = None) -> None:
        if getattr(self, "_enforcing_sidebar_widths", False):
            return
        if getattr(self, "_board_focus_mode", False):
            return
        if not hasattr(self, "root_splitter"):
            return
        sizes = self.root_splitter.sizes()
        if len(sizes) != 3:
            return
        total = max(1, sum(sizes))
        collapse_left = sizes[0] <= SIDEBAR_COLLAPSE_THRESHOLD
        collapse_right = sizes[2] <= SIDEBAR_COLLAPSE_THRESHOLD
        if _index == 1 and _pos is not None:
            collapse_left = _pos <= SIDEBAR_COLLAPSE_THRESHOLD
        if _index == 2 and _pos is not None:
            collapse_right = (total - _pos) <= SIDEBAR_COLLAPSE_THRESHOLD
        left = 0 if collapse_left else LEFT_SIDEBAR_WIDTH
        right = 0 if collapse_right else RIGHT_SIDEBAR_WIDTH
        center = max(1, total - left - right)
        self._enforcing_sidebar_widths = True
        try:
            self.root_splitter.setSizes([left, center, right])
        finally:
            self._enforcing_sidebar_widths = False

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(0)
        panel.setMaximumWidth(LEFT_SIDEBAR_WIDTH)
        panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(panel)

        self.add_folder_button = QPushButton("导入图片")
        self.import_folder_tree_button = QPushButton("导入文件夹")
        self.rescan_button = QPushButton("重新扫描")
        self.add_folder_button.setMinimumWidth(0)
        self.import_folder_tree_button.setMinimumWidth(0)
        self.rescan_button.setMinimumWidth(0)
        self.rescan_button.hide()
        self.folder_tree = QTreeWidget()
        self.folder_tree.setColumnCount(2)
        self.folder_tree.setHeaderLabels(["文件夹", "张"])
        self.folder_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.folder_tree.setIndentation(14)
        self.folder_tree.setMinimumWidth(0)
        self._configure_sidebar_count_tree(self.folder_tree)
        self.folder_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.add_collection_button = QPushButton("新建文件夹")
        self.add_collection_button.setMinimumWidth(0)
        self.collection_tree = CollectionTreeWidget()
        self.collection_tree.setColumnCount(2)
        self.collection_tree.setHeaderLabels(["文件夹", "张"])
        self.collection_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.collection_tree.setIndentation(14)
        self.collection_tree.setMinimumWidth(0)
        self._configure_sidebar_count_tree(self.collection_tree)
        self.collection_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.temp_project_list = QListWidget()
        self.temp_project_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.temp_project_list.setMinimumWidth(0)
        self.temp_project_list.setMinimumHeight(220)
        self.temp_project_list.setItemDelegate(ProjectListItemDelegate(self.temp_project_list))
        self.temp_project_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.temp_project_list.setToolTip(
            "统一管理语义探针项目、暂时收藏和创作节点项目；删除项目不会影响源文件。"
        )

        self.creative_project_combo = QComboBox()
        self.creative_project_combo.setMinimumWidth(0)
        self.creative_project_combo.setToolTip("选择创作项目；项目只保存图库图片链接，不移动源文件。")
        self.creative_project_combo.hide()
        self.creative_project_list = QListWidget()
        self.creative_project_list.setMinimumHeight(150)
        self.creative_project_list.setMaximumHeight(190)
        self.creative_project_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.creative_project_list.setToolTip("选择创作项目；项目只保存图库图片链接，不移动源文件。")
        self.creative_template_combo = QComboBox()
        for template in CREATIVE_TEMPLATES:
            self.creative_template_combo.addItem(template.label, template.id)
        self.creative_new_project_button = QPushButton("按模板新建项目")
        self.creative_add_child_button = QPushButton("新建子节点")
        self.creative_delete_node_button = QPushButton("删除节点")
        for button in [
            self.creative_new_project_button,
            self.creative_add_child_button,
            self.creative_delete_node_button,
        ]:
            button.setMinimumWidth(0)
        self.creative_node_tree = QTreeWidget()
        self.creative_node_tree.setColumnCount(2)
        self.creative_node_tree.setHeaderLabels(["创作节点", "图"])
        self.creative_node_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.creative_node_tree.setIndentation(14)
        self.creative_node_tree.setMinimumWidth(0)
        self.creative_node_tree.setMinimumHeight(260)
        self.creative_node_tree.setMaximumHeight(430)
        self.creative_node_tree.setToolTip("节点用于组织创作思路和对应参考图。")
        self._configure_sidebar_count_tree(self.creative_node_tree)
        self.creative_node_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.status_filter_combo = QComboBox()
        for label, value in [
            ("全部", "all"),
            ("收藏", "favorite"),
            ("未完成语义", "unindexed"),
            ("文件丢失", "missing"),
        ]:
            self.status_filter_combo.addItem(label, value)
        self._set_combo_to_data(self.status_filter_combo, self.initial_status_filter)

        self.tag_list = QListWidget()
        self.tag_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tag_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tag_search_input = QLineEdit()
        self.tag_search_input.setPlaceholderText("搜索标签")
        self.tag_sort_combo = QComboBox()
        self.tag_sort_combo.addItem("按名称", "name")
        self.tag_sort_combo.addItem("数量最多", "count_desc")
        self.tag_sort_combo.addItem("数量最少", "count_asc")
        self._set_combo_to_data(self.tag_sort_combo, self.initial_tag_sort)
        self.tag_match_combo = QComboBox()
        self.tag_match_combo.addItem("匹配全部", "all")
        self.tag_match_combo.addItem("匹配任一", "any")
        self._set_combo_to_data(self.tag_match_combo, self.initial_tag_match_mode)
        self.rename_tag_button = QPushButton("重命名")
        self.delete_tag_button = QPushButton("删除")
        self.merge_tag_button = QPushButton("合并")

        self.start_embedding_button = QPushButton("开始索引")
        self.pause_embedding_button = QPushButton("暂停索引")
        self.retry_failed_button = QPushButton("重试失败")
        self.rescan_all_button = QPushButton("重新扫描全部导入目录")
        self.clean_orphan_thumbnails_button = QPushButton("清理孤立缩略图")
        self.embedding_progress_bar = QProgressBar()
        self.embedding_progress_bar.setRange(0, 100)
        self.embedding_progress_bar.setValue(0)
        self.embedding_stats_label = QLabel("索引：0 / 0")
        self.embedding_stats_label.setWordWrap(True)
        self.saved_view_combo = QComboBox()
        self.save_view_button = QPushButton("保存")
        self.apply_view_button = QPushButton("载入")
        self.rename_view_button = QPushButton("重命名")
        self.delete_view_button = QPushButton("删除")

        layout.addWidget(QLabel("文件夹"))
        layout.addWidget(self.add_collection_button)
        layout.addWidget(self.add_folder_button)
        layout.addWidget(self.import_folder_tree_button)
        layout.addWidget(self.collection_tree, 3)
        layout.addWidget(QLabel("项目"))
        layout.addWidget(self.temp_project_list, 2)
        return panel

    @staticmethod
    def _configure_sidebar_count_tree(tree: QTreeWidget) -> None:
        header = tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        tree.setColumnWidth(1, SIDEBAR_COUNT_COLUMN_WIDTH)

    @staticmethod
    def _configure_sidebar_form(form: QFormLayout) -> None:
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

    @staticmethod
    def _configure_board_icon_button(
        button: QPushButton,
        icon_name: str,
        accessible_name: str,
        tooltip: str,
    ) -> None:
        button.setText("")
        button.setIcon(_board_control_icon(icon_name))
        button.setIconSize(BOARD_ICON_SIZE)
        button.setFixedSize(BOARD_ICON_BUTTON_SIZE)
        button.setToolTip(tooltip)
        button.setAccessibleName(accessible_name)
        button.setAccessibleDescription(tooltip)

    def _build_library_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(0)
        panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setSpacing(4)

        search_row = QHBoxLayout()
        self.search_row = search_row
        search_row.setSpacing(0)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("文件名、标签、备注，或语义搜索文本")
        self.search_mode_group = QButtonGroup(self)
        self.search_mode_group.setExclusive(True)
        self.color_mode_button = QPushButton("颜色")
        self.color_mode_button.setCheckable(True)
        self.keyword_mode_button = QPushButton("关键词")
        self.keyword_mode_button.setCheckable(True)
        self.semantic_mode_button = QPushButton("语义")
        self.semantic_mode_button.setCheckable(True)
        self.semantic_mode_button.setChecked(True)
        self.collection_filter_button = QPushButton("文件夹")
        self.collection_filter_button.setToolTip("选择文件夹作为筛选或反向排除条件")
        self.tag_filter_button = QPushButton("标签")
        self.tag_filter_button.setToolTip("选择用户标签作为筛选或反向排除条件")
        self.similar_image_button = QPushButton("相似图")
        self.similar_image_button.setToolTip("用当前选中的图片查找相似图片")
        self.search_mode_group.addButton(self.color_mode_button)
        self.search_mode_group.addButton(self.keyword_mode_button)
        self.search_mode_group.addButton(self.semantic_mode_button)
        self.search_button = QPushButton("搜索")
        self.clear_search_button = QPushButton("清空")
        self.reverse_exclusion_button = QPushButton("反向排除")
        self.reverse_exclusion_button.setCheckable(True)
        self.reverse_exclusion_button.setToolTip("打开后，颜色/关键词/语义/标签会从当前结果中反向扣除")
        self.advanced_search_toggle_button = QPushButton("筛选/排序")
        self.advanced_search_toggle_button.setCheckable(True)
        self.advanced_search_toggle_button.setToolTip("展开搜索逻辑、元数据筛选、排序和结果管理")
        for search_action_button in [
            self.reverse_exclusion_button,
            self.color_mode_button,
            self.keyword_mode_button,
            self.semantic_mode_button,
            self.collection_filter_button,
            self.tag_filter_button,
            self.similar_image_button,
            self.search_button,
            self.clear_search_button,
            self.advanced_search_toggle_button,
        ]:
            search_action_button.setMinimumWidth(TOOL_BUTTON_MIN_WIDTH)
        self._update_color_swatch()
        search_row.addWidget(self.search_input, 1)
        search_row.addSpacing(8)
        search_row.addWidget(self.reverse_exclusion_button)
        search_row.addSpacing(16)
        search_row.addWidget(self.color_mode_button)
        search_row.addWidget(self.keyword_mode_button)
        search_row.addWidget(self.semantic_mode_button)
        search_row.addWidget(self.collection_filter_button)
        search_row.addWidget(self.tag_filter_button)
        search_row.addWidget(self.similar_image_button)
        search_row.addSpacing(16)
        search_row.addWidget(self.search_button)
        search_row.addWidget(self.clear_search_button)
        search_row.addWidget(self.advanced_search_toggle_button)

        search_operation_row = QHBoxLayout()
        search_operation_row.setContentsMargins(0, 0, 0, 0)
        search_operation_row.setSpacing(0)
        self.search_operation_group = QButtonGroup(self)
        self.search_operation_group.setExclusive(True)
        self.search_within_results_button = QPushButton("在结果中搜")
        self.search_merge_results_button = QPushButton("合并结果")
        self.search_replace_results_button = QPushButton("重新搜索")
        self.search_within_results_button.setCheckable(True)
        self.search_merge_results_button.setCheckable(True)
        self.search_replace_results_button.setCheckable(True)
        self.search_replace_results_button.setChecked(True)
        self.search_within_results_button.setToolTip("第二轮搜索时，只在当前结果中继续筛选")
        self.search_merge_results_button.setToolTip("第二轮搜索时，重新搜索当前图库/文件夹范围，并与当前结果合并")
        self.search_replace_results_button.setToolTip("清空当前搜索条件，用这次搜索替换当前结果")
        self.search_operation_label = QLabel("搜索逻辑")
        search_operation_row.addWidget(self.search_operation_label)
        search_operation_row.addSpacing(8)
        for operation_button in [
            self.search_within_results_button,
            self.search_merge_results_button,
            self.search_replace_results_button,
        ]:
            operation_button.setMinimumWidth(TOOL_BUTTON_MIN_WIDTH)
            self.search_operation_group.addButton(operation_button)
            search_operation_row.addWidget(operation_button)
        search_operation_row.addStretch(1)

        saved_view_row = QHBoxLayout()
        saved_view_row.setContentsMargins(0, 0, 0, 0)
        saved_view_row.setSpacing(6)
        saved_view_row.addWidget(QLabel("筛选预设"))
        saved_view_row.addWidget(self.saved_view_combo, 1)
        saved_view_row.addWidget(self.save_view_button)
        saved_view_row.addWidget(self.apply_view_button)
        saved_view_row.addWidget(self.rename_view_button)
        saved_view_row.addWidget(self.delete_view_button)
        saved_view_row.addSpacing(12)
        saved_view_row.addWidget(QLabel("状态"))
        self.status_filter_combo.setMinimumWidth(110)
        saved_view_row.addWidget(self.status_filter_combo)

        metadata_filter_row = QHBoxLayout()
        self.file_type_filter_combo = QComboBox()
        self.file_type_filter_combo.addItem("文件类型", None)
        for label, value in [
            ("图片", "media:image"),
            ("视频", "media:video"),
            ("JPG", "ext:.jpg"),
            ("JPEG", "ext:.jpeg"),
            ("PNG", "ext:.png"),
            ("WebP", "ext:.webp"),
            ("MP4", "ext:.mp4"),
            ("MOV", "ext:.mov"),
            ("M4V", "ext:.m4v"),
            ("AVI", "ext:.avi"),
            ("MKV", "ext:.mkv"),
            ("WebM", "ext:.webm"),
        ]:
            self.file_type_filter_combo.addItem(label, value)
        self.add_file_type_filter_button = QPushButton("添加类型")

        self.dimension_filter_combo = QComboBox()
        self.dimension_filter_combo.addItem("尺寸/方向", None)
        for label, value in [
            ("横图", "orientation:landscape"),
            ("竖图", "orientation:portrait"),
            ("正方形", "orientation:square"),
            ("大图 >= 2MP", "size:large"),
            ("小图 <= 0.5MP", "size:small"),
        ]:
            self.dimension_filter_combo.addItem(label, value)
        self.add_dimension_filter_button = QPushButton("添加尺寸")

        self.ai_vision_field_filter_combo = QComboBox()
        self.ai_vision_value_filter_combo = QComboBox()
        self.add_ai_vision_filter_button = QPushButton("添加AI筛选")
        for field, label in [
            ("scene_location", "室内外"),
            ("environment_type", "环境"),
            ("time_of_day", "时间"),
            ("weather", "天气"),
            ("shot_scale", "景别"),
            ("view_angle", "视角"),
            ("lighting", "光照"),
        ]:
            self.ai_vision_field_filter_combo.addItem(label, field)
        self._refresh_ai_vision_value_filter_combo()

        metadata_filter_row.addWidget(QLabel("元数据筛选"))
        metadata_filter_row.addWidget(self.file_type_filter_combo)
        metadata_filter_row.addWidget(self.add_file_type_filter_button)
        metadata_filter_row.addWidget(self.dimension_filter_combo)
        metadata_filter_row.addWidget(self.add_dimension_filter_button)
        metadata_filter_row.addSpacing(12)
        self.sort_combo = QComboBox()
        for label, value in [
            ("默认排序", "default"),
            ("相似度", "score"),
            ("导入时间", "imported"),
            ("修改时间", "modified"),
            ("文件名", "name"),
            ("文件大小", "file_size"),
            ("宽度", "width"),
            ("高度", "height"),
            ("像素数", "pixels"),
            ("视频时长", "duration"),
        ]:
            self.sort_combo.addItem(label, value)
        self.sort_order_combo = QComboBox()
        self.sort_order_combo.addItem("降序", "desc")
        self.sort_order_combo.addItem("升序", "asc")
        self._set_combo_to_data(self.sort_combo, self.current_sort_key)
        self._set_combo_to_data(
            self.sort_order_combo,
            "desc" if self.current_sort_desc else "asc",
        )
        metadata_filter_row.addWidget(QLabel("排序"))
        metadata_filter_row.addWidget(self.sort_combo)
        metadata_filter_row.addWidget(self.sort_order_combo)
        self.shuffle_results_button = QPushButton("打乱排序")
        self.shuffle_results_button.setToolTip("随机打乱当前显示的图片顺序，不改变筛选条件")
        self.shuffle_results_button.setMinimumWidth(TOOL_BUTTON_MIN_WIDTH)
        metadata_filter_row.addStretch(1)

        ai_scene_filter_row = QHBoxLayout()
        ai_scene_filter_row.setContentsMargins(0, 0, 0, 0)
        ai_scene_filter_row.setSpacing(6)
        ai_scene_filter_row.addWidget(QLabel("AI 场景标签"))
        ai_scene_filter_row.addWidget(self.ai_vision_field_filter_combo)
        ai_scene_filter_row.addWidget(self.ai_vision_value_filter_combo)
        ai_scene_filter_row.addWidget(self.add_ai_vision_filter_button)
        ai_scene_filter_row.addStretch(1)

        self.filter_chain_widget = QWidget()
        self.filter_chain_layout = QHBoxLayout(self.filter_chain_widget)
        self.filter_chain_layout.setContentsMargins(0, 0, 0, 0)
        self.filter_chain_label = QLabel("筛选：无")
        self.filter_chain_layout.addWidget(self.filter_chain_label)
        self.filter_chain_layout.addStretch(1)

        result_tools_row = QHBoxLayout()
        result_tools_row.setContentsMargins(0, 0, 0, 0)
        result_tools_row.setSpacing(0)
        self.save_result_set_button = QPushButton("保存结果集")
        self.save_result_set_button.setToolTip("把当前可见搜索结果整体保存为语义探针项目")
        self.save_result_set_button.setMinimumWidth(TOOL_BUTTON_MIN_WIDTH)
        self.save_result_set_button.setEnabled(False)
        result_tools_row.addWidget(QLabel("结果管理"))
        result_tools_row.addSpacing(8)
        result_tools_row.addWidget(self.save_result_set_button)
        result_tools_row.addStretch(1)

        threshold_row = QHBoxLayout()
        threshold_row.setContentsMargins(0, 0, 0, 0)
        threshold_row.setSpacing(8)
        self.score_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.score_threshold_slider.setRange(0, 100)
        self.score_threshold_slider.setTracking(False)
        self.score_threshold_slider.setValue(self.initial_score_threshold)
        self.score_threshold_label = QLabel(
            self._format_score_threshold_label(self.initial_score_threshold)
        )
        self.score_threshold_label.setMinimumWidth(112)
        self.score_threshold_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        threshold_row.addWidget(self.score_threshold_label)
        threshold_row.addWidget(self.score_threshold_slider, 1)

        self.result_state_label = QLabel("全部图库")
        self.result_state_label.setWordWrap(False)
        self.result_state_label.setFixedHeight(20)
        self.result_state_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.search_diagnostics_label = QLabel("搜索诊断：-")
        self.search_diagnostics_label.setWordWrap(False)
        self.search_diagnostics_label.setFixedHeight(20)
        self.search_diagnostics_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.search_diagnostics_label.hide()

        self.grid_view = JustifiedImageGridView(
            thumbnail_size=self.initial_thumbnail_size,
            spacing=4,
        )
        self.project_board_view = ProjectBoardView()
        self.center_result_stack = QStackedWidget()
        self.center_result_stack.addWidget(self.grid_view)
        self.center_result_stack.addWidget(self.project_board_view)

        self.load_more_button = QPushButton("加载更多")
        thumbnail_size_row = QHBoxLayout()
        self.show_gallery_button = QPushButton("图片墙")
        self.show_project_board_button = QPushButton("看板")
        self.save_project_board_layout_button = QPushButton("保存看板")
        self.board_pin_button = QPushButton()
        self.board_pin_button.setCheckable(True)
        self.board_hide_selected_button = QPushButton()
        self.board_fit_all_button = QPushButton()
        self.board_flip_button = QPushButton()
        self.board_grayscale_button = QPushButton()
        self.board_import_button = QPushButton("导入图片")
        self.board_show_all_button = QPushButton("显示全部")
        for board_button in [
            self.show_gallery_button,
            self.show_project_board_button,
            self.save_project_board_layout_button,
            self.board_import_button,
            self.board_show_all_button,
        ]:
            board_button.setMinimumWidth(TOOL_BUTTON_MIN_WIDTH)
        self._configure_board_icon_button(
            self.board_pin_button,
            "pin",
            "图钉",
            "让 Eidory 窗口置顶，避免被其他软件遮挡",
        )
        self._configure_board_icon_button(
            self.board_hide_selected_button,
            "focus",
            "隐藏界面",
            "隐藏左右侧栏和非看板控件；按 Tab 可恢复",
        )
        self._configure_board_icon_button(
            self.board_fit_all_button,
            "fit",
            "适应全部",
            "让所有看板图片适应当前画布窗口",
        )
        self._configure_board_icon_button(
            self.board_flip_button,
            "flip",
            "左右翻转",
            "左右镜像翻转选中的看板图片",
        )
        self._configure_board_icon_button(
            self.board_grayscale_button,
            "grayscale",
            "黑白",
            "把选中的看板图片切换为黑白显示",
        )
        for board_action_button in [
            self.board_pin_button,
            self.board_hide_selected_button,
            self.board_fit_all_button,
            self.board_flip_button,
            self.board_grayscale_button,
            self.board_import_button,
            self.board_show_all_button,
        ]:
            board_action_button.hide()
        self.save_project_board_layout_button.setEnabled(False)
        self.thumbnail_size_label = QLabel(f"缩略图：{self.initial_thumbnail_size}")
        self.thumbnail_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.thumbnail_size_slider.setRange(96, 320)
        self.thumbnail_size_slider.setValue(self.initial_thumbnail_size)
        self.thumbnail_size_slider.setMaximumWidth(220)
        thumbnail_size_row.addWidget(self.show_gallery_button)
        thumbnail_size_row.addWidget(self.show_project_board_button)
        thumbnail_size_row.addWidget(self.save_project_board_layout_button)
        thumbnail_size_row.addSpacing(16)
        thumbnail_size_row.addWidget(self.board_pin_button)
        thumbnail_size_row.addWidget(self.board_hide_selected_button)
        thumbnail_size_row.addWidget(self.board_fit_all_button)
        thumbnail_size_row.addWidget(self.board_flip_button)
        thumbnail_size_row.addWidget(self.board_grayscale_button)
        thumbnail_size_row.addSpacing(16)
        thumbnail_size_row.addWidget(self.board_import_button)
        thumbnail_size_row.addWidget(self.board_show_all_button)
        thumbnail_size_row.addStretch(1)
        thumbnail_size_row.addWidget(self.shuffle_results_button)
        thumbnail_size_row.addSpacing(12)
        thumbnail_size_row.addWidget(self.thumbnail_size_label)
        thumbnail_size_row.addWidget(self.thumbnail_size_slider)

        self.advanced_search_widget = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_search_widget)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(4)
        advanced_layout.addLayout(search_operation_row)
        advanced_layout.addLayout(saved_view_row)
        advanced_layout.addLayout(metadata_filter_row)
        advanced_layout.addLayout(ai_scene_filter_row)
        advanced_layout.addLayout(result_tools_row)
        self.advanced_search_widget.hide()

        compact_status_row = QHBoxLayout()
        compact_status_row.setContentsMargins(0, 0, 0, 0)
        compact_status_row.setSpacing(12)
        compact_status_row.addWidget(self.result_state_label, 1)
        compact_status_row.addWidget(self.search_diagnostics_label)

        layout.addLayout(search_row)
        layout.addWidget(self.advanced_search_widget)
        layout.addWidget(self.filter_chain_widget)
        layout.addLayout(threshold_row)
        layout.addLayout(compact_status_row)
        layout.addWidget(self.center_result_stack, 1)
        layout.addWidget(self.load_more_button)
        layout.addLayout(thumbnail_size_row)
        return panel

    def _build_detail_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(0)
        panel.setMaximumWidth(RIGHT_SIDEBAR_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.preview_label = QLabel("未选择图片")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(180)
        self.preview_label.setStyleSheet("background:#2d3138;color:#d8dee9;")
        self.video_widget: QVideoWidget | None = None
        self.video_player: QMediaPlayer | None = None
        self.video_audio_output: QAudioOutput | None = None

        self.preview_stack = QStackedWidget()
        self.preview_stack.addWidget(self.preview_label)

        self.file_name_input = QLineEdit()
        self.file_name_input.setPlaceholderText("文件名")
        self.path_label = QTextEdit()
        self.path_label.setReadOnly(True)
        self.path_label.setAcceptRichText(False)
        self.path_label.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        path_wrap = QTextOption()
        path_wrap.setWrapMode(QTextOption.WrapMode.WrapAnywhere)
        self.path_label.document().setDefaultTextOption(path_wrap)
        self.path_label.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.path_label.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setStyleSheet("background: transparent; border: 0; padding: 0;")
        self.path_label.setFixedHeight(34)
        self.size_label = QLabel("-")
        self.modified_label = QLabel("-")
        self.embedding_label = QLabel("-")
        self.score_label = QLabel("-")
        self.ai_vision_detail_label = QLabel("-")
        self.ai_vision_detail_label.setWordWrap(True)
        self.ai_vision_detail_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.image_collections_label = QLabel("-")
        self.image_collections_label.setWordWrap(True)
        self.image_collections_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.feedback_group = QButtonGroup(self)
        self.feedback_group.setExclusive(True)
        self.feedback_relevant_button = QPushButton("相关")
        self.feedback_irrelevant_button = QPushButton("不相关")
        self.feedback_ignored_button = QPushButton("忽略")
        self.feedback_buttons = {
            "relevant": self.feedback_relevant_button,
            "irrelevant": self.feedback_irrelevant_button,
            "ignored": self.feedback_ignored_button,
        }
        self.feedback_widget = QWidget(panel)
        self.feedback_widget.hide()
        feedback_layout = QHBoxLayout(self.feedback_widget)
        feedback_layout.setContentsMargins(0, 0, 0, 0)
        for button in self.feedback_buttons.values():
            button.setCheckable(True)
            button.setEnabled(False)
            button.setToolTip("只记录这次语义搜索的反馈，不改变当前排序。")
            self.feedback_group.addButton(button)
            feedback_layout.addWidget(button)
        self.favorite_checkbox = QCheckBox("收藏")
        self.tag_completion_model = QStringListModel(self)
        self.tags_display = QTextEdit()
        self.tags_display.setReadOnly(True)
        self.tags_display.setAcceptRichText(False)
        self.tags_display.setPlaceholderText("无标签")
        self.tags_display.setMinimumHeight(58)
        self.tags_display.setMaximumHeight(86)
        self.tags_display.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.tags_display.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.tags_display.setStyleSheet("QTextEdit { background: transparent; border: 0; padding: 0; }")

        self.batch_tag_summary_label = QLabel("-")
        self.batch_tag_summary_label.setWordWrap(True)
        self.batch_tags_input = QLineEdit()
        self.batch_tags_input.setPlaceholderText("给选中项添加标签，逗号分隔")
        self.batch_tags_input.setCompleter(self._make_tag_completer())
        self.batch_add_tags_button = QPushButton("添加到选中")
        self.batch_remove_tag_combo = QComboBox()
        self.batch_remove_tag_button = QPushButton("移除标签")
        self.batch_clear_tags_button = QPushButton("清空选中")
        self.batch_tags_widget = QWidget()
        batch_tags_layout = QVBoxLayout(self.batch_tags_widget)
        batch_tags_layout.setContentsMargins(0, 0, 0, 0)
        batch_add_row = QHBoxLayout()
        batch_add_row.setContentsMargins(0, 0, 0, 0)
        batch_add_row.addWidget(self.batch_tags_input, 1)
        batch_add_row.addWidget(self.batch_add_tags_button)
        batch_remove_row = QHBoxLayout()
        batch_remove_row.setContentsMargins(0, 0, 0, 0)
        batch_remove_row.addWidget(self.batch_remove_tag_combo, 1)
        batch_remove_row.addWidget(self.batch_remove_tag_button)
        batch_remove_row.addWidget(self.batch_clear_tags_button)
        batch_tags_layout.addWidget(self.batch_tag_summary_label)
        batch_tags_layout.addLayout(batch_add_row)
        batch_tags_layout.addLayout(batch_remove_row)

        self.creative_selection_summary_label = QLabel("-")
        self.creative_selection_summary_label.setWordWrap(True)
        self.creative_selection_add_button = QPushButton("存入当前节点")
        self.creative_selection_remove_button = QPushButton("移出当前节点")
        self.creative_selection_widget = QWidget()
        creative_selection_layout = QVBoxLayout(self.creative_selection_widget)
        creative_selection_layout.setContentsMargins(0, 0, 0, 0)
        creative_selection_layout.setSpacing(6)
        creative_selection_actions = QHBoxLayout()
        creative_selection_actions.setContentsMargins(0, 0, 0, 0)
        creative_selection_actions.addWidget(self.creative_selection_add_button)
        creative_selection_actions.addWidget(self.creative_selection_remove_button)
        creative_selection_layout.addWidget(self.creative_selection_summary_label)
        creative_selection_layout.addLayout(creative_selection_actions)

        self.tag_panel_selection_label = QLabel("先在图片墙选择 1 张或多张图片。")
        self.tag_panel_selection_label.setWordWrap(True)
        self.tag_panel_input = QTextEdit()
        self.tag_panel_input.setPlaceholderText("给选中图片添加标签，每行一个")
        self.tag_panel_input.setAcceptRichText(False)
        self.tag_panel_input.setTabChangesFocus(True)
        self.tag_panel_input.setMinimumHeight(150)
        self.tag_panel_input.setMaximumHeight(180)
        self.tag_panel_input.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.tag_panel_add_button = QPushButton("添加到选中")
        self.tag_panel_remove_combo = QComboBox()
        self.tag_panel_remove_button = QPushButton("移除标签")
        self.tag_panel_clear_button = QPushButton("清空标签")
        self.note_input = QTextEdit()
        self.note_input.setPlaceholderText("备注")
        self.note_input.setAcceptRichText(False)
        self.rename_file_button = QPushButton("重命名")
        self.rename_file_button.setToolTip("重命名硬盘上的源文件")
        self.note_auto_save_timer = QTimer(self)
        self.note_auto_save_timer.setSingleShot(True)
        self.note_auto_save_timer.setInterval(700)
        self._suppress_detail_auto_save = False
        self._pending_note_image_id: int | None = None
        self._pending_note_text: str | None = None
        self.delete_source_button = QPushButton("删除/移除图片")
        self.delete_source_button.setToolTip("选择删除源文件，或只从 Eidory 移除索引")
        self.play_pause_button = QPushButton("播放", panel)
        self.open_original_button = QPushButton("打开源文件", panel)
        self.reveal_in_finder_button = QPushButton("Finder 中显示", panel)
        self.copy_path_button = QPushButton("复制路径", panel)
        for hidden_action_button in [
            self.play_pause_button,
            self.open_original_button,
            self.reveal_in_finder_button,
            self.copy_path_button,
        ]:
            hidden_action_button.hide()

        self.creative_node_status_label = QLabel("未选择创作项目。")
        self.creative_node_status_label.setWordWrap(True)
        self.creative_node_note_input = QTextEdit()
        self.creative_node_note_input.setPlaceholderText("当前节点说明")
        self.creative_node_note_input.setAcceptRichText(False)
        self.creative_node_note_input.setMinimumHeight(118)
        self.creative_node_note_input.setMaximumHeight(170)
        self.creative_node_query_input = QLineEdit()
        self.creative_node_query_input.setPlaceholderText("当前节点默认语义搜索语句")
        self.save_creative_node_button = QPushButton("保存节点")
        self.save_creative_node_button.hide()
        self.generate_creative_children_button = QPushButton("节点信息AI补全")
        self.generate_creative_copy_button = QPushButton("生成文案")
        self.generate_creative_copy_tab_button = QPushButton("生成文案")
        self.creative_project_copy_input = QTextEdit()
        self.creative_project_copy_input.setPlaceholderText("生成文案后会显示在这里，也可以手动修改。")
        self.creative_project_copy_input.setAcceptRichText(False)
        self.creative_project_copy_input.setMinimumHeight(220)
        self.search_creative_node_button = QPushButton("搜索当前节点")
        self.save_selection_to_creative_node_button = QPushButton("存入当前节点")
        self.open_creative_board_button = QPushButton("看板")
        for creative_button in [
            self.save_creative_node_button,
            self.generate_creative_children_button,
            self.generate_creative_copy_button,
            self.generate_creative_copy_tab_button,
            self.search_creative_node_button,
            self.save_selection_to_creative_node_button,
            self.open_creative_board_button,
        ]:
            creative_button.setEnabled(False)

        self.inspiration_brief_input = QTextEdit()
        self.inspiration_brief_input.setPlaceholderText("用一句话描述画面的创作主题")
        self.inspiration_brief_input.setAcceptRichText(False)
        self.inspiration_brief_input.setFixedHeight(34)
        self.inspiration_answers_input = QTextEdit()
        self.inspiration_answers_input.setPlaceholderText("补充信息：时代、天气、光源、画面气质等，可留空")
        self.inspiration_answers_input.setAcceptRichText(False)
        self.inspiration_answers_input.setFixedHeight(60)
        self.inspiration_questions_label = QLabel("AI 追问：-")
        self.inspiration_questions_label.setWordWrap(True)
        self.inspiration_history_list = QListWidget()
        self.inspiration_history_list.setMaximumHeight(116)
        self.inspiration_history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.inspiration_term_list = QListWidget()
        self.inspiration_term_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.inspiration_term_list.setMinimumHeight(360)
        self.inspiration_filter_list = QListWidget()
        self.inspiration_filter_list.setMaximumHeight(130)
        self.inspiration_status_label = QLabel("生成后最多选择 7 个语义探针。")
        self.inspiration_status_label.setWordWrap(True)
        self.use_ai_scene_tags_checkbox = QCheckBox("生成探针时同时生成 AI 场景标签")
        self.use_ai_scene_tags_checkbox.setToolTip(
            "勾选后，生成语义探针时会同时让模型给出室内外、时间、天气、光照等场景标签，并在保存搜索时参与筛选。"
        )
        self.generate_inspiration_button = QPushButton("生成语义探针")
        self.search_inspiration_button = QPushButton("保存并搜索")
        self.search_inspiration_button.setEnabled(False)
        self.save_temp_project_button = QPushButton("存为语义探针项目")
        self.save_temp_project_button.setEnabled(False)
        self.ai_project_mode_button = QPushButton("创作节点")
        self.ai_probe_mode_button = QPushButton("语义探针")
        self.ai_project_mode_button.setCheckable(True)
        self.ai_probe_mode_button.setCheckable(True)
        self.ai_workflow_mode_group = QButtonGroup(self)
        self.ai_workflow_mode_group.setExclusive(True)
        self.ai_workflow_mode_group.addButton(self.ai_project_mode_button, 0)
        self.ai_workflow_mode_group.addButton(self.ai_probe_mode_button, 1)
        self.ai_project_mode_button.setChecked(True)

        self.ai_vision_progress_bar = QProgressBar()
        self.ai_vision_progress_bar.setRange(0, 100)
        self.ai_vision_progress_bar.setValue(0)
        self.ai_vision_stats_label = QLabel("AI 场景标签：0 / 0")
        self.ai_vision_stats_label.setWordWrap(True)
        self.ai_vision_virtual_filter_list = QListWidget()
        self.ai_vision_virtual_filter_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.ai_vision_virtual_filter_list.setMaximumHeight(34)
        self.ai_vision_virtual_filter_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ai_vision_virtual_filter_list.setToolTip("按 AI 场景标签状态筛选图库。")
        self.ai_vision_rule_tree = QTreeWidget()
        self.ai_vision_rule_tree.setColumnCount(6)
        self.ai_vision_rule_tree.setHeaderLabels(["规则", "文件夹", "完成", "失败", "未处理", "总数"])
        self.ai_vision_rule_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.ai_vision_rule_tree.setMaximumHeight(180)
        self.add_ai_vision_include_rule_button = QPushButton("识别选中文件夹")
        self.add_ai_vision_exclude_rule_button = QPushButton("排除选中文件夹")
        self.remove_ai_vision_rule_button = QPushButton("移除规则")
        self.start_ai_vision_button = QPushButton("开始AI识别")
        self.pause_ai_vision_button = QPushButton("暂停AI识别")
        self.retry_failed_ai_vision_button = QPushButton("重试AI失败")
        self.refresh_ai_vision_button = QPushButton("刷新AI统计")

        form = QFormLayout()
        self.detail_form = form
        self._configure_sidebar_form(form)
        file_name_row = QHBoxLayout()
        file_name_row.setContentsMargins(0, 0, 0, 0)
        file_name_row.setSpacing(6)
        file_name_row.addWidget(self.file_name_input, 1)
        file_name_row.addWidget(self.rename_file_button)
        form.addRow("文件名", file_name_row)
        form.addRow("路径", self.path_label)
        form.addRow("文件夹", self.image_collections_label)
        form.addRow("尺寸", self.size_label)
        form.addRow("修改时间", self.modified_label)
        form.addRow("索引状态", self.embedding_label)
        form.addRow("AI 标签", self.ai_vision_detail_label)
        form.addRow("相似度", self.score_label)
        form.addRow("", self.favorite_checkbox)
        form.addRow("创作节点", self.creative_selection_widget)
        form.addRow("批量标签", self.batch_tags_widget)
        self.creative_selection_widget.hide()
        self.batch_tags_widget.hide()

        detail_tab = QWidget()
        detail_layout = QVBoxLayout(detail_tab)
        detail_layout.setContentsMargins(6, 6, 6, 6)
        detail_layout.setSpacing(6)
        detail_tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.note_input.setMaximumHeight(96)
        self.note_input.setMinimumHeight(76)

        self.image_detail_widget = QWidget()
        image_detail_layout = QVBoxLayout(self.image_detail_widget)
        image_detail_layout.setContentsMargins(0, 0, 0, 0)
        image_detail_layout.setSpacing(6)
        image_detail_layout.addWidget(self.preview_stack)
        image_detail_layout.addLayout(form)
        image_detail_layout.addWidget(QLabel("标签"))
        image_detail_layout.addWidget(self.tags_display)
        image_detail_layout.addWidget(QLabel("备注"))
        image_detail_layout.addWidget(self.note_input)
        image_detail_layout.addWidget(self.delete_source_button)
        image_detail_layout.addStretch(1)

        self.collection_detail_widget = QWidget()
        collection_detail_layout = QVBoxLayout(self.collection_detail_widget)
        collection_detail_layout.setContentsMargins(0, 0, 0, 0)
        collection_detail_layout.setSpacing(8)
        collection_form = QFormLayout()
        self._configure_sidebar_form(collection_form)
        self.collection_detail_name_label = QLabel("-")
        self.collection_detail_name_label.setWordWrap(True)
        self.collection_detail_path_label = QLabel("-")
        self.collection_detail_path_label.setWordWrap(True)
        self.collection_detail_count_label = QLabel("-")
        self.collection_detail_import_dir_label = QLabel("-")
        self.collection_detail_import_dir_label.setWordWrap(True)
        self.collection_detail_import_dir_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.collection_detail_help_label = QLabel(
            "当前没有选中图片。选择图片后，这里会显示路径、标签、备注、AI 标签和删除/移除操作。"
        )
        self.collection_detail_help_label.setWordWrap(True)
        collection_form.addRow("文件夹", self.collection_detail_name_label)
        collection_form.addRow("层级", self.collection_detail_path_label)
        collection_form.addRow("图片/视频", self.collection_detail_count_label)
        collection_form.addRow("保存位置", self.collection_detail_import_dir_label)
        self.open_collection_import_dir_button = QPushButton("打开保存位置")
        collection_detail_layout.addLayout(collection_form)
        collection_detail_layout.addWidget(self.collection_detail_help_label)
        collection_detail_layout.addWidget(self.open_collection_import_dir_button)
        collection_detail_layout.addStretch(1)
        self.collection_detail_widget.hide()

        detail_layout.addWidget(self.image_detail_widget, 1)
        detail_layout.addWidget(self.collection_detail_widget, 1)

        inspiration_tab = QWidget()
        inspiration_layout = QVBoxLayout(inspiration_tab)
        inspiration_layout.setContentsMargins(8, 8, 8, 8)
        inspiration_layout.setSpacing(8)
        inspiration_layout.addWidget(QLabel("创作主题"))
        inspiration_layout.addWidget(self.inspiration_brief_input)
        inspiration_layout.addWidget(QLabel("补充信息"))
        inspiration_layout.addWidget(self.inspiration_answers_input)
        ai_mode_row = QHBoxLayout()
        ai_mode_row.setContentsMargins(0, 0, 0, 0)
        ai_mode_row.setSpacing(0)
        ai_mode_row.addWidget(self.ai_project_mode_button)
        ai_mode_row.addWidget(self.ai_probe_mode_button)
        inspiration_layout.addLayout(ai_mode_row)

        self.ai_workflow_stack = QStackedWidget()
        project_mode_widget = QWidget()
        project_mode_layout = QVBoxLayout(project_mode_widget)
        project_mode_layout.setContentsMargins(0, 0, 0, 0)
        project_mode_layout.setSpacing(7)
        project_mode_layout.addWidget(QLabel("绘画类型模板"))
        project_mode_layout.addWidget(self.creative_template_combo)
        project_mode_layout.addSpacing(4)
        project_mode_layout.addWidget(self.creative_new_project_button)
        project_mode_layout.addSpacing(8)
        self.creative_project_list.hide()
        project_mode_layout.addWidget(self.creative_project_combo)

        self.creative_content_tabs = QTabWidget()
        self.creative_content_tabs.setObjectName("creativeContentTabs")
        self.creative_content_tabs.setTabBar(EqualWidthTabBar())
        node_tab = QWidget()
        node_tab_layout = QVBoxLayout(node_tab)
        node_tab_layout.setContentsMargins(0, 0, 0, 0)
        node_tab_layout.setSpacing(7)
        node_tab_layout.addWidget(self.creative_node_tree)
        node_tab_layout.addWidget(self.creative_node_status_label)
        node_tab_layout.addWidget(QLabel("当前节点说明"))
        node_tab_layout.addWidget(self.creative_node_note_input)
        node_tab_layout.addWidget(self.creative_node_query_input)
        creative_node_button_row = QHBoxLayout()
        creative_node_button_row.setContentsMargins(0, 0, 0, 0)
        creative_node_button_row.addWidget(self.generate_creative_children_button)
        node_tab_layout.addLayout(creative_node_button_row)
        creative_node_action_row = QHBoxLayout()
        creative_node_action_row.setContentsMargins(0, 0, 0, 0)
        creative_node_action_row.addWidget(self.search_creative_node_button)
        creative_node_action_row.addWidget(self.save_selection_to_creative_node_button)
        node_tab_layout.addLayout(creative_node_action_row)
        node_tab_layout.addWidget(self.open_creative_board_button)
        copy_tab = QWidget()
        copy_tab_layout = QVBoxLayout(copy_tab)
        copy_tab_layout.setContentsMargins(0, 0, 0, 0)
        copy_tab_layout.setSpacing(7)
        copy_tab_layout.addWidget(QLabel("项目文案"))
        copy_tab_layout.addWidget(self.creative_project_copy_input, 1)
        copy_tab_layout.addWidget(self.generate_creative_copy_tab_button)
        self.creative_content_tabs.addTab(node_tab, "节点树")
        self.creative_content_tabs.addTab(copy_tab, "文案")
        project_mode_layout.addWidget(self.creative_content_tabs, 1)
        project_mode_layout.addStretch(1)

        probe_mode_widget = QWidget()
        probe_mode_layout = QVBoxLayout(probe_mode_widget)
        probe_mode_layout.setContentsMargins(0, 0, 0, 0)
        probe_mode_layout.setSpacing(7)
        probe_mode_layout.addWidget(self.inspiration_questions_label)
        probe_mode_layout.addWidget(QLabel("历史探针"))
        probe_mode_layout.addWidget(self.inspiration_history_list)
        probe_mode_layout.addWidget(QLabel("语义探针"))
        probe_mode_layout.addWidget(self.inspiration_term_list, 1)
        probe_mode_layout.addWidget(self.use_ai_scene_tags_checkbox)
        probe_mode_layout.addWidget(QLabel("AI 场景标签"))
        probe_mode_layout.addWidget(self.inspiration_filter_list)
        probe_mode_layout.addWidget(self.inspiration_status_label)
        inspiration_button_row = QHBoxLayout()
        inspiration_button_row.setContentsMargins(0, 0, 0, 0)
        inspiration_button_row.addWidget(self.generate_inspiration_button)
        inspiration_button_row.addWidget(self.search_inspiration_button)
        probe_mode_layout.addLayout(inspiration_button_row)
        probe_mode_layout.addWidget(self.save_temp_project_button)

        self.ai_workflow_stack.addWidget(project_mode_widget)
        self.ai_workflow_stack.addWidget(probe_mode_widget)
        inspiration_layout.addWidget(self.ai_workflow_stack, 1)

        filter_tab = QWidget()
        filter_layout = QVBoxLayout(filter_tab)
        filter_layout.setContentsMargins(6, 6, 6, 6)
        filter_layout.setSpacing(6)
        filter_layout.addWidget(QLabel("给选中图片打标签"))
        filter_layout.addWidget(self.tag_panel_selection_label)
        tag_panel_add_row = QHBoxLayout()
        tag_panel_add_row.setContentsMargins(0, 0, 0, 0)
        filter_layout.addWidget(self.tag_panel_input)
        tag_panel_add_row.addStretch(1)
        tag_panel_add_row.addWidget(self.tag_panel_add_button)
        filter_layout.addLayout(tag_panel_add_row)
        tag_panel_remove_row = QHBoxLayout()
        tag_panel_remove_row.setContentsMargins(0, 0, 0, 0)
        tag_panel_remove_row.addWidget(self.tag_panel_remove_combo, 1)
        tag_panel_remove_row.addWidget(self.tag_panel_remove_button)
        tag_panel_remove_row.addWidget(self.tag_panel_clear_button)
        filter_layout.addLayout(tag_panel_remove_row)
        filter_layout.addSpacing(8)
        filter_layout.addWidget(QLabel("标签管理"))
        tag_tools_row = QHBoxLayout()
        tag_tools_row.setContentsMargins(0, 0, 0, 0)
        tag_tools_row.addWidget(self.tag_search_input, 1)
        tag_tools_row.addWidget(self.tag_sort_combo)
        filter_layout.addLayout(tag_tools_row)
        filter_layout.addWidget(self.tag_list, 1)
        tag_action_row = QHBoxLayout()
        tag_action_row.setContentsMargins(0, 0, 0, 0)
        tag_action_row.addWidget(self.rename_tag_button)
        tag_action_row.addWidget(self.delete_tag_button)
        tag_action_row.addWidget(self.merge_tag_button)
        filter_layout.addLayout(tag_action_row)

        index_tab = QWidget()
        index_layout = QVBoxLayout(index_tab)
        index_layout.setContentsMargins(6, 6, 6, 6)
        index_layout.setSpacing(6)
        index_layout.addWidget(QLabel("语义 embedding"))
        index_layout.addWidget(self.embedding_progress_bar)
        index_layout.addWidget(self.embedding_stats_label)
        index_layout.addWidget(self.start_embedding_button)
        index_layout.addWidget(self.pause_embedding_button)
        index_layout.addWidget(self.retry_failed_button)
        index_layout.addSpacing(8)
        index_layout.addWidget(QLabel("AI 场景视觉标签"))
        index_layout.addWidget(self.ai_vision_progress_bar)
        index_layout.addWidget(self.ai_vision_stats_label)
        index_layout.addWidget(self.ai_vision_virtual_filter_list)
        index_layout.addWidget(self.ai_vision_rule_tree)
        ai_rule_row = QHBoxLayout()
        ai_rule_row.setContentsMargins(0, 0, 0, 0)
        ai_rule_row.addWidget(self.add_ai_vision_include_rule_button)
        ai_rule_row.addWidget(self.add_ai_vision_exclude_rule_button)
        index_layout.addLayout(ai_rule_row)
        ai_rule_action_row = QHBoxLayout()
        ai_rule_action_row.setContentsMargins(0, 0, 0, 0)
        ai_rule_action_row.addWidget(self.remove_ai_vision_rule_button)
        ai_rule_action_row.addWidget(self.refresh_ai_vision_button)
        index_layout.addLayout(ai_rule_action_row)
        ai_worker_row = QHBoxLayout()
        ai_worker_row.setContentsMargins(0, 0, 0, 0)
        ai_worker_row.addWidget(self.start_ai_vision_button)
        ai_worker_row.addWidget(self.pause_ai_vision_button)
        ai_worker_row.addWidget(self.retry_failed_ai_vision_button)
        index_layout.addLayout(ai_worker_row)
        index_layout.addStretch(1)

        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.setSpacing(8)
        settings_layout.addWidget(QLabel("文本模型设置"))
        settings_form = QFormLayout()
        self._configure_sidebar_form(settings_form)
        self.llm_service_combo = QComboBox()
        for service_key, service_label in LLM_SERVICE_OPTIONS:
            self.llm_service_combo.addItem(service_label, service_key)
        self.llm_endpoint_input = QLineEdit()
        self.llm_model_input = QLineEdit()
        self.llm_api_key_input = QLineEdit()
        self.llm_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.llm_temperature_spin = QDoubleSpinBox()
        self.llm_temperature_spin.setRange(0.0, 2.0)
        self.llm_temperature_spin.setSingleStep(0.1)
        self.llm_temperature_spin.setDecimals(2)
        self.language_combo = QComboBox()
        self.language_combo.addItem("中文", "zh")
        self.language_combo.addItem("English", "en")
        self.data_dir_label = QLabel(str(self.paths.data_dir))
        self.data_dir_label.setWordWrap(True)
        self.database_path_label = QLabel(str(self.paths.database_path))
        self.database_path_label.setWordWrap(True)
        settings_form.addRow("模型服务", self.llm_service_combo)
        settings_form.addRow("Endpoint", self.llm_endpoint_input)
        settings_form.addRow("Model", self.llm_model_input)
        settings_form.addRow("API Key", self.llm_api_key_input)
        settings_form.addRow("温度", self.llm_temperature_spin)
        settings_form.addRow("界面语言", self.language_combo)
        settings_form.addRow("数据目录", self.data_dir_label)
        settings_form.addRow("数据库", self.database_path_label)
        settings_layout.addLayout(settings_form)
        self.save_settings_button = QPushButton("保存设置")
        self.open_data_dir_button = QPushButton("打开数据目录")
        self.backup_database_button = QPushButton("备份数据库")
        self.restore_database_button = QPushButton("恢复数据库")
        self.run_self_check_button = QPushButton("运行启动自检")
        self.show_error_log_button = QPushButton("查看错误日志")
        self.show_operation_history_button = QPushButton("操作历史")
        self.file_watch_checkbox = QCheckBox("自动监听文件变化")
        self.file_watch_checkbox.setChecked(self.file_watch_enabled)
        self.show_missing_files_button = QPushButton("查看/修复丢失文件")
        self.rescan_new_button = QPushButton("扫描新增/变化")
        self.rescan_missing_button = QPushButton("扫描缺失所在目录")
        self.clean_missing_index_button = QPushButton("清理丢失索引")
        self.detect_duplicates_button = QPushButton("检测重复/近重复")
        self.rebuild_selected_thumbnails_button = QPushButton("重建选中缩略图")
        self.remove_selected_index_button = QPushButton("移除选中索引")
        self.rebuild_selected_thumbnails_button.setEnabled(False)
        self.remove_selected_index_button.setEnabled(False)
        self.run_performance_check_button = QPushButton("性能压测")
        self.more_maintenance_button = QPushButton("更多维护")
        self.more_maintenance_button.setCheckable(True)
        self.export_library_button = QPushButton("导出图库")
        self.export_selection_button = QPushButton("导出图片")
        self.export_selection_button.setEnabled(False)
        self.path_remap_old_combo = QComboBox()
        self.path_remap_old_combo.setEditable(True)
        self.path_remap_new_input = QLineEdit()
        self.path_remap_new_input.setPlaceholderText("选择移动后的新根目录")
        self.refresh_path_candidates_button = QPushButton("刷新旧位置")
        self.choose_remap_new_path_button = QPushButton("选择新位置")
        self.apply_path_remap_button = QPushButton("应用路径重映射")
        self.settings_status_label = QLabel(
            "LM Studio 默认使用 http://localhost:1234/v1；Ollama 默认使用 http://localhost:11434/v1；API Key 不会写入日志。"
        )
        self.settings_status_label.setWordWrap(True)
        settings_layout.addWidget(self.save_settings_button)
        settings_layout.addSpacing(4)
        settings_layout.addWidget(QLabel("路径修复"))
        path_remap_form = QFormLayout()
        self._configure_sidebar_form(path_remap_form)
        path_remap_form.addRow("旧位置", self.path_remap_old_combo)
        path_remap_form.addRow("新位置", self.path_remap_new_input)
        settings_layout.addLayout(path_remap_form)
        path_remap_actions_row = QHBoxLayout()
        path_remap_actions_row.setContentsMargins(0, 0, 0, 0)
        path_remap_actions_row.addWidget(self.refresh_path_candidates_button)
        path_remap_actions_row.addWidget(self.choose_remap_new_path_button)
        settings_layout.addLayout(path_remap_actions_row)
        settings_layout.addWidget(self.apply_path_remap_button)
        settings_layout.addSpacing(4)
        settings_layout.addWidget(QLabel("图库维护"))
        settings_layout.addWidget(self.file_watch_checkbox)
        maintenance_primary_row = QHBoxLayout()
        maintenance_primary_row.setContentsMargins(0, 0, 0, 0)
        maintenance_primary_row.addWidget(self.rescan_new_button)
        maintenance_primary_row.addWidget(self.show_missing_files_button)
        settings_layout.addLayout(maintenance_primary_row)
        maintenance_secondary_row = QHBoxLayout()
        maintenance_secondary_row.setContentsMargins(0, 0, 0, 0)
        maintenance_secondary_row.addWidget(self.detect_duplicates_button)
        maintenance_secondary_row.addWidget(self.more_maintenance_button)
        settings_layout.addLayout(maintenance_secondary_row)

        self.more_maintenance_widget = QWidget()
        more_maintenance_layout = QVBoxLayout(self.more_maintenance_widget)
        more_maintenance_layout.setContentsMargins(0, 0, 0, 0)
        more_maintenance_layout.setSpacing(6)
        more_maintenance_help = QLabel("低频维护操作。遇到导入异常、文件迁移、缩略图异常或性能排查时再使用。")
        more_maintenance_help.setWordWrap(True)
        more_maintenance_layout.addWidget(more_maintenance_help)
        maintenance_scan_row = QHBoxLayout()
        maintenance_scan_row.setContentsMargins(0, 0, 0, 0)
        maintenance_scan_row.addWidget(self.rescan_all_button)
        maintenance_scan_row.addWidget(self.rescan_missing_button)
        more_maintenance_layout.addLayout(maintenance_scan_row)
        maintenance_clean_row = QHBoxLayout()
        maintenance_clean_row.setContentsMargins(0, 0, 0, 0)
        maintenance_clean_row.addWidget(self.clean_missing_index_button)
        maintenance_clean_row.addWidget(self.clean_orphan_thumbnails_button)
        more_maintenance_layout.addLayout(maintenance_clean_row)
        maintenance_selected_row = QHBoxLayout()
        maintenance_selected_row.setContentsMargins(0, 0, 0, 0)
        maintenance_selected_row.addWidget(self.rebuild_selected_thumbnails_button)
        maintenance_selected_row.addWidget(self.remove_selected_index_button)
        more_maintenance_layout.addLayout(maintenance_selected_row)
        more_maintenance_layout.addWidget(self.run_performance_check_button)
        self.more_maintenance_widget.hide()
        settings_layout.addWidget(self.more_maintenance_widget)
        settings_layout.addSpacing(4)
        settings_layout.addWidget(QLabel("导出"))
        export_row = QHBoxLayout()
        export_row.setContentsMargins(0, 0, 0, 0)
        export_row.addWidget(self.export_library_button)
        export_row.addWidget(self.export_selection_button)
        settings_layout.addLayout(export_row)
        settings_layout.addSpacing(4)
        settings_layout.addWidget(QLabel("数据库维护"))
        settings_actions_row = QHBoxLayout()
        settings_actions_row.setContentsMargins(0, 0, 0, 0)
        settings_actions_row.addWidget(self.open_data_dir_button)
        settings_actions_row.addWidget(self.backup_database_button)
        settings_actions_row.addWidget(self.restore_database_button)
        settings_layout.addLayout(settings_actions_row)
        settings_health_row = QHBoxLayout()
        settings_health_row.setContentsMargins(0, 0, 0, 0)
        settings_health_row.addWidget(self.run_self_check_button)
        settings_health_row.addWidget(self.show_error_log_button)
        settings_layout.addLayout(settings_health_row)
        settings_layout.addWidget(self.show_operation_history_button)
        settings_layout.addWidget(self.settings_status_label)
        settings_layout.addStretch(1)
        self._load_settings_controls()
        self._refresh_inspiration_history()

        self.right_tab_widget = QTabWidget()
        self.right_tab_widget.setObjectName("rightSidebarTabs")
        self.right_tab_widget.setTabBar(EqualWidthTabBar())
        self.right_tab_widget.addTab(detail_tab, "详情")
        self.right_tab_widget.addTab(inspiration_tab, "AI")
        self.right_tab_widget.addTab(filter_tab, "标签")
        self.right_tab_widget.addTab(index_tab, "索引")
        self.right_tab_widget.addTab(settings_tab, "设置")
        self.right_tab_widget.setElideMode(Qt.TextElideMode.ElideRight)
        tab_bar = self.right_tab_widget.tabBar()
        tab_bar.setExpanding(True)
        tab_bar.setUsesScrollButtons(False)
        tab_bar.setTabToolTip(1, "AI 语义探针")
        tab_bar.setMinimumWidth(
            EqualWidthTabBar.minimum_width_for_tab_count(self.right_tab_widget.count())
        )
        self.right_tab_widget.setCurrentIndex(
            self._setting_int("ui.right_tab_index", 0, 0, self.right_tab_widget.count() - 1)
        )

        layout.addSpacing(10)
        layout.addWidget(self.right_tab_widget, 1)
        return panel

    def _connect_signals(self) -> None:
        self.add_folder_button.clicked.connect(self._choose_folder)
        self.import_folder_tree_button.clicked.connect(self._choose_folder_tree_import)
        self.rescan_button.clicked.connect(self._rescan_selected_folder)
        self.add_collection_button.clicked.connect(self._create_collection_from_button)
        self.creative_project_combo.currentIndexChanged.connect(self._on_creative_project_combo_changed)
        self.creative_project_list.itemSelectionChanged.connect(self._on_creative_project_list_changed)
        self.creative_project_list.customContextMenuRequested.connect(self._show_creative_project_context_menu)
        self.creative_node_tree.customContextMenuRequested.connect(self._show_creative_node_context_menu)
        self.creative_node_tree.itemSelectionChanged.connect(self._on_creative_node_selection_changed)
        self.creative_new_project_button.clicked.connect(self._create_creative_project_from_current_brief)
        self.creative_add_child_button.clicked.connect(self._create_manual_creative_child_node)
        self.creative_delete_node_button.clicked.connect(self._delete_selected_creative_node)
        self.shuffle_results_button.clicked.connect(self._shuffle_current_grid_images)
        self.advanced_search_toggle_button.toggled.connect(self._toggle_advanced_search_tools)
        self.search_button.clicked.connect(self._run_search)
        self.clear_search_button.clicked.connect(self._clear_search)
        self.save_result_set_button.clicked.connect(self._save_current_visible_results_as_temporary_project)
        self.search_input.returnPressed.connect(self._run_search)
        self.similar_image_button.clicked.connect(self._find_similar_to_selected_image)
        self.color_mode_button.clicked.connect(self._choose_search_color)
        self.collection_filter_button.clicked.connect(self._choose_collection_filter)
        self.tag_filter_button.clicked.connect(self._choose_tag_filter)
        self.add_file_type_filter_button.clicked.connect(self._add_file_type_filter_from_controls)
        self.add_dimension_filter_button.clicked.connect(self._add_dimension_filter_from_controls)
        self.ai_vision_field_filter_combo.currentIndexChanged.connect(self._refresh_ai_vision_value_filter_combo)
        self.add_ai_vision_filter_button.clicked.connect(self._add_ai_vision_filter_from_controls)
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        self.sort_order_combo.currentIndexChanged.connect(self._on_sort_changed)
        self.score_threshold_slider.sliderMoved.connect(self._preview_score_threshold)
        self.score_threshold_slider.valueChanged.connect(self._update_score_threshold)
        self.thumbnail_size_slider.valueChanged.connect(self._update_thumbnail_size)
        self.show_gallery_button.clicked.connect(self._show_gallery_view)
        self.show_project_board_button.clicked.connect(self._show_current_project_board)
        self.save_project_board_layout_button.clicked.connect(self._save_current_creative_board_layout)
        self.board_pin_button.clicked.connect(self._toggle_board_window_pin)
        self.board_hide_selected_button.clicked.connect(self._toggle_board_focus_mode)
        self.board_fit_all_button.clicked.connect(self._fit_board_all_images)
        self.board_flip_button.clicked.connect(self._flip_board_selected_images)
        self.board_grayscale_button.clicked.connect(self._toggle_board_selected_grayscale)
        self.board_import_button.clicked.connect(self._import_images_to_current_board)
        self.board_show_all_button.clicked.connect(self._show_all_board_images)
        self.project_board_view.removeImagesRequested.connect(self._remove_images_from_current_board)
        self.project_board_view.undoRemovalRequested.connect(self._undo_last_board_removal)
        self.load_more_button.clicked.connect(self._load_more)
        self.grid_view.selectionChanged.connect(self._on_grid_image_selected)
        self.grid_view.selectionSetChanged.connect(self._on_grid_selection_changed)
        self.grid_view.imageDoubleClicked.connect(self._open_image_preview)
        self.grid_view.imagePreviewRequested.connect(self._open_image_preview)
        self.grid_view.imageContextMenuRequested.connect(self._show_grid_context_menu)
        self.grid_view.filesDropped.connect(self._import_dropped_files_to_selected_collection)
        self.grid_view.dropPayloadDropped.connect(self._import_drop_payload_to_selected_collection)
        self.rename_file_button.clicked.connect(self._rename_current_file)
        self.file_name_input.returnPressed.connect(self._rename_current_file)
        self.favorite_checkbox.toggled.connect(self._save_current_favorite)
        self.note_input.textChanged.connect(self._queue_note_auto_save)
        self.note_auto_save_timer.timeout.connect(self._save_pending_note)
        self.delete_source_button.clicked.connect(self._delete_selected_source_files)
        self.undo_removal_action = QAction("撤销删除/移除", self)
        self.undo_removal_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_removal_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.undo_removal_action.setEnabled(False)
        self.undo_removal_action.triggered.connect(self._handle_undo_shortcut)
        self.addAction(self.undo_removal_action)
        self.minimize_window_action = QAction("最小化窗口", self)
        self.minimize_window_action.setShortcuts([QKeySequence("Meta+M"), QKeySequence("Ctrl+M")])
        self.minimize_window_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.minimize_window_action.triggered.connect(self._minimize_window)
        self.addAction(self.minimize_window_action)
        self.board_focus_shortcut = QShortcut(QKeySequence("Tab"), self)
        self.board_focus_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.board_focus_shortcut.setEnabled(True)
        self.board_focus_shortcut.activated.connect(self._toggle_board_focus_mode)
        self.open_collection_import_dir_button.clicked.connect(self._open_selected_collection_import_dir)
        self.open_original_button.clicked.connect(self._open_selected_original)
        self.reveal_in_finder_button.clicked.connect(self._reveal_selected_in_finder)
        self.copy_path_button.clicked.connect(self._copy_selected_path)
        self.play_pause_button.clicked.connect(self._toggle_video_playback)
        self.batch_add_tags_button.clicked.connect(self._batch_add_tags_from_panel)
        self.batch_remove_tag_button.clicked.connect(self._batch_remove_selected_tag)
        self.batch_clear_tags_button.clicked.connect(self._batch_clear_tags)
        self.creative_selection_add_button.clicked.connect(self._save_selection_to_current_creative_node)
        self.creative_selection_remove_button.clicked.connect(self._remove_selection_from_current_creative_node)
        self.tag_panel_add_button.clicked.connect(self._tag_panel_add_tags)
        self.tag_panel_remove_button.clicked.connect(self._tag_panel_remove_selected_tag)
        self.tag_panel_clear_button.clicked.connect(self._tag_panel_clear_tags)
        self.ai_workflow_mode_group.idClicked.connect(self._on_ai_workflow_mode_clicked)
        self.generate_inspiration_button.clicked.connect(self._generate_inspiration_terms_from_panel)
        self.save_creative_node_button.clicked.connect(self._save_current_creative_node_details)
        self.generate_creative_children_button.clicked.connect(self._generate_creative_children_for_selected_node)
        self.generate_creative_copy_button.clicked.connect(self._generate_creative_project_copy)
        self.generate_creative_copy_tab_button.clicked.connect(self._generate_creative_project_copy)
        self.search_creative_node_button.clicked.connect(self._search_selected_creative_node)
        self.save_selection_to_creative_node_button.clicked.connect(self._save_selection_to_current_creative_node)
        self.open_creative_board_button.clicked.connect(self._show_current_creative_board)
        self.search_inspiration_button.clicked.connect(self._save_and_search_inspiration)
        self.inspiration_history_list.itemClicked.connect(self._load_selected_inspiration_history)
        self.inspiration_history_list.customContextMenuRequested.connect(self._show_inspiration_history_context_menu)
        self.inspiration_term_list.itemChanged.connect(self._enforce_inspiration_selection_limit)
        self.inspiration_term_list.customContextMenuRequested.connect(self._show_inspiration_term_context_menu)
        self.inspiration_filter_list.itemChanged.connect(self._handle_inspiration_filter_changed)
        self.save_temp_project_button.clicked.connect(
            lambda _checked=False: self._save_selected_images_as_temporary_project(kind="semantic")
        )
        self.feedback_relevant_button.clicked.connect(lambda: self._save_search_feedback("relevant"))
        self.feedback_irrelevant_button.clicked.connect(lambda: self._save_search_feedback("irrelevant"))
        self.feedback_ignored_button.clicked.connect(lambda: self._save_search_feedback("ignored"))
        self.start_embedding_button.clicked.connect(self._start_embedding)
        self.pause_embedding_button.clicked.connect(self._pause_embedding)
        self.retry_failed_button.clicked.connect(self._retry_failed_embeddings)
        self.add_ai_vision_include_rule_button.clicked.connect(lambda: self._set_ai_vision_rule_for_selected_collection("include"))
        self.add_ai_vision_exclude_rule_button.clicked.connect(lambda: self._set_ai_vision_rule_for_selected_collection("exclude"))
        self.remove_ai_vision_rule_button.clicked.connect(self._remove_selected_ai_vision_rule)
        self.start_ai_vision_button.clicked.connect(self._start_ai_vision)
        self.pause_ai_vision_button.clicked.connect(self._pause_ai_vision)
        self.retry_failed_ai_vision_button.clicked.connect(self._retry_failed_ai_vision)
        self.refresh_ai_vision_button.clicked.connect(self._refresh_ai_vision_stats)
        self.rescan_all_button.clicked.connect(self._rescan_all_folders)
        self.rescan_new_button.clicked.connect(self._rescan_new_or_changed_folders)
        self.rescan_missing_button.clicked.connect(self._rescan_missing_folders)
        self.file_watch_checkbox.toggled.connect(self._set_file_watch_enabled)
        self.show_missing_files_button.clicked.connect(self._show_missing_files_dialog)
        self.clean_missing_index_button.clicked.connect(self._clean_missing_index)
        self.detect_duplicates_button.clicked.connect(self._detect_duplicates)
        self.clean_orphan_thumbnails_button.clicked.connect(self._clean_orphan_thumbnails)
        self.rebuild_selected_thumbnails_button.clicked.connect(self._batch_rebuild_thumbnails)
        self.remove_selected_index_button.clicked.connect(self._batch_remove_from_library)
        self.export_library_button.clicked.connect(self._export_library)
        self.export_selection_button.clicked.connect(self._export_selected_images)
        self.folder_tree.itemSelectionChanged.connect(self._refresh_current_results_for_filters)
        self.collection_tree.itemSelectionChanged.connect(self._on_collection_selection_changed)
        self.collection_tree.treeReordered.connect(self._save_collection_tree_order)
        self.collection_tree.imagesDropped.connect(self._assign_dropped_images_to_collection)
        self.collection_tree.filesDropped.connect(self._import_dropped_files_to_collection)
        self.collection_tree.rootFilesDropped.connect(self._import_dropped_files_to_root)
        self.status_filter_combo.currentIndexChanged.connect(self._refresh_current_results_for_filters)
        self.status_filter_combo.currentIndexChanged.connect(self._save_status_filter)
        self.tag_list.itemSelectionChanged.connect(self._on_tag_list_selection_changed)
        self.ai_vision_virtual_filter_list.itemSelectionChanged.connect(
            self._on_ai_vision_virtual_filter_selection_changed
        )
        self.tag_search_input.textChanged.connect(self._on_tag_search_changed)
        self.tag_sort_combo.currentIndexChanged.connect(self._on_tag_sort_changed)
        self.tag_match_combo.currentIndexChanged.connect(self._on_tag_match_changed)
        self.rename_tag_button.clicked.connect(self._rename_selected_tag)
        self.delete_tag_button.clicked.connect(self._delete_selected_tag)
        self.merge_tag_button.clicked.connect(self._merge_selected_tag)
        self.folder_tree.customContextMenuRequested.connect(self._show_folder_context_menu)
        self.collection_tree.customContextMenuRequested.connect(self._show_collection_context_menu)
        self.temp_project_list.itemClicked.connect(self._handle_project_sidebar_item_clicked)
        self.temp_project_list.itemSelectionChanged.connect(self._load_selected_temporary_project)
        self.temp_project_list.customContextMenuRequested.connect(self._show_temporary_project_context_menu)
        self.tag_list.customContextMenuRequested.connect(self._show_tag_context_menu)
        self.right_tab_widget.currentChanged.connect(self._save_right_tab_index)
        self.llm_service_combo.currentIndexChanged.connect(self._on_llm_service_changed)
        self.save_settings_button.clicked.connect(self._save_settings)
        self.open_data_dir_button.clicked.connect(self._open_data_directory)
        self.backup_database_button.clicked.connect(self._backup_database)
        self.restore_database_button.clicked.connect(self._restore_database_from_file)
        self.run_self_check_button.clicked.connect(self._run_self_check_from_settings)
        self.show_error_log_button.clicked.connect(self._show_error_log_window)
        self.show_operation_history_button.clicked.connect(self._show_operation_history)
        self.refresh_path_candidates_button.clicked.connect(self._refresh_path_remap_candidates)
        self.choose_remap_new_path_button.clicked.connect(self._choose_remap_new_path)
        self.apply_path_remap_button.clicked.connect(self._apply_path_remap)
        self.run_performance_check_button.clicked.connect(self._run_performance_check)
        self.more_maintenance_button.toggled.connect(self._toggle_more_maintenance)
        self.saved_view_combo.currentIndexChanged.connect(self._refresh_saved_view_buttons)
        self.save_view_button.clicked.connect(self._save_current_view)
        self.apply_view_button.clicked.connect(self._apply_selected_saved_view)
        self.rename_view_button.clicked.connect(self._rename_selected_saved_view)
        self.delete_view_button.clicked.connect(self._delete_selected_saved_view)
        self.file_watcher.directoryChanged.connect(self._handle_watched_path_changed)
        self.file_watcher.fileChanged.connect(self._handle_watched_path_changed)

    def _minimize_window(self) -> None:
        self.showMinimized()

    def _toggle_advanced_search_tools(self, checked: bool) -> None:
        self.advanced_search_widget.setVisible(checked)
        if self.current_language == "en":
            self.advanced_search_toggle_button.setText("Hide Filters" if checked else "Filters / Sort")
        else:
            self.advanced_search_toggle_button.setText("收起筛选" if checked else "筛选/排序")

    def _load_settings_controls(self) -> None:
        service = self._llm_service_key()
        self._set_combo_to_data(self.llm_service_combo, service)
        self._load_llm_service_controls(service)
        self.llm_temperature_spin.setValue(self._llm_temperature())
        self._set_combo_to_data(self.language_combo, self.current_language)
        if hasattr(self, "file_watch_checkbox"):
            self.file_watch_checkbox.blockSignals(True)
            self.file_watch_checkbox.setChecked(self.file_watch_enabled)
            self.file_watch_checkbox.blockSignals(False)
        if hasattr(self, "path_remap_old_combo"):
            self._refresh_path_remap_candidates()

    def _llm_service_key(self) -> str:
        raw = self.store.get_setting("llm.provider", "lm_studio")
        allowed = {key for key, _label in LLM_SERVICE_OPTIONS}
        return raw if raw in allowed else "lm_studio"

    def _llm_service_label(self, service: str) -> str:
        return dict(LLM_SERVICE_OPTIONS).get(service, "LM Studio")

    def _llm_endpoint(self, service: str) -> str:
        return (
            self.store.get_setting(f"llm.{service}.base_url")
            or self.store.get_setting("llm.lmstudio.base_url")
            or DEFAULT_LLM_ENDPOINTS.get(service)
            or DEFAULT_LLM_ENDPOINTS["lm_studio"]
        )

    def _llm_model(self, service: str) -> str:
        return self.store.get_setting(f"llm.{service}.model") or (
            self.store.get_setting("llm.lmstudio.model") if service == "lm_studio" else ""
        ) or ""

    def _llm_api_key(self, service: str) -> str:
        return self.store.get_setting(f"llm.{service}.api_key") or ""

    def _llm_temperature(self) -> float:
        raw = self.store.get_setting("llm.temperature", "0.7")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.7
        return max(0.0, min(2.0, value))

    def _make_llm_provider(self) -> LMStudioProvider:
        service = self._llm_service_key()
        return LMStudioProvider(
            base_url=self._llm_endpoint(service),
            model_name=self._llm_model(service) or None,
            api_key=self._llm_api_key(service),
            service_name=self._llm_service_label(service),
            temperature=self._llm_temperature(),
        )

    def _make_ai_vision_provider(self) -> AIVisionProvider:
        service = self._llm_service_key()
        return AIVisionProvider(
            base_url=self._llm_endpoint(service),
            model_name=self._llm_model(service) or None,
            api_key=self._llm_api_key(service),
            service_name=self._llm_service_label(service),
            temperature=0.1,
        )

    def _ai_vision_provider_name(self) -> str:
        return self._llm_service_label(self._llm_service_key())

    def _ai_vision_model_name_for_stats(self) -> str:
        service = self._llm_service_key()
        return self._llm_model(service) or "local-vision-model"

    def _load_llm_service_controls(self, service: str) -> None:
        self.llm_endpoint_input.setText(self._llm_endpoint(service))
        self.llm_model_input.setText(self._llm_model(service))
        self.llm_api_key_input.setText(self._llm_api_key(service))

    def _on_llm_service_changed(self) -> None:
        service = str(self.llm_service_combo.currentData() or "lm_studio")
        self._load_llm_service_controls(service)

    def _save_settings(self) -> None:
        service = str(self.llm_service_combo.currentData() or "lm_studio")
        language = str(self.language_combo.currentData() or "zh")
        self.store.set_setting("llm.provider", service)
        self.store.set_setting(f"llm.{service}.base_url", self.llm_endpoint_input.text().strip())
        self.store.set_setting(f"llm.{service}.model", self.llm_model_input.text().strip())
        self.store.set_setting(f"llm.{service}.api_key", self.llm_api_key_input.text().strip())
        self.store.set_setting("llm.temperature", f"{self.llm_temperature_spin.value():.2f}")
        self.store.set_setting("ui.language", language)
        self.current_language = language
        self._apply_runtime_language_settings()
        self._refresh_ai_vision_stats()
        self.settings_status_label.setText("设置已保存。API Key 不会写入日志。")
        self.statusBar().showMessage("设置已保存")

    def _open_data_directory(self) -> None:
        self.paths.ensure()
        ok = QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.paths.data_dir)))
        self.statusBar().showMessage("已打开数据目录" if ok else "打开数据目录失败")

    @staticmethod
    @contextmanager
    def _connect_sqlite_database(path: str | Path, *, uri: bool = False) -> Iterator[sqlite3.Connection]:
        with MetadataStore._connection_lock:
            conn = sqlite3.connect(
                path,
                uri=uri,
                timeout=MetadataStore._busy_timeout_ms / 1000,
            )
            MetadataStore.configure_connection(conn, readonly=uri)
            try:
                yield conn
            finally:
                conn.close()

    def _backup_database(self) -> None:
        try:
            with self._database_maintenance("数据库备份", restart_index_workers=True):
                backup_path = self._backup_database_to_default_location()
        except Exception as exc:
            self._record_error(f"数据库备份失败：{exc}")
            QMessageBox.warning(self, "Eidory", f"数据库备份失败：{exc}")
            return
        self.settings_status_label.setText(f"数据库已备份：{backup_path}")
        self.statusBar().showMessage("数据库已备份")

    def _backup_database_to_default_location(self) -> Path:
        backup_dir = self.paths.data_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = backup_dir / f"eidory-{timestamp}.sqlite3"
        with self._connect_sqlite_database(self.paths.database_path) as source:
            with self._connect_sqlite_database(backup_path) as target:
                source.backup(target)
        return backup_path

    def _restore_database_from_file(self) -> None:
        database_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择要恢复的 Eidory 数据库",
            str(self.paths.data_dir),
            "SQLite Database (*.sqlite3 *.db);;All Files (*)",
        )
        if not database_path:
            return
        source_path = Path(database_path).expanduser()
        ok, message = self._validate_database_file(source_path)
        if not ok:
            QMessageBox.warning(self, "Eidory", f"数据库文件无效：{message}")
            return
        try:
            if source_path.resolve() == self.paths.database_path.resolve():
                QMessageBox.information(self, "Eidory", "选择的是当前正在使用的数据库。")
                return
        except OSError:
            pass
        confirm = QMessageBox.question(
            self,
            "恢复数据库",
            "恢复会先备份当前数据库，然后用所选数据库替换当前数据库。继续？",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        restore_temp_path: Path | None = None
        try:
            with self._database_maintenance("数据库恢复"):
                backup_path = self._backup_database_to_default_location()
                self.paths.database_path.parent.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                restore_temp_path = self.paths.database_path.with_name(
                    f".eidory-restore-{timestamp}.sqlite3"
                )
                source_uri = f"{source_path.resolve().as_uri()}?mode=ro"
                with self._connect_sqlite_database(source_uri, uri=True) as source:
                    with self._connect_sqlite_database(restore_temp_path) as target:
                        source.backup(target)
                shutil.move(str(restore_temp_path), self.paths.database_path)
                for suffix in ("-wal", "-shm"):
                    self.paths.database_path.with_name(
                        f"{self.paths.database_path.name}{suffix}"
                    ).unlink(missing_ok=True)
                self.store.initialize()
            self._refresh_after_database_change()
        except Exception as exc:
            if restore_temp_path is not None:
                restore_temp_path.unlink(missing_ok=True)
            self._record_error(f"数据库恢复失败：{exc}")
            QMessageBox.critical(self, "Eidory", f"数据库恢复失败：{exc}")
            return
        self.settings_status_label.setText(f"数据库已恢复。恢复前备份：{backup_path}")
        self.statusBar().showMessage("数据库已恢复")

    @staticmethod
    def _validate_database_file(database_path: Path) -> tuple[bool, str]:
        if not database_path.is_file():
            return False, "文件不存在"
        try:
            uri = f"{database_path.resolve().as_uri()}?mode=ro"
            with MainWindow._connect_sqlite_database(uri, uri=True) as conn:
                row = conn.execute("PRAGMA integrity_check").fetchone()
                if row is None or str(row[0]) != "ok":
                    return False, str(row[0]) if row is not None else "integrity_check 无结果"
                table_rows = conn.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name IN ('folders', 'images', 'app_settings')
                    """
                ).fetchall()
        except Exception as exc:
            return False, str(exc)
        found_tables = {str(row[0]) for row in table_rows}
        missing = {"folders", "images", "app_settings"} - found_tables
        if missing:
            return False, f"缺少表：{', '.join(sorted(missing))}"
        return True, "ok"

    def _refresh_path_remap_candidates(self) -> None:
        if not hasattr(self, "path_remap_old_combo"):
            return
        current_text = self.path_remap_old_combo.currentText().strip()
        candidates = self.store.path_remap_candidates()
        self.path_remap_old_combo.blockSignals(True)
        self.path_remap_old_combo.clear()
        self.path_remap_old_combo.addItems(candidates)
        if current_text:
            index = self.path_remap_old_combo.findText(current_text)
            if index >= 0:
                self.path_remap_old_combo.setCurrentIndex(index)
            else:
                self.path_remap_old_combo.setEditText(current_text)
        self.path_remap_old_combo.blockSignals(False)

    def _choose_remap_new_path(self) -> None:
        folder_path = QFileDialog.getExistingDirectory(self, "选择移动后的新根目录")
        if folder_path:
            self.path_remap_new_input.setText(folder_path)

    def _apply_path_remap(self) -> None:
        old_prefix = self.path_remap_old_combo.currentText().strip()
        new_prefix = self.path_remap_new_input.text().strip()
        if not old_prefix or not new_prefix:
            self.statusBar().showMessage("请先填写旧位置和新位置")
            return
        try:
            counts = self.store.path_prefix_match_counts(old_prefix)
        except Exception as exc:
            QMessageBox.warning(self, "Eidory", f"检查旧位置失败：{exc}")
            return
        if counts["folders"] == 0 and counts["images"] == 0:
            QMessageBox.information(self, "Eidory", "旧位置没有匹配到图库记录。")
            return
        confirm = QMessageBox.question(
            self,
            "应用路径重映射",
            (
                f"旧位置匹配到 {counts['folders']} 个导入目录、"
                f"{counts['images']} 个文件记录，其中 {counts['missing']} 个丢失。继续？"
            ),
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._set_maintenance_controls_enabled(False)
        self.settings_status_label.setText("正在应用路径重映射...")

        def run() -> None:
            try:
                result = self.store.remap_path_prefix(old_prefix, new_prefix)
                self.events.put(("path_remap_done", result))
            except Exception as exc:
                self.events.put(("error", f"路径重映射失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=lambda: self._set_maintenance_controls_enabled(True),
        )

    def _handle_path_remap_done(self, result: dict[str, int]) -> None:
        self._set_maintenance_controls_enabled(True)
        self.vector_index.invalidate()
        self._refresh_after_database_change()
        message = (
            "路径重映射完成："
            f"目录更新 {result.get('folders_updated', 0)}，"
            f"目录合并 {result.get('folders_merged', 0)}，"
            f"文件更新 {result.get('images_updated', 0)}，"
            f"已恢复 {result.get('relinked', 0)}，"
            f"仍丢失 {result.get('still_missing', 0)}，"
            f"冲突跳过 {result.get('conflicts', 0)}"
        )
        self.settings_status_label.setText(message)
        self.statusBar().showMessage(message)

    def _refresh_after_database_change(self) -> None:
        self._invalidate_near_duplicate_hash_cache()
        self.current_language = self._setting_choice("ui.language", "zh", {"zh", "en"})
        self._load_settings_controls()
        self._apply_runtime_language_settings()
        self.vector_index.invalidate()
        self._refresh_folders()
        self._refresh_collections()
        self._refresh_temporary_projects()
        self._refresh_tags()
        self._refresh_saved_views()
        self._refresh_inspiration_history()
        self._reload_images()
        self._refresh_embedding_stats()
        self._refresh_file_watcher()

    def _set_maintenance_controls_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self._maintenance_controls_enabled = enabled
        for attr in [
            "rescan_all_button",
            "rescan_new_button",
            "rescan_missing_button",
            "show_missing_files_button",
            "clean_missing_index_button",
            "detect_duplicates_button",
            "clean_orphan_thumbnails_button",
            "run_performance_check_button",
            "restore_database_button",
            "apply_path_remap_button",
        ]:
            if hasattr(self, attr):
                widget = getattr(self, attr)
                if widget.isEnabled() != enabled:
                    widget.setEnabled(enabled)

    def _run_startup_self_check(self) -> None:
        if not hasattr(self, "settings_status_label"):
            return
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return
        self._run_self_check(show_success=False)

    def _run_self_check_from_settings(self) -> None:
        self._run_self_check(show_success=True)

    def _run_self_check(self, *, show_success: bool) -> None:
        self.run_self_check_button.setEnabled(False)
        self.settings_status_label.setText("正在自检...")

        def run() -> None:
            report = self._build_self_check_report()
            self.events.put(("self_check_done", (report, show_success)))

        self._start_background_task(
            run,
            on_rejected=lambda: self.run_self_check_button.setEnabled(True),
        )

    def _build_self_check_report(self) -> list[tuple[str, bool, str]]:
        report: list[tuple[str, bool, str]] = []
        report.append(self._check_directory("数据目录", self.paths.data_dir))
        report.append(self._check_directory("缩略图目录", self.paths.thumbnail_dir))
        report.append(self._check_database())
        ffmpeg_path = find_media_tool("ffmpeg")
        report.append(("ffmpeg", ffmpeg_path is not None, ffmpeg_path or "未找到，视频缩略图会失败"))
        report.append(self._check_llm_endpoint())
        return report

    @staticmethod
    def _check_directory(label: str, path: Path) -> tuple[str, bool, str]:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".eidory-write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return label, True, str(path)
        except Exception as exc:
            return label, False, f"{path}: {exc}"

    def _check_database(self) -> tuple[str, bool, str]:
        try:
            with self._connect_sqlite_database(self.paths.database_path) as conn:
                row = conn.execute("PRAGMA integrity_check").fetchone()
            status = str(row[0]) if row is not None else "no result"
            return "数据库", status == "ok", status
        except Exception as exc:
            return "数据库", False, str(exc)

    def _check_llm_endpoint(self) -> tuple[str, bool, str]:
        service = self._llm_service_key()
        endpoint = self._llm_endpoint(service).rstrip("/")
        if not endpoint:
            return "模型服务", False, "Endpoint 为空"
        request = urllib.request.Request(f"{endpoint}/models")
        api_key = self._llm_api_key(service)
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(request, timeout=2.0) as response:
                status = int(getattr(response, "status", 0))
            return "模型服务", 200 <= status < 500, f"{self._llm_service_label(service)} / HTTP {status}"
        except urllib.error.HTTPError as exc:
            return "模型服务", exc.code < 500, f"{self._llm_service_label(service)} / HTTP {exc.code}"
        except Exception as exc:
            return "模型服务", False, f"{self._llm_service_label(service)} / {exc}"

    def _handle_self_check_done(self, payload: object) -> None:
        report, show_success = payload
        self.run_self_check_button.setEnabled(True)
        lines = [
            f"{'OK' if ok else 'FAIL'} {label}: {detail}"
            for label, ok, detail in report
        ]
        failed = [line for line, (_label, ok, _detail) in zip(lines, report, strict=False) if not ok]
        message = "\n".join(lines)
        self.settings_status_label.setText(message)
        if failed:
            self._record_error("启动自检发现问题：\n" + "\n".join(failed))
            self.statusBar().showMessage(f"自检发现 {len(failed)} 个问题")
        elif show_success:
            self.statusBar().showMessage("自检通过")

    def _show_error_log_window(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("错误日志")
        dialog.resize(720, 420)
        layout = QVBoxLayout(dialog)
        output = QTextEdit()
        output.setReadOnly(True)
        output.setAcceptRichText(False)
        output.setPlainText("\n\n".join(self.error_log_messages) or "暂无错误。")
        layout.addWidget(output)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def _record_error(self, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        entry = f"[{timestamp}] {message}"
        self.error_log_messages.append(entry)
        self.error_log_messages = self.error_log_messages[-200:]
        try:
            self.paths.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.paths.log_dir / "eidory-ui.log").open("a", encoding="utf-8") as handle:
                handle.write(entry + "\n")
        except Exception:
            pass

    def _configure_accessibility_labels(self) -> None:
        pairs = [
            (self, "Eidory main window", "Local image library manager"),
            (self.search_input, "Search text", "File name, tags, notes, or semantic search text"),
            (self.reverse_exclusion_button, "Reverse exclusion toggle", "Use selected filter buttons as exclusion filters"),
            (self.color_mode_button, "Color filter", "Pick a color and filter images by dominant color similarity"),
            (self.keyword_mode_button, "Keyword search mode", "Search file names, tags and notes"),
            (self.semantic_mode_button, "Semantic search mode", "Search images by semantic embedding"),
            (self.collection_filter_button, "Folder filter", "Filter or exclude selected folders"),
            (self.tag_filter_button, "Tag filter", "Filter or exclude selected user tags"),
            (self.similar_image_button, "Similar image search", "Search images similar to the selected image"),
            (self.search_button, "Run search", "Run the selected search"),
            (self.clear_search_button, "Clear search", "Clear current filters and search state"),
            (self.advanced_search_toggle_button, "Search and sort panel", "Show or hide advanced search controls"),
            (self.search_replace_results_button, "Replace results", "Search the current scope from scratch"),
            (self.search_within_results_button, "Search within results", "Narrow the current visible result set"),
            (self.search_merge_results_button, "Merge results", "Union a new search with current results"),
            (self.result_state_label, "Result status", "Unified library and search result status"),
            (self.grid_view, "Image wall", "Selectable image and video result grid"),
            (self.collection_tree, "Library folders", "Nested Eidory folder tree"),
            (self.temp_project_list, "Project list", "Saved temporary reference groups and creative projects"),
            (self.right_tab_widget, "Right side panel", "Details, AI, tag, index and settings pages"),
            (self.preview_label, "Selected item preview", "Preview of selected image or folder state"),
            (self.tags_display, "Selected item tags", "Read-only tags for the selected image"),
            (self.use_ai_scene_tags_checkbox, "Use AI scene tags", "Generate scene tag filters together with semantic probes"),
            (self.tag_panel_selection_label, "Tag page selection status", "Shows which selected images can be tagged"),
            (self.tag_panel_input, "Add tags to selection", "One tag per line to add to selected images"),
            (self.tag_list, "Tag management list", "Existing user tags for filtering and management"),
            (self.settings_status_label, "Settings status", "Settings, self-check and error status output"),
        ]
        for widget, name, description in pairs:
            try:
                widget.setAccessibleName(name)
                widget.setAccessibleDescription(description)
            except RuntimeError:
                continue

    def _apply_runtime_language_settings(self) -> None:
        if self.current_language == "en":
            self.add_collection_button.setText("New Folder")
            self.add_folder_button.setText("Import Images")
            self.import_folder_tree_button.setText("Import Folder")
            self.search_input.setPlaceholderText("File name, tags, notes, or semantic search text")
            self.keyword_mode_button.setText("Keyword")
            self.semantic_mode_button.setText("Semantic")
            self.collection_filter_button.setText("Folder")
            self.collection_filter_button.setToolTip("Choose folders as include or reverse-exclude conditions")
            self.tag_filter_button.setText("Tags")
            self.tag_filter_button.setToolTip("Choose user tags as include or reverse-exclude conditions")
            self.similar_image_button.setText("Similar")
            self.search_button.setText("Search")
            self.clear_search_button.setText("Clear")
            self.reverse_exclusion_button.setText("Exclude")
            self.reverse_exclusion_button.setToolTip("When enabled, Color/Keyword/Semantic/Tag conditions subtract matches from the current results")
            self.advanced_search_toggle_button.setText(
                "Hide Filters" if self.advanced_search_toggle_button.isChecked() else "Filters / Sort"
            )
            self.advanced_search_toggle_button.setToolTip("Show search logic, metadata filters, sorting, and result tools")
            self.search_within_results_button.setText("Within Results")
            self.search_merge_results_button.setText("Merge Results")
            self.search_replace_results_button.setText("Replace")
            self.search_operation_label.setText("Search Logic")
            self.shuffle_results_button.setText("Shuffle")
            self.save_result_set_button.setText("Save Results")
            self.rename_file_button.setText("Rename")
            self.rename_file_button.setToolTip("Rename the source file on disk")
            self.delete_source_button.setText("Delete / Remove")
            self.delete_source_button.setToolTip("Delete source files, or only remove them from Eidory")
            self.tag_panel_input.setPlaceholderText("Add tags to selected items, one per line")
            self.tag_panel_add_button.setText("Add to Selected")
            self.tag_panel_remove_button.setText("Remove Tag")
            self.tag_panel_clear_button.setText("Clear Tags")
            self.creative_selection_add_button.setText("Save to Node")
            self.creative_selection_remove_button.setText("Remove from Node")
            self.inspiration_brief_input.setPlaceholderText("Describe the image concept in one sentence")
            self.inspiration_answers_input.setPlaceholderText("Extra context: era, weather, lighting, mood, optional")
            self.inspiration_questions_label.setText("AI questions: -")
            self.inspiration_status_label.setText("Select up to 7 semantic probes.")
            self.use_ai_scene_tags_checkbox.setText("Generate AI scene tags with probes")
            self.ai_project_mode_button.setText("Project Nodes")
            self.ai_probe_mode_button.setText("Semantic Probes")
            self.creative_new_project_button.setText("Create from Template")
            self.creative_add_child_button.setText("New Child")
            self.creative_delete_node_button.setText("Delete Node")
            self.save_creative_node_button.setText("Save Node")
            self.generate_creative_children_button.setText("AI Complete Node")
            self.search_creative_node_button.setText("Search Node")
            self.save_selection_to_creative_node_button.setText("Save to Node")
            self.open_creative_board_button.setText("Board")
            self.creative_content_tabs.setTabText(0, "Node Tree")
            self.creative_content_tabs.setTabText(1, "Copy")
            self._configure_board_icon_button(
                self.board_pin_button,
                "pin",
                "Pin board window",
                "Keep the Eidory window above other apps",
            )
            self._configure_board_icon_button(
                self.board_hide_selected_button,
                "focus",
                "Focus board",
                "Hide sidebars and non-board controls; press Tab to restore",
            )
            self._configure_board_icon_button(
                self.board_fit_all_button,
                "fit",
                "Fit all board images",
                "Fit all board images into the current board viewport",
            )
            self._configure_board_icon_button(
                self.board_flip_button,
                "flip",
                "Flip selected board images",
                "Flip selected board images horizontally",
            )
            self._configure_board_icon_button(
                self.board_grayscale_button,
                "grayscale",
                "Toggle grayscale board images",
                "Toggle grayscale display for selected board images",
            )
            self.board_import_button.setText("Import")
            self.board_show_all_button.setText("Show All")
            self.generate_inspiration_button.setText("Generate Probes")
            self.search_inspiration_button.setText("Save and Search")
            self.save_temp_project_button.setText("Save Selected")
            self.add_ai_vision_filter_button.setText("Add Scene Tag")
            self.add_ai_vision_include_rule_button.setText("Include Folder")
            self.add_ai_vision_exclude_rule_button.setText("Exclude Folder")
            self.remove_ai_vision_rule_button.setText("Remove Rule")
            self.start_ai_vision_button.setText("Start AI Vision")
            self.pause_ai_vision_button.setText("Pause AI Vision")
            self.retry_failed_ai_vision_button.setText("Retry Failed")
            self.refresh_ai_vision_button.setText("Refresh Stats")
            self.rescan_all_button.setText("Scan All")
            self.rescan_new_button.setText("Scan New")
            self.rescan_missing_button.setText("Scan Missing")
            self.file_watch_checkbox.setText("Watch local file changes")
            self.show_missing_files_button.setText("Repair Missing")
            self.clean_missing_index_button.setText("Clean Missing")
            self.detect_duplicates_button.setText("Find Duplicates")
            self.clean_orphan_thumbnails_button.setText("Clean Thumbs")
            self.rebuild_selected_thumbnails_button.setText("Rebuild Selected")
            self.remove_selected_index_button.setText("Remove Selected")
            self.run_performance_check_button.setText("Benchmark")
            self.more_maintenance_button.setText("Hide More" if self.more_maintenance_button.isChecked() else "More Maintenance")
            self.export_library_button.setText("Export Library")
            self.export_selection_button.setText("Export Images")
            self.open_data_dir_button.setText("Open Data")
            self.backup_database_button.setText("Backup DB")
            self.restore_database_button.setText("Restore DB")
            self.run_self_check_button.setText("Startup Check")
            self.show_error_log_button.setText("Error Log")
            self.show_operation_history_button.setText("Operation History")
            self.path_remap_new_input.setPlaceholderText("Choose the new root folder")
            self.refresh_path_candidates_button.setText("Refresh")
            self.choose_remap_new_path_button.setText("Choose")
            self.apply_path_remap_button.setText("Apply Remap")
            self.right_tab_widget.setTabText(0, "Details")
            self.right_tab_widget.setTabText(1, "AI")
            self.right_tab_widget.setTabText(2, "Tags")
            self.right_tab_widget.setTabText(3, "Index")
            self.right_tab_widget.setTabText(4, "Settings")
        else:
            self.add_collection_button.setText("新建文件夹")
            self.add_folder_button.setText("导入图片")
            self.import_folder_tree_button.setText("导入文件夹")
            self.search_input.setPlaceholderText("文件名、标签、备注，或语义搜索文本")
            self.keyword_mode_button.setText("关键词")
            self.semantic_mode_button.setText("语义")
            self.collection_filter_button.setText("文件夹")
            self.collection_filter_button.setToolTip("选择文件夹作为筛选或反向排除条件")
            self.tag_filter_button.setText("标签")
            self.tag_filter_button.setToolTip("选择用户标签作为筛选或反向排除条件")
            self.similar_image_button.setText("相似图")
            self.search_button.setText("搜索")
            self.clear_search_button.setText("清空")
            self.reverse_exclusion_button.setText("反向排除")
            self.reverse_exclusion_button.setToolTip("打开后，颜色/关键词/语义/标签会从当前结果中反向扣除")
            self.advanced_search_toggle_button.setText(
                "收起筛选" if self.advanced_search_toggle_button.isChecked() else "筛选/排序"
            )
            self.advanced_search_toggle_button.setToolTip("展开搜索逻辑、元数据筛选、排序和结果管理")
            self.search_within_results_button.setText("在结果中搜")
            self.search_merge_results_button.setText("合并结果")
            self.search_replace_results_button.setText("重新搜索")
            self.search_operation_label.setText("搜索逻辑")
            self.shuffle_results_button.setText("打乱排序")
            self.save_result_set_button.setText("保存结果集")
            self.rename_file_button.setText("重命名")
            self.rename_file_button.setToolTip("重命名硬盘上的源文件")
            self.delete_source_button.setText("删除/移除图片")
            self.delete_source_button.setToolTip("选择删除源文件，或只从 Eidory 移除索引")
            self.tag_panel_input.setPlaceholderText("给选中图片添加标签，每行一个")
            self.tag_panel_add_button.setText("添加到选中")
            self.tag_panel_remove_button.setText("移除标签")
            self.tag_panel_clear_button.setText("清空标签")
            self.creative_selection_add_button.setText("存入当前节点")
            self.creative_selection_remove_button.setText("移出当前节点")
            self.inspiration_brief_input.setPlaceholderText("用一句话描述画面的创作主题")
            self.inspiration_answers_input.setPlaceholderText("补充信息：时代、天气、光源、画面气质等，可留空")
            self.inspiration_questions_label.setText("AI 追问：-")
            self.inspiration_status_label.setText("生成后最多选择 7 个语义探针。")
            self.use_ai_scene_tags_checkbox.setText("生成探针时同时生成 AI 场景标签")
            self.ai_project_mode_button.setText("创作节点")
            self.ai_probe_mode_button.setText("语义探针")
            self.creative_new_project_button.setText("按模板新建项目")
            self.creative_add_child_button.setText("新建子节点")
            self.creative_delete_node_button.setText("删除节点")
            self.save_creative_node_button.setText("保存节点")
            self.generate_creative_children_button.setText("节点信息AI补全")
            self.search_creative_node_button.setText("搜索当前节点")
            self.save_selection_to_creative_node_button.setText("存入当前节点")
            self.open_creative_board_button.setText("看板")
            self.creative_content_tabs.setTabText(0, "节点树")
            self.creative_content_tabs.setTabText(1, "文案")
            self._configure_board_icon_button(
                self.board_pin_button,
                "pin",
                "图钉",
                "让 Eidory 窗口置顶，避免被其他软件遮挡",
            )
            self._configure_board_icon_button(
                self.board_hide_selected_button,
                "focus",
                "隐藏界面",
                "隐藏左右侧栏和非看板控件；按 Tab 可恢复",
            )
            self._configure_board_icon_button(
                self.board_fit_all_button,
                "fit",
                "适应全部",
                "让所有看板图片适应当前画布窗口",
            )
            self._configure_board_icon_button(
                self.board_flip_button,
                "flip",
                "左右翻转",
                "左右镜像翻转选中的看板图片",
            )
            self._configure_board_icon_button(
                self.board_grayscale_button,
                "grayscale",
                "黑白",
                "把选中的看板图片切换为黑白显示",
            )
            self.board_import_button.setText("导入图片")
            self.board_show_all_button.setText("显示全部")
            self.generate_inspiration_button.setText("生成语义探针")
            self.search_inspiration_button.setText("保存并搜索")
            self.save_temp_project_button.setText("存为语义探针项目")
            self.add_ai_vision_filter_button.setText("添加场景标签")
            self.add_ai_vision_include_rule_button.setText("识别选中文件夹")
            self.add_ai_vision_exclude_rule_button.setText("排除选中文件夹")
            self.remove_ai_vision_rule_button.setText("移除规则")
            self.start_ai_vision_button.setText("开始AI识别")
            self.pause_ai_vision_button.setText("暂停AI识别")
            self.retry_failed_ai_vision_button.setText("重试AI失败")
            self.refresh_ai_vision_button.setText("刷新AI统计")
            self.rescan_all_button.setText("扫描全部")
            self.rescan_new_button.setText("扫描新增")
            self.rescan_missing_button.setText("扫描缺失")
            self.file_watch_checkbox.setText("自动监听文件变化")
            self.show_missing_files_button.setText("查看/修复丢失")
            self.clean_missing_index_button.setText("清理丢失")
            self.detect_duplicates_button.setText("检测重复")
            self.clean_orphan_thumbnails_button.setText("清理缩略图")
            self.rebuild_selected_thumbnails_button.setText("重建选中")
            self.remove_selected_index_button.setText("移除选中")
            self.run_performance_check_button.setText("性能压测")
            self.more_maintenance_button.setText("收起维护" if self.more_maintenance_button.isChecked() else "更多维护")
            self.export_library_button.setText("导出图库")
            self.export_selection_button.setText("导出图片")
            self.open_data_dir_button.setText("打开数据")
            self.backup_database_button.setText("备份数据库")
            self.restore_database_button.setText("恢复数据库")
            self.run_self_check_button.setText("启动自检")
            self.show_error_log_button.setText("错误日志")
            self.show_operation_history_button.setText("操作历史")
            self.path_remap_new_input.setPlaceholderText("选择移动后的新根目录")
            self.refresh_path_candidates_button.setText("刷新")
            self.choose_remap_new_path_button.setText("选择")
            self.apply_path_remap_button.setText("应用重映射")
            self.right_tab_widget.setTabText(0, "详情")
            self.right_tab_widget.setTabText(1, "AI")
            self.right_tab_widget.setTabText(2, "标签")
            self.right_tab_widget.setTabText(3, "索引")
            self.right_tab_widget.setTabText(4, "设置")
        self._update_color_swatch()
        self._refresh_ai_vision_value_filter_combo()

    def _choose_folder(self) -> None:
        collection_id = self._selected_collection_id()
        if collection_id is None:
            self.statusBar().showMessage("请先选择或新建一个 Eidory 文件夹")
            return
        filters = "Media Files (*.jpg *.jpeg *.png *.webp *.mp4 *.mov *.m4v *.avi *.mkv *.webm)"
        files, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "选择要导入的图片或视频",
            str(Path.home()),
            filters,
        )
        if files:
            self._start_file_import(files, collection_id)

    def _choose_folder_tree_import(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择要导入的磁盘文件夹")
        if folder:
            self._start_folder_tree_import([folder], parent_collection_id=None)

    def _rescan_selected_folder(self) -> None:
        item = self.folder_tree.currentItem()
        folder_path = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        if not folder_path:
            self.statusBar().showMessage("没有选中文件夹")
            return
        self._start_scan(folder_path)

    def _start_scan(self, folder_path: str) -> None:
        self.statusBar().showMessage(f"扫描中：{folder_path}")
        self._set_import_controls_enabled(False)

        def run() -> None:
            try:
                result = self.scanner.scan_folder(folder_path)
                self.events.put(("scan_done", result))
            except Exception as exc:
                self.events.put(("error", f"扫描失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=lambda: self._set_import_controls_enabled(True),
        )

    def _rescan_all_folders(self) -> None:
        folders = self.store.list_folders_with_collection_images()
        if not folders:
            self.statusBar().showMessage("没有可重新扫描的导入目录")
            return
        self.statusBar().showMessage(f"重新扫描全部导入目录：{len(folders)} 个")
        self._set_import_controls_enabled(False)
        self._set_maintenance_controls_enabled(False)
        self.settings_status_label.setText(f"正在重新扫描全部导入目录：{len(folders)} 个")

        def run() -> None:
            results: list[ScanResult] = []
            try:
                for folder in folders:
                    results.append(self.scanner.scan_folder(folder.folder_path))
                self.events.put(("scan_all_done", results))
            except Exception as exc:
                self.events.put(("error", f"重新扫描全部导入目录失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=self._restore_import_and_maintenance_controls,
        )

    def _rescan_new_or_changed_folders(self) -> None:
        folders = self.store.list_folders_with_collection_images()
        if not folders:
            self.statusBar().showMessage("没有可扫描的导入目录")
            return
        self.statusBar().showMessage(f"扫描新增/变化：{len(folders)} 个目录")
        self._set_import_controls_enabled(False)
        self._set_maintenance_controls_enabled(False)
        self.settings_status_label.setText("正在扫描新增/变化；不会处理已删除文件。")

        def run() -> None:
            results: list[ScanResult] = []
            try:
                for folder in folders:
                    results.append(self.scanner.scan_folder_new_only(folder.folder_path))
                self.events.put(("scan_new_done", results))
            except Exception as exc:
                self.events.put(("error", f"扫描新增/变化失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=self._restore_import_and_maintenance_controls,
        )

    def _rescan_missing_folders(self) -> None:
        folders = self.store.folders_with_missing_images()
        if not folders:
            self.statusBar().showMessage("没有包含丢失文件的导入目录")
            return
        self.statusBar().showMessage(f"扫描缺失所在目录：{len(folders)} 个")
        self._set_import_controls_enabled(False)
        self._set_maintenance_controls_enabled(False)
        self.settings_status_label.setText(f"正在重新扫描 {len(folders)} 个包含丢失文件的目录...")

        def run() -> None:
            results: list[ScanResult] = []
            try:
                for folder in folders:
                    results.append(self.scanner.scan_folder(folder.folder_path))
                self.events.put(("scan_missing_done", results))
            except Exception as exc:
                self.events.put(("error", f"扫描缺失所在目录失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=self._restore_import_and_maintenance_controls,
        )

    def _clean_missing_index(self) -> None:
        missing_count = self.store.count_missing_images()
        if missing_count == 0:
            self.statusBar().showMessage("没有丢失索引可清理")
            return
        confirm = QMessageBox.question(
            self,
            "清理丢失索引",
            f"将从 Eidory 中移除 {missing_count} 个丢失文件记录，不删除硬盘源文件。继续？",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        thumbnail_paths, removed = self.store.remove_missing_images_from_library()
        self._delete_thumbnail_files(thumbnail_paths)
        self.vector_index.invalidate()
        self._refresh_after_database_change()
        message = f"已清理 {removed} 个丢失索引"
        self.settings_status_label.setText(message)
        self.statusBar().showMessage(message)

    def _clean_orphan_thumbnails(self) -> None:
        thumbnail_root = self.paths.thumbnail_dir
        thumbnail_root.mkdir(parents=True, exist_ok=True)
        used_paths = {
            str(Path(path).resolve())
            for path in self.store.thumbnail_paths_in_use()
        }
        removed = 0
        for path in thumbnail_root.glob("thumb_*.webp"):
            try:
                resolved = str(path.resolve())
                if resolved in used_paths:
                    continue
                path.unlink(missing_ok=True)
                removed += 1
            except Exception as exc:
                self._record_error(f"清理孤立缩略图失败：{path}: {exc}")
        message = f"已清理 {removed} 个孤立缩略图"
        self.settings_status_label.setText(message)
        self.statusBar().showMessage(message)

    def _run_performance_check(self) -> None:
        self.run_performance_check_button.setEnabled(False)
        self.settings_status_label.setText("正在性能压测...")
        self.statusBar().showMessage("正在性能压测")

        def run() -> None:
            try:
                report = self._build_performance_report()
                self.events.put(("performance_done", report))
            except Exception as exc:
                self.events.put(("error", f"性能压测失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=lambda: self.run_performance_check_button.setEnabled(True),
        )

    def _build_performance_report(self) -> str:
        total_started = time.perf_counter()
        image_count = self.store.count_images()

        started = time.perf_counter()
        first_page = self.store.list_images(
            limit=500,
            sort_key=self._database_sort_key(),
            sort_desc=self.current_sort_desc,
        )
        first_page_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        sample = self.store.list_images(limit=5_000)
        sample_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        image_ids, matrix = self.store.embeddings_for_model(
            model_name=self.embedding_provider.model_name,
            model_revision=self.embedding_provider.model_revision,
            embedding_dim=self.embedding_provider.dim,
        )
        vector_load_ms = (time.perf_counter() - started) * 1000

        vector_search_ms: float | None = None
        if matrix.shape[0] > 0:
            started = time.perf_counter()
            normalized = matrix.astype(np.float32, copy=False)
            norms = np.linalg.norm(normalized, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = normalized / norms
            query = normalized[0]
            scores = normalized @ query
            top_k = min(500, scores.shape[0])
            if top_k > 0:
                np.argpartition(-scores, top_k - 1)[:top_k]
            vector_search_ms = (time.perf_counter() - started) * 1000

        total_ms = (time.perf_counter() - total_started) * 1000
        search_line = (
            f"NumPy 精确检索 top500：{vector_search_ms:.1f} ms"
            if vector_search_ms is not None
            else "NumPy 精确检索：无可用 embedding"
        )
        return (
            "性能压测完成\n"
            f"图片记录：{image_count}\n"
            f"首屏列表 500 条：{first_page_ms:.1f} ms / 返回 {len(first_page)}\n"
            f"样本列表 5000 条：{sample_ms:.1f} ms / 返回 {len(sample)}\n"
            f"向量加载：{vector_load_ms:.1f} ms / {len(image_ids)} 条\n"
            f"{search_line}\n"
            f"总耗时：{total_ms:.1f} ms"
        )

    def _export_library(self) -> None:
        target_dir = QFileDialog.getExistingDirectory(self, "选择图库导出位置")
        if not target_dir:
            return
        self._set_export_controls_enabled(False)
        self.settings_status_label.setText("正在导出图库...")
        self.statusBar().showMessage("正在导出图库")

        def run() -> None:
            try:
                result = export_library_to_directory(self.store, Path(target_dir))
                self.events.put(("export_done", ("图库导出", result)))
            except Exception as exc:
                self.events.put(("error", f"导出图库失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=lambda: self._set_export_controls_enabled(True),
        )

    def _export_selected_images(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        if len(images) > 1 and not self._confirm_batch_operation(
            "导出图片",
            "复制导出选中图片",
            images,
        ):
            return
        target_dir = QFileDialog.getExistingDirectory(self, "选择图片导出位置")
        if not target_dir:
            return
        self._set_export_controls_enabled(False)
        self.settings_status_label.setText(f"正在导出选中图片：{len(images)} 个...")
        self.statusBar().showMessage("正在导出选中图片")

        def run() -> None:
            try:
                result = export_images_to_directory(images, Path(target_dir))
                self.events.put(("export_done", ("选中图片导出", result)))
            except Exception as exc:
                self.events.put(("error", f"导出选中图片失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=lambda: self._set_export_controls_enabled(True),
        )

    def _set_export_controls_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        selection_enabled = enabled and bool(self._selected_grid_images())
        if (
            self._export_controls_enabled == enabled
            and self.export_selection_button.isEnabled() == selection_enabled
        ):
            return
        self._export_controls_enabled = enabled
        if self.export_library_button.isEnabled() != enabled:
            self.export_library_button.setEnabled(enabled)
        if self.export_selection_button.isEnabled() != selection_enabled:
            self.export_selection_button.setEnabled(selection_enabled)

    def _set_import_controls_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self._import_controls_enabled = enabled
        for widget in [self.add_folder_button, self.import_folder_tree_button, self.rescan_button]:
            if widget.isEnabled() != enabled:
                widget.setEnabled(enabled)

    def _restore_import_and_maintenance_controls(self) -> None:
        self._set_import_controls_enabled(True)
        self._set_maintenance_controls_enabled(True)

    def _restore_inspiration_generation_controls(self) -> None:
        self.generate_inspiration_button.setEnabled(True)
        self.search_inspiration_button.setEnabled(bool(self._selected_inspiration_terms()))

    def _restore_search_task_controls(self) -> None:
        self.search_button.setEnabled(True)
        self._restore_inspiration_generation_controls()

    def _on_ai_workflow_mode_clicked(self, mode_id: int) -> None:
        self._set_ai_workflow_mode("probe" if mode_id == 1 else "project")

    def _set_ai_workflow_mode(self, mode: str) -> None:
        if not hasattr(self, "ai_workflow_stack"):
            return
        is_probe = mode == "probe"
        self.ai_project_mode_button.blockSignals(True)
        self.ai_probe_mode_button.blockSignals(True)
        self.ai_project_mode_button.setChecked(not is_probe)
        self.ai_probe_mode_button.setChecked(is_probe)
        self.ai_project_mode_button.blockSignals(False)
        self.ai_probe_mode_button.blockSignals(False)
        self.ai_workflow_stack.setCurrentIndex(1 if is_probe else 0)

    def _toggle_more_maintenance(self, checked: bool) -> None:
        self.more_maintenance_widget.setVisible(checked)
        if self.current_language == "en":
            self.more_maintenance_button.setText("Hide More" if checked else "More Maintenance")
        else:
            self.more_maintenance_button.setText("收起维护" if checked else "更多维护")

    def _handle_export_done(self, payload: object) -> None:
        label, result = payload
        if not isinstance(result, ExportResult):
            return
        self._set_export_controls_enabled(True)
        message = (
            f"{label}完成：复制 {result.copied}，"
            f"跳过缺失 {result.skipped_missing}，失败 {result.failed}，"
            f"目录 {result.directories}。位置：{result.target_dir}"
        )
        self._record_operation_history(message)
        self.settings_status_label.setText(message)
        self.statusBar().showMessage(message)

    def _start_import(
        self,
        folder_path: str,
        collection_id: int | None,
        *,
        preserve_structure: bool,
    ) -> None:
        collection_name = self._collection_name(collection_id) if collection_id is not None else "全部文件夹"
        if collection_id is None and not preserve_structure:
            self.events.put(("error", "请先选择一个 Eidory 文件夹"))
            return
        self.statusBar().showMessage(f"导入中：{folder_path}")
        self._set_import_controls_enabled(False)

        def run() -> None:
            try:
                result = self.scanner.scan_folder(folder_path)
                assigned = self._assign_import_result(
                    result=result,
                    folder_path=folder_path,
                    collection_id=collection_id,
                    preserve_structure=preserve_structure,
                )
                self.events.put((
                    "import_done",
                    (result, collection_id, collection_name, assigned, preserve_structure),
                ))
            except Exception as exc:
                self.events.put(("error", f"导入失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=lambda: self._set_import_controls_enabled(True),
        )

    def _start_file_import(self, file_paths: list[str], collection_id: int) -> None:
        supported_files = [
            os.path.abspath(os.path.expanduser(path))
            for path in file_paths
            if os.path.isfile(os.path.abspath(os.path.expanduser(path)))
            and is_supported_media(os.path.abspath(os.path.expanduser(path)))
        ]
        if not supported_files:
            self.statusBar().showMessage("没有可导入的受支持图片或视频")
            return
        self._start_near_duplicate_resolution(
            supported_files,
            include_same_path=True,
            on_resolved=lambda accepted_paths, _skip_paths, replaced_ids, skipped_count: (
                self._continue_file_import_after_duplicate_resolution(
                    accepted_paths,
                    collection_id=collection_id,
                    replaced_count=len(replaced_ids),
                    skipped_count=skipped_count,
                )
            ),
        )
        return

    def _continue_file_import_after_duplicate_resolution(
        self,
        supported_files: list[str],
        *,
        collection_id: int,
        replaced_count: int,
        skipped_count: int,
    ) -> None:
        if not supported_files:
            self._refresh_after_scan_database_change(select_collection_id=collection_id)
            self.statusBar().showMessage(
                f"近似图片处理完成：替换 {replaced_count}，放弃 {skipped_count}，没有新图片导入"
            )
            return
        collection_name = self._collection_name(collection_id) or "当前文件夹"
        self.statusBar().showMessage(f"导入图片到“{collection_name}”")
        self._set_import_controls_enabled(False)

        def run() -> None:
            try:
                import_paths = self._copy_local_import_files_to_collection(
                    supported_files,
                    collection_id,
                )
                if not import_paths:
                    raise FileNotFoundError("没有可导入的受支持图片或视频")
                result = self.scanner.import_files(import_paths)
                assigned = self.store.assign_images_to_collection(
                    list(result.image_ids),
                    collection_id,
                )
                self.events.put((
                    "drop_import_done",
                    (
                        collection_id,
                        collection_name,
                        result.scanned_files,
                        result.new_files,
                        result.changed_files,
                        assigned,
                        list(result.image_ids),
                    ),
                ))
            except Exception as exc:
                self.events.put(("error", f"导入图片失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=lambda: self._set_import_controls_enabled(True),
        )

    def _start_folder_tree_import(
        self,
        folder_paths: list[str],
        *,
        parent_collection_id: int | None,
    ) -> None:
        folders = self._valid_import_folders(folder_paths)
        if not folders:
            self.statusBar().showMessage("没有可导入的磁盘文件夹")
            return
        folder_media_paths = self._folder_supported_media_paths(folders)
        self._start_near_duplicate_resolution(
            folder_media_paths,
            include_same_path=False,
            on_resolved=lambda _accepted_paths, skip_paths, _replaced_ids, _skipped_count: (
                self._continue_folder_tree_import_after_duplicate_resolution(
                    folders,
                    skip_paths=skip_paths,
                    parent_collection_id=parent_collection_id,
                )
            ),
        )
        return

    def _continue_folder_tree_import_after_duplicate_resolution(
        self,
        folders: list[str],
        *,
        skip_paths: set[str],
        parent_collection_id: int | None,
    ) -> None:
        parent_name = (
            self._collection_name(parent_collection_id)
            if parent_collection_id is not None
            else "根层级"
        ) or "文件夹"
        self.statusBar().showMessage(f"导入文件夹树到“{parent_name}”：{len(folders)} 个")
        self._set_import_controls_enabled(False)

        def run() -> None:
            try:
                results: list[ScanResult] = []
                assigned_total = 0
                imported_image_ids: list[int] = []
                for folder in folders:
                    result = self.scanner.scan_folder(folder, skip_paths=skip_paths)
                    results.append(result)
                    imported_image_ids.extend(result.image_ids)
                    assigned_total += self._assign_import_result(
                        result=result,
                        folder_path=folder,
                        collection_id=parent_collection_id,
                        preserve_structure=True,
                    )
                self.events.put((
                    "folder_tree_import_done",
                    (
                        results,
                        parent_collection_id,
                        parent_name,
                        assigned_total,
                        imported_image_ids,
                        folders,
                    ),
                ))
            except Exception as exc:
                self.events.put(("error", f"导入文件夹失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=lambda: self._set_import_controls_enabled(True),
        )

    def _start_local_paths_import(
        self,
        *,
        file_paths: list[str],
        folder_paths: list[str],
        parent_collection_id: int | None,
    ) -> None:
        target_name = (
            self._collection_name(parent_collection_id)
            if parent_collection_id is not None
            else "根层级"
        ) or "文件夹"
        folder_media_paths = self._folder_supported_media_paths(folder_paths)
        self._start_near_duplicate_resolution(
            file_paths,
            include_same_path=True,
            on_resolved=lambda accepted_files, file_skip_paths, file_replaced_ids, file_skipped_count: (
                self._start_near_duplicate_resolution(
                    folder_media_paths,
                    include_same_path=False,
                    on_resolved=lambda _accepted_folder_paths, folder_skip_paths, folder_replaced_ids, folder_skipped_count: (
                        self._continue_local_paths_import_after_duplicate_resolution(
                            target_name=target_name,
                            file_paths=accepted_files,
                            folder_paths=folder_paths,
                            parent_collection_id=parent_collection_id,
                            skip_paths=file_skip_paths | folder_skip_paths,
                            replaced_count=len(file_replaced_ids) + len(folder_replaced_ids),
                            skipped_count=file_skipped_count + folder_skipped_count,
                        )
                    ),
                )
            ),
        )
        return

    def _continue_local_paths_import_after_duplicate_resolution(
        self,
        *,
        target_name: str,
        file_paths: list[str],
        folder_paths: list[str],
        parent_collection_id: int | None,
        skip_paths: set[str],
        replaced_count: int,
        skipped_count: int,
    ) -> None:
        if not file_paths and not folder_paths:
            self._refresh_after_scan_database_change(select_collection_id=parent_collection_id)
            self.statusBar().showMessage(
                "近似图片处理完成："
                f"替换 {replaced_count}，"
                f"放弃 {skipped_count}，没有新图片导入"
            )
            return
        self.statusBar().showMessage(f"导入拖入内容到“{target_name}”")
        self._set_import_controls_enabled(False)

        def run() -> None:
            try:
                results: list[ScanResult] = []
                assigned_total = 0
                imported_image_ids: list[int] = []
                if file_paths:
                    if parent_collection_id is None:
                        raise ValueError("拖入图片必须先选择或拖到一个 Eidory 文件夹")
                    copied_file_paths = self._copy_local_import_files_to_collection(
                        file_paths,
                        parent_collection_id,
                    )
                    if not copied_file_paths:
                        raise FileNotFoundError("没有可导入的受支持图片或视频")
                    file_result = self.scanner.import_files(copied_file_paths)
                    results.append(file_result)
                    imported_image_ids.extend(file_result.image_ids)
                    assigned_total += self.store.assign_images_to_collection(
                        list(file_result.image_ids),
                        parent_collection_id,
                    )
                for folder in folder_paths:
                    folder_result = self.scanner.scan_folder(folder, skip_paths=skip_paths)
                    results.append(folder_result)
                    imported_image_ids.extend(folder_result.image_ids)
                    assigned_total += self._assign_import_result(
                        result=folder_result,
                        folder_path=folder,
                        collection_id=parent_collection_id,
                        preserve_structure=True,
                    )
                self.events.put((
                    "local_paths_import_done",
                    (
                        results,
                        parent_collection_id,
                        target_name,
                        assigned_total,
                        imported_image_ids,
                        file_paths,
                        folder_paths,
                    ),
                ))
            except Exception as exc:
                self.events.put(("error", f"拖入导入失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=lambda: self._set_import_controls_enabled(True),
        )

    @staticmethod
    def _valid_import_folders(folder_paths: list[str]) -> list[str]:
        folders: list[str] = []
        seen: set[str] = set()
        for raw_path in folder_paths:
            path = Path(os.path.abspath(os.path.expanduser(str(raw_path))))
            if not path.is_dir() or path.is_symlink() or path.name.startswith("."):
                continue
            normalized = str(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            folders.append(normalized)
        return folders

    def _start_near_duplicate_resolution(
        self,
        file_paths: list[str],
        *,
        include_same_path: bool,
        on_resolved: Callable[[list[str], set[str], list[int], int], None],
    ) -> None:
        normalized_paths = self._normalized_near_duplicate_import_paths(file_paths)
        if not normalized_paths:
            on_resolved([], set(), [], 0)
            return
        image_paths = [path for path in normalized_paths if is_supported_image(path)]
        if not image_paths:
            on_resolved(normalized_paths, set(), [], 0)
            return

        self._near_duplicate_job_counter += 1
        job_id = self._near_duplicate_job_counter
        self._near_duplicate_callbacks[job_id] = (on_resolved, normalized_paths)
        self.statusBar().showMessage("正在后台检查近似图片...")

        def run() -> None:
            try:
                candidate_map = self._near_duplicate_candidate_map(
                    normalized_paths,
                    include_same_path=include_same_path,
                )
                self.events.put((
                    "near_duplicate_candidates_ready",
                    (job_id, normalized_paths, candidate_map),
                ))
            except Exception as exc:
                self.events.put(("near_duplicate_candidates_failed", (job_id, str(exc))))

        self._start_background_task(
            run,
            name="near-duplicate-check",
            on_rejected=lambda: self._finish_rejected_near_duplicate_job(job_id),
        )

    def _finish_rejected_near_duplicate_job(self, job_id: int) -> None:
        entry = self._near_duplicate_callbacks.pop(job_id, None)
        if entry is not None:
            callback, _normalized_paths = entry
            callback([], set(), [], 0)

    @staticmethod
    def _normalized_near_duplicate_import_paths(file_paths: list[str]) -> list[str]:
        normalized_paths = [
            os.path.abspath(os.path.expanduser(str(path)))
            for path in file_paths
            if os.path.isfile(os.path.abspath(os.path.expanduser(str(path))))
            and is_supported_media(os.path.abspath(os.path.expanduser(str(path))))
        ]
        return sorted(set(normalized_paths))

    def _near_duplicate_candidate_map(
        self,
        file_paths: list[str],
        *,
        include_same_path: bool,
    ) -> dict[str, list[NearDuplicateCandidate]]:
        image_paths = [path for path in file_paths if is_supported_image(path)]
        candidates_by_path: dict[str, list[NearDuplicateCandidate]] = {}
        if include_same_path and image_paths:
            candidates_by_path.update(self._same_path_near_duplicate_candidates(image_paths))
        for file_path in image_paths:
            if file_path in candidates_by_path:
                continue
            candidates = self._find_import_near_duplicate_candidates(
                file_path,
                include_same_path=include_same_path,
            )
            if candidates:
                candidates_by_path[file_path] = candidates
        return candidates_by_path

    def _handle_near_duplicate_candidates_ready(self, payload: object) -> None:
        job_id, normalized_paths, candidate_map = payload
        entry = self._near_duplicate_callbacks.pop(int(job_id), None)
        if entry is None:
            return
        callback, _original_paths = entry
        accepted, skipped, replaced, skipped_count = self._resolve_near_duplicate_decisions(
            list(normalized_paths),
            dict(candidate_map),
        )
        callback(accepted, skipped, replaced, skipped_count)

    def _handle_near_duplicate_candidates_failed(self, payload: object) -> None:
        job_id, error = payload
        entry = self._near_duplicate_callbacks.pop(int(job_id), None)
        QMessageBox.warning(
            self,
            "近似图片检查失败",
            f"近似图片检查失败：{error}\n\n这次导入将继续进行。",
        )
        if entry is not None:
            callback, original_paths = entry
            callback(original_paths, set(), [], 0)

    def _resolve_near_duplicate_import_paths(
        self,
        file_paths: list[str],
        *,
        include_same_path: bool = False,
    ) -> tuple[list[str], set[str], list[int], int]:
        normalized_paths = self._normalized_near_duplicate_import_paths(file_paths)
        if not normalized_paths:
            return [], set(), [], 0
        candidate_map = self._near_duplicate_candidate_map(
            normalized_paths,
            include_same_path=include_same_path,
        )
        return self._resolve_near_duplicate_decisions(normalized_paths, candidate_map)

    def _resolve_near_duplicate_decisions(
        self,
        normalized_paths: list[str],
        candidate_map: dict[str, list[NearDuplicateCandidate]],
    ) -> tuple[list[str], set[str], list[int], int]:
        accepted_paths: list[str] = []
        skipped_paths: set[str] = set()
        replaced_image_ids: list[int] = []
        skipped_count = 0
        for file_path in normalized_paths:
            if not is_supported_image(file_path):
                accepted_paths.append(file_path)
                continue
            candidates = candidate_map.get(file_path, [])
            if not candidates:
                accepted_paths.append(file_path)
                continue
            decision, candidate = self._ask_near_duplicate_decision(file_path, candidates)
            if decision == NearDuplicateDecision.IMPORT:
                accepted_paths.append(file_path)
            elif decision == NearDuplicateDecision.REPLACE and candidate is not None:
                try:
                    replaced_id = self._replace_existing_image_from_import(
                        candidate.image,
                        Path(file_path),
                    )
                except Exception as exc:
                    QMessageBox.warning(
                        self,
                        "替换失败",
                        f"替换已有图片失败：{exc}\n\n这张新图片将继续按新图片导入。",
                    )
                    accepted_paths.append(file_path)
                    continue
                replaced_image_ids.append(replaced_id)
                skipped_paths.add(file_path)
                skipped_count += 1
            else:
                skipped_paths.add(file_path)
                skipped_count += 1
        return accepted_paths, skipped_paths, replaced_image_ids, skipped_count

    def _find_import_near_duplicate_candidates(
        self,
        file_path: str,
        *,
        include_same_path: bool,
    ) -> list[NearDuplicateCandidate]:
        width, height, file_size = self._quick_image_file_metadata(file_path)
        candidate_images = self.store.near_duplicate_metadata_candidates(
            width=width,
            height=height,
            file_size=file_size,
            limit=400,
        )
        if not candidate_images:
            return []
        exact_metadata_images = [
            image
            for image in candidate_images
            if image.width == width
            and image.height == height
            and image.file_size == file_size
        ]
        candidates = self._near_duplicate_candidates_from_images(
            file_path,
            exact_metadata_images,
            include_same_path=include_same_path,
        )
        if candidates:
            return candidates
        exact_image_ids = {image.id for image in exact_metadata_images}
        remaining_images = [
            image
            for image in candidate_images
            if image.id not in exact_image_ids
        ]
        return self._near_duplicate_candidates_from_images(
            file_path,
            remaining_images,
            include_same_path=include_same_path,
        )

    @staticmethod
    def _near_duplicate_candidates_from_images(
        file_path: str,
        images: list[ImageItem],
        *,
        include_same_path: bool,
    ) -> list[NearDuplicateCandidate]:
        if not images:
            return []
        hash_records = build_image_dhash_records(images)
        return find_near_duplicate_candidates(
            file_path,
            hash_records=hash_records,
            near_distance=8,
            limit=5,
            include_same_path=include_same_path,
        )

    @staticmethod
    def _quick_image_file_metadata(file_path: str) -> tuple[int | None, int | None, int]:
        try:
            file_size = Path(file_path).stat().st_size
        except OSError:
            file_size = 0
        reader = QImageReader(file_path)
        reader.setAutoTransform(True)
        size = reader.size()
        if not size.isValid():
            return None, None, file_size
        return int(size.width()), int(size.height()), int(file_size)

    def _same_path_near_duplicate_candidates(
        self,
        file_paths: list[str],
    ) -> dict[str, list[NearDuplicateCandidate]]:
        candidates_by_path: dict[str, list[NearDuplicateCandidate]] = {}
        for file_path in file_paths:
            image = self.store.get_image_by_path(file_path)
            if image is None:
                continue
            candidates_by_path[file_path] = [
                NearDuplicateCandidate(
                    image=image,
                    distance=0,
                    similarity=1.0,
                    hash_source=image.thumbnail_path or image.file_path,
                )
            ]
        return candidates_by_path

    def _build_near_duplicate_hash_records(self) -> list[ImageDHashRecord]:
        image_count = max(self.store.count_images(), 1)
        if (
            self._near_duplicate_hash_records_cache is not None
            and self._near_duplicate_hash_records_cache_count == image_count
        ):
            return self._near_duplicate_hash_records_cache
        images = self.store.list_images(limit=image_count, include_missing=False)
        records = build_image_dhash_records(images)
        self._near_duplicate_hash_records_cache = records
        self._near_duplicate_hash_records_cache_count = image_count
        return records

    def _invalidate_near_duplicate_hash_cache(self) -> None:
        self._near_duplicate_hash_records_cache = None
        self._near_duplicate_hash_records_cache_count = None

    def _ask_near_duplicate_decision(
        self,
        file_path: str,
        candidates: list[NearDuplicateCandidate],
    ) -> tuple[str, NearDuplicateCandidate | None]:
        dialog = NearDuplicateDialog(file_path, candidates, self)
        dialog.exec()
        return dialog.decision, dialog.selected_candidate()

    def _replace_existing_image_from_import(
        self,
        existing_image: ImageItem,
        source_path: Path,
    ) -> int:
        target_path = Path(existing_image.file_path)
        if not target_path.parent.is_dir():
            raise FileNotFoundError(f"已有图片所在文件夹不存在：{target_path.parent}")
        if source_path.resolve() != target_path.resolve():
            temp_path = target_path.with_name(
                f".{target_path.name}.eidory-replace-{uuid.uuid4().hex}.tmp"
            )
            try:
                shutil.copy2(source_path, temp_path)
                os.replace(temp_path, target_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        result = self.scanner.import_files([str(target_path)])
        if result.image_ids:
            self._invalidate_near_duplicate_hash_cache()
            self.vector_index.invalidate()
            self._refresh_embedding_stats()
            self._refresh_ai_vision_stats()
            return int(result.image_ids[0])
        return existing_image.id

    @staticmethod
    def _folder_supported_media_paths(folder_paths: list[str]) -> list[str]:
        paths: list[str] = []
        for folder in folder_paths:
            if os.path.isdir(folder):
                paths.extend(ImageScanner._iter_image_files(folder))
        return sorted(set(paths))

    @staticmethod
    def _remove_materialized_import_files(file_paths: list[str], skip_paths: set[str]) -> None:
        for file_path in skip_paths:
            if file_path not in file_paths:
                continue
            try:
                Path(file_path).unlink()
            except FileNotFoundError:
                continue
            except OSError:
                continue

    def _assign_import_result(
        self,
        *,
        result: ScanResult,
        folder_path: str,
        collection_id: int | None,
        preserve_structure: bool,
    ) -> int:
        image_ids = list(result.image_ids)
        if not image_ids:
            return 0
        if not preserve_structure:
            if collection_id is None:
                return 0
            return self.store.assign_images_to_collection(image_ids, collection_id)

        base = self._normalize_folder_path(folder_path)
        base_name = Path(base).name or base
        root_collection_id = self.store.ensure_collection_path([base_name], parent_id=collection_id)
        if root_collection_id is None:
            return 0

        assigned = 0
        for image in self.store.images_by_ids(image_ids):
            image_dir = self._normalize_folder_path(os.path.dirname(image.file_path))
            relative_dir = os.path.relpath(image_dir, base)
            names = []
            if relative_dir != ".":
                names.extend(part for part in relative_dir.split(os.sep) if part)
            target_id = (
                self.store.ensure_collection_path(names, parent_id=root_collection_id)
                if names
                else root_collection_id
            )
            if target_id is not None:
                assigned += self.store.assign_images_to_collection([image.id], target_id)
        return assigned

    def _run_search(self) -> None:
        self._show_gallery_view()
        search_filter = self._search_filter_from_controls()
        if search_filter is None:
            return
        if self.reverse_exclusion_button.isChecked():
            self._start_reverse_exclusion_with_filter(search_filter)
            return
        self._start_search_with_filter(search_filter)

    def _generate_inspiration_terms_from_panel(self) -> None:
        self._set_ai_workflow_mode("probe")
        brief = self.inspiration_brief_input.toPlainText().strip()
        if not brief:
            self.inspiration_status_label.setText("先输入创作主题。")
            return
        self.inspiration_plan_filters = []
        self.current_inspiration_plan_filters = []
        self.current_inspiration_raw_term_results = []
        self._populate_inspiration_filter_list([])
        self.generate_inspiration_button.setEnabled(False)
        self.search_inspiration_button.setEnabled(False)
        service = self._llm_service_key()
        use_ai_scene_tags = self.use_ai_scene_tags_checkbox.isChecked()
        self.inspiration_status_label.setText(
            f"正在请求 {self._llm_service_label(service)} 生成"
            f"{'语义探针和 AI 场景标签' if use_ai_scene_tags else '语义探针'}..."
        )
        provider = self._make_llm_provider()
        answers = self.inspiration_answers_input.toPlainText()

        def run() -> None:
            try:
                if use_ai_scene_tags:
                    proposal = provider.generate_search_plan(
                        brief=brief,
                        answers=answers,
                        language=self.current_language,
                    )
                    self.events.put(("search_plan_proposal", proposal))
                else:
                    proposal = provider.generate_inspiration_terms(
                        brief=brief,
                        answers=answers,
                        language=self.current_language,
                    )
                    self.events.put(("inspiration_proposal", proposal))
            except Exception as exc:
                self.events.put(("inspiration_error", exc))

        self._start_background_task(
            run,
            on_rejected=self._restore_inspiration_generation_controls,
        )

    def _show_inspiration_proposal(self, proposal) -> None:
        self._set_ai_workflow_mode("probe")
        self.current_inspiration_project_id = None
        self.inspiration_proposal_terms = list(proposal.terms)
        self.inspiration_plan_filters = []
        self.current_inspiration_plan_filters = []
        self.current_inspiration_raw_term_results = []
        self.inspiration_questions = list(proposal.questions)
        self.inspiration_model_name = proposal.model_name
        self._populate_inspiration_term_list(
            self.inspiration_proposal_terms,
            default_selected_count=5,
        )
        self._populate_inspiration_filter_list([])
        self._sync_inspiration_questions_label()
        self._refresh_inspiration_status()
        self.search_inspiration_button.setEnabled(bool(self._selected_inspiration_terms()))

    def _show_search_plan_proposal(self, proposal) -> None:
        self._set_ai_workflow_mode("probe")
        self.current_inspiration_project_id = None
        self.inspiration_proposal_terms = list(proposal.terms)
        self.inspiration_plan_filters = list(proposal.filters)
        self.current_inspiration_plan_filters = []
        self.current_inspiration_raw_term_results = []
        self.inspiration_questions = list(proposal.questions)
        self.inspiration_model_name = proposal.model_name
        self._populate_inspiration_term_list(
            self.inspiration_proposal_terms,
            default_selected_count=5,
        )
        self._populate_inspiration_filter_list(self.inspiration_plan_filters)
        self._sync_inspiration_questions_label()
        self._refresh_inspiration_status()
        self.search_inspiration_button.setEnabled(bool(self._selected_inspiration_terms()))

    def _populate_inspiration_term_list(
        self,
        terms: list[InspirationTerm],
        *,
        default_selected_count: int | None = None,
    ) -> None:
        self.inspiration_term_list.blockSignals(True)
        self.inspiration_term_list.clear()
        for index, term in enumerate(terms):
            item = QListWidgetItem(f"{term.title}\n{term.query}\n{term.reason}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = term.selected or (
                default_selected_count is not None and index < default_selected_count
            )
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, term)
            item.setSizeHint(QSize(0, 58))
            item.setToolTip(f"{term.query}\n{term.reason}")
            self.inspiration_term_list.addItem(item)
        self.inspiration_term_list.blockSignals(False)

    def _populate_inspiration_filter_list(self, filters: list[SearchPlanFilter]) -> None:
        if not hasattr(self, "inspiration_filter_list"):
            return
        self.inspiration_filter_list.blockSignals(True)
        self.inspiration_filter_list.clear()
        for plan_filter in filters:
            label = ai_vision_label(
                plan_filter.field,
                plan_filter.value,
                language=self.current_language,
            )
            prefix = "Optional" if self.current_language == "en" else "可选"
            text = label if not plan_filter.optional else f"{prefix}：{label}"
            if plan_filter.reason:
                text = f"{text}\n{plan_filter.reason}"
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Unchecked if plan_filter.optional else Qt.CheckState.Checked
            )
            item.setData(Qt.ItemDataRole.UserRole, plan_filter)
            item.setSizeHint(QSize(0, 42 if plan_filter.reason else 28))
            item.setToolTip(text)
            self.inspiration_filter_list.addItem(item)
        self.inspiration_filter_list.blockSignals(False)

    def _sync_inspiration_questions_label(self) -> None:
        question_prefix = "AI questions: " if self.current_language == "en" else "AI 追问："
        self.inspiration_questions_label.setText(
            question_prefix + " / ".join(self.inspiration_questions)
            if self.inspiration_questions
            else question_prefix + "-"
        )

    def _show_inspiration_error(self, payload: object) -> None:
        message = str(payload) if isinstance(payload, LLMProviderError) else f"生成失败：{payload}"
        self.inspiration_status_label.setText(message)
        self.statusBar().showMessage(message)

    def _refresh_inspiration_history(self, select_project_id: int | None = None) -> None:
        if not hasattr(self, "inspiration_history_list"):
            return
        self.inspiration_history_list.blockSignals(True)
        self.inspiration_history_list.clear()
        selected_item: QListWidgetItem | None = None
        for project in self.store.list_inspiration_projects():
            item = QListWidgetItem(
                f"{project.title}    {project.selected_count}/{project.term_count}"
            )
            item.setData(Qt.ItemDataRole.UserRole, project.id)
            item.setToolTip(
                "\n".join([
                    project.brief,
                    project.answers,
                    f"{project.provider_name} / {project.model_name}",
                ]).strip()
            )
            self.inspiration_history_list.addItem(item)
            if project.id == select_project_id:
                selected_item = item
        if selected_item is not None:
            self.inspiration_history_list.setCurrentItem(selected_item)
        self.inspiration_history_list.blockSignals(False)

    def _load_selected_inspiration_history(self, item: QListWidgetItem | None = None) -> None:
        item = item or self.inspiration_history_list.currentItem()
        if item is None:
            return
        project_id = item.data(Qt.ItemDataRole.UserRole)
        if project_id is None:
            return
        self._load_inspiration_project(int(project_id))

    def _load_inspiration_project(self, project_id: int) -> None:
        project = self.store.get_inspiration_project(project_id)
        if project is None:
            self._refresh_inspiration_history()
            self.statusBar().showMessage("该 AI 探针历史已不存在")
            return
        self._set_ai_workflow_mode("probe")
        terms = self.store.inspiration_terms_for_project(project_id)
        self.current_inspiration_project_id = project.id
        self.inspiration_proposal_terms = list(terms)
        self.inspiration_plan_filters = []
        self.current_inspiration_plan_filters = []
        self.current_inspiration_raw_term_results = []
        self.inspiration_questions = list(project.questions)
        self.inspiration_model_name = project.model_name
        self.inspiration_brief_input.setPlainText(project.brief)
        self.inspiration_answers_input.setPlainText(project.answers)
        self._populate_inspiration_term_list(self.inspiration_proposal_terms)
        self._populate_inspiration_filter_list([])
        self._sync_inspiration_questions_label()
        self._refresh_inspiration_status()
        self.search_inspiration_button.setEnabled(bool(self._selected_inspiration_terms()))
        self._refresh_inspiration_history(select_project_id=project_id)
        self.statusBar().showMessage(f"已恢复 AI 探针历史：{project.title}")

    def _show_inspiration_history_context_menu(self, position) -> None:
        item = self.inspiration_history_list.itemAt(position)
        if item is None:
            return
        self.inspiration_history_list.setCurrentItem(item)
        project_id = item.data(Qt.ItemDataRole.UserRole)
        if project_id is None:
            return
        menu = QMenu(self)
        restore_action = menu.addAction("恢复此探针")
        delete_action = menu.addAction("删除历史")
        action = menu.exec(self.inspiration_history_list.viewport().mapToGlobal(position))
        if action == restore_action:
            self._load_inspiration_project(int(project_id))
        elif action == delete_action:
            self._delete_inspiration_project(int(project_id))

    def _delete_inspiration_project(self, project_id: int) -> None:
        project = self.store.get_inspiration_project(project_id)
        if project is None:
            self._refresh_inspiration_history()
            self.statusBar().showMessage("该 AI 探针历史已不存在")
            return
        answer = QMessageBox.question(
            self,
            "删除 AI 探针历史",
            f"删除“{project.title}”？这只删除探针历史，不会删除图片或语义探针项目。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        deleted = self.store.delete_inspiration_project(project_id)
        if self.current_inspiration_project_id == project_id:
            self.current_inspiration_project_id = None
        self._refresh_inspiration_history()
        if deleted:
            self.statusBar().showMessage(f"已删除 AI 探针历史：{project.title}")

    def _enforce_inspiration_selection_limit(self, changed_item: QListWidgetItem) -> None:
        selected = self._selected_inspiration_terms()
        if len(selected) > 7:
            self.inspiration_term_list.blockSignals(True)
            changed_item.setCheckState(Qt.CheckState.Unchecked)
            self.inspiration_term_list.blockSignals(False)
            self.inspiration_status_label.setText("最多选择 7 个语义探针。")
            return
        self._refresh_inspiration_status()
        self.search_inspiration_button.setEnabled(bool(selected))

    def _refresh_inspiration_status(self) -> None:
        count = len(self._selected_inspiration_terms())
        total = self.inspiration_term_list.count()
        filter_count = len(self._selected_inspiration_plan_filters())
        if total == 0:
            self.inspiration_status_label.setText("生成后最多选择 7 个语义探针；右键探针可单条搜索。")
        else:
            scoped_count = sum(
                1
                for term in self._selected_inspiration_terms()
                if self._inspiration_term_uses_plan_filters(term)
            )
            self.inspiration_status_label.setText(
                f"已选择 {count} / 7 个语义探针，{filter_count} 个 AI 场景标签；"
                "保存并搜索会混排所有已选探针，"
                f"筛选作用于 {scoped_count} 个场景类探针。"
            )

    def _selected_inspiration_terms(self) -> list[InspirationTerm]:
        terms: list[InspirationTerm] = []
        for row in range(self.inspiration_term_list.count()):
            item = self.inspiration_term_list.item(row)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            term = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(term, InspirationTerm):
                terms.append(InspirationTerm(
                    id=term.id,
                    title=term.title,
                    query=term.query,
                    axis=term.axis,
                    reason=term.reason,
                    selected=True,
                ))
        return terms

    def _selected_inspiration_plan_filters(self) -> list[SearchPlanFilter]:
        filters: list[SearchPlanFilter] = []
        for row in range(self.inspiration_filter_list.count()):
            item = self.inspiration_filter_list.item(row)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            plan_filter = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(plan_filter, SearchPlanFilter):
                filters.append(plan_filter)
        return filters

    def _show_inspiration_term_context_menu(self, position) -> None:
        item = self.inspiration_term_list.itemAt(position)
        if item is None:
            return
        term = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(term, InspirationTerm):
            return
        menu = QMenu(self)
        search_action = menu.addAction("单独搜索此探针")
        action = menu.exec(self.inspiration_term_list.viewport().mapToGlobal(position))
        if action == search_action:
            self._search_clicked_inspiration_term(item)

    def _search_clicked_inspiration_term(self, item: QListWidgetItem) -> None:
        term = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(term, InspirationTerm):
            return
        self._run_single_inspiration_term_search(term)

    def _run_single_inspiration_term_search(self, term: InspirationTerm) -> None:
        self.inspiration_status_label.setText(f"正在单条搜索：{term.title}")
        self._run_inspiration_search(
            self.current_inspiration_project_id or 0,
            [
                InspirationTerm(
                    id=term.id,
                    title=term.title,
                    query=term.query,
                    axis=term.axis,
                    reason=term.reason,
                    selected=True,
                )
            ],
            plan_filters=self._selected_inspiration_plan_filters(),
        )

    def _save_and_search_inspiration(self) -> None:
        selected_terms = self._selected_inspiration_terms()
        if not selected_terms:
            self.inspiration_status_label.setText("至少选择 1 个语义探针。")
            return
        if len(selected_terms) > 7:
            self.inspiration_status_label.setText("最多选择 7 个语义探针。")
            return
        brief = self.inspiration_brief_input.toPlainText().strip()
        if not brief:
            self.inspiration_status_label.setText("先输入创作主题。")
            return
        selected_titles = {term.title for term in selected_terms}
        plan_filters = self._selected_inspiration_plan_filters()
        project_id = self.current_inspiration_project_id
        if project_id is not None and self.store.get_inspiration_project(project_id) is not None:
            self.store.update_inspiration_project_selection(
                project_id,
                selected_titles=selected_titles,
            )
        else:
            service = self._llm_service_key()
            project_id = self.store.create_inspiration_project(
                title=brief[:28] or "灵感项目",
                brief=brief,
                answers=self.inspiration_answers_input.toPlainText(),
                questions=self.inspiration_questions,
                provider_name=self._llm_service_label(service),
                model_name=self.inspiration_model_name or self._llm_model(service) or "local-model",
                terms=self.inspiration_proposal_terms,
                selected_titles=selected_titles,
            )
        self._refresh_inspiration_history(select_project_id=project_id)
        self._run_inspiration_search(project_id, selected_terms, plan_filters=plan_filters)

    def _run_inspiration_search(
        self,
        project_id: int,
        selected_terms: list[InspirationTerm],
        *,
        plan_filters: list[SearchPlanFilter] | None = None,
    ) -> None:
        if not selected_terms:
            self.statusBar().showMessage("没有可搜索的语义探针")
            return
        self.semantic_search_revision += 1
        self._clear_manual_result_order()
        self._clear_result_management_state()
        revision = self.semantic_search_revision
        folder_path_prefix = self._selected_folder_path_prefix()
        collection_id = self._selected_collection_id()
        virtual_filter = self._selected_virtual_filter()
        tag_ids: list[int] = []
        tag_match_mode = "any"
        status_filter = self._selected_status_filter()
        self.current_result_mode = "inspiration"
        self.current_inspiration_project_id = project_id
        self.current_inspiration_terms = list(selected_terms)
        self.current_inspiration_plan_filters = list(plan_filters or [])
        self.current_inspiration_raw_term_results = []
        self.current_inspiration_images = []
        self.current_inspiration_filtered_images = []
        self.current_inspiration_matches = {}
        self.current_temp_project_id = None
        self.current_temp_project_images = []
        self.current_temp_project_badges = {}
        self.search_filters.clear()
        self.active_filter_index = None
        self.current_offset = 0
        self.load_more_button.setEnabled(False)
        self.generate_inspiration_button.setEnabled(False)
        self.search_inspiration_button.setEnabled(False)
        filter_text = (
            f"，{len(self.current_inspiration_plan_filters)} 个 AI 场景标签"
            if self.current_inspiration_plan_filters else ""
        )
        self._set_result_status(f"灵感项目搜索中：{len(selected_terms)} 个语义探针{filter_text}")
        self.search_diagnostics_label.setText("搜索诊断：-")

        def run() -> None:
            try:
                term_results = []
                for term in selected_terms:
                    result = self.search_service.semantic_search(
                        term.query,
                        folder_path_prefix=folder_path_prefix,
                        collection_id=collection_id,
                        virtual_filter=virtual_filter,
                        tag_ids=tag_ids,
                        tag_match_mode=tag_match_mode,
                        status_filter=status_filter,
                    )
                    term_results.append((term, list(result.images)))
                self.events.put((
                    "inspiration_done",
                    (revision, project_id, selected_terms, term_results, list(plan_filters or [])),
                ))
            except Exception as exc:
                self.events.put(("error", f"灵感项目搜索失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=self._restore_search_task_controls,
        )

    def _handle_inspiration_filter_changed(self, _item: QListWidgetItem | None = None) -> None:
        self._refresh_inspiration_status()
        if self.current_result_mode != "inspiration" or not self.current_inspiration_raw_term_results:
            return
        self.current_inspiration_plan_filters = self._selected_inspiration_plan_filters()
        self._rebuild_current_inspiration_results_from_raw()
        self._apply_inspiration_result_filters()
        images = self.current_inspiration_filtered_images
        self.grid_view.set_images(images, badges_by_image_id=self._inspiration_badges_by_image_id())
        self._set_inspiration_result_status(images)
        self._update_inspiration_diagnostics(images)

    def _rebuild_current_inspiration_results_from_raw(self) -> None:
        result = self._mix_inspiration_raw_term_results(
            self.current_inspiration_raw_term_results,
            self.current_inspiration_plan_filters,
        )
        self.current_inspiration_images = list(result.images)
        self.current_inspiration_matches = dict(result.matches_by_image_id)

    def _mix_inspiration_raw_term_results(
        self,
        term_results: list[tuple[InspirationTerm, list[ImageItem]]],
        plan_filters: list[SearchPlanFilter],
    ):
        scoped_results: list[tuple[InspirationTerm, list[ImageItem]]] = []
        for term, images in term_results:
            scoped_images = self._images_for_inspiration_term_with_plan_filters(
                term,
                list(images),
                plan_filters,
            )
            scoped_results.append((term, scoped_images[:100]))
        return mix_inspiration_search_results(scoped_results, limit=500)

    def _images_for_inspiration_term_with_plan_filters(
        self,
        term: InspirationTerm,
        images: list[ImageItem],
        plan_filters: list[SearchPlanFilter],
    ) -> list[ImageItem]:
        if not plan_filters or not self._inspiration_term_uses_plan_filters(term):
            return list(images)
        return self._apply_inspiration_plan_filters_to_images(images, plan_filters)

    def _inspiration_term_uses_plan_filters(self, term: InspirationTerm) -> bool:
        axis = term.axis.strip().casefold()
        scene_axes = {"environment", "lighting", "mood", "composition", "era"}
        object_axes = {"object_detail", "object", "material", "character", "vehicle", "prop"}
        if axis in scene_axes:
            return True
        if axis in object_axes:
            return False
        text = f"{term.title} {term.query} {term.reason}".casefold()
        object_markers = (
            "摩托", "车辆", "飞行器", "引擎", "发动机", "机械", "结构", "造型", "细节",
            "物件", "道具", "装备", "服装", "材质", "纹理", "machine", "vehicle",
            "motorcycle", "engine", "object", "prop", "material", "texture", "detail",
        )
        scene_markers = (
            "室内", "室外", "场景", "环境", "天气", "白天", "夜晚", "清晨", "黄昏",
            "光照", "逆光", "侧光", "低调", "高调", "气氛", "氛围", "构图", "视角",
            "interior", "exterior", "environment", "scene", "weather", "day", "night",
            "dawn", "dusk", "lighting", "mood", "atmosphere", "composition", "angle",
        )
        if any(marker in text for marker in object_markers):
            return False
        return any(marker in text for marker in scene_markers)

    def _apply_inspiration_plan_filters_to_images(
        self,
        images: list[ImageItem],
        plan_filters: list[SearchPlanFilter],
    ) -> list[ImageItem]:
        if not images or not plan_filters:
            return list(images)
        allowed_by_field: dict[str, set[int]] = {}
        for plan_filter in plan_filters:
            matching_ids = self.store.image_ids_matching_ai_vision(
                plan_filter.field,
                plan_filter.value,
            )
            if plan_filter.field in allowed_by_field:
                allowed_by_field[plan_filter.field].update(matching_ids)
            else:
                allowed_by_field[plan_filter.field] = set(matching_ids)
        if not allowed_by_field:
            return list(images)
        return [
            image
            for image in images
            if all(image.id in allowed_ids for allowed_ids in allowed_by_field.values())
        ]

    def _find_similar_to_selected_image(self) -> None:
        self._find_similar_to_image(self._selected_grid_image())

    def _find_similar_to_image(self, image: ImageItem | None) -> None:
        if self.reverse_exclusion_button.isChecked():
            self.statusBar().showMessage("反向排除暂不支持相似图")
            return
        reason = self._similar_image_blocking_reason(image, check_vector=True)
        if reason is not None:
            self.statusBar().showMessage(reason)
            return
        assert image is not None
        self._start_search_with_filter(SearchFilter("similar", image.id))

    def _similar_image_blocking_reason(
        self,
        image: ImageItem | None,
        *,
        check_vector: bool,
    ) -> str | None:
        if image is None:
            return "请先选中一张图片"
        if image.file_ext not in SUPPORTED_IMAGE_EXTENSIONS:
            return "视频暂不支持相似图片搜索"
        if image.is_missing:
            return "源文件丢失，不能查找相似图片"
        if image.embedding_status != "ready":
            return "这张图片还没有完成语义索引，先在索引页开始索引"
        if check_vector and self.store.embedding_vector_for_image(
            image.id,
            model_name=self.embedding_provider.model_name,
            model_revision=self.embedding_provider.model_revision,
            embedding_dim=self.embedding_provider.dim,
        ) is None:
            return "当前模型没有这张图片的语义向量，请重新索引"
        return None

    def _add_file_type_filter_from_controls(self) -> None:
        value = self.file_type_filter_combo.currentData()
        if not value:
            self.statusBar().showMessage("请选择文件类型")
            return
        self._start_search_with_filter(SearchFilter("file_type", str(value)))

    def _add_dimension_filter_from_controls(self) -> None:
        value = self.dimension_filter_combo.currentData()
        if not value:
            self.statusBar().showMessage("请选择尺寸或方向")
            return
        kind, _separator, filter_value = str(value).partition(":")
        if kind not in {"orientation", "size"} or not filter_value:
            self.statusBar().showMessage("未知的尺寸筛选条件")
            return
        self._start_search_with_filter(SearchFilter(kind, filter_value))

    def _refresh_ai_vision_value_filter_combo(self) -> None:
        if not hasattr(self, "ai_vision_field_filter_combo"):
            return
        field = str(self.ai_vision_field_filter_combo.currentData() or "scene_location")
        previous = self.ai_vision_value_filter_combo.currentData() if hasattr(self, "ai_vision_value_filter_combo") else None
        values = AI_VISION_LIGHTING_VALUES if field == "lighting" else AI_VISION_FIELD_VALUES.get(field, [])
        self.ai_vision_value_filter_combo.blockSignals(True)
        self.ai_vision_value_filter_combo.clear()
        for value in values:
            label = ai_vision_label(field, value, language=self.current_language)
            label = label.split(": ", 1)[1] if ": " in label else label
            self.ai_vision_value_filter_combo.addItem(label, value)
        if previous is not None:
            self._set_combo_to_data(self.ai_vision_value_filter_combo, previous)
        self.ai_vision_value_filter_combo.blockSignals(False)

    def _add_ai_vision_filter_from_controls(self) -> None:
        field = str(self.ai_vision_field_filter_combo.currentData() or "")
        value = str(self.ai_vision_value_filter_combo.currentData() or "")
        if not field or not value:
            self.statusBar().showMessage("请选择 AI 场景筛选条件")
            return
        self._start_search_with_filter(SearchFilter("ai_vision", f"{field}:{value}"))

    def _save_current_view(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "保存筛选预设",
            "预设名称：",
            QLineEdit.EchoMode.Normal,
            self._suggest_saved_view_name(),
        )
        if not ok:
            return
        clean_name = name.strip()
        if not clean_name:
            self.statusBar().showMessage("预设名称不能为空")
            return
        payload = self._current_view_payload()
        saved_view_id = self.store.upsert_saved_view(
            clean_name,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )
        self._refresh_saved_views(select_saved_view_id=saved_view_id)
        self.statusBar().showMessage(f"已保存筛选预设：{clean_name}")

    def _apply_selected_saved_view(self) -> None:
        saved_view = self._selected_saved_view()
        if saved_view is None:
            self.statusBar().showMessage("请先选择一个筛选预设")
            return
        try:
            payload = json.loads(saved_view.payload_json)
        except json.JSONDecodeError:
            QMessageBox.warning(self, "Eidory", "该筛选预设数据损坏，无法载入。")
            return
        self._apply_view_payload(payload)
        self.statusBar().showMessage(f"已载入筛选预设：{saved_view.name}")

    def _rename_selected_saved_view(self) -> None:
        saved_view = self._selected_saved_view()
        if saved_view is None:
            self.statusBar().showMessage("请先选择一个筛选预设")
            return
        name, ok = QInputDialog.getText(
            self,
            "重命名筛选预设",
            "新名称：",
            QLineEdit.EchoMode.Normal,
            saved_view.name,
        )
        if not ok:
            return
        clean_name = name.strip()
        if not clean_name:
            self.statusBar().showMessage("预设名称不能为空")
            return
        try:
            changed = self.store.rename_saved_view(saved_view.id, clean_name)
        except ValueError:
            QMessageBox.warning(self, "Eidory", "该预设名称已存在。")
            return
        if changed:
            self._refresh_saved_views(select_saved_view_id=saved_view.id)
            self.statusBar().showMessage(f"筛选预设已重命名为：{clean_name}")

    def _delete_selected_saved_view(self) -> None:
        saved_view = self._selected_saved_view()
        if saved_view is None:
            self.statusBar().showMessage("请先选择一个筛选预设")
            return
        answer = QMessageBox.question(
            self,
            "删除筛选预设",
            f"删除筛选预设“{saved_view.name}”？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.store.delete_saved_view(saved_view.id):
            self._refresh_saved_views()
            self.statusBar().showMessage(f"已删除筛选预设：{saved_view.name}")

    def _search_filter_from_controls(self) -> SearchFilter | None:
        mode = self._selected_search_mode()
        if mode == "color":
            return SearchFilter("color", self.current_color_rgb)

        query = self.search_input.text().strip()
        if not query:
            self.statusBar().showMessage("请输入搜索内容")
            return None
        if mode == "semantic":
            return SearchFilter("semantic", query)
        return SearchFilter("keyword", query)

    def _selected_search_operation_mode(self) -> str:
        if not self._has_visible_result_context():
            return "replace"
        if self.search_merge_results_button.isChecked():
            return "merge"
        if self.search_replace_results_button.isChecked():
            return "replace"
        return "refine"

    def _set_search_operation_mode(self, mode: str) -> None:
        if mode == "merge":
            self.search_merge_results_button.setChecked(True)
        elif mode == "replace":
            self.search_replace_results_button.setChecked(True)
        else:
            self.search_within_results_button.setChecked(True)

    def _refresh_search_operation_controls(self) -> None:
        if not hasattr(self, "search_within_results_button"):
            return
        has_results = self._has_visible_result_context()
        self.search_within_results_button.setEnabled(has_results)
        self.search_merge_results_button.setEnabled(has_results)
        if not has_results:
            self.search_replace_results_button.setChecked(True)

    def _start_search_with_filter(self, search_filter: SearchFilter) -> None:
        operation_context = self._capture_search_operation_context()
        operation_mode = self._search_operation_mode_for_new_filter(search_filter)
        if operation_mode is None:
            return
        self._set_search_operation_mode(operation_mode)
        if operation_mode in {"merge", "replace"}:
            self.search_filters.clear()
            self.active_filter_index = None
        self._add_search_filter(
            search_filter,
            replace_same_kind=operation_mode == "replace",
        )
        self._execute_search_chain(
            operation_mode=operation_mode,
            operation_context=operation_context,
        )

    def _search_operation_mode_for_new_filter(self, search_filter: SearchFilter) -> str | None:
        if not self._should_prompt_search_operation(search_filter):
            return self._selected_search_operation_mode()
        return self._prompt_search_operation_choice(search_filter)

    def _should_prompt_search_operation(self, search_filter: SearchFilter) -> bool:
        if not self._has_visible_result_context():
            return False
        if not self.grid_view.images():
            return False
        active_kinds = self._active_positive_filter_kinds()
        if not active_kinds:
            return False
        return search_filter.kind not in active_kinds

    def _active_positive_filter_kinds(self) -> set[str]:
        if self.search_filters:
            return {search_filter.kind for search_filter in self.search_filters}
        if self.current_result_mode in {"semantic", "color", "keyword"}:
            return {self.current_result_mode}
        if self.current_result_mode in {"inspiration", "temp_project", "duplicate_group"}:
            return {self.current_result_mode}
        return set()

    def _capture_search_operation_context(
        self,
    ) -> tuple[list[ImageItem], list[ImageItem], set[int] | None, str | None]:
        visible_images = list(self.grid_view.images()) if self._has_visible_result_context() else []
        source_images = self._search_operation_source_images()
        base_image_ids, base_label = self._search_chain_base_context()
        return visible_images, source_images, base_image_ids, base_label

    def _search_operation_source_images(self) -> list[ImageItem]:
        if not self._has_visible_result_context():
            return []
        if self.current_result_mode == "search_chain" or self.search_filters:
            return list(self.current_chain_images)
        if self.current_result_mode == "semantic":
            return list(self.current_semantic_images)
        if self.current_result_mode == "color":
            return list(self.current_color_images)
        if self.current_result_mode == "inspiration":
            return list(self.current_inspiration_images)
        if self.current_result_mode == "creative_node":
            return list(self.current_creative_node_images)
        if self.current_result_mode == "temp_project":
            return list(self.current_temp_project_images)
        return list(self.grid_view.images())

    def _prompt_search_operation_choice(self, search_filter: SearchFilter) -> str | None:
        if self.current_language == "en":
            title = "Choose Search Logic"
            text = "There are existing results. How should Eidory apply this new filter?"
            info = f"New filter: {self._filter_label(search_filter)}"
            refine_label = "Search Within Results"
            merge_label = "Merge Results"
            replace_label = "New Search"
        else:
            title = "选择筛选逻辑"
            text = "当前已经有筛选结果。这个新条件要怎么作用？"
            info = f"新条件：{self._filter_label(search_filter)}"
            refine_label = "在结果中搜"
            merge_label = "合并结果"
            replace_label = "重新搜索"

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setWindowTitle(title)
        dialog.setText(text)
        dialog.setInformativeText(info)
        refine_button = dialog.addButton(refine_label, QMessageBox.ButtonRole.AcceptRole)
        merge_button = dialog.addButton(merge_label, QMessageBox.ButtonRole.ActionRole)
        replace_button = dialog.addButton(replace_label, QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(refine_button)
        dialog.exec()

        clicked = dialog.clickedButton()
        if clicked == refine_button:
            return "refine"
        if clicked == merge_button:
            return "merge"
        if clicked == replace_button:
            return "replace"
        if clicked == cancel_button:
            return None
        return None

    def _start_reverse_exclusion_with_filter(self, search_filter: SearchFilter) -> None:
        if search_filter.kind not in {"keyword", "semantic", "color", "tag"}:
            self.statusBar().showMessage("反向排除暂只支持颜色、关键词、语义和标签")
            return
        if not self._has_visible_result_context():
            self.statusBar().showMessage("先做一次正向筛选，再使用反向排除")
            return
        source_images = self._result_exclusion_source_images()
        if not source_images:
            self.statusBar().showMessage("当前结果为空，不能反向排除")
            return
        source_ids = {image.id for image in source_images}
        self.semantic_search_revision += 1
        revision = self.semantic_search_revision
        self.search_button.setEnabled(False)
        self._set_result_status(f"反向排除计算中：{self._filter_label(search_filter)}")

        def run() -> None:
            try:
                matches = self._compute_result_exclusion_filter_matches(
                    search_filter,
                    source_ids,
                )
                self.events.put(("result_exclusion_filter_done", (revision, search_filter, matches)))
            except Exception as exc:
                self.events.put(("error", f"反向排除失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=lambda: self.search_button.setEnabled(True),
        )

    def _add_search_filter(
        self,
        search_filter: SearchFilter,
        *,
        replace_same_kind: bool = True,
    ) -> None:
        search_filter = self._search_filter_with_default_threshold(search_filter)
        if replace_same_kind and self.search_filters and self.search_filters[-1].kind == search_filter.kind:
            self.search_filters[-1] = search_filter
            filter_index = len(self.search_filters) - 1
        else:
            self.search_filters.append(search_filter)
            filter_index = len(self.search_filters) - 1
        if search_filter.kind in SCORED_FILTER_KINDS:
            self.active_filter_index = filter_index
        else:
            self._ensure_active_filter_index()
        self._sync_legacy_search_state_from_filters()
        self._refresh_filter_chain_ui()
        self._refresh_score_threshold_controls()

    def _search_filter_with_default_threshold(self, search_filter: SearchFilter) -> SearchFilter:
        if search_filter.kind not in SCORED_FILTER_KINDS or search_filter.score_threshold is not None:
            return search_filter
        if self.score_threshold_slider.value() <= 0:
            return search_filter
        return SearchFilter(
            search_filter.kind,
            search_filter.value,
            self.score_threshold_slider.value(),
        )

    def _last_score_filter_index(self, filters: Sequence[SearchFilter] | None = None) -> int | None:
        filter_list = list(self.search_filters if filters is None else filters)
        for index in range(len(filter_list) - 1, -1, -1):
            if filter_list[index].kind in SCORED_FILTER_KINDS:
                return index
        return None

    def _ensure_active_filter_index(self) -> None:
        if not self.search_filters:
            self.active_filter_index = None
            return
        if (
            self.active_filter_index is not None
            and 0 <= self.active_filter_index < len(self.search_filters)
        ):
            return
        self.active_filter_index = self._last_score_filter_index()

    def _add_result_exclusion_filter_from_matches(
        self,
        search_filter: SearchFilter,
        matches: list[ImageItem],
    ) -> None:
        if search_filter not in self.result_exclusion_filters:
            self.result_exclusion_filters.append(search_filter)
        self.result_exclusion_filter_matches[search_filter] = list(matches)
        self._refresh_visible_results_after_result_management_change()
        self.statusBar().showMessage(
            f"已加入反向排除：{self._filter_label(search_filter)}，命中 {len(matches)} 张"
        )

    def _remove_result_exclusion_filter(self, index: int) -> None:
        if index < 0 or index >= len(self.result_exclusion_filters):
            return
        search_filter = self.result_exclusion_filters.pop(index)
        self.result_exclusion_filter_matches.pop(search_filter, None)
        self._refresh_visible_results_after_result_management_change()

    def _recompute_result_exclusion_filter_matches_for_current_source(self) -> None:
        if not self.result_exclusion_filters:
            return
        source_ids = {image.id for image in self._result_exclusion_source_images()}
        if not source_ids:
            self.result_exclusion_filter_matches = {
                search_filter: []
                for search_filter in self.result_exclusion_filters
            }
            return
        self.result_exclusion_filter_matches = {
            search_filter: self._compute_result_exclusion_filter_matches(search_filter, source_ids)
            for search_filter in self.result_exclusion_filters
        }

    def _compute_result_exclusion_filter_matches(
        self,
        search_filter: SearchFilter,
        source_ids: set[int],
    ) -> list[ImageItem]:
        if not source_ids:
            return []
        if search_filter.kind == "keyword":
            return [
                image
                for image in self.store.images_by_ids(sorted(source_ids))
                if self._image_matches_keyword(image, str(search_filter.value))
            ]
        if search_filter.kind == "semantic":
            return self.search_service.semantic_search(
                str(search_filter.value),
                allowed_image_ids=set(source_ids),
            ).images
        if search_filter.kind == "color":
            return self.search_service.color_search(
                search_filter.value,  # type: ignore[arg-type]
                allowed_image_ids=set(source_ids),
            ).images
        if search_filter.kind == "tag":
            selected_ids, match_mode = self._tag_filter_parts(search_filter)
            matching_ids = self._image_ids_for_tags(selected_ids, match_mode) & set(source_ids)
            return self.store.images_by_ids(sorted(matching_ids))
        return []

    def _result_exclusion_source_images(self) -> list[ImageItem]:
        if self.current_result_mode == "search_chain" or self.search_filters:
            return list(self.current_chain_images)
        if self.current_result_mode == "inspiration":
            return list(self.current_inspiration_images)
        if self.current_result_mode == "creative_node":
            return list(self.current_creative_node_images)
        if self.current_result_mode == "temp_project":
            return list(self.current_temp_project_images)
        if self.current_result_mode == "semantic":
            return list(self.current_semantic_images)
        if self.current_result_mode == "color":
            return list(self.current_color_images)
        if self.current_result_mode == "keyword":
            return list(self.grid_view.images())
        return []

    def _result_exclusion_filter_image_ids(self) -> set[int]:
        excluded: set[int] = set()
        for search_filter in self.result_exclusion_filters:
            matches = self.result_exclusion_filter_matches.get(search_filter, [])
            threshold: float | None = None
            if search_filter.kind == "semantic":
                threshold = self._semantic_score_threshold(matches)
            elif search_filter.kind == "color":
                threshold = self._color_score_threshold(matches)
            for image in matches:
                if threshold is not None and (
                    image.score is None or image.score < threshold
                ):
                    continue
                excluded.add(image.id)
        return excluded

    def _execute_search_chain(
        self,
        operation_mode: str = "refine",
        operation_context: tuple[list[ImageItem], list[ImageItem], set[int] | None, str | None] | None = None,
    ) -> None:
        if not self.search_filters:
            self._reload_images()
            return
        if operation_mode != "recompute":
            self._clear_manual_result_order()

        self.semantic_search_revision += 1
        revision = self.semantic_search_revision
        filters = tuple(self.search_filters)
        folder_path_prefix = self._selected_folder_path_prefix()
        collection_id = self._selected_collection_id()
        virtual_filter = self._selected_virtual_filter()
        tag_ids: list[int] = []
        tag_match_mode = "any"
        status_filter = self._selected_status_filter()
        base_image_ids, base_label, merge_base_images = self._search_operation_context(
            operation_mode,
            operation_context=operation_context,
        )

        self.current_result_mode = "search_chain"
        self.current_offset = 0
        self.load_more_button.setEnabled(False)
        self.search_button.setEnabled(False)
        self.current_chain_images = []
        self.current_chain_filtered_images = []
        self.current_chain_result = SearchChainResult(images=[])
        self.current_chain_base_image_ids = base_image_ids
        self.current_chain_base_label = base_label
        self.current_chain_operation_mode = operation_mode
        base_prefix = f"{base_label} ｜ " if base_label else ""
        self._set_result_status(f"筛选中：{base_prefix}{self._format_filter_chain(filters)}")
        self.search_diagnostics_label.setText("搜索诊断：-")
        self._refresh_feedback_buttons(self.selected_image)
        self._refresh_filter_chain_ui()

        def run() -> None:
            try:
                result = self._compute_search_chain(
                    filters=filters,
                    folder_path_prefix=folder_path_prefix,
                    collection_id=collection_id,
                    virtual_filter=virtual_filter,
                    tag_ids=tag_ids,
                    tag_match_mode=tag_match_mode,
                    status_filter=status_filter,
                    base_image_ids=base_image_ids,
                    merge_base_images=merge_base_images,
                )
                self.events.put(("search_chain_done", (revision, filters, result)))
            except Exception as exc:
                self.events.put(("error", f"筛选失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=self._restore_search_task_controls,
        )

    def _search_chain_base_context(self) -> tuple[set[int] | None, str | None]:
        if self.current_result_mode == "temp_project" and self.current_temp_project_id is not None:
            project = self.store.get_temporary_project(self.current_temp_project_id)
            if project is None:
                return None, None
            image_ids = {image.id for image in self.current_temp_project_images}
            return image_ids, f"基于{self._temporary_project_label(project)}：{project.name}"
        if self.current_result_mode == "creative_node" and self.current_creative_node_id is not None:
            node = self.store.get_creative_node(self.current_creative_node_id)
            image_ids = {image.id for image in self.current_creative_node_images}
            label = f"基于创作节点：{node.title}" if node is not None else "基于创作节点"
            return image_ids, label
        if self.current_result_mode == "search_chain" and self.current_chain_base_image_ids is not None:
            return set(self.current_chain_base_image_ids), self.current_chain_base_label
        return None, None

    def _search_operation_context(
        self,
        operation_mode: str,
        operation_context: tuple[list[ImageItem], list[ImageItem], set[int] | None, str | None] | None = None,
    ) -> tuple[set[int] | None, str | None, list[ImageItem] | None]:
        if operation_mode == "recompute":
            base_image_ids, base_label = self._search_chain_base_context()
            return base_image_ids, base_label, None

        if operation_mode == "replace":
            self._clear_result_management_state()
            return None, None, None

        if operation_context is None:
            visible_images = list(self.grid_view.images()) if self._has_visible_result_context() else []
            source_images = self._search_operation_source_images()
            snapshot_base_ids, snapshot_base_label = self._search_chain_base_context()
        else:
            visible_images, source_images, snapshot_base_ids, snapshot_base_label = operation_context
        if operation_mode == "merge":
            if visible_images:
                return None, "合并当前结果", visible_images
            return None, None, None

        if source_images:
            return {image.id for image in source_images}, "在当前结果中", None
        if visible_images:
            return {image.id for image in visible_images}, "在当前结果中", None

        return snapshot_base_ids, snapshot_base_label, None

    def _compute_search_chain(
        self,
        *,
        filters: tuple[SearchFilter, ...],
        folder_path_prefix: str | None,
        collection_id: int | None,
        tag_ids: list[int],
        tag_match_mode: str,
        status_filter: str | None,
        virtual_filter: str | None = None,
        base_image_ids: set[int] | None = None,
        merge_base_images: list[ImageItem] | None = None,
    ) -> SearchChainResult:
        images: list[ImageItem] = (
            self.store.images_by_ids(sorted(base_image_ids))
            if base_image_ids is not None
            else []
        )
        allowed_image_ids: set[int] | None = set(base_image_ids) if base_image_ids is not None else None
        semantic_searchable_count = 0
        semantic_candidate_limit = 0
        similar_searchable_count = 0
        similar_candidate_limit = 0
        color_searchable_count = 0
        color_indexed_count = 0
        color_candidate_limit = 0
        last_scored_index = self._last_score_filter_index(filters)

        for filter_index, search_filter in enumerate(filters):
            if search_filter.kind == "semantic":
                result = self.search_service.semantic_search(
                    str(search_filter.value),
                    folder_path_prefix=folder_path_prefix,
                    collection_id=collection_id,
                    virtual_filter=virtual_filter,
                    tag_ids=tag_ids,
                    tag_match_mode=tag_match_mode,
                    status_filter=status_filter,
                    allowed_image_ids=allowed_image_ids,
                )
                images = result.images
                semantic_searchable_count = result.searchable_count
                semantic_candidate_limit = result.candidate_limit
            elif search_filter.kind == "similar":
                result = self.search_service.similar_image_search(
                    int(search_filter.value),
                    folder_path_prefix=folder_path_prefix,
                    collection_id=collection_id,
                    virtual_filter=virtual_filter,
                    tag_ids=tag_ids,
                    tag_match_mode=tag_match_mode,
                    status_filter=status_filter,
                    allowed_image_ids=allowed_image_ids,
                )
                images = result.images
                similar_searchable_count = result.searchable_count
                similar_candidate_limit = result.candidate_limit
            elif search_filter.kind == "color":
                result = self.search_service.color_search(
                    search_filter.value,  # type: ignore[arg-type]
                    folder_path_prefix=folder_path_prefix,
                    collection_id=collection_id,
                    virtual_filter=virtual_filter,
                    tag_ids=tag_ids,
                    tag_match_mode=tag_match_mode,
                    status_filter=status_filter,
                    allowed_image_ids=allowed_image_ids,
                )
                images = result.images
                color_searchable_count = result.searchable_count
                color_indexed_count = result.indexed_count
                color_candidate_limit = result.candidate_limit
            elif search_filter.kind == "keyword":
                query = str(search_filter.value)
                if allowed_image_ids is None:
                    images = self.store.list_images(
                        text_query=query,
                        status_filter=status_filter,
                        tag_ids=tag_ids,
                        tag_match_mode=tag_match_mode,
                        folder_path_prefix=folder_path_prefix,
                        collection_id=collection_id,
                        virtual_filter=virtual_filter,
                        limit=5_000,
                    )
                else:
                    images = [
                        image
                        for image in images
                        if self._image_matches_keyword(image, query)
                    ]
            elif search_filter.kind == "file_type":
                if allowed_image_ids is None:
                    images = self._list_chain_base_images(
                        folder_path_prefix=folder_path_prefix,
                        collection_id=collection_id,
                        virtual_filter=virtual_filter,
                        tag_ids=tag_ids,
                        tag_match_mode=tag_match_mode,
                        status_filter=status_filter,
                    )
                images = [
                    image
                    for image in images
                    if self._image_matches_file_type(image, str(search_filter.value))
                ]
            elif search_filter.kind == "orientation":
                if allowed_image_ids is None:
                    images = self._list_chain_base_images(
                        folder_path_prefix=folder_path_prefix,
                        collection_id=collection_id,
                        virtual_filter=virtual_filter,
                        tag_ids=tag_ids,
                        tag_match_mode=tag_match_mode,
                        status_filter=status_filter,
                    )
                images = [
                    image
                    for image in images
                    if self._image_matches_orientation(image, str(search_filter.value))
                ]
            elif search_filter.kind == "size":
                if allowed_image_ids is None:
                    images = self._list_chain_base_images(
                        folder_path_prefix=folder_path_prefix,
                        collection_id=collection_id,
                        virtual_filter=virtual_filter,
                        tag_ids=tag_ids,
                        tag_match_mode=tag_match_mode,
                        status_filter=status_filter,
                    )
                images = [
                    image
                    for image in images
                    if self._image_matches_size(image, str(search_filter.value))
                ]
            elif search_filter.kind == "ai_vision":
                if allowed_image_ids is None:
                    images = self._list_chain_base_images(
                        folder_path_prefix=folder_path_prefix,
                        collection_id=collection_id,
                        virtual_filter=virtual_filter,
                        tag_ids=tag_ids,
                        tag_match_mode=tag_match_mode,
                        status_filter=status_filter,
                    )
                field, value = ai_vision_filter_parts(str(search_filter.value))
                matching_ids = self.store.image_ids_matching_ai_vision(field, value)
                images = [
                    image
                    for image in images
                    if image.id in matching_ids
                ]
            elif search_filter.kind == "collection":
                selected_ids = self._collection_filter_ids(search_filter)
                if allowed_image_ids is None:
                    images = self._list_chain_base_images(
                        folder_path_prefix=folder_path_prefix,
                        collection_id=collection_id,
                        virtual_filter=virtual_filter,
                        tag_ids=tag_ids,
                        tag_match_mode=tag_match_mode,
                        status_filter=status_filter,
                    )
                matching_ids = self._image_ids_for_collections(selected_ids)
                images = [
                    image
                    for image in images
                    if image.id in matching_ids
                ]
            elif search_filter.kind == "tag":
                selected_ids, match_mode = self._tag_filter_parts(search_filter)
                if allowed_image_ids is None:
                    images = self._list_chain_base_images(
                        folder_path_prefix=folder_path_prefix,
                        collection_id=collection_id,
                        virtual_filter=virtual_filter,
                        tag_ids=tag_ids,
                        tag_match_mode=tag_match_mode,
                        status_filter=status_filter,
                    )
                matching_ids = self._image_ids_for_tags(selected_ids, match_mode)
                images = [
                    image
                    for image in images
                    if image.id in matching_ids
                ]
            if (
                search_filter.kind in SCORED_FILTER_KINDS
                and filter_index != last_scored_index
            ):
                images = self._apply_score_threshold_for_filter(search_filter, images)
            allowed_image_ids = {image.id for image in images}
            if not allowed_image_ids:
                break

        if merge_base_images:
            images = self._merge_search_result_images(merge_base_images, images)

        return SearchChainResult(
            images=images,
            semantic_searchable_count=semantic_searchable_count,
            semantic_candidate_limit=semantic_candidate_limit,
            similar_searchable_count=similar_searchable_count,
            similar_candidate_limit=similar_candidate_limit,
            color_searchable_count=color_searchable_count,
            color_indexed_count=color_indexed_count,
            color_candidate_limit=color_candidate_limit,
        )

    @staticmethod
    def _merge_search_result_images(
        base_images: list[ImageItem],
        new_images: list[ImageItem],
    ) -> list[ImageItem]:
        merged: list[ImageItem] = []
        seen: set[int] = set()
        for image in [*base_images, *new_images]:
            if image.id in seen:
                continue
            seen.add(image.id)
            merged.append(image)
        return merged

    def _list_chain_base_images(
        self,
        *,
        folder_path_prefix: str | None,
        collection_id: int | None,
        virtual_filter: str | None,
        tag_ids: list[int],
        tag_match_mode: str,
        status_filter: str | None,
    ) -> list[ImageItem]:
        return self.store.list_images(
            status_filter=status_filter,
            tag_ids=tag_ids,
            tag_match_mode=tag_match_mode,
            folder_path_prefix=folder_path_prefix,
            collection_id=collection_id,
            virtual_filter=virtual_filter,
            limit=50_000,
        )

    def _apply_score_threshold_for_filter(
        self,
        search_filter: SearchFilter,
        images: list[ImageItem],
    ) -> list[ImageItem]:
        threshold = self._score_threshold_for_filter(search_filter, images)
        if threshold is None:
            return list(images)
        return [
            image
            for image in images
            if image.score is not None and image.score >= threshold
        ]

    def _image_matches_keyword(self, image: ImageItem, query: str) -> bool:
        needle = query.strip().casefold()
        if not needle:
            return True
        values = [
            image.file_name,
            image.file_path,
            image.note or "",
        ]
        if any(needle in value.casefold() for value in values):
            return True
        return any(
            needle in tag.casefold()
            for tag in self.store.get_image_tags(image.id)
        )

    @staticmethod
    def _image_matches_file_type(image: ImageItem, value: str) -> bool:
        if value == "media:image":
            return image.file_ext in SUPPORTED_IMAGE_EXTENSIONS
        if value == "media:video":
            return image.file_ext in SUPPORTED_VIDEO_EXTENSIONS
        if value.startswith("ext:"):
            return image.file_ext == value.removeprefix("ext:")
        return False

    @staticmethod
    def _image_matches_orientation(image: ImageItem, value: str) -> bool:
        if not image.width or not image.height:
            return False
        ratio = image.width / image.height
        if value == "landscape":
            return ratio > 1.08
        if value == "portrait":
            return ratio < 0.92
        if value == "square":
            return 0.92 <= ratio <= 1.08
        return False

    @staticmethod
    def _image_matches_size(image: ImageItem, value: str) -> bool:
        if not image.width or not image.height:
            return False
        pixels = image.width * image.height
        if value == "large":
            return pixels >= 2_000_000
        if value == "small":
            return pixels <= 500_000
        return False

    def _handle_search_chain_done(
        self,
        *,
        filters: tuple[SearchFilter, ...],
        result: SearchChainResult,
    ) -> None:
        self.current_chain_result = result
        self.current_chain_images = list(result.images)
        self.current_semantic_searchable_count = result.semantic_searchable_count
        self.current_semantic_candidate_limit = result.semantic_candidate_limit
        self.current_similar_searchable_count = result.similar_searchable_count
        self.current_similar_candidate_limit = result.similar_candidate_limit
        self.current_color_searchable_count = result.color_searchable_count
        self.current_color_indexed_count = result.color_indexed_count
        self.current_color_candidate_limit = result.color_candidate_limit
        self._sync_legacy_search_state_from_filters()
        self._recompute_result_exclusion_filter_matches_for_current_source()
        self._apply_search_chain_filters()
        images = self.current_chain_filtered_images
        self.grid_view.set_images(images)
        self._set_search_chain_result_status(filters, images)
        self._update_search_chain_diagnostics(filters, images)
        self._refresh_feedback_buttons(self.selected_image)

    def _handle_inspiration_done(
        self,
        *,
        project_id: int,
        selected_terms: list[InspirationTerm],
        result=None,
        raw_term_results: list[tuple[InspirationTerm, list[ImageItem]]] | None = None,
        plan_filters: list[SearchPlanFilter] | None = None,
    ) -> None:
        self.current_result_mode = "inspiration"
        self.current_inspiration_project_id = project_id
        self.current_inspiration_terms = list(selected_terms)
        self.current_inspiration_plan_filters = list(plan_filters or [])
        if raw_term_results is not None:
            self.current_inspiration_raw_term_results = [
                (term, list(images))
                for term, images in raw_term_results
            ]
            self._rebuild_current_inspiration_results_from_raw()
        elif result is not None:
            self.current_inspiration_raw_term_results = []
            self.current_inspiration_images = list(result.images)
            self.current_inspiration_matches = dict(result.matches_by_image_id)
        self._recompute_result_exclusion_filter_matches_for_current_source()
        self._apply_inspiration_result_filters()
        images = self.current_inspiration_filtered_images
        badges = self._inspiration_badges_by_image_id()
        self.grid_view.set_images(images, badges_by_image_id=badges)
        self._set_inspiration_result_status(images)
        self._update_inspiration_diagnostics(images)
        self._refresh_feedback_buttons(self.selected_image)

    def _inspiration_badges_by_image_id(self) -> dict[int, list[str]]:
        return {
            image_id: [self._format_inspiration_badge(matches)]
            for image_id, matches in self.current_inspiration_matches.items()
            if matches
        }

    def _update_inspiration_diagnostics(self, images: list[ImageItem]) -> None:
        if not images:
            self.search_diagnostics_label.setText("搜索诊断：-")
            return
        visible_term_titles = {
            match.term_title
            for image in images
            for match in self.current_inspiration_matches.get(image.id, [])
        }
        covered_count = sum(
            1
            for term in self.current_inspiration_terms
            if term.title in visible_term_titles
        )
        multi_hit_count = sum(
            1
            for image in images
            if len(self.current_inspiration_matches.get(image.id, [])) > 1
        )
        scores = [image.score for image in images if image.score is not None]
        threshold = self._semantic_score_threshold(self.current_inspiration_images)
        threshold_text = "不限" if threshold is None else f"{threshold:.2f}"
        protected_count = (
            0
            if threshold is None
            else sum(
                1
                for image in images
                if image.score is not None and image.score < threshold
            )
        )
        parts = [
            f"显示 {len(images)}",
            f"探针覆盖 {covered_count}/{len(self.current_inspiration_terms)}",
            f"多重命中 {multi_hit_count}",
            f"阈值 {threshold_text}（强度 {self.score_threshold_slider.value()}%）",
        ]
        if self.current_inspiration_plan_filters:
            parts.append(
                "AI 场景标签 "
                f"{len(self.current_inspiration_plan_filters)}，"
                f"作用 {self._inspiration_plan_filtered_term_count()}/{len(self.current_inspiration_terms)}"
            )
        if protected_count:
            parts.append(f"覆盖保留 {protected_count}")
        if scores:
            parts.extend([
                f"最高 {max(scores):.3f}",
                f"最低 {min(scores):.3f}",
                f"平均 {sum(scores) / len(scores):.3f}",
            ])
        self.search_diagnostics_label.setText("搜索诊断：" + "，".join(parts))

    def _set_inspiration_result_status(self, images: list[ImageItem]) -> None:
        source_count = len(self.current_inspiration_images)
        term_titles = "、".join(term.title for term in self.current_inspiration_terms)
        filter_suffix = (
            " ｜ 筛选："
            f"{self._format_plan_filter_summary(self.current_inspiration_plan_filters)}"
            f"（作用 {self._inspiration_plan_filtered_term_count()}/{len(self.current_inspiration_terms)} 个探针）"
            if self.current_inspiration_plan_filters else ""
        )
        suffix = self._result_management_status_suffix()
        if len(images) == source_count:
            self._set_result_status(f"灵感项目结果：{len(images)} 张 ｜ {term_titles}{filter_suffix}{suffix}")
        else:
            self._set_result_status(
                f"灵感项目结果：{len(images)} / 原始 {source_count} ｜ {term_titles}{filter_suffix}{suffix}"
            )

    def _format_plan_filter_summary(self, filters: list[SearchPlanFilter]) -> str:
        labels = [
            ai_vision_label(plan_filter.field, plan_filter.value, language=self.current_language)
            for plan_filter in filters
        ]
        return "、".join(label.split(": ", 1)[-1] for label in labels[:6])

    def _inspiration_plan_filtered_term_count(self) -> int:
        return sum(
            1
            for term in self.current_inspiration_terms
            if self._inspiration_term_uses_plan_filters(term)
        )

    def _apply_inspiration_result_filters(self) -> None:
        images = self.current_inspiration_images
        threshold = self._semantic_score_threshold(images)
        if threshold is None:
            filtered = list(images)
        else:
            passing = [
                image
                for image in images
                if image.score is not None and image.score >= threshold
            ]
            filtered = self._unique_images([
                *self._inspiration_coverage_images(images),
                *passing,
            ])
        filtered = self._apply_result_management_filters(self._apply_sidebar_filters(filtered))
        self.current_inspiration_filtered_images = self._sort_images(filtered)

    def _inspiration_coverage_images(self, images: list[ImageItem]) -> list[ImageItem]:
        coverage: list[ImageItem] = []
        used_ids: set[int] = set()
        for term in self.current_inspiration_terms:
            best_unused = self._best_inspiration_image_for_term(
                term.title,
                images,
                excluded_ids=used_ids,
            )
            best = best_unused or self._best_inspiration_image_for_term(
                term.title,
                images,
                excluded_ids=set(),
            )
            if best is None:
                continue
            coverage.append(best)
            used_ids.add(best.id)
        return coverage

    def _best_inspiration_image_for_term(
        self,
        term_title: str,
        images: list[ImageItem],
        *,
        excluded_ids: set[int],
    ) -> ImageItem | None:
        best_image: ImageItem | None = None
        best_score = float("-inf")
        for image in images:
            if image.id in excluded_ids:
                continue
            matches = self.current_inspiration_matches.get(image.id, [])
            term_scores = [
                match.score
                for match in matches
                if match.term_title == term_title and match.score is not None
            ]
            if not term_scores:
                continue
            score = max(float(term_score) for term_score in term_scores)
            if best_image is None or score > best_score:
                best_image = image
                best_score = score
        return best_image

    @staticmethod
    def _unique_images(images: list[ImageItem]) -> list[ImageItem]:
        unique: list[ImageItem] = []
        seen: set[int] = set()
        for image in images:
            if image.id in seen:
                continue
            seen.add(image.id)
            unique.append(image)
        return unique

    def _apply_search_chain_filters(self) -> None:
        images = self.current_chain_images
        threshold = self._active_score_threshold(images)
        if threshold is not None:
            images = [
                image
                for image in images
                if image.score is not None and image.score >= threshold
            ]
        images = self._apply_result_management_filters(self._apply_sidebar_filters(images))
        self.current_chain_filtered_images = self._sort_images(images)

    def _active_score_threshold(self, images: list[ImageItem]) -> float | None:
        score_index = self._last_score_filter_index()
        if score_index is None:
            return None
        return self._score_threshold_for_filter(self.search_filters[score_index], images)

    def _score_threshold_for_filter(
        self,
        search_filter: SearchFilter,
        images: list[ImageItem],
    ) -> float | None:
        value = self._score_threshold_value_for_filter(search_filter)
        if search_filter.kind in {"semantic", "similar"}:
            return self._semantic_score_threshold(images, value=value)
        if search_filter.kind == "color":
            return self._color_score_threshold(images, value=value)
        return None

    def _score_threshold_value_for_filter(self, search_filter: SearchFilter) -> int:
        if search_filter.score_threshold is None:
            return self.score_threshold_slider.value()
        return max(0, min(100, int(search_filter.score_threshold)))

    def _set_search_chain_result_status(
        self,
        filters: tuple[SearchFilter, ...],
        images: list[ImageItem],
    ) -> None:
        source_count = len(self.current_chain_images)
        chain = self._format_filter_chain(filters)
        base_prefix = f"{self.current_chain_base_label} ｜ " if self.current_chain_base_label else ""
        suffix = self._result_management_status_suffix()
        if len(images) == source_count:
            self._set_result_status(f"筛选结果：{base_prefix}{len(images)} ｜ {chain}{suffix}")
        else:
            self._set_result_status(f"筛选结果：{base_prefix}{len(images)} / 原始 {source_count} ｜ {chain}{suffix}")

    def _update_search_chain_diagnostics(
        self,
        filters: tuple[SearchFilter, ...],
        images: list[ImageItem],
    ) -> None:
        if not filters or not images:
            self.search_diagnostics_label.setText("搜索诊断：-")
            return

        parts = [f"显示 {len(images)}", f"条件 {len(filters)}"]
        if self.current_chain_base_image_ids is not None:
            parts.append(f"基础范围 {len(self.current_chain_base_image_ids)}")
        last_index = self._last_score_filter_index(filters)
        last_filter = filters[last_index] if last_index is not None else None
        last_kind = last_filter.kind if last_filter is not None else None
        scores = [image.score for image in images if image.score is not None]
        if last_kind == "semantic":
            assert last_filter is not None
            threshold_value = self._score_threshold_value_for_filter(last_filter)
            threshold = self._semantic_score_threshold(self.current_chain_images, value=threshold_value)
            threshold_text = "不限" if threshold is None else f"{threshold:.2f}"
            parts.extend([
                f"语义可搜索 {self.current_semantic_searchable_count}",
                f"候选上限 {self.current_semantic_candidate_limit}",
                f"阈值 {threshold_text}（强度 {threshold_value}%）",
            ])
        elif last_kind == "similar":
            assert last_filter is not None
            threshold_value = self._score_threshold_value_for_filter(last_filter)
            threshold = self._semantic_score_threshold(self.current_chain_images, value=threshold_value)
            threshold_text = "不限" if threshold is None else f"{threshold:.2f}"
            parts.extend([
                f"相似可搜索 {self.current_similar_searchable_count}",
                f"候选上限 {self.current_similar_candidate_limit}",
                f"阈值 {threshold_text}（强度 {threshold_value}%）",
            ])
        elif last_kind == "color":
            assert last_filter is not None
            threshold_value = self._score_threshold_value_for_filter(last_filter)
            threshold = self._color_score_threshold(self.current_chain_images, value=threshold_value)
            threshold_text = (
                "不限"
                if threshold is None
                else f"{threshold:.3f}（强度 {threshold_value}%）"
            )
            parts.extend([
                f"颜色候选 {self.current_color_searchable_count}",
                f"颜色索引 {self.current_color_indexed_count}",
                f"候选上限 {self.current_color_candidate_limit}",
                f"阈值 {threshold_text}",
            ])
        if scores:
            parts.extend([
                f"最高 {max(scores):.3f}",
                f"最低 {min(scores):.3f}",
                f"平均 {sum(scores) / len(scores):.3f}",
            ])
        self.search_diagnostics_label.setText("搜索诊断：" + "，".join(parts))

    def _sync_legacy_search_state_from_filters(self) -> None:
        self.current_keyword_query = None
        self.current_semantic_query = None
        for search_filter in self.search_filters:
            if search_filter.kind == "keyword":
                self.current_keyword_query = str(search_filter.value)
            elif search_filter.kind == "semantic":
                self.current_semantic_query = str(search_filter.value)
            elif search_filter.kind == "color":
                self.current_color_rgb = search_filter.value  # type: ignore[assignment]
        if hasattr(self, "color_mode_button"):
            self._update_color_swatch()

    def _format_filter_chain(self, filters: tuple[SearchFilter, ...] | list[SearchFilter]) -> str:
        return format_filter_chain(filters, image_label_for_id=self._image_label_for_id)

    def _filter_label(self, search_filter: SearchFilter) -> str:
        if search_filter.kind == "collection":
            return f"文件夹：{self._collection_filter_label(search_filter)}"
        if search_filter.kind == "tag":
            return f"标签：{self._tag_filter_label(search_filter)}"
        return filter_label(search_filter, image_label_for_id=self._image_label_for_id)

    def _collection_filter_value(self, collection_ids: list[int] | set[int]) -> str:
        return ",".join(str(collection_id) for collection_id in sorted(set(collection_ids)))

    def _collection_filter_ids(self, search_filter: SearchFilter) -> list[int]:
        if search_filter.kind != "collection":
            return []
        ids: list[int] = []
        for part in str(search_filter.value).split(","):
            try:
                collection_id = int(part.strip())
            except ValueError:
                continue
            if collection_id > 0:
                ids.append(collection_id)
        return ids

    def _collection_filter_label(self, search_filter: SearchFilter) -> str:
        labels = [
            self._collection_path_text(collection_id)
            for collection_id in self._collection_filter_ids(search_filter)
            if self._collection_by_id(collection_id) is not None
        ]
        if not labels:
            return "无"
        if len(labels) <= 2:
            return " + ".join(labels)
        return f"{' + '.join(labels[:2])} 等 {len(labels)} 个"

    def _tag_filter_value(self, tag_ids: list[int] | set[int], match_mode: str) -> str:
        mode = "any" if match_mode == "any" else "all"
        ids = ",".join(str(tag_id) for tag_id in sorted(set(tag_ids)))
        return f"{mode}:{ids}"

    def _tag_filter_parts(self, search_filter: SearchFilter) -> tuple[list[int], str]:
        if search_filter.kind != "tag":
            return [], "all"
        raw_value = str(search_filter.value)
        raw_mode, separator, raw_ids = raw_value.partition(":")
        if separator:
            mode = "any" if raw_mode == "any" else "all"
            id_text = raw_ids
        else:
            mode = "all"
            id_text = raw_value
        ids: list[int] = []
        for part in id_text.split(","):
            try:
                tag_id = int(part.strip())
            except ValueError:
                continue
            if tag_id > 0:
                ids.append(tag_id)
        return ids, mode

    def _tag_filter_label(self, search_filter: SearchFilter) -> str:
        tag_ids, mode = self._tag_filter_parts(search_filter)
        tags_by_id = {tag.id: tag.tag_name for tag in self.store.list_tags()}
        labels = [tags_by_id[tag_id] for tag_id in tag_ids if tag_id in tags_by_id]
        if not labels:
            return "无"
        prefix = "任一" if mode == "any" else "全部"
        if len(labels) <= 3:
            return f"{prefix}：{' + '.join(labels)}"
        return f"{prefix}：{' + '.join(labels[:3])} 等 {len(labels)} 个"

    def _image_ids_for_tags(self, tag_ids: list[int] | set[int], match_mode: str) -> set[int]:
        if not tag_ids:
            return set()
        images = self.store.list_images(
            tag_ids=sorted(set(tag_ids)),
            tag_match_mode="any" if match_mode == "any" else "all",
            limit=50_000,
        )
        return {image.id for image in images}

    def _image_ids_for_collections(self, collection_ids: list[int] | set[int]) -> set[int]:
        image_ids: set[int] = set()
        for collection_id in collection_ids:
            image_ids.update(self.store.image_ids_for_collection(collection_id))
        return image_ids

    def _image_label_for_id(self, image_id: int) -> str:
        image = self.store.get_image(int(image_id))
        return image.file_name if image is not None else f"#{image_id}"

    @staticmethod
    def _file_type_filter_label(value: str) -> str:
        return file_type_filter_label(value)

    @staticmethod
    def _orientation_filter_label(value: str) -> str:
        return orientation_filter_label(value)

    @staticmethod
    def _size_filter_label(value: str) -> str:
        return size_filter_label(value)

    def _refresh_filter_chain_ui(self) -> None:
        while self.filter_chain_layout.count():
            item = self.filter_chain_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.filter_chain_layout.addWidget(QLabel("筛选："))
        has_filter = False
        for index, search_filter in enumerate(self.search_filters):
            has_filter = True
            self.filter_chain_layout.addWidget(self._search_filter_chip(index, search_filter))

        for label, callback in self._context_filter_actions():
            has_filter = True
            button = QPushButton(f"× {label}")
            button.setToolTip("移除此侧栏筛选")
            button.clicked.connect(lambda _checked=False, action=callback: action())
            self.filter_chain_layout.addWidget(button)

        for label, callback in self._result_management_filter_actions():
            has_filter = True
            button = QPushButton(f"× {label}")
            button.setToolTip("移除此结果管理条件")
            button.clicked.connect(lambda _checked=False, action=callback: action())
            self.filter_chain_layout.addWidget(button)

        for index, search_filter in enumerate(self.result_exclusion_filters):
            has_filter = True
            button = QPushButton(f"× 反向排除：{self._filter_label(search_filter)}")
            button.setToolTip("移除此反向排除条件")
            button.clicked.connect(
                lambda _checked=False, filter_index=index: self._remove_result_exclusion_filter(filter_index)
            )
            self.filter_chain_layout.addWidget(button)

        if not has_filter:
            self.filter_chain_layout.addWidget(QLabel("无"))
        self.filter_chain_layout.addStretch(1)
        self._refresh_score_threshold_controls()
        self._refresh_result_management_buttons()

    def _search_filter_chip(self, index: int, search_filter: SearchFilter) -> QWidget:
        chip = QWidget()
        layout = QHBoxLayout(chip)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        body = QPushButton(self._filter_label(search_filter))
        body.setCheckable(True)
        body.setChecked(index == self.active_filter_index)
        body.setToolTip(
            "点击后用相似度滑块调整此条件"
            if search_filter.kind in SCORED_FILTER_KINDS
            else "点击后选中此条件；此条件没有相似度滑块"
        )
        body.clicked.connect(
            lambda _checked=False, filter_index=index: self._select_search_filter_chip(filter_index)
        )
        remove = QPushButton("×")
        remove.setFixedWidth(34)
        remove.setMinimumHeight(26)
        remove_font = remove.font()
        remove_font.setPointSize(remove_font.pointSize() + 2)
        remove_font.setBold(True)
        remove.setFont(remove_font)
        remove.setStyleSheet(
            "QPushButton {"
            "border-left: 1px solid #6f7782;"
            "padding: 0 8px;"
            "}"
            "QPushButton:hover {"
            "background: #5a6370;"
            "}"
        )
        remove.setToolTip("移除此筛选条件")
        remove.clicked.connect(
            lambda _checked=False, filter_index=index: self._remove_search_filter(filter_index)
        )
        layout.addWidget(body)
        layout.addWidget(remove)
        return chip

    def _select_search_filter_chip(self, index: int) -> None:
        if index < 0 or index >= len(self.search_filters):
            return
        self.active_filter_index = index
        self._refresh_filter_chain_ui()
        search_filter = self.search_filters[index]
        if search_filter.kind not in SCORED_FILTER_KINDS:
            self.statusBar().showMessage(f"{self._filter_label(search_filter)} 没有相似度强度")

    def _refresh_result_management_buttons(self) -> None:
        if not hasattr(self, "save_result_set_button"):
            return
        has_result_context = self._has_result_context()
        has_visible_results = bool(self.grid_view.images()) if hasattr(self, "grid_view") else False
        self.save_result_set_button.setEnabled(has_result_context and has_visible_results)

    def _remove_search_filter(self, index: int) -> None:
        if index < 0 or index >= len(self.search_filters):
            return
        del self.search_filters[index]
        if self.active_filter_index is not None:
            if self.active_filter_index == index:
                self.active_filter_index = None
            elif self.active_filter_index > index:
                self.active_filter_index -= 1
        self._ensure_active_filter_index()
        self._sync_legacy_search_state_from_filters()
        self._refresh_filter_chain_ui()
        if self.search_filters:
            self._execute_search_chain(operation_mode="recompute")
        else:
            self._reload_images()

    def _context_filter_actions(self) -> list[tuple[str, object]]:
        actions: list[tuple[str, object]] = []
        virtual_filter = self._selected_virtual_filter()
        collection_id = self._selected_collection_id()
        if virtual_filter is not None:
            actions.append((f"聚类：{self._virtual_filter_label(virtual_filter)}", self._clear_collection_filter))
        elif collection_id is not None:
            collection_name = self._collection_name(collection_id) or "当前文件夹"
            actions.append((f"文件夹：{collection_name}", self._clear_collection_filter))

        status_value = self.status_filter_combo.currentData()
        if status_value is not None and status_value != "all":
            actions.append((f"状态：{self.status_filter_combo.currentText()}", self._clear_status_filter))
        return actions

    def _result_management_filter_actions(self) -> list[tuple[str, object]]:
        actions: list[tuple[str, object]] = []
        if self.result_excluded_image_ids:
            actions.append((f"结果排除：{len(self.result_excluded_image_ids)} 张", self._clear_result_exclusions))
        for collection_id in sorted(self.result_excluded_collection_ids):
            collection = self._collection_by_id(collection_id)
            label = self._collection_path_text(collection_id) if collection is not None else f"#{collection_id}"
            actions.append((
                f"排除文件夹：{label}",
                lambda collection_id=collection_id: self._remove_result_collection_exclusion(collection_id),
            ))
        return actions

    def _clear_result_exclusions(self) -> None:
        self.result_excluded_image_ids.clear()
        self._refresh_visible_results_after_result_management_change()

    def _clear_collection_filter(self) -> None:
        self.current_virtual_filter = None
        self._refresh_virtual_collection_filters()
        item = self.collection_tree.topLevelItem(0)
        if item is not None:
            self.collection_tree.setCurrentItem(item)

    def _clear_tag_filter(self) -> None:
        self.tag_list.clearSelection()
        item = self.tag_list.item(0)
        if item is not None:
            self.tag_list.setCurrentItem(item)
            item.setSelected(True)

    def _clear_status_filter(self) -> None:
        self._set_combo_to_data(self.status_filter_combo, "all")

    def _choose_collection_filter(self) -> None:
        if not self.store.list_collections():
            self.statusBar().showMessage("还没有可筛选的文件夹")
            return
        reverse_mode = self.reverse_exclusion_button.isChecked()
        if reverse_mode and not self._has_result_context():
            self.statusBar().showMessage("先做一次正向筛选，再使用反向排除")
            return
        selected_ids = self._select_collections_for_search_dialog(
            reverse_mode=reverse_mode,
            context_image=self._selected_grid_image(),
        )
        if selected_ids is None:
            return
        if not selected_ids:
            self.statusBar().showMessage("没有选择文件夹")
            return
        if reverse_mode:
            self._exclude_collections_from_results(selected_ids)
            return
        search_filter = SearchFilter("collection", self._collection_filter_value(selected_ids))
        self._start_search_with_filter(search_filter)

    def _choose_tag_filter(self) -> None:
        if not self.store.list_tags():
            self.right_tab_widget.setCurrentIndex(2)
            QMessageBox.information(
                self,
                "标签筛选",
                "当前还没有用户标签。先选中图片，在右侧“标签”页添加标签；然后这里才能按标签筛选或反向排除。",
            )
            self.statusBar().showMessage("还没有可筛选的标签")
            return
        reverse_mode = self.reverse_exclusion_button.isChecked()
        if reverse_mode and not self._has_result_context():
            self.statusBar().showMessage("先做一次正向筛选，再使用反向排除")
            return
        selection = self._select_tags_for_search_dialog(reverse_mode=reverse_mode)
        if selection is None:
            return
        selected_ids, match_mode = selection
        if not selected_ids:
            self.statusBar().showMessage("没有选择标签")
            return
        search_filter = SearchFilter("tag", self._tag_filter_value(selected_ids, match_mode))
        if reverse_mode:
            self._start_reverse_exclusion_with_filter(search_filter)
        else:
            self._start_search_with_filter(search_filter)

    def _select_tags_for_search_dialog(
        self,
        *,
        reverse_mode: bool,
    ) -> tuple[list[int], str] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("反向排除标签" if reverse_mode else "标签筛选")
        layout = QVBoxLayout(dialog)
        instruction = (
            "选择要从当前结果中排除的标签。"
            if reverse_mode
            else "选择要纳入搜索条件的标签。"
        )
        layout.addWidget(QLabel(instruction))

        match_combo = QComboBox()
        match_combo.addItem("匹配全部", "all")
        match_combo.addItem("匹配任一", "any")
        self._set_combo_to_data(match_combo, self._selected_tag_match_mode())
        layout.addWidget(match_combo)

        tag_list = QListWidget()
        tag_list.setMinimumHeight(260)
        current_ids = set(self._selected_tag_ids())
        for tag, count in self.store.list_tags_with_counts():
            item = QListWidgetItem(f"{tag.tag_name}    {count}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if tag.id in current_ids else Qt.CheckState.Unchecked
            )
            item.setData(Qt.ItemDataRole.UserRole, tag.id)
            item.setToolTip(tag.tag_name)
            tag_list.addItem(item)
        layout.addWidget(tag_list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        selected_ids: list[int] = []
        for index in range(tag_list.count()):
            item = tag_list.item(index)
            tag_id = item.data(Qt.ItemDataRole.UserRole)
            if tag_id is not None and item.checkState() == Qt.CheckState.Checked:
                selected_ids.append(int(tag_id))
        match_mode = "any" if match_combo.currentData() == "any" else "all"
        self._set_combo_to_data(self.tag_match_combo, match_mode)
        return selected_ids, match_mode

    def _select_collections_for_search_dialog(
        self,
        *,
        reverse_mode: bool,
        context_image: ImageItem | None,
    ) -> list[int] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("反向排除文件夹" if reverse_mode else "文件夹筛选")
        layout = QVBoxLayout(dialog)
        instruction = (
            "选择要从当前结果中排除的文件夹。父级文件夹会包含其子文件夹。"
            if reverse_mode
            else "选择要纳入搜索范围的文件夹。父级文件夹会包含其子文件夹。"
        )
        layout.addWidget(QLabel(instruction))

        chain_list: QListWidget | None = None
        if reverse_mode and context_image is not None:
            chain_entries = self._collection_chain_entries_for_image(context_image.id)
            if chain_entries:
                layout.addWidget(QLabel("当前图片所在文件夹链"))
                chain_list = QListWidget()
                chain_list.setMaximumHeight(96)
                for label, collection_id in chain_entries:
                    item = QListWidgetItem(label)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    item.setData(Qt.ItemDataRole.UserRole, collection_id)
                    chain_list.addItem(item)
                layout.addWidget(chain_list)

        tree = QTreeWidget()
        tree.setHeaderLabels(["文件夹", "张"])
        tree.setMinimumHeight(260)
        tree.setAlternatingRowColors(True)
        rows = self.store.list_collections_with_counts()
        collection_by_id = {collection.id: collection for collection, _count in rows}
        counts = {collection.id: count for collection, count in rows}
        children_by_parent: dict[int | None, list[int]] = {}
        for collection, _count in rows:
            children_by_parent.setdefault(collection.parent_id, []).append(collection.id)
        for children in children_by_parent.values():
            children.sort(key=lambda collection_id: collection_by_id[collection_id].name.casefold())

        def add_items(parent_item: QTreeWidgetItem | None, parent_id: int | None) -> None:
            for collection_id in children_by_parent.get(parent_id, []):
                collection = collection_by_id[collection_id]
                item = QTreeWidgetItem([collection.name, str(counts.get(collection_id, 0))])
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Unchecked)
                item.setData(0, Qt.ItemDataRole.UserRole, collection_id)
                if parent_item is None:
                    tree.addTopLevelItem(item)
                else:
                    parent_item.addChild(item)
                add_items(item, collection_id)

        add_items(None, None)
        tree.collapseAll()
        tree.resizeColumnToContents(0)
        layout.addWidget(tree)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        selected_ids: set[int] = set()

        def collect_checked(item: QTreeWidgetItem) -> None:
            collection_id = item.data(0, Qt.ItemDataRole.UserRole)
            if collection_id is not None and item.checkState(0) == Qt.CheckState.Checked:
                selected_ids.add(int(collection_id))
            for index in range(item.childCount()):
                collect_checked(item.child(index))

        for index in range(tree.topLevelItemCount()):
            collect_checked(tree.topLevelItem(index))

        if chain_list is not None:
            for index in range(chain_list.count()):
                item = chain_list.item(index)
                collection_id = item.data(Qt.ItemDataRole.UserRole)
                if collection_id is not None and item.checkState() == Qt.CheckState.Checked:
                    selected_ids.add(int(collection_id))

        return sorted(selected_ids)

    def _collection_chain_entries_for_image(self, image_id: int) -> list[tuple[str, int]]:
        entries: list[tuple[str, int]] = []
        seen: set[int] = set()
        for chain in self.store.collection_chains_for_image(image_id):
            for level, collection in enumerate(reversed(chain)):
                if collection.id in seen:
                    continue
                seen.add(collection.id)
                path = " / ".join(item.name for item in chain[: len(chain) - level])
                prefix = "当前" if level == 0 else f"上级 {level}"
                entries.append((f"{prefix}：{path}", collection.id))
        return entries

    def _reload_images(self) -> None:
        self.current_offset = 0
        self._clear_manual_result_order()
        self._clear_result_management_state()
        self._clear_temporary_project_selection()
        self.search_filters.clear()
        self.active_filter_index = None
        self.current_keyword_query = None
        self.current_semantic_query = None
        self.current_result_mode = "library"
        self.current_semantic_images = []
        self.current_semantic_filtered_images = []
        self.current_similar_searchable_count = 0
        self.current_similar_candidate_limit = 0
        self.current_color_images = []
        self.current_color_filtered_images = []
        self.current_search_scope_count = None
        self.current_inspiration_project_id = None
        self.current_inspiration_terms = []
        self.current_inspiration_plan_filters = []
        self.current_inspiration_raw_term_results = []
        self.current_inspiration_images = []
        self.current_inspiration_filtered_images = []
        self.current_inspiration_matches = {}
        self.current_temp_project_id = None
        self.current_temp_project_images = []
        self.current_temp_project_badges = {}
        self.current_creative_node_images = []
        self.current_creative_node_filtered_images = []
        self.current_creative_node_searchable_count = 0
        self.current_creative_node_candidate_limit = 0
        self.current_creative_node_badges = {}
        self.current_chain_images = []
        self.current_chain_filtered_images = []
        self.current_chain_result = SearchChainResult(images=[])
        self.current_chain_base_image_ids = None
        self.current_chain_base_label = None
        self.current_chain_operation_mode = "replace"
        self._show_gallery_view()
        self.load_more_button.setEnabled(True)
        images = self.store.list_images(
            status_filter=self._selected_status_filter(),
            tag_ids=[],
            tag_match_mode="any",
            folder_path_prefix=self._selected_folder_path_prefix(),
            collection_id=self._selected_collection_id(),
            virtual_filter=self._selected_virtual_filter(),
            limit=self.page_size,
            offset=0,
            sort_key=self._database_sort_key(),
            sort_desc=self.current_sort_desc,
        )
        self.grid_view.set_images(images)
        self._set_result_status(f"全部图库：已加载 {len(images)} 张")
        self._update_search_diagnostics([])
        self._refresh_feedback_buttons(self.selected_image)
        self._refresh_filter_chain_ui()

    def _clear_temporary_project_selection(self) -> None:
        if not hasattr(self, "temp_project_list"):
            return
        self.temp_project_list.blockSignals(True)
        self.temp_project_list.clearSelection()
        self.temp_project_list.setCurrentItem(None)
        self.temp_project_list.blockSignals(False)

    def _load_more(self) -> None:
        if self.search_filters or self.current_result_mode in {
            "semantic",
            "color",
            "search_chain",
            "inspiration",
            "temp_project",
            "creative_node",
        }:
            return
        self.current_offset += self.page_size
        images = self.store.list_images(
            text_query=self.current_keyword_query,
            status_filter=self._selected_status_filter(),
            tag_ids=[],
            tag_match_mode="any",
            folder_path_prefix=self._selected_folder_path_prefix(),
            collection_id=self._selected_collection_id(),
            virtual_filter=self._selected_virtual_filter(),
            limit=self.page_size,
            offset=self.current_offset,
            sort_key=self._database_sort_key(),
            sort_desc=self.current_sort_desc,
        )
        self.grid_view.append_images(images)
        self._set_result_status(f"已加载 {self.grid_view.rowCount()} 张，新增加载 {len(images)} 张")

    def _clear_search(self) -> None:
        self.semantic_search_revision += 1
        self._clear_result_management_state()
        self.search_filters.clear()
        self.active_filter_index = None
        self.current_semantic_filtered_images = []
        self.current_semantic_searchable_count = 0
        self.current_semantic_candidate_limit = 0
        self.current_semantic_query = None
        self.current_similar_searchable_count = 0
        self.current_similar_candidate_limit = 0
        self.current_color_images = []
        self.current_color_filtered_images = []
        self.current_color_searchable_count = 0
        self.current_color_indexed_count = 0
        self.current_color_candidate_limit = 0
        self.current_search_scope_count = None
        self.current_inspiration_project_id = None
        self.current_inspiration_terms = []
        self.current_inspiration_plan_filters = []
        self.current_inspiration_raw_term_results = []
        self.current_inspiration_images = []
        self.current_inspiration_filtered_images = []
        self.current_inspiration_matches = {}
        self.current_creative_node_images = []
        self.current_creative_node_filtered_images = []
        self.current_creative_node_searchable_count = 0
        self.current_creative_node_candidate_limit = 0
        self.current_creative_node_badges = {}
        self.current_chain_images = []
        self.current_chain_filtered_images = []
        self.current_chain_result = SearchChainResult(images=[])
        self.current_chain_base_image_ids = None
        self.current_chain_base_label = None
        self.current_chain_operation_mode = "replace"
        self.search_input.clear()
        self.search_button.setEnabled(True)
        self._refresh_filter_chain_ui()
        self._reload_images()

    def _refresh_current_results_for_filters(self) -> None:
        self._refresh_filter_chain_ui()
        if self.search_filters:
            if self.search_button.isEnabled():
                self._execute_search_chain(operation_mode="recompute")
            return

        if self.current_result_mode == "color":
            if self.search_button.isEnabled():
                self._run_search()
            else:
                self._apply_color_result_filters()
                images = self.current_color_filtered_images
                self.grid_view.set_images(images)
                self._set_color_result_status(images)
                self._update_color_search_diagnostics(images)
            return

        if self.current_result_mode == "semantic":
            if self.search_input.text().strip() and self.search_button.isEnabled():
                self._run_search()
            else:
                self._apply_semantic_result_filters()
                images = self.current_semantic_filtered_images
                self.grid_view.set_images(images)
                self._set_semantic_result_status(images)
                self._update_search_diagnostics(images)
            return

        if self.current_result_mode == "keyword":
            images = self.store.list_images(
                text_query=self.current_keyword_query or "",
                status_filter=self._selected_status_filter(),
                tag_ids=[],
                tag_match_mode="any",
                folder_path_prefix=self._selected_folder_path_prefix(),
                collection_id=self._selected_collection_id(),
                virtual_filter=self._selected_virtual_filter(),
                limit=self.page_size,
                offset=0,
                sort_key=self._database_sort_key(),
                sort_desc=self.current_sort_desc,
            )
            self.current_offset = 0
            images = self._apply_result_management_filters(images)
            self.grid_view.set_images(images)
            self._set_result_status(f"关键词结果：{len(images)}{self._result_management_status_suffix()}")
            return

        if self.current_result_mode == "inspiration" and self.current_inspiration_terms:
            self._run_inspiration_search(
                self.current_inspiration_project_id or 0,
                self.current_inspiration_terms,
                plan_filters=self._selected_inspiration_plan_filters()
                or self.current_inspiration_plan_filters,
            )
            return

        if self.current_result_mode == "creative_node":
            self._apply_creative_node_result_filters()
            images = self.current_creative_node_filtered_images
            self.grid_view.set_images(images, badges_by_image_id=self.current_creative_node_badges)
            self._set_creative_node_result_status(images)
            self._update_creative_node_search_diagnostics(images)
            return

        if self.current_result_mode == "temp_project":
            images = self._apply_result_management_filters(
                self._apply_sidebar_filters(self.current_temp_project_images)
            )
            images = self._sort_images(images)
            self.grid_view.set_images(images, badges_by_image_id=self.current_temp_project_badges)
            project = (
                self.store.get_temporary_project(self.current_temp_project_id)
                if self.current_temp_project_id is not None
                else None
            )
            label = self._temporary_project_label(project)
            name = project.name if project is not None else label
            suffix = f" ｜ {project.summary}" if project is not None and project.summary else ""
            self._set_result_status(
                f"{label}：{name} ｜ {len(images)} 张{suffix}{self._result_management_status_suffix()}"
            )
            return

        if self.current_result_mode == "duplicate_group":
            images = self._apply_result_management_filters(
                self._apply_sidebar_filters(self.current_duplicate_images)
            )
            images = self._sort_images(images)
            self.load_more_button.setEnabled(False)
            self.grid_view.set_images(images)
            self._set_result_status(f"重复候选组：{len(images)} 张{self._result_management_status_suffix()}")
            return

        self._reload_images()

    def _refresh_visible_results_after_result_management_change(self) -> None:
        self._refresh_filter_chain_ui()
        if self.search_filters or self.current_result_mode == "search_chain":
            self._apply_search_chain_filters()
            images = self.current_chain_filtered_images
            self.grid_view.set_images(images)
            self._set_search_chain_result_status(tuple(self.search_filters), images)
            self._update_search_chain_diagnostics(tuple(self.search_filters), images)
            return

        if self.current_result_mode == "semantic":
            self._apply_semantic_result_filters()
            images = self.current_semantic_filtered_images
            self.grid_view.set_images(images)
            self._set_semantic_result_status(images)
            self._update_search_diagnostics(images)
            return

        if self.current_result_mode == "color":
            self._apply_color_result_filters()
            images = self.current_color_filtered_images
            self.grid_view.set_images(images)
            self._set_color_result_status(images)
            self._update_color_search_diagnostics(images)
            return

        if self.current_result_mode == "inspiration":
            self._apply_inspiration_result_filters()
            images = self.current_inspiration_filtered_images
            self.grid_view.set_images(images, badges_by_image_id=self._inspiration_badges_by_image_id())
            self._set_inspiration_result_status(images)
            self._update_inspiration_diagnostics(images)
            return

        if self.current_result_mode == "creative_node":
            self._apply_creative_node_result_filters()
            images = self.current_creative_node_filtered_images
            self.grid_view.set_images(images, badges_by_image_id=self.current_creative_node_badges)
            self._set_creative_node_result_status(images)
            self._update_creative_node_search_diagnostics(images)
            return

        if self.current_result_mode == "temp_project":
            images = self._apply_result_management_filters(
                self._apply_sidebar_filters(self.current_temp_project_images)
            )
            images = self._sort_images(images)
            self.grid_view.set_images(images, badges_by_image_id=self.current_temp_project_badges)
            project = (
                self.store.get_temporary_project(self.current_temp_project_id)
                if self.current_temp_project_id is not None
                else None
            )
            label = self._temporary_project_label(project)
            name = project.name if project is not None else label
            suffix = f" ｜ {project.summary}" if project is not None and project.summary else ""
            self._set_result_status(
                f"{label}：{name} ｜ {len(images)} 张{suffix}{self._result_management_status_suffix()}"
            )
            return

        if self.current_result_mode == "duplicate_group":
            images = self._apply_result_management_filters(
                self._apply_sidebar_filters(self.current_duplicate_images)
            )
            images = self._sort_images(images)
            self.load_more_button.setEnabled(False)
            self.grid_view.set_images(images)
            self._set_result_status(f"重复候选组：{len(images)} 张{self._result_management_status_suffix()}")
            return

        if self.current_result_mode == "keyword":
            self._refresh_current_results_for_filters()
            return

        self._reload_images()

    def _set_result_status(self, message: str) -> None:
        display_message = self._format_unified_result_status(message)
        self.result_state_label.setText(display_message)
        self.statusBar().showMessage(message)
        self._refresh_result_management_buttons()
        self._refresh_search_operation_controls()

    def _format_unified_result_status(self, context: str) -> str:
        try:
            total = self.store.count_images()
            missing = self.store.count_missing_images()
            scope_count = self._current_scope_count_for_status(total=total, missing=missing)
        except Exception:
            total = 0
            missing = 0
            scope_count = 0
        loaded = self.grid_view.rowCount() if hasattr(self, "grid_view") else 0
        result_count = loaded if self._has_visible_result_context() else "-"
        context_label = self._compact_status_context(context)
        parts = [
            f"总数 {total}",
            f"当前范围 {scope_count}",
            f"已加载 {loaded}",
            f"缺失 {missing}",
            f"结果 {result_count}",
        ]
        if context_label:
            parts.append(context_label)
        return " ｜ ".join(parts)

    def _current_scope_count_for_status(self, *, total: int, missing: int) -> int:
        collection_id = self._selected_collection_id()
        if collection_id is not None:
            return self.store.collection_image_counts().get(collection_id, 0)
        return max(0, total - missing)

    @staticmethod
    def _compact_status_context(context: str) -> str:
        clean = " ".join(str(context or "").split())
        if not clean:
            return ""
        if clean.startswith("全部图库"):
            return "全部图库"
        if clean.startswith("已加载 "):
            return "全部图库"
        return clean

    def _shuffle_current_grid_images(self) -> None:
        images = self.grid_view.images()
        if len(images) <= 1:
            self.statusBar().showMessage("当前没有足够的图片可打乱")
            return
        random.shuffle(images)
        self._set_manual_result_order(images)
        self.grid_view.set_images(
            images,
            badges_by_image_id=self._current_grid_badges_by_image_id(),
        )
        self.statusBar().showMessage(f"已打乱当前显示的 {len(images)} 张图片")

    def _current_grid_badges_by_image_id(self) -> dict[int, list[str]]:
        return {
            image_id: list(badges)
            for image_id, badges in getattr(self.grid_view, "_badges_by_image_id", {}).items()
        }

    def _selected_grid_image(self) -> ImageItem | None:
        return self.grid_view.current_image() or self.selected_image

    def _open_selected_original(self) -> None:
        image = self._selected_grid_image()
        self._open_image_original(image)

    def _open_image_original(self, image: ImageItem | None) -> None:
        if image is None:
            self.statusBar().showMessage("没有选中文件")
            return
        path = Path(image.file_path)
        if image.is_missing or not path.exists():
            QMessageBox.warning(self, "Eidory", "源文件不存在，无法打开。")
            return
        ok = QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        self.statusBar().showMessage("已打开源文件" if ok else "打开源文件失败")

    def _reveal_selected_in_finder(self) -> None:
        image = self._selected_grid_image()
        if image is None:
            self.statusBar().showMessage("没有选中图片")
            return
        path = Path(image.file_path)
        if image.is_missing or not path.exists():
            QMessageBox.warning(self, "Eidory", "源文件不存在，无法在 Finder 中显示。")
            return
        subprocess.run(["open", "-R", str(path)], check=False)
        self.statusBar().showMessage("已在 Finder 中显示")

    def _copy_selected_path(self) -> None:
        image = self._selected_grid_image()
        if image is None:
            self.statusBar().showMessage("没有选中图片")
            return
        QApplication.clipboard().setText(image.file_path)
        self.statusBar().showMessage("已复制路径")

    def _selected_grid_images(self) -> list[ImageItem]:
        selected = self.grid_view.selected_images()
        if selected:
            return selected
        image = self._selected_grid_image()
        return [image] if image is not None else []

    def _refresh_temp_project_save_button(self) -> None:
        if not hasattr(self, "save_temp_project_button"):
            return
        count = len(self._selected_grid_images())
        self.save_temp_project_button.setEnabled(count > 0)
        if hasattr(self, "save_selection_to_creative_node_button"):
            self.save_selection_to_creative_node_button.setEnabled(
                count > 0 and self.current_creative_node_id is not None
            )
        if hasattr(self, "export_selection_button"):
            self.export_selection_button.setEnabled(count > 0)
        if hasattr(self, "rebuild_selected_thumbnails_button"):
            self.rebuild_selected_thumbnails_button.setEnabled(count > 0)
        if hasattr(self, "remove_selected_index_button"):
            self.remove_selected_index_button.setEnabled(count > 0)
        if self.current_language == "en":
            self.save_temp_project_button.setText(
                f"Save {count} Selected" if count else "Save Selected"
            )
            if hasattr(self, "export_selection_button"):
                self.export_selection_button.setText("Export Images")
        else:
            self.save_temp_project_button.setText(
                f"存为语义探针 {count} 张" if count else "存为语义探针项目"
            )
            if hasattr(self, "export_selection_button"):
                self.export_selection_button.setText("导出图片")

    @staticmethod
    def _temporary_project_ui_kind(project) -> str:
        return "quick" if project is not None and getattr(project, "kind", "semantic") == "quick" else "temporary"

    @staticmethod
    def _temporary_project_label(project) -> str:
        return "暂时收藏" if project is not None and getattr(project, "kind", "semantic") == "quick" else "语义探针项目"

    def _current_temporary_project_kind(self) -> str | None:
        if self.current_result_mode != "temp_project" or self.current_temp_project_id is None:
            return None
        project = self.store.get_temporary_project(self.current_temp_project_id)
        if project is None:
            return None
        return project.kind

    def _save_selected_images_as_temporary_project(self, *, kind: str = "semantic") -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        clean_kind = "quick" if kind == "quick" else "semantic"
        label = "暂时收藏" if clean_kind == "quick" else "语义探针项目"
        if len(images) > 1 and not self._confirm_batch_operation(
            f"保存为{label}",
            f"把选中图片保存为一个{label}",
            images,
        ):
            return
        default_name = self._suggest_temporary_project_name(images)
        name, ok = QInputDialog.getText(
            self,
            f"保存为{label}",
            "项目名称：",
            QLineEdit.EchoMode.Normal,
            default_name,
        )
        if not ok:
            return
        clean_name = name.strip()
        if not clean_name:
            self.statusBar().showMessage("项目名称不能为空")
            return
        project_id = self.store.create_temporary_project(
            clean_name,
            [image.id for image in images],
            kind=clean_kind,
        )
        intent_labels, intent_queries = self._temporary_project_intents_for_images(images)
        if intent_labels or intent_queries:
            self.store.add_images_to_temporary_project(
                project_id,
                [image.id for image in images],
                intent_labels=intent_labels,
                intent_queries=intent_queries,
            )
        select_kind = "quick" if clean_kind == "quick" else "temporary"
        self._expand_project_sidebar_section(select_kind)
        self._refresh_temporary_projects(select_project_id=project_id, select_kind=select_kind)
        project = self.store.get_temporary_project(project_id)
        project_name = project.name if project is not None else clean_name
        self._record_operation_history(f"新建{label}“{project_name}”：{len(images)} 张")
        self.statusBar().showMessage(f"已保存 {len(images)} 张到{label}“{project_name}”")
        if clean_kind == "semantic":
            self._suggest_temporary_project_details(
                project_id=project_id,
                images=images,
                can_rename=clean_name == default_name,
            )

    def _save_current_visible_results_as_temporary_project(self) -> None:
        images = self.grid_view.images()
        if not images:
            self.statusBar().showMessage("当前没有可保存的结果")
            return
        if not self._confirm_batch_operation(
            "保存当前结果集",
            "把当前可见结果保存为语义探针项目",
            images,
        ):
            return
        default_name = self._suggest_current_result_set_name(images)
        name, ok = QInputDialog.getText(
            self,
            "保存当前结果集",
            f"项目名称（当前结果 {len(images)} 张）：",
            QLineEdit.EchoMode.Normal,
            default_name,
        )
        if not ok:
            return
        clean_name = name.strip()
        if not clean_name:
            self.statusBar().showMessage("项目名称不能为空")
            return
        project_id = self.store.create_temporary_project(
            clean_name,
            [image.id for image in images],
            kind="semantic",
        )
        intent_labels, intent_queries = self._temporary_project_intents_for_images(images)
        if intent_labels or intent_queries:
            self.store.add_images_to_temporary_project(
                project_id,
                [image.id for image in images],
                intent_labels=intent_labels,
                intent_queries=intent_queries,
        )
        self._expand_project_sidebar_section("temporary")
        self._refresh_temporary_projects(select_project_id=project_id, select_kind="temporary")
        self._record_operation_history(f"保存当前结果集为语义探针项目“{clean_name}”：{len(images)} 张")
        self.statusBar().showMessage(f"已保存当前结果集 {len(images)} 张到语义探针项目“{clean_name}”")

    def _suggest_current_result_set_name(self, images: list[ImageItem]) -> str:
        if self.current_result_mode == "temp_project" and self.current_temp_project_id is not None:
            project = self.store.get_temporary_project(self.current_temp_project_id)
            if project is not None:
                return f"{project.name} 结果"
        if self.current_result_mode == "inspiration" and self.current_inspiration_terms:
            return " / ".join(term.title for term in self.current_inspiration_terms[:2])[:28]
        if self.search_filters:
            return self._format_filter_chain(self.search_filters[:2])[:28]
        return f"结果集 {len(images)} 张"

    def _exclude_selection_from_results(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        if not self._has_result_context():
            self.statusBar().showMessage("当前不是搜索或项目结果")
            return
        before = len(self.result_excluded_image_ids)
        self.result_excluded_image_ids.update(image.id for image in images)
        added = len(self.result_excluded_image_ids) - before
        self._refresh_visible_results_after_result_management_change()
        self._record_operation_history(f"从当前结果排除 {added} 张")
        self.statusBar().showMessage(f"已从当前结果排除 {added} 张")

    def _exclude_collection_from_results(self, collection_id: int) -> None:
        self._exclude_collections_from_results([collection_id])

    def _exclude_collections_from_results(self, collection_ids: list[int]) -> None:
        if not self._has_result_context():
            self.statusBar().showMessage("当前不是搜索或项目结果")
            return
        valid_ids = [
            collection_id
            for collection_id in collection_ids
            if self._collection_by_id(collection_id) is not None
        ]
        if not valid_ids:
            self.statusBar().showMessage("所选文件夹已不存在")
            return
        before = len(self.result_excluded_collection_ids)
        self.result_excluded_collection_ids.update(valid_ids)
        self._rebuild_result_excluded_collection_image_ids()
        self._refresh_visible_results_after_result_management_change()
        added = len(self.result_excluded_collection_ids) - before
        affected = len(self._image_ids_for_collections(valid_ids))
        label = (
            self._collection_path_text(valid_ids[0])
            if len(valid_ids) == 1
            else f"{len(valid_ids)} 个文件夹"
        )
        self.statusBar().showMessage(
            f"已从当前结果排除文件夹：{label}（新增 {added} 个，影响 {affected} 张）"
        )
        self._record_operation_history(f"从当前结果排除文件夹“{label}”，影响 {affected} 张")

    def _remove_result_collection_exclusion(self, collection_id: int) -> None:
        self.result_excluded_collection_ids.discard(collection_id)
        self._rebuild_result_excluded_collection_image_ids()
        self._refresh_visible_results_after_result_management_change()

    def _clear_result_collection_exclusions(self) -> None:
        self.result_excluded_collection_ids.clear()
        self.result_excluded_collection_image_ids.clear()
        self._refresh_visible_results_after_result_management_change()

    def _rebuild_result_excluded_collection_image_ids(self) -> None:
        excluded: set[int] = set()
        for collection_id in self.result_excluded_collection_ids:
            excluded.update(self.store.image_ids_for_collection(collection_id))
        self.result_excluded_collection_image_ids = excluded

    def _add_selection_to_temporary_project(self, project_id: int) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        project = self.store.get_temporary_project(project_id)
        if project is None:
            self._refresh_temporary_projects()
            self.statusBar().showMessage("该项目已不存在")
            return
        label = self._temporary_project_label(project)
        if len(images) > 1 and not self._confirm_batch_operation(
            f"加入{label}",
            f"加入已有{label}“{project.name}”",
            images,
        ):
            return
        intent_labels, intent_queries = self._temporary_project_intents_for_images(images)
        self.store.add_images_to_temporary_project(
            project_id,
            [image.id for image in images],
            intent_labels=intent_labels,
            intent_queries=intent_queries,
        )
        select_kind = self._temporary_project_ui_kind(project)
        self._expand_project_sidebar_section(select_kind)
        self._refresh_temporary_projects(
            select_project_id=project_id,
            select_kind=select_kind,
        )
        if self.current_result_mode == "temp_project" and self.current_temp_project_id == project_id:
            if (
                self.center_result_stack.currentWidget() is self.project_board_view
                and self._current_board_temp_project_id == project_id
            ):
                self._show_temporary_project_board(project_id)
            else:
                self._load_temporary_project(project_id)
        self._record_operation_history(f"加入 {len(images)} 张到{label}“{project.name}”")
        self.statusBar().showMessage(f"已加入 {len(images)} 张到“{project.name}”")

    def _remove_selection_from_current_temporary_project(self) -> None:
        if self.current_result_mode != "temp_project" or self.current_temp_project_id is None:
            self.statusBar().showMessage("当前不在项目结果中")
            return
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        project_id = self.current_temp_project_id
        project = self.store.get_temporary_project(project_id)
        label = self._temporary_project_label(project)
        project_name = project.name if project is not None else label
        if not self._confirm_batch_operation(
            f"从{label}移除",
            f"从“{project_name}”移除选中图片，不删除源文件",
            images,
            destructive=True,
        ):
            return
        removed = self.store.remove_images_from_temporary_project(
            project_id,
            [image.id for image in images],
        )
        self._refresh_temporary_projects(
            select_project_id=project_id,
            select_kind=self._temporary_project_ui_kind(project),
        )
        if self.store.get_temporary_project(project_id) is not None:
            self._load_temporary_project(project_id)
        else:
            self._reload_images()
        self._record_operation_history(f"从{label}“{project_name}”移除 {removed} 张")
        self.statusBar().showMessage(f"已从“{project_name}”移除 {removed} 张")

    def _group_selected_images_with_ai(self) -> None:
        images = self._selected_grid_images()
        if len(images) < 4:
            self.statusBar().showMessage("至少选中 4 张图片才能分组")
            return
        vectors_by_image_id = self._selected_ready_embedding_vectors(images)
        if len(vectors_by_image_id) < 4:
            QMessageBox.warning(
                self,
                "AI 分组",
                "至少需要 4 张已完成语义索引的图片。请先完成索引，或减少未索引/视频文件。",
            )
            return
        groups = cluster_reference_vectors(vectors_by_image_id, max_groups=6)
        if len(groups) <= 1:
            self.statusBar().showMessage("选中图片的视觉差异不足，未拆分成多个组")
            return
        group_contexts = self._reference_group_contexts(groups, images)
        provider = self._make_llm_provider()
        self.statusBar().showMessage(f"已分成 {len(groups)} 组，正在让 AI 命名...")

        def run() -> None:
            error_message = ""
            try:
                suggestions = provider.suggest_reference_group_names(
                    groups=group_contexts,
                    language=self.current_language,
                )
            except Exception as exc:
                error_message = str(exc)
                suggestions = self._fallback_reference_group_suggestions(groups)
            self.events.put(("reference_groups_done", (groups, suggestions, error_message)))

        self._start_background_task(run)

    def _selected_ready_embedding_vectors(self, images: list[ImageItem]) -> dict[int, object]:
        vectors: dict[int, object] = {}
        for image in images:
            if image.embedding_status != "ready" or is_supported_video(image.file_path):
                continue
            vector = self.store.embedding_vector_for_image(
                image.id,
                model_name=self.embedding_provider.model_name,
                model_revision=self.embedding_provider.model_revision,
                embedding_dim=self.embedding_provider.dim,
            )
            if vector is not None:
                vectors[image.id] = vector
        return vectors

    def _reference_group_contexts(
        self,
        groups: list[ReferenceGroup],
        images: list[ImageItem],
    ) -> list[dict[str, object]]:
        image_by_id = {image.id: image for image in images}
        contexts: list[dict[str, object]] = []
        for group in groups:
            file_names = [
                image_by_id[image_id].file_name
                for image_id in group.image_ids
                if image_id in image_by_id
            ]
            badges = [
                badge
                for image_id in group.image_ids
                for badge in self._badges_for_image_id(image_id)
            ]
            contexts.append({
                "file_names": file_names,
                "badges": list(dict.fromkeys(badges)),
            })
        return contexts

    def _badges_for_image_id(self, image_id: int) -> list[str]:
        if self.current_result_mode == "inspiration":
            matches = self.current_inspiration_matches.get(image_id, [])
            label = self._format_inspiration_badge(matches)
            return [label] if label else []
        if self.current_result_mode == "temp_project":
            return self.current_temp_project_badges.get(image_id, [])
        return self.grid_view._badges_by_image_id.get(image_id, [])

    @staticmethod
    def _fallback_reference_group_suggestions(groups: list[ReferenceGroup]) -> list[object]:
        from eidory.core.llm_provider import GroupNameSuggestion

        return [
            GroupNameSuggestion(name=f"AI 分组 {index}", summary=f"{len(group.image_ids)} 张参考图")
            for index, group in enumerate(groups, start=1)
        ]

    def _create_reference_group_projects(self, payload: object, *, confirm: bool = False) -> None:
        groups, suggestions, error_message = payload
        group_pairs = [
            (group, suggestion)
            for group, suggestion in zip(groups, suggestions, strict=False)
            if group.image_ids
        ]
        if not group_pairs:
            self.statusBar().showMessage("AI 分组没有可保存的图片")
            return
        if confirm and not self._confirm_reference_group_projects(group_pairs, str(error_message or "")):
            self.statusBar().showMessage("已取消 AI 分组保存")
            return
        created = 0
        batch_color = self.store.next_temporary_project_color()
        for group, suggestion in group_pairs:
            project_id = self.store.create_temporary_project(
                suggestion.name,
                group.image_ids,
                summary=suggestion.summary,
                color_hex=batch_color,
                kind="semantic",
            )
            self.store.add_images_to_temporary_project(
                project_id,
                group.image_ids,
                intent_labels={image_id: suggestion.name for image_id in group.image_ids},
            )
            created += 1
        self._expand_project_sidebar_section("temporary")
        self._refresh_temporary_projects()
        if error_message:
            self.statusBar().showMessage(f"已创建 {created} 个 AI 分组；命名失败，使用备用名称：{error_message}")
        else:
            self.statusBar().showMessage(f"已创建 {created} 个 AI 分组语义探针项目")

    def _confirm_reference_group_projects(self, group_pairs: list[tuple[ReferenceGroup, object]], error_message: str) -> bool:
        preview = "\n".join(self._reference_group_preview_lines(group_pairs))
        message = f"将创建 {len(group_pairs)} 个语义探针项目：\n\n{preview}\n\n继续？"
        if error_message:
            message += f"\n\nAI 命名失败时会使用备用名称：{error_message}"
        answer = QMessageBox.question(
            self,
            "确认 AI 分组",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _reference_group_preview_lines(self, group_pairs: list[tuple[ReferenceGroup, object]]) -> list[str]:
        image_ids = [
            image_id
            for group, _suggestion in group_pairs
            for image_id in group.image_ids[:3]
        ]
        image_by_id = {image.id: image for image in self.store.images_by_ids(image_ids)}
        lines: list[str] = []
        for index, (group, suggestion) in enumerate(group_pairs, start=1):
            name = str(getattr(suggestion, "name", f"AI 分组 {index}")).strip() or f"AI 分组 {index}"
            summary = str(getattr(suggestion, "summary", "")).strip()
            file_names = [
                image_by_id[image_id].file_name
                for image_id in group.image_ids[:3]
                if image_id in image_by_id
            ]
            suffix = "..." if len(group.image_ids) > 3 else ""
            samples = f"：{', '.join(file_names)}{suffix}" if file_names else ""
            summary_text = f" - {summary}" if summary else ""
            lines.append(f"{index}. {name}（{len(group.image_ids)} 张）{summary_text}{samples}")
        return lines[:8]

    def _temporary_project_intents_for_images(
        self,
        images: list[ImageItem],
    ) -> tuple[dict[int, str], dict[int, str]]:
        if self.current_result_mode == "inspiration":
            labels: dict[int, str] = {}
            queries: dict[int, str] = {}
            for image in images:
                matches = self.current_inspiration_matches.get(image.id, [])
                if not matches:
                    continue
                labels[image.id] = self._format_inspiration_badge(matches)
                queries[image.id] = matches[0].query
            return labels, queries
        labels: dict[int, str] = {}
        for image in images:
            badges = self._badges_for_image_id(image.id)
            if badges:
                labels[image.id] = badges[0]
        return labels, {}

    @staticmethod
    def _format_inspiration_badge(matches: list[InspirationMatch]) -> str:
        if not matches:
            return ""
        text = matches[0].term_title
        if len(matches) > 1:
            text = f"{text} +{len(matches) - 1}"
        return text

    def _suggest_temporary_project_details(
        self,
        *,
        project_id: int,
        images: list[ImageItem],
        can_rename: bool,
    ) -> None:
        provider = self._make_llm_provider()
        brief = self.inspiration_brief_input.toPlainText().strip()
        labels, _queries = self._temporary_project_intents_for_images(images)
        if not labels:
            project_badges = self.store.temporary_project_image_badges(project_id)
            labels = {
                image_id: badges[0]
                for image_id, badges in project_badges.items()
                if badges
            }
        selected_terms = sorted(set(labels.values()))
        file_names = [image.file_name for image in images[:24]]

        def run() -> None:
            try:
                suggestion = provider.suggest_project_details(
                    brief=brief,
                    selected_terms=selected_terms,
                    file_names=file_names,
                    language=self.current_language,
                )
                self.events.put(("temp_project_suggestion", (project_id, can_rename, suggestion)))
            except Exception as exc:
                self.events.put(("temp_project_suggestion_error", (project_id, exc)))

        self._start_background_task(run)

    def _apply_temporary_project_suggestion(self, payload: object) -> None:
        project_id, can_rename, suggestion = payload
        project = self.store.get_temporary_project(int(project_id))
        if project is None:
            return
        updated = self.store.update_temporary_project_details(
            int(project_id),
            name=suggestion.name if can_rename else None,
            summary=suggestion.summary,
        )
        self._refresh_temporary_projects(
            select_project_id=int(project_id) if self.current_temp_project_id == int(project_id) else None,
            select_kind=self._temporary_project_ui_kind(project),
        )
        if self.current_temp_project_id == int(project_id):
            self._load_temporary_project(int(project_id))
        if updated is not None:
            self.statusBar().showMessage(f"AI 已更新{self._temporary_project_label(updated)}：{updated.name}")

    def _show_temporary_project_suggestion_error(self, payload: object) -> None:
        _project_id, exc = payload
        self.statusBar().showMessage(f"AI 项目命名失败：{exc}")

    def _suggest_temporary_project_name(self, images: list[ImageItem]) -> str:
        if self.current_result_mode == "inspiration":
            brief = self.inspiration_brief_input.toPlainText().strip()
            if brief:
                return brief[:18]
            if self.current_inspiration_terms:
                return " / ".join(term.title for term in self.current_inspiration_terms[:2])
        current = self._selected_grid_image()
        if current is not None:
            return Path(current.file_name).stem[:24]
        return f"暂存 {len(images)} 张"

    def _open_image_preview(self, image: ImageItem | None = None) -> None:
        start_image = image or self._selected_grid_image()
        if start_image is None:
            self.statusBar().showMessage("没有选中图片")
            return
        self._stop_video_preview()
        images = self.grid_view.images()
        if not images:
            return
        start_index = next(
            (index for index, item in enumerate(images) if item.id == start_image.id),
            max(0, self.grid_view.current_index()),
        )
        dialog = ImagePreviewDialog(
            images=images,
            start_index=start_index,
            store=self.store,
            semantic_query=self.current_semantic_query if self.current_result_mode == "semantic" else None,
            model_name=self.embedding_provider.model_name,
            model_revision=self.embedding_provider.model_revision,
            embedding_dim=self.embedding_provider.dim,
            thumbnail_dir=self.paths.thumbnail_dir,
            parent=self,
        )
        dialog.imageChanged.connect(self._sync_preview_selection)
        dialog.favoriteChanged.connect(self._handle_preview_favorite_changed)
        dialog.feedbackSaved.connect(self._handle_preview_feedback_saved)
        dialog.indexRemoved.connect(self._handle_preview_index_removed)
        dialog.exec()
        current = dialog.current_image()
        if current is not None and (self.selected_image is None or self.selected_image.id != current.id):
            self._sync_preview_selection(current)

    def _sync_preview_selection(self, image: ImageItem) -> None:
        self.selected_image = image
        self.grid_view.select_image_id(image.id)
        self._show_image_details(image)

    def _handle_preview_favorite_changed(self, image: ImageItem) -> None:
        self.selected_image = image
        self._show_image_details(image)
        if self._selected_status_filter() == "favorite" and not image.is_favorite:
            self._refresh_current_results_for_filters()

    def _handle_preview_feedback_saved(self, image: ImageItem, _label: str) -> None:
        self.selected_image = image
        self._refresh_feedback_buttons(image)
        self._update_search_diagnostics(self.current_semantic_filtered_images)

    def _handle_preview_index_removed(self, image: ImageItem) -> None:
        if self.selected_image is not None and self.selected_image.id == image.id:
            self.selected_image = None
        self.vector_index.invalidate()
        self._refresh_folders()
        self._refresh_collections()
        self._refresh_tags()
        if self.current_result_mode == "temp_project" and self.current_temp_project_id is not None:
            if (
                self.center_result_stack.currentWidget() is self.project_board_view
                and self._current_board_temp_project_id == self.current_temp_project_id
            ):
                self._show_temporary_project_board(self.current_temp_project_id)
            else:
                self._load_temporary_project(self.current_temp_project_id)
        else:
            self._refresh_current_results_for_filters()
        self._refresh_embedding_stats()
        self._refresh_ai_vision_stats()
        self.statusBar().showMessage(f"已从图库移除索引：{image.file_name}")

    def _refresh_feedback_buttons(self, image: ImageItem | None) -> None:
        can_feedback = (
            image is not None
            and self.current_result_mode == "semantic"
            and bool(self.current_semantic_query)
        )

        self.feedback_group.setExclusive(False)
        for button in self.feedback_buttons.values():
            button.blockSignals(True)
            button.setChecked(False)
            button.setEnabled(can_feedback)
            button.blockSignals(False)
        self.feedback_group.setExclusive(True)

        if not can_feedback or image is None or self.current_semantic_query is None:
            return

        label = self.store.get_search_feedback(
            query=self.current_semantic_query,
            image_id=image.id,
            model_name=self.embedding_provider.model_name,
            model_revision=self.embedding_provider.model_revision,
            embedding_dim=self.embedding_provider.dim,
        )
        if label in self.feedback_buttons:
            button = self.feedback_buttons[label]
            button.blockSignals(True)
            button.setChecked(True)
            button.blockSignals(False)

    def _save_search_feedback(self, label: str) -> None:
        image = self._selected_grid_image()
        query = self.current_semantic_query
        if image is None or self.current_result_mode != "semantic" or not query:
            self.statusBar().showMessage("只有语义搜索结果可以标注反馈")
            self._refresh_feedback_buttons(image)
            return

        self.store.upsert_search_feedback(
            query=query,
            image_id=image.id,
            model_name=self.embedding_provider.model_name,
            model_revision=self.embedding_provider.model_revision,
            embedding_dim=self.embedding_provider.dim,
            score=image.score,
            label=label,
        )
        self._refresh_feedback_buttons(image)
        self._update_search_diagnostics(self.current_semantic_filtered_images)
        self.statusBar().showMessage(f"搜索反馈已保存：{self._feedback_label_text(label)}")

    @staticmethod
    def _feedback_label_text(label: str) -> str:
        return {
            "relevant": "相关",
            "irrelevant": "不相关",
            "ignored": "忽略",
        }.get(label, label)

    def _make_tag_completer(self) -> QCompleter:
        completer = QCompleter(self.tag_completion_model, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        return completer

    def _show_grid_context_menu(self, image: ImageItem | None, global_position) -> None:
        self.selected_image = image
        selected_images = self._selected_grid_images()
        if len(selected_images) > 1:
            self._show_multi_selection_details(selected_images)
        else:
            self._show_image_details(image)
        context_image = image or (selected_images[0] if len(selected_images) == 1 else None)

        menu = self._build_grid_context_menu(
            selected_images=selected_images,
            context_image=context_image,
        )
        action = menu.exec(global_position)
        if action is None:
            return
        command = action.data()
        if command == "preview":
            self._open_image_preview(context_image)
        elif command == "compare_selection":
            self._compare_selected_images()
        elif command == "find_similar":
            self._find_similar_to_image(context_image)
        elif command == "open":
            self._open_selected_original()
        elif command == "reveal":
            self._reveal_selected_in_finder()
        elif command == "copy_path":
            self._copy_selected_path()
        elif command == "favorite":
            self._batch_set_favorite(True)
        elif command == "unfavorite":
            self._batch_set_favorite(False)
        elif command == "add_tags":
            self._batch_add_tags()
        elif command == "clear_tags":
            self._batch_clear_tags()
        elif command == "save_temp":
            self._save_selected_images_as_temporary_project(kind="semantic")
        elif command == "save_quick_project":
            self._save_selected_images_as_temporary_project(kind="quick")
        elif command == "save_to_creative_node":
            self._save_selection_to_current_creative_node()
        elif command == "remove_from_creative_node":
            self._remove_selection_from_current_creative_node()
        elif command == "export_selection":
            self._export_selected_images()
        elif command == "delete_source":
            self._delete_selected_source_files()
        elif command == "save_result_set":
            self._save_current_visible_results_as_temporary_project()
        elif command == "group_selection":
            self._group_selected_images_with_ai()
        elif isinstance(command, str) and command.startswith("temporary_project:"):
            self._add_selection_to_temporary_project(int(command.split(":", 1)[1]))
        elif isinstance(command, str) and command.startswith("quick_project:"):
            self._add_selection_to_temporary_project(int(command.split(":", 1)[1]))
        elif command == "remove_from_temp":
            self._remove_selection_from_current_temporary_project()
        elif command == "remove_from_quick_project":
            self._remove_selection_from_current_temporary_project()
        elif command == "add_to_collection":
            self._add_selection_to_collection_dialog()
        elif command == "move_to_collection":
            self._move_selection_to_collection_dialog()
        elif command == "remove_from_collection":
            self._remove_selection_from_current_collection()
        elif command == "exclude_from_results":
            self._exclude_selection_from_results()
        elif isinstance(command, str) and command.startswith("exclude_collection:"):
            self._exclude_collection_from_results(int(command.split(":", 1)[1]))

    def _build_grid_context_menu(
        self,
        *,
        selected_images: list[ImageItem],
        context_image: ImageItem | None,
    ) -> QMenu:
        menu = QMenu(self)
        actions: dict[str, object] = {}

        def add_action(target_menu: QMenu, key: str, text: str):
            action = target_menu.addAction(text)
            action.setData(key)
            actions[key] = action
            return action

        add_action(menu, "preview", "快速预览")
        add_action(menu, "compare_selection", "对比查看")
        add_action(menu, "find_similar", "查找相似图片")
        has_single_context = context_image is not None and len(selected_images) <= 1
        actions["preview"].setEnabled(has_single_context)
        actions["compare_selection"].setEnabled(len(selected_images) >= 2)
        actions["find_similar"].setEnabled(
            has_single_context
            and self._similar_image_blocking_reason(context_image, check_vector=False) is None
        )

        menu.addSeparator()

        file_menu = menu.addMenu("文件与导出")
        add_action(file_menu, "open", "打开源文件")
        add_action(file_menu, "reveal", "在 Finder 中显示")
        add_action(file_menu, "copy_path", "复制路径")
        file_menu.addSeparator()
        add_action(file_menu, "export_selection", "导出选中图片")
        file_menu.addSeparator()
        add_action(file_menu, "delete_source", "删除/移除图片...")

        marker_menu = menu.addMenu("收藏与标签")
        add_action(marker_menu, "favorite", f"收藏选中 {len(selected_images)} 张")
        add_action(marker_menu, "unfavorite", "取消收藏")
        marker_menu.addSeparator()
        add_action(marker_menu, "add_tags", "批量添加标签")
        add_action(marker_menu, "clear_tags", "清除选中图片标签")

        semantic_menu = menu.addMenu("语义探针项目")
        add_action(semantic_menu, "save_temp", "保存选中为语义探针项目")
        temp_project_menu = semantic_menu.addMenu("加入已有语义探针项目")
        temp_project_actions: dict[object, int] = {}
        for project in self.store.list_temporary_projects(kind="semantic"):
            project_action = temp_project_menu.addAction(f"{project.name} ({project.image_count})")
            project_action.setData(f"temporary_project:{project.id}")
            temp_project_actions[project_action] = project.id
        if not temp_project_actions:
            empty_action = temp_project_menu.addAction("没有可用语义探针项目")
            empty_action.setEnabled(False)
        add_action(semantic_menu, "remove_from_temp", "从当前语义探针项目移除")
        semantic_menu.addSeparator()
        add_action(semantic_menu, "group_selection", "AI 分组选中图片")

        creative_menu = menu.addMenu("创作节点项目")
        add_action(creative_menu, "save_to_creative_node", "存入当前创作节点")
        add_action(creative_menu, "remove_from_creative_node", "从当前节点移除")

        quick_menu = menu.addMenu("暂时收藏")
        add_action(quick_menu, "save_quick_project", "新建暂时收藏")
        quick_project_menu = quick_menu.addMenu("加入已有暂时收藏")
        quick_project_actions: dict[object, int] = {}
        for project in self.store.list_temporary_projects(kind="quick"):
            project_action = quick_project_menu.addAction(f"{project.name} ({project.image_count})")
            project_action.setData(f"quick_project:{project.id}")
            quick_project_actions[project_action] = project.id
        if not quick_project_actions:
            empty_action = quick_project_menu.addAction("没有可用暂时收藏")
            empty_action.setEnabled(False)
        add_action(quick_menu, "remove_from_quick_project", "从当前暂时收藏移除")

        collection_menu = menu.addMenu("文件夹归类")
        add_action(collection_menu, "add_to_collection", "添加到文件夹")
        add_action(collection_menu, "move_to_collection", "移动到文件夹")
        add_action(collection_menu, "remove_from_collection", "从当前文件夹移出")

        result_menu = menu.addMenu("当前结果")
        add_action(result_menu, "save_result_set", "保存当前结果集")
        add_action(result_menu, "exclude_from_results", "从当前结果排除选中")
        exclude_collection_menu = result_menu.addMenu("排除此图所在的文件夹")
        exclude_collection_actions: dict[object, int] = {}
        if context_image is not None:
            seen_collection_ids: set[int] = set()
            chains = self.store.collection_chains_for_image(context_image.id)
            for chain_index, chain in enumerate(chains):
                if chain_index > 0:
                    exclude_collection_menu.addSeparator()
                for level, collection in enumerate(reversed(chain)):
                    if collection.id in seen_collection_ids:
                        continue
                    seen_collection_ids.add(collection.id)
                    path = " / ".join(item.name for item in chain[: len(chain) - level])
                    prefix = "当前" if level == 0 else f"上级 {level}"
                    action = exclude_collection_menu.addAction(f"{prefix}：{path}")
                    action.setData(f"exclude_collection:{collection.id}")
                    exclude_collection_actions[action] = collection.id
        if not exclude_collection_actions:
            empty_action = exclude_collection_menu.addAction("这张图没有所属文件夹")
            empty_action.setEnabled(False)

        menu._eidory_submenus = [  # type: ignore[attr-defined]
            file_menu,
            marker_menu,
            semantic_menu,
            temp_project_menu,
            creative_menu,
            quick_menu,
            quick_project_menu,
            collection_menu,
            result_menu,
            exclude_collection_menu,
        ]

        has_selection = bool(selected_images)
        has_visible_results = bool(self.grid_view.images())
        has_result_context = self._has_result_context()
        for key in ["open", "reveal", "copy_path"]:
            actions[key].setEnabled(has_single_context)
        for key in [
            "export_selection",
            "delete_source",
            "favorite",
            "unfavorite",
            "add_tags",
            "clear_tags",
            "save_temp",
            "save_quick_project",
            "save_to_creative_node",
            "remove_from_creative_node",
            "remove_from_temp",
            "remove_from_quick_project",
            "add_to_collection",
            "move_to_collection",
            "remove_from_collection",
            "exclude_from_results",
        ]:
            actions[key].setEnabled(has_selection)
        actions["save_result_set"].setEnabled(has_visible_results and has_result_context)
        actions["group_selection"].setEnabled(len(selected_images) >= 4)
        actions["save_to_creative_node"].setEnabled(has_selection and self.current_creative_node_id is not None)
        actions["remove_from_creative_node"].setEnabled(
            has_selection
            and self.current_result_mode == "creative_node"
            and self.current_creative_node_id is not None
        )
        temp_project_menu.setEnabled(has_selection and bool(temp_project_actions))
        quick_project_menu.setEnabled(has_selection and bool(quick_project_actions))
        current_temp_kind = self._current_temporary_project_kind()
        actions["remove_from_temp"].setEnabled(
            has_selection
            and self.current_result_mode == "temp_project"
            and current_temp_kind == "semantic"
        )
        actions["remove_from_quick_project"].setEnabled(
            has_selection
            and self.current_result_mode == "temp_project"
            and current_temp_kind == "quick"
        )
        has_current_collection = self._selected_collection_id() is not None
        actions["move_to_collection"].setEnabled(has_selection and has_current_collection)
        actions["remove_from_collection"].setEnabled(has_selection and has_current_collection)
        actions["exclude_from_results"].setEnabled(has_selection and has_result_context)
        exclude_collection_menu.setEnabled(
            has_single_context and has_result_context and bool(exclude_collection_actions)
        )

        file_menu.setEnabled(has_single_context or has_selection)
        marker_menu.setEnabled(has_selection)
        semantic_menu.setEnabled(has_selection)
        quick_menu.setEnabled(has_selection)
        collection_menu.setEnabled(has_selection)
        result_menu.setEnabled(has_result_context and (has_selection or has_visible_results))
        return menu

    def _show_folder_context_menu(self, position) -> None:
        item = self.folder_tree.itemAt(position)
        if item is None:
            return
        self.folder_tree.setCurrentItem(item)
        scan_path = item.data(0, Qt.ItemDataRole.UserRole)
        folder_id = item.data(0, Qt.ItemDataRole.UserRole + 1)
        filter_path = item.data(0, Qt.ItemDataRole.UserRole + 2)

        menu = QMenu(self)
        rescan_action = menu.addAction("重新扫描所在根目录")
        import_collections_action = menu.addAction("从磁盘目录生成分类树")
        remove_action = menu.addAction("从图库移除该文件夹索引")
        has_folder = bool(scan_path and folder_id and filter_path)
        rescan_action.setEnabled(has_folder)
        import_collections_action.setEnabled(has_folder)
        remove_action.setEnabled(has_folder)

        action = menu.exec(self.folder_tree.viewport().mapToGlobal(position))
        if not has_folder:
            return
        if action == rescan_action:
            self._start_scan(str(scan_path))
        elif action == import_collections_action:
            self._create_collections_from_disk_folder(str(filter_path))
        elif action == remove_action:
            self._remove_folder_index(
                scan_path=str(scan_path),
                folder_id=int(folder_id),
                filter_path=str(filter_path),
            )

    def _show_tag_context_menu(self, position) -> None:
        item = self.tag_list.itemAt(position)
        if item is None:
            return
        self.tag_list.setCurrentItem(item)
        tag_id = item.data(Qt.ItemDataRole.UserRole)
        tag_name = item.data(Qt.ItemDataRole.UserRole + 1)

        menu = QMenu(self)
        rename_action = menu.addAction("重命名标签")
        delete_action = menu.addAction("删除标签")
        merge_action = menu.addAction("合并到其他标签")
        has_tag = tag_id is not None and tag_name is not None
        for action in [rename_action, delete_action, merge_action]:
            action.setEnabled(has_tag)

        action = menu.exec(self.tag_list.viewport().mapToGlobal(position))
        if not has_tag:
            return
        if action == rename_action:
            self._rename_tag(int(tag_id), str(tag_name))
        elif action == delete_action:
            self._delete_tag(int(tag_id), str(tag_name))
        elif action == merge_action:
            self._merge_tag(int(tag_id), str(tag_name))

    def _rename_selected_tag(self) -> None:
        tag_id, tag_name = self._selected_tag_context()
        if tag_id is None or tag_name is None:
            self.statusBar().showMessage("请先选择一个标签")
            return
        self._rename_tag(tag_id, tag_name)

    def _delete_selected_tag(self) -> None:
        tag_id, tag_name = self._selected_tag_context()
        if tag_id is None or tag_name is None:
            self.statusBar().showMessage("请先选择一个标签")
            return
        self._delete_tag(tag_id, tag_name)

    def _merge_selected_tag(self) -> None:
        tag_id, tag_name = self._selected_tag_context()
        if tag_id is None or tag_name is None:
            self.statusBar().showMessage("请先选择一个标签")
            return
        self._merge_tag(tag_id, tag_name)

    def _selected_tag_context(self) -> tuple[int | None, str | None]:
        item = self.tag_list.currentItem()
        if item is None:
            return None, None
        tag_id = item.data(Qt.ItemDataRole.UserRole)
        tag_name = item.data(Qt.ItemDataRole.UserRole + 1)
        if tag_id is None or tag_name is None:
            return None, None
        return int(tag_id), str(tag_name)

    def _show_collection_context_menu(self, position) -> None:
        item = self.collection_tree.itemAt(position)
        if item is not None:
            self.collection_tree.setCurrentItem(item)
        collection_id = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        collection_name = item.data(0, Qt.ItemDataRole.UserRole + 1) if item is not None else None

        menu, actions = self._build_collection_context_menu(item)
        has_collection = collection_id is not None

        action = menu.exec(self.collection_tree.viewport().mapToGlobal(position))
        if action == actions["new_root"]:
            self._create_collection(parent_id=None)
        elif action == actions["new_child"] and has_collection:
            self._create_collection(parent_id=int(collection_id))
        elif action == actions["rename"] and has_collection:
            self._rename_collection(int(collection_id), str(collection_name))
        elif action == actions["delete"] and has_collection:
            self._delete_collection(int(collection_id), str(collection_name))
        elif action == actions["add_selected"] and has_collection:
            self._assign_selected_images_to_collection(int(collection_id), str(collection_name))
        elif action == actions["import_flat"] and has_collection:
            self._choose_import_folder_for_collection(int(collection_id), preserve_structure=False)
        elif action == actions["import_tree"]:
            parent_id = int(collection_id) if collection_id is not None else None
            self._choose_import_folder_for_collection(parent_id, preserve_structure=True)

    def _build_collection_context_menu(self, item: QTreeWidgetItem | None) -> tuple[QMenu, dict[str, object]]:
        collection_id = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        has_collection = collection_id is not None
        is_virtual = item is not None and item.data(0, COLLECTION_VIRTUAL_FILTER_ROLE) is not None
        menu = QMenu(self)
        actions = {
            "new_root": menu.addAction("新建文件夹"),
            "new_child": menu.addAction("新建子文件夹"),
            "rename": menu.addAction("重命名"),
            "delete": menu.addAction("删除文件夹"),
            "add_selected": menu.addAction("把选中图片加入此文件夹"),
            "import_flat": menu.addAction("导入图片"),
            "import_tree": menu.addAction("导入文件夹"),
        }
        for key in ["new_child", "rename", "delete", "add_selected", "import_flat"]:
            actions[key].setEnabled(has_collection)
        actions["import_tree"].setEnabled(not is_virtual)
        return menu, actions

    def _on_grid_image_selected(self, image: ImageItem | None) -> None:
        self.selected_image = image
        self._show_image_details(image)
        if self.current_result_mode == "creative_node":
            self._refresh_creative_selection_panel([image] if image is not None else [])
        else:
            self._refresh_tag_panel_assignment([image] if image is not None else [])
        self._refresh_temp_project_save_button()

    def _on_grid_selection_changed(self, images: list[ImageItem]) -> None:
        if len(images) > 1:
            self.selected_image = images[-1]
            self._show_multi_selection_details(images)
        elif not images:
            self.selected_image = None
            self._show_collection_details(self._selected_collection_id())
        if self.current_result_mode == "creative_node":
            self._refresh_creative_selection_panel(images)
        else:
            self._refresh_tag_panel_assignment(images)
        self._refresh_temp_project_save_button()

    def _on_collection_selection_changed(self) -> None:
        item = self.collection_tree.currentItem()
        virtual_filter = item.data(0, COLLECTION_VIRTUAL_FILTER_ROLE) if item is not None else None
        if virtual_filter:
            self.current_virtual_filter = str(virtual_filter)
            self._clear_tag_virtual_filter_selection()
            self._clear_ai_vision_virtual_filter_selection()
        else:
            self.current_virtual_filter = None
            self._refresh_virtual_collection_filters()
        self.selected_image = None
        self._refresh_current_results_for_filters()
        self._show_collection_details(self._selected_collection_id())
        self._refresh_tag_panel_assignment([])
        self._refresh_temp_project_save_button()

    def _on_tag_list_selection_changed(self) -> None:
        selected_virtual_filter = self._selected_tag_virtual_filter()
        previous_virtual_filter = self.current_virtual_filter
        if selected_virtual_filter:
            self.current_virtual_filter = selected_virtual_filter
            self._clear_collection_selection()
            self._clear_ai_vision_virtual_filter_selection()
            self.selected_image = None
            self._refresh_current_results_for_filters()
            self._show_collection_details(None)
            self._refresh_tag_panel_assignment([])
            self._refresh_temp_project_save_button()
        elif previous_virtual_filter == "untagged":
            self.current_virtual_filter = None
            self._refresh_virtual_collection_filters()
            self._refresh_current_results_for_filters()
        self._refresh_tag_action_buttons()
        self._save_selected_tag_filter()

    def _on_ai_vision_virtual_filter_selection_changed(self) -> None:
        selected_virtual_filter = self._selected_ai_vision_virtual_filter()
        previous_virtual_filter = self.current_virtual_filter
        if selected_virtual_filter:
            self.current_virtual_filter = selected_virtual_filter
            self._clear_collection_selection()
            self._clear_tag_virtual_filter_selection()
            self.selected_image = None
            self._refresh_current_results_for_filters()
            self._show_collection_details(None)
            self._refresh_tag_panel_assignment([])
            self._refresh_temp_project_save_button()
        elif previous_virtual_filter == "un_ai_tagged":
            self.current_virtual_filter = None
            self._refresh_virtual_collection_filters()
            self._refresh_current_results_for_filters()

    def _clear_collection_selection(self) -> None:
        self.collection_tree.blockSignals(True)
        self.collection_tree.clearSelection()
        self.collection_tree.setCurrentItem(None)
        self.collection_tree.blockSignals(False)

    def _clear_tag_virtual_filter_selection(self) -> None:
        if not hasattr(self, "tag_list"):
            return
        self.tag_list.blockSignals(True)
        for index in range(self.tag_list.count()):
            item = self.tag_list.item(index)
            if item.data(COLLECTION_VIRTUAL_FILTER_ROLE):
                item.setSelected(False)
        self.tag_list.blockSignals(False)

    def _clear_ai_vision_virtual_filter_selection(self) -> None:
        if not hasattr(self, "ai_vision_virtual_filter_list"):
            return
        self.ai_vision_virtual_filter_list.blockSignals(True)
        self.ai_vision_virtual_filter_list.clearSelection()
        self.ai_vision_virtual_filter_list.setCurrentItem(None)
        self.ai_vision_virtual_filter_list.blockSignals(False)

    def _set_detail_controls_enabled(self, enabled: bool) -> None:
        for widget in [
            self.file_name_input,
            self.favorite_checkbox,
            self.note_input,
            self.rename_file_button,
            self.delete_source_button,
            self.play_pause_button,
            self.open_original_button,
            self.reveal_in_finder_button,
            self.copy_path_button,
        ]:
            widget.setEnabled(enabled)

    def _show_multi_selection_details(self, images: list[ImageItem]) -> None:
        self._save_pending_note()
        count = len(images)
        total_size = sum(image.file_size for image in images)
        ready_count = sum(1 for image in images if image.embedding_status == "ready")
        favorite_count = sum(1 for image in images if image.is_favorite)

        self._stop_video_preview()
        self.image_detail_widget.show()
        self.collection_detail_widget.hide()
        self.preview_stack.setCurrentWidget(self.preview_label)
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText(f"已选择 {count} 张")
        self._suppress_detail_auto_save = True
        self.file_name_input.setText(f"已选择 {count} 张")
        self._set_path_text("-")
        self.image_collections_label.setText("-")
        self.size_label.setText(f"{total_size:,} bytes")
        self.modified_label.setText("-")
        self.embedding_label.setText(f"ready {ready_count} / {count}")
        self.ai_vision_detail_label.setText("-")
        self.score_label.setText("-")
        self.favorite_checkbox.setChecked(favorite_count == count)
        self.tags_display.setPlainText("多选标签请到“标签”页编辑。")
        self.note_input.clear()
        self._suppress_detail_auto_save = False
        self._refresh_feedback_buttons(None)
        self._set_detail_controls_enabled(False)
        self.delete_source_button.setEnabled(True)
        if self.current_result_mode == "creative_node":
            self.tags_display.setPlainText("创作节点结果：使用下方按钮存入或移出当前节点。")
            self._set_batch_tag_controls_visible(False)
            self._refresh_creative_selection_panel(images)
        else:
            self._set_creative_selection_controls_visible(False)
            self._set_batch_tag_controls_visible(True)
            self._refresh_batch_tag_panel(images)
        self.statusBar().showMessage(f"已选择 {count} 张")

    def _show_image_details(self, image: ImageItem | None) -> None:
        self._save_pending_note()
        if image is None:
            self._show_collection_details(self._selected_collection_id())
            return

        self._set_detail_controls_enabled(True)
        self._set_batch_tag_controls_visible(False)
        self._set_creative_selection_controls_visible(False)
        self.image_detail_widget.show()
        self.collection_detail_widget.hide()
        self._suppress_detail_auto_save = True
        self.file_name_input.setText(image.file_name)
        self._set_path_text(image.file_path)
        collection_paths = self.store.collection_paths_for_image(image.id)
        self.image_collections_label.setText("；".join(collection_paths) if collection_paths else "未归类")
        self.size_label.setText(self._format_media_dimensions(image))
        self.modified_label.setText(image.modified_at or "-")
        self.embedding_label.setText(image.embedding_status)
        self.ai_vision_detail_label.setText(self._format_ai_vision_details(image.id))
        if self.current_result_mode == "inspiration" and image.id in self.current_inspiration_matches:
            matches = self.current_inspiration_matches[image.id]
            titles = "、".join(match.term_title for match in matches[:3])
            score_text = "-" if image.score is None else f"{image.score:.4f}"
            self.score_label.setText(f"{score_text} | {titles}")
        else:
            self.score_label.setText("-" if image.score is None else f"{image.score:.4f}")
        self.favorite_checkbox.setChecked(image.is_favorite)
        tags = self.store.get_image_tags(image.id)
        self.tags_display.setPlainText("\n".join(tags) if tags else "无标签")
        self.note_input.setPlainText(image.note or "")
        self._suppress_detail_auto_save = False
        self._refresh_feedback_buttons(image)
        if self.current_result_mode == "creative_node":
            self._refresh_creative_selection_panel([image])

        if is_supported_video(image.file_path):
            self._show_video_details(image)
            return

        self._stop_video_preview()
        self.preview_stack.setCurrentWidget(self.preview_label)
        self.play_pause_button.setEnabled(False)
        preview_path = image.thumbnail_path if image.thumbnail_path and Path(image.thumbnail_path).exists() else image.file_path
        pixmap = self._load_detail_preview_pixmap(preview_path) if not image.is_missing else QPixmap()
        if pixmap.isNull():
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("无法预览")
            return
        self.preview_label.setText("")
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _load_detail_preview_pixmap(self, image_path: str) -> QPixmap:
        target_size = self.preview_label.size()
        max_width = max(1, target_size.width() * 2)
        max_height = max(1, target_size.height() * 2)
        return _load_scaled_qt_pixmap(image_path, max_width, max_height)

    def _ensure_detail_video_preview(self) -> tuple[QVideoWidget, QMediaPlayer]:
        if self.video_widget is None:
            self.video_widget = QVideoWidget()
            self.video_widget.setMinimumHeight(180)
            self.video_widget.setStyleSheet("background:#2d3138;")
            self.preview_stack.addWidget(self.video_widget)
        if self.video_player is None:
            self.video_player = QMediaPlayer(self)
            self.video_audio_output = QAudioOutput(self)
            self.video_player.setAudioOutput(self.video_audio_output)
            self.video_player.playbackStateChanged.connect(self._update_video_play_button)
        self.video_player.setVideoOutput(self.video_widget)
        return self.video_widget, self.video_player

    def _show_collection_details(self, collection_id: int | None) -> None:
        if not hasattr(self, "collection_detail_widget"):
            return
        self._save_pending_note()
        self._stop_video_preview()
        self.selected_image = None
        self.image_detail_widget.hide()
        self.collection_detail_widget.show()
        self._refresh_feedback_buttons(None)
        self._set_detail_controls_enabled(False)
        self._set_batch_tag_controls_visible(False)
        self._set_creative_selection_controls_visible(False)

        virtual_filter = self._selected_virtual_filter() if collection_id is None else None
        if virtual_filter is not None:
            label = self._virtual_filter_label(virtual_filter)
            count = self.store.count_images_for_virtual_filter(virtual_filter)
            self.collection_detail_name_label.setText(label)
            self.collection_detail_path_label.setText(f"聚类 / {label}")
            self.collection_detail_count_label.setText(f"{count} 个")
            self.collection_detail_import_dir_label.setText("-")
            self.collection_detail_help_label.setText(
                f"{self._virtual_filter_help(virtual_filter)}选择图片后，这里会切换为图片详情。"
            )
            self.open_collection_import_dir_button.setEnabled(False)
            self.ai_vision_detail_label.setText("-")
            return

        if collection_id is None:
            total = self.store.count_images()
            missing = self.store.count_missing_images()
            available = max(0, total - missing)
            self.collection_detail_name_label.setText("全部文件夹")
            self.collection_detail_path_label.setText("全部文件夹")
            self.collection_detail_count_label.setText(f"{available} 个可用 / {missing} 个丢失 / 共 {total} 个")
            self.collection_detail_import_dir_label.setText("-")
            self.collection_detail_help_label.setText(
                "当前范围是全部文件夹。选择左侧文件夹可缩小范围；在图片墙选择图片后，这里会显示路径、标签、备注、AI 标签和删除/移除操作。"
            )
            self.open_collection_import_dir_button.setEnabled(False)
            self.ai_vision_detail_label.setText("-")
            return

        collection = self._collection_by_id(collection_id)
        if collection is None:
            self.collection_detail_name_label.setText("-")
            self.collection_detail_path_label.setText("-")
            self.collection_detail_count_label.setText("-")
            self.collection_detail_import_dir_label.setText("-")
            self.collection_detail_help_label.setText(
                "当前没有可显示的文件夹详情。请选择左侧文件夹，或在图片墙选择图片。"
            )
            self.open_collection_import_dir_button.setEnabled(False)
            self.ai_vision_detail_label.setText("-")
            return

        counts = self.store.collection_image_counts()
        import_dir = self._collection_import_directory(collection_id)
        self.collection_detail_name_label.setText(collection.name)
        self.collection_detail_path_label.setText(self._collection_path_text(collection_id))
        self.collection_detail_count_label.setText(f"{counts.get(collection_id, 0)} 个")
        self.collection_detail_import_dir_label.setText(str(import_dir))
        self.collection_detail_help_label.setText(
            "当前显示的是所选文件夹范围。可以把图片拖入中间图片墙导入；选择图片后，这里会切换为图片详情。"
        )
        self.open_collection_import_dir_button.setEnabled(True)

    def _show_video_details(self, image: ImageItem) -> None:
        video_widget, video_player = self._ensure_detail_video_preview()
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("")
        self.preview_stack.setCurrentWidget(video_widget)
        self.play_pause_button.setEnabled(not image.is_missing and Path(image.file_path).exists())
        self.play_pause_button.setText("播放")
        self.size_label.setText(self._format_media_dimensions(image))
        self.embedding_label.setText("无需语义索引")
        video_player.stop()
        if image.is_missing or not Path(image.file_path).exists():
            self.preview_stack.setCurrentWidget(self.preview_label)
            self.preview_label.setText("视频文件不存在")
            return
        video_player.setSource(QUrl.fromLocalFile(image.file_path))

    def _format_ai_vision_details(self, image_id: int) -> str:
        tags = self.store.ai_vision_tags_for_image(image_id)
        if tags is None:
            return "未识别" if self.current_language != "en" else "Not indexed"
        status = str(tags.get("status") or "")
        if status != "ready":
            error = str(tags.get("error_message") or "").strip()
            if error:
                return f"{status}: {error[:120]}"
            return status
        parts: list[str] = []
        for field in [
            "scene_location",
            "environment_type",
            "time_of_day",
            "weather",
            "shot_scale",
            "view_angle",
        ]:
            value = tags.get(field)
            if isinstance(value, str) and value:
                parts.append(ai_vision_label(field, value, language=self.current_language))
        lighting = tags.get("lighting")
        if isinstance(lighting, list) and lighting:
            label = " / ".join(
                ai_vision_label("lighting", str(value), language=self.current_language).split(": ", 1)[-1]
                for value in lighting
            )
            field_label = "Lighting" if self.current_language == "en" else "光照"
            parts.append(f"{field_label}: {label}")
        notes = str(tags.get("notes") or "").strip()
        if notes:
            parts.append(notes)
        return "\n".join(parts) if parts else "-"

    def _set_path_text(self, path_text: str) -> None:
        self.path_label.setPlainText(path_text)
        self.path_label.setToolTip(path_text)
        self._fit_path_label_height()

    def _fit_path_label_height(self) -> None:
        width = max(40, self.path_label.viewport().width())
        self.path_label.document().setTextWidth(width)
        height = int(self.path_label.document().size().height()) + 8
        self.path_label.setFixedHeight(max(34, height))

    def _set_batch_tag_controls_visible(self, visible: bool) -> None:
        self.batch_tags_widget.setVisible(visible)
        label = self.detail_form.labelForField(self.batch_tags_widget)
        if label is not None:
            label.setVisible(visible)
        if not visible:
            self.batch_tag_summary_label.setText("-")
            self.batch_tags_input.clear()
            self.batch_remove_tag_combo.clear()

    def _set_creative_selection_controls_visible(self, visible: bool) -> None:
        self.creative_selection_widget.setVisible(visible)
        label = self.detail_form.labelForField(self.creative_selection_widget)
        if label is not None:
            label.setVisible(visible)
        if not visible:
            self.creative_selection_summary_label.setText("-")
            self.creative_selection_add_button.setEnabled(False)
            self.creative_selection_remove_button.setEnabled(False)

    def _refresh_creative_selection_panel(self, images: list[ImageItem]) -> None:
        node = self._current_creative_node()
        if node is None or not images:
            self._set_creative_selection_controls_visible(False)
            return
        selected_ids = [image.id for image in images]
        node_image_ids = set(self.store.creative_node_image_ids(node.id, include_descendants=False))
        already_count = sum(1 for image_id in selected_ids if image_id in node_image_ids)
        can_add = already_count < len(selected_ids)
        can_remove = already_count > 0
        self.creative_selection_summary_label.setText(
            f"当前节点：{node.title}\n已选择 {len(selected_ids)} 张，{already_count} 张已在当前节点。"
        )
        self.creative_selection_add_button.setEnabled(can_add)
        self.creative_selection_remove_button.setEnabled(can_remove)
        self._set_creative_selection_controls_visible(True)

    def _refresh_batch_tag_panel(self, images: list[ImageItem]) -> None:
        image_ids = [image.id for image in images]
        count = len(image_ids)
        tag_counts = self.store.tag_counts_for_images(image_ids)
        tagged_count = self.store.count_images_with_tags(image_ids)
        no_tag_count = count - tagged_count

        common = [
            tag_name
            for tag_name, tag_count in tag_counts.items()
            if tag_count == count
        ]
        partial = [
            (tag_name, tag_count)
            for tag_name, tag_count in tag_counts.items()
            if tag_count < count
        ]
        self.batch_tag_summary_label.setText(
            self._format_batch_tag_summary(
                total=count,
                common=common,
                partial=partial,
                no_tag_count=no_tag_count,
            )
        )

        self.batch_remove_tag_combo.blockSignals(True)
        self.batch_remove_tag_combo.clear()
        for tag_name, tag_count in tag_counts.items():
            self.batch_remove_tag_combo.addItem(f"{tag_name} ({tag_count}/{count})", tag_name)
        self.batch_remove_tag_combo.blockSignals(False)
        has_tags = bool(tag_counts)
        self.batch_remove_tag_combo.setEnabled(has_tags)
        self.batch_remove_tag_button.setEnabled(has_tags)
        self.batch_clear_tags_button.setEnabled(has_tags)
        self.batch_add_tags_button.setEnabled(bool(images))
        self.batch_tags_input.setEnabled(bool(images))

    def _refresh_tag_panel_assignment(self, images: list[ImageItem] | None = None) -> None:
        if not hasattr(self, "tag_panel_selection_label"):
            return
        selected = list(images) if images is not None else self._selected_grid_images()
        image_ids = [image.id for image in selected]
        count = len(image_ids)
        if count == 0:
            self.tag_panel_selection_label.setText(
                "先在图片墙选择 1 张或多张图片，然后在这里添加、移除或清空标签。顶栏“标签”用于筛选；这里用于编辑标签。"
            )
            tag_counts: dict[str, int] = {}
        elif count == 1:
            self.tag_panel_selection_label.setText(f"已选择 1 张：{selected[0].file_name}")
            tag_counts = self.store.tag_counts_for_images(image_ids)
        else:
            tag_counts = self.store.tag_counts_for_images(image_ids)
            common = [
                tag_name
                for tag_name, tag_count in tag_counts.items()
                if tag_count == count
            ]
            partial = [
                f"{tag_name} ({tag_count}/{count})"
                for tag_name, tag_count in tag_counts.items()
                if tag_count < count
            ]
            common_text = "、".join(common) if common else "无"
            partial_text = "、".join(partial) if partial else "无"
            self.tag_panel_selection_label.setText(
                f"已选择 {count} 张\n共同标签：{common_text}\n部分标签：{partial_text}"
            )

        self.tag_panel_remove_combo.blockSignals(True)
        self.tag_panel_remove_combo.clear()
        for tag_name, tag_count in tag_counts.items():
            label = tag_name if count <= 1 else f"{tag_name} ({tag_count}/{count})"
            self.tag_panel_remove_combo.addItem(label, tag_name)
        self.tag_panel_remove_combo.blockSignals(False)
        has_selection = count > 0
        has_tags = bool(tag_counts)
        self.tag_panel_input.setEnabled(has_selection)
        self.tag_panel_add_button.setEnabled(has_selection)
        self.tag_panel_remove_combo.setEnabled(has_tags)
        self.tag_panel_remove_button.setEnabled(has_tags)
        self.tag_panel_clear_button.setEnabled(has_tags)

    @staticmethod
    def _format_batch_tag_summary(
        *,
        total: int,
        common: list[str],
        partial: list[tuple[str, int]],
        no_tag_count: int,
    ) -> str:
        common_text = "、".join(common) if common else "无"
        partial_text = (
            "、".join(f"{tag} ({count}/{total})" for tag, count in partial)
            if partial
            else "无"
        )
        return (
            f"共同标签：{common_text}\n"
            f"部分标签：{partial_text}\n"
            f"无标签：{no_tag_count}"
        )

    def _toggle_video_playback(self) -> None:
        image = self._selected_grid_image()
        if image is None or not is_supported_video(image.file_path):
            self.statusBar().showMessage("当前选中项不是视频")
            return
        if image.is_missing or not Path(image.file_path).exists():
            QMessageBox.warning(self, "Eidory", "视频文件不存在，无法播放。")
            return
        _video_widget, video_player = self._ensure_detail_video_preview()
        if video_player.source().isEmpty():
            video_player.setSource(QUrl.fromLocalFile(image.file_path))
        if video_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            video_player.pause()
        else:
            video_player.play()

    def _update_video_play_button(self, state) -> None:
        self.play_pause_button.setText(
            "暂停" if state == QMediaPlayer.PlaybackState.PlayingState else "播放"
        )

    @staticmethod
    def _format_media_dimensions(image: ImageItem) -> str:
        parts: list[str] = []
        if image.width and image.height:
            parts.append(f"{image.width} x {image.height}")
        if image.duration_ms is not None:
            parts.append(MainWindow._format_duration(image.duration_ms))
        parts.append(f"{image.file_size:,} bytes")
        return " / ".join(parts)

    @staticmethod
    def _format_duration(milliseconds: int) -> str:
        total_seconds = max(0, int(milliseconds / 1000))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _stop_video_preview(self) -> None:
        if self.video_player is not None:
            self.video_player.stop()
            self.video_player.setSource(QUrl())
        self.play_pause_button.setText("播放")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "root_splitter"):
            QTimer.singleShot(0, self._enforce_fixed_sidebar_widths)
        if hasattr(self, "path_label"):
            self._fit_path_label_height()
        selected_images = self.grid_view.selected_images()
        if len(selected_images) > 1:
            self._show_multi_selection_details(selected_images)
        elif self.selected_image is not None:
            if is_supported_video(self.selected_image.file_path):
                return
            self._show_image_details(self.selected_image)

    def _rename_current_file(self) -> None:
        if self.selected_image is None:
            return
        self._save_pending_note()
        image = self._rename_selected_image_if_needed(self.selected_image)
        if image is None:
            return
        self._refresh_current_results_for_filters()
        refreshed = self.store.get_image(image.id)
        self.selected_image = refreshed
        self._show_image_details(refreshed)

    def _queue_note_auto_save(self) -> None:
        if self._suppress_detail_auto_save or self.selected_image is None:
            return
        if not self.note_input.isEnabled():
            return
        self._pending_note_image_id = self.selected_image.id
        self._pending_note_text = self.note_input.toPlainText()
        self.note_auto_save_timer.start()

    def _save_pending_note(self) -> None:
        image_id = self._pending_note_image_id
        note_text = self._pending_note_text
        if image_id is None or note_text is None:
            return
        self.note_auto_save_timer.stop()
        self._pending_note_image_id = None
        self._pending_note_text = None
        self.store.update_note(image_id, note_text)
        if self.selected_image is not None and self.selected_image.id == image_id:
            refreshed = self.store.get_image(image_id)
            if refreshed is not None:
                self.selected_image = refreshed
        self.statusBar().showMessage("备注已自动保存")

    def _save_current_favorite(self, checked: bool) -> None:
        if self._suppress_detail_auto_save or self.selected_image is None:
            return
        image_id = self.selected_image.id
        self.store.update_favorite(image_id, checked)
        self._refresh_virtual_collection_counts()
        self._refresh_current_results_for_filters()
        refreshed = self.store.get_image(image_id)
        if refreshed is not None:
            self.selected_image = refreshed
        self.statusBar().showMessage("收藏已自动保存")

    def _rename_selected_image_if_needed(self, image: ImageItem) -> ImageItem | None:
        desired = self.file_name_input.text().strip()
        if not desired or desired == image.file_name:
            return image
        if "/" in desired or "\\" in desired:
            QMessageBox.warning(self, "Eidory", "文件名不能包含路径分隔符。")
            return None
        current_path = Path(image.file_path)
        if image.is_missing or not current_path.exists():
            QMessageBox.warning(self, "Eidory", "源文件不存在，不能重命名。")
            return None

        current_suffix = current_path.suffix
        desired_path = Path(desired)
        if not desired_path.suffix:
            desired = f"{desired}{current_suffix}"
            desired_path = Path(desired)
        if desired_path.suffix.lower() != current_suffix.lower():
            QMessageBox.warning(self, "Eidory", "暂不支持修改文件扩展名。")
            return None

        target_path = current_path.with_name(desired)
        if target_path == current_path:
            return image
        if target_path.exists():
            QMessageBox.warning(self, "Eidory", "同目录下已存在这个文件名。")
            return None
        try:
            current_path.rename(target_path)
            stat = target_path.stat()
            self.store.update_image_path_after_rename(
                image.id,
                file_path=str(target_path),
                file_size=stat.st_size,
                modified_time_ns=stat.st_mtime_ns,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Eidory", f"重命名失败：{exc}")
            return None

        refreshed = self.store.get_image(image.id)
        if refreshed is not None:
            self.statusBar().showMessage(f"已重命名为：{refreshed.file_name}")
            return refreshed
        return image

    def _clear_selected_tags(self) -> None:
        image = self._selected_grid_image()
        if image is None:
            self.statusBar().showMessage("没有选中图片")
            return
        removed = self.store.clear_tags_for_images([image.id])
        self.tags_display.clear()
        self._refresh_tags()
        self._refresh_virtual_collection_counts()
        self._refresh_current_results_for_filters()
        refreshed = self.store.get_image(image.id)
        self.selected_image = refreshed
        self._show_image_details(refreshed)
        self.statusBar().showMessage(f"已清除 {removed} 个标签关联")

    def _create_collection_from_button(self) -> None:
        self._create_collection(parent_id=self._selected_collection_id())

    def _create_collection(self, parent_id: int | None) -> None:
        text, ok = QInputDialog.getText(
            self,
            "新建文件夹",
            "文件夹名称：",
        )
        if not ok:
            return
        name = text.strip()
        if not name:
            self.statusBar().showMessage("文件夹名称不能为空")
            return
        try:
            collection_id = self.store.create_collection(name, parent_id)
        except ValueError:
            QMessageBox.warning(self, "Eidory", "同级文件夹下已存在这个名称。")
            return
        self._refresh_collections(select_collection_id=collection_id)
        self.statusBar().showMessage(f"已创建文件夹：{name}")

    def _rename_collection(self, collection_id: int, current_name: str) -> None:
        text, ok = QInputDialog.getText(
            self,
            "重命名文件夹",
            "新文件夹名称：",
            QLineEdit.EchoMode.Normal,
            current_name,
        )
        if not ok:
            return
        name = text.strip()
        if not name:
            self.statusBar().showMessage("文件夹名称不能为空")
            return
        try:
            changed = self.store.rename_collection(collection_id, name)
        except ValueError:
            QMessageBox.warning(self, "Eidory", "同级文件夹下已存在这个名称。")
            return
        if changed:
            self._refresh_collections(select_collection_id=collection_id)
            self._refresh_current_results_for_filters()
            self.statusBar().showMessage(f"文件夹已重命名为：{name}")

    def _delete_collection(self, collection_id: int, collection_name: str) -> None:
        answer = QMessageBox.question(
            self,
            "删除文件夹",
            f"删除文件夹“{collection_name}”及其子文件夹。文件夹内只属于这里的图片会从 Eidory 移除索引，但不会删除硬盘源文件。继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        affected_links, deleted_images, thumbnail_paths = self.store.delete_collection(collection_id)
        self._delete_thumbnail_files(thumbnail_paths)
        self.vector_index.invalidate()
        self.selected_image = None
        self._refresh_collections()
        self._refresh_tags()
        self._refresh_current_results_for_filters()
        self._refresh_embedding_stats()
        self._refresh_ai_vision_stats()
        self.statusBar().showMessage(
            f"已删除文件夹，移除 {affected_links} 个归类关联，"
            f"从 Eidory 移除 {deleted_images} 张图片索引，源文件未删除"
        )

    def _save_collection_tree_order(self, updates: list[tuple[int, int | None, int]]) -> None:
        try:
            self.store.update_collection_tree(updates)
        except ValueError:
            QMessageBox.warning(self, "Eidory", "无法移动：同级文件夹下存在重名。")
            self._refresh_collections()
            return
        self._refresh_collections(select_collection_id=self._selected_collection_id())

    def _assign_dropped_images_to_collection(
        self,
        collection_id: int,
        image_ids: list[int],
    ) -> None:
        collection_name = self._collection_name(collection_id) or "文件夹"
        inserted = self.store.assign_images_to_collection(image_ids, collection_id)
        self._refresh_collections(select_collection_id=collection_id)
        self._refresh_current_results_for_filters()
        self.statusBar().showMessage(f"已添加 {inserted} 张到文件夹“{collection_name}”")

    def _import_dropped_files_to_collection(
        self,
        collection_id: int,
        paths: list[str],
    ) -> None:
        file_paths, folder_paths = self._split_local_import_paths(paths)
        if not file_paths and not folder_paths:
            self.statusBar().showMessage("拖入内容里没有可导入的图片、视频或文件夹")
            return
        if not self._confirm_local_paths_import(
            parent_collection_id=collection_id,
            file_paths=file_paths,
            folder_paths=folder_paths,
        ):
            return
        self._start_local_paths_import(
            file_paths=file_paths,
            folder_paths=folder_paths,
            parent_collection_id=collection_id,
        )

    def _import_dropped_files_to_root(self, paths: list[str]) -> None:
        file_paths, folder_paths = self._split_local_import_paths(paths)
        if file_paths:
            self.statusBar().showMessage("拖入图片必须拖到具体 Eidory 文件夹；拖入文件夹可放到根层级")
            return
        if not folder_paths:
            self.statusBar().showMessage("拖入内容里没有可导入的文件夹")
            return
        if not self._confirm_local_paths_import(
            parent_collection_id=None,
            file_paths=[],
            folder_paths=folder_paths,
        ):
            return
        self._start_local_paths_import(
            file_paths=[],
            folder_paths=folder_paths,
            parent_collection_id=None,
        )

    def _import_dropped_files_to_selected_collection(self, paths: list[str]) -> None:
        collection_id = self._selected_collection_id()
        if collection_id is None:
            file_paths, folder_paths = self._split_local_import_paths(paths)
            if file_paths:
                self.statusBar().showMessage("拖入图片必须先选择一个 Eidory 文件夹")
                return
            if not folder_paths:
                self.statusBar().showMessage("拖入内容里没有可导入的文件夹")
                return
            if not self._confirm_local_paths_import(
                parent_collection_id=None,
                file_paths=[],
                folder_paths=folder_paths,
            ):
                return
            self._start_local_paths_import(
                file_paths=[],
                folder_paths=folder_paths,
                parent_collection_id=None,
            )
            return
        self._import_dropped_files_to_collection(collection_id, paths)

    def _import_drop_payload_to_selected_collection(self, payload: dict[str, object]) -> None:
        collection_id = self._selected_collection_id()
        raw_paths = payload.get("local_paths", [])
        if isinstance(raw_paths, list) and raw_paths:
            file_paths, folder_paths = self._split_local_import_paths([str(path) for path in raw_paths])
            if not file_paths and not folder_paths:
                self.statusBar().showMessage("拖入内容里没有可导入的图片、视频或文件夹")
                return
            if collection_id is None:
                if file_paths:
                    self.statusBar().showMessage("拖入图片必须先选择一个 Eidory 文件夹")
                    return
                if folder_paths and self._confirm_local_paths_import(
                    parent_collection_id=None,
                    file_paths=[],
                    folder_paths=folder_paths,
                ):
                    self._start_local_paths_import(
                        file_paths=[],
                        folder_paths=folder_paths,
                        parent_collection_id=None,
                    )
                return
            if file_paths or folder_paths:
                if not self._confirm_local_paths_import(
                    parent_collection_id=collection_id,
                    file_paths=file_paths,
                    folder_paths=folder_paths,
                ):
                    return
                self._start_local_paths_import(
                    file_paths=file_paths,
                    folder_paths=folder_paths,
                    parent_collection_id=collection_id,
                )
                return
        if collection_id is None:
            self.statusBar().showMessage("请选择左侧具体文件夹后再拖入图片")
            return
        self._import_drop_payload_to_collection(collection_id, payload)

    def _import_drop_payload_to_collection(
        self,
        collection_id: int,
        payload: dict[str, object],
    ) -> None:
        if not self._confirm_drop_import(collection_id, payload):
            return
        collection_name = self._collection_name(collection_id) or "文件夹"
        target_dir = self._collection_import_directory(collection_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        prepared_payload = dict(payload)
        dropped_image = prepared_payload.pop("image", None)
        if dropped_image is not None:
            image_bytes = self._dropped_image_png_bytes(dropped_image)
            if image_bytes is not None:
                prepared_payload["image_png_bytes"] = image_bytes

        self.statusBar().showMessage(f"保存拖入图片到“{collection_name}”")
        self._set_import_controls_enabled(False)

        def run() -> None:
            try:
                saved_paths = self._materialize_drop_payload(prepared_payload, target_dir)
                if not saved_paths:
                    raise FileNotFoundError("没有可导入的受支持图片或视频")
                self.events.put((
                    "drop_payload_materialized",
                    (
                        collection_id,
                        collection_name,
                        [str(path) for path in saved_paths],
                    ),
                ))
            except Exception as exc:
                self.events.put(("error", f"拖入保存失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=lambda: self._set_import_controls_enabled(True),
        )

    def _handle_drop_payload_materialized(
        self,
        *,
        collection_id: int,
        collection_name: str,
        saved_paths: list[str],
    ) -> None:
        self._start_near_duplicate_resolution(
            saved_paths,
            include_same_path=False,
            on_resolved=lambda accepted_paths, skip_paths, replaced_ids, skipped_count: (
                self._continue_drop_payload_materialized_after_duplicate_resolution(
                    collection_id=collection_id,
                    collection_name=collection_name,
                    saved_paths=saved_paths,
                    accepted_paths=accepted_paths,
                    skip_paths=skip_paths,
                    replaced_count=len(replaced_ids),
                    skipped_count=skipped_count,
                )
            ),
        )
        return

    def _continue_drop_payload_materialized_after_duplicate_resolution(
        self,
        *,
        collection_id: int,
        collection_name: str,
        saved_paths: list[str],
        accepted_paths: list[str],
        skip_paths: set[str],
        replaced_count: int,
        skipped_count: int,
    ) -> None:
        self._remove_materialized_import_files(saved_paths, skip_paths)
        if not accepted_paths:
            self._set_import_controls_enabled(True)
            self._refresh_after_scan_database_change(select_collection_id=collection_id)
            self.statusBar().showMessage(
                f"近似图片处理完成：替换 {replaced_count}，放弃 {skipped_count}，没有新图片导入"
            )
            return
        self.statusBar().showMessage(f"导入拖入图片到“{collection_name}”")

        def run() -> None:
            try:
                result = self.scanner.import_files(accepted_paths)
                assigned = self.store.assign_images_to_collection(
                    list(result.image_ids),
                    collection_id,
                )
                self.events.put((
                    "drop_import_done",
                    (
                        collection_id,
                        collection_name,
                        result.scanned_files,
                        result.new_files,
                        result.changed_files,
                        assigned,
                        list(result.image_ids),
                    ),
                ))
            except Exception as exc:
                self.events.put(("error", f"拖入导入失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=lambda: self._set_import_controls_enabled(True),
        )

    def _split_local_import_paths(self, paths: list[str]) -> tuple[list[str], list[str]]:
        file_paths: list[str] = []
        folder_paths: list[str] = []
        seen_files: set[str] = set()
        seen_folders: set[str] = set()
        for path in paths:
            expanded = os.path.abspath(os.path.expanduser(str(path)))
            if os.path.isdir(expanded):
                folder = Path(expanded)
                if folder.is_symlink() or folder.name.startswith(".") or expanded in seen_folders:
                    continue
                seen_folders.add(expanded)
                folder_paths.append(expanded)
            elif os.path.isfile(expanded) and is_supported_media(expanded):
                if expanded in seen_files:
                    continue
                seen_files.add(expanded)
                file_paths.append(expanded)
        return file_paths, folder_paths

    def _confirm_local_paths_import(
        self,
        *,
        parent_collection_id: int | None,
        file_paths: list[str],
        folder_paths: list[str],
    ) -> bool:
        if file_paths and parent_collection_id is None:
            self.statusBar().showMessage("拖入图片必须指定一个 Eidory 文件夹")
            return False
        target_name = (
            self._collection_path_text(parent_collection_id)
            if parent_collection_id is not None
            else "根层级"
        )
        lines = [f"目标：{target_name}"]
        if file_paths:
            target_dir = self._collection_import_directory(parent_collection_id)  # type: ignore[arg-type]
            lines.append(f"图片/视频：{len(file_paths)} 个，会复制到：{target_dir}")
        if folder_paths:
            lines.append(f"磁盘文件夹：{len(folder_paths)} 个，会按原目录树原地索引，不复制、不移动源文件。")
            preview_names = [Path(path).name for path in folder_paths[:6]]
            if preview_names:
                lines.append("文件夹：" + "、".join(preview_names))
            if len(folder_paths) > 6:
                lines.append(f"另有 {len(folder_paths) - 6} 个文件夹。")
        message = "\n".join(lines) + "\n\n继续导入？"
        answer = QMessageBox.question(
            self,
            "确认导入",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_drop_import(self, collection_id: int, payload: dict[str, object]) -> bool:
        local_paths = payload.get("local_paths", [])
        if not isinstance(local_paths, list):
            local_paths = []
        remote_urls = self._remote_urls_from_drop_payload(payload)
        has_image_data = payload.get("image") is not None
        if not local_paths and not remote_urls and not has_image_data:
            self.statusBar().showMessage("拖入内容里没有可导入的图片或视频")
            return False

        target_name = self._collection_path_text(collection_id)
        target_dir = self._collection_import_directory(collection_id)
        source_lines: list[str] = []
        if local_paths:
            source_lines.append(f"本地文件/文件夹：{len(local_paths)} 个")
        elif remote_urls:
            source_lines.append("网页图片/链接：将导入第一张可下载图片")
        elif has_image_data:
            source_lines.append("网页直接图片数据：1 张")
        if remote_urls and has_image_data:
            source_lines.append("如果网页图片链接不可下载，将用网页提供的直接图片数据兜底。")

        message = (
            f"导入到文件夹：{target_name}\n"
            f"保存位置：{target_dir}\n\n"
            + "\n".join(source_lines)
            + "\n\n导入会复制/下载文件，不移动原文件。继续？"
        )
        answer = QMessageBox.question(
            self,
            "确认导入图片",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _materialize_drop_payload(
        self,
        payload: dict[str, object],
        target_dir: Path,
    ) -> list[Path]:
        raw_paths = payload.get("local_paths", [])
        if not isinstance(raw_paths, list):
            raw_paths = []
        saved_paths: list[Path] = []
        if raw_paths:
            for raw_path in raw_paths:
                saved_paths.extend(self._copy_local_drop_source(str(raw_path), target_dir))
            return saved_paths

        for url in self._remote_urls_from_drop_payload(payload):
            downloaded = self._download_remote_media(url, target_dir)
            if downloaded is not None:
                return [downloaded]

        image_bytes = payload.get("image_png_bytes")
        if isinstance(image_bytes, bytes):
            saved_image = self._write_dropped_image_bytes(image_bytes, target_dir)
            return [saved_image] if saved_image is not None else []
        return []

    def _copy_local_drop_source(self, raw_path: str, target_dir: Path) -> list[Path]:
        source = Path(os.path.abspath(os.path.expanduser(raw_path)))
        if not source.exists() or source.is_symlink():
            return []
        candidates: list[Path]
        if source.is_dir():
            candidates = [
                path
                for path in source.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and is_supported_media(str(path))
            ]
        elif source.is_file() and is_supported_media(str(source)):
            candidates = [source]
        else:
            return []

        saved_paths: list[Path] = []
        for candidate in candidates:
            destination = self._unique_destination_path(
                target_dir,
                self._safe_import_filename(candidate.name),
            )
            if candidate.resolve() == destination.resolve():
                saved_paths.append(destination)
                continue
            shutil.copy2(candidate, destination)
            saved_paths.append(destination)
        return saved_paths

    def _copy_local_import_files_to_collection(
        self,
        file_paths: list[str],
        collection_id: int,
    ) -> list[str]:
        target_dir = self._collection_import_directory(collection_id)
        copied_paths: list[str] = []
        seen: set[str] = set()
        for raw_path in file_paths:
            source = Path(os.path.abspath(os.path.expanduser(raw_path)))
            if (
                not source.is_file()
                or source.is_symlink()
                or not is_supported_media(str(source))
            ):
                continue
            copied = self._copy_local_import_file(source, target_dir)
            normalized = str(copied)
            if normalized in seen:
                continue
            seen.add(normalized)
            copied_paths.append(normalized)
        return copied_paths

    def _copy_local_import_file(self, source: Path, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self._safe_import_filename(source.name)
        target_dir_resolved = target_dir.resolve()
        try:
            if source.resolve().parent == target_dir_resolved and source.name == safe_name:
                return source
        except OSError:
            pass
        destination = self._unique_destination_path(target_dir, safe_name)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return destination

    def _download_remote_media(self, url: str, target_dir: Path) -> Path | None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return None
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 Eidory/0.1",
                "Referer": f"{parsed.scheme}://{parsed.netloc}/",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                content_type = response.headers.get_content_type()
                suffix = self._suffix_for_remote_media(parsed.path, content_type)
                if suffix not in SUPPORTED_MEDIA_EXTENSIONS:
                    return None
                max_bytes = 80 * 1024 * 1024
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("远程文件超过 80MB")
                    chunks.append(chunk)
        except (urllib.error.URLError, TimeoutError):
            return None

        raw_name = Path(urllib.parse.unquote(parsed.path)).name
        if not raw_name or Path(raw_name).suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
            raw_name = f"web-image-{uuid.uuid4().hex[:8]}{suffix}"
        destination = self._unique_destination_path(target_dir, self._safe_import_filename(raw_name))
        destination.write_bytes(b"".join(chunks))
        return destination

    @staticmethod
    def _suffix_for_remote_media(path: str, content_type: str) -> str:
        suffix = Path(urllib.parse.unquote(path)).suffix.lower()
        if suffix in SUPPORTED_MEDIA_EXTENSIONS:
            return suffix
        guessed = mimetypes.guess_extension(content_type) or ""
        if guessed == ".jpe":
            guessed = ".jpg"
        return guessed.lower()

    def _dropped_image_png_bytes(self, image_data: object) -> bytes | None:
        qimage: QImage | None = None
        if isinstance(image_data, QImage):
            qimage = image_data
        elif hasattr(image_data, "toImage"):
            qimage = image_data.toImage()
        if qimage is None or qimage.isNull():
            return None
        buffer = QBuffer()
        if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
            return None
        if not qimage.save(buffer, "PNG"):
            return None
        return bytes(buffer.data())

    def _write_dropped_image_bytes(self, image_bytes: bytes, target_dir: Path) -> Path | None:
        destination = self._unique_destination_path(
            target_dir,
            f"clipboard-image-{uuid.uuid4().hex[:8]}.png",
        )
        destination.write_bytes(image_bytes)
        return destination

    def _remote_urls_from_drop_payload(self, payload: dict[str, object]) -> list[str]:
        candidates: list[str] = []
        raw_urls = payload.get("urls", [])
        if isinstance(raw_urls, list):
            candidates.extend(str(url) for url in raw_urls)
        for key in ["text", "html"]:
            value = payload.get(key, "")
            if isinstance(value, str) and value:
                candidates.extend(self._urls_from_text_or_html(value))

        unique: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            clean = html_lib.unescape(candidate).strip().strip("'\"")
            if not clean:
                continue
            parsed = urllib.parse.urlparse(clean)
            if parsed.scheme not in {"http", "https"} or clean in seen:
                continue
            seen.add(clean)
            unique.append(clean)
        return unique

    @staticmethod
    def _urls_from_text_or_html(value: str) -> list[str]:
        urls = re.findall(r"https?://[^\s\"'<>]+", value)
        urls.extend(
            match
            for match in re.findall(r"(?:src|href|data-src)=['\"]([^'\"]+)['\"]", value)
            if match.startswith(("http://", "https://"))
        )
        return urls

    def _open_selected_collection_import_dir(self) -> None:
        collection_id = self._selected_collection_id()
        if collection_id is None:
            self.statusBar().showMessage("请先选择具体文件夹")
            return
        directory = self._collection_import_directory(collection_id)
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def _assign_selected_images_to_collection(
        self,
        collection_id: int,
        collection_name: str,
    ) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        if len(images) > 1 and not self._confirm_batch_operation(
            "添加到文件夹",
            f"添加到文件夹“{collection_name}”",
            images,
        ):
            return
        inserted = self.store.assign_images_to_collection(
            [image.id for image in images],
            collection_id,
        )
        self._refresh_collections(select_collection_id=collection_id)
        self._refresh_current_results_for_filters()
        self._record_operation_history(f"添加 {inserted} 张到文件夹“{collection_name}”")
        self.statusBar().showMessage(f"已添加 {inserted} 张到文件夹“{collection_name}”")

    def _add_selection_to_collection_dialog(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        choices = self._collection_choices()
        if not choices:
            self.statusBar().showMessage("还没有文件夹")
            return
        labels = [label for label, _collection_id in choices]
        selected_label, ok = QInputDialog.getItem(
            self,
            "添加到文件夹",
            "选择文件夹：",
            labels,
            0,
            False,
        )
        if not ok:
            return
        index = labels.index(selected_label)
        collection_id = choices[index][1]
        collection_name = self._collection_name(collection_id) or selected_label
        self._assign_selected_images_to_collection(collection_id, collection_name)

    def _move_selection_to_collection_dialog(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        source_collection_id = self._selected_collection_id()
        if source_collection_id is None:
            self.statusBar().showMessage("请先选中一个 Eidory 文件夹")
            return
        choices = [
            (label, collection_id)
            for label, collection_id in self._collection_choices()
            if collection_id != source_collection_id
        ]
        if not choices:
            self.statusBar().showMessage("没有可移动到的其他文件夹")
            return
        labels = [label for label, _collection_id in choices]
        selected_label, ok = QInputDialog.getItem(
            self,
            "移动到文件夹",
            "选择目标文件夹：",
            labels,
            0,
            False,
        )
        if not ok:
            return
        target_collection_id = choices[labels.index(selected_label)][1]
        target_name = self._collection_name(target_collection_id) or selected_label
        if not self._confirm_batch_operation(
            "移动到文件夹",
            f"从当前文件夹移动到“{target_name}”",
            images,
        ):
            return
        inserted, removed_links, deleted_images, thumbnail_paths = self.store.move_images_to_collection(
            [image.id for image in images],
            source_collection_id=source_collection_id,
            target_collection_id=target_collection_id,
        )
        if deleted_images:
            self._delete_thumbnail_files(thumbnail_paths)
            self.vector_index.invalidate()
        self._refresh_collections(select_collection_id=source_collection_id)
        self._refresh_current_results_for_filters()
        self._refresh_embedding_stats()
        self._refresh_ai_vision_stats()
        self._record_operation_history(
            f"移动 {len(images)} 张到“{target_name}”：新增关联 {inserted}，移出关联 {removed_links}"
        )
        self.statusBar().showMessage(
            f"已移动到“{target_name}”：新增关联 {inserted}，移出关联 {removed_links}"
        )

    def _remove_selection_from_current_collection(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        collection_id = self._selected_collection_id()
        if collection_id is None:
            self.statusBar().showMessage("请先选中一个 Eidory 文件夹")
            return
        collection_name = self._collection_name(collection_id) or "当前文件夹"
        if not self._confirm_batch_operation(
            "从当前文件夹移出",
            f"从“{collection_name}”及其子文件夹移出。不会删除源文件；孤立项目会从图库索引移除",
            images,
            destructive=True,
        ):
            return
        removed_links, deleted_images, thumbnail_paths = self.store.remove_images_from_collection_subtree(
            [image.id for image in images],
            collection_id,
        )
        if deleted_images:
            self._delete_thumbnail_files(thumbnail_paths)
            self.vector_index.invalidate()
        self._refresh_collections(select_collection_id=collection_id)
        self._refresh_current_results_for_filters()
        self._refresh_embedding_stats()
        self._refresh_ai_vision_stats()
        self._record_operation_history(
            f"从“{collection_name}”移出关联 {removed_links} 个；移除孤立索引 {deleted_images} 个"
        )
        self.statusBar().showMessage(
            f"已移出关联 {removed_links} 个；从图库移除索引 {deleted_images} 个，源文件未删除"
        )

    def _choose_import_folder_for_collection(
        self,
        collection_id: int | None,
        *,
        preserve_structure: bool,
    ) -> None:
        if preserve_structure:
            folder = QFileDialog.getExistingDirectory(self, "选择要导入的磁盘文件夹")
            if folder:
                self._start_folder_tree_import([folder], parent_collection_id=collection_id)
            return
        if collection_id is None:
            self.statusBar().showMessage("导入图片必须先选择一个 Eidory 文件夹")
            return
        filters = "Media Files (*.jpg *.jpeg *.png *.webp *.mp4 *.mov *.m4v *.avi *.mkv *.webm)"
        files, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "选择要导入的图片或视频",
            str(Path.home()),
            filters,
        )
        if files:
            self._start_file_import(files, collection_id)

    def _create_collections_from_disk_folder(self, folder_path: str) -> None:
        base = self._normalize_folder_path(folder_path)
        images = self.store.list_images_for_folder_path_prefix(base)
        if not images:
            self.statusBar().showMessage("该磁盘文件夹下没有可生成分类的媒体文件")
            return
        base_name = Path(base).name or base
        assigned = 0
        for image in images:
            image_dir = self._normalize_folder_path(os.path.dirname(image.file_path))
            relative_dir = os.path.relpath(image_dir, base)
            names = [base_name]
            if relative_dir != ".":
                names.extend(part for part in relative_dir.split(os.sep) if part)
            collection_id = self.store.ensure_collection_path(names)
            if collection_id is not None:
                assigned += self.store.assign_images_to_collection([image.id], collection_id)
        self._refresh_collections()
        self._refresh_current_results_for_filters()
        self.statusBar().showMessage(f"已从磁盘目录生成分类树，新增 {assigned} 个媒体文件分类关联")

    def _remove_folder_index(self, *, scan_path: str, folder_id: int, filter_path: str) -> None:
        is_root = self._normalize_folder_path(scan_path) == self._normalize_folder_path(filter_path)
        label = Path(filter_path).name or filter_path
        message = (
            f"只从 Eidory 移除“{label}”的索引记录，不删除源文件。继续？"
            if not is_root
            else f"只从 Eidory 移除根目录“{label}”及其全部索引记录，不删除源文件。继续？"
        )
        answer = QMessageBox.question(
            self,
            "移除文件夹索引",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        if is_root:
            thumbnail_paths, removed = self.store.remove_folder_from_library(folder_id)
        else:
            thumbnail_paths, removed = self.store.remove_images_by_folder_path_prefix(filter_path)
        self._delete_thumbnail_files(thumbnail_paths)
        self.vector_index.invalidate()
        self.selected_image = None
        self._refresh_folders()
        self._refresh_collections()
        self._refresh_tags()
        self._refresh_current_results_for_filters()
        self._refresh_embedding_stats()
        self._refresh_ai_vision_stats()
        self.statusBar().showMessage(f"已移除 {removed} 张图片索引，源文件未删除")

    def _rename_tag(self, tag_id: int, current_name: str) -> None:
        text, ok = QInputDialog.getText(
            self,
            "重命名标签",
            "新标签名：",
            QLineEdit.EchoMode.Normal,
            current_name,
        )
        if not ok:
            return
        new_name = text.strip()
        if not new_name:
            self.statusBar().showMessage("标签名不能为空")
            return
        try:
            changed = self.store.rename_tag(tag_id, new_name)
        except ValueError:
            QMessageBox.warning(self, "Eidory", "该标签名已存在；如果要合并，请使用“合并到其他标签”。")
            return
        if changed:
            self._refresh_tags()
            self._refresh_current_results_for_filters()
            self.statusBar().showMessage(f"标签已重命名为：{new_name}")

    def _delete_tag(self, tag_id: int, tag_name: str) -> None:
        answer = QMessageBox.question(
            self,
            "删除标签",
            f"删除标签“{tag_name}”，并从所有图片上移除这个标签。继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        removed = self.store.delete_tag(tag_id)
        self._refresh_tags()
        self._refresh_virtual_collection_counts()
        self._refresh_current_results_for_filters()
        self.statusBar().showMessage(f"已删除标签“{tag_name}”，移除 {removed} 个关联")

    def _merge_tag(self, source_tag_id: int, source_name: str) -> None:
        candidates = [
            (tag, count)
            for tag, count in self.store.list_tags_with_counts()
            if tag.id != source_tag_id
        ]
        if not candidates:
            self.statusBar().showMessage("没有可合并的目标标签")
            return

        labels = [f"{tag.tag_name} ({count})" for tag, count in candidates]
        selected_label, ok = QInputDialog.getItem(
            self,
            "合并标签",
            f"把“{source_name}”合并到：",
            labels,
            0,
            False,
        )
        if not ok:
            return
        target_index = labels.index(selected_label)
        target_tag = candidates[target_index][0]
        answer = QMessageBox.question(
            self,
            "合并标签",
            f"把“{source_name}”合并到“{target_tag.tag_name}”，源标签会删除。继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        moved = self.store.merge_tag(source_tag_id, target_tag.id)
        self._refresh_tags()
        self._refresh_current_results_for_filters()
        self.statusBar().showMessage(
            f"已合并“{source_name}”到“{target_tag.tag_name}”，处理 {moved} 个关联"
        )

    def _batch_set_favorite(self, is_favorite: bool) -> None:
        images = self._selected_grid_images()
        image_ids = [image.id for image in images]
        if not image_ids:
            self.statusBar().showMessage("没有选中图片")
            return
        if len(images) > 1 and not self._confirm_batch_operation(
            "批量收藏",
            "收藏选中图片" if is_favorite else "取消收藏选中图片",
            images,
        ):
            return
        count = self.store.update_favorites(image_ids, is_favorite)
        self._refresh_current_results_for_filters()
        action = "收藏" if is_favorite else "取消收藏"
        self._record_operation_history(f"{action} {count} 张")
        self.statusBar().showMessage(f"已{action} {count} 张")

    def _tag_panel_add_tags(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        tags = self._parse_tag_panel_input(self.tag_panel_input.toPlainText())
        if not tags:
            self.statusBar().showMessage("没有输入标签")
            return
        if len(images) > 1 and not self._confirm_batch_operation(
            "批量添加标签",
            f"添加标签：{', '.join(tags)}",
            images,
        ):
            return
        inserted = self.store.add_tags_to_images([image.id for image in images], tags)
        self.tag_panel_input.clear()
        self._refresh_after_tag_assignment(images)
        self._record_operation_history(f"添加标签 {tags} 到 {len(images)} 张，新增关联 {inserted}")
        self.statusBar().showMessage(f"已添加 {inserted} 个标签关联")

    def _tag_panel_remove_selected_tag(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        tag_name = self.tag_panel_remove_combo.currentData()
        if not tag_name:
            self.statusBar().showMessage("没有可移除的标签")
            return
        if len(images) > 1 and not self._confirm_batch_operation(
            "批量移除标签",
            f"移除标签：{tag_name}",
            images,
        ):
            return
        removed = self.store.remove_tags_from_images(
            [image.id for image in images],
            [str(tag_name)],
        )
        self._refresh_after_tag_assignment(images)
        self._record_operation_history(f"移除标签 {tag_name}：{removed} 个关联")
        self.statusBar().showMessage(f"已移除 {removed} 个标签关联：{tag_name}")

    def _tag_panel_clear_tags(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        answer = QMessageBox.question(
            self,
            "清除标签",
            f"清除选中 {len(images)} 张图片的全部标签。继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        removed = self.store.clear_tags_for_images([image.id for image in images])
        self._refresh_after_tag_assignment(images)
        self._record_operation_history(f"清除 {len(images)} 张图片标签，移除关联 {removed}")
        self.statusBar().showMessage(f"已清除 {removed} 个标签关联")

    def _refresh_after_tag_assignment(self, previous_selection: list[ImageItem]) -> None:
        self._refresh_tags()
        self._refresh_virtual_collection_counts()
        self._refresh_current_results_for_filters()
        refreshed = self.grid_view.selected_images() or self.store.images_by_ids(
            [image.id for image in previous_selection]
        )
        if len(refreshed) == 1:
            self.selected_image = refreshed[0]
            self._show_image_details(refreshed[0])
        elif len(refreshed) > 1:
            self._show_multi_selection_details(refreshed)
        self._refresh_tag_panel_assignment(refreshed)

    def _batch_add_tags_from_panel(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        tags = self._parse_tag_input(self.batch_tags_input.text())
        if not tags:
            self.statusBar().showMessage("没有输入标签")
            return
        if not self._confirm_batch_operation(
            "批量添加标签",
            f"添加标签：{', '.join(tags)}",
            images,
        ):
            return
        inserted = self.store.add_tags_to_images([image.id for image in images], tags)
        self.batch_tags_input.clear()
        self._refresh_tags()
        self._refresh_virtual_collection_counts()
        self._refresh_current_results_for_filters()
        refreshed = self.grid_view.selected_images()
        if len(refreshed) > 1:
            self._show_multi_selection_details(refreshed)
        self._record_operation_history(f"添加标签 {tags} 到 {len(images)} 张，新增关联 {inserted}")
        self.statusBar().showMessage(f"已添加 {inserted} 个标签关联")

    def _batch_remove_selected_tag(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        tag_name = self.batch_remove_tag_combo.currentData()
        if not tag_name:
            self.statusBar().showMessage("没有可移除的标签")
            return
        if not self._confirm_batch_operation(
            "批量移除标签",
            f"移除标签：{tag_name}",
            images,
        ):
            return
        removed = self.store.remove_tags_from_images(
            [image.id for image in images],
            [str(tag_name)],
        )
        self._refresh_tags()
        self._refresh_virtual_collection_counts()
        self._refresh_current_results_for_filters()
        refreshed = self.grid_view.selected_images()
        if len(refreshed) > 1:
            self._show_multi_selection_details(refreshed)
        self._record_operation_history(f"移除标签 {tag_name}：{removed} 个关联")
        self.statusBar().showMessage(f"已移除 {removed} 个标签关联：{tag_name}")

    def _batch_add_tags(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        text, ok = QInputDialog.getText(
            self,
            "批量添加标签",
            "输入要追加的标签，用逗号或换行分隔：",
        )
        if not ok:
            return
        tags = self._parse_tag_input(text)
        if not tags:
            self.statusBar().showMessage("没有输入标签")
            return
        if not self._confirm_batch_operation(
            "批量添加标签",
            f"添加标签：{', '.join(tags)}",
            images,
        ):
            return
        inserted = self.store.add_tags_to_images([image.id for image in images], tags)
        self._refresh_tags()
        self._refresh_virtual_collection_counts()
        self._refresh_current_results_for_filters()
        self._record_operation_history(f"添加标签 {tags} 到 {len(images)} 张，新增关联 {inserted}")
        self.statusBar().showMessage(f"已添加 {inserted} 个标签关联")

    def _batch_clear_tags(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        answer = QMessageBox.question(
            self,
            "清除标签",
            f"清除选中 {len(images)} 张图片的全部标签。继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        removed = self.store.clear_tags_for_images([image.id for image in images])
        self._refresh_tags()
        self._refresh_virtual_collection_counts()
        self._refresh_current_results_for_filters()
        self._record_operation_history(f"清除 {len(images)} 张图片标签，移除关联 {removed}")
        self.statusBar().showMessage(f"已清除 {removed} 个标签关联")

    def _confirm_batch_operation(
        self,
        title: str,
        action_text: str,
        images: list[ImageItem],
        *,
        destructive: bool = False,
    ) -> bool:
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return True
        preview = "\n".join(f"• {image.file_name}" for image in images[:8])
        if len(images) > 8:
            preview += f"\n……另有 {len(images) - 8} 个"
        icon = QMessageBox.Icon.Warning if destructive else QMessageBox.Icon.Information
        message = QMessageBox(self)
        message.setIcon(icon)
        message.setWindowTitle(title)
        message.setText(f"{action_text}\n\n选择数量：{len(images)}")
        message.setInformativeText(preview)
        message.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        message.setDefaultButton(QMessageBox.StandardButton.No)
        return message.exec() == QMessageBox.StandardButton.Yes

    def _record_operation_history(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.operation_history_messages.append(f"[{timestamp}] {message}")
        self.operation_history_messages = self.operation_history_messages[-200:]

    def _show_operation_history(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("操作历史")
        dialog.resize(760, 460)
        layout = QVBoxLayout(dialog)
        intro = QLabel("这里记录本次启动后的批量操作。删除/移除类操作仍以 Cmd+Z 撤销为准。")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        output = QTextEdit()
        output.setReadOnly(True)
        output.setPlainText("\n".join(reversed(self.operation_history_messages)) or "暂无批量操作。")
        layout.addWidget(output, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    @staticmethod
    def _parse_tag_input(text: str) -> list[str]:
        seen: set[str] = set()
        tags: list[str] = []
        for part in re.split(r"[,，\n]", text):
            tag = part.strip()
            if not tag or tag in seen:
                continue
            seen.add(tag)
            tags.append(tag)
        return tags

    @staticmethod
    def _parse_tag_panel_input(text: str) -> list[str]:
        seen: set[str] = set()
        tags: list[str] = []
        for line in text.splitlines():
            tag = line.strip()
            if not tag or tag in seen:
                continue
            seen.add(tag)
            tags.append(tag)
        return tags

    def _batch_remove_from_library(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        if not self._confirm_batch_operation(
            "从图库移除索引",
            "只从 Eidory 移除索引记录，不删除源文件",
            images,
            destructive=True,
        ):
            return
        removed = self._remove_images_from_library_with_undo(
            images,
            undo_label=f"移除索引 {len(images)} 个",
        )
        self._record_operation_history(f"移除索引 {removed} 张，可 Cmd+Z 撤销")
        self.statusBar().showMessage(f"已从图库移除 {removed} 张，源文件未删除。按 Cmd+Z 可撤销")

    def _delete_selected_source_files(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        mode = self._ask_delete_or_remove_mode(len(images))
        if mode == "index":
            removed = self._remove_images_from_library_with_undo(
                images,
                undo_label=f"移除索引 {len(images)} 个",
            )
            self._record_operation_history(f"移除索引 {removed} 张，可 Cmd+Z 撤销")
            self.statusBar().showMessage(f"已从图库移除 {removed} 张，源文件未删除。按 Cmd+Z 可撤销")
        elif mode == "source":
            self._delete_source_files_with_undo(images)

    def _ask_delete_or_remove_mode(self, count: int) -> str | None:
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("删除/移除图片")
        message.setText(f"已选择 {count} 个项目。请选择要执行的操作。")
        message.setInformativeText(
            "“只从软件移除”不会删除硬盘源文件。\n"
            "“删除源文件”会先备份到临时撤销区，再把硬盘源文件移到废纸篓。"
        )
        remove_button = message.addButton("只从软件移除", QMessageBox.ButtonRole.ActionRole)
        delete_button = message.addButton("删除源文件", QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = message.addButton(QMessageBox.StandardButton.Cancel)
        message.setDefaultButton(cancel_button)
        message.exec()
        clicked = message.clickedButton()
        if clicked == remove_button:
            return "index"
        if clicked == delete_button:
            return "source"
        return None

    def _remove_images_from_library_with_undo(
        self,
        images: list[ImageItem],
        *,
        undo_label: str,
        snapshot: dict[str, object] | None = None,
        source_backups: list[dict[str, object]] | None = None,
        backup_dir: Path | None = None,
    ) -> int:
        clean_images = [image for image in images if image.id > 0]
        if not clean_images:
            return 0
        image_ids = [image.id for image in clean_images]
        undo_snapshot = snapshot or self.store.snapshot_images_for_restore(image_ids)
        self.store.remove_images_from_library(image_ids)
        self._invalidate_near_duplicate_hash_cache()
        self.vector_index.invalidate()
        self.selected_image = None
        self._register_removal_undo(
            image_ids=image_ids,
            snapshot=undo_snapshot,
            source_backups=source_backups or [],
            backup_dir=backup_dir,
            label=undo_label,
        )
        self._refresh_after_library_removal()
        return len(image_ids)

    def _delete_source_files_with_undo(self, images: list[ImageItem]) -> None:
        image_ids = [image.id for image in images]
        snapshot = self.store.snapshot_images_for_restore(image_ids)
        backup_dir = self.paths.data_dir / "undo-removal" / uuid.uuid4().hex
        deleted_ids: list[int] = []
        source_backups: list[dict[str, object]] = []
        failures: list[str] = []

        for image in images:
            path = Path(image.file_path)
            if image.is_missing or not path.exists():
                deleted_ids.append(image.id)
                continue
            if not path.is_file():
                failures.append(f"{image.file_name}: 不是普通文件")
                continue
            backup_path = backup_dir / f"{image.id}-{self._safe_import_filename(path.name)}"
            try:
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup_path)
            except Exception as exc:
                failures.append(f"{image.file_name}: 无法建立撤销备份：{exc}")
                continue
            ok, error = self._move_source_file_to_trash(path)
            if ok:
                deleted_ids.append(image.id)
                source_backups.append(
                    {
                        "image_id": image.id,
                        "original_path": str(path),
                        "backup_path": str(backup_path),
                    }
                )
            else:
                backup_path.unlink(missing_ok=True)
                failures.append(f"{image.file_name}: {error or '删除失败'}")

        if deleted_ids:
            deleted_images = [image for image in images if image.id in set(deleted_ids)]
            self._remove_images_from_library_with_undo(
                deleted_images,
                undo_label=f"删除源文件 {len(deleted_ids)} 个",
                snapshot=snapshot,
                source_backups=source_backups,
                backup_dir=backup_dir,
            )
        elif backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

        if failures:
            preview = "\n".join(failures[:6])
            if len(failures) > 6:
                preview += f"\n……另有 {len(failures) - 6} 个失败"
            QMessageBox.warning(
                self,
                "部分图片删除失败",
                f"已删除并移除索引 {len(deleted_ids)} 个；失败 {len(failures)} 个。\n\n{preview}",
            )
        if deleted_ids:
            self.statusBar().showMessage(
                f"已删除源文件并移除索引 {len(deleted_ids)} 个；失败 {len(failures)} 个。按 Cmd+Z 可撤销"
            )
            self._record_operation_history(
                f"删除源文件并移除索引 {len(deleted_ids)} 个，失败 {len(failures)} 个，可 Cmd+Z 撤销"
            )
        else:
            self.statusBar().showMessage(f"没有删除源文件；失败 {len(failures)} 个")

    @staticmethod
    def _move_source_file_to_trash(path: Path) -> tuple[bool, str]:
        try:
            result = QFile.moveToTrash(str(path))
            if isinstance(result, tuple):
                moved = bool(result[0])
            else:
                moved = bool(result)
            if moved:
                return True, ""
        except Exception as exc:
            return False, str(exc)
        return False, "无法移到废纸篓"

    def _register_removal_undo(
        self,
        *,
        image_ids: list[int],
        snapshot: dict[str, object],
        source_backups: list[dict[str, object]],
        backup_dir: Path | None,
        label: str,
    ) -> None:
        self._clear_last_removal_undo(cleanup_backups=True)
        self._last_removal_undo = {
            "image_ids": list(image_ids),
            "snapshot": snapshot,
            "source_backups": list(source_backups),
            "backup_dir": str(backup_dir) if backup_dir is not None else "",
            "label": label,
        }
        if hasattr(self, "undo_removal_action"):
            self.undo_removal_action.setEnabled(True)

    def _clear_last_removal_undo(self, *, cleanup_backups: bool) -> None:
        undo = self._last_removal_undo
        if undo is not None and cleanup_backups:
            backup_dir = undo.get("backup_dir")
            if isinstance(backup_dir, str) and backup_dir:
                shutil.rmtree(backup_dir, ignore_errors=True)
        self._last_removal_undo = None
        if hasattr(self, "undo_removal_action"):
            self.undo_removal_action.setEnabled(False)

    def _undo_last_library_removal(self) -> None:
        undo = self._last_removal_undo
        if undo is None:
            self.statusBar().showMessage("没有可撤销的删除/移除操作")
            return
        image_ids = [int(image_id) for image_id in undo.get("image_ids", [])]  # type: ignore[arg-type]
        restore_ids = set(image_ids)
        failures: list[str] = []
        for backup in undo.get("source_backups", []):  # type: ignore[union-attr]
            if not isinstance(backup, dict):
                continue
            image_id = int(backup.get("image_id", 0) or 0)
            original_path = Path(str(backup.get("original_path", "")))
            backup_path = Path(str(backup.get("backup_path", "")))
            if not original_path.exists():
                if not backup_path.exists():
                    failures.append(f"{original_path.name}: 撤销备份不存在")
                    restore_ids.discard(image_id)
                    continue
                try:
                    original_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(backup_path), str(original_path))
                except Exception as exc:
                    failures.append(f"{original_path.name}: 无法恢复源文件：{exc}")
                    restore_ids.discard(image_id)
                    continue

        snapshot = undo.get("snapshot")
        restored = 0
        if isinstance(snapshot, dict) and restore_ids:
            restored = self.store.restore_images_snapshot(snapshot, image_ids=sorted(restore_ids))
            self.vector_index.invalidate()
        backup_dir = undo.get("backup_dir")
        self._last_removal_undo = None
        if hasattr(self, "undo_removal_action"):
            self.undo_removal_action.setEnabled(False)
        if isinstance(backup_dir, str) and backup_dir:
            shutil.rmtree(backup_dir, ignore_errors=True)
        self._refresh_after_library_removal()
        if failures:
            QMessageBox.warning(
                self,
                "撤销不完整",
                f"已恢复索引 {restored} 个；失败 {len(failures)} 个。\n\n" + "\n".join(failures[:6]),
            )
        self._record_operation_history(f"撤销删除/移除：恢复 {restored} 个项目")
        self.statusBar().showMessage(f"已撤销：恢复 {restored} 个项目")

    def _refresh_after_library_removal(self) -> None:
        self._invalidate_near_duplicate_hash_cache()
        self._refresh_folders()
        self._refresh_collections()
        self._refresh_tags()
        self._refresh_current_results_for_filters()
        self._refresh_embedding_stats()
        self._refresh_temp_project_save_button()

    def _batch_rebuild_thumbnails(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        if not self._confirm_batch_operation(
            "重建缩略图",
            "重新生成选中项目的缩略图",
            images,
        ):
            return
        rebuilt = 0
        failed = 0
        for image in images:
            path = Path(image.file_path)
            if image.is_missing or not path.exists():
                failed += 1
                self.store.update_thumbnail(image.id, None, "failed")
                continue
            try:
                thumbnail_path = (
                    self.thumbnailer.generate_video(image.id, image.file_path)
                    if is_supported_video(image.file_path)
                    else self.thumbnailer.generate(image.id, image.file_path)
                )
                self.store.update_thumbnail(image.id, str(thumbnail_path), "ready")
                rebuilt += 1
            except Exception:
                failed += 1
                self.store.update_thumbnail(image.id, None, "failed")
        self._invalidate_near_duplicate_hash_cache()
        self._refresh_current_results_for_filters()
        self._record_operation_history(f"重建缩略图：成功 {rebuilt}，失败 {failed}")
        self.statusBar().showMessage(
            f"缩略图重建完成：成功 {rebuilt}，失败 {failed}"
        )

    def _set_file_watch_enabled(self, enabled: bool) -> None:
        self.file_watch_enabled = bool(enabled)
        self.store.set_setting("ui.file_watch_enabled", "1" if self.file_watch_enabled else "0")
        self._refresh_file_watcher()
        state = "已开启" if self.file_watch_enabled else "已关闭"
        self.statusBar().showMessage(f"自动监听文件变化{state}")

    def _refresh_file_watcher(self) -> None:
        if not hasattr(self, "file_watcher"):
            return
        watched = list(self.file_watcher.directories()) + list(self.file_watcher.files())
        if watched:
            self.file_watcher.removePaths(watched)
        self._watch_path_roots.clear()
        if self._database_maintenance_active:
            return
        removed_roots, removed_images = self._cleanup_missing_active_scan_roots()
        if removed_images:
            message = (
                f"已移除 {removed_roots} 个已不存在导入目录，"
                f"{removed_images} 张失效索引"
            )
            self._record_operation_history(message)
            self.statusBar().showMessage(message)
        if not self.file_watch_enabled:
            return
        paths_to_watch: list[str] = []
        for folder in self.store.list_folders_with_collection_images():
            root = self._normalize_folder_path(folder.folder_path)
            if not os.path.isdir(root):
                continue
            for directory in self._watched_directories_for_root(root):
                if directory in self._watch_path_roots:
                    continue
                paths_to_watch.append(directory)
                self._watch_path_roots[directory] = root
        if paths_to_watch:
            self.file_watcher.addPaths(paths_to_watch)
            self.statusBar().showMessage(f"已监听 {len(paths_to_watch)} 个本地目录")

    @staticmethod
    def _watched_directories_for_root(root: str, *, max_directories: int = 1200) -> list[str]:
        directories: list[str] = []
        for current_root, dirnames, _filenames in os.walk(root, topdown=True, followlinks=False):
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not dirname.startswith(".")
                and not os.path.islink(os.path.join(current_root, dirname))
            ]
            directories.append(os.path.abspath(current_root))
            if len(directories) >= max_directories:
                dirnames[:] = []
                break
        return directories

    def _handle_watched_path_changed(self, changed_path: str) -> None:
        if self._database_maintenance_active:
            return
        if not self.file_watch_enabled:
            return
        normalized = self._normalize_folder_path(changed_path)
        root = self._watch_path_roots.get(normalized)
        if root is None:
            for watched_path, watched_root in self._watch_path_roots.items():
                if self._path_is_in_folder(normalized, watched_path):
                    root = watched_root
                    break
        if root is None:
            return
        self._pending_watch_scan_roots.add(root)
        self.watch_scan_timer.start(1500)
        self.statusBar().showMessage("检测到本地文件变化，准备增量扫描")

    def _run_pending_watch_scans(self) -> None:
        if self._database_maintenance_active:
            self._pending_watch_scan_roots.clear()
            return
        if self._watch_scan_running:
            if self._pending_watch_scan_roots:
                self.watch_scan_timer.start(1500)
            return
        roots = sorted(self._pending_watch_scan_roots)
        self._pending_watch_scan_roots.clear()
        if not roots:
            return
        if not self.file_watch_enabled:
            return
        self._watch_scan_running = True

        def run() -> None:
            results: list[ScanResult] = []
            failures: list[str] = []
            for root in roots:
                try:
                    results.append(self.scanner.scan_folder(root))
                except Exception as exc:
                    failures.append(f"{root}: {exc}")
            self.events.put(("watch_scan_done", (roots, results, failures)))

        self._start_background_task(
            run,
            on_rejected=self._reset_watch_scan_running,
        )

    def _reset_watch_scan_running(self) -> None:
        self._watch_scan_running = False

    def _handle_watch_scan_done(self, payload: object) -> None:
        roots, results, failures = payload
        self._watch_scan_running = False
        self._delete_scan_removed_thumbnails(results)
        self._refresh_after_scan_database_change(preserve_current_view=True)
        scanned = sum(result.scanned_files for result in results)
        new_files = sum(result.new_files for result in results)
        changed_files = sum(result.changed_files for result in results)
        removed = sum(result.missing_marked for result in results)
        message = (
            f"自动扫描完成：目录 {len(roots)}，扫描 {scanned}，"
            f"新增 {new_files}，变化 {changed_files}，移除索引 {removed}，失败 {len(failures)}"
        )
        if failures:
            self._record_error("自动监听扫描失败：" + " | ".join(failures[:5]))
        self.settings_status_label.setText(message)
        self.statusBar().showMessage(message)
        if self._pending_watch_scan_roots and self.file_watch_enabled:
            self.watch_scan_timer.start(1500)

    def _refresh_after_scan_database_change(
        self,
        *,
        select_collection_id: int | None = None,
        preserve_current_view: bool = False,
    ) -> None:
        self._invalidate_near_duplicate_hash_cache()
        self._refresh_folders()
        self._refresh_collections(select_collection_id=select_collection_id)
        self.store.seed_default_ai_vision_collection_rules()
        if preserve_current_view:
            self._refresh_filter_chain_ui()
        elif self._has_visible_result_context() or self.search_filters:
            self._refresh_current_results_for_filters()
        else:
            self._reload_images()
        self._refresh_embedding_stats()
        self._refresh_ai_vision_stats()
        self._refresh_path_remap_candidates()
        self._refresh_file_watcher()

    def _folder_label_for_image(self, image: ImageItem) -> str:
        chains = self.store.collection_chains_for_image(image.id)
        if chains:
            labels = [" / ".join(collection.name for collection in chain) for chain in chains]
            return "；".join(labels)
        parent = Path(image.file_path).parent
        return parent.name or str(parent)

    def _show_missing_files_dialog(self) -> None:
        missing_count = self.store.count_missing_images()
        if missing_count <= 0:
            QMessageBox.information(self, "缺失文件修复", "当前没有缺失文件。")
            return
        images = self.store.list_images(
            status_filter="missing",
            include_missing=True,
            limit=max(1, missing_count),
            sort_key="name",
            sort_desc=False,
        )
        folder_labels = {image.id: self._folder_label_for_image(image) for image in images}
        dialog = MissingFilesDialog(images=images, folder_labels=folder_labels, parent=self)

        def refresh_dialog() -> None:
            refreshed_count = self.store.count_missing_images()
            refreshed = self.store.list_images(
                status_filter="missing",
                include_missing=True,
                limit=max(1, refreshed_count),
                sort_key="name",
                sort_desc=False,
            )
            labels = {image.id: self._folder_label_for_image(image) for image in refreshed}
            dialog.set_images(refreshed, labels)

        def relink_selected() -> None:
            image_id = dialog.current_image_id()
            image = self.store.get_image(image_id) if image_id is not None else None
            if image is None:
                return
            file_path, _selected_filter = QFileDialog.getOpenFileName(
                dialog,
                "重新指定源文件",
                str(Path(image.file_path).parent),
                "Media (*.jpg *.jpeg *.png *.webp *.mp4 *.mov *.m4v);;All Files (*)",
            )
            if not file_path:
                return
            try:
                self._repair_missing_image_path(image.id, Path(file_path))
            except Exception as exc:
                QMessageBox.warning(dialog, "修复失败", str(exc))
                return
            self._record_operation_history(f"重新定位缺失文件：{image.file_name} -> {file_path}")
            refresh_dialog()

        def remap_selected() -> None:
            image_id = dialog.current_image_id()
            image = self.store.get_image(image_id) if image_id is not None else None
            if image is None:
                return
            old_prefix = str(Path(image.file_path).parent)
            new_prefix = QFileDialog.getExistingDirectory(dialog, "选择移动后的新目录", str(Path.home()))
            if not new_prefix:
                return
            try:
                counts = self.store.path_prefix_match_counts(old_prefix)
            except Exception as exc:
                QMessageBox.warning(dialog, "检查失败", str(exc))
                return
            answer = QMessageBox.question(
                dialog,
                "批量重定位",
                (
                    f"旧目录：{old_prefix}\n新目录：{new_prefix}\n\n"
                    f"匹配 {counts['images']} 个文件记录，其中 {counts['missing']} 个缺失。继续？"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            result = self.store.remap_path_prefix(old_prefix, new_prefix)
            self.vector_index.invalidate()
            self._refresh_after_database_change()
            refresh_dialog()
            self._record_operation_history(
                f"批量路径重定位：{old_prefix} -> {new_prefix}，恢复 {result.get('relinked', 0)}"
            )
            self.statusBar().showMessage(
                f"批量重定位完成：恢复 {result.get('relinked', 0)}，仍丢失 {result.get('still_missing', 0)}"
            )

        def remove_selected() -> None:
            image_ids = dialog.selected_image_ids()
            if not image_ids:
                return
            images_to_remove = self.store.images_by_ids(image_ids)
            if not images_to_remove:
                return
            if not self._confirm_batch_operation("移除缺失索引", "从 Eidory 移除缺失文件索引", images_to_remove):
                return
            removed = self._remove_images_from_library_with_undo(
                images_to_remove,
                undo_label=f"移除缺失索引 {len(images_to_remove)} 个",
            )
            refresh_dialog()
            self._record_operation_history(f"移除缺失索引 {removed} 个，可 Cmd+Z 撤销")
            self.statusBar().showMessage(f"已移除缺失索引 {removed} 个。按 Cmd+Z 可撤销")

        dialog.relink_button.clicked.connect(relink_selected)
        dialog.remap_button.clicked.connect(remap_selected)
        dialog.remove_button.clicked.connect(remove_selected)
        dialog.exec()
        self._refresh_after_database_change()

    def _repair_missing_image_path(self, image_id: int, file_path: Path) -> None:
        path = Path(file_path).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError("请选择一个存在的普通媒体文件。")
        if not is_supported_media(str(path)):
            raise ValueError("文件格式不受支持。")
        stat = path.stat()
        width, height, duration_ms = self.scanner._read_media_metadata(str(path))
        self.store.repair_missing_image_path(
            image_id,
            file_path=str(path),
            file_size=stat.st_size,
            width=width,
            height=height,
            modified_time_ns=stat.st_mtime_ns,
            duration_ms=duration_ms,
        )
        try:
            thumbnail_path = (
                self.thumbnailer.generate_video(image_id, str(path))
                if is_supported_video(str(path))
                else self.thumbnailer.generate(image_id, str(path))
            )
            self.store.update_thumbnail(image_id, str(thumbnail_path), "ready")
        except Exception:
            self.store.update_thumbnail(image_id, None, "failed")
        if is_supported_image(str(path)):
            self.scanner._update_color_feature(image_id, str(path))
        else:
            self.store.mark_embedding_not_required(image_id)
        self.vector_index.invalidate()

    def _detect_duplicates(self) -> None:
        image_count = self.store.count_images()
        if image_count <= 1:
            QMessageBox.information(self, "重复检测", "图库里没有足够的图片可检测。")
            return
        self._set_maintenance_controls_enabled(False)
        self.settings_status_label.setText("正在检测重复/近重复图片...")
        self.statusBar().showMessage("正在检测重复/近重复图片")

        def run() -> None:
            try:
                images = self.store.list_images(
                    include_missing=False,
                    limit=max(1, image_count),
                    sort_key="name",
                    sort_desc=False,
                )
                folder_labels = {image.id: self._folder_label_for_image(image) for image in images}
                groups = find_duplicate_groups(
                    images,
                    folder_label_for_image=folder_labels,
                    near_distance=8,
                )
                self.events.put(("duplicates_done", groups))
            except Exception as exc:
                self.events.put(("error", f"重复检测失败：{exc}"))

        self._start_background_task(
            run,
            on_rejected=lambda: self._set_maintenance_controls_enabled(True),
        )

    def _handle_duplicates_done(self, groups: list[DuplicateGroup]) -> None:
        self._set_maintenance_controls_enabled(True)
        if not groups:
            self.settings_status_label.setText("没有发现重复/近重复图片。")
            self.statusBar().showMessage("没有发现重复/近重复图片")
            return
        exact_count = sum(1 for group in groups if group.kind == "exact")
        near_count = len(groups) - exact_count
        dialog = DuplicateResultsDialog(groups, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_group_image_ids:
            images = self.store.images_by_ids(dialog.selected_group_image_ids)
            self.current_duplicate_images = images
            self.current_result_mode = "duplicate_group"
            self._clear_result_management_state()
            self.search_filters.clear()
            self.active_filter_index = None
            self.current_offset = len(images)
            self.load_more_button.setEnabled(False)
            self.grid_view.set_images(images)
            self._set_result_status(f"重复候选组：{len(images)} 张")
            self.search_diagnostics_label.setText("搜索诊断：重复/近重复候选组")
        message = f"重复检测完成：完全重复 {exact_count} 组，近重复 {near_count} 组"
        self.settings_status_label.setText(message)
        self.statusBar().showMessage(message)

    def _compare_selected_images(self) -> None:
        images = self._selected_grid_images()
        if len(images) < 2:
            self.statusBar().showMessage("至少选择 2 张图片才能对比")
            return
        ImageCompareDialog(images, parent=self).exec()

    def _delete_thumbnail_files(self, thumbnail_paths: list[str]) -> None:
        thumbnail_root = self.paths.thumbnail_dir.resolve()
        for thumbnail_path in thumbnail_paths:
            path = Path(thumbnail_path)
            try:
                resolved = path.resolve()
                if resolved.is_relative_to(thumbnail_root):
                    resolved.unlink(missing_ok=True)
            except Exception:
                continue

    def _delete_scan_removed_thumbnails(self, results: list[ScanResult]) -> None:
        thumbnail_paths = [
            thumbnail_path
            for result in results
            for thumbnail_path in result.removed_thumbnail_paths
        ]
        self._delete_thumbnail_files(thumbnail_paths)

    def _cleanup_unclassified_active_roots(self) -> tuple[int, int]:
        thumbnail_paths, removed_roots, removed_images = self.store.remove_unclassified_active_roots()
        if removed_images:
            self._delete_thumbnail_files(thumbnail_paths)
            self.vector_index.invalidate()
        return removed_roots, removed_images

    def _cleanup_missing_active_scan_roots(self) -> tuple[int, int]:
        removed_roots = 0
        removed_images = 0
        thumbnail_paths: list[str] = []
        for folder in self.store.list_folders_with_collection_images():
            if not folder.last_scanned_at:
                continue
            root = self._normalize_folder_path(folder.folder_path)
            if os.path.isdir(root):
                continue
            folder_thumbnails, folder_removed = self.store.remove_folder_from_library(folder.id)
            thumbnail_paths.extend(folder_thumbnails)
            removed_roots += 1
            removed_images += folder_removed
        if removed_images:
            self._delete_thumbnail_files(thumbnail_paths)
            self.vector_index.invalidate()
        return removed_roots, removed_images

    def _start_embedding(self) -> None:
        if self._database_maintenance_active:
            self.statusBar().showMessage("数据库维护中，暂不启动语义索引")
            return
        if self.embedding_worker is not None and self.embedding_worker.is_alive():
            self.embedding_worker.resume_work()
            self._refresh_embedding_stats()
            return
        self.embedding_worker = EmbeddingWorker(
            store=self.store,
            provider=self.embedding_provider,
            vector_index=self.vector_index,
            on_progress=lambda progress: self.events.put(("embedding", progress)),
        )
        self.embedding_worker.start()
        self._refresh_embedding_stats()

    def _pause_embedding(self) -> None:
        if self.embedding_worker is not None:
            self.embedding_worker.pause()

    def _retry_failed_embeddings(self) -> None:
        count = self.store.retry_failed_embeddings(
            model_name=self.embedding_provider.model_name,
            model_revision=self.embedding_provider.model_revision,
            embedding_dim=self.embedding_provider.dim,
        )
        self.statusBar().showMessage(f"已重试 {count} 个失败项")
        self._refresh_embedding_stats()
        self._start_embedding()

    def _start_ai_vision(self) -> None:
        if self._database_maintenance_active:
            self.statusBar().showMessage("数据库维护中，暂不启动 AI 识别")
            return
        if self.ai_vision_worker is not None and self.ai_vision_worker.is_alive():
            self.ai_vision_worker.resume_work()
            self._refresh_ai_vision_stats()
            return
        provider = self._make_ai_vision_provider()
        self.ai_vision_worker = AIVisionWorker(
            store=self.store,
            provider=provider,
            on_progress=lambda progress: self.events.put(("ai_vision", progress)),
        )
        self.ai_vision_worker.start()
        self._refresh_ai_vision_stats()

    def _pause_ai_vision(self) -> None:
        if self.ai_vision_worker is not None:
            self.ai_vision_worker.pause()

    def _retry_failed_ai_vision(self) -> None:
        count = self.store.retry_failed_ai_vision()
        self.statusBar().showMessage(f"已重试 {count} 个 AI 识别失败项")
        self._refresh_ai_vision_stats()
        if count:
            self._start_ai_vision()

    def _set_ai_vision_rule_for_selected_collection(self, mode: str) -> None:
        collection_id = self._selected_collection_id()
        if collection_id is None:
            self.statusBar().showMessage("请先在左侧选择一个普通文件夹")
            return
        self.store.set_ai_vision_collection_rule(collection_id, mode=mode, include_descendants=True)
        self._refresh_ai_vision_stats()
        label = "识别" if mode == "include" else "排除"
        self.statusBar().showMessage(f"已设置 AI 场景标签规则：{label} {self._collection_path_text(collection_id)}")

    def _remove_selected_ai_vision_rule(self) -> None:
        item = self.ai_vision_rule_tree.currentItem()
        if item is None:
            self.statusBar().showMessage("请先选择一条 AI 规则")
            return
        collection_id = item.data(0, Qt.ItemDataRole.UserRole)
        if collection_id is None:
            return
        removed = self.store.remove_ai_vision_collection_rule(int(collection_id))
        self._refresh_ai_vision_stats()
        self.statusBar().showMessage("已移除 AI 规则" if removed else "AI 规则不存在")

    def _handle_ai_vision_progress(self, progress: AIVisionProgress) -> None:
        self._refresh_ai_vision_stats()
        if progress.status in {"ready", "failed", "stale", "pending"}:
            self._refresh_virtual_collection_counts()
        if progress.image_id is None:
            self.statusBar().showMessage(progress.message)
        else:
            self.statusBar().showMessage(f"{progress.file_name}: {progress.status}")
            if self.selected_image is not None and self.selected_image.id == progress.image_id:
                image = self.store.get_image(progress.image_id)
                if image is not None:
                    self.selected_image = image
                    self._show_image_details(image)

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "scan_done":
                self._handle_scan_done(payload)
            elif kind == "scan_all_done":
                self._handle_scan_all_done(payload)
            elif kind == "scan_new_done":
                self._handle_scan_new_done(payload)
            elif kind == "scan_missing_done":
                self._handle_scan_missing_done(payload)
            elif kind == "watch_scan_done":
                self._handle_watch_scan_done(payload)
            elif kind == "duplicates_done":
                self._handle_duplicates_done(payload)
            elif kind == "import_done":
                result, collection_id, collection_name, assigned, preserve_structure = payload
                self._handle_import_done(
                    result,
                    collection_id=collection_id,
                    collection_name=collection_name,
                    assigned=assigned,
                    preserve_structure=preserve_structure,
                )
            elif kind == "drop_import_done":
                imported_image_ids: list[int] = []
                if len(payload) == 7:
                    (
                        collection_id,
                        collection_name,
                        scanned,
                        new_files,
                        changed_files,
                        assigned,
                        imported_image_ids,
                    ) = payload
                else:
                    collection_id, collection_name, scanned, new_files, changed_files, assigned = payload
                self._handle_drop_import_done(
                    collection_id=collection_id,
                    collection_name=collection_name,
                    scanned=scanned,
                    new_files=new_files,
                    changed_files=changed_files,
                    assigned=assigned,
                    imported_image_ids=list(imported_image_ids),
                )
            elif kind == "drop_payload_materialized":
                collection_id, collection_name, saved_paths = payload
                self._handle_drop_payload_materialized(
                    collection_id=collection_id,
                    collection_name=collection_name,
                    saved_paths=list(saved_paths),
                )
            elif kind == "near_duplicate_candidates_ready":
                self._handle_near_duplicate_candidates_ready(payload)
            elif kind == "near_duplicate_candidates_failed":
                self._handle_near_duplicate_candidates_failed(payload)
            elif kind == "search_done":
                self.search_button.setEnabled(True)
                revision, result = payload
                if revision != self.semantic_search_revision:
                    continue
                self.current_semantic_images = list(result.images)
                self.current_semantic_searchable_count = result.searchable_count
                self.current_semantic_candidate_limit = result.candidate_limit
                self._recompute_result_exclusion_filter_matches_for_current_source()
                self._apply_semantic_result_filters()
                images = self.current_semantic_filtered_images
                self.grid_view.set_images(images)
                self._set_semantic_result_status(images)
                self._update_search_diagnostics(images)
            elif kind == "color_search_done":
                self.search_button.setEnabled(True)
                revision, result = payload
                if revision != self.semantic_search_revision:
                    continue
                self.current_color_images = list(result.images)
                self.current_color_searchable_count = result.searchable_count
                self.current_color_indexed_count = result.indexed_count
                self.current_color_candidate_limit = result.candidate_limit
                self._recompute_result_exclusion_filter_matches_for_current_source()
                self._apply_color_result_filters()
                images = self.current_color_filtered_images
                self.grid_view.set_images(images)
                self._set_color_result_status(images)
                self._update_color_search_diagnostics(images)
            elif kind == "search_chain_done":
                self.search_button.setEnabled(True)
                revision, filters, result = payload
                if revision != self.semantic_search_revision:
                    continue
                self._handle_search_chain_done(filters=filters, result=result)
            elif kind == "result_exclusion_filter_done":
                self.search_button.setEnabled(True)
                revision, search_filter, matches = payload
                if revision != self.semantic_search_revision:
                    continue
                self._add_result_exclusion_filter_from_matches(search_filter, matches)
            elif kind == "inspiration_proposal":
                self.generate_inspiration_button.setEnabled(True)
                self._show_inspiration_proposal(payload)
            elif kind == "inspiration_error":
                self.generate_inspiration_button.setEnabled(True)
                self.search_inspiration_button.setEnabled(bool(self._selected_inspiration_terms()))
                self._show_inspiration_error(payload)
            elif kind == "search_plan_proposal":
                self.generate_inspiration_button.setEnabled(True)
                self._show_search_plan_proposal(payload)
            elif kind == "search_plan_error":
                self.generate_inspiration_button.setEnabled(True)
                self.search_inspiration_button.setEnabled(bool(self._selected_inspiration_terms()))
                self._show_inspiration_error(payload)
            elif kind == "inspiration_done":
                self.generate_inspiration_button.setEnabled(True)
                self.search_inspiration_button.setEnabled(bool(self._selected_inspiration_terms()))
                plan_filters = None
                raw_term_results = None
                result = None
                if isinstance(payload, tuple) and len(payload) == 5:
                    revision, project_id, selected_terms, raw_term_results, plan_filters = payload
                else:
                    revision, project_id, selected_terms, result = payload
                if revision != self.semantic_search_revision:
                    continue
                self._handle_inspiration_done(
                    project_id=project_id,
                    selected_terms=selected_terms,
                    result=result,
                    raw_term_results=raw_term_results,
                    plan_filters=plan_filters,
                )
            elif kind == "creative_node_note_done":
                node_id, suggestion, model_name = payload
                self._handle_creative_node_note_done(node_id, suggestion, model_name)
            elif kind == "creative_node_note_error":
                node_id, exc = payload
                self._handle_creative_node_note_error(node_id, exc)
            elif kind == "creative_project_copy_done":
                project_id, selected_node_id, fill_empty_only, suggestion, model_name = payload
                self._handle_creative_project_copy_done(
                    int(project_id),
                    int(selected_node_id) if selected_node_id is not None else None,
                    bool(fill_empty_only),
                    suggestion,
                    str(model_name),
                )
            elif kind == "creative_project_copy_error":
                project_id, exc = payload
                self._handle_creative_project_copy_error(int(project_id), exc)
            elif kind == "creative_node_search_done":
                revision, node_id, query, result = payload
                self._handle_creative_node_search_done(revision, node_id, query, result)
            elif kind == "creative_node_search_error":
                revision, exc = payload
                if revision == self.semantic_search_revision:
                    self.search_creative_node_button.setEnabled(self.current_creative_node_id is not None)
                    QMessageBox.warning(self, "Eidory", f"创作节点搜索失败：{exc}")
            elif kind == "temp_project_suggestion":
                self._apply_temporary_project_suggestion(payload)
            elif kind == "temp_project_suggestion_error":
                self._show_temporary_project_suggestion_error(payload)
            elif kind == "reference_groups_done":
                self._create_reference_group_projects(payload, confirm=True)
            elif kind == "self_check_done":
                self._handle_self_check_done(payload)
            elif kind == "path_remap_done":
                self._handle_path_remap_done(payload)
            elif kind == "performance_done":
                self._handle_performance_done(payload)
            elif kind == "export_done":
                self._handle_export_done(payload)
            elif kind == "folder_tree_import_done":
                (
                    results,
                    parent_collection_id,
                    parent_name,
                    assigned,
                    imported_image_ids,
                    folder_paths,
                ) = payload
                self._handle_folder_tree_import_done(
                    results=results,
                    parent_collection_id=parent_collection_id,
                    parent_name=parent_name,
                    assigned=assigned,
                    imported_image_ids=imported_image_ids,
                    folder_paths=folder_paths,
                )
            elif kind == "local_paths_import_done":
                (
                    results,
                    parent_collection_id,
                    target_name,
                    assigned,
                    imported_image_ids,
                    file_paths,
                    folder_paths,
                ) = payload
                self._handle_local_paths_import_done(
                    results=results,
                    parent_collection_id=parent_collection_id,
                    target_name=target_name,
                    assigned=assigned,
                    imported_image_ids=imported_image_ids,
                    file_paths=file_paths,
                    folder_paths=folder_paths,
                )
            elif kind == "embedding":
                self._handle_embedding_progress(payload)
            elif kind == "ai_vision":
                self._handle_ai_vision_progress(payload)
            elif kind == "error":
                self._record_error(str(payload))
                self.search_button.setEnabled(True)
                self.generate_inspiration_button.setEnabled(True)
                self.search_inspiration_button.setEnabled(bool(self._selected_inspiration_terms()))
                self._set_import_controls_enabled(True)
                if hasattr(self, "start_ai_vision_button"):
                    self.start_ai_vision_button.setEnabled(True)
                    self.pause_ai_vision_button.setEnabled(True)
                    self.retry_failed_ai_vision_button.setEnabled(True)
                self._set_maintenance_controls_enabled(True)
                if hasattr(self, "export_library_button"):
                    self._set_export_controls_enabled(True)
                QMessageBox.critical(self, "Eidory", str(payload))

    def _handle_scan_done(self, result: ScanResult) -> None:
        self._set_import_controls_enabled(True)
        self._set_maintenance_controls_enabled(True)
        self._delete_scan_removed_thumbnails([result])
        self._refresh_after_scan_database_change()
        self.statusBar().showMessage(
            f"扫描完成：新增 {result.new_files}，变化 {result.changed_files}，移除索引 {result.missing_marked}"
        )

    def _handle_scan_all_done(self, results: list[ScanResult]) -> None:
        self._set_import_controls_enabled(True)
        self._set_maintenance_controls_enabled(True)
        self._delete_scan_removed_thumbnails(results)
        self._refresh_after_scan_database_change()
        scanned = sum(result.scanned_files for result in results)
        new_files = sum(result.new_files for result in results)
        changed_files = sum(result.changed_files for result in results)
        removed = sum(result.missing_marked for result in results)
        message = (
            f"全部重新扫描完成：目录 {len(results)}，扫描 {scanned}，"
            f"新增 {new_files}，变化 {changed_files}，移除索引 {removed}"
        )
        self.settings_status_label.setText(message)
        self.statusBar().showMessage(message)

    def _handle_scan_new_done(self, results: list[ScanResult]) -> None:
        self._set_import_controls_enabled(True)
        self._set_maintenance_controls_enabled(True)
        self._delete_scan_removed_thumbnails(results)
        self._refresh_after_scan_database_change()
        scanned = sum(result.scanned_files for result in results)
        new_files = sum(result.new_files for result in results)
        changed_files = sum(result.changed_files for result in results)
        message = (
            f"扫描新增/变化完成：目录 {len(results)}，扫描 {scanned}，"
            f"新增 {new_files}，变化 {changed_files}，未处理删除"
        )
        self.settings_status_label.setText(message)
        self.statusBar().showMessage(message)

    def _handle_scan_missing_done(self, results: list[ScanResult]) -> None:
        self._set_import_controls_enabled(True)
        self._set_maintenance_controls_enabled(True)
        self._delete_scan_removed_thumbnails(results)
        self._refresh_after_scan_database_change()
        scanned = sum(result.scanned_files for result in results)
        recovered = sum(result.changed_files for result in results)
        removed = sum(result.missing_marked for result in results)
        message = (
            f"扫描缺失所在目录完成：目录 {len(results)}，扫描 {scanned}，"
            f"恢复/变化 {recovered}，移除索引 {removed}"
        )
        self.settings_status_label.setText(message)
        self.statusBar().showMessage(message)

    def _handle_performance_done(self, report: str) -> None:
        self.run_performance_check_button.setEnabled(True)
        self.settings_status_label.setText(report)
        self.statusBar().showMessage("性能压测完成")

    def _handle_import_done(
        self,
        result: ScanResult,
        *,
        collection_id: int | None,
        collection_name: str,
        assigned: int,
        preserve_structure: bool,
    ) -> None:
        self._set_import_controls_enabled(True)
        self._refresh_after_scan_database_change(select_collection_id=collection_id)
        mode = "按目录结构导入" if preserve_structure else "导入"
        self.statusBar().showMessage(
            f"{mode}完成：{collection_name}，扫描 {result.scanned_files}，"
            f"新增 {result.new_files}，变化 {result.changed_files}，加入 {assigned}"
        )

    def _handle_folder_tree_import_done(
        self,
        *,
        results: list[ScanResult],
        parent_collection_id: int | None,
        parent_name: str,
        assigned: int,
        imported_image_ids: list[int],
        folder_paths: list[str],
    ) -> None:
        self._set_import_controls_enabled(True)
        self._delete_scan_removed_thumbnails(results)
        self._refresh_after_scan_database_change(select_collection_id=parent_collection_id)
        self.store.seed_default_ai_vision_collection_rules()
        clean_imported_ids = [
            int(image_id)
            for image_id in imported_image_ids
            if int(image_id) > 0
        ]
        if clean_imported_ids:
            self._show_imported_images_first(parent_collection_id, clean_imported_ids)
        scanned = sum(result.scanned_files for result in results)
        new_files = sum(result.new_files for result in results)
        changed_files = sum(result.changed_files for result in results)
        removed = sum(result.missing_marked for result in results)
        message = (
            f"导入文件夹完成：目标 {parent_name}，文件夹 {len(folder_paths)}，"
            f"扫描 {scanned}，新增 {new_files}，变化 {changed_files}，"
            f"移除索引 {removed}，加入 {assigned}"
        )
        self._record_operation_history(message)
        self.statusBar().showMessage(message)

    def _handle_local_paths_import_done(
        self,
        *,
        results: list[ScanResult],
        parent_collection_id: int | None,
        target_name: str,
        assigned: int,
        imported_image_ids: list[int],
        file_paths: list[str],
        folder_paths: list[str],
    ) -> None:
        self._set_import_controls_enabled(True)
        self._delete_scan_removed_thumbnails(results)
        self._refresh_after_scan_database_change(select_collection_id=parent_collection_id)
        self.store.seed_default_ai_vision_collection_rules()
        clean_imported_ids = [
            int(image_id)
            for image_id in imported_image_ids
            if int(image_id) > 0
        ]
        if clean_imported_ids:
            self._show_imported_images_first(parent_collection_id, clean_imported_ids)
        scanned = sum(result.scanned_files for result in results)
        new_files = sum(result.new_files for result in results)
        changed_files = sum(result.changed_files for result in results)
        removed = sum(result.missing_marked for result in results)
        message = (
            f"拖入导入完成：目标 {target_name}，图片 {len(file_paths)}，"
            f"文件夹 {len(folder_paths)}，扫描 {scanned}，新增 {new_files}，"
            f"变化 {changed_files}，移除索引 {removed}，加入 {assigned}"
        )
        self._record_operation_history(message)
        self.statusBar().showMessage(message)

    def _handle_drop_import_done(
        self,
        *,
        collection_id: int,
        collection_name: str,
        scanned: int,
        new_files: int,
        changed_files: int,
        assigned: int,
        imported_image_ids: list[int] | None = None,
    ) -> None:
        self._set_import_controls_enabled(True)
        self._refresh_folders()
        self._refresh_collections(select_collection_id=collection_id)
        self.store.seed_default_ai_vision_collection_rules()
        clean_imported_ids = [
            int(image_id)
            for image_id in imported_image_ids or []
            if int(image_id) > 0
        ]
        pending_node_id = self._pending_board_import_node_id
        self._pending_board_import_node_id = None
        imported_to_board = False
        if pending_node_id is not None and clean_imported_ids:
            node = self.store.get_creative_node(pending_node_id)
            if node is not None:
                self.store.add_images_to_creative_node(
                    node.id,
                    clean_imported_ids,
                    intent_label=node.title,
                    intent_query=node.search_query or node.title,
                )
                self._expand_project_sidebar_section("creative")
                self._refresh_creative_projects(select_project_id=node.project_id)
                imported_to_board = True
                if self.center_result_stack.currentWidget() is self.project_board_view:
                    self._show_current_creative_board()
        if clean_imported_ids and not imported_to_board:
            self._show_imported_images_first(collection_id, clean_imported_ids)
        else:
            self._refresh_after_scan_database_change(select_collection_id=collection_id)
        if clean_imported_ids:
            self._refresh_embedding_stats()
            self._refresh_ai_vision_stats()
            self._refresh_file_watcher()
        self.statusBar().showMessage(
            f"拖入导入成功：{collection_name}，扫描 {scanned}，"
            f"新增 {new_files}，变化 {changed_files}，加入 {assigned}"
        )

    def _show_imported_images_first(
        self,
        collection_id: int | None,
        imported_image_ids: list[int],
    ) -> None:
        imported_images = self.store.images_by_ids(imported_image_ids)
        imported_id_set = {image.id for image in imported_images}
        remaining_images = [
            image
            for image in self.store.list_images(
                limit=self.page_size,
                collection_id=collection_id,
                sort_key=self._database_sort_key(),
                sort_desc=self.current_sort_desc,
            )
            if image.id not in imported_id_set
        ]
        images = imported_images + remaining_images
        self.semantic_search_revision += 1
        self._clear_manual_result_order()
        self._clear_result_management_state()
        self.search_filters.clear()
        self.active_filter_index = None
        self.current_keyword_query = None
        self.current_semantic_query = None
        self.current_result_mode = "library"
        self.current_offset = len(images)
        self.load_more_button.setEnabled(len(remaining_images) >= self.page_size)
        self._refresh_filter_chain_ui()
        selected_ids = [imported_images[0].id] if imported_images else []
        current_id = selected_ids[0] if selected_ids else None
        self.grid_view.set_images(
            images,
            selected_image_ids=selected_ids,
            current_image_id=current_id,
        )
        self._set_result_status(
            f"已导入 {len(imported_images)} 张到“{self._collection_path_text(collection_id)}”"
        )
        self.search_diagnostics_label.setText("搜索诊断：-")

    def _handle_embedding_progress(self, progress: EmbeddingProgress) -> None:
        self._refresh_embedding_stats()
        if progress.image_id is None:
            self.statusBar().showMessage(progress.message)
        else:
            self.statusBar().showMessage(f"{progress.file_name}: {progress.status}")

    def _refresh_folders(self) -> None:
        current_prefix = self._selected_folder_path_prefix()
        self.folder_tree.blockSignals(True)
        self.folder_tree.clear()

        selected_item: QTreeWidgetItem | None = None
        all_item = QTreeWidgetItem(["全部文件夹", ""])
        all_item.setData(0, Qt.ItemDataRole.UserRole, None)
        all_item.setData(0, Qt.ItemDataRole.UserRole + 1, None)
        all_item.setData(0, Qt.ItemDataRole.UserRole + 2, None)
        self.folder_tree.addTopLevelItem(all_item)
        if current_prefix is None:
            selected_item = all_item

        for folder, counts in self.store.folder_subtree_counts():
            root_path = self._normalize_folder_path(folder.folder_path)
            root_count = counts.get(root_path, 0)
            root_item = self._make_folder_tree_item(
                label=Path(root_path).name or root_path,
                count=root_count,
                scan_path=root_path,
                folder_id=folder.id,
                filter_path=root_path,
            )
            root_item.setToolTip(0, root_path)
            root_item.setExpanded(True)
            self.folder_tree.addTopLevelItem(root_item)
            path_items = {root_path: root_item}
            if current_prefix == root_path:
                selected_item = root_item

            for path in sorted(
                (path for path in counts if path != root_path),
                key=lambda value: (value.count(os.sep), value.casefold()),
            ):
                parent_path = self._normalize_folder_path(os.path.dirname(path))
                parent_item = path_items.get(parent_path, root_item)
                item = self._make_folder_tree_item(
                    label=Path(path).name or path,
                    count=counts[path],
                    scan_path=root_path,
                    folder_id=folder.id,
                    filter_path=path,
                )
                item.setToolTip(0, path)
                parent_item.addChild(item)
                path_items[path] = item
                if current_prefix == path:
                    selected_item = item

        self.folder_tree.setCurrentItem(selected_item or all_item)
        self._expand_folder_tree_parents(self.folder_tree.currentItem())
        self.folder_tree.blockSignals(False)

    def _refresh_collections(self, select_collection_id: int | None = None) -> None:
        current = select_collection_id
        current_virtual_filter = self.current_virtual_filter
        if current is None:
            current = self._selected_collection_id()
        else:
            current_virtual_filter = None
            self.current_virtual_filter = None
        expanded_ids = self._expanded_collection_ids(self.collection_tree)
        self.collection_tree.blockSignals(True)
        self.collection_tree.clear()

        selected_item: QTreeWidgetItem | None = None
        all_item = QTreeWidgetItem(["全部文件夹", ""])
        all_item.setData(0, Qt.ItemDataRole.UserRole, None)
        all_item.setData(0, Qt.ItemDataRole.UserRole + 1, None)
        self._apply_collection_tree_level_style(all_item, -1)
        all_item.setFlags(all_item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled & ~Qt.ItemFlag.ItemIsDropEnabled)
        self.collection_tree.addTopLevelItem(all_item)
        if current is None and current_virtual_filter is None:
            selected_item = all_item

        collections_with_counts = self.store.list_collections_with_counts()
        children_by_parent: dict[int | None, list[tuple[object, int]]] = {}
        for collection, count in collections_with_counts:
            children_by_parent.setdefault(collection.parent_id, []).append((collection, count))

        def add_children(parent_item: QTreeWidgetItem | None, parent_id: int | None, depth: int) -> None:
            for collection, count in children_by_parent.get(parent_id, []):
                item = self._make_collection_tree_item(
                    collection_id=collection.id,
                    name=collection.name,
                    count=count,
                    depth=depth,
                )
                if parent_item is None:
                    self.collection_tree.addTopLevelItem(item)
                else:
                    parent_item.addChild(item)
                item.setExpanded(collection.id in expanded_ids)
                if current == collection.id:
                    nonlocal_selected[0] = item
                add_children(item, collection.id, depth + 1)

        nonlocal_selected: list[QTreeWidgetItem | None] = [selected_item]
        add_children(None, None, 0)
        selected_item = nonlocal_selected[0]

        counts = self.store.virtual_image_filter_counts()
        uncategorized_item = self._make_virtual_collection_tree_item(
            "uncategorized",
            counts.get("uncategorized", 0),
        )
        self.collection_tree.addTopLevelItem(uncategorized_item)
        if current_virtual_filter == "uncategorized":
            selected_item = uncategorized_item

        if selected_item is not None or current_virtual_filter is None:
            self.collection_tree.setCurrentItem(selected_item or all_item)
        else:
            self.collection_tree.clearSelection()
            self.collection_tree.setCurrentItem(None)
        if select_collection_id is not None:
            self._expand_folder_tree_parents(self.collection_tree.currentItem())
        self.collection_tree.blockSignals(False)
        self._refresh_virtual_collection_filters(select_virtual_filter=current_virtual_filter)
        if hasattr(self, "collection_detail_widget") and self.selected_image is None:
            self._show_collection_details(self._selected_collection_id())

    def _refresh_virtual_collection_filters(self, select_virtual_filter: str | None = None) -> None:
        if select_virtual_filter is not None:
            self.current_virtual_filter = select_virtual_filter
        counts = self.store.virtual_image_filter_counts()
        self._refresh_collection_virtual_filter_entry(counts)
        self._refresh_tag_virtual_filter_entry(counts)
        self._refresh_ai_vision_virtual_filter_entry(counts)

    def _refresh_virtual_collection_counts(self) -> None:
        counts = self.store.virtual_image_filter_counts()
        self._refresh_collection_virtual_filter_entry(counts)
        self._refresh_tag_virtual_filter_entry(counts)
        self._refresh_ai_vision_virtual_filter_entry(counts)
        if self.selected_image is None and self._selected_virtual_filter() is not None:
            self._show_collection_details(None)

    def _make_virtual_collection_tree_item(self, virtual_filter: str, count: int) -> QTreeWidgetItem:
        label = self._virtual_filter_label(virtual_filter)
        item = QTreeWidgetItem([label, str(count)])
        item.setData(0, Qt.ItemDataRole.UserRole, None)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, None)
        item.setData(0, COLLECTION_VIRTUAL_FILTER_ROLE, virtual_filter)
        item.setToolTip(0, self._virtual_filter_help(virtual_filter))
        item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        for column in range(item.columnCount()):
            item.setBackground(column, QBrush(QColor("#343b44")))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled & ~Qt.ItemFlag.ItemIsDropEnabled)
        return item

    def _refresh_collection_virtual_filter_entry(self, counts: Mapping[str, int]) -> None:
        if not hasattr(self, "collection_tree"):
            return
        previous = self.collection_tree.blockSignals(True)
        try:
            for index in range(self.collection_tree.topLevelItemCount()):
                item = self.collection_tree.topLevelItem(index)
                virtual_filter = item.data(0, COLLECTION_VIRTUAL_FILTER_ROLE)
                if virtual_filter:
                    item.setText(1, str(counts.get(str(virtual_filter), 0)))
                    if self.current_virtual_filter == str(virtual_filter):
                        self.collection_tree.setCurrentItem(item)
                    elif self.collection_tree.currentItem() is item:
                        self.collection_tree.clearSelection()
                        self.collection_tree.setCurrentItem(None)
        finally:
            self.collection_tree.blockSignals(previous)

    def _refresh_tag_virtual_filter_entry(self, counts: Mapping[str, int]) -> None:
        if not hasattr(self, "tag_list"):
            return
        previous = self.tag_list.blockSignals(True)
        try:
            for index in range(self.tag_list.count()):
                item = self.tag_list.item(index)
                if item.data(COLLECTION_VIRTUAL_FILTER_ROLE) == "untagged":
                    item.setText(f"未标签    {counts.get('untagged', 0)}")
                    item.setSelected(self.current_virtual_filter == "untagged")
                    if self.current_virtual_filter == "untagged":
                        self.tag_list.setCurrentItem(item)
                    return
        finally:
            self.tag_list.blockSignals(previous)

    def _refresh_ai_vision_virtual_filter_entry(self, counts: Mapping[str, int]) -> None:
        if not hasattr(self, "ai_vision_virtual_filter_list"):
            return
        previous = self.ai_vision_virtual_filter_list.blockSignals(True)
        try:
            self.ai_vision_virtual_filter_list.clear()
            item = QListWidgetItem(f"未AI标签    {counts.get('un_ai_tagged', 0)}")
            item.setData(COLLECTION_VIRTUAL_FILTER_ROLE, "un_ai_tagged")
            item.setToolTip(self._virtual_filter_help("un_ai_tagged"))
            self.ai_vision_virtual_filter_list.addItem(item)
            if self.current_virtual_filter == "un_ai_tagged":
                item.setSelected(True)
                self.ai_vision_virtual_filter_list.setCurrentItem(item)
        finally:
            self.ai_vision_virtual_filter_list.blockSignals(previous)

    def _refresh_creative_projects(self, select_project_id: int | None = None) -> None:
        if not hasattr(self, "creative_project_combo"):
            return
        projects = self.store.list_creative_projects()
        target_id = select_project_id if select_project_id is not None else self.current_creative_project_id
        if target_id is None and projects:
            target_id = projects[0].id
        self._refreshing_creative_projects = True
        self.creative_project_combo.blockSignals(True)
        self.creative_project_list.blockSignals(True)
        self.creative_project_combo.clear()
        self.creative_project_combo.addItem("未选择创作项目", None)
        self.creative_project_list.clear()
        selected_index = 0
        selected_item: QListWidgetItem | None = None
        for project in projects:
            label = f"{'⬆ ' if project.is_pinned else ''}{project.title} ({project.node_count}/{project.image_count})"
            self.creative_project_combo.addItem(label, project.id)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, project.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, project.is_pinned)
            item.setToolTip(project.brief or project.title)
            self.creative_project_list.addItem(item)
            if project.id == target_id:
                selected_index = self.creative_project_combo.count() - 1
                selected_item = item
        self.creative_project_combo.setCurrentIndex(selected_index)
        if selected_item is not None:
            self.creative_project_list.setCurrentItem(selected_item)
        else:
            self.creative_project_list.clearSelection()
        self.creative_project_combo.blockSignals(False)
        self.creative_project_list.blockSignals(False)
        self._refreshing_creative_projects = False
        self._refresh_project_sidebar(
            select_kind="creative" if selected_index > 0 else None,
            select_id=target_id if selected_index > 0 else None,
        )
        self._load_creative_project(target_id if selected_index > 0 else None)

    def _show_creative_project_context_menu(self, pos) -> None:
        item = self.creative_project_list.itemAt(pos)
        if item is None:
            return
        self.creative_project_list.setCurrentItem(item)
        project_id = item.data(Qt.ItemDataRole.UserRole)
        if project_id is None:
            return
        project = self.store.get_creative_project(int(project_id))
        if project is None:
            return
        menu = QMenu(self)
        pin_action = menu.addAction("取消置顶项目" if project.is_pinned else "置顶项目")
        delete_action = menu.addAction("删除创作项目")
        chosen = menu.exec(self.creative_project_list.viewport().mapToGlobal(pos))
        if chosen == pin_action:
            self.store.set_creative_project_pinned(project.id, not project.is_pinned)
            self._refresh_creative_projects(select_project_id=project.id)
            self.statusBar().showMessage("已置顶项目" if not project.is_pinned else "已取消置顶项目")
        elif chosen == delete_action:
            self._delete_creative_project(project.id)

    def _delete_creative_project(self, project_id: int) -> None:
        project = self.store.get_creative_project(project_id)
        if project is None:
            return
        answer = QMessageBox.question(
            self,
            "删除创作项目",
            f"删除创作项目“{project.title}”？\n这只删除项目、节点和图片链接，不会删除图库图片或源文件。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.store.delete_creative_project(project_id):
            if self.current_creative_project_id == project_id:
                self.current_creative_project_id = None
                self.current_creative_node_id = None
            self._refresh_creative_projects()
            self.statusBar().showMessage(f"已删除创作项目：{project.title}")

    def _on_creative_project_combo_changed(self, _index: int) -> None:
        if self._refreshing_creative_projects:
            return
        project_id = self.creative_project_combo.currentData()
        self._load_creative_project(int(project_id) if project_id is not None else None, show_board=project_id is not None)

    def _on_creative_project_list_changed(self) -> None:
        if self._refreshing_creative_projects:
            return
        item = self.creative_project_list.currentItem()
        project_id = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        self.creative_project_combo.blockSignals(True)
        self._set_combo_to_data(self.creative_project_combo, project_id)
        self.creative_project_combo.blockSignals(False)
        self._load_creative_project(int(project_id) if project_id is not None else None, show_board=project_id is not None)

    def _load_creative_project(
        self,
        project_id: int | None,
        *,
        select_node_id: int | None = None,
        show_board: bool = False,
    ) -> None:
        self.current_creative_project_id = project_id
        if project_id is None:
            self.current_creative_node_id = None
            self._current_board_node_id = None
            self._current_board_image_ids = ()
            self.creative_node_tree.clear()
            self.creative_project_copy_input.blockSignals(True)
            self.creative_project_copy_input.clear()
            self.creative_project_copy_input.blockSignals(False)
            self._sync_creative_node_panel()
            return
        project = self.store.get_creative_project(project_id)
        if project is not None:
            self.creative_project_copy_input.blockSignals(True)
            self.creative_project_copy_input.setPlainText(project.copy_text)
            self.creative_project_copy_input.blockSignals(False)
        if select_node_id is None:
            select_node_id = self.current_creative_node_id or self.store.creative_root_node_id(project_id)
        self._refresh_creative_node_tree(select_node_id=select_node_id)
        self._sync_creative_node_panel()
        self._refresh_project_sidebar(select_kind="creative", select_id=project_id)
        if show_board and self.current_creative_node_id is not None:
            QTimer.singleShot(0, self._show_current_creative_board)

    def _refresh_creative_node_tree(self, select_node_id: int | None = None) -> None:
        self.creative_node_tree.blockSignals(True)
        self.creative_node_tree.clear()
        if self.current_creative_project_id is None:
            self.creative_node_tree.blockSignals(False)
            return
        nodes = self.store.list_creative_nodes(self.current_creative_project_id)
        items_by_id: dict[int, QTreeWidgetItem] = {}
        children_by_parent: dict[int | None, list[CreativeNodeItem]] = {}
        for node in nodes:
            children_by_parent.setdefault(node.parent_id, []).append(node)

        def add_node(node: CreativeNodeItem, parent_item: QTreeWidgetItem | None) -> None:
            branch_image_count = len(self.store.creative_node_image_ids(node.id, include_descendants=True))
            item = QTreeWidgetItem([node.title, str(branch_image_count)])
            item.setData(0, Qt.ItemDataRole.UserRole, node.id)
            item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if parent_item is None:
                self.creative_node_tree.addTopLevelItem(item)
            else:
                parent_item.addChild(item)
            items_by_id[node.id] = item
            for child in children_by_parent.get(node.id, []):
                add_node(child, item)

        for root_node in children_by_parent.get(None, []):
            add_node(root_node, None)
        self.creative_node_tree.expandAll()
        if select_node_id in items_by_id:
            self.creative_node_tree.setCurrentItem(items_by_id[select_node_id])
            self.current_creative_node_id = select_node_id
        elif nodes:
            first = nodes[0]
            self.current_creative_node_id = first.id
            self.creative_node_tree.setCurrentItem(items_by_id.get(first.id))
        else:
            self.current_creative_node_id = None
        self.creative_node_tree.blockSignals(False)

    def _on_creative_node_selection_changed(self) -> None:
        item = self.creative_node_tree.currentItem()
        node_id = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        self.current_creative_node_id = int(node_id) if node_id is not None else None
        self._sync_creative_node_panel()
        if self.current_creative_node_id is not None:
            self._show_current_creative_board()

    def _current_creative_node(self) -> CreativeNodeItem | None:
        if self.current_creative_node_id is None:
            return None
        return self.store.get_creative_node(self.current_creative_node_id)

    def _creative_node_path_text(self, node_id: int) -> str:
        if self.current_creative_project_id is None:
            node = self.store.get_creative_node(node_id)
            return node.title if node is not None else ""
        nodes = self.store.list_creative_nodes(self.current_creative_project_id)
        by_id = {node.id: node for node in nodes}
        parts: list[str] = []
        current = by_id.get(node_id)
        visited: set[int] = set()
        while current is not None and current.id not in visited:
            visited.add(current.id)
            parts.append(current.title)
            current = by_id.get(current.parent_id) if current.parent_id is not None else None
        return " / ".join(reversed(parts))

    def _sync_creative_node_panel(self) -> None:
        if not hasattr(self, "creative_node_status_label"):
            return
        node = self._current_creative_node()
        project = (
            self.store.get_creative_project(self.current_creative_project_id)
            if self.current_creative_project_id is not None
            else None
        )
        has_node = node is not None and project is not None
        self.creative_node_note_input.blockSignals(True)
        self.creative_node_query_input.blockSignals(True)
        if not has_node:
            self.creative_node_status_label.setText("未选择创作项目。")
            self.creative_node_note_input.clear()
            self.creative_node_query_input.clear()
        else:
            assert node is not None and project is not None
            self.creative_node_status_label.setText(
                f"项目：{project.title}\n节点：{node.title} ｜ 已存 {node.image_count} 张"
            )
            self.creative_node_note_input.setPlainText(node.note)
            self.creative_node_query_input.setText(node.search_query)
        self.creative_node_note_input.blockSignals(False)
        self.creative_node_query_input.blockSignals(False)
        for button in [
            self.generate_creative_children_button,
            self.generate_creative_copy_button,
            self.generate_creative_copy_tab_button,
            self.search_creative_node_button,
            self.open_creative_board_button,
            self.creative_add_child_button,
            self.creative_delete_node_button,
        ]:
            button.setEnabled(has_node)
        self.save_selection_to_creative_node_button.setEnabled(
            has_node and bool(self._selected_grid_images())
        )
        if node is not None and node.parent_id is None:
            self.creative_delete_node_button.setEnabled(False)

    def _create_creative_project_from_current_brief(self) -> None:
        self._set_ai_workflow_mode("project")
        brief = self.inspiration_brief_input.toPlainText().strip()
        default_title = brief[:32].strip() if brief else "新创作项目"
        title, ok = QInputDialog.getText(self, "新建创作项目", "项目名称", text=default_title)
        if not ok:
            return
        title = title.strip()
        if not title:
            self.statusBar().showMessage("项目名称不能为空")
            return
        template_id = str(self.creative_template_combo.currentData() or "story")
        template = creative_template_by_id(template_id)
        service = self._llm_service_key()
        project_id = self.store.create_creative_project(
            title=title,
            brief=brief,
            language=self.current_language,
            provider_name=self._llm_service_label(service),
            model_name=self._llm_model(service),
        )
        root_id = self.store.creative_root_node_id(project_id)
        if root_id is not None:
            self._seed_creative_template(
                project_id=project_id,
                root_id=root_id,
                project_brief=brief or title,
                template_root=template.root,
            )
        self._refresh_creative_projects(select_project_id=project_id)
        self.right_tab_widget.setCurrentIndex(1)
        self.statusBar().showMessage(f"已新建创作项目：{title} / {template.label}")

    def _seed_creative_template(
        self,
        *,
        project_id: int,
        root_id: int,
        project_brief: str,
        template_root: CreativeTemplateNode,
    ) -> None:
        existing_nodes = self.store.list_creative_nodes(project_id)
        if any(node.parent_id == root_id for node in existing_nodes):
            return
        self.store.update_creative_node(
            root_id,
            note=project_brief or template_root.note,
            search_query=project_brief or template_root.title,
        )

        def add_template_node(parent_id: int, template_node: CreativeTemplateNode) -> int:
            node_id = self.store.create_creative_node(
                project_id=project_id,
                parent_id=parent_id,
                title=template_node.title,
                note=template_node.note,
                search_query=template_search_query(
                    template_node.title,
                    template_node.note,
                    project_brief,
                ),
            )
            for child_template in template_node.children:
                add_template_node(node_id, child_template)
            return node_id

        for template_node in template_root.children:
            add_template_node(root_id, template_node)

    def _create_manual_creative_child_node(self) -> None:
        parent = self._current_creative_node()
        if parent is None or self.current_creative_project_id is None:
            return
        title, ok = QInputDialog.getText(self, "新建子节点", "节点名称")
        if not ok or not title.strip():
            return
        node_id = self.store.create_creative_node(
            project_id=self.current_creative_project_id,
            parent_id=parent.id,
            title=title.strip(),
            note="",
            search_query=title.strip(),
        )
        self._push_creative_node_undo({"kind": "created", "node_id": node_id})
        self._load_creative_project(self.current_creative_project_id, select_node_id=node_id)

    def _delete_selected_creative_node(self) -> None:
        node = self._current_creative_node()
        if node is None or node.parent_id is None:
            return
        answer = QMessageBox.question(
            self,
            "删除节点",
            f"删除节点“{node.title}”及其子节点？\n这只删除项目链接，不会删除图库图片。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        snapshot = self._creative_node_subtree_snapshot(node.id)
        parent_id = node.parent_id
        if self.store.delete_creative_node(node.id):
            self._push_creative_node_undo({"kind": "deleted", "snapshot": snapshot})
            self._load_creative_project(node.project_id, select_node_id=parent_id)

    def _show_creative_node_context_menu(self, pos) -> None:
        item = self.creative_node_tree.itemAt(pos)
        if item is not None:
            self.creative_node_tree.setCurrentItem(item)
        menu = QMenu(self)
        add_action = menu.addAction("新建子节点")
        add_action.setEnabled(self._current_creative_node() is not None)
        delete_action = menu.addAction("删除节点")
        node = self._current_creative_node()
        delete_action.setEnabled(node is not None and node.parent_id is not None)
        undo_action = menu.addAction("撤销节点操作")
        undo_action.setEnabled(bool(self._creative_node_undo_stack))
        chosen = menu.exec(self.creative_node_tree.viewport().mapToGlobal(pos))
        if chosen == add_action:
            self._create_manual_creative_child_node()
        elif chosen == delete_action:
            self._delete_selected_creative_node()
        elif chosen == undo_action:
            self._undo_last_creative_node_operation()

    def _push_creative_node_undo(self, payload: dict[str, object]) -> None:
        self._creative_node_undo_stack.append(payload)
        if len(self._creative_node_undo_stack) > 12:
            self._creative_node_undo_stack.pop(0)

    def _creative_node_subtree_snapshot(self, root_node_id: int) -> dict[str, object]:
        root = self.store.get_creative_node(root_node_id)
        if root is None:
            return {"nodes": [], "root_id": root_node_id}
        nodes = self.store.list_creative_nodes(root.project_id)
        children_by_parent: dict[int | None, list[CreativeNodeItem]] = {}
        for node in nodes:
            children_by_parent.setdefault(node.parent_id, []).append(node)
        ordered: list[CreativeNodeItem] = []

        def visit(node_id: int) -> None:
            node = next((candidate for candidate in nodes if candidate.id == node_id), None)
            if node is None:
                return
            ordered.append(node)
            for child in children_by_parent.get(node.id, []):
                visit(child.id)

        visit(root_node_id)
        return {
            "project_id": root.project_id,
            "parent_id": root.parent_id,
            "root_id": root_node_id,
            "nodes": [
                {
                    "id": node.id,
                    "parent_id": node.parent_id,
                    "title": node.title,
                    "note": node.note,
                    "search_query": node.search_query,
                    "image_ids": self.store.creative_node_image_ids(node.id),
                    "layout": self.store.get_creative_node_board_layout(node.id) or "",
                }
                for node in ordered
            ],
        }

    def _undo_last_creative_node_operation(self) -> None:
        if not self._creative_node_undo_stack:
            self.statusBar().showMessage("没有可撤销的节点操作")
            return
        payload = self._creative_node_undo_stack.pop()
        kind = payload.get("kind")
        if kind == "created":
            node_id = int(payload.get("node_id", 0) or 0)
            node = self.store.get_creative_node(node_id)
            if node is None:
                self.statusBar().showMessage("节点已不存在，无法撤销")
                return
            project_id = node.project_id
            parent_id = node.parent_id
            if self.store.delete_creative_node(node_id):
                self._load_creative_project(project_id, select_node_id=parent_id)
                self.statusBar().showMessage("已撤销新建节点")
            return
        if kind != "deleted":
            return
        snapshot = payload.get("snapshot")
        if not isinstance(snapshot, dict):
            self.statusBar().showMessage("撤销数据无效")
            return
        project_id = int(snapshot.get("project_id", 0) or 0)
        parent_id = snapshot.get("parent_id")
        restored_parent_id = int(parent_id) if parent_id is not None else None
        raw_nodes = snapshot.get("nodes", [])
        if project_id <= 0 or not isinstance(raw_nodes, list):
            self.statusBar().showMessage("撤销数据无效")
            return
        id_map: dict[int, int] = {}
        first_new_id: int | None = None
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                continue
            old_id = int(raw_node.get("id", 0) or 0)
            old_parent = raw_node.get("parent_id")
            new_parent_id = restored_parent_id
            if old_parent is not None:
                new_parent_id = id_map.get(int(old_parent), restored_parent_id)
            new_id = self.store.create_creative_node(
                project_id=project_id,
                parent_id=new_parent_id,
                title=str(raw_node.get("title", "未命名节点")),
                note=str(raw_node.get("note", "")),
                search_query=str(raw_node.get("search_query", "")),
            )
            id_map[old_id] = new_id
            first_new_id = first_new_id or new_id
            image_ids: list[int] = []
            raw_image_ids = raw_node.get("image_ids", [])
            if isinstance(raw_image_ids, list):
                for raw_image_id in raw_image_ids:
                    try:
                        image_id = int(raw_image_id)
                    except (TypeError, ValueError):
                        continue
                    if image_id > 0:
                        image_ids.append(image_id)
            if image_ids:
                self.store.add_images_to_creative_node(
                    new_id,
                    image_ids,
                    intent_label=str(raw_node.get("title", "")),
                    intent_query=str(raw_node.get("search_query", "")),
                )
            layout = str(raw_node.get("layout", "") or "")
            if layout:
                self.store.save_creative_node_board_layout(new_id, layout)
        self._load_creative_project(project_id, select_node_id=first_new_id)
        self.statusBar().showMessage("已撤销删除节点")

    def _save_current_creative_node_details(self) -> None:
        node = self._current_creative_node()
        if node is None:
            return
        updated = self.store.update_creative_node(
            node.id,
            note=self.creative_node_note_input.toPlainText(),
            search_query=self.creative_node_query_input.text(),
        )
        if updated is None:
            self.statusBar().showMessage("节点不存在")
            return
        self._load_creative_project(updated.project_id, select_node_id=updated.id)
        self.statusBar().showMessage("节点已保存")

    def _generate_creative_children_for_selected_node(self) -> None:
        self._set_ai_workflow_mode("project")
        node = self._current_creative_node()
        project = (
            self.store.get_creative_project(self.current_creative_project_id)
            if self.current_creative_project_id is not None
            else None
        )
        if node is None or project is None:
            self.statusBar().showMessage("先选择创作项目节点")
            return
        self.generate_creative_children_button.setEnabled(False)
        self.creative_node_status_label.setText(f"正在补全“{node.title}”...")
        provider = self._make_llm_provider()
        current_note = self.creative_node_note_input.toPlainText() or node.note
        node_path = self._creative_node_path_text(node.id)

        def run() -> None:
            try:
                suggestion, model_name = provider.generate_creative_node_note(
                    project_brief=project.brief or project.title,
                    node_title=node.title,
                    current_note=current_note,
                    node_path=node_path,
                    language=self.current_language,
                )
                self.events.put(("creative_node_note_done", (node.id, suggestion, model_name)))
            except Exception as exc:
                self.events.put(("creative_node_note_error", (node.id, exc)))

        self._start_background_task(
            run,
            on_rejected=lambda: self.generate_creative_children_button.setEnabled(True),
        )

    def _handle_creative_node_note_done(self, node_id: int, suggestion: object, _model_name: str) -> None:
        node = self.store.get_creative_node(node_id)
        if node is None:
            self.statusBar().showMessage("节点已不存在，AI补全结果已丢弃")
            self._sync_creative_node_panel()
            return
        note = str(getattr(suggestion, "note", "")).strip()
        query = str(getattr(suggestion, "search_query", "")).strip()
        if not note and not query:
            self.statusBar().showMessage("AI没有返回可用节点内容")
            self.generate_creative_children_button.setEnabled(True)
            return
        updated = self.store.update_creative_node(
            node.id,
            note=note or node.note,
            search_query=query or node.search_query or node.title,
        )
        self._load_creative_project(node.project_id, select_node_id=node.id)
        self.generate_creative_children_button.setEnabled(True)
        if updated is not None:
            self.statusBar().showMessage(f"已补全节点“{updated.title}”")

    def _handle_creative_node_note_error(self, node_id: int, exc: Exception) -> None:
        if self.current_creative_node_id == node_id:
            self.generate_creative_children_button.setEnabled(True)
            self._sync_creative_node_panel()
        QMessageBox.warning(self, "Eidory", f"AI补全当前节点失败：{exc}")

    def _creative_project_node_payload(self, project_id: int) -> list[dict[str, str]]:
        nodes = self.store.list_creative_nodes(project_id)
        by_id = {node.id: node for node in nodes}
        payload: list[dict[str, str]] = []
        for node in nodes:
            parts: list[str] = []
            current: CreativeNodeItem | None = node
            visited: set[int] = set()
            while current is not None and current.id not in visited:
                visited.add(current.id)
                parts.append(current.title)
                current = by_id.get(current.parent_id) if current.parent_id is not None else None
            payload.append(
                {
                    "title": node.title,
                    "path": " / ".join(reversed(parts)),
                    "note": node.note,
                    "search_query": node.search_query,
                }
            )
        return payload

    def _generate_creative_project_copy(self) -> None:
        self._set_ai_workflow_mode("project")
        project = (
            self.store.get_creative_project(self.current_creative_project_id)
            if self.current_creative_project_id is not None
            else None
        )
        if project is None:
            self.statusBar().showMessage("先选择创作项目")
            return
        nodes_payload = self._creative_project_node_payload(project.id)
        fill_empty_only = any(
            str(node.get("note", "")).strip() or str(node.get("search_query", "")).strip()
            for node in nodes_payload
        )
        selected_node_id = self.current_creative_node_id
        for button in [self.generate_creative_copy_button, self.generate_creative_copy_tab_button]:
            button.setEnabled(False)
        self.statusBar().showMessage(f"正在生成项目文案：{project.title}")
        provider = self._make_llm_provider()

        def run() -> None:
            try:
                suggestion, model_name = provider.generate_creative_project_copy(
                    project_brief=project.brief or project.title,
                    nodes=nodes_payload,
                    language=self.current_language,
                )
                self.events.put(
                    (
                        "creative_project_copy_done",
                        (project.id, selected_node_id, fill_empty_only, suggestion, model_name),
                    )
                )
            except Exception as exc:
                self.events.put(("creative_project_copy_error", (project.id, exc)))

        self._start_background_task(
            run,
            on_rejected=lambda: self._set_creative_copy_buttons_enabled(True),
        )

    def _set_creative_copy_buttons_enabled(self, enabled: bool) -> None:
        has_project = self.current_creative_project_id is not None
        for button in [self.generate_creative_copy_button, self.generate_creative_copy_tab_button]:
            button.setEnabled(bool(enabled and has_project))

    def _handle_creative_project_copy_done(
        self,
        project_id: int,
        selected_node_id: int | None,
        fill_empty_only: bool,
        suggestion: object,
        _model_name: str,
    ) -> None:
        project = self.store.get_creative_project(project_id)
        if project is None:
            self.statusBar().showMessage("项目已不存在，文案结果已丢弃")
            self._set_creative_copy_buttons_enabled(True)
            return
        copy_text = str(getattr(suggestion, "copy_text", "")).strip()
        if copy_text:
            self.store.update_creative_project_copy(project_id, copy_text)
        node_suggestions = getattr(suggestion, "nodes", [])
        if isinstance(node_suggestions, list):
            nodes = self.store.list_creative_nodes(project_id)
            by_title = {node.title: node for node in nodes}
            for node_suggestion in node_suggestions:
                title = str(getattr(node_suggestion, "title", "")).strip()
                node = by_title.get(title)
                if node is None:
                    continue
                if fill_empty_only and (node.note.strip() or node.search_query.strip()):
                    continue
                note = str(getattr(node_suggestion, "note", "")).strip()
                search_query = str(getattr(node_suggestion, "search_query", "")).strip()
                if note or search_query:
                    self.store.update_creative_node(
                        node.id,
                        note=note or node.note,
                        search_query=search_query or node.search_query or node.title,
                    )
        self._load_creative_project(project_id, select_node_id=selected_node_id)
        if hasattr(self, "creative_content_tabs"):
            self.creative_content_tabs.setCurrentIndex(1)
        self._set_creative_copy_buttons_enabled(True)
        self.statusBar().showMessage(f"已生成项目文案：{project.title}")

    def _handle_creative_project_copy_error(self, project_id: int, exc: Exception) -> None:
        if self.current_creative_project_id == project_id:
            self._set_creative_copy_buttons_enabled(True)
        QMessageBox.warning(self, "Eidory", f"生成项目文案失败：{exc}")

    def _search_selected_creative_node(self) -> None:
        node = self._current_creative_node()
        if node is None:
            self.statusBar().showMessage("先选择创作项目节点")
            return
        note_text = self.creative_node_note_input.toPlainText()
        query_text = self.creative_node_query_input.text().strip()
        updated = self.store.update_creative_node(
            node.id,
            note=note_text,
            search_query=query_text,
        )
        if updated is not None:
            node = updated
        query = query_text or node.search_query or note_text.strip() or node.note or node.title
        if not query.strip():
            self.statusBar().showMessage("当前节点没有可搜索内容")
            return
        self.semantic_search_revision += 1
        self._clear_manual_result_order()
        self._clear_result_management_state()
        revision = self.semantic_search_revision
        self.current_result_mode = "creative_node"
        self.current_creative_node_id = node.id
        self.current_creative_node_images = []
        self.current_creative_node_filtered_images = []
        self.current_creative_node_searchable_count = 0
        self.current_creative_node_candidate_limit = 0
        self.current_creative_node_badges = {}
        self.current_temp_project_id = None
        self.current_temp_project_images = []
        self.current_temp_project_badges = {}
        self.search_filters.clear()
        self.active_filter_index = None
        self._refresh_filter_chain_ui()
        self._show_gallery_view()
        self.search_creative_node_button.setEnabled(False)
        self._set_result_status(f"创作节点搜索中：{node.title}")

        def run() -> None:
            try:
                result = self.search_service.semantic_search(query)
                self.events.put(("creative_node_search_done", (revision, node.id, query, result)))
            except Exception as exc:
                self.events.put(("creative_node_search_error", (revision, exc)))

        self._start_background_task(
            run,
            on_rejected=lambda: self.search_creative_node_button.setEnabled(True),
        )

    def _handle_creative_node_search_done(self, revision: int, node_id: int, query: str, result) -> None:
        self.search_creative_node_button.setEnabled(self.current_creative_node_id is not None)
        if revision != self.semantic_search_revision:
            return
        node = self.store.get_creative_node(node_id)
        if node is None:
            return
        self.current_result_mode = "creative_node"
        self.current_creative_node_id = node_id
        self.current_creative_node_images = list(result.images)
        self.current_creative_node_searchable_count = int(result.searchable_count)
        self.current_creative_node_candidate_limit = int(result.candidate_limit)
        self.current_creative_node_badges = {
            image.id: [node.title]
            for image in self.current_creative_node_images
        }
        self._apply_creative_node_result_filters()
        images = self.current_creative_node_filtered_images
        self.grid_view.set_images(images, badges_by_image_id=self.current_creative_node_badges)
        self._set_creative_node_result_status(images, node=node, query=query)
        self._update_creative_node_search_diagnostics(images)
        self._refresh_temp_project_save_button()

    def _apply_creative_node_result_filters(self) -> None:
        images = self.current_creative_node_images
        threshold = self._semantic_score_threshold(images)
        if threshold is not None:
            images = [
                image
                for image in images
                if image.score is not None and image.score >= threshold
            ]
        images = self._apply_result_management_filters(images)
        self.current_creative_node_filtered_images = self._sort_images(images)

    def _set_creative_node_result_status(
        self,
        images: list[ImageItem],
        *,
        node: CreativeNodeItem | None = None,
        query: str | None = None,
    ) -> None:
        node = node or self._current_creative_node()
        title = node.title if node is not None else "创作节点"
        source_count = len(self.current_creative_node_images)
        suffix = self._result_management_status_suffix()
        query_text = f" ｜ {query}" if query else ""
        if len(images) == source_count:
            self._set_result_status(f"创作节点结果：{title} ｜ {len(images)} 张{query_text}{suffix}")
        else:
            self._set_result_status(
                f"创作节点结果：{title} ｜ {len(images)} / 原始 {source_count}{query_text}{suffix}"
            )

    def _update_creative_node_search_diagnostics(self, images: list[ImageItem]) -> None:
        if not images:
            self.search_diagnostics_label.setText("搜索诊断：-")
            return
        scores = [image.score for image in images if image.score is not None]
        parts = [
            f"显示 {len(images)}",
            f"可搜索 {self.current_creative_node_searchable_count}",
            f"候选上限 {self.current_creative_node_candidate_limit}",
        ]
        if scores:
            parts.extend([
                f"最高 {max(scores):.3f}",
                f"最低 {min(scores):.3f}",
                f"平均 {sum(scores) / len(scores):.3f}",
            ])
        self.search_diagnostics_label.setText("搜索诊断：" + "，".join(parts))

    def _save_selection_to_current_creative_node(self) -> None:
        node = self._current_creative_node()
        if node is None:
            self.statusBar().showMessage("先选择创作项目节点")
            return
        selected_images = self._selected_grid_images()
        if not selected_images:
            self.statusBar().showMessage("先在图片墙选择要存入节点的图片")
            return
        query = self.creative_node_query_input.text().strip() or node.search_query or node.title
        changed = self.store.add_images_to_creative_node(
            node.id,
            [image.id for image in selected_images],
            intent_label=node.title,
            intent_query=query,
        )
        self._expand_project_sidebar_section("creative")
        self._refresh_creative_projects(select_project_id=node.project_id)
        self._refresh_creative_selection_panel(selected_images)
        self.statusBar().showMessage(f"已存入 {changed} 张到节点“{node.title}”")

    def _remove_selection_from_current_creative_node(self) -> None:
        node = self._current_creative_node()
        if node is None:
            self.statusBar().showMessage("先选择创作项目节点")
            return
        selected_images = self._selected_grid_images()
        if not selected_images:
            return
        removed = self.store.remove_images_from_creative_node(
            node.id,
            [image.id for image in selected_images],
        )
        if removed:
            self.current_creative_node_images = [
                image for image in self.current_creative_node_images
                if image.id not in {selected.id for selected in selected_images}
            ]
            self._apply_creative_node_result_filters()
            images = self.current_creative_node_filtered_images
            self.grid_view.set_images(images, badges_by_image_id=self.current_creative_node_badges)
            self._set_creative_node_result_status(images)
            self._refresh_creative_projects(select_project_id=node.project_id)
            self._refresh_creative_selection_panel(self._selected_grid_images())
        self.statusBar().showMessage(f"已从节点移除 {removed} 张")

    def _show_creative_node_saved_images(self, node_id: int) -> None:
        node = self.store.get_creative_node(node_id)
        if node is None:
            return
        image_ids = self.store.creative_node_image_ids(node.id, include_descendants=True)
        images = self.store.images_by_ids(image_ids)
        self.current_result_mode = "creative_node"
        self.current_creative_node_id = node.id
        self.current_creative_node_images = images
        self.current_creative_node_searchable_count = 0
        self.current_creative_node_candidate_limit = 0
        self.current_creative_node_badges = self.store.creative_node_image_badges(node.project_id)
        self.current_temp_project_id = None
        self.current_temp_project_images = []
        self.current_temp_project_badges = {}
        self.search_filters.clear()
        self.active_filter_index = None
        self._refresh_filter_chain_ui()
        self._apply_creative_node_result_filters()
        filtered = self.current_creative_node_filtered_images
        self.grid_view.set_images(filtered, badges_by_image_id=self.current_creative_node_badges)
        self._show_gallery_view()
        if filtered:
            self._set_result_status(f"创作节点已存图片：{node.title} ｜ {len(filtered)} 张")
            self.search_diagnostics_label.setText("搜索诊断：-")
        else:
            self._set_result_status(f"创作节点暂无已存图片：{node.title}")
            self.search_diagnostics_label.setText("搜索诊断：点击“搜索当前节点”可从图库匹配参考图。")

    def _show_gallery_view(self) -> None:
        self._save_current_board_layout_if_needed()
        self._current_board_node_id = None
        self._current_board_temp_project_id = None
        self._current_board_image_ids = ()
        self._set_board_focus_mode(False)
        self._set_board_window_pinned(False)
        if hasattr(self, "center_result_stack"):
            self.center_result_stack.setCurrentWidget(self.grid_view)
        if hasattr(self, "load_more_button"):
            self.load_more_button.show()
        if hasattr(self, "save_project_board_layout_button"):
            self.save_project_board_layout_button.setEnabled(False)
        self._set_board_toolbar_visible(False)

    def _show_current_creative_board(self) -> None:
        node = self._current_creative_node()
        if node is None:
            self.statusBar().showMessage("先选择创作项目节点")
            return
        image_ids = self.store.creative_node_image_ids(node.id, include_descendants=True)
        image_id_tuple = tuple(image_ids)
        self._save_current_board_layout_if_needed()
        if (
            self.center_result_stack.currentWidget() is self.project_board_view
            and self._current_board_node_id == node.id
            and self._current_board_image_ids == image_id_tuple
        ):
            self.load_more_button.hide()
            self.save_project_board_layout_button.setEnabled(True)
            self._set_board_toolbar_visible(True)
            self._set_result_status(f"项目看板：{node.title} ｜ {len(image_ids)} 张")
            self.project_board_view.setFocus(Qt.FocusReason.OtherFocusReason)
            return
        images = self.store.images_by_ids(image_ids)
        layout_payload = self._creative_node_board_layout_payload(node.id)
        title = self._creative_node_path_text(node.id) or node.title
        badges = self.store.creative_node_image_badges(node.project_id)
        self.project_board_view.set_images(
            images,
            title=title,
            layout_payload=layout_payload,
            badges_by_image_id=badges,
        )
        self._current_board_node_id = node.id
        self._current_board_temp_project_id = None
        self._current_board_image_ids = image_id_tuple
        self.center_result_stack.setCurrentWidget(self.project_board_view)
        self.project_board_view.setFocus(Qt.FocusReason.OtherFocusReason)
        self.load_more_button.hide()
        self.save_project_board_layout_button.setEnabled(True)
        self._set_board_toolbar_visible(True)
        self._set_result_status(f"项目看板：{node.title} ｜ {len(images)} 张")

    def _show_current_project_board(self) -> None:
        if self.current_result_mode == "temp_project" and self.current_temp_project_id is not None:
            self._show_temporary_project_board(self.current_temp_project_id)
            return
        self._show_current_creative_board()

    def _set_board_toolbar_visible(self, visible: bool) -> None:
        if not hasattr(self, "board_pin_button"):
            return
        board_buttons = [
            self.board_pin_button,
            self.board_hide_selected_button,
            self.board_fit_all_button,
            self.board_flip_button,
            self.board_grayscale_button,
            self.board_import_button,
            self.board_show_all_button,
        ]
        for button in board_buttons:
            button.setVisible(visible)
        self.board_pin_button.setChecked(self._board_window_pinned)
        if hasattr(self, "board_focus_shortcut"):
            self.board_focus_shortcut.setEnabled(True)
        self.shuffle_results_button.setVisible(not visible)
        self.thumbnail_size_label.setVisible(not visible)
        self.thumbnail_size_slider.setVisible(not visible)
        if self._board_focus_mode:
            if visible and not self._board_focus_widget_visibility:
                self._board_focus_widget_visibility = {
                    widget: widget.isVisible()
                    for widget in self._board_focus_chrome_widgets()
                }
            self._hide_board_focus_chrome_widgets()

    def _save_current_creative_board_layout(self) -> None:
        if self._save_current_board_layout_if_needed():
            self.statusBar().showMessage("看板布局已保存")
        else:
            self.statusBar().showMessage("当前没有可保存的看板布局")

    def _toggle_board_window_pin(self, checked: bool = False) -> None:
        self._set_board_window_pinned(bool(checked))

    def _set_board_window_pinned(self, pinned: bool) -> None:
        pinned = bool(pinned)
        if self._board_window_pinned == pinned:
            if hasattr(self, "board_pin_button"):
                self.board_pin_button.setChecked(pinned)
            return
        if not self._set_native_window_pinned(pinned):
            if sys.platform == "darwin":
                if hasattr(self, "board_pin_button"):
                    self.board_pin_button.setChecked(self._board_window_pinned)
                self.statusBar().showMessage("看板窗口置顶失败，已保持原状态")
                return
            self._set_qt_window_pinned_fallback(pinned)
        self._board_window_pinned = pinned
        if hasattr(self, "board_pin_button"):
            self.board_pin_button.setChecked(pinned)
        self.statusBar().showMessage("看板窗口已置顶" if pinned else "看板窗口已取消置顶")

    def _set_native_window_pinned(self, pinned: bool) -> bool:
        if sys.platform != "darwin":
            return False
        app = QApplication.instance()
        if app is None or app.platformName().lower() != "cocoa":
            return False
        try:
            return self._set_macos_window_level(floating=pinned)
        except Exception as exc:
            self._record_error(f"macOS 窗口置顶设置失败：{exc}")
            return False

    def _set_macos_window_level(self, *, floating: bool) -> bool:
        import ctypes
        import ctypes.util

        objc_path = ctypes.util.find_library("objc")
        if not objc_path:
            return False
        objc = ctypes.cdll.LoadLibrary(objc_path)
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        msg_send = objc.objc_msgSend

        def sel(name: str) -> int:
            return int(objc.sel_registerName(name.encode("utf-8")) or 0)

        def send_id(receiver: int, selector: str, *args) -> int:
            if not receiver:
                return 0
            msg_send.restype = ctypes.c_void_p
            return int(msg_send(ctypes.c_void_p(receiver), ctypes.c_void_p(sel(selector)), *args) or 0)

        def send_void_long(receiver: int, selector: str, value: int) -> None:
            if receiver:
                msg_send.restype = None
                msg_send(ctypes.c_void_p(receiver), ctypes.c_void_p(sel(selector)), ctypes.c_long(value))

        native_view = int(self.winId())
        window = send_id(native_view, "window")
        if not window:
            return False
        normal_window_level = 0
        floating_window_level = 3
        send_void_long(window, "setLevel:", floating_window_level if floating else normal_window_level)
        if self.isVisible():
            self.raise_()
            if floating:
                self.activateWindow()
        return True

    def _set_qt_window_pinned_fallback(self, pinned: bool) -> None:
        geometry = self.saveGeometry()
        was_visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, pinned)
        self.restoreGeometry(geometry)
        if was_visible:
            self.show()
            self.raise_()
            if pinned:
                self.activateWindow()

    def _toggle_board_focus_mode(self, _checked: bool = False) -> None:
        self._set_board_focus_mode(not self._board_focus_mode)

    def _set_board_focus_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._board_focus_mode == enabled:
            return
        is_board_view = self.center_result_stack.currentWidget() is self.project_board_view
        if enabled:
            self._board_focus_previous_splitter_sizes = self.root_splitter.sizes()
            self._board_focus_widget_visibility = {}
            if is_board_view:
                self._board_focus_widget_visibility = {
                    widget: widget.isVisible()
                    for widget in self._board_focus_chrome_widgets()
                }
            self._board_focus_mode = True
            if hasattr(self, "board_focus_shortcut"):
                self.board_focus_shortcut.setEnabled(True)
            if is_board_view:
                self._hide_board_focus_chrome_widgets()
            total = max(1, sum(self.root_splitter.sizes()))
            self.root_splitter.setSizes([0, total, 0])
            focus_widget = self.project_board_view if is_board_view else self.grid_view
            focus_widget.setFocus(Qt.FocusReason.ShortcutFocusReason)
            QTimer.singleShot(0, self._reapply_board_window_pin)
            if is_board_view:
                QTimer.singleShot(0, self.project_board_view.fit_all_images)
            return
        self._board_focus_mode = False
        for widget, was_visible in list(self._board_focus_widget_visibility.items()):
            widget.setVisible(was_visible)
        self._board_focus_widget_visibility.clear()
        if self._board_focus_previous_splitter_sizes is not None:
            self.root_splitter.setSizes(self._board_focus_previous_splitter_sizes)
            self._board_focus_previous_splitter_sizes = None
            QTimer.singleShot(0, self._enforce_fixed_sidebar_widths)
        if hasattr(self, "board_focus_shortcut"):
            self.board_focus_shortcut.setEnabled(True)
        if self.center_result_stack.currentWidget() is self.project_board_view:
            self.project_board_view.setFocus(Qt.FocusReason.ShortcutFocusReason)
            QTimer.singleShot(0, self.project_board_view.fit_all_images)
        else:
            self.grid_view.setFocus(Qt.FocusReason.ShortcutFocusReason)
        QTimer.singleShot(0, self._reapply_board_window_pin)

    def _board_focus_chrome_widgets(self) -> list[QWidget]:
        widgets: list[QWidget] = [
            self.search_input,
            self.reverse_exclusion_button,
            self.color_mode_button,
            self.keyword_mode_button,
            self.semantic_mode_button,
            self.collection_filter_button,
            self.tag_filter_button,
            self.similar_image_button,
            self.search_button,
            self.clear_search_button,
            self.advanced_search_toggle_button,
            self.advanced_search_widget,
            self.filter_chain_widget,
            self.score_threshold_label,
            self.score_threshold_slider,
            self.result_state_label,
            self.search_diagnostics_label,
            self.load_more_button,
            self.shuffle_results_button,
            self.thumbnail_size_label,
            self.thumbnail_size_slider,
            self.statusBar(),
        ]
        return [widget for widget in widgets if widget is not None]

    def _hide_board_focus_chrome_widgets(self) -> None:
        for widget in self._board_focus_chrome_widgets():
            widget.hide()

    def _reapply_board_window_pin(self) -> None:
        if not self._board_window_pinned:
            return
        if not self._set_native_window_pinned(True) and sys.platform != "darwin":
            self._set_qt_window_pinned_fallback(True)

    def _fit_board_all_images(self) -> None:
        self.project_board_view.fit_all_images()

    def _flip_board_selected_images(self) -> None:
        changed = self.project_board_view.toggle_selected_flipped()
        self.statusBar().showMessage(f"已左右翻转：{changed} 张")

    def _toggle_board_selected_grayscale(self) -> None:
        changed = self.project_board_view.toggle_selected_grayscale()
        self.statusBar().showMessage(f"已切换黑白显示：{changed} 张")

    def _show_all_board_images(self) -> None:
        self.project_board_view.show_all_items()
        self.statusBar().showMessage("已显示全部看板图片")

    def _import_images_to_current_board(self) -> None:
        self._save_current_board_layout_if_needed()
        node = self._current_creative_node()
        if node is None:
            self.statusBar().showMessage("先选择创作节点")
            return
        collection_id = self._selected_collection_id()
        if collection_id is None:
            self.statusBar().showMessage("先在左侧选择导入目标文件夹")
            return
        filters = "Media Files (*.jpg *.jpeg *.png *.webp *.mp4 *.mov *.m4v *.avi *.mkv *.webm)"
        files, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "导入图片到当前节点",
            str(Path.home()),
            filters,
        )
        if not files:
            return
        self._pending_board_import_node_id = node.id
        self._start_file_import(files, collection_id)

    def _open_project_board_image_preview(self, image_id: int) -> None:
        image = self.store.get_image(int(image_id))
        if image is not None:
            self._open_image_preview(image)

    def _remove_images_from_current_board(self, image_ids: list[int]) -> None:
        if self.center_result_stack.currentWidget() is not self.project_board_view:
            return
        clean_ids = [int(image_id) for image_id in image_ids if int(image_id) > 0]
        if not clean_ids:
            return
        self._save_current_board_layout_if_needed()
        if self._current_board_temp_project_id is not None:
            self._remove_images_from_current_temporary_board(clean_ids)
            return
        if self._current_board_node_id is not None:
            self._remove_images_from_current_creative_board(clean_ids)
            return
        self.statusBar().showMessage("当前看板没有可移除的项目链接")

    def _remove_images_from_current_temporary_board(self, image_ids: list[int]) -> None:
        project_id = self._current_board_temp_project_id
        if project_id is None:
            return
        project = self.store.get_temporary_project(project_id)
        if project is None:
            self._refresh_temporary_projects()
            self.statusBar().showMessage("该项目已不存在")
            return
        links = self.store.temporary_project_image_links(project_id, image_ids)
        if not links:
            self.statusBar().showMessage("选中图片已不在当前项目里")
            return
        removed = self.store.remove_images_from_temporary_project(project_id, image_ids)
        if not removed:
            self.statusBar().showMessage("没有移除图片链接")
            return
        self._push_board_removal_undo(
            {
                "kind": "temporary",
                "project_id": project_id,
                "project_kind": self._temporary_project_ui_kind(project),
                "project_name": project.name,
                "links": links,
            }
        )
        self._refresh_temporary_projects(
            select_project_id=project_id,
            select_kind=self._temporary_project_ui_kind(project),
        )
        self._show_temporary_project_board(project_id)
        label = self._temporary_project_label(project)
        self._record_operation_history(f"从{label}“{project.name}”看板移除 {removed} 个图片链接")
        self.statusBar().showMessage(f"已从“{project.name}”移除 {removed} 个图片链接，源文件未删除。按 Cmd+Z 可恢复")

    def _remove_images_from_current_creative_board(self, image_ids: list[int]) -> None:
        node_id = self._current_board_node_id or self.current_creative_node_id
        node = self.store.get_creative_node(int(node_id)) if node_id is not None else None
        if node is None or not image_ids:
            return
        links = self.store.creative_node_image_links_for_branch(node.id, image_ids)
        if not links:
            self.statusBar().showMessage("选中图片已不在当前节点分支里")
            return
        removed = self.store.remove_images_from_creative_node_branch(node.id, image_ids)
        if removed:
            self._push_board_removal_undo(
                {
                    "kind": "creative",
                    "node_id": node.id,
                    "project_id": node.project_id,
                    "node_title": node.title,
                    "links": links,
                }
            )
            self.current_creative_node_id = node.id
            self._refresh_creative_node_tree(select_node_id=node.id)
            self._sync_creative_node_panel()
            self._show_current_creative_board()
            self._refresh_creative_projects(select_project_id=node.project_id)
            self._record_operation_history(f"从创作节点“{node.title}”看板移除 {removed} 个图片链接")
            self.statusBar().showMessage(f"已从“{node.title}”分支移除 {removed} 个图片链接，源文件未删除。按 Cmd+Z 可恢复")

    def _push_board_removal_undo(self, payload: dict[str, object]) -> None:
        self._board_removal_undo_stack.append(payload)
        if len(self._board_removal_undo_stack) > 12:
            self._board_removal_undo_stack.pop(0)

    def _undo_last_board_removal(self) -> None:
        if self.center_result_stack.currentWidget() is not self.project_board_view:
            self.statusBar().showMessage("当前不在看板里，不能撤销看板移除")
            return
        if not self._board_removal_undo_stack:
            self.statusBar().showMessage("没有可撤销的看板移除操作")
            return
        undo = self._board_removal_undo_stack[-1]
        kind = undo.get("kind")
        links = undo.get("links")
        if not isinstance(links, list) or not links:
            self._board_removal_undo_stack.pop()
            self.statusBar().showMessage("撤销数据无效")
            return
        if kind == "temporary":
            project_id = int(undo.get("project_id", 0) or 0)
            if self._current_board_temp_project_id != project_id:
                self.statusBar().showMessage("只能在原看板里撤销这次移除")
                return
            project = self.store.get_temporary_project(project_id)
            if project is None:
                self._board_removal_undo_stack.pop()
                self._refresh_temporary_projects()
                self.statusBar().showMessage("该项目已不存在，无法恢复")
                return
            restored = self.store.restore_temporary_project_image_links(project_id, links)
            self._board_removal_undo_stack.pop()
            self._refresh_temporary_projects(
                select_project_id=project_id,
                select_kind=self._temporary_project_ui_kind(project),
            )
            self._show_temporary_project_board(project_id)
            self.statusBar().showMessage(f"已恢复 {restored} 个图片链接到“{project.name}”")
            return
        if kind == "creative":
            node_id = int(undo.get("node_id", 0) or 0)
            if self._current_board_node_id != node_id:
                self.statusBar().showMessage("只能在原看板里撤销这次移除")
                return
            node = self.store.get_creative_node(node_id)
            if node is None:
                self._board_removal_undo_stack.pop()
                self._refresh_creative_projects()
                self.statusBar().showMessage("该创作节点已不存在，无法恢复")
                return
            restored = self.store.restore_creative_node_image_links(links)
            self._board_removal_undo_stack.pop()
            self.current_creative_node_id = node.id
            self._refresh_creative_node_tree(select_node_id=node.id)
            self._sync_creative_node_panel()
            self._show_current_creative_board()
            self._refresh_creative_projects(select_project_id=node.project_id)
            self.statusBar().showMessage(f"已恢复 {restored} 个图片链接到“{node.title}”分支")
            return
        self._board_removal_undo_stack.pop()
        self.statusBar().showMessage("撤销数据无效")

    def _project_board_has_keyboard_focus(self) -> bool:
        if self.center_result_stack.currentWidget() is not self.project_board_view:
            return False
        focus_widget = QApplication.focusWidget()
        if focus_widget is None:
            return False
        return (
            focus_widget is self.project_board_view
            or focus_widget is self.project_board_view.viewport()
            or self.project_board_view.isAncestorOf(focus_widget)
        )

    def _handle_undo_shortcut(self) -> None:
        if self._project_board_has_keyboard_focus():
            self._undo_last_board_removal()
            return
        self._undo_last_library_removal()

    def _creative_board_groups(self, project_id: int) -> list[dict[str, object]]:
        groups: list[dict[str, object]] = []
        nodes = self.store.list_creative_nodes(project_id)
        for node in nodes:
            image_ids = self.store.creative_node_image_ids(node.id)
            if not image_ids:
                continue
            images = self.store.images_by_ids(image_ids)
            if images:
                groups.append({"node_id": node.id, "title": node.title, "images": images})
        return groups

    def _creative_board_layout_payload(self, project_id: int) -> dict[str, object] | None:
        payload_json = self.store.get_creative_board_layout(project_id)
        if not payload_json:
            return None
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _creative_node_board_layout_payload(self, node_id: int) -> dict[str, object] | None:
        payload_json = self.store.get_creative_node_board_layout(node_id)
        if not payload_json:
            return None
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _temporary_project_board_layout_payload(self, project_id: int) -> dict[str, object] | None:
        payload_json = self.store.get_temporary_project_board_layout(project_id)
        if not payload_json:
            return None
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _save_current_board_layout_if_needed(self) -> bool:
        if not hasattr(self, "center_result_stack"):
            return False
        if self.center_result_stack.currentWidget() is not self.project_board_view:
            return False
        if self._current_board_temp_project_id is not None:
            payload = self._merged_board_layout_payload(
                self.project_board_view.layout_payload(),
                self._temporary_project_board_layout_payload(self._current_board_temp_project_id),
            )
            self.store.save_temporary_project_board_layout(
                self._current_board_temp_project_id,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            )
            return True
        if self._current_board_node_id is not None:
            payload = self._merged_board_layout_payload(
                self.project_board_view.layout_payload(),
                self._creative_node_board_layout_payload(self._current_board_node_id),
            )
            self.store.save_creative_node_board_layout(
                self._current_board_node_id,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            )
            return True
        return False

    @staticmethod
    def _merged_board_layout_payload(
        current_payload: dict[str, object],
        existing_payload: dict[str, object] | None,
    ) -> dict[str, object]:
        if not isinstance(existing_payload, dict):
            return current_payload
        existing_items = existing_payload.get("items")
        current_items = current_payload.get("items")
        if not isinstance(existing_items, dict) or not isinstance(current_items, dict):
            return current_payload
        merged = dict(current_payload)
        merged_items = dict(existing_items)
        merged_items.update(current_items)
        merged["items"] = merged_items
        return merged

    def _refresh_temporary_projects(
        self,
        select_project_id: int | None = None,
        *,
        select_kind: str = "temporary",
    ) -> None:
        self._refresh_project_sidebar(select_kind=select_kind, select_id=select_project_id)

    def _refresh_project_sidebar(
        self,
        *,
        select_kind: str | None = None,
        select_id: int | None = None,
    ) -> None:
        if not hasattr(self, "temp_project_list"):
            return
        semantic_projects = self.store.list_temporary_projects(kind="semantic")
        quick_projects = self.store.list_temporary_projects(kind="quick")
        creative_projects = self.store.list_creative_projects()
        current_item = self.temp_project_list.currentItem()
        if select_kind is None and current_item is not None:
            current_kind = current_item.data(PROJECT_LIST_KIND_ROLE)
            current_id = current_item.data(PROJECT_LIST_ID_ROLE)
            if current_kind in {"temporary", "quick", "creative"} and current_id is not None:
                select_kind = str(current_kind)
                select_id = int(current_id)
        selected_item: QListWidgetItem | None = None

        self.temp_project_list.blockSignals(True)
        self.temp_project_list.clear()

        def section_expanded(section_id: str) -> bool:
            return bool(self.project_sidebar_expanded_sections.get(section_id, False))

        def add_section(title: str, section_id: str) -> None:
            item = QListWidgetItem(title)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            item.setData(PROJECT_LIST_KIND_ROLE, PROJECT_LIST_SECTION_KIND)
            item.setData(PROJECT_LIST_SECTION_ID_ROLE, section_id)
            item.setData(PROJECT_LIST_SECTION_EXPANDED_ROLE, section_expanded(section_id))
            item.setBackground(QBrush(QColor("#2d343d")))
            item.setToolTip("点击展开或收起")
            self.temp_project_list.addItem(item)

        def add_temporary_items(projects, *, kind_role: str, tooltip_label: str) -> None:
            nonlocal selected_item
            for project in projects:
                item = QListWidgetItem(f"◇ {project.name}    {project.image_count}")
                item.setData(PROJECT_LIST_KIND_ROLE, kind_role)
                item.setData(PROJECT_LIST_ID_ROLE, project.id)
                item.setData(PROJECT_LIST_NAME_ROLE, project.name)
                item.setData(PROJECT_LIST_COLOR_ROLE, project.color_hex)
                item.setData(PROJECT_LIST_COUNT_ROLE, project.image_count)
                item.setData(PROJECT_LIST_PINNED_ROLE, False)
                item.setData(Qt.ItemDataRole.UserRole, project.id)
                item.setData(Qt.ItemDataRole.UserRole + 1, project.name)
                item.setData(Qt.ItemDataRole.UserRole + 2, project.color_hex)
                tooltip_parts = [tooltip_label, project.name, f"{project.image_count} 张"]
                if project.summary:
                    tooltip_parts.append(project.summary)
                item.setToolTip("\n".join(tooltip_parts))
                self.temp_project_list.addItem(item)
                if select_kind == kind_role and select_id == project.id:
                    selected_item = item

        add_section("创作节点项目", "creative")
        if section_expanded("creative") and creative_projects:
            for project in creative_projects:
                pin = "⬆ " if project.is_pinned else ""
                item = QListWidgetItem(f"◇ {pin}{project.title}    {project.image_count}")
                item.setData(PROJECT_LIST_KIND_ROLE, "creative")
                item.setData(PROJECT_LIST_ID_ROLE, project.id)
                item.setData(PROJECT_LIST_NAME_ROLE, project.title)
                item.setData(PROJECT_LIST_COUNT_ROLE, project.image_count)
                item.setData(PROJECT_LIST_PINNED_ROLE, project.is_pinned)
                item.setData(Qt.ItemDataRole.UserRole, project.id)
                item.setData(Qt.ItemDataRole.UserRole + 1, project.title)
                item.setData(Qt.ItemDataRole.UserRole + 2, "")
                item.setToolTip(
                    "\n".join(["创作节点项目", project.title, f"{project.node_count} 个节点", f"{project.image_count} 张"])
                )
                if project.is_pinned:
                    item.setForeground(QBrush(QColor("#f1d58a")))
                self.temp_project_list.addItem(item)
                if select_kind == "creative" and select_id == project.id:
                    selected_item = item
        add_section("语义探针项目", "temporary")
        if section_expanded("temporary"):
            add_temporary_items(semantic_projects, kind_role="temporary", tooltip_label="语义探针项目")
        add_section("暂时收藏", "quick")
        if section_expanded("quick"):
            add_temporary_items(quick_projects, kind_role="quick", tooltip_label="暂时收藏")

        if selected_item is not None:
            self.temp_project_list.setCurrentItem(selected_item)
        self.temp_project_list.blockSignals(False)

    def _handle_project_sidebar_item_clicked(self, item: QListWidgetItem) -> None:
        if item.data(PROJECT_LIST_KIND_ROLE) != PROJECT_LIST_SECTION_KIND:
            return
        section_id = item.data(PROJECT_LIST_SECTION_ID_ROLE)
        if section_id is None:
            return
        section_key = str(section_id)
        was_expanded = bool(self.project_sidebar_expanded_sections.get(section_key, False))
        self._set_project_sidebar_expanded_section(None if was_expanded else section_key)
        self._refresh_project_sidebar()

    def _expand_project_sidebar_section(self, section_id: str) -> None:
        if section_id not in self.project_sidebar_expanded_sections:
            return
        self._set_project_sidebar_expanded_section(section_id)

    def _set_project_sidebar_expanded_section(self, section_id: str | None) -> None:
        for key in self.project_sidebar_expanded_sections:
            self.project_sidebar_expanded_sections[key] = key == section_id

    def _load_selected_temporary_project(self) -> None:
        item = self.temp_project_list.currentItem()
        if item is None:
            return
        kind = item.data(PROJECT_LIST_KIND_ROLE) or "temporary"
        if kind == PROJECT_LIST_SECTION_KIND:
            return
        project_id = item.data(PROJECT_LIST_ID_ROLE)
        if project_id is None:
            project_id = item.data(Qt.ItemDataRole.UserRole)
        if project_id is None:
            return
        if kind == "creative":
            self._set_ai_workflow_mode("project")
            self.creative_project_combo.blockSignals(True)
            self._set_combo_to_data(self.creative_project_combo, int(project_id))
            self.creative_project_combo.blockSignals(False)
            self._load_creative_project(int(project_id), show_board=True)
            self.right_tab_widget.setCurrentIndex(1)
            self._refresh_project_sidebar(select_kind="creative", select_id=int(project_id))
            return
        self._show_temporary_project_board(int(project_id))

    def _load_temporary_project(self, project_id: int, *, update_grid: bool = True) -> None:
        project = self.store.get_temporary_project(project_id)
        if project is None:
            self._refresh_temporary_projects()
            self.statusBar().showMessage("该项目已不存在")
            return
        label = self._temporary_project_label(project)
        image_ids = self.store.temporary_project_image_ids(project_id)
        images = self.store.images_by_ids(image_ids)
        badges = self.store.temporary_project_image_badges(project_id)
        self.semantic_search_revision += 1
        self._clear_manual_result_order()
        self._clear_result_management_state()
        self.search_filters.clear()
        self.active_filter_index = None
        self.current_keyword_query = None
        self.current_semantic_query = None
        self.current_result_mode = "temp_project"
        self.current_temp_project_id = project_id
        self.current_temp_project_images = list(images)
        self.current_temp_project_badges = dict(badges)
        self.current_inspiration_project_id = None
        self.current_inspiration_terms = []
        self.current_inspiration_plan_filters = []
        self.current_inspiration_raw_term_results = []
        self.current_inspiration_images = []
        self.current_inspiration_filtered_images = []
        self.current_inspiration_matches = {}
        self.current_chain_images = []
        self.current_chain_filtered_images = []
        self.current_chain_result = SearchChainResult(images=[])
        self.current_chain_base_image_ids = None
        self.current_chain_base_label = None
        self.current_chain_operation_mode = "replace"
        self.current_offset = 0
        self.load_more_button.setEnabled(False)
        self._refresh_filter_chain_ui()
        if update_grid:
            self.grid_view.set_images(
                self._sort_images(images),
                selected_image_ids=[],
                badges_by_image_id=badges,
            )
        suffix = f" ｜ {project.summary}" if project.summary else ""
        self._set_result_status(f"{label}：{project.name} ｜ {len(images)} 张{suffix}")
        self.search_diagnostics_label.setText("搜索诊断：-")
        self._refresh_project_sidebar(
            select_kind=self._temporary_project_ui_kind(project),
            select_id=project_id,
        )

    def _show_temporary_project_board(self, project_id: int) -> None:
        project = self.store.get_temporary_project(project_id)
        if project is None:
            self._refresh_temporary_projects()
            self.statusBar().showMessage("该项目已不存在")
            return
        label = self._temporary_project_label(project)
        self._save_current_board_layout_if_needed()
        self._load_temporary_project(project_id, update_grid=False)
        images = list(self.current_temp_project_images)
        image_id_tuple = tuple(image.id for image in images)
        if (
            self.center_result_stack.currentWidget() is self.project_board_view
            and self._current_board_temp_project_id == project_id
            and self._current_board_image_ids == image_id_tuple
        ):
            self.load_more_button.hide()
            self.save_project_board_layout_button.setEnabled(False)
            self._set_board_toolbar_visible(True)
            self._set_result_status(f"{label}看板：{project.name} ｜ {len(images)} 张")
            self.project_board_view.setFocus(Qt.FocusReason.OtherFocusReason)
            return
        title = f"{label}：{project.name}"
        layout_payload = self._temporary_project_board_layout_payload(project_id)
        self.project_board_view.set_images(
            images,
            title=title,
            layout_payload=layout_payload,
            badges_by_image_id=self.current_temp_project_badges,
        )
        self._current_board_node_id = None
        self._current_board_temp_project_id = project_id
        self._current_board_image_ids = image_id_tuple
        self.center_result_stack.setCurrentWidget(self.project_board_view)
        self.project_board_view.setFocus(Qt.FocusReason.OtherFocusReason)
        self.load_more_button.hide()
        self.save_project_board_layout_button.setEnabled(False)
        self._set_board_toolbar_visible(True)
        suffix = f" ｜ {project.summary}" if project.summary else ""
        self._set_result_status(f"{label}看板：{project.name} ｜ {len(images)} 张{suffix}")
        self.search_diagnostics_label.setText("搜索诊断：-")

    def _show_temporary_project_context_menu(self, position) -> None:
        item = self.temp_project_list.itemAt(position)
        if item is None:
            menu = QMenu(self)
            clear_semantic_action = menu.addAction("清空语义探针项目")
            clear_quick_action = menu.addAction("清空暂时收藏")
            clear_semantic_action.setEnabled(bool(self.store.list_temporary_projects(kind="semantic")))
            clear_quick_action.setEnabled(bool(self.store.list_temporary_projects(kind="quick")))
            action = menu.exec(self.temp_project_list.viewport().mapToGlobal(position))
            if action == clear_semantic_action:
                self._clear_all_temporary_projects(kind="semantic")
            elif action == clear_quick_action:
                self._clear_all_temporary_projects(kind="quick")
            return
        self.temp_project_list.setCurrentItem(item)
        kind = item.data(PROJECT_LIST_KIND_ROLE) or "temporary"
        project_id = item.data(PROJECT_LIST_ID_ROLE)
        if project_id is None:
            project_id = item.data(Qt.ItemDataRole.UserRole)
        project_name = item.data(PROJECT_LIST_NAME_ROLE)
        if project_name is None:
            project_name = item.data(Qt.ItemDataRole.UserRole + 1)
        if project_id is None:
            return
        if kind == "creative":
            self._show_creative_sidebar_project_context_menu(
                int(project_id),
                self.temp_project_list.viewport().mapToGlobal(position),
            )
            return
        project = self.store.get_temporary_project(int(project_id))
        if project is None:
            self._refresh_temporary_projects()
            self.statusBar().showMessage("该项目已不存在")
            return
        label = self._temporary_project_label(project)
        menu = QMenu(self)
        open_action = menu.addAction(f"打开{label}")
        edit_action = menu.addAction("编辑名称和摘要")
        ai_details_action = menu.addAction("AI 重新命名和摘要")
        ai_details_action.setEnabled(project.kind == "semantic")
        move_up_action = menu.addAction("上移")
        move_down_action = menu.addAction("下移")
        delete_action = menu.addAction(f"删除{label}")
        action = menu.exec(self.temp_project_list.viewport().mapToGlobal(position))
        if action == open_action:
            self._show_temporary_project_board(int(project_id))
        elif action == edit_action:
            self._edit_temporary_project_details(int(project_id))
        elif action == ai_details_action:
            self._request_temporary_project_ai_details(int(project_id))
        elif action == move_up_action:
            self._move_temporary_project(int(project_id), -1)
        elif action == move_down_action:
            self._move_temporary_project(int(project_id), 1)
        elif action == delete_action:
            self._delete_temporary_project(int(project_id), str(project_name or label))

    def _build_creative_sidebar_project_context_menu(self) -> tuple[QMenu, dict[str, QAction]]:
        menu = QMenu(self)
        actions = {
            "open": menu.addAction("打开创作节点项目"),
            "edit": menu.addAction("编辑名称和摘要"),
            "ai_details": menu.addAction("AI 重新命名和摘要"),
            "move_up": menu.addAction("上移"),
            "move_down": menu.addAction("下移"),
            "delete": menu.addAction("删除创作节点项目"),
        }
        actions["ai_details"].setEnabled(False)
        return menu, actions

    def _show_creative_sidebar_project_context_menu(self, project_id: int, global_pos) -> None:
        project = self.store.get_creative_project(project_id)
        if project is None:
            self._refresh_creative_projects()
            self.statusBar().showMessage("该创作项目已不存在")
            return
        menu, actions = self._build_creative_sidebar_project_context_menu()
        chosen = menu.exec(global_pos)
        if chosen == actions["open"]:
            self._set_ai_workflow_mode("project")
            self._load_creative_project(project.id, show_board=True)
            self.right_tab_widget.setCurrentIndex(1)
        elif chosen == actions["edit"]:
            self._edit_creative_project_details(project.id)
        elif chosen == actions["move_up"]:
            self._move_creative_sidebar_project(project.id, -1)
        elif chosen == actions["move_down"]:
            self._move_creative_sidebar_project(project.id, 1)
        elif chosen == actions["delete"]:
            self._delete_creative_project(project.id)

    def _move_creative_sidebar_project(self, project_id: int, direction: int) -> None:
        project = self.store.get_creative_project(project_id)
        if project is None:
            self._refresh_creative_projects()
            self.statusBar().showMessage("该创作项目已不存在")
            return
        moved = self.store.move_creative_project(project_id, direction)
        self._refresh_creative_projects(select_project_id=project_id)
        if moved:
            self.statusBar().showMessage("创作节点项目顺序已更新")
        else:
            self.statusBar().showMessage("创作节点项目已经在边界位置")

    def _move_temporary_project(self, project_id: int, direction: int) -> None:
        project = self.store.get_temporary_project(project_id)
        if project is None:
            self._refresh_temporary_projects()
            self.statusBar().showMessage("该项目已不存在")
            return
        label = self._temporary_project_label(project)
        moved = self.store.move_temporary_project(project_id, direction, kind=project.kind)
        self._refresh_temporary_projects(
            select_project_id=project_id,
            select_kind=self._temporary_project_ui_kind(project),
        )
        if moved:
            self.statusBar().showMessage(f"{label}顺序已更新")
        else:
            self.statusBar().showMessage(f"{label}已经在边界位置")

    def _edit_creative_project_details(self, project_id: int) -> None:
        project = self.store.get_creative_project(project_id)
        if project is None:
            self._refresh_creative_projects()
            self.statusBar().showMessage("该创作项目已不存在")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("编辑创作节点项目")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        title_input = QLineEdit(project.title)
        brief_input = QTextEdit()
        brief_input.setAcceptRichText(False)
        brief_input.setPlainText(project.brief)
        brief_input.setFixedHeight(86)
        form.addRow("名称", title_input)
        form.addRow("摘要", brief_input)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._update_creative_project_details_from_values(
            project_id,
            title=title_input.text(),
            brief=brief_input.toPlainText(),
        )

    def _update_creative_project_details_from_values(
        self,
        project_id: int,
        *,
        title: str,
        brief: str,
    ) -> None:
        clean_title = title.strip()
        if not clean_title:
            self.statusBar().showMessage("项目名称不能为空")
            return
        try:
            updated = self.store.update_creative_project_details(
                project_id,
                title=clean_title,
                brief=brief,
            )
        except ValueError as exc:
            self.statusBar().showMessage(str(exc))
            return
        if updated is None:
            self._refresh_creative_projects()
            self.statusBar().showMessage("该创作项目已不存在")
            return
        self._refresh_creative_projects(select_project_id=project_id)
        if self.current_creative_project_id == project_id:
            self._load_creative_project(project_id, show_board=False)
            if (
                hasattr(self, "center_result_stack")
                and self.center_result_stack.currentWidget() is self.project_board_view
                and self._current_board_node_id is not None
            ):
                self._show_current_creative_board()
        self.statusBar().showMessage(f"已更新创作节点项目：{updated.title}")

    def _edit_temporary_project_details(self, project_id: int) -> None:
        project = self.store.get_temporary_project(project_id)
        if project is None:
            self._refresh_temporary_projects()
            self.statusBar().showMessage("该项目已不存在")
            return
        label = self._temporary_project_label(project)
        dialog = QDialog(self)
        dialog.setWindowTitle(f"编辑{label}")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        name_input = QLineEdit(project.name)
        summary_input = QTextEdit()
        summary_input.setAcceptRichText(False)
        summary_input.setPlainText(project.summary)
        summary_input.setFixedHeight(86)
        form.addRow("名称", name_input)
        form.addRow("摘要", summary_input)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._update_temporary_project_details_from_values(
            project_id,
            name=name_input.text(),
            summary=summary_input.toPlainText(),
        )

    def _update_temporary_project_details_from_values(
        self,
        project_id: int,
        *,
        name: str,
        summary: str,
    ) -> None:
        clean_name = name.strip()
        if not clean_name:
            self.statusBar().showMessage("项目名称不能为空")
            return
        try:
            updated = self.store.update_temporary_project_details(
                project_id,
                name=clean_name,
                summary=summary,
            )
        except ValueError as exc:
            self.statusBar().showMessage(str(exc))
            return
        if updated is None:
            self._refresh_temporary_projects()
            self.statusBar().showMessage("该项目已不存在")
            return
        self._refresh_temporary_projects(
            select_project_id=project_id,
            select_kind=self._temporary_project_ui_kind(updated),
        )
        if self.current_temp_project_id == project_id:
            if (
                hasattr(self, "center_result_stack")
                and self.center_result_stack.currentWidget() is self.project_board_view
                and self._current_board_temp_project_id == project_id
            ):
                self._show_temporary_project_board(project_id)
            else:
                self._load_temporary_project(project_id)
        self.statusBar().showMessage(f"已更新{self._temporary_project_label(updated)}：{updated.name}")

    def _request_temporary_project_ai_details(self, project_id: int) -> None:
        project = self.store.get_temporary_project(project_id)
        if project is None:
            self._refresh_temporary_projects()
            self.statusBar().showMessage("该项目已不存在")
            return
        if project.kind != "semantic":
            self.statusBar().showMessage("暂时收藏不做 AI 命名和摘要")
            return
        image_ids = self.store.temporary_project_image_ids(project_id)
        images = self.store.images_by_ids(image_ids)
        if not images:
            self.statusBar().showMessage("该语义探针项目没有图片，无法生成名称和摘要")
            return
        self.statusBar().showMessage(f"正在用 AI 更新“{project.name}”的名称和摘要...")
        self._suggest_temporary_project_details(
            project_id=project_id,
            images=images,
            can_rename=True,
        )

    def _delete_temporary_project(self, project_id: int, project_name: str) -> None:
        project = self.store.get_temporary_project(project_id)
        label = self._temporary_project_label(project)
        answer = QMessageBox.question(
            self,
            f"删除{label}",
            f"删除“{project_name}”？这只删除{label}，不会删除图片源文件。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        deleted = self.store.delete_temporary_project(project_id)
        if self.current_temp_project_id == project_id:
            self.current_temp_project_id = None
            self.current_temp_project_images = []
            self.current_temp_project_badges = {}
            self._reload_images()
        self._refresh_temporary_projects()
        if deleted:
            self.statusBar().showMessage(f"已删除{label}：{project_name}")

    def _clear_all_temporary_projects(self, *, confirm: bool = True, kind: str | None = None) -> None:
        label = "暂时收藏" if kind == "quick" else "语义探针项目" if kind == "semantic" else "项目"
        projects = self.store.list_temporary_projects(kind=kind) if kind is not None else self.store.list_temporary_projects()
        if not projects:
            self.statusBar().showMessage(f"没有可清空的{label}")
            return
        if confirm:
            answer = QMessageBox.question(
                self,
                f"清空{label}",
                f"清空全部 {len(projects)} 个{label}？这不会删除图片源文件。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        cleared = self.store.clear_temporary_projects(kind=kind)
        was_viewing_temporary_project = self.current_result_mode == "temp_project"
        self.current_temp_project_id = None
        self.current_temp_project_images = []
        self.current_temp_project_badges = {}
        if was_viewing_temporary_project:
            self._reload_images()
        self._refresh_temporary_projects()
        self.statusBar().showMessage(f"已清空 {cleared} 个{label}")

    def _make_collection_tree_item(
        self,
        *,
        collection_id: int,
        name: str,
        count: int,
        depth: int,
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem([name, str(count)])
        item.setData(0, Qt.ItemDataRole.UserRole, collection_id)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, name)
        item.setData(0, COLLECTION_VIRTUAL_FILTER_ROLE, None)
        item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._apply_collection_tree_level_style(item, depth)
        item.setFlags(
            item.flags()
            | Qt.ItemFlag.ItemIsDragEnabled
            | Qt.ItemFlag.ItemIsDropEnabled
        )
        return item

    @staticmethod
    def _expanded_collection_ids(tree: QTreeWidget) -> set[int]:
        expanded_ids: set[int] = set()

        def visit(item: QTreeWidgetItem) -> None:
            collection_id = item.data(0, Qt.ItemDataRole.UserRole)
            if collection_id is not None and item.isExpanded():
                expanded_ids.add(int(collection_id))
            for index in range(item.childCount()):
                visit(item.child(index))

        for index in range(tree.topLevelItemCount()):
            visit(tree.topLevelItem(index))
        return expanded_ids

    @staticmethod
    def _apply_collection_tree_level_style(item: QTreeWidgetItem, depth: int) -> None:
        colors = [
            "#424852",
            "#3b414a",
            "#353b44",
            "#303640",
            "#2d333c",
        ]
        color = "#474e59" if depth < 0 else colors[min(depth, len(colors) - 1)]
        brush = QBrush(QColor(color))
        for column in range(item.columnCount()):
            item.setBackground(column, brush)

    def _refresh_tags(self) -> None:
        current_ids = set(self._selected_tag_ids())
        if not current_ids and self._pending_tag_restore_ids is not None:
            current_ids = set(self._pending_tag_restore_ids)
        self.tag_list.blockSignals(True)
        self.tag_list.clear()
        selected_item: QListWidgetItem | None = None
        all_item = QListWidgetItem("全部标签")
        all_item.setData(Qt.ItemDataRole.UserRole, None)
        all_item.setData(Qt.ItemDataRole.UserRole + 1, None)
        all_item.setData(Qt.ItemDataRole.UserRole + 2, 0)
        self.tag_list.addItem(all_item)
        if not current_ids:
            selected_item = all_item
            all_item.setSelected(True)
        tag_names: list[str] = []
        visible_tags = self._visible_tags_with_counts()
        for tag, _count in self.store.list_tags_with_counts():
            tag_names.append(tag.tag_name)
        for tag, count in visible_tags:
            item = QListWidgetItem(f"{tag.tag_name}    {count}")
            item.setData(Qt.ItemDataRole.UserRole, tag.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, tag.tag_name)
            item.setData(Qt.ItemDataRole.UserRole + 2, count)
            item.setToolTip(tag.tag_name)
            self.tag_list.addItem(item)
            if tag.id in current_ids:
                item.setSelected(True)
                selected_item = item
        tag_filter_text = self.tag_search_input.text().strip().casefold()
        show_untagged = (
            not tag_filter_text
            or tag_filter_text in "未标签".casefold()
            or self.current_virtual_filter == "untagged"
        )
        if show_untagged:
            counts = self.store.virtual_image_filter_counts()
            untagged_item = QListWidgetItem(f"未标签    {counts.get('untagged', 0)}")
            untagged_item.setData(Qt.ItemDataRole.UserRole, None)
            untagged_item.setData(Qt.ItemDataRole.UserRole + 1, None)
            untagged_item.setData(Qt.ItemDataRole.UserRole + 2, counts.get("untagged", 0))
            untagged_item.setData(COLLECTION_VIRTUAL_FILTER_ROLE, "untagged")
            untagged_item.setToolTip(self._virtual_filter_help("untagged"))
            self.tag_list.addItem(untagged_item)
            if self.current_virtual_filter == "untagged":
                selected_item = untagged_item
                untagged_item.setSelected(True)
                all_item.setSelected(False)
        self.tag_completion_model.setStringList(tag_names)
        self.tag_list.setCurrentItem(selected_item or all_item)
        self.tag_list.blockSignals(False)
        self._pending_tag_restore_ids = None
        self._refresh_tag_action_buttons()
        self._save_selected_tag_filter()

    def _visible_tags_with_counts(self) -> list[tuple[object, int]]:
        filter_text = self.tag_search_input.text().strip().casefold()
        tags = [
            (tag, count)
            for tag, count in self.store.list_tags_with_counts()
            if not filter_text or filter_text in tag.tag_name.casefold()
        ]
        sort_mode = self.tag_sort_combo.currentData() if hasattr(self, "tag_sort_combo") else "name"
        if sort_mode == "count_desc":
            return sorted(tags, key=lambda item: (-item[1], item[0].tag_name.casefold()))
        if sort_mode == "count_asc":
            return sorted(tags, key=lambda item: (item[1], item[0].tag_name.casefold()))
        return sorted(tags, key=lambda item: item[0].tag_name.casefold())

    def _on_tag_search_changed(self) -> None:
        self._refresh_tags()

    def _on_tag_sort_changed(self) -> None:
        self.store.set_setting("ui.tag_sort", str(self.tag_sort_combo.currentData() or "name"))
        self._refresh_tags()

    def _on_tag_match_changed(self) -> None:
        self.store.set_setting("ui.tag_match_mode", self._selected_tag_match_mode())

    def _save_status_filter(self) -> None:
        value = self.status_filter_combo.currentData()
        self.store.set_setting("ui.status_filter", str(value or "all"))

    def _save_selected_tag_filter(self) -> None:
        self.store.set_setting(
            "ui.selected_tag_ids",
            ",".join(str(tag_id) for tag_id in self._selected_tag_ids()),
        )

    def _save_right_tab_index(self, index: int) -> None:
        self.store.set_setting("ui.right_tab_index", str(max(0, index)))

    def _refresh_tag_action_buttons(self) -> None:
        tag_id, _tag_name = self._selected_tag_context()
        enabled = tag_id is not None
        self.rename_tag_button.setEnabled(enabled)
        self.delete_tag_button.setEnabled(enabled)
        self.merge_tag_button.setEnabled(enabled)

    def _selected_folder_path_prefix(self) -> str | None:
        item = self.folder_tree.currentItem()
        if item is None:
            return None
        path = item.data(0, Qt.ItemDataRole.UserRole + 2)
        return str(path) if path else None

    def _selected_collection_id(self) -> int | None:
        item = self.collection_tree.currentItem()
        if item is None:
            return None
        collection_id = item.data(0, Qt.ItemDataRole.UserRole)
        return int(collection_id) if collection_id is not None else None

    def _selected_virtual_filter(self) -> str | None:
        valid_filters = {key for key, _label, _help_text in VIRTUAL_COLLECTION_FILTERS}
        return self.current_virtual_filter if self.current_virtual_filter in valid_filters else None

    @staticmethod
    def _virtual_filter_label(virtual_filter: str | None) -> str:
        for key, label, _help_text in VIRTUAL_COLLECTION_FILTERS:
            if key == virtual_filter:
                return label
        return "聚类"

    @staticmethod
    def _virtual_filter_help(virtual_filter: str | None) -> str:
        for key, _label, help_text in VIRTUAL_COLLECTION_FILTERS:
            if key == virtual_filter:
                return help_text
        return "当前显示的是虚拟聚类。"

    def _collection_by_id(self, collection_id: int):
        for collection in self.store.list_collections():
            if collection.id == collection_id:
                return collection
        return None

    def _collection_path_parts(self, collection_id: int) -> list[str]:
        collections = self.store.list_collections()
        by_id = {collection.id: collection for collection in collections}
        parts: list[str] = []
        current = by_id.get(collection_id)
        seen: set[int] = set()
        while current is not None and current.id not in seen:
            seen.add(current.id)
            parts.append(current.name)
            current = by_id.get(current.parent_id) if current.parent_id is not None else None
        return list(reversed(parts))

    def _collection_path_text(self, collection_id: int) -> str:
        parts = self._collection_path_parts(collection_id)
        return " / ".join(parts) if parts else "-"

    def _collection_import_directory(self, collection_id: int) -> Path:
        parts = self._collection_path_parts(collection_id)
        safe_parts = [self._safe_path_component(part) for part in parts if part.strip()]
        if not safe_parts:
            safe_parts = [f"collection-{collection_id}"]
        return Path.home() / "Pictures" / "Eidory Imports" / Path(*safe_parts)

    @staticmethod
    def _safe_path_component(value: str) -> str:
        clean = re.sub(r"[\\/:*?\"<>|]+", "_", value.strip())
        clean = re.sub(r"\s+", " ", clean).strip(" .")
        return clean[:80] or "Untitled"

    @staticmethod
    def _safe_import_filename(value: str) -> str:
        name = Path(value).name
        suffix = Path(name).suffix.lower()
        stem = Path(name).stem
        clean_stem = re.sub(r"[\\/:*?\"<>|]+", "_", stem.strip())
        clean_stem = re.sub(r"\s+", " ", clean_stem).strip(" .")[:120] or "untitled"
        return f"{clean_stem}{suffix}"

    @staticmethod
    def _unique_destination_path(target_dir: Path, filename: str) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        candidate = target_dir / filename
        if not candidate.exists():
            return candidate
        stem = candidate.stem
        suffix = candidate.suffix
        for index in range(1, 10000):
            candidate = target_dir / f"{stem}-{index}{suffix}"
            if not candidate.exists():
                return candidate
        return target_dir / f"{stem}-{uuid.uuid4().hex[:8]}{suffix}"

    def _make_folder_tree_item(
        self,
        *,
        label: str,
        count: int,
        scan_path: str,
        folder_id: int,
        filter_path: str,
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem([label, str(count)])
        item.setData(0, Qt.ItemDataRole.UserRole, scan_path)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, folder_id)
        item.setData(0, Qt.ItemDataRole.UserRole + 2, filter_path)
        item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return item

    @staticmethod
    def _expand_folder_tree_parents(item: QTreeWidgetItem | None) -> None:
        parent = item.parent() if item is not None else None
        while parent is not None:
            parent.setExpanded(True)
            parent = parent.parent()

    @staticmethod
    def _normalize_folder_path(folder_path: str) -> str:
        normalized = os.path.abspath(os.path.expanduser(folder_path))
        return normalized.rstrip(os.sep) or os.sep

    def _setting_int(self, key: str, default: int, minimum: int, maximum: int) -> int:
        raw = self.store.get_setting(key)
        try:
            value = int(raw) if raw is not None else default
        except ValueError:
            value = default
        return max(minimum, min(maximum, value))

    def _setting_int_list(self, key: str, default: list[int], expected_len: int) -> list[int]:
        raw = self.store.get_setting(key)
        if raw is None:
            return list(default)
        try:
            values = [int(part.strip()) for part in raw.split(",")]
        except ValueError:
            return list(default)
        if len(values) != expected_len or any(value <= 0 for value in values):
            return list(default)
        return values

    def _setting_int_csv(self, key: str) -> list[int]:
        raw = self.store.get_setting(key)
        if not raw:
            return []
        values: list[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                value = int(part)
            except ValueError:
                continue
            if value > 0:
                values.append(value)
        return values

    def _setting_choice(self, key: str, default: str, allowed: set[str]) -> str:
        raw = self.store.get_setting(key)
        return raw if raw in allowed else default

    @staticmethod
    def _set_combo_to_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _set_list_current_by_data(list_widget: QListWidget, value: str) -> None:
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == value:
                list_widget.setCurrentItem(item)
                return
        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)

    def _setting_color(self, key: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
        raw = self.store.get_setting(key)
        if raw:
            color = QColor(raw)
            if color.isValid():
                return (color.red(), color.green(), color.blue())
        return default

    @staticmethod
    def _format_score_threshold_label(value: int, prefix: str = "相似度筛选") -> str:
        if value <= 0:
            return f"{prefix}：不限"
        return f"{prefix}：{value}%"

    def _score_threshold_label_prefix(self, search_filter: SearchFilter) -> str:
        if search_filter.kind == "color":
            return "颜色相似度"
        if search_filter.kind == "semantic":
            return "语义相似度"
        if search_filter.kind == "similar":
            return "相似图相似度"
        return "相似度筛选"

    def _refresh_score_threshold_controls(self) -> None:
        if not hasattr(self, "score_threshold_slider"):
            return
        if self.search_filters:
            self._ensure_active_filter_index()
            search_filter = (
                self.search_filters[self.active_filter_index]
                if self.active_filter_index is not None
                and 0 <= self.active_filter_index < len(self.search_filters)
                else None
            )
            if search_filter is None:
                self.score_threshold_slider.setEnabled(False)
                self.score_threshold_label.setText("相似度筛选：无")
                return
            if search_filter.kind not in SCORED_FILTER_KINDS:
                self.score_threshold_slider.setEnabled(False)
                self.score_threshold_label.setText(f"{self._filter_label(search_filter)}：无相似度")
                return
            value = self._score_threshold_value_for_filter(search_filter)
            previous = self.score_threshold_slider.blockSignals(True)
            self.score_threshold_slider.setValue(value)
            self.score_threshold_slider.blockSignals(previous)
            self.score_threshold_slider.setEnabled(True)
            self.score_threshold_label.setText(
                self._format_score_threshold_label(
                    value,
                    self._score_threshold_label_prefix(search_filter),
                )
            )
            return
        self.score_threshold_slider.setEnabled(True)
        self.score_threshold_label.setText(
            self._format_score_threshold_label(self.score_threshold_slider.value())
        )

    def _set_search_filter_score_threshold(self, index: int, value: int) -> bool:
        if index < 0 or index >= len(self.search_filters):
            return False
        search_filter = self.search_filters[index]
        if search_filter.kind not in SCORED_FILTER_KINDS:
            return False
        threshold = max(0, min(100, int(value)))
        self.search_filters[index] = SearchFilter(
            search_filter.kind,
            search_filter.value,
            threshold,
        )
        return True

    def _selected_tag_id(self) -> int | None:
        item = self.tag_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _selected_tag_ids(self) -> list[int]:
        ids: list[int] = []
        for item in self.tag_list.selectedItems():
            tag_id = item.data(Qt.ItemDataRole.UserRole)
            if tag_id is not None:
                ids.append(int(tag_id))
        return ids

    def _selected_tag_virtual_filter(self) -> str | None:
        for item in self.tag_list.selectedItems():
            value = item.data(COLLECTION_VIRTUAL_FILTER_ROLE)
            if value:
                return str(value)
        return None

    def _selected_ai_vision_virtual_filter(self) -> str | None:
        if not hasattr(self, "ai_vision_virtual_filter_list"):
            return None
        item = self.ai_vision_virtual_filter_list.currentItem()
        if item is None:
            return None
        value = item.data(COLLECTION_VIRTUAL_FILTER_ROLE)
        return str(value) if value else None

    def _selected_tag_match_mode(self) -> str:
        return "any" if self.tag_match_combo.currentData() == "any" else "all"

    def _selected_tag_name(self) -> str | None:
        item = self.tag_list.currentItem()
        if item is None:
            return None
        tag_name = item.data(Qt.ItemDataRole.UserRole + 1)
        return str(tag_name) if tag_name else None

    def _selected_tag_names(self) -> list[str]:
        names: list[str] = []
        for item in self.tag_list.selectedItems():
            tag_name = item.data(Qt.ItemDataRole.UserRole + 1)
            if tag_name:
                names.append(str(tag_name))
        return names

    def _collection_name(self, collection_id: int) -> str | None:
        for collection in self.store.list_collections():
            if collection.id == collection_id:
                return collection.name
        return None

    def _collection_choices(self) -> list[tuple[str, int]]:
        collections_with_counts = self.store.list_collections_with_counts()
        by_id = {collection.id: collection for collection, _count in collections_with_counts}
        count_by_id = {collection.id: count for collection, count in collections_with_counts}
        path_cache: dict[int, str] = {}

        def path_for(collection_id: int) -> str:
            cached = path_cache.get(collection_id)
            if cached is not None:
                return cached
            collection = by_id[collection_id]
            if collection.parent_id is None or collection.parent_id not in by_id:
                path = collection.name
            else:
                path = f"{path_for(collection.parent_id)} / {collection.name}"
            path_cache[collection_id] = path
            return path

        return [
            (f"{path_for(collection.id)} ({count_by_id[collection.id]})", collection.id)
            for collection, _count in collections_with_counts
        ]

    def _selected_status_filter(self) -> str | None:
        value = self.status_filter_combo.currentData()
        return None if value == "all" else value

    def _selected_search_mode(self) -> str:
        if self.color_mode_button.isChecked():
            return "color"
        return "semantic" if self.semantic_mode_button.isChecked() else "keyword"

    def _refresh_saved_views(self, select_saved_view_id: int | None = None) -> None:
        current_id = select_saved_view_id
        if current_id is None:
            current_id = self._selected_saved_view_id()
        self.saved_view_combo.blockSignals(True)
        self.saved_view_combo.clear()
        self.saved_view_combo.addItem("未选择预设", None)
        selected_index = 0
        for saved_view in self.store.list_saved_views():
            self.saved_view_combo.addItem(saved_view.name, saved_view.id)
            if current_id == saved_view.id:
                selected_index = self.saved_view_combo.count() - 1
        self.saved_view_combo.setCurrentIndex(selected_index)
        self.saved_view_combo.blockSignals(False)
        self._refresh_saved_view_buttons()

    def _refresh_saved_view_buttons(self) -> None:
        has_view = self._selected_saved_view_id() is not None
        self.apply_view_button.setEnabled(has_view)
        self.rename_view_button.setEnabled(has_view)
        self.delete_view_button.setEnabled(has_view)

    def _selected_saved_view_id(self) -> int | None:
        if not hasattr(self, "saved_view_combo"):
            return None
        value = self.saved_view_combo.currentData()
        return int(value) if value is not None else None

    def _selected_saved_view(self):
        saved_view_id = self._selected_saved_view_id()
        if saved_view_id is None:
            return None
        return self.store.get_saved_view(saved_view_id)

    def _current_view_payload(self) -> dict[str, object]:
        return {
            "version": 1,
            "status_filter": self._selected_status_filter() or "all",
            "selected_tag_ids": self._selected_tag_ids(),
            "tag_match_mode": self._selected_tag_match_mode(),
            "collection_id": self._selected_collection_id(),
            "virtual_filter": self._selected_virtual_filter(),
            "search_filters": [
                self._search_filter_to_payload(search_filter)
                for search_filter in self.search_filters
            ],
            "sort_key": self.current_sort_key,
            "sort_order": "desc" if self.current_sort_desc else "asc",
            "score_threshold": self.score_threshold_slider.value(),
        }

    @staticmethod
    def _search_filter_to_payload(search_filter: SearchFilter) -> dict[str, object]:
        return search_filter_to_payload(search_filter)

    @staticmethod
    def _search_filter_from_payload(payload: object) -> SearchFilter | None:
        return search_filter_from_payload(payload)

    def _apply_view_payload(self, payload: object) -> None:
        if not isinstance(payload, dict):
            QMessageBox.warning(self, "Eidory", "该筛选预设格式无效。")
            return
        signal_widgets = [
            self.status_filter_combo,
            self.tag_match_combo,
            self.sort_combo,
            self.sort_order_combo,
            self.score_threshold_slider,
        ]
        previous_signal_states = [widget.blockSignals(True) for widget in signal_widgets]
        self._applying_view_payload = True
        self.semantic_search_revision += 1
        self.search_button.setEnabled(True)
        try:
            self._apply_collection_from_payload(
                payload.get("collection_id"),
                payload.get("virtual_filter"),
            )
            self._apply_status_from_payload(payload.get("status_filter"))
            self._apply_tags_from_payload(payload.get("selected_tag_ids"))
            self._set_combo_to_data(
                self.tag_match_combo,
                str(payload.get("tag_match_mode") or "all"),
            )
            self._set_combo_to_data(
                self.sort_combo,
                str(payload.get("sort_key") or "default"),
            )
            self._set_combo_to_data(
                self.sort_order_combo,
                str(payload.get("sort_order") or "desc"),
            )
            try:
                threshold = int(payload.get("score_threshold", 0))
            except (TypeError, ValueError):
                threshold = 0
            self.score_threshold_slider.setValue(max(0, min(100, threshold)))
            self.search_filters = [
                search_filter
                for raw_filter in payload.get("search_filters", [])
                if (search_filter := self._search_filter_from_payload(raw_filter)) is not None
            ]
            self.active_filter_index = self._last_score_filter_index()
        finally:
            for widget, previous_state in zip(signal_widgets, previous_signal_states):
                widget.blockSignals(previous_state)
            self._applying_view_payload = False
        self.current_sort_key = str(self.sort_combo.currentData() or "default")
        self.current_sort_desc = self.sort_order_combo.currentData() != "asc"
        self.score_threshold_label.setText(
            self._format_score_threshold_label(self.score_threshold_slider.value())
        )
        self._save_status_filter()
        self._save_selected_tag_filter()
        self.store.set_setting("ui.tag_match_mode", self._selected_tag_match_mode())
        self.store.set_setting("ui.sort_key", self.current_sort_key)
        self.store.set_setting("ui.sort_order", "desc" if self.current_sort_desc else "asc")
        self.store.set_setting("ui.score_threshold", str(self.score_threshold_slider.value()))
        self._sync_legacy_search_state_from_filters()
        self._refresh_filter_chain_ui()
        if self.search_filters:
            self._execute_search_chain()
        else:
            self._reload_images()

    def _apply_collection_from_payload(
        self,
        raw_collection_id: object,
        raw_virtual_filter: object = None,
    ) -> None:
        virtual_filter = str(raw_virtual_filter) if raw_virtual_filter else None
        if virtual_filter in {key for key, _label, _help in VIRTUAL_COLLECTION_FILTERS}:
            self.current_virtual_filter = virtual_filter
            self._refresh_collections()
            self._refresh_tags()
            self._refresh_virtual_collection_filters(select_virtual_filter=virtual_filter)
            return
        collection_id = None
        try:
            collection_id = int(raw_collection_id) if raw_collection_id is not None else None
        except (TypeError, ValueError):
            collection_id = None
        self.current_virtual_filter = None
        self._refresh_collections(select_collection_id=collection_id)

    def _apply_status_from_payload(self, raw_status: object) -> None:
        status = raw_status if raw_status in {"all", "favorite", "unindexed", "missing"} else "all"
        self._set_combo_to_data(self.status_filter_combo, str(status))

    def _apply_tags_from_payload(self, raw_tag_ids: object) -> None:
        tag_ids: set[int] = set()
        if isinstance(raw_tag_ids, list):
            for raw_tag_id in raw_tag_ids:
                try:
                    tag_ids.add(int(raw_tag_id))
                except (TypeError, ValueError):
                    continue
        self.tag_list.blockSignals(True)
        self.tag_list.clearSelection()
        fallback_item = self.tag_list.item(0)
        selected_any = False
        for row in range(1, self.tag_list.count()):
            item = self.tag_list.item(row)
            tag_id = item.data(Qt.ItemDataRole.UserRole)
            if tag_id is not None and int(tag_id) in tag_ids:
                item.setSelected(True)
                self.tag_list.setCurrentItem(item)
                selected_any = True
        if not selected_any and fallback_item is not None:
            fallback_item.setSelected(True)
            self.tag_list.setCurrentItem(fallback_item)
        self.tag_list.blockSignals(False)
        self._save_selected_tag_filter()
        self._refresh_tag_action_buttons()

    def _suggest_saved_view_name(self) -> str:
        parts: list[str] = []
        collection_id = self._selected_collection_id()
        if collection_id is not None:
            parts.append(self._collection_name(collection_id) or "当前文件夹")
        tag_names = self._selected_tag_names()
        if tag_names:
            parts.append("+".join(tag_names[:2]))
        if self.search_filters:
            parts.append(self._format_filter_chain(self.search_filters[:2]))
        return " / ".join(parts) if parts else "新视图"

    def _on_sort_changed(self) -> None:
        self._clear_manual_result_order()
        self.current_sort_key = str(self.sort_combo.currentData() or "default")
        self.current_sort_desc = self.sort_order_combo.currentData() != "asc"
        self.store.set_setting("ui.sort_key", self.current_sort_key)
        self.store.set_setting("ui.sort_order", "desc" if self.current_sort_desc else "asc")
        if self.search_filters:
            self._apply_search_chain_filters()
            images = self.current_chain_filtered_images
            self.grid_view.set_images(images)
            self._set_search_chain_result_status(tuple(self.search_filters), images)
            self._update_search_chain_diagnostics(tuple(self.search_filters), images)
            return
        if self.current_result_mode == "temp_project":
            images = self._apply_result_management_filters(
                self._apply_sidebar_filters(self.current_temp_project_images)
            )
            images = self._sort_images(images)
            self.grid_view.set_images(images, badges_by_image_id=self.current_temp_project_badges)
            project = (
                self.store.get_temporary_project(self.current_temp_project_id)
                if self.current_temp_project_id is not None
                else None
            )
            label = self._temporary_project_label(project)
            name = project.name if project is not None else label
            suffix = f" ｜ {project.summary}" if project is not None and project.summary else ""
            self._set_result_status(
                f"{label}：{name} ｜ {len(images)} 张{suffix}{self._result_management_status_suffix()}"
            )
            return
        self._refresh_current_results_for_filters()

    def _database_sort_key(self) -> str:
        if self.current_sort_key == "score":
            return "default"
        return self.current_sort_key

    def _sort_images(self, images: list[ImageItem]) -> list[ImageItem]:
        if self.manual_result_order_ids:
            return self._apply_manual_result_order(images)
        sort_key = self.current_sort_key
        if sort_key == "default":
            return list(images)

        def value_rank(image: ImageItem):
            value = self._sort_value(image, sort_key)
            if isinstance(value, str):
                return value.casefold()
            return value

        present = [
            image
            for image in images
            if self._sort_value(image, sort_key) is not None
        ]
        missing = [
            image
            for image in images
            if self._sort_value(image, sort_key) is None
        ]
        present.sort(
            key=lambda image: (value_rank(image), image.id),
            reverse=self.current_sort_desc,
        )
        return present + missing

    def _set_manual_result_order(self, images: list[ImageItem]) -> None:
        self.manual_result_order_ids = [image.id for image in images]

    def _clear_manual_result_order(self) -> None:
        self.manual_result_order_ids = None

    def _apply_manual_result_order(self, images: list[ImageItem]) -> list[ImageItem]:
        order = {
            image_id: index
            for index, image_id in enumerate(self.manual_result_order_ids or [])
        }
        in_order: list[ImageItem] = []
        new_images: list[ImageItem] = []
        for image in images:
            if image.id in order:
                in_order.append(image)
            else:
                new_images.append(image)
        in_order.sort(key=lambda image: order[image.id])
        return in_order + new_images

    @staticmethod
    def _sort_value(image: ImageItem, sort_key: str):
        if sort_key == "score":
            return image.score
        if sort_key == "imported":
            return image.imported_at
        if sort_key == "modified":
            return image.modified_time_ns
        if sort_key == "name":
            return image.file_name
        if sort_key == "file_size":
            return image.file_size
        if sort_key == "width":
            return image.width
        if sort_key == "height":
            return image.height
        if sort_key == "pixels":
            if image.width is None or image.height is None:
                return None
            return image.width * image.height
        if sort_key == "duration":
            return image.duration_ms
        return None

    def _stacked_search_scope_ids(self, mode: str) -> list[int] | None:
        if mode == "color" and (
            self.current_result_mode == "semantic" or self.current_semantic_query is not None
        ):
            return [image.id for image in self.current_semantic_filtered_images]
        if mode == "semantic" and (
            self.current_result_mode == "color" or bool(self.current_color_images)
        ):
            return [image.id for image in self.current_color_filtered_images]
        return None

    @staticmethod
    def _format_color_hex(rgb: tuple[int, int, int]) -> str:
        return format_color_hex(rgb)

    def _update_color_swatch(self) -> None:
        color_hex = self._format_color_hex(self.current_color_rgb)
        label = "Color" if getattr(self, "current_language", "zh") == "en" else "颜色"
        self.color_mode_button.setText(label)
        self.color_mode_button.setToolTip(f"选择颜色：{color_hex}")
        self.color_mode_button.setStyleSheet(
            "QPushButton {"
            f"background:{color_hex};"
            "border: 1px solid #6f7782;"
            f"color:{self._swatch_text_color(self.current_color_rgb)};"
            "}"
            "QPushButton:checked {"
            "border: 1px solid #8fb2ff;"
            "}"
        )

    def _choose_search_color(self) -> None:
        current = QColor(*self.current_color_rgb)
        color = QColorDialog.getColor(current, self, "选择搜索颜色")
        if not color.isValid():
            return
        self.current_color_rgb = (color.red(), color.green(), color.blue())
        self.store.set_setting("ui.search_color", self._format_color_hex(self.current_color_rgb))
        self._update_color_swatch()
        self.color_mode_button.setChecked(True)
        self._run_search()

    def _apply_result_management_filters(self, images: list[ImageItem]) -> list[ImageItem]:
        excluded_image_ids = (
            self.result_excluded_image_ids
            | self.result_excluded_collection_image_ids
            | self._result_exclusion_filter_image_ids()
        )
        return [
            image
            for image in images
            if image.id not in excluded_image_ids
        ]

    def _result_management_status_suffix(self) -> str:
        parts: list[str] = []
        if self.result_excluded_image_ids:
            parts.append(f"排除 {len(self.result_excluded_image_ids)}")
        if self.result_excluded_collection_ids:
            parts.append(f"排除文件夹 {len(self.result_excluded_collection_ids)}")
        if self.result_exclusion_filters:
            parts.append(
                f"反向排除 {len(self.result_exclusion_filters)} 项/"
                f"{len(self._result_exclusion_filter_image_ids())} 张"
            )
        return "，" + "，".join(parts) if parts else ""

    def _has_result_context(self) -> bool:
        return self.current_result_mode in {
            "semantic",
            "color",
            "search_chain",
            "inspiration",
            "creative_node",
            "temp_project",
            "duplicate_group",
            "keyword",
        } or bool(self.search_filters)

    def _has_visible_result_context(self) -> bool:
        return self.current_result_mode in {
            "semantic",
            "color",
            "search_chain",
            "inspiration",
            "creative_node",
            "temp_project",
            "duplicate_group",
            "keyword",
        }

    def _clear_result_management_state(self) -> None:
        self.result_excluded_image_ids.clear()
        self.result_excluded_collection_ids.clear()
        self.result_excluded_collection_image_ids.clear()
        self.result_exclusion_filters.clear()
        self.result_exclusion_filter_matches.clear()

    @staticmethod
    def _swatch_text_color(rgb: tuple[int, int, int]) -> str:
        red, green, blue = rgb
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return "#111318" if luminance > 150 else "#f4f6fb"

    def _update_thumbnail_size(self) -> None:
        size = self.thumbnail_size_slider.value()
        self.thumbnail_size_label.setText(f"缩略图：{size}")
        self.grid_view.set_thumbnail_size(size)
        self.store.set_setting("ui.thumbnail_size", str(size))

    def _preview_score_threshold(self, value: int) -> None:
        if (
            self.search_filters
            and self.active_filter_index is not None
            and 0 <= self.active_filter_index < len(self.search_filters)
        ):
            search_filter = self.search_filters[self.active_filter_index]
            if search_filter.kind in SCORED_FILTER_KINDS:
                self.score_threshold_label.setText(
                    self._format_score_threshold_label(
                        value,
                        self._score_threshold_label_prefix(search_filter),
                    )
                )
                return
        self.score_threshold_label.setText(self._format_score_threshold_label(value))

    def _update_score_threshold(self) -> None:
        value = self.score_threshold_slider.value()
        self.store.set_setting("ui.score_threshold", str(value))
        if self.search_filters:
            self._ensure_active_filter_index()
            if self.active_filter_index is None:
                self._refresh_score_threshold_controls()
                return
            active_index = self.active_filter_index
            active_filter = self.search_filters[active_index]
            if active_filter.kind not in SCORED_FILTER_KINDS:
                self._refresh_score_threshold_controls()
                return
            self._set_search_filter_score_threshold(active_index, value)
            self._refresh_score_threshold_controls()
            if active_index == self._last_score_filter_index():
                self._apply_search_chain_filters()
                images = self.current_chain_filtered_images
                self.grid_view.set_images(images)
                self._set_search_chain_result_status(tuple(self.search_filters), images)
                self._update_search_chain_diagnostics(tuple(self.search_filters), images)
            else:
                self._execute_search_chain(operation_mode="recompute")
            return
        self.score_threshold_label.setText(self._format_score_threshold_label(value))
        if self.current_result_mode == "semantic":
            self._apply_semantic_result_filters()
            images = self.current_semantic_filtered_images
            self.grid_view.set_images(images)
            self._set_semantic_result_status(images)
            self._update_search_diagnostics(images)
        elif self.current_result_mode == "color":
            self._apply_color_result_filters()
            images = self.current_color_filtered_images
            self.grid_view.set_images(images)
            self._set_color_result_status(images)
            self._update_color_search_diagnostics(images)
        elif self.current_result_mode == "inspiration":
            self._apply_inspiration_result_filters()
            images = self.current_inspiration_filtered_images
            self.grid_view.set_images(images, badges_by_image_id=self._inspiration_badges_by_image_id())
            self._set_inspiration_result_status(images)
            self._update_inspiration_diagnostics(images)
        elif self.current_result_mode == "creative_node":
            self._apply_creative_node_result_filters()
            images = self.current_creative_node_filtered_images
            self.grid_view.set_images(images, badges_by_image_id=self.current_creative_node_badges)
            self._set_creative_node_result_status(images)
            self._update_creative_node_search_diagnostics(images)

    def _semantic_score_threshold(self, images: list[ImageItem], *, value: int | None = None) -> float | None:
        if value is None:
            value = self.score_threshold_slider.value()
        if value <= 0 or not images:
            return None
        scores = [float(image.score) for image in images if image.score is not None]
        if not scores:
            return None
        low = min(scores)
        high = max(scores)
        if high <= low:
            return high if value >= 100 else low
        return low + (high - low) * (value / 100.0)

    def _apply_semantic_result_filters(self) -> None:
        images = self.current_semantic_images
        threshold = self._semantic_score_threshold(images)
        if threshold is not None:
            images = [
                image
                for image in images
                if image.score is not None and image.score >= threshold
            ]
        images = self._apply_result_management_filters(self._apply_sidebar_filters(images))
        self.current_semantic_filtered_images = self._sort_images(images)

    def _apply_color_result_filters(self) -> None:
        threshold = self._color_score_threshold(self.current_color_images)
        images = self.current_color_images
        if threshold is not None:
            images = [
                image
                for image in images
                if image.score is not None and image.score >= threshold
            ]
        images = self._apply_result_management_filters(self._apply_sidebar_filters(images))
        self.current_color_filtered_images = self._sort_images(images)

    def _set_semantic_result_status(self, images: list[ImageItem]) -> None:
        source_count = len(self.current_semantic_images)
        scope_text = (
            ""
            if self.current_search_scope_count is None
            else f"，叠加范围 {self.current_search_scope_count}"
        )
        suffix = self._result_management_status_suffix()
        if len(images) == source_count:
            self._set_result_status(f"语义结果：{len(images)}{scope_text}{suffix}")
        else:
            self._set_result_status(f"语义结果：{len(images)} / 原始 {source_count}{scope_text}{suffix}")

    def _set_color_result_status(self, images: list[ImageItem]) -> None:
        source_count = len(self.current_color_images)
        color_hex = self._format_color_hex(self.current_color_rgb)
        scope_text = (
            ""
            if self.current_search_scope_count is None
            else f"，叠加范围 {self.current_search_scope_count}"
        )
        suffix = self._result_management_status_suffix()
        if len(images) == source_count:
            self._set_result_status(f"颜色结果 {color_hex}：{len(images)}{scope_text}{suffix}")
        else:
            self._set_result_status(f"颜色结果 {color_hex}：{len(images)} / 原始 {source_count}{scope_text}{suffix}")

    def _update_search_diagnostics(self, images: list[ImageItem]) -> None:
        if self.current_result_mode != "semantic" or not images:
            self.search_diagnostics_label.setText("搜索诊断：-")
            return

        scores = [image.score for image in images if image.score is not None]
        if not scores:
            self.search_diagnostics_label.setText("搜索诊断：无相似度分数")
            return

        threshold = self._semantic_score_threshold(self.current_semantic_images)
        threshold_text = "不限" if threshold is None else f"{threshold:.2f}"
        avg_score = sum(scores) / len(scores)
        self.search_diagnostics_label.setText(
            "搜索诊断："
            f"显示 {len(images)}，可搜索 {self.current_semantic_searchable_count}，"
            f"候选上限 {self.current_semantic_candidate_limit}，"
            f"最高 {max(scores):.3f}，最低 {min(scores):.3f}，"
            f"平均 {avg_score:.3f}，阈值 {threshold_text}（强度 {self.score_threshold_slider.value()}%）"
        )

    def _update_color_search_diagnostics(self, images: list[ImageItem]) -> None:
        if self.current_result_mode != "color" or not images:
            self.search_diagnostics_label.setText("搜索诊断：-")
            return

        scores = [image.score for image in images if image.score is not None]
        if not scores:
            self.search_diagnostics_label.setText("搜索诊断：无颜色分数")
            return

        threshold = self._color_score_threshold(self.current_color_images)
        threshold_text = (
            "不限"
            if threshold is None
            else f"{threshold:.3f}（强度 {self.score_threshold_slider.value()}%）"
        )
        avg_score = sum(scores) / len(scores)
        self.search_diagnostics_label.setText(
            "搜索诊断："
            f"显示 {len(images)}，候选图片 {self.current_color_searchable_count}，"
            f"已有颜色索引 {self.current_color_indexed_count}，"
            f"候选上限 {self.current_color_candidate_limit}，"
            f"最高 {max(scores):.3f}，最低 {min(scores):.3f}，"
            f"平均 {avg_score:.3f}，阈值 {threshold_text}"
        )

    def _color_score_threshold(self, images: list[ImageItem], *, value: int | None = None) -> float | None:
        if value is None:
            value = self.score_threshold_slider.value()
        if value <= 0 or not images:
            return None
        scores = [image.score for image in images if image.score is not None]
        if not scores:
            return None
        return max(scores) * (value / 100.0)

    def _refresh_embedding_stats(self) -> None:
        stats = self.store.embedding_stats(
            model_name=self.embedding_provider.model_name,
            model_revision=self.embedding_provider.model_revision,
            embedding_dim=self.embedding_provider.dim,
        )
        total = stats["total"]
        ready = stats["ready"]
        failed = stats["failed"]
        processing = stats["processing"]
        pending = stats["pending"] + stats["stale"]
        percent = int((ready / total) * 100) if total else 0
        self.embedding_progress_bar.setValue(percent)
        self.embedding_stats_label.setText(
            f"已完成：{ready} / {total}\n"
            f"剩余：{pending}    处理中：{processing}\n"
            f"失败：{failed}    进度：{percent}%"
        )

    def _refresh_ai_vision_stats(self) -> None:
        if not hasattr(self, "ai_vision_progress_bar"):
            return
        provider_name = self._ai_vision_provider_name()
        model_name = self._ai_vision_model_name_for_stats()
        stats = self.store.ai_vision_stats(
            provider_name=provider_name,
            model_name=model_name,
            prompt_version=AI_VISION_PROMPT_VERSION,
        )
        total = stats["total"]
        ready = stats["ready"]
        failed = stats["failed"]
        processing = stats["processing"]
        pending = stats["pending"] + stats["stale"]
        percent = int((ready / total) * 100) if total else 0
        self.ai_vision_progress_bar.setValue(percent)
        self.ai_vision_stats_label.setText(
            f"已完成：{ready} / {total}\n"
            f"剩余：{pending}    处理中：{processing}\n"
            f"失败：{failed}    进度：{percent}%"
        )
        self.ai_vision_rule_tree.blockSignals(True)
        self.ai_vision_rule_tree.clear()
        for rule in self.store.list_ai_vision_collection_rules_with_stats(
            provider_name=provider_name,
            model_name=model_name,
            prompt_version=AI_VISION_PROMPT_VERSION,
        ):
            rule_stats = rule["stats"]
            assert isinstance(rule_stats, dict)
            mode = "识别" if rule["mode"] == "include" else "排除"
            if self.current_language == "en":
                mode = "Include" if rule["mode"] == "include" else "Exclude"
            pending_count = int(rule_stats["pending"]) + int(rule_stats["stale"])
            item = QTreeWidgetItem(
                [
                    mode,
                    str(rule["path"]),
                    str(rule_stats["ready"]),
                    str(rule_stats["failed"]),
                    str(pending_count),
                    str(rule_stats["total"]),
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, int(rule["collection_id"]))
            self.ai_vision_rule_tree.addTopLevelItem(item)
        self.ai_vision_rule_tree.blockSignals(False)
        self._refresh_ai_vision_virtual_filter_entry(self.store.virtual_image_filter_counts())

    def _apply_sidebar_filters(self, images: list[ImageItem]) -> list[ImageItem]:
        folder_path_prefix = self._selected_folder_path_prefix()
        collection_id = self._selected_collection_id()
        collection_image_ids = (
            self.store.image_ids_for_collection(collection_id)
            if collection_id is not None
            else None
        )
        virtual_filter = self._selected_virtual_filter()
        virtual_image_ids = (
            self.store.image_ids_for_virtual_filter(virtual_filter)
            if virtual_filter is not None
            else None
        )
        status = self._selected_status_filter()

        filtered: list[ImageItem] = []
        for image in images:
            if folder_path_prefix and not self._path_is_in_folder(image.file_path, folder_path_prefix):
                continue
            if collection_image_ids is not None and image.id not in collection_image_ids:
                continue
            if virtual_image_ids is not None and image.id not in virtual_image_ids:
                continue
            if status == "favorite" and not image.is_favorite:
                continue
            if status == "unindexed" and image.embedding_status == "ready":
                continue
            if status == "missing" and not image.is_missing:
                continue
            filtered.append(image)
        return filtered

    @staticmethod
    def _path_is_in_folder(file_path: str, folder_path_prefix: str) -> bool:
        normalized_file = os.path.abspath(os.path.expanduser(file_path))
        normalized_folder = os.path.abspath(os.path.expanduser(folder_path_prefix)).rstrip(os.sep) or os.sep
        if normalized_folder == os.sep:
            return normalized_file.startswith(os.sep)
        return normalized_file.startswith(f"{normalized_folder}{os.sep}")
