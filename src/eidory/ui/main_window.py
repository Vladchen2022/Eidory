from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import json
from pathlib import Path

from PySide6.QtCore import QSize, QStringListModel, Qt, QTimer, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QPixmap, QTextOption
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
    QSlider,
    QSplitter,
    QStackedWidget,
    QStatusBar,
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
from eidory.core.embedding_provider import JinaClipV2Provider
from eidory.core.embedding_worker import EmbeddingProgress, EmbeddingWorker
from eidory.core.inspiration import (
    InspirationMatch,
    InspirationTerm,
    mix_inspiration_search_results,
)
from eidory.core.llm_provider import LMStudioProvider, LLMProviderError
from eidory.core.media_types import (
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    is_supported_video,
)
from eidory.core.metadata_store import MetadataStore
from eidory.core.reference_grouping import ReferenceGroup, cluster_reference_vectors
from eidory.core.scanner import ImageScanner, ScanResult
from eidory.core.search_filters import (
    SearchChainResult,
    SearchFilter,
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
from eidory.models import ImageItem
from eidory.ui.collection_tree import CollectionTreeWidget
from eidory.ui.image_preview_dialog import ImagePreviewDialog
from eidory.ui.justified_image_grid import JustifiedImageGridView


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


class EqualWidthTabBar(QTabBar):
    BUTTON_MATCH_HEIGHT = 26
    BOTTOM_GAP = 8

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.updateGeometry()

    def tabSizeHint(self, index: int) -> QSize:
        size = super().tabSizeHint(index)
        count = max(1, self.count())
        parent_width = self.parentWidget().width() if self.parentWidget() is not None else 0
        available_width = max(self.width(), parent_width)
        if available_width > 0:
            size.setWidth(max(34, available_width // count - 1))
        size.setHeight(self.BUTTON_MATCH_HEIGHT + self.BOTTOM_GAP)
        return size


class MainWindow(QMainWindow):
    def __init__(self, *, paths: AppPaths, store: MetadataStore):
        super().__init__()
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
        self.current_inspiration_images: list[ImageItem] = []
        self.current_inspiration_filtered_images: list[ImageItem] = []
        self.current_inspiration_matches: dict[int, list[InspirationMatch]] = {}
        self.inspiration_proposal_terms: list[InspirationTerm] = []
        self.inspiration_questions: list[str] = []
        self.inspiration_model_name = ""
        self.current_temp_project_id: int | None = None
        self.current_temp_project_images: list[ImageItem] = []
        self.current_temp_project_badges: dict[int, list[str]] = {}
        self.search_filters: list[SearchFilter] = []
        self.current_chain_images: list[ImageItem] = []
        self.current_chain_filtered_images: list[ImageItem] = []
        self.current_chain_result = SearchChainResult(images=[])
        self.semantic_search_revision = 0
        self.selected_image: ImageItem | None = None
        self.embedding_refresh_counter = 0
        self._applying_view_payload = False
        self.current_language = self._setting_choice("ui.language", "zh", {"zh", "en"})

        self.setWindowTitle("Eidory")
        self.setStatusBar(QStatusBar())
        self._build_ui()
        self._apply_runtime_language_settings()
        self._connect_signals()
        self._refresh_folders()
        self._refresh_collections()
        self._refresh_temporary_projects()
        self._refresh_tags()
        self._refresh_saved_views()
        self._reload_images()
        self._refresh_embedding_stats()

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_events)
        self.poll_timer.start(250)

    def closeEvent(self, event) -> None:
        if self.embedding_worker is not None:
            self.embedding_worker.stop()
        if hasattr(self, "video_player"):
            self.video_player.stop()
        if hasattr(self, "root_splitter"):
            self.store.set_setting(
                "ui.root_splitter_sizes",
                ",".join(str(size) for size in self.root_splitter.sizes()),
            )
        size = self.size()
        self.store.set_setting("ui.window_width", str(size.width()))
        self.store.set_setting("ui.window_height", str(size.height()))
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self.root_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.root_splitter.addWidget(self._build_sidebar())
        self.root_splitter.addWidget(self._build_library_panel())
        self.root_splitter.addWidget(self._build_detail_panel())
        self.root_splitter.setSizes(
            self._setting_int_list("ui.root_splitter_sizes", [220, 944, 216], 3)
        )
        self.setCentralWidget(self.root_splitter)

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.add_folder_button = QPushButton("导入到当前文件夹")
        self.rescan_button = QPushButton("重新扫描")
        self.rescan_button.hide()
        self.folder_tree = QTreeWidget()
        self.folder_tree.setColumnCount(2)
        self.folder_tree.setHeaderLabels(["文件夹", "张"])
        self.folder_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.folder_tree.setIndentation(14)
        self.folder_tree.setColumnWidth(0, 170)
        self.folder_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.add_collection_button = QPushButton("新建文件夹")
        self.collection_tree = CollectionTreeWidget()
        self.collection_tree.setColumnCount(2)
        self.collection_tree.setHeaderLabels(["文件夹", "张"])
        self.collection_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.collection_tree.setIndentation(14)
        self.collection_tree.setColumnWidth(0, 170)
        self.collection_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.temp_project_list = QListWidget()
        self.temp_project_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.temp_project_list.setMinimumHeight(130)
        self.temp_project_list.setToolTip("保存的临时图片组；删除项目不会影响源文件。")

        self.status_list = QListWidget()
        for label, value in [
            ("全部", "all"),
            ("收藏", "favorite"),
            ("未索引", "unindexed"),
            ("文件丢失", "missing"),
        ]:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, value)
            self.status_list.addItem(item)
        self._set_list_current_by_data(self.status_list, self.initial_status_filter)

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
        layout.addWidget(self.collection_tree, 3)
        layout.addWidget(QLabel("灵感暂存"))
        layout.addWidget(self.temp_project_list, 1)
        return panel

    def _build_library_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("文件名、标签、备注，或语义搜索文本")
        self.search_mode_group = QButtonGroup(self)
        self.search_mode_group.setExclusive(True)
        self.color_mode_button = QPushButton("颜色")
        self.color_mode_button.setCheckable(True)
        self.color_swatch_button = QPushButton()
        self.color_swatch_button.setMaximumWidth(92)
        self.color_swatch_button.setToolTip("选择颜色")
        self._update_color_swatch()
        self.keyword_mode_button = QPushButton("关键词")
        self.keyword_mode_button.setCheckable(True)
        self.semantic_mode_button = QPushButton("语义")
        self.semantic_mode_button.setCheckable(True)
        self.semantic_mode_button.setChecked(True)
        self.similar_image_button = QPushButton("相似图")
        self.similar_image_button.setToolTip("用当前选中的图片查找相似图片")
        self.search_mode_group.addButton(self.color_mode_button)
        self.search_mode_group.addButton(self.keyword_mode_button)
        self.search_mode_group.addButton(self.semantic_mode_button)
        self.search_button = QPushButton("搜索")
        self.clear_search_button = QPushButton("清空")
        search_row.addWidget(self.search_input, 1)
        search_row.addWidget(self.color_mode_button)
        search_row.addWidget(self.color_swatch_button)
        search_row.addWidget(self.keyword_mode_button)
        search_row.addWidget(self.semantic_mode_button)
        search_row.addWidget(self.similar_image_button)
        search_row.addWidget(self.search_button)
        search_row.addWidget(self.clear_search_button)

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
        metadata_filter_row.addStretch(1)

        self.filter_chain_widget = QWidget()
        self.filter_chain_layout = QHBoxLayout(self.filter_chain_widget)
        self.filter_chain_layout.setContentsMargins(0, 0, 0, 0)
        self.filter_chain_label = QLabel("筛选：无")
        self.filter_chain_layout.addWidget(self.filter_chain_label)
        self.filter_chain_layout.addStretch(1)

        threshold_row = QHBoxLayout()
        self.score_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.score_threshold_slider.setRange(0, 100)
        self.score_threshold_slider.setValue(self.initial_score_threshold)
        self.score_threshold_label = QLabel(
            self._format_score_threshold_label(self.initial_score_threshold)
        )
        threshold_row.addWidget(self.score_threshold_label)
        threshold_row.addWidget(self.score_threshold_slider, 1)

        self.result_state_label = QLabel("全部图库")
        self.search_diagnostics_label = QLabel("搜索诊断：-")
        self.search_diagnostics_label.setWordWrap(True)

        self.grid_view = JustifiedImageGridView(
            thumbnail_size=self.initial_thumbnail_size,
            spacing=4,
        )

        self.load_more_button = QPushButton("加载更多")
        thumbnail_size_row = QHBoxLayout()
        self.thumbnail_size_label = QLabel(f"缩略图：{self.initial_thumbnail_size}")
        self.thumbnail_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.thumbnail_size_slider.setRange(96, 320)
        self.thumbnail_size_slider.setValue(self.initial_thumbnail_size)
        self.thumbnail_size_slider.setMaximumWidth(220)
        thumbnail_size_row.addStretch(1)
        thumbnail_size_row.addWidget(self.thumbnail_size_label)
        thumbnail_size_row.addWidget(self.thumbnail_size_slider)

        layout.addLayout(search_row)
        layout.addLayout(metadata_filter_row)
        layout.addWidget(self.filter_chain_widget)
        layout.addLayout(threshold_row)
        layout.addWidget(self.result_state_label)
        layout.addWidget(self.search_diagnostics_label)
        layout.addWidget(self.grid_view, 1)
        layout.addWidget(self.load_more_button)
        layout.addLayout(thumbnail_size_row)
        return panel

    def _build_detail_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(220)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.preview_label = QLabel("未选择图片")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(180)
        self.preview_label.setStyleSheet("background:#2d3138;color:#d8dee9;")
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(180)
        self.video_widget.setStyleSheet("background:#2d3138;")
        self.video_player = QMediaPlayer(self)
        self.video_audio_output = QAudioOutput(self)
        self.video_player.setAudioOutput(self.video_audio_output)
        self.video_player.setVideoOutput(self.video_widget)

        self.preview_stack = QStackedWidget()
        self.preview_stack.addWidget(self.preview_label)
        self.preview_stack.addWidget(self.video_widget)

        self.file_name_label = QLabel("-")
        self.file_name_label.setWordWrap(True)
        self.file_name_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
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
        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("用逗号分隔标签")
        self.tag_completion_model = QStringListModel(self)
        self.tags_input_completer = self._make_tag_completer()
        self.tags_input.setCompleter(self.tags_input_completer)
        self.clear_tags_button = QPushButton("清除")
        tags_widget = QWidget()
        tags_layout = QHBoxLayout(tags_widget)
        tags_layout.setContentsMargins(0, 0, 0, 0)
        tags_layout.addWidget(self.tags_input, 1)
        tags_layout.addWidget(self.clear_tags_button)

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
        self.note_input = QTextEdit()
        self.note_input.setPlaceholderText("备注")
        self.note_input.setAcceptRichText(False)
        self.save_detail_button = QPushButton("保存详情")
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
        self.inspiration_status_label = QLabel("生成后最多选择 7 个语义探针。")
        self.inspiration_status_label.setWordWrap(True)
        self.generate_inspiration_button = QPushButton("生成语义探针")
        self.search_inspiration_button = QPushButton("保存并搜索")
        self.search_inspiration_button.setEnabled(False)
        self.save_temp_project_button = QPushButton("暂存选中图片")
        self.save_temp_project_button.setEnabled(False)

        form = QFormLayout()
        self.detail_form = form
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(4)
        form.addRow("文件名", self.file_name_label)
        form.addRow("路径", self.path_label)
        form.addRow("尺寸", self.size_label)
        form.addRow("修改时间", self.modified_label)
        form.addRow("索引状态", self.embedding_label)
        form.addRow("相似度", self.score_label)
        form.addRow("", self.favorite_checkbox)
        form.addRow("标签", tags_widget)
        form.addRow("批量标签", self.batch_tags_widget)
        self.batch_tags_widget.hide()
        batch_tag_label = form.labelForField(self.batch_tags_widget)
        if batch_tag_label is not None:
            batch_tag_label.hide()

        detail_tab = QWidget()
        detail_layout = QVBoxLayout(detail_tab)
        detail_layout.setContentsMargins(6, 6, 6, 6)
        detail_layout.setSpacing(6)
        detail_tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.note_input.setMaximumHeight(96)
        self.note_input.setMinimumHeight(76)
        detail_layout.addWidget(self.preview_stack)
        detail_layout.addLayout(form)
        detail_layout.addWidget(QLabel("备注"))
        detail_layout.addWidget(self.note_input)
        detail_layout.addWidget(self.save_detail_button)
        detail_layout.addStretch(1)

        inspiration_tab = QWidget()
        inspiration_layout = QVBoxLayout(inspiration_tab)
        inspiration_layout.setContentsMargins(8, 8, 8, 8)
        inspiration_layout.setSpacing(8)
        inspiration_layout.addWidget(QLabel("创作主题"))
        inspiration_layout.addWidget(self.inspiration_brief_input)
        inspiration_layout.addWidget(QLabel("补充信息"))
        inspiration_layout.addWidget(self.inspiration_answers_input)
        inspiration_layout.addWidget(self.inspiration_questions_label)
        inspiration_layout.addWidget(QLabel("历史探针"))
        inspiration_layout.addWidget(self.inspiration_history_list)
        inspiration_layout.addWidget(QLabel("语义探针"))
        inspiration_layout.addWidget(self.inspiration_term_list, 1)
        inspiration_layout.addWidget(self.inspiration_status_label)
        inspiration_button_row = QHBoxLayout()
        inspiration_button_row.setContentsMargins(0, 0, 0, 0)
        inspiration_button_row.addWidget(self.generate_inspiration_button)
        inspiration_button_row.addWidget(self.search_inspiration_button)
        inspiration_layout.addLayout(inspiration_button_row)
        inspiration_layout.addWidget(self.save_temp_project_button)

        filter_tab = QWidget()
        filter_layout = QVBoxLayout(filter_tab)
        filter_layout.setContentsMargins(6, 6, 6, 6)
        filter_layout.setSpacing(6)
        saved_view_row = QHBoxLayout()
        saved_view_row.setContentsMargins(0, 0, 0, 0)
        saved_view_row.addWidget(self.saved_view_combo, 1)
        saved_view_row.addWidget(self.save_view_button)
        saved_view_row.addWidget(self.apply_view_button)
        saved_view_row.addWidget(self.rename_view_button)
        saved_view_row.addWidget(self.delete_view_button)
        filter_layout.addWidget(QLabel("视图预设"))
        filter_layout.addLayout(saved_view_row)
        self.status_list.setMaximumHeight(96)
        filter_layout.addWidget(QLabel("状态"))
        filter_layout.addWidget(self.status_list)
        filter_layout.addWidget(QLabel("标签"))
        tag_tools_row = QHBoxLayout()
        tag_tools_row.setContentsMargins(0, 0, 0, 0)
        tag_tools_row.addWidget(self.tag_search_input, 1)
        tag_tools_row.addWidget(self.tag_sort_combo)
        tag_tools_row.addWidget(self.tag_match_combo)
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
        index_layout.addWidget(self.embedding_progress_bar)
        index_layout.addWidget(self.embedding_stats_label)
        index_layout.addWidget(self.start_embedding_button)
        index_layout.addWidget(self.pause_embedding_button)
        index_layout.addWidget(self.retry_failed_button)
        index_layout.addStretch(1)

        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.setSpacing(8)
        settings_layout.addWidget(QLabel("文本模型设置"))
        settings_form = QFormLayout()
        settings_form.setContentsMargins(0, 0, 0, 0)
        settings_form.setHorizontalSpacing(8)
        settings_form.setVerticalSpacing(8)
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
        settings_form.addRow("模型服务", self.llm_service_combo)
        settings_form.addRow("Endpoint", self.llm_endpoint_input)
        settings_form.addRow("Model", self.llm_model_input)
        settings_form.addRow("API Key", self.llm_api_key_input)
        settings_form.addRow("温度", self.llm_temperature_spin)
        settings_form.addRow("界面语言", self.language_combo)
        settings_layout.addLayout(settings_form)
        self.save_settings_button = QPushButton("保存设置")
        self.settings_status_label = QLabel(
            "LM Studio 默认使用 http://localhost:1234/v1；Ollama 默认使用 http://localhost:11434/v1；API Key 不会写入日志。"
        )
        self.settings_status_label.setWordWrap(True)
        settings_layout.addWidget(self.save_settings_button)
        settings_layout.addWidget(self.settings_status_label)
        settings_layout.addStretch(1)
        self._load_settings_controls()
        self._refresh_inspiration_history()

        self.right_tab_widget = QTabWidget()
        self.right_tab_widget.setObjectName("rightSidebarTabs")
        self.right_tab_widget.setTabBar(EqualWidthTabBar())
        self.right_tab_widget.addTab(detail_tab, "详情")
        self.right_tab_widget.addTab(inspiration_tab, "AI")
        self.right_tab_widget.addTab(filter_tab, "筛选")
        self.right_tab_widget.addTab(index_tab, "索引")
        self.right_tab_widget.addTab(settings_tab, "设置")
        self.right_tab_widget.setElideMode(Qt.TextElideMode.ElideRight)
        tab_bar = self.right_tab_widget.tabBar()
        tab_bar.setExpanding(True)
        tab_bar.setUsesScrollButtons(False)
        tab_bar.setTabToolTip(1, "AI 语义探针")
        self.right_tab_widget.setCurrentIndex(
            self._setting_int("ui.right_tab_index", 0, 0, self.right_tab_widget.count() - 1)
        )

        layout.addWidget(self.right_tab_widget, 1)
        return panel

    def _connect_signals(self) -> None:
        self.add_folder_button.clicked.connect(self._choose_folder)
        self.rescan_button.clicked.connect(self._rescan_selected_folder)
        self.add_collection_button.clicked.connect(self._create_collection_from_button)
        self.search_button.clicked.connect(self._run_search)
        self.clear_search_button.clicked.connect(self._clear_search)
        self.search_input.returnPressed.connect(self._run_search)
        self.similar_image_button.clicked.connect(self._find_similar_to_selected_image)
        self.color_swatch_button.clicked.connect(self._choose_search_color)
        self.add_file_type_filter_button.clicked.connect(self._add_file_type_filter_from_controls)
        self.add_dimension_filter_button.clicked.connect(self._add_dimension_filter_from_controls)
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        self.sort_order_combo.currentIndexChanged.connect(self._on_sort_changed)
        self.score_threshold_slider.valueChanged.connect(self._update_score_threshold)
        self.thumbnail_size_slider.valueChanged.connect(self._update_thumbnail_size)
        self.load_more_button.clicked.connect(self._load_more)
        self.grid_view.selectionChanged.connect(self._on_grid_image_selected)
        self.grid_view.selectionSetChanged.connect(self._on_grid_selection_changed)
        self.grid_view.imageDoubleClicked.connect(self._open_image_preview)
        self.grid_view.imagePreviewRequested.connect(self._open_image_preview)
        self.grid_view.imageContextMenuRequested.connect(self._show_grid_context_menu)
        self.grid_view.filesDropped.connect(self._import_dropped_files_to_selected_collection)
        self.save_detail_button.clicked.connect(self._save_current_details)
        self.open_original_button.clicked.connect(self._open_selected_original)
        self.reveal_in_finder_button.clicked.connect(self._reveal_selected_in_finder)
        self.copy_path_button.clicked.connect(self._copy_selected_path)
        self.play_pause_button.clicked.connect(self._toggle_video_playback)
        self.video_player.playbackStateChanged.connect(self._update_video_play_button)
        self.clear_tags_button.clicked.connect(self._clear_selected_tags)
        self.batch_add_tags_button.clicked.connect(self._batch_add_tags_from_panel)
        self.batch_remove_tag_button.clicked.connect(self._batch_remove_selected_tag)
        self.batch_clear_tags_button.clicked.connect(self._batch_clear_tags)
        self.generate_inspiration_button.clicked.connect(self._generate_inspiration_terms_from_panel)
        self.search_inspiration_button.clicked.connect(self._save_and_search_inspiration)
        self.inspiration_history_list.itemClicked.connect(self._load_selected_inspiration_history)
        self.inspiration_history_list.customContextMenuRequested.connect(self._show_inspiration_history_context_menu)
        self.inspiration_term_list.itemChanged.connect(self._enforce_inspiration_selection_limit)
        self.inspiration_term_list.customContextMenuRequested.connect(self._show_inspiration_term_context_menu)
        self.save_temp_project_button.clicked.connect(self._save_selected_images_as_temporary_project)
        self.feedback_relevant_button.clicked.connect(lambda: self._save_search_feedback("relevant"))
        self.feedback_irrelevant_button.clicked.connect(lambda: self._save_search_feedback("irrelevant"))
        self.feedback_ignored_button.clicked.connect(lambda: self._save_search_feedback("ignored"))
        self.start_embedding_button.clicked.connect(self._start_embedding)
        self.pause_embedding_button.clicked.connect(self._pause_embedding)
        self.retry_failed_button.clicked.connect(self._retry_failed_embeddings)
        self.folder_tree.itemSelectionChanged.connect(self._refresh_current_results_for_filters)
        self.collection_tree.itemSelectionChanged.connect(self._refresh_current_results_for_filters)
        self.collection_tree.treeReordered.connect(self._save_collection_tree_order)
        self.collection_tree.imagesDropped.connect(self._assign_dropped_images_to_collection)
        self.collection_tree.filesDropped.connect(self._import_dropped_files_to_collection)
        self.status_list.itemSelectionChanged.connect(self._refresh_current_results_for_filters)
        self.status_list.itemSelectionChanged.connect(self._save_status_filter)
        self.tag_list.itemSelectionChanged.connect(self._refresh_current_results_for_filters)
        self.tag_list.itemSelectionChanged.connect(self._refresh_tag_action_buttons)
        self.tag_list.itemSelectionChanged.connect(self._save_selected_tag_filter)
        self.tag_search_input.textChanged.connect(self._on_tag_search_changed)
        self.tag_sort_combo.currentIndexChanged.connect(self._on_tag_sort_changed)
        self.tag_match_combo.currentIndexChanged.connect(self._on_tag_match_changed)
        self.rename_tag_button.clicked.connect(self._rename_selected_tag)
        self.delete_tag_button.clicked.connect(self._delete_selected_tag)
        self.merge_tag_button.clicked.connect(self._merge_selected_tag)
        self.folder_tree.customContextMenuRequested.connect(self._show_folder_context_menu)
        self.collection_tree.customContextMenuRequested.connect(self._show_collection_context_menu)
        self.temp_project_list.itemSelectionChanged.connect(self._load_selected_temporary_project)
        self.temp_project_list.customContextMenuRequested.connect(self._show_temporary_project_context_menu)
        self.tag_list.customContextMenuRequested.connect(self._show_tag_context_menu)
        self.right_tab_widget.currentChanged.connect(self._save_right_tab_index)
        self.llm_service_combo.currentIndexChanged.connect(self._on_llm_service_changed)
        self.save_settings_button.clicked.connect(self._save_settings)
        self.saved_view_combo.currentIndexChanged.connect(self._refresh_saved_view_buttons)
        self.save_view_button.clicked.connect(self._save_current_view)
        self.apply_view_button.clicked.connect(self._apply_selected_saved_view)
        self.rename_view_button.clicked.connect(self._rename_selected_saved_view)
        self.delete_view_button.clicked.connect(self._delete_selected_saved_view)

    def _load_settings_controls(self) -> None:
        service = self._llm_service_key()
        self._set_combo_to_data(self.llm_service_combo, service)
        self._load_llm_service_controls(service)
        self.llm_temperature_spin.setValue(self._llm_temperature())
        self._set_combo_to_data(self.language_combo, self.current_language)

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
        self.settings_status_label.setText("设置已保存。API Key 不会写入日志。")
        self.statusBar().showMessage("设置已保存")

    def _apply_runtime_language_settings(self) -> None:
        if self.current_language == "en":
            self.add_collection_button.setText("New Folder")
            self.add_folder_button.setText("Import Here")
            self.search_input.setPlaceholderText("File name, tags, notes, or semantic search text")
            self.color_mode_button.setText("Color")
            self.keyword_mode_button.setText("Keyword")
            self.semantic_mode_button.setText("Semantic")
            self.similar_image_button.setText("Similar")
            self.search_button.setText("Search")
            self.clear_search_button.setText("Clear")
            self.save_detail_button.setText("Save Details")
            self.inspiration_brief_input.setPlaceholderText("Describe the image concept in one sentence")
            self.inspiration_answers_input.setPlaceholderText("Extra context: era, weather, lighting, mood, optional")
            self.inspiration_questions_label.setText("AI questions: -")
            self.inspiration_status_label.setText("Select up to 7 semantic probes.")
            self.generate_inspiration_button.setText("Generate Probes")
            self.search_inspiration_button.setText("Save and Search")
            self.save_temp_project_button.setText("Save Selected")
            self.right_tab_widget.setTabText(0, "Details")
            self.right_tab_widget.setTabText(1, "AI")
            self.right_tab_widget.setTabText(2, "Filters")
            self.right_tab_widget.setTabText(3, "Index")
            self.right_tab_widget.setTabText(4, "Settings")
        else:
            self.add_collection_button.setText("新建文件夹")
            self.add_folder_button.setText("导入到当前文件夹")
            self.search_input.setPlaceholderText("文件名、标签、备注，或语义搜索文本")
            self.color_mode_button.setText("颜色")
            self.keyword_mode_button.setText("关键词")
            self.semantic_mode_button.setText("语义")
            self.similar_image_button.setText("相似图")
            self.search_button.setText("搜索")
            self.clear_search_button.setText("清空")
            self.save_detail_button.setText("保存详情")
            self.inspiration_brief_input.setPlaceholderText("用一句话描述画面的创作主题")
            self.inspiration_answers_input.setPlaceholderText("补充信息：时代、天气、光源、画面气质等，可留空")
            self.inspiration_questions_label.setText("AI 追问：-")
            self.inspiration_status_label.setText("生成后最多选择 7 个语义探针。")
            self.generate_inspiration_button.setText("生成语义探针")
            self.search_inspiration_button.setText("保存并搜索")
            self.save_temp_project_button.setText("暂存选中图片")
            self.right_tab_widget.setTabText(0, "详情")
            self.right_tab_widget.setTabText(1, "AI")
            self.right_tab_widget.setTabText(2, "筛选")
            self.right_tab_widget.setTabText(3, "索引")
            self.right_tab_widget.setTabText(4, "设置")

    def _choose_folder(self) -> None:
        collection_id = self._selected_collection_id()
        if collection_id is None:
            self.statusBar().showMessage("请先选择或新建一个 Eidory 文件夹")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择要导入的磁盘文件夹")
        if folder:
            self._start_import(folder, collection_id, preserve_structure=False)

    def _rescan_selected_folder(self) -> None:
        item = self.folder_tree.currentItem()
        folder_path = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        if not folder_path:
            self.statusBar().showMessage("没有选中文件夹")
            return
        self._start_scan(folder_path)

    def _start_scan(self, folder_path: str) -> None:
        self.statusBar().showMessage(f"扫描中：{folder_path}")
        self.add_folder_button.setEnabled(False)
        self.rescan_button.setEnabled(False)

        def run() -> None:
            try:
                result = self.scanner.scan_folder(folder_path)
                self.events.put(("scan_done", result))
            except Exception as exc:
                self.events.put(("error", f"扫描失败：{exc}"))

        threading.Thread(target=run, daemon=True).start()

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
        self.add_folder_button.setEnabled(False)
        self.rescan_button.setEnabled(False)

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

        threading.Thread(target=run, daemon=True).start()

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
        search_filter = self._search_filter_from_controls()
        if search_filter is None:
            return
        self._add_search_filter(search_filter)
        self._execute_search_chain()

    def _generate_inspiration_terms_from_panel(self) -> None:
        brief = self.inspiration_brief_input.toPlainText().strip()
        if not brief:
            self.inspiration_status_label.setText("先输入创作主题。")
            return
        self.generate_inspiration_button.setEnabled(False)
        self.search_inspiration_button.setEnabled(False)
        service = self._llm_service_key()
        self.inspiration_status_label.setText(f"正在请求 {self._llm_service_label(service)} 生成语义探针...")
        provider = self._make_llm_provider()
        answers = self.inspiration_answers_input.toPlainText()

        def run() -> None:
            try:
                proposal = provider.generate_inspiration_terms(
                    brief=brief,
                    answers=answers,
                    language=self.current_language,
                )
                self.events.put(("inspiration_proposal", proposal))
            except Exception as exc:
                self.events.put(("inspiration_error", exc))

        threading.Thread(target=run, daemon=True).start()

    def _show_inspiration_proposal(self, proposal) -> None:
        self.current_inspiration_project_id = None
        self.inspiration_proposal_terms = list(proposal.terms)
        self.inspiration_questions = list(proposal.questions)
        self.inspiration_model_name = proposal.model_name
        self._populate_inspiration_term_list(
            self.inspiration_proposal_terms,
            default_selected_count=5,
        )
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
        terms = self.store.inspiration_terms_for_project(project_id)
        self.current_inspiration_project_id = project.id
        self.inspiration_proposal_terms = list(terms)
        self.inspiration_questions = list(project.questions)
        self.inspiration_model_name = project.model_name
        self.inspiration_brief_input.setPlainText(project.brief)
        self.inspiration_answers_input.setPlainText(project.answers)
        self._populate_inspiration_term_list(self.inspiration_proposal_terms)
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
            f"删除“{project.title}”？这只删除探针历史，不会删除图片或灵感暂存项目。",
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
        if total == 0:
            self.inspiration_status_label.setText("生成后最多选择 7 个语义探针；右键探针可单条搜索。")
        else:
            self.inspiration_status_label.setText(
                f"已选择 {count} / 7 个语义探针；保存并搜索会混排所有已选探针，右键可单条搜索。"
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
        self._run_inspiration_search(project_id, selected_terms)

    def _run_inspiration_search(
        self,
        project_id: int,
        selected_terms: list[InspirationTerm],
    ) -> None:
        if not selected_terms:
            self.statusBar().showMessage("没有可搜索的语义探针")
            return
        self.semantic_search_revision += 1
        revision = self.semantic_search_revision
        folder_path_prefix = self._selected_folder_path_prefix()
        collection_id = self._selected_collection_id()
        tag_ids = self._selected_tag_ids()
        tag_match_mode = self._selected_tag_match_mode()
        status_filter = self._selected_status_filter()
        self.current_result_mode = "inspiration"
        self.current_inspiration_project_id = project_id
        self.current_inspiration_terms = list(selected_terms)
        self.current_inspiration_images = []
        self.current_inspiration_filtered_images = []
        self.current_inspiration_matches = {}
        self.current_temp_project_id = None
        self.current_temp_project_images = []
        self.current_temp_project_badges = {}
        self.search_filters.clear()
        self.current_offset = 0
        self.load_more_button.setEnabled(False)
        self.generate_inspiration_button.setEnabled(False)
        self.search_inspiration_button.setEnabled(False)
        self._set_result_status(f"灵感项目搜索中：{len(selected_terms)} 个语义探针")
        self.search_diagnostics_label.setText("搜索诊断：-")

        def run() -> None:
            try:
                term_results = []
                for term in selected_terms:
                    result = self.search_service.semantic_search(
                        term.query,
                        folder_path_prefix=folder_path_prefix,
                        collection_id=collection_id,
                        tag_ids=tag_ids,
                        tag_match_mode=tag_match_mode,
                        status_filter=status_filter,
                    )
                    term_results.append((term, result.images[:100]))
                mixed = mix_inspiration_search_results(term_results, limit=500)
                self.events.put(("inspiration_done", (revision, project_id, selected_terms, mixed)))
            except Exception as exc:
                self.events.put(("error", f"灵感项目搜索失败：{exc}"))

        threading.Thread(target=run, daemon=True).start()

    def _find_similar_to_selected_image(self) -> None:
        self._find_similar_to_image(self._selected_grid_image())

    def _find_similar_to_image(self, image: ImageItem | None) -> None:
        reason = self._similar_image_blocking_reason(image, check_vector=True)
        if reason is not None:
            self.statusBar().showMessage(reason)
            return
        assert image is not None
        self._add_search_filter(SearchFilter("similar", image.id))
        self._execute_search_chain()

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
        self._add_search_filter(SearchFilter("file_type", str(value)))
        self._execute_search_chain()

    def _add_dimension_filter_from_controls(self) -> None:
        value = self.dimension_filter_combo.currentData()
        if not value:
            self.statusBar().showMessage("请选择尺寸或方向")
            return
        kind, _separator, filter_value = str(value).partition(":")
        if kind not in {"orientation", "size"} or not filter_value:
            self.statusBar().showMessage("未知的尺寸筛选条件")
            return
        self._add_search_filter(SearchFilter(kind, filter_value))
        self._execute_search_chain()

    def _save_current_view(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "保存视图预设",
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
        self.statusBar().showMessage(f"已保存视图预设：{clean_name}")

    def _apply_selected_saved_view(self) -> None:
        saved_view = self._selected_saved_view()
        if saved_view is None:
            self.statusBar().showMessage("请先选择一个视图预设")
            return
        try:
            payload = json.loads(saved_view.payload_json)
        except json.JSONDecodeError:
            QMessageBox.warning(self, "Eidory", "该视图预设数据损坏，无法载入。")
            return
        self._apply_view_payload(payload)
        self.statusBar().showMessage(f"已载入视图预设：{saved_view.name}")

    def _rename_selected_saved_view(self) -> None:
        saved_view = self._selected_saved_view()
        if saved_view is None:
            self.statusBar().showMessage("请先选择一个视图预设")
            return
        name, ok = QInputDialog.getText(
            self,
            "重命名视图预设",
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
            self.statusBar().showMessage(f"视图预设已重命名为：{clean_name}")

    def _delete_selected_saved_view(self) -> None:
        saved_view = self._selected_saved_view()
        if saved_view is None:
            self.statusBar().showMessage("请先选择一个视图预设")
            return
        answer = QMessageBox.question(
            self,
            "删除视图预设",
            f"删除视图预设“{saved_view.name}”？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.store.delete_saved_view(saved_view.id):
            self._refresh_saved_views()
            self.statusBar().showMessage(f"已删除视图预设：{saved_view.name}")

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

    def _add_search_filter(self, search_filter: SearchFilter) -> None:
        if self.search_filters and self.search_filters[-1].kind == search_filter.kind:
            self.search_filters[-1] = search_filter
        else:
            self.search_filters.append(search_filter)
        self._sync_legacy_search_state_from_filters()
        self._refresh_filter_chain_ui()

    def _execute_search_chain(self) -> None:
        if not self.search_filters:
            self._reload_images()
            return

        self.semantic_search_revision += 1
        revision = self.semantic_search_revision
        filters = tuple(self.search_filters)
        folder_path_prefix = self._selected_folder_path_prefix()
        collection_id = self._selected_collection_id()
        tag_ids = self._selected_tag_ids()
        tag_match_mode = self._selected_tag_match_mode()
        status_filter = self._selected_status_filter()

        self.current_result_mode = "search_chain"
        self.current_offset = 0
        self.load_more_button.setEnabled(False)
        self.search_button.setEnabled(False)
        self.current_chain_images = []
        self.current_chain_filtered_images = []
        self.current_chain_result = SearchChainResult(images=[])
        self._set_result_status(f"筛选中：{self._format_filter_chain(filters)}")
        self.search_diagnostics_label.setText("搜索诊断：-")
        self._refresh_feedback_buttons(self.selected_image)
        self._refresh_filter_chain_ui()

        def run() -> None:
            try:
                result = self._compute_search_chain(
                    filters=filters,
                    folder_path_prefix=folder_path_prefix,
                    collection_id=collection_id,
                    tag_ids=tag_ids,
                    tag_match_mode=tag_match_mode,
                    status_filter=status_filter,
                )
                self.events.put(("search_chain_done", (revision, filters, result)))
            except Exception as exc:
                self.events.put(("error", f"筛选失败：{exc}"))

        threading.Thread(target=run, daemon=True).start()

    def _compute_search_chain(
        self,
        *,
        filters: tuple[SearchFilter, ...],
        folder_path_prefix: str | None,
        collection_id: int | None,
        tag_ids: list[int],
        tag_match_mode: str,
        status_filter: str | None,
    ) -> SearchChainResult:
        images: list[ImageItem] = []
        allowed_image_ids: set[int] | None = None
        semantic_searchable_count = 0
        semantic_candidate_limit = 0
        similar_searchable_count = 0
        similar_candidate_limit = 0
        color_searchable_count = 0
        color_indexed_count = 0
        color_candidate_limit = 0

        for search_filter in filters:
            if search_filter.kind == "semantic":
                result = self.search_service.semantic_search(
                    str(search_filter.value),
                    folder_path_prefix=folder_path_prefix,
                    collection_id=collection_id,
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
                        tag_ids=tag_ids,
                        tag_match_mode=tag_match_mode,
                        status_filter=status_filter,
                    )
                images = [
                    image
                    for image in images
                    if self._image_matches_size(image, str(search_filter.value))
                ]
            allowed_image_ids = {image.id for image in images}
            if not allowed_image_ids:
                break

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

    def _list_chain_base_images(
        self,
        *,
        folder_path_prefix: str | None,
        collection_id: int | None,
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
            limit=50_000,
        )

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
        result,
    ) -> None:
        self.current_result_mode = "inspiration"
        self.current_inspiration_project_id = project_id
        self.current_inspiration_terms = list(selected_terms)
        self.current_inspiration_images = list(result.images)
        self.current_inspiration_matches = dict(result.matches_by_image_id)
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
        threshold = self._score_threshold()
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
            f"阈值 {threshold_text}",
        ]
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
        if len(images) == source_count:
            self._set_result_status(f"灵感项目结果：{len(images)} 张 ｜ {term_titles}")
        else:
            self._set_result_status(
                f"灵感项目结果：{len(images)} / 原始 {source_count} ｜ {term_titles}"
            )

    def _apply_inspiration_result_filters(self) -> None:
        threshold = self._score_threshold()
        images = self.current_inspiration_images
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
        self.current_inspiration_filtered_images = self._sort_images(
            self._apply_sidebar_filters(filtered)
        )

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
        self.current_chain_filtered_images = self._sort_images(self._apply_sidebar_filters(images))

    def _active_score_threshold(self, images: list[ImageItem]) -> float | None:
        score_kind = last_score_filter_kind(self.search_filters)
        if score_kind in {"semantic", "similar"}:
            return self._score_threshold()
        if score_kind == "color":
            return self._color_score_threshold(images)
        return None

    def _set_search_chain_result_status(
        self,
        filters: tuple[SearchFilter, ...],
        images: list[ImageItem],
    ) -> None:
        source_count = len(self.current_chain_images)
        chain = self._format_filter_chain(filters)
        if len(images) == source_count:
            self._set_result_status(f"筛选结果：{len(images)} ｜ {chain}")
        else:
            self._set_result_status(f"筛选结果：{len(images)} / 原始 {source_count} ｜ {chain}")

    def _update_search_chain_diagnostics(
        self,
        filters: tuple[SearchFilter, ...],
        images: list[ImageItem],
    ) -> None:
        if not filters or not images:
            self.search_diagnostics_label.setText("搜索诊断：-")
            return

        parts = [f"显示 {len(images)}", f"条件 {len(filters)}"]
        last_kind = last_score_filter_kind(filters)
        scores = [image.score for image in images if image.score is not None]
        if last_kind == "semantic":
            threshold = self._score_threshold()
            threshold_text = "不限" if threshold is None else f"{threshold:.2f}"
            parts.extend([
                f"语义可搜索 {self.current_semantic_searchable_count}",
                f"候选上限 {self.current_semantic_candidate_limit}",
                f"阈值 {threshold_text}",
            ])
        elif last_kind == "similar":
            threshold = self._score_threshold()
            threshold_text = "不限" if threshold is None else f"{threshold:.2f}"
            parts.extend([
                f"相似可搜索 {self.current_similar_searchable_count}",
                f"候选上限 {self.current_similar_candidate_limit}",
                f"阈值 {threshold_text}",
            ])
        elif last_kind == "color":
            threshold = self._color_score_threshold(self.current_chain_images)
            threshold_text = (
                "不限"
                if threshold is None
                else f"{threshold:.3f}（强度 {self.score_threshold_slider.value()}%）"
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
        if hasattr(self, "color_swatch_button"):
            self._update_color_swatch()

    def _format_filter_chain(self, filters: tuple[SearchFilter, ...] | list[SearchFilter]) -> str:
        return format_filter_chain(filters, image_label_for_id=self._image_label_for_id)

    def _filter_label(self, search_filter: SearchFilter) -> str:
        return filter_label(search_filter, image_label_for_id=self._image_label_for_id)

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
            button = QPushButton(f"× {self._filter_label(search_filter)}")
            button.setToolTip("移除此筛选条件")
            button.clicked.connect(
                lambda _checked=False, filter_index=index: self._remove_search_filter(filter_index)
            )
            self.filter_chain_layout.addWidget(button)

        for label, callback in self._context_filter_actions():
            has_filter = True
            button = QPushButton(f"× {label}")
            button.setToolTip("移除此侧栏筛选")
            button.clicked.connect(lambda _checked=False, action=callback: action())
            self.filter_chain_layout.addWidget(button)

        if not has_filter:
            self.filter_chain_layout.addWidget(QLabel("无"))
        self.filter_chain_layout.addStretch(1)

    def _remove_search_filter(self, index: int) -> None:
        if index < 0 or index >= len(self.search_filters):
            return
        del self.search_filters[index]
        self._sync_legacy_search_state_from_filters()
        self._refresh_filter_chain_ui()
        if self.search_filters:
            self._execute_search_chain()
        else:
            self._reload_images()

    def _context_filter_actions(self) -> list[tuple[str, object]]:
        actions: list[tuple[str, object]] = []
        collection_id = self._selected_collection_id()
        if collection_id is not None:
            collection_name = self._collection_name(collection_id) or "当前文件夹"
            actions.append((f"文件夹：{collection_name}", self._clear_collection_filter))

        tag_names = self._selected_tag_names()
        if tag_names:
            label = " + ".join(tag_names[:3])
            if len(tag_names) > 3:
                label = f"{label} 等 {len(tag_names)} 个"
            mode = "全部" if self._selected_tag_match_mode() == "all" else "任一"
            actions.append((f"标签({mode})：{label}", self._clear_tag_filter))

        status_item = self.status_list.currentItem()
        if status_item is not None and status_item.data(Qt.ItemDataRole.UserRole) != "all":
            actions.append((f"状态：{status_item.text()}", self._clear_status_filter))
        return actions

    def _clear_collection_filter(self) -> None:
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
        item = self.status_list.item(0)
        if item is not None:
            self.status_list.setCurrentItem(item)

    def _reload_images(self) -> None:
        self.current_offset = 0
        self._clear_temporary_project_selection()
        self.search_filters.clear()
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
        self.current_inspiration_images = []
        self.current_inspiration_filtered_images = []
        self.current_inspiration_matches = {}
        self.current_temp_project_id = None
        self.current_temp_project_images = []
        self.current_temp_project_badges = {}
        self.current_chain_images = []
        self.current_chain_filtered_images = []
        self.current_chain_result = SearchChainResult(images=[])
        self.load_more_button.setEnabled(True)
        images = self.store.list_images(
            status_filter=self._selected_status_filter(),
            tag_ids=self._selected_tag_ids(),
            tag_match_mode=self._selected_tag_match_mode(),
            folder_path_prefix=self._selected_folder_path_prefix(),
            collection_id=self._selected_collection_id(),
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
        }:
            return
        self.current_offset += self.page_size
        images = self.store.list_images(
            text_query=self.current_keyword_query,
            status_filter=self._selected_status_filter(),
            tag_ids=self._selected_tag_ids(),
            tag_match_mode=self._selected_tag_match_mode(),
            folder_path_prefix=self._selected_folder_path_prefix(),
            collection_id=self._selected_collection_id(),
            limit=self.page_size,
            offset=self.current_offset,
            sort_key=self._database_sort_key(),
            sort_desc=self.current_sort_desc,
        )
        self.grid_view.append_images(images)
        self._set_result_status(f"已加载 {self.grid_view.rowCount()} 张，新增加载 {len(images)} 张")

    def _clear_search(self) -> None:
        self.semantic_search_revision += 1
        self.search_filters.clear()
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
        self.current_inspiration_images = []
        self.current_inspiration_filtered_images = []
        self.current_inspiration_matches = {}
        self.current_chain_images = []
        self.current_chain_filtered_images = []
        self.current_chain_result = SearchChainResult(images=[])
        self.search_input.clear()
        self.search_button.setEnabled(True)
        self._refresh_filter_chain_ui()
        self._reload_images()

    def _refresh_current_results_for_filters(self) -> None:
        self._refresh_filter_chain_ui()
        if self.search_filters:
            if self.search_button.isEnabled():
                self._execute_search_chain()
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
                tag_ids=self._selected_tag_ids(),
                tag_match_mode=self._selected_tag_match_mode(),
                folder_path_prefix=self._selected_folder_path_prefix(),
                collection_id=self._selected_collection_id(),
                limit=self.page_size,
                offset=0,
                sort_key=self._database_sort_key(),
                sort_desc=self.current_sort_desc,
            )
            self.current_offset = 0
            self.grid_view.set_images(images)
            self._set_result_status(f"关键词结果：{len(images)}")
            return

        if self.current_result_mode == "inspiration" and self.current_inspiration_terms:
            self._run_inspiration_search(
                self.current_inspiration_project_id or 0,
                self.current_inspiration_terms,
            )
            return

        if self.current_result_mode == "temp_project":
            images = self._sort_images(self._apply_sidebar_filters(self.current_temp_project_images))
            self.grid_view.set_images(images, badges_by_image_id=self.current_temp_project_badges)
            project = (
                self.store.get_temporary_project(self.current_temp_project_id)
                if self.current_temp_project_id is not None
                else None
            )
            name = project.name if project is not None else "灵感暂存"
            suffix = f" ｜ {project.summary}" if project is not None and project.summary else ""
            self._set_result_status(f"灵感暂存：{name} ｜ {len(images)} 张{suffix}")
            return

        self._reload_images()

    def _set_result_status(self, message: str) -> None:
        self.result_state_label.setText(message)
        self.statusBar().showMessage(message)

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
        if self.current_language == "en":
            self.save_temp_project_button.setText(
                f"Save {count} Selected" if count else "Save Selected"
            )
        else:
            self.save_temp_project_button.setText(
                f"暂存选中 {count} 张" if count else "暂存选中图片"
            )

    def _save_selected_images_as_temporary_project(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        default_name = self._suggest_temporary_project_name(images)
        name, ok = QInputDialog.getText(
            self,
            "保存为灵感暂存",
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
        )
        intent_labels, intent_queries = self._temporary_project_intents_for_images(images)
        if intent_labels or intent_queries:
            self.store.add_images_to_temporary_project(
                project_id,
                [image.id for image in images],
                intent_labels=intent_labels,
                intent_queries=intent_queries,
            )
        self._refresh_temporary_projects(select_project_id=project_id)
        project = self.store.get_temporary_project(project_id)
        project_name = project.name if project is not None else clean_name
        self.statusBar().showMessage(f"已暂存 {len(images)} 张到“{project_name}”")
        self._suggest_temporary_project_details(
            project_id=project_id,
            images=images,
            can_rename=clean_name == default_name,
        )

    def _add_selection_to_temporary_project(self, project_id: int) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        project = self.store.get_temporary_project(project_id)
        if project is None:
            self._refresh_temporary_projects()
            self.statusBar().showMessage("该灵感暂存已不存在")
            return
        intent_labels, intent_queries = self._temporary_project_intents_for_images(images)
        self.store.add_images_to_temporary_project(
            project_id,
            [image.id for image in images],
            intent_labels=intent_labels,
            intent_queries=intent_queries,
        )
        self._refresh_temporary_projects()
        if self.current_result_mode == "temp_project" and self.current_temp_project_id == project_id:
            self._load_temporary_project(project_id)
        self.statusBar().showMessage(f"已加入 {len(images)} 张到“{project.name}”")

    def _remove_selection_from_current_temporary_project(self) -> None:
        if self.current_result_mode != "temp_project" or self.current_temp_project_id is None:
            self.statusBar().showMessage("当前不在灵感暂存结果中")
            return
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        project_id = self.current_temp_project_id
        project = self.store.get_temporary_project(project_id)
        project_name = project.name if project is not None else "灵感暂存"
        removed = self.store.remove_images_from_temporary_project(
            project_id,
            [image.id for image in images],
        )
        self._refresh_temporary_projects(select_project_id=project_id)
        if self.store.get_temporary_project(project_id) is not None:
            self._load_temporary_project(project_id)
        else:
            self._reload_images()
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

        threading.Thread(target=run, daemon=True).start()

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
            )
            self.store.add_images_to_temporary_project(
                project_id,
                group.image_ids,
                intent_labels={image_id: suggestion.name for image_id in group.image_ids},
            )
            created += 1
        self._refresh_temporary_projects()
        if error_message:
            self.statusBar().showMessage(f"已创建 {created} 个 AI 分组；命名失败，使用备用名称：{error_message}")
        else:
            self.statusBar().showMessage(f"已创建 {created} 个 AI 分组暂存项目")

    def _confirm_reference_group_projects(self, group_pairs: list[tuple[ReferenceGroup, object]], error_message: str) -> bool:
        preview = "\n".join(self._reference_group_preview_lines(group_pairs))
        message = f"将创建 {len(group_pairs)} 个灵感暂存项目：\n\n{preview}\n\n继续？"
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

        threading.Thread(target=run, daemon=True).start()

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
            select_project_id=int(project_id) if self.current_temp_project_id == int(project_id) else None
        )
        if self.current_temp_project_id == int(project_id):
            self._load_temporary_project(int(project_id))
        if updated is not None:
            self.statusBar().showMessage(f"AI 已更新灵感暂存：{updated.name}")

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
            parent=self,
        )
        dialog.imageChanged.connect(self._sync_preview_selection)
        dialog.favoriteChanged.connect(self._handle_preview_favorite_changed)
        dialog.feedbackSaved.connect(self._handle_preview_feedback_saved)
        dialog.exec()
        current = dialog.current_image()
        if current is not None:
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

        menu = QMenu(self)
        preview_action = menu.addAction("快速预览")
        find_similar_action = menu.addAction("查找相似图片")
        open_action = menu.addAction("打开源文件")
        reveal_action = menu.addAction("在 Finder 中显示")
        copy_action = menu.addAction("复制路径")
        has_single_context = context_image is not None and len(selected_images) <= 1
        preview_action.setEnabled(has_single_context)
        find_similar_action.setEnabled(
            has_single_context
            and self._similar_image_blocking_reason(context_image, check_vector=False) is None
        )
        open_action.setEnabled(has_single_context)
        reveal_action.setEnabled(has_single_context)
        copy_action.setEnabled(has_single_context)

        menu.addSeparator()
        favorite_action = menu.addAction(f"收藏选中 {len(selected_images)} 张")
        unfavorite_action = menu.addAction("取消收藏")
        add_tags_action = menu.addAction("批量添加标签")
        clear_tags_action = menu.addAction("清除选中图片标签")
        save_temp_action = menu.addAction("暂存选中图片")
        group_selection_action = menu.addAction("AI 分组选中图片")
        temp_project_menu = menu.addMenu("加入已有灵感暂存")
        temp_project_actions: dict[object, int] = {}
        for project in self.store.list_temporary_projects():
            project_action = temp_project_menu.addAction(f"{project.name} ({project.image_count})")
            temp_project_actions[project_action] = project.id
        if not temp_project_actions:
            empty_action = temp_project_menu.addAction("没有可用暂存项目")
            empty_action.setEnabled(False)
        remove_from_temp_action = menu.addAction("从当前灵感暂存移除")
        add_to_collection_action = menu.addAction("添加到文件夹")
        move_to_collection_action = menu.addAction("移动到文件夹")
        remove_from_collection_action = menu.addAction("从当前文件夹移出")
        rebuild_thumbnails_action = menu.addAction("重建缩略图")
        remove_index_action = menu.addAction("从图库移除索引")
        has_selection = bool(selected_images)
        for batch_action in [
            favorite_action,
            unfavorite_action,
            add_tags_action,
            clear_tags_action,
            save_temp_action,
            group_selection_action,
            remove_from_temp_action,
            add_to_collection_action,
            move_to_collection_action,
            remove_from_collection_action,
            rebuild_thumbnails_action,
            remove_index_action,
        ]:
            batch_action.setEnabled(has_selection)
        group_selection_action.setEnabled(len(selected_images) >= 4)
        temp_project_menu.setEnabled(has_selection and bool(temp_project_actions))
        remove_from_temp_action.setEnabled(has_selection and self.current_result_mode == "temp_project")
        has_current_collection = self._selected_collection_id() is not None
        move_to_collection_action.setEnabled(has_selection and has_current_collection)
        remove_from_collection_action.setEnabled(has_selection and has_current_collection)

        action = menu.exec(global_position)
        if action == preview_action:
            self._open_image_preview(context_image)
        elif action == find_similar_action:
            self._find_similar_to_image(context_image)
        elif action == open_action:
            self._open_selected_original()
        elif action == reveal_action:
            self._reveal_selected_in_finder()
        elif action == copy_action:
            self._copy_selected_path()
        elif action == favorite_action:
            self._batch_set_favorite(True)
        elif action == unfavorite_action:
            self._batch_set_favorite(False)
        elif action == add_tags_action:
            self._batch_add_tags()
        elif action == clear_tags_action:
            self._batch_clear_tags()
        elif action == save_temp_action:
            self._save_selected_images_as_temporary_project()
        elif action == group_selection_action:
            self._group_selected_images_with_ai()
        elif action in temp_project_actions:
            self._add_selection_to_temporary_project(temp_project_actions[action])
        elif action == remove_from_temp_action:
            self._remove_selection_from_current_temporary_project()
        elif action == add_to_collection_action:
            self._add_selection_to_collection_dialog()
        elif action == move_to_collection_action:
            self._move_selection_to_collection_dialog()
        elif action == remove_from_collection_action:
            self._remove_selection_from_current_collection()
        elif action == rebuild_thumbnails_action:
            self._batch_rebuild_thumbnails()
        elif action == remove_index_action:
            self._batch_remove_from_library()

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
        elif action == actions["import_tree"] and item is not None:
            parent_id = int(collection_id) if collection_id is not None else None
            self._choose_import_folder_for_collection(parent_id, preserve_structure=True)

    def _build_collection_context_menu(self, item: QTreeWidgetItem | None) -> tuple[QMenu, dict[str, object]]:
        collection_id = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        has_collection = collection_id is not None
        menu = QMenu(self)
        actions = {
            "new_root": menu.addAction("新建文件夹"),
            "new_child": menu.addAction("新建子文件夹"),
            "rename": menu.addAction("重命名"),
            "delete": menu.addAction("删除文件夹"),
            "add_selected": menu.addAction("把选中图片加入此文件夹"),
            "import_flat": menu.addAction("导入磁盘文件夹到此文件夹"),
            "import_tree": menu.addAction("按磁盘目录生成子文件夹导入"),
        }
        for key in ["new_child", "rename", "delete", "add_selected", "import_flat"]:
            actions[key].setEnabled(has_collection)
        actions["import_tree"].setEnabled(item is not None)
        return menu, actions

    def _on_grid_image_selected(self, image: ImageItem | None) -> None:
        self.selected_image = image
        self._show_image_details(image)
        self._refresh_temp_project_save_button()

    def _on_grid_selection_changed(self, images: list[ImageItem]) -> None:
        if len(images) > 1:
            self.selected_image = images[-1]
            self._show_multi_selection_details(images)
        self._refresh_temp_project_save_button()

    def _set_detail_controls_enabled(self, enabled: bool) -> None:
        for widget in [
            self.favorite_checkbox,
            self.tags_input,
            self.clear_tags_button,
            self.note_input,
            self.save_detail_button,
            self.play_pause_button,
            self.open_original_button,
            self.reveal_in_finder_button,
            self.copy_path_button,
        ]:
            widget.setEnabled(enabled)

    def _show_multi_selection_details(self, images: list[ImageItem]) -> None:
        count = len(images)
        total_size = sum(image.file_size for image in images)
        ready_count = sum(1 for image in images if image.embedding_status == "ready")
        favorite_count = sum(1 for image in images if image.is_favorite)

        self._stop_video_preview()
        self.preview_stack.setCurrentWidget(self.preview_label)
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText(f"已选择 {count} 张")
        self.file_name_label.setText(f"已选择 {count} 张")
        self._set_path_text("-")
        self.size_label.setText(f"{total_size:,} bytes")
        self.modified_label.setText("-")
        self.embedding_label.setText(f"ready {ready_count} / {count}")
        self.score_label.setText("-")
        self.favorite_checkbox.setChecked(favorite_count == count)
        self.tags_input.clear()
        self.note_input.clear()
        self._refresh_feedback_buttons(None)
        self._set_detail_controls_enabled(False)
        self._set_batch_tag_controls_visible(True)
        self._refresh_batch_tag_panel(images)
        self.statusBar().showMessage(f"已选择 {count} 张")

    def _show_image_details(self, image: ImageItem | None) -> None:
        if image is None:
            self._stop_video_preview()
            self.preview_stack.setCurrentWidget(self.preview_label)
            self.preview_label.setText("未选择图片")
            self.preview_label.setPixmap(QPixmap())
            self.file_name_label.setText("-")
            self._set_path_text("-")
            self.size_label.setText("-")
            self.modified_label.setText("-")
            self.embedding_label.setText("-")
            self.score_label.setText("-")
            self.favorite_checkbox.setChecked(False)
            self.tags_input.clear()
            self.note_input.clear()
            self._refresh_feedback_buttons(None)
            self._set_detail_controls_enabled(False)
            self._set_batch_tag_controls_visible(False)
            return

        self._set_detail_controls_enabled(True)
        self._set_batch_tag_controls_visible(False)
        self.file_name_label.setText(image.file_name)
        self._set_path_text(image.file_path)
        self.size_label.setText(self._format_media_dimensions(image))
        self.modified_label.setText(image.modified_at or "-")
        self.embedding_label.setText(image.embedding_status)
        if self.current_result_mode == "inspiration" and image.id in self.current_inspiration_matches:
            matches = self.current_inspiration_matches[image.id]
            titles = "、".join(match.term_title for match in matches[:3])
            score_text = "-" if image.score is None else f"{image.score:.4f}"
            self.score_label.setText(f"{score_text} | {titles}")
        else:
            self.score_label.setText("-" if image.score is None else f"{image.score:.4f}")
        self.favorite_checkbox.setChecked(image.is_favorite)
        self.tags_input.setText(", ".join(self.store.get_image_tags(image.id)))
        self.note_input.setPlainText(image.note or "")
        self._refresh_feedback_buttons(image)

        if is_supported_video(image.file_path):
            self._show_video_details(image)
            return

        self._stop_video_preview()
        self.preview_stack.setCurrentWidget(self.preview_label)
        self.play_pause_button.setEnabled(False)
        preview_path = image.thumbnail_path if image.thumbnail_path and Path(image.thumbnail_path).exists() else image.file_path
        pixmap = QPixmap(preview_path) if not image.is_missing else QPixmap()
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

    def _show_video_details(self, image: ImageItem) -> None:
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("")
        self.preview_stack.setCurrentWidget(self.video_widget)
        self.play_pause_button.setEnabled(not image.is_missing and Path(image.file_path).exists())
        self.play_pause_button.setText("播放")
        self.size_label.setText(self._format_media_dimensions(image))
        self.embedding_label.setText("无需语义索引")
        self.video_player.stop()
        if image.is_missing or not Path(image.file_path).exists():
            self.preview_stack.setCurrentWidget(self.preview_label)
            self.preview_label.setText("视频文件不存在")
            return
        self.video_player.setSource(QUrl.fromLocalFile(image.file_path))

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
        if self.video_player.source().isEmpty():
            self.video_player.setSource(QUrl.fromLocalFile(image.file_path))
        if self.video_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.video_player.pause()
        else:
            self.video_player.play()

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
        self.video_player.stop()
        self.video_player.setSource(QUrl())
        self.play_pause_button.setText("播放")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "path_label"):
            self._fit_path_label_height()
        selected_images = self.grid_view.selected_images()
        if len(selected_images) > 1:
            self._show_multi_selection_details(selected_images)
        elif self.selected_image is not None:
            if is_supported_video(self.selected_image.file_path):
                return
            self._show_image_details(self.selected_image)

    def _save_current_details(self) -> None:
        if self.selected_image is None:
            return
        image_id = self.selected_image.id
        tags = self._parse_tag_input(self.tags_input.text())
        self.store.update_note(image_id, self.note_input.toPlainText())
        self.store.update_favorite(image_id, self.favorite_checkbox.isChecked())
        self.store.set_image_tags(image_id, tags)
        self._refresh_tags()
        self._refresh_current_results_for_filters()
        refreshed = self.store.get_image(image_id)
        self.selected_image = refreshed
        self._show_image_details(refreshed)
        self.statusBar().showMessage("详情已保存")

    def _clear_selected_tags(self) -> None:
        image = self._selected_grid_image()
        if image is None:
            self.statusBar().showMessage("没有选中图片")
            return
        removed = self.store.clear_tags_for_images([image.id])
        self.tags_input.clear()
        self._refresh_tags()
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
        collection_name = self._collection_name(collection_id) or "文件夹"
        self.statusBar().showMessage(f"导入拖入的文件到“{collection_name}”")
        self.add_folder_button.setEnabled(False)
        self.rescan_button.setEnabled(False)

        def run() -> None:
            try:
                file_paths: list[str] = []
                folder_paths: list[str] = []
                for path in paths:
                    expanded = os.path.abspath(os.path.expanduser(path))
                    if os.path.isdir(expanded):
                        folder_paths.append(expanded)
                    elif os.path.isfile(expanded):
                        file_paths.append(expanded)

                total_scanned = 0
                total_new = 0
                total_changed = 0
                total_assigned = 0
                if file_paths:
                    result = self.scanner.import_files(file_paths)
                    total_scanned += result.scanned_files
                    total_new += result.new_files
                    total_changed += result.changed_files
                    total_assigned += self.store.assign_images_to_collection(
                        list(result.image_ids),
                        collection_id,
                    )
                for folder_path in folder_paths:
                    result = self.scanner.scan_folder(folder_path)
                    total_scanned += result.scanned_files
                    total_new += result.new_files
                    total_changed += result.changed_files
                    total_assigned += self.store.assign_images_to_collection(
                        list(result.image_ids),
                        collection_id,
                    )
                self.events.put((
                    "drop_import_done",
                    (collection_id, collection_name, total_scanned, total_new, total_changed, total_assigned),
                ))
            except Exception as exc:
                self.events.put(("error", f"拖入导入失败：{exc}"))

        threading.Thread(target=run, daemon=True).start()

    def _import_dropped_files_to_selected_collection(self, paths: list[str]) -> None:
        collection_id = self._selected_collection_id()
        if collection_id is None:
            self.statusBar().showMessage("请先选择一个 Eidory 文件夹，再拖入硬盘图片")
            return
        self._import_dropped_files_to_collection(collection_id, paths)

    def _assign_selected_images_to_collection(
        self,
        collection_id: int,
        collection_name: str,
    ) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        inserted = self.store.assign_images_to_collection(
            [image.id for image in images],
            collection_id,
        )
        self._refresh_collections(select_collection_id=collection_id)
        self._refresh_current_results_for_filters()
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
        answer = QMessageBox.question(
            self,
            "从当前文件夹移出",
            f"从“{collection_name}”及其子文件夹移出 {len(images)} 个项目。"
            "不会删除硬盘源文件；如果项目不属于其他 Eidory 文件夹，会从图库索引中移除。继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
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
        self.statusBar().showMessage(
            f"已移出关联 {removed_links} 个；从图库移除索引 {deleted_images} 个，源文件未删除"
        )

    def _choose_import_folder_for_collection(
        self,
        collection_id: int | None,
        *,
        preserve_structure: bool,
    ) -> None:
        title = "选择要按目录结构导入的磁盘文件夹" if preserve_structure else "选择要导入的磁盘文件夹"
        folder = QFileDialog.getExistingDirectory(self, title)
        if folder:
            self._start_import(folder, collection_id, preserve_structure=preserve_structure)

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
        image_ids = [image.id for image in self._selected_grid_images()]
        if not image_ids:
            self.statusBar().showMessage("没有选中图片")
            return
        count = self.store.update_favorites(image_ids, is_favorite)
        self._refresh_current_results_for_filters()
        action = "收藏" if is_favorite else "取消收藏"
        self.statusBar().showMessage(f"已{action} {count} 张")

    def _batch_add_tags_from_panel(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        tags = self._parse_tag_input(self.batch_tags_input.text())
        if not tags:
            self.statusBar().showMessage("没有输入标签")
            return
        inserted = self.store.add_tags_to_images([image.id for image in images], tags)
        self.batch_tags_input.clear()
        self._refresh_tags()
        self._refresh_current_results_for_filters()
        refreshed = self.grid_view.selected_images()
        if len(refreshed) > 1:
            self._show_multi_selection_details(refreshed)
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
        removed = self.store.remove_tags_from_images(
            [image.id for image in images],
            [str(tag_name)],
        )
        self._refresh_tags()
        self._refresh_current_results_for_filters()
        refreshed = self.grid_view.selected_images()
        if len(refreshed) > 1:
            self._show_multi_selection_details(refreshed)
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
        inserted = self.store.add_tags_to_images([image.id for image in images], tags)
        self._refresh_tags()
        self._refresh_current_results_for_filters()
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
        self._refresh_current_results_for_filters()
        self.statusBar().showMessage(f"已清除 {removed} 个标签关联")

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

    def _batch_remove_from_library(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
            return
        answer = QMessageBox.question(
            self,
            "从图库移除索引",
            f"只从 Eidory 移除 {len(images)} 张图片的索引记录，不删除源文件。继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        thumbnail_paths = self.store.remove_images_from_library([image.id for image in images])
        self._delete_thumbnail_files(thumbnail_paths)
        self.vector_index.invalidate()
        self.selected_image = None
        self._refresh_folders()
        self._refresh_tags()
        self._refresh_current_results_for_filters()
        self._refresh_embedding_stats()
        self.statusBar().showMessage(f"已从图库移除 {len(images)} 张，源文件未删除")

    def _batch_rebuild_thumbnails(self) -> None:
        images = self._selected_grid_images()
        if not images:
            self.statusBar().showMessage("没有选中图片")
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
        self._refresh_current_results_for_filters()
        self.statusBar().showMessage(
            f"缩略图重建完成：成功 {rebuilt}，失败 {failed}"
        )

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

    def _start_embedding(self) -> None:
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

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "scan_done":
                self._handle_scan_done(payload)
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
                collection_id, collection_name, scanned, new_files, changed_files, assigned = payload
                self._handle_drop_import_done(
                    collection_id=collection_id,
                    collection_name=collection_name,
                    scanned=scanned,
                    new_files=new_files,
                    changed_files=changed_files,
                    assigned=assigned,
                )
            elif kind == "search_done":
                self.search_button.setEnabled(True)
                revision, result = payload
                if revision != self.semantic_search_revision:
                    continue
                self.current_semantic_images = list(result.images)
                self.current_semantic_searchable_count = result.searchable_count
                self.current_semantic_candidate_limit = result.candidate_limit
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
            elif kind == "inspiration_proposal":
                self.generate_inspiration_button.setEnabled(True)
                self._show_inspiration_proposal(payload)
            elif kind == "inspiration_error":
                self.generate_inspiration_button.setEnabled(True)
                self.search_inspiration_button.setEnabled(bool(self._selected_inspiration_terms()))
                self._show_inspiration_error(payload)
            elif kind == "inspiration_done":
                self.generate_inspiration_button.setEnabled(True)
                self.search_inspiration_button.setEnabled(bool(self._selected_inspiration_terms()))
                revision, project_id, selected_terms, result = payload
                if revision != self.semantic_search_revision:
                    continue
                self._handle_inspiration_done(
                    project_id=project_id,
                    selected_terms=selected_terms,
                    result=result,
                )
            elif kind == "temp_project_suggestion":
                self._apply_temporary_project_suggestion(payload)
            elif kind == "temp_project_suggestion_error":
                self._show_temporary_project_suggestion_error(payload)
            elif kind == "reference_groups_done":
                self._create_reference_group_projects(payload, confirm=True)
            elif kind == "embedding":
                self._handle_embedding_progress(payload)
            elif kind == "error":
                self.search_button.setEnabled(True)
                self.generate_inspiration_button.setEnabled(True)
                self.search_inspiration_button.setEnabled(bool(self._selected_inspiration_terms()))
                self.add_folder_button.setEnabled(True)
                self.rescan_button.setEnabled(True)
                QMessageBox.critical(self, "Eidory", str(payload))

    def _handle_scan_done(self, result: ScanResult) -> None:
        self.add_folder_button.setEnabled(True)
        self.rescan_button.setEnabled(True)
        self._refresh_folders()
        self._refresh_collections()
        self._reload_images()
        self._refresh_embedding_stats()
        self.statusBar().showMessage(
            f"扫描完成：新增 {result.new_files}，变化 {result.changed_files}，丢失 {result.missing_marked}"
        )

    def _handle_import_done(
        self,
        result: ScanResult,
        *,
        collection_id: int | None,
        collection_name: str,
        assigned: int,
        preserve_structure: bool,
    ) -> None:
        self.add_folder_button.setEnabled(True)
        self.rescan_button.setEnabled(True)
        self._refresh_folders()
        self._refresh_collections(select_collection_id=collection_id)
        self._reload_images()
        self._refresh_embedding_stats()
        mode = "按目录结构导入" if preserve_structure else "导入"
        self.statusBar().showMessage(
            f"{mode}完成：{collection_name}，扫描 {result.scanned_files}，"
            f"新增 {result.new_files}，变化 {result.changed_files}，加入 {assigned}"
        )

    def _handle_drop_import_done(
        self,
        *,
        collection_id: int,
        collection_name: str,
        scanned: int,
        new_files: int,
        changed_files: int,
        assigned: int,
    ) -> None:
        self.add_folder_button.setEnabled(True)
        self.rescan_button.setEnabled(True)
        self._refresh_folders()
        self._refresh_collections(select_collection_id=collection_id)
        self._reload_images()
        self._refresh_embedding_stats()
        self.statusBar().showMessage(
            f"拖入导入完成：{collection_name}，扫描 {scanned}，"
            f"新增 {new_files}，变化 {changed_files}，加入 {assigned}"
        )

    def _handle_embedding_progress(self, progress: EmbeddingProgress) -> None:
        self._refresh_embedding_stats()
        if progress.image_id is None:
            self.statusBar().showMessage(progress.message)
            if progress.status in {"idle", "stopped"} and self.current_result_mode not in {"semantic", "color", "search_chain", "inspiration"}:
                self._reload_images()
        else:
            self.statusBar().showMessage(f"{progress.file_name}: {progress.status}")
        if progress.status in {"ready", "failed"} and self.current_result_mode not in {"semantic", "color", "search_chain", "inspiration"}:
            self.embedding_refresh_counter += 1
            if self.embedding_refresh_counter >= 20:
                self.embedding_refresh_counter = 0
                self._reload_images()

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
        if current is None:
            current = self._selected_collection_id()
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
        if current is None:
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
        self.collection_tree.setCurrentItem(selected_item or all_item)
        if select_collection_id is not None:
            self._expand_folder_tree_parents(self.collection_tree.currentItem())
        self.collection_tree.blockSignals(False)

    def _refresh_temporary_projects(self, select_project_id: int | None = None) -> None:
        self.temp_project_list.blockSignals(True)
        self.temp_project_list.clear()
        selected_item: QListWidgetItem | None = None
        for project in self.store.list_temporary_projects():
            item = QListWidgetItem(f"{project.name}    {project.image_count}")
            item.setData(Qt.ItemDataRole.UserRole, project.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, project.name)
            item.setData(Qt.ItemDataRole.UserRole + 2, project.color_hex)
            self._apply_temporary_project_item_color(item, project.color_hex)
            tooltip_parts = [project.name, f"{project.image_count} 张"]
            if project.summary:
                tooltip_parts.append(project.summary)
            item.setToolTip("\n".join(tooltip_parts))
            self.temp_project_list.addItem(item)
            if select_project_id == project.id:
                selected_item = item
        if selected_item is not None:
            self.temp_project_list.setCurrentItem(selected_item)
        self.temp_project_list.blockSignals(False)

    @staticmethod
    def _apply_temporary_project_item_color(item: QListWidgetItem, color_hex: str) -> None:
        color = QColor(color_hex)
        if not color.isValid():
            return
        item.setBackground(QBrush(color))
        item.setForeground(QBrush(QColor("#f4f6fb")))

    def _load_selected_temporary_project(self) -> None:
        item = self.temp_project_list.currentItem()
        if item is None:
            return
        project_id = item.data(Qt.ItemDataRole.UserRole)
        if project_id is None:
            return
        self._load_temporary_project(int(project_id))

    def _load_temporary_project(self, project_id: int) -> None:
        project = self.store.get_temporary_project(project_id)
        if project is None:
            self._refresh_temporary_projects()
            self.statusBar().showMessage("该灵感暂存已不存在")
            return
        image_ids = self.store.temporary_project_image_ids(project_id)
        images = self.store.images_by_ids(image_ids)
        badges = self.store.temporary_project_image_badges(project_id)
        self.semantic_search_revision += 1
        self.search_filters.clear()
        self.current_keyword_query = None
        self.current_semantic_query = None
        self.current_result_mode = "temp_project"
        self.current_temp_project_id = project_id
        self.current_temp_project_images = list(images)
        self.current_temp_project_badges = dict(badges)
        self.current_inspiration_project_id = None
        self.current_inspiration_terms = []
        self.current_inspiration_images = []
        self.current_inspiration_filtered_images = []
        self.current_inspiration_matches = {}
        self.current_chain_images = []
        self.current_chain_filtered_images = []
        self.current_chain_result = SearchChainResult(images=[])
        self.current_offset = 0
        self.load_more_button.setEnabled(False)
        self._refresh_filter_chain_ui()
        self.grid_view.set_images(
            self._sort_images(images),
            selected_image_ids=[],
            badges_by_image_id=badges,
        )
        suffix = f" ｜ {project.summary}" if project.summary else ""
        self._set_result_status(f"灵感暂存：{project.name} ｜ {len(images)} 张{suffix}")
        self.search_diagnostics_label.setText("搜索诊断：-")

    def _show_temporary_project_context_menu(self, position) -> None:
        item = self.temp_project_list.itemAt(position)
        if item is None:
            menu = QMenu(self)
            clear_action = menu.addAction("清空暂存")
            clear_action.setEnabled(bool(self.store.list_temporary_projects()))
            action = menu.exec(self.temp_project_list.viewport().mapToGlobal(position))
            if action == clear_action:
                self._clear_all_temporary_projects()
            return
        self.temp_project_list.setCurrentItem(item)
        project_id = item.data(Qt.ItemDataRole.UserRole)
        project_name = item.data(Qt.ItemDataRole.UserRole + 1)
        if project_id is None:
            return
        menu = QMenu(self)
        open_action = menu.addAction("打开暂存项目")
        edit_action = menu.addAction("编辑名称和摘要")
        ai_details_action = menu.addAction("AI 重新命名和摘要")
        delete_action = menu.addAction("删除暂存项目")
        action = menu.exec(self.temp_project_list.viewport().mapToGlobal(position))
        if action == open_action:
            self._load_temporary_project(int(project_id))
        elif action == edit_action:
            self._edit_temporary_project_details(int(project_id))
        elif action == ai_details_action:
            self._request_temporary_project_ai_details(int(project_id))
        elif action == delete_action:
            self._delete_temporary_project(int(project_id), str(project_name or "暂存项目"))

    def _edit_temporary_project_details(self, project_id: int) -> None:
        project = self.store.get_temporary_project(project_id)
        if project is None:
            self._refresh_temporary_projects()
            self.statusBar().showMessage("该灵感暂存已不存在")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("编辑灵感暂存")
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
            self.statusBar().showMessage("灵感暂存名称不能为空")
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
            self.statusBar().showMessage("该灵感暂存已不存在")
            return
        self._refresh_temporary_projects(select_project_id=project_id)
        if self.current_temp_project_id == project_id:
            self._load_temporary_project(project_id)
        self.statusBar().showMessage(f"已更新灵感暂存：{updated.name}")

    def _request_temporary_project_ai_details(self, project_id: int) -> None:
        project = self.store.get_temporary_project(project_id)
        if project is None:
            self._refresh_temporary_projects()
            self.statusBar().showMessage("该灵感暂存已不存在")
            return
        image_ids = self.store.temporary_project_image_ids(project_id)
        images = self.store.images_by_ids(image_ids)
        if not images:
            self.statusBar().showMessage("该灵感暂存没有图片，无法生成名称和摘要")
            return
        self.statusBar().showMessage(f"正在用 AI 更新“{project.name}”的名称和摘要...")
        self._suggest_temporary_project_details(
            project_id=project_id,
            images=images,
            can_rename=True,
        )

    def _delete_temporary_project(self, project_id: int, project_name: str) -> None:
        answer = QMessageBox.question(
            self,
            "删除灵感暂存",
            f"删除“{project_name}”？这只删除暂存项目，不会删除图片源文件。",
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
            self.statusBar().showMessage(f"已删除灵感暂存：{project_name}")

    def _clear_all_temporary_projects(self, *, confirm: bool = True) -> None:
        projects = self.store.list_temporary_projects()
        if not projects:
            self.statusBar().showMessage("没有可清空的灵感暂存")
            return
        if confirm:
            answer = QMessageBox.question(
                self,
                "清空灵感暂存",
                f"清空全部 {len(projects)} 个灵感暂存项目？这不会删除图片源文件。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        cleared = self.store.clear_temporary_projects()
        was_viewing_temporary_project = self.current_result_mode == "temp_project"
        self.current_temp_project_id = None
        self.current_temp_project_images = []
        self.current_temp_project_badges = {}
        if was_viewing_temporary_project:
            self._reload_images()
        self._refresh_temporary_projects()
        self.statusBar().showMessage(f"已清空 {cleared} 个灵感暂存项目")

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
        previous_ids = set(current_ids)
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
        self.tag_completion_model.setStringList(tag_names)
        self.tag_list.setCurrentItem(selected_item or all_item)
        self.tag_list.blockSignals(False)
        self._pending_tag_restore_ids = None
        self._refresh_tag_action_buttons()
        self._save_selected_tag_filter()
        if previous_ids != set(self._selected_tag_ids()):
            self._refresh_current_results_for_filters()

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
        self._refresh_current_results_for_filters()

    def _save_status_filter(self) -> None:
        item = self.status_list.currentItem()
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else "all"
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
    def _format_score_threshold_label(value: int) -> str:
        if value <= 0:
            return "最低相似度：不限"
        return f"最低相似度：{value / 100.0:.2f}"

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
        item = self.status_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
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
            QMessageBox.warning(self, "Eidory", "该视图预设格式无效。")
            return
        signal_widgets = [
            self.status_list,
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
            self._apply_collection_from_payload(payload.get("collection_id"))
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

    def _apply_collection_from_payload(self, raw_collection_id: object) -> None:
        collection_id = None
        try:
            collection_id = int(raw_collection_id) if raw_collection_id is not None else None
        except (TypeError, ValueError):
            collection_id = None
        self._refresh_collections(select_collection_id=collection_id)

    def _apply_status_from_payload(self, raw_status: object) -> None:
        status = raw_status if raw_status in {"all", "favorite", "unindexed", "missing"} else "all"
        self._set_list_current_by_data(self.status_list, str(status))

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
            images = self._sort_images(self.current_temp_project_images)
            self.grid_view.set_images(images, badges_by_image_id=self.current_temp_project_badges)
            project = (
                self.store.get_temporary_project(self.current_temp_project_id)
                if self.current_temp_project_id is not None
                else None
            )
            name = project.name if project is not None else "灵感暂存"
            suffix = f" ｜ {project.summary}" if project is not None and project.summary else ""
            self._set_result_status(f"灵感暂存：{name} ｜ {len(images)} 张{suffix}")
            return
        self._refresh_current_results_for_filters()

    def _database_sort_key(self) -> str:
        if self.current_sort_key == "score":
            return "default"
        return self.current_sort_key

    def _sort_images(self, images: list[ImageItem]) -> list[ImageItem]:
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
        self.color_swatch_button.setText(color_hex)
        self.color_swatch_button.setStyleSheet(
            f"background:{color_hex}; color:{self._swatch_text_color(self.current_color_rgb)};"
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

    def _update_score_threshold(self) -> None:
        self.score_threshold_label.setText(
            self._format_score_threshold_label(self.score_threshold_slider.value())
        )
        self.store.set_setting("ui.score_threshold", str(self.score_threshold_slider.value()))
        if self.search_filters:
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

    def _score_threshold(self) -> float | None:
        value = self.score_threshold_slider.value()
        if value <= 0:
            return None
        return value / 100.0

    def _apply_semantic_result_filters(self) -> None:
        threshold = self._score_threshold()
        images = self.current_semantic_images
        if threshold is not None:
            images = [
                image
                for image in images
                if image.score is not None and image.score >= threshold
            ]
        self.current_semantic_filtered_images = self._sort_images(self._apply_sidebar_filters(images))

    def _apply_color_result_filters(self) -> None:
        threshold = self._color_score_threshold(self.current_color_images)
        images = self.current_color_images
        if threshold is not None:
            images = [
                image
                for image in images
                if image.score is not None and image.score >= threshold
            ]
        self.current_color_filtered_images = self._sort_images(self._apply_sidebar_filters(images))

    def _set_semantic_result_status(self, images: list[ImageItem]) -> None:
        source_count = len(self.current_semantic_images)
        scope_text = (
            ""
            if self.current_search_scope_count is None
            else f"，叠加范围 {self.current_search_scope_count}"
        )
        if len(images) == source_count:
            self._set_result_status(f"语义结果：{len(images)}{scope_text}")
        else:
            self._set_result_status(f"语义结果：{len(images)} / 原始 {source_count}{scope_text}")

    def _set_color_result_status(self, images: list[ImageItem]) -> None:
        source_count = len(self.current_color_images)
        color_hex = self._format_color_hex(self.current_color_rgb)
        scope_text = (
            ""
            if self.current_search_scope_count is None
            else f"，叠加范围 {self.current_search_scope_count}"
        )
        if len(images) == source_count:
            self._set_result_status(f"颜色结果 {color_hex}：{len(images)}{scope_text}")
        else:
            self._set_result_status(f"颜色结果 {color_hex}：{len(images)} / 原始 {source_count}{scope_text}")

    def _update_search_diagnostics(self, images: list[ImageItem]) -> None:
        if self.current_result_mode != "semantic" or not images:
            self.search_diagnostics_label.setText("搜索诊断：-")
            return

        scores = [image.score for image in images if image.score is not None]
        if not scores:
            self.search_diagnostics_label.setText("搜索诊断：无相似度分数")
            return

        threshold = self._score_threshold()
        threshold_text = "不限" if threshold is None else f"{threshold:.2f}"
        avg_score = sum(scores) / len(scores)
        self.search_diagnostics_label.setText(
            "搜索诊断："
            f"显示 {len(images)}，可搜索 {self.current_semantic_searchable_count}，"
            f"候选上限 {self.current_semantic_candidate_limit}，"
            f"最高 {max(scores):.3f}，最低 {min(scores):.3f}，"
            f"平均 {avg_score:.3f}，阈值 {threshold_text}"
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

    def _color_score_threshold(self, images: list[ImageItem]) -> float | None:
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

    def _apply_sidebar_filters(self, images: list[ImageItem]) -> list[ImageItem]:
        folder_path_prefix = self._selected_folder_path_prefix()
        collection_id = self._selected_collection_id()
        collection_image_ids = (
            self.store.image_ids_for_collection(collection_id)
            if collection_id is not None
            else None
        )
        status = self._selected_status_filter()
        tag_names = self._selected_tag_names()
        tag_match_mode = self._selected_tag_match_mode()

        filtered: list[ImageItem] = []
        for image in images:
            if folder_path_prefix and not self._path_is_in_folder(image.file_path, folder_path_prefix):
                continue
            if collection_image_ids is not None and image.id not in collection_image_ids:
                continue
            if status == "favorite" and not image.is_favorite:
                continue
            if status == "unindexed" and image.embedding_status == "ready":
                continue
            if status == "missing" and not image.is_missing:
                continue
            if tag_names:
                image_tags = set(self.store.get_image_tags(image.id))
                selected_tags = set(tag_names)
                if tag_match_mode == "all":
                    if not selected_tags.issubset(image_tags):
                        continue
                elif image_tags.isdisjoint(selected_tags):
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
