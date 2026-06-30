from __future__ import annotations

import os
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible, QColor, QKeySequence, QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QListWidgetItem, QMessageBox, QPushButton, QTreeWidget, QTreeWidgetItem
from PIL import Image

from eidory.config import AppPaths
from eidory.core.ai_vision import AIVisionAnalysis, AI_VISION_PROMPT_VERSION
from eidory.core.duplicate_detection import DuplicateGroup, DuplicateMember
from eidory.core.embedding_worker import EmbeddingProgress
from eidory.core.inspiration import InspirationMatch, InspirationTerm
from eidory.core.llm_provider import GroupNameSuggestion, ProjectSuggestion, SearchPlanFilter
from eidory.core.metadata_store import MetadataStore, TEMPORARY_PROJECT_COLORS
from eidory.core.reference_grouping import ReferenceGroup
from eidory.core.scanner import ScanResult
from eidory.core.search_service import SemanticSearchResult
from eidory.core.search_filters import (
    SearchFilter,
    last_score_filter_kind,
    search_filter_from_payload,
    search_filter_to_payload,
)
from eidory.models import ImageItem
from eidory.ui.main_window import (
    COLLECTION_SEARCH_EXCLUDED_DIRECT_ROLE,
    COLLECTION_SEARCH_EXCLUDED_MATCH_ROLE,
    CREATIVE_NODE_HAS_NOTE_ROLE,
    DuplicateResultsDialog,
    EqualWidthTabBar,
    FolderTreeItemDelegate,
    FOLDER_SEARCH_EXCLUDED_DIRECT_ROLE,
    LEFT_SIDEBAR_WIDTH,
    MainWindow,
    PROJECT_LIST_ID_ROLE,
    PROJECT_LIST_KIND_ROLE,
    PROJECT_LIST_SECTION_ID_ROLE,
    RIGHT_SIDEBAR_WIDTH,
    SAVED_VIEW_DELETE_ACTION,
    SAVED_VIEW_RENAME_ACTION,
    SAVED_VIEW_SAVE_ACTION,
    SIDEBAR_COLLAPSE_THRESHOLD,
    SIDEBAR_COUNT_COLUMN_WIDTH,
    TAG_GROUP_HEADER_ROLE,
    TagPickerDialog,
    TOP_TOOL_BUTTON_MIN_WIDTH,
    TOP_TOOL_BUTTON_SPACING,
    TOOL_BUTTON_MIN_WIDTH,
)


class MainWindowContextMenuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_search_defaults_to_semantic_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            self.assertTrue(window.semantic_mode_button.isChecked())
            self.assertEqual(window._selected_search_mode(), "semantic")
            window.close()

    def test_advanced_search_tools_are_collapsed_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            self.assertTrue(window.advanced_search_widget.isHidden())
            self.assertEqual(window.advanced_search_toggle_button.text(), "▾")
            self.assertIs(window.saved_view_combo.parentWidget(), window.advanced_search_widget)
            window.advanced_search_toggle_button.click()
            self.app.processEvents()
            self.assertFalse(window.advanced_search_widget.isHidden())
            self.assertEqual(window.advanced_search_toggle_button.text(), "▴")
            self.assertTrue(window.saved_view_combo.isVisible())
            self.assertEqual(window.saved_view_combo.currentText(), "未选择预设")
            self.assertEqual(window.saved_view_combo.contextMenuPolicy(), Qt.ContextMenuPolicy.NoContextMenu)
            self.assertGreaterEqual(window.saved_view_combo.count(), 2)
            self.assertEqual(window.saved_view_combo.itemData(1), SAVED_VIEW_SAVE_ACTION)
            self.assertEqual(window.saved_view_combo.itemText(1), "保存当前筛选为预设")
            self.assertEqual(window.saved_view_combo.findData(SAVED_VIEW_RENAME_ACTION), -1)
            self.assertEqual(window.saved_view_combo.findData(SAVED_VIEW_DELETE_ACTION), -1)
            window.advanced_search_toggle_button.click()
            self.app.processEvents()
            self.assertTrue(window.advanced_search_widget.isHidden())
            self.assertEqual(window.advanced_search_toggle_button.text(), "▾")
            window.close()

    def test_saved_view_combo_save_action_is_available_from_left_click_menu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            save_index = window.saved_view_combo.findData(SAVED_VIEW_SAVE_ACTION)
            self.assertGreaterEqual(save_index, 0)
            window.saved_view_combo.setCurrentIndex(save_index)
            with patch.object(window, "_save_current_view", return_value=None) as save_current_view:
                window._handle_saved_view_combo_activated(save_index)
            save_current_view.assert_called_once()
            self.assertEqual(window.saved_view_combo.currentText(), "未选择预设")
            window.close()

    def test_saved_view_combo_management_actions_are_left_click_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            saved_view_id = store.upsert_saved_view("常用筛选", json.dumps({"version": 1}))
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window._refresh_saved_views(select_saved_view_id=saved_view_id)
            self.assertEqual(window.saved_view_combo.findData(SAVED_VIEW_RENAME_ACTION), 2)
            self.assertEqual(window.saved_view_combo.findData(SAVED_VIEW_DELETE_ACTION), 3)
            self.assertEqual(window.saved_view_combo.contextMenuPolicy(), Qt.ContextMenuPolicy.NoContextMenu)
            rename_index = window.saved_view_combo.findData(SAVED_VIEW_RENAME_ACTION)
            window.saved_view_combo.setCurrentIndex(rename_index)
            with patch.object(window, "_rename_selected_saved_view") as rename_saved_view:
                window._handle_saved_view_combo_activated(rename_index)
            rename_saved_view.assert_called_once()
            self.assertEqual(window._selected_saved_view_id(), saved_view_id)
            window.close()

    def test_main_window_disables_qt_accessibility_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            QAccessible.setActive(True)

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            self.assertFalse(QAccessible.isActive())
            window.close()

    def test_sqlite_utility_connection_uses_store_busy_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "eidory.sqlite3"

            with MainWindow._connect_sqlite_database(database_path) as conn:
                busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()

            self.assertEqual(busy_timeout[0], MetadataStore._busy_timeout_ms)

    def test_database_maintenance_blocks_background_tasks_and_file_watcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            watched_dir = Path(tmp) / "watched"
            watched_dir.mkdir()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            self.assertTrue(window.file_watcher.addPath(str(watched_dir)))
            window._pending_watch_scan_roots.add(str(watched_dir))
            window.watch_scan_timer.start(10_000)
            ran: list[bool] = []

            with window._database_maintenance("test"):
                self.assertTrue(window._database_maintenance_active)
                self.assertFalse(window.watch_scan_timer.isActive())
                self.assertEqual(window.file_watcher.directories(), [])
                self.assertFalse(window._start_background_task(lambda: ran.append(True)))

            self.assertEqual(ran, [])
            window.close()

    def test_database_maintenance_aborts_when_index_workers_do_not_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            entered = False

            with (
                patch.object(window, "_stop_index_workers_for_maintenance", return_value=False),
                patch.object(window, "_wait_for_background_tasks") as wait_for_background,
            ):
                with self.assertRaisesRegex(RuntimeError, "索引 worker"):
                    with window._database_maintenance("数据库恢复"):
                        entered = True

            self.assertFalse(entered)
            wait_for_background.assert_not_called()
            self.assertFalse(window._database_maintenance_active)
            window.close()

    def test_database_maintenance_aborts_when_background_tasks_do_not_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            entered = False

            with (
                patch.object(window, "_stop_index_workers_for_maintenance", return_value=True),
                patch.object(window, "_wait_for_background_tasks", return_value=False),
            ):
                with self.assertRaisesRegex(RuntimeError, "后台任务"):
                    with window._database_maintenance("数据库恢复"):
                        entered = True

            self.assertFalse(entered)
            self.assertFalse(window._database_maintenance_active)
            window.close()

    def test_database_backup_maintenance_restarts_previously_running_index_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window.embedding_worker = SimpleNamespace(is_alive=lambda: True)
            window.ai_vision_worker = SimpleNamespace(is_alive=lambda: True)

            with (
                patch.object(window, "_stop_index_workers_for_maintenance", return_value=True),
                patch.object(window, "_wait_for_background_tasks", return_value=True),
                patch.object(window, "_start_embedding") as start_embedding,
                patch.object(window, "_start_ai_vision") as start_ai_vision,
            ):
                with window._database_maintenance("数据库备份", restart_index_workers=True):
                    pass

            start_embedding.assert_called_once_with()
            start_ai_vision.assert_called_once_with()
            window.embedding_worker = None
            window.ai_vision_worker = None
            window.close()

    def test_ai_vision_index_panel_uses_folder_list_inclusion_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            included = store.create_collection("需要识别")
            excluded = store.create_collection("旧排除项")
            store.set_ai_vision_collection_rule(included, mode="include")
            store.set_ai_vision_collection_rule(excluded, mode="exclude")

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._refresh_ai_vision_stats()

            self.assertEqual(window.ai_vision_rule_tree.columnCount(), 5)
            self.assertEqual(window.ai_vision_rule_tree.headerItem().text(0), "文件夹")
            self.assertEqual(window.add_ai_vision_include_rule_button.text(), "添加选中文件夹")
            self.assertEqual(window.remove_ai_vision_rule_button.text(), "移除选中文件夹")
            self.assertFalse(hasattr(window, "add_ai_vision_exclude_rule_button"))
            self.assertEqual(window.ai_vision_rule_tree.topLevelItemCount(), 1)
            self.assertEqual(window.ai_vision_rule_tree.topLevelItem(0).text(0), "需要识别")
            window.close()

    def test_result_status_reuses_short_lived_count_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._invalidate_result_status_count_cache()

            with (
                patch.object(store, "count_images", wraps=store.count_images) as count_images,
                patch.object(store, "count_missing_images", wraps=store.count_missing_images) as count_missing,
            ):
                window._set_result_status("第一次刷新")
                window._set_result_status("第二次刷新")

            self.assertEqual(count_images.call_count, 1)
            self.assertEqual(count_missing.call_count, 1)
            window.close()

    def test_refresh_collections_reuses_virtual_filter_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            with patch.object(store, "virtual_image_filter_counts", wraps=store.virtual_image_filter_counts) as counts:
                window._refresh_collections()

            self.assertEqual(counts.call_count, 1)
            window.close()

    def test_refresh_tags_reuses_tag_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=100,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            store.set_image_tags(image_id, ["室内"])
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            with patch.object(store, "list_tags_with_counts", wraps=store.list_tags_with_counts) as tag_counts:
                window._refresh_tags()

            self.assertEqual(tag_counts.call_count, 1)
            window.close()

    def test_detail_preview_reuses_decoded_pixmap_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            image_path = Path(tmp) / "first.jpg"
            preview = QPixmap(80, 50)
            preview.fill(QColor("#223344"))
            self.assertTrue(preview.save(str(image_path)))
            image = self._image(1, file_path=str(image_path), width=80, height=50)

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            decoded = QPixmap(40, 25)
            decoded.fill(QColor("#556677"))
            with patch("eidory.ui.main_window._load_scaled_qt_pixmap", return_value=decoded) as load_pixmap:
                first = window._load_detail_preview_pixmap(image)
                second = window._load_detail_preview_pixmap(image)

            self.assertFalse(first.isNull())
            self.assertFalse(second.isNull())
            self.assertEqual(load_pixmap.call_count, 1)
            window.close()

    def test_creative_selection_panel_reuses_current_node_image_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            image = store.get_image(image_id)
            self.assertIsNotNone(image)
            project_id = store.create_creative_project(
                title="测试项目",
                brief="测试",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            node_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(node_id)
            store.add_images_to_creative_node(int(node_id), [image_id], intent_label="世界观")

            window = MainWindow(paths=paths, store=store)
            window.current_creative_project_id = project_id
            window.current_creative_node_id = int(node_id)
            window.show()
            self.app.processEvents()
            with patch.object(store, "creative_node_image_ids", wraps=store.creative_node_image_ids) as image_ids:
                window._refresh_creative_selection_panel([image])
                window._refresh_creative_selection_panel([image])

            self.assertEqual(image_ids.call_count, 1)
            window.close()

    def test_database_restore_maintenance_does_not_restart_stopped_index_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window.embedding_worker = SimpleNamespace(is_alive=lambda: True)
            window.ai_vision_worker = SimpleNamespace(is_alive=lambda: True)

            with (
                patch.object(window, "_stop_index_workers_for_maintenance", return_value=True),
                patch.object(window, "_wait_for_background_tasks", return_value=True),
                patch.object(window, "_start_embedding") as start_embedding,
                patch.object(window, "_start_ai_vision") as start_ai_vision,
            ):
                with window._database_maintenance("数据库恢复"):
                    pass

            start_embedding.assert_not_called()
            start_ai_vision.assert_not_called()
            window.embedding_worker = None
            window.ai_vision_worker = None
            window.close()

    def test_database_backup_does_not_restart_index_workers_if_stop_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window.embedding_worker = SimpleNamespace(is_alive=lambda: True)
            window.ai_vision_worker = SimpleNamespace(is_alive=lambda: True)

            with (
                patch.object(window, "_stop_index_workers_for_maintenance", return_value=False),
                patch.object(window, "_start_embedding") as start_embedding,
                patch.object(window, "_start_ai_vision") as start_ai_vision,
            ):
                with self.assertRaisesRegex(RuntimeError, "索引 worker"):
                    with window._database_maintenance("数据库备份", restart_index_workers=True):
                        pass

            start_embedding.assert_not_called()
            start_ai_vision.assert_not_called()
            window.embedding_worker = None
            window.ai_vision_worker = None
            window.close()

    def test_background_task_rejection_runs_ui_rollback_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._database_maintenance_active = True
            calls: list[str] = []

            started = window._start_background_task(
                lambda: calls.append("ran"),
                on_rejected=lambda: calls.append("rollback"),
            )

            self.assertFalse(started)
            self.assertEqual(calls, ["rollback"])
            window._database_maintenance_active = False
            window.close()

    def test_sidebars_have_fixed_visible_widths_and_can_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.resize(1500, 900)
            window.show()
            self.app.processEvents()

            self.assertEqual(window.root_splitter.widget(0).maximumWidth(), LEFT_SIDEBAR_WIDTH)
            self.assertEqual(window.root_splitter.widget(2).maximumWidth(), RIGHT_SIDEBAR_WIDTH)
            self.assertGreaterEqual(window.root_splitter.sizes()[0], 300)
            self.assertEqual(window.collection_tree.columnWidth(1), SIDEBAR_COUNT_COLUMN_WIDTH)
            self.assertGreaterEqual(window.collection_tree.columnWidth(0), 210)

            window.resize(2200, 1000)
            self.app.processEvents()
            window._enforce_fixed_sidebar_widths()
            self.assertEqual(window.root_splitter.sizes()[0], LEFT_SIDEBAR_WIDTH)
            self.assertGreaterEqual(window.collection_tree.columnWidth(0), 210)

            window.root_splitter.setSizes([360, 700, 640])
            window._enforce_fixed_sidebar_widths()
            left, _center, right = window.root_splitter.sizes()
            self.assertEqual(left, LEFT_SIDEBAR_WIDTH)
            self.assertEqual(right, RIGHT_SIDEBAR_WIDTH)

            window.root_splitter.setSizes([10, 1200, 20])
            window._enforce_fixed_sidebar_widths(10, 1)
            window._enforce_fixed_sidebar_widths(sum(window.root_splitter.sizes()) - 20, 2)
            left, _center, right = window.root_splitter.sizes()
            self.assertEqual(left, 0)
            self.assertEqual(right, 0)
            window.close()

    def test_tab_focus_mode_collapses_sidebars_in_gallery_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.resize(1500, 900)
            window.show()
            self.app.processEvents()
            window._show_gallery_view()
            window.root_splitter.setSizes([LEFT_SIDEBAR_WIDTH, 900, RIGHT_SIDEBAR_WIDTH])
            self.app.processEvents()

            self.assertTrue(window.board_focus_shortcut.isEnabled())
            self.assertIs(window.center_result_stack.currentWidget(), window.grid_view)

            window._toggle_board_focus_mode()
            self.app.processEvents()
            left, _center, right = window.root_splitter.sizes()
            self.assertEqual(left, 0)
            self.assertEqual(right, 0)
            self.assertIs(window.center_result_stack.currentWidget(), window.grid_view)
            self.assertTrue(window.search_input.isVisible())

            window._toggle_board_focus_mode()
            self.app.processEvents()
            left, _center, right = window.root_splitter.sizes()
            self.assertEqual(left, LEFT_SIDEBAR_WIDTH)
            self.assertEqual(right, RIGHT_SIDEBAR_WIDTH)
            self.assertIs(window.center_result_stack.currentWidget(), window.grid_view)
            window.close()

    def test_tab_focus_mode_keeps_board_toolbar_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.resize(1500, 900)
            window.show()
            self.app.processEvents()
            window.center_result_stack.setCurrentWidget(window.project_board_view)
            window._set_board_toolbar_visible(True)
            window.root_splitter.setSizes([LEFT_SIDEBAR_WIDTH, 900, RIGHT_SIDEBAR_WIDTH])
            self.app.processEvents()

            window._toggle_board_focus_mode()
            self.app.processEvents()

            left, _center, right = window.root_splitter.sizes()
            self.assertEqual(left, 0)
            self.assertEqual(right, 0)
            self.assertFalse(window.search_input.isVisible())
            self.assertTrue(window.show_gallery_button.isVisible())
            self.assertTrue(window.show_project_board_button.isVisible())
            self.assertTrue(window.board_hide_selected_button.isVisible())
            self.assertTrue(window.board_fit_all_button.isVisible())
            self.assertTrue(window.board_grayscale_button.isVisible())
            window.close()

    def test_search_operation_defaults_to_replace_until_results_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            self.assertTrue(window.search_replace_results_button.isChecked())
            self.assertFalse(window.search_within_results_button.isEnabled())
            self.assertFalse(window.search_merge_results_button.isEnabled())

            window.search_within_results_button.setChecked(True)
            self.assertEqual(window._selected_search_operation_mode(), "replace")

            window.current_result_mode = "keyword"
            window._refresh_search_operation_controls()
            self.assertTrue(window.search_within_results_button.isEnabled())
            self.assertTrue(window.search_merge_results_button.isEnabled())
            window.close()

    def test_new_filter_kind_prompts_for_search_operation_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window.current_result_mode = "search_chain"
            window.grid_view.set_images([self._image(1), self._image(2)])
            window.search_filters = [SearchFilter("color", (240, 152, 196))]

            with (
                patch.object(window, "_prompt_search_operation_choice", return_value="merge") as prompt,
                patch.object(window, "_execute_search_chain") as execute_search_chain,
            ):
                window._start_search_with_filter(SearchFilter("semantic", "水"))

            prompt.assert_called_once_with(SearchFilter("semantic", "水"))
            self.assertTrue(window.search_merge_results_button.isChecked())
            self.assertEqual(window.search_filters, [SearchFilter("semantic", "水")])
            execute_search_chain.assert_called_once()
            self.assertEqual(execute_search_chain.call_args.kwargs["operation_mode"], "merge")
            self.assertIn("operation_context", execute_search_chain.call_args.kwargs)
            window.close()

    def test_same_filter_kind_prompts_for_search_operation_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window.current_result_mode = "search_chain"
            window.grid_view.set_images([self._image(1)])
            window.search_filters = [SearchFilter("color", (240, 152, 196))]

            with (
                patch.object(window, "_prompt_search_operation_choice", return_value="replace") as prompt,
                patch.object(window, "_execute_search_chain") as execute_search_chain,
            ):
                window._start_search_with_filter(SearchFilter("color", (255, 0, 0)))

            prompt.assert_called_once_with(SearchFilter("color", (255, 0, 0)))
            self.assertEqual(window.search_filters, [SearchFilter("color", (255, 0, 0))])
            execute_search_chain.assert_called_once()
            self.assertEqual(execute_search_chain.call_args.kwargs["operation_mode"], "replace")
            self.assertIn("operation_context", execute_search_chain.call_args.kwargs)
            window.close()

    def test_typing_query_after_color_search_switches_to_semantic_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window.color_mode_button.setChecked(True)
            window.search_input.setText("旅客")
            window._switch_to_semantic_search_for_typed_query("旅客")

            self.assertTrue(window.semantic_mode_button.isChecked())
            self.assertEqual(window._search_filter_from_controls(), SearchFilter("semantic", "旅客"))
            window.close()

    def test_color_picker_runs_color_search_even_when_query_text_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window.search_input.setText("旅客")
            with (
                patch("eidory.ui.main_window.QColorDialog.getColor", return_value=QColor(1, 2, 3)),
                patch.object(window, "_start_search_with_filter") as start_search,
            ):
                window._choose_search_color()

            self.assertTrue(window.color_mode_button.isChecked())
            start_search.assert_called_once_with(SearchFilter("color", (1, 2, 3)))
            window.close()

    def test_cancel_search_operation_prompt_keeps_current_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window.current_result_mode = "search_chain"
            window.grid_view.set_images([self._image(1), self._image(2)])
            original_filters = [SearchFilter("color", (240, 152, 196))]
            window.search_filters = list(original_filters)

            with (
                patch.object(window, "_prompt_search_operation_choice", return_value=None) as prompt,
                patch.object(window, "_execute_search_chain") as execute_search_chain,
            ):
                window._start_search_with_filter(SearchFilter("semantic", "水"))

            prompt.assert_called_once_with(SearchFilter("semantic", "水"))
            self.assertEqual(window.search_filters, original_filters)
            execute_search_chain.assert_not_called()
            window.close()

    def test_scan_refresh_preserves_active_result_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            for mode in [
                "semantic",
                "color",
                "keyword",
                "inspiration",
                "temp_project",
                "duplicate_group",
            ]:
                window.current_result_mode = mode
                window.search_filters.clear()
                with patch.object(window, "_reload_images") as reload_images, patch.object(
                    window,
                    "_refresh_current_results_for_filters",
                ) as refresh_results:
                    window._refresh_after_scan_database_change()
                    self.assertFalse(reload_images.called, mode)
                    self.assertTrue(refresh_results.called, mode)

            window.current_result_mode = "library"
            window.search_filters = [object()]  # type: ignore[list-item]
            with patch.object(window, "_reload_images") as reload_images, patch.object(
                window,
                "_refresh_current_results_for_filters",
            ) as refresh_results:
                window._refresh_after_scan_database_change()
                self.assertFalse(reload_images.called)
                self.assertTrue(refresh_results.called)

            window.current_result_mode = "library"
            window.search_filters.clear()
            with patch.object(window, "_reload_images") as reload_images, patch.object(
                window,
                "_refresh_current_results_for_filters",
            ) as refresh_results:
                window._refresh_after_scan_database_change()
                self.assertTrue(reload_images.called)
                self.assertFalse(refresh_results.called)

            window.close()

    def test_empty_side_panels_explain_next_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            self.assertIn("先在图片墙选择", window.tag_panel_selection_label.text())
            self.assertIn("顶栏“标签”用于筛选", window.tag_panel_selection_label.text())
            self.assertIn("选择图片后", window.collection_detail_help_label.text())
            self.assertIn("AI 标签", window.collection_detail_help_label.text())
            self.assertEqual(window.grid_view.accessibleName(), "Image wall")
            self.assertEqual(window.search_input.accessibleName(), "Search text")
            window.close()

    def test_result_status_uses_unified_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "image.jpg"),
                file_size=100,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            status = window.result_state_label.text()
            self.assertIn("总数 1", status)
            self.assertIn("当前范围 1", status)
            self.assertIn("已加载 1", status)
            self.assertIn("缺失 0", status)
            self.assertIn("结果 -", status)
            window.close()

    def test_near_duplicate_import_prompts_for_same_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            library = Path(tmp) / "library"
            library.mkdir()
            image_path = library / "existing.jpg"
            Image.new("RGB", (120, 80), color="white").save(image_path)

            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(library))
            stat = image_path.stat()
            store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=stat.st_size,
                width=120,
                height=80,
                created_time_ns=None,
                modified_time_ns=stat.st_mtime_ns,
            )
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            with (
                patch.object(window, "_ask_near_duplicate_decision", return_value=("skip", None)) as ask,
                patch.object(
                    window,
                    "_build_near_duplicate_hash_records",
                    side_effect=AssertionError("same-path import should not scan library hashes"),
                ),
            ):
                accepted, skipped, replaced, skipped_count = window._resolve_near_duplicate_import_paths(
                    [str(image_path)],
                    include_same_path=True,
                )

            self.assertEqual(accepted, [])
            self.assertEqual(skipped, {str(image_path)})
            self.assertEqual(replaced, [])
            self.assertEqual(skipped_count, 1)
            ask.assert_called_once()
            self.assertEqual(ask.call_args.args[1][0].image.file_path, str(image_path))
            window.close()

    def test_near_duplicate_import_uses_metadata_candidates_for_different_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            library = Path(tmp) / "library"
            incoming = Path(tmp) / "incoming"
            library.mkdir()
            incoming.mkdir()
            image_path = library / "existing.jpg"
            import_path = incoming / "existing-copy.jpg"
            Image.new("RGB", (160, 90), color="white").save(image_path)
            shutil.copy2(image_path, import_path)

            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(library))
            stat = image_path.stat()
            store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=stat.st_size,
                width=160,
                height=90,
                created_time_ns=None,
                modified_time_ns=stat.st_mtime_ns,
            )
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            with (
                patch.object(window, "_ask_near_duplicate_decision", return_value=("skip", None)) as ask,
                patch.object(
                    window,
                    "_build_near_duplicate_hash_records",
                    side_effect=AssertionError("near import should not build full-library hashes"),
                ),
            ):
                accepted, skipped, replaced, skipped_count = window._resolve_near_duplicate_import_paths(
                    [str(import_path)],
                    include_same_path=True,
                )

            self.assertEqual(accepted, [])
            self.assertEqual(skipped, {str(import_path)})
            self.assertEqual(replaced, [])
            self.assertEqual(skipped_count, 1)
            ask.assert_called_once()
            self.assertEqual(ask.call_args.args[1][0].image.file_path, str(image_path))
            window.close()

    def test_file_import_near_duplicate_check_does_not_block_ui_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            library = Path(tmp) / "library"
            library.mkdir()
            image_path = library / "incoming.jpg"
            Image.new("RGB", (120, 80), color="white").save(image_path)

            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(library))
            collection_id = store.create_collection("测试")
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            started = threading.Event()
            release = threading.Event()

            def slow_candidate_map(_paths, *, include_same_path):
                started.set()
                release.wait(timeout=1.0)
                return {}

            with (
                patch.object(window, "_near_duplicate_candidate_map", side_effect=slow_candidate_map),
                patch.object(
                    window,
                    "_resolve_near_duplicate_import_paths",
                    side_effect=AssertionError("file import must not do synchronous duplicate checking"),
                ),
                patch.object(window, "_continue_file_import_after_duplicate_resolution") as continue_import,
            ):
                started_at = time.monotonic()
                window._start_file_import([str(image_path)], collection_id)
                elapsed = time.monotonic() - started_at
                self.assertLess(elapsed, 0.2)
                self.assertTrue(started.wait(timeout=1.0))
                self.assertFalse(continue_import.called)
                release.set()
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and not continue_import.called:
                    self.app.processEvents()
                    time.sleep(0.01)

            self.assertTrue(continue_import.called)
            self.assertEqual(continue_import.call_args.args[0], [str(image_path)])
            window.close()

    def test_file_import_copies_to_collection_import_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            source_dir = Path(tmp) / "source"
            source_dir.mkdir()
            source_image = source_dir / "incoming.jpg"
            Image.new("RGB", (120, 80), color="white").save(source_image)
            import_dir = Path(tmp) / "imports" / "AI生成图库"

            store = MetadataStore(paths.database_path)
            store.initialize()
            collection_id = store.create_collection("AI生成图库")
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            with patch.object(window, "_collection_import_directory", return_value=import_dir):
                window._continue_file_import_after_duplicate_resolution(
                    [str(source_image)],
                    collection_id=collection_id,
                    replaced_count=0,
                    skipped_count=0,
                )
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    window._poll_events()
                    self.app.processEvents()
                    if store.collection_image_counts().get(collection_id) == 1:
                        break
                    time.sleep(0.01)

            images = store.list_images(limit=10)
            self.assertEqual(len(images), 1)
            self.assertEqual(Path(images[0].file_path), import_dir / "incoming.jpg")
            self.assertTrue((import_dir / "incoming.jpg").exists())
            self.assertEqual(store.collection_image_counts().get(collection_id), 1)
            window.close()

    def test_local_file_drop_copies_to_selected_collection_import_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            source_dir = Path(tmp) / "external"
            source_dir.mkdir()
            source_image = source_dir / "dragged.jpg"
            Image.new("RGB", (128, 96), color="black").save(source_image)
            import_dir = Path(tmp) / "imports" / "AI生成图库"

            store = MetadataStore(paths.database_path)
            store.initialize()
            collection_id = store.create_collection("AI生成图库")
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            with patch.object(window, "_collection_import_directory", return_value=import_dir):
                window._continue_local_paths_import_after_duplicate_resolution(
                    target_name="AI生成图库",
                    file_paths=[str(source_image)],
                    folder_paths=[],
                    parent_collection_id=collection_id,
                    skip_paths=set(),
                    replaced_count=0,
                    skipped_count=0,
                )
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    window._poll_events()
                    self.app.processEvents()
                    if store.collection_image_counts().get(collection_id) == 1:
                        break
                    time.sleep(0.01)

            images = store.list_images(limit=10)
            self.assertEqual(len(images), 1)
            self.assertEqual(Path(images[0].file_path), import_dir / "dragged.jpg")
            self.assertTrue((import_dir / "dragged.jpg").exists())
            self.assertEqual(store.collection_image_counts().get(collection_id), 1)
            window.close()

    def test_main_window_has_minimize_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            self.assertIn(QKeySequence("Meta+M"), window.minimize_window_action.shortcuts())
            self.assertIn(QKeySequence("Ctrl+M"), window.minimize_window_action.shortcuts())
            self.assertEqual(
                window.minimize_window_action.shortcutContext(),
                Qt.ShortcutContext.WindowShortcut,
            )
            window.close()

    def test_collection_tree_starts_collapsed_and_preserves_user_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            parent_id = store.create_collection("一级")
            child_id = store.create_collection("二级", parent_id)
            grandchild_id = store.create_collection("三级", child_id)
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            parent_item = self._collection_item(window.collection_tree, parent_id)
            child_item = self._collection_item(window.collection_tree, child_id)
            grandchild_item = self._collection_item(window.collection_tree, grandchild_id)
            self.assertIsNotNone(parent_item)
            self.assertIsNotNone(child_item)
            self.assertIsNotNone(grandchild_item)
            self.assertFalse(parent_item.isExpanded())
            self.assertFalse(child_item.isExpanded())
            self.assertNotEqual(
                parent_item.background(0).color().name(),
                child_item.background(0).color().name(),
            )
            self.assertNotEqual(
                child_item.background(0).color().name(),
                grandchild_item.background(0).color().name(),
            )

            parent_item.setExpanded(True)
            window._refresh_collections()
            refreshed_parent = self._collection_item(window.collection_tree, parent_id)
            refreshed_child = self._collection_item(window.collection_tree, child_id)

            self.assertIsNotNone(refreshed_parent)
            self.assertIsNotNone(refreshed_child)
            self.assertTrue(refreshed_parent.isExpanded())
            self.assertFalse(refreshed_child.isExpanded())
            window.close()

    def test_collection_filter_dialog_starts_collapsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            parent_id = store.create_collection("一级")
            child_id = store.create_collection("二级", parent_id)
            store.create_collection("三级", child_id)
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            inspected = False

            def inspect_dialog(dialog: QDialog) -> QDialog.DialogCode:
                nonlocal inspected
                tree = dialog.findChild(QTreeWidget)
                self.assertIsNotNone(tree)
                assert tree is not None
                parent_item = tree.topLevelItem(0)
                self.assertIsNotNone(parent_item)
                assert parent_item is not None
                self.assertFalse(parent_item.isExpanded())
                child_item = parent_item.child(0)
                self.assertIsNotNone(child_item)
                assert child_item is not None
                self.assertFalse(child_item.isExpanded())
                inspected = True
                return QDialog.DialogCode.Rejected

            with patch("eidory.ui.main_window.QDialog.exec", new=inspect_dialog):
                self.assertIsNone(
                    window._select_collections_for_search_dialog(
                        reverse_mode=False,
                        context_image=None,
                    )
                )

            self.assertTrue(inspected)
            window.close()

    def test_detail_panel_hides_duplicate_file_action_buttons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            self.assertTrue(window.play_pause_button.isHidden())
            self.assertTrue(window.open_original_button.isHidden())
            self.assertTrue(window.reveal_in_finder_button.isHidden())
            self.assertTrue(window.copy_path_button.isHidden())
            self.assertTrue(window.feedback_widget.isHidden())
            self.assertLessEqual(window.note_input.maximumHeight(), 96)
            self.assertIn("border: 0", window.tags_display.styleSheet())
            window.close()

    def test_duplicate_results_loads_parent_group_from_child_without_closing(self) -> None:
        group = DuplicateGroup(
            kind="exact",
            reason="same hash",
            members=(
                DuplicateMember(self._image(1), "A", "hash", None),
                DuplicateMember(self._image(2), "A", "hash", None),
            ),
        )
        dialog = DuplicateResultsDialog([group])
        dialog.show()
        self.app.processEvents()
        emitted: list[list[int]] = []
        dialog.groupLoadRequested.connect(lambda image_ids: emitted.append(list(image_ids)))

        dialog.tree.setCurrentItem(dialog.tree.topLevelItem(0).child(1))
        dialog._accept_selected_group()

        self.assertEqual(emitted, [[1, 2]])
        self.assertTrue(dialog.isVisible())
        dialog.close()

    def test_clear_search_restores_open_duplicate_results_dialog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            dialog = DuplicateResultsDialog([
                DuplicateGroup(
                    kind="exact",
                    reason="same hash",
                    members=(
                        DuplicateMember(self._image(1), "A", "hash", None),
                        DuplicateMember(self._image(2), "A", "hash", None),
                    ),
                )
            ], parent=window)
            window._duplicate_results_dialog = dialog
            window.current_result_mode = "duplicate_group"
            dialog.hide()

            with patch.object(window, "_reload_images"):
                window._clear_search()
                self.app.processEvents()

            self.assertTrue(dialog.isVisible())
            dialog.close()
            window.close()

    def test_duplicate_detection_loads_all_groups_into_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            images = [self._image(image_id) for image_id in [1, 2, 3, 4]]
            groups = [
                DuplicateGroup(
                    kind="exact",
                    reason="same hash",
                    members=(
                        DuplicateMember(images[0], "A", "hash-a", None),
                        DuplicateMember(images[1], "A", "hash-a", None),
                    ),
                ),
                DuplicateGroup(
                    kind="near",
                    reason="near hash",
                    members=(
                        DuplicateMember(images[2], "B", "hash-b", 1),
                        DuplicateMember(images[3], "B", "hash-c", 2),
                    ),
                ),
            ]

            window._handle_duplicates_done(groups)

            self.assertEqual([image.id for image in window.grid_view.images()], [1, 2, 3, 4])
            self.assertEqual(window.current_result_mode, "duplicate_group")
            self.assertEqual(window.manual_result_order_ids, [1, 2, 3, 4])
            self.assertEqual(window.current_duplicate_badges[1], ["完全重复 #1"])
            self.assertEqual(window.current_duplicate_badges[3], ["近重复 #2"])
            self.assertIsNotNone(window._duplicate_results_dialog)
            window._duplicate_results_dialog.close()
            window.close()

    def test_duplicate_results_load_all_button_restores_all_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            images = [self._image(image_id) for image_id in [1, 2, 3, 4]]
            groups = [
                DuplicateGroup(
                    kind="exact",
                    reason="same hash",
                    members=(
                        DuplicateMember(images[0], "A", "hash-a", None),
                        DuplicateMember(images[1], "A", "hash-a", None),
                    ),
                ),
                DuplicateGroup(
                    kind="near",
                    reason="near hash",
                    members=(
                        DuplicateMember(images[2], "B", "hash-b", 1),
                        DuplicateMember(images[3], "B", "hash-c", 2),
                    ),
                ),
            ]
            window._handle_duplicates_done(groups)
            dialog = window._duplicate_results_dialog
            self.assertIsNotNone(dialog)
            assert dialog is not None

            window._show_duplicate_images([images[0], images[1]], badges_by_image_id={1: ["重复组"], 2: ["重复组"]})
            dialog.allGroupsLoadRequested.emit()

            self.assertEqual([image.id for image in window.grid_view.images()], [1, 2, 3, 4])
            self.assertEqual(window.current_duplicate_badges[3], ["近重复 #2"])
            dialog.close()
            window.close()

    def test_duplicate_group_library_removal_updates_loaded_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            library_dir = Path(tmp) / "library"
            library_dir.mkdir()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(library_dir))
            image_ids: list[int] = []
            for index in range(3):
                image_path = library_dir / f"{index}.jpg"
                image_path.write_bytes(b"fake")
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(image_path),
                    file_size=image_path.stat().st_size,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            images = store.images_by_ids(image_ids)
            group = DuplicateGroup(
                kind="exact",
                reason="same hash",
                members=tuple(DuplicateMember(image, "A", "hash", None) for image in images),
            )
            dialog = DuplicateResultsDialog([group], parent=window)
            window._duplicate_groups = [group]
            window._duplicate_results_dialog = dialog
            window.current_result_mode = "duplicate_group"
            window.current_duplicate_images = list(images)
            window.current_duplicate_badges = {image.id: ["完全重复 #1"] for image in images}

            removed = window._remove_images_from_library_with_undo(
                [images[0]],
                undo_label="移除重复候选",
            )

            self.assertEqual(removed, 1)
            self.assertEqual([image.id for image in window.current_duplicate_images], image_ids[1:])
            self.assertEqual([[member.image.id for member in group.members] for group in window._duplicate_groups], [image_ids[1:]])
            self.assertEqual(dialog.tree.topLevelItemCount(), 1)
            self.assertEqual(dialog.tree.topLevelItem(0).childCount(), 2)
            dialog.close()
            window.close()

    def test_detail_path_shows_complete_wrapped_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            long_path = (
                "/Users/victorcloux/Pictures/Eidory图库/创作参考/绘画设计/"
                "特别长的目录名称/另一层特别长的目录名称/Screenshot_2019-01-19-00-28-23-9.jpg"
            )
            window.path_label.setFixedWidth(120)
            window._show_image_details(self._image(1, file_path=long_path))
            self.app.processEvents()

            self.assertEqual(window.path_label.toPlainText(), long_path)
            self.assertGreater(window.path_label.height(), 34)
            window.close()

    def test_selection_detail_state_handles_empty_single_and_multi_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            first = self._image(1, file_path=str(Path(tmp) / "first.jpg"))
            second = self._image(2, file_path=str(Path(tmp) / "second.jpg"))

            window._apply_selection_detail_state([first])
            self.assertEqual(window.selected_image.id, 1)
            self.assertTrue(window.image_detail_widget.isVisible())
            self.assertEqual(window.file_name_input.text(), "first.jpg")

            window._apply_selection_detail_state([first, second])
            self.assertEqual(window.selected_image.id, 2)
            self.assertIn("已选择 2 张", window.preview_label.text())
            self.assertTrue(window.batch_tags_widget.isVisible())

            window._apply_selection_detail_state([])
            self.assertIsNone(window.selected_image)
            self.assertTrue(window.collection_detail_widget.isVisible())
            window.close()

    def test_hidden_detail_tab_does_not_refresh_image_details_on_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            image = self._image(1)
            window.right_tab_widget.setCurrentIndex(1)
            self.app.processEvents()

            with patch.object(window, "_show_image_details") as show_image_details:
                window.grid_view.set_images([image], selected_image_ids=[image.id])
                self.app.processEvents()

            show_image_details.assert_not_called()
            self.assertEqual(window.selected_image, image)
            self.assertTrue(window._detail_panel_dirty)
            window.close()

    def test_project_board_selection_uses_loaded_images_without_store_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            image = self._image(1)
            window._board_image_by_id = {image.id: image}
            window.center_result_stack.setCurrentWidget(window.project_board_view)

            with patch.object(store, "get_image", side_effect=AssertionError("unexpected database lookup")):
                window._on_project_board_selection_changed([image.id])
                self.app.processEvents()

            self.assertEqual(window.selected_image, image)
            window.close()

    def test_detail_note_and_favorite_auto_save_without_renaming_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            library = Path(tmp) / "library"
            library.mkdir()
            image_path = library / "original.jpg"
            image_path.write_bytes(b"fake image bytes")
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(library))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=100,
                height=80,
                created_time_ns=None,
                modified_time_ns=image_path.stat().st_mtime_ns,
            )
            image = store.get_image(image_id)
            self.assertIsNotNone(image)
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window.selected_image = image
            window._show_image_details(image)
            window.file_name_input.setText("not-renamed.jpg")
            window.note_input.setPlainText("new note")
            window._save_pending_note()
            window.favorite_checkbox.setChecked(True)
            self.app.processEvents()

            updated = store.get_image(image_id)
            self.assertIsNotNone(updated)
            self.assertEqual(updated.note, "new note")
            self.assertTrue(updated.is_favorite)
            self.assertEqual(updated.file_name, "original.jpg")
            self.assertTrue(image_path.exists())
            self.assertFalse((library / "not-renamed.jpg").exists())
            window.close()

    def test_detail_rename_requires_explicit_rename_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            library = Path(tmp) / "library"
            library.mkdir()
            image_path = library / "original.jpg"
            image_path.write_bytes(b"fake image bytes")
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(library))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=100,
                height=80,
                created_time_ns=None,
                modified_time_ns=image_path.stat().st_mtime_ns,
            )
            image = store.get_image(image_id)
            self.assertIsNotNone(image)
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window.selected_image = image
            window._show_image_details(image)
            window.file_name_input.setText("renamed.jpg")
            window.rename_file_button.click()
            self.app.processEvents()

            renamed_path = library / "renamed.jpg"
            updated = store.get_image(image_id)
            self.assertIsNotNone(updated)
            self.assertFalse(image_path.exists())
            self.assertTrue(renamed_path.exists())
            self.assertEqual(updated.file_name, "renamed.jpg")
            self.assertEqual(updated.file_path, str(renamed_path))
            window.close()

    def test_image_detail_does_not_initialize_video_preview_stack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            library = Path(tmp) / "library"
            library.mkdir()
            image_path = library / "image.jpg"
            image_path.write_bytes(b"fake image bytes")
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(library))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=100,
                height=80,
                created_time_ns=None,
                modified_time_ns=image_path.stat().st_mtime_ns,
            )
            image = store.get_image(image_id)
            self.assertIsNotNone(image)
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window._show_image_details(image)
            self.app.processEvents()

            self.assertIsNone(window.video_player)
            self.assertIsNone(window.video_widget)
            self.assertIs(window.preview_stack.currentWidget(), window.preview_label)
            window.close()

    def test_sort_preference_restores_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            store.set_setting("ui.sort_key", "name")
            store.set_setting("ui.sort_order", "asc")
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            self.assertEqual(window.sort_combo.currentData(), "name")
            self.assertEqual(window.sort_order_combo.currentData(), "asc")
            self.assertEqual(window.current_sort_key, "name")
            self.assertFalse(window.current_sort_desc)

            window.sort_combo.setCurrentIndex(window.sort_combo.findData("modified"))
            window.sort_order_combo.setCurrentIndex(window.sort_order_combo.findData("desc"))
            self.app.processEvents()
            window.close()

            self.assertEqual(store.get_setting("ui.sort_key"), "modified")
            self.assertEqual(store.get_setting("ui.sort_order"), "desc")

    def test_root_splitter_sizes_persist_on_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.resize(1200, 800)
            window.show()
            window.root_splitter.setSizes([200, 800, 200])
            self.app.processEvents()
            window.close()

            saved = store.get_setting("ui.root_splitter_sizes")
            self.assertIsNotNone(saved)
            sizes = [int(part) for part in saved.split(",")]
            self.assertEqual(len(sizes), 3)
            self.assertTrue(all(size > 0 for size in sizes))
            self.assertEqual(sizes[0], LEFT_SIDEBAR_WIDTH)
            self.assertEqual(sizes[2], RIGHT_SIDEBAR_WIDTH)

    def test_root_splitter_recovers_from_saved_narrow_sidebar_width(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            store.set_setting(
                "ui.root_splitter_sizes",
                f"{SIDEBAR_COLLAPSE_THRESHOLD + 40},900,{RIGHT_SIDEBAR_WIDTH - 80}",
            )

            window = MainWindow(paths=paths, store=store)
            window.resize(1500, 900)
            window.show()
            self.app.processEvents()
            window._enforce_fixed_sidebar_widths()

            left, _center, right = window.root_splitter.sizes()
            self.assertEqual(left, LEFT_SIDEBAR_WIDTH)
            self.assertEqual(right, RIGHT_SIDEBAR_WIDTH)
            self.assertEqual(window.collection_tree.columnWidth(1), SIDEBAR_COUNT_COLUMN_WIDTH)
            self.assertGreaterEqual(window.collection_tree.columnWidth(0), 210)
            window.close()

    def test_filter_panel_preferences_restore_and_persist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp)))
            first, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "first.jpg"),
                file_size=100,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            second, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "second.jpg"),
                file_size=100,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=2,
            )
            store.set_image_tags(first, ["室内", "夜晚"])
            store.set_image_tags(second, ["室内"])
            indoor_id = self._tag_id(store, "室内")
            night_id = self._tag_id(store, "夜晚")
            store.set_setting("ui.status_filter", "favorite")
            store.set_setting("ui.tag_sort", "count_desc")
            store.set_setting("ui.tag_match_mode", "any")
            store.set_setting("ui.selected_tag_ids", f"{indoor_id},{night_id}")
            store.set_setting("ui.right_tab_index", "1")

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            self.assertEqual(window.status_filter_combo.currentData(), "favorite")
            self.assertEqual(window.tag_sort_combo.currentData(), "count_desc")
            self.assertEqual(window.tag_match_combo.currentData(), "any")
            self.assertEqual(window.right_tab_widget.currentIndex(), 1)
            self.assertEqual(set(window._selected_tag_ids()), {indoor_id, night_id})

            window._set_combo_to_data(window.status_filter_combo, "missing")
            window.tag_sort_combo.setCurrentIndex(window.tag_sort_combo.findData("count_asc"))
            window.tag_match_combo.setCurrentIndex(window.tag_match_combo.findData("all"))
            window.right_tab_widget.setCurrentIndex(2)
            window.tag_list.clearSelection()
            night_item = self._tag_item(window, "夜晚")
            night_item.setSelected(True)
            window.tag_list.setCurrentItem(night_item)
            self.app.processEvents()

            window.close()
            self.assertEqual(store.get_setting("ui.status_filter"), "missing")
            self.assertEqual(store.get_setting("ui.tag_sort"), "count_asc")
            self.assertEqual(store.get_setting("ui.tag_match_mode"), "all")
            self.assertEqual(store.get_setting("ui.selected_tag_ids"), str(night_id))
            self.assertEqual(store.get_setting("ui.right_tab_index"), "2")

    def test_tag_panel_filters_sorts_and_enables_management_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp)))
            first, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "first.jpg"),
                file_size=100,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            second, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "second.jpg"),
                file_size=100,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=2,
            )
            third, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "third.jpg"),
                file_size=100,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=3,
            )
            store.set_image_tags(first, ["室内", "人物"])
            store.set_image_tags(second, ["室内"])
            store.set_image_tags(third, ["机械"])
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            self.assertFalse(window.rename_tag_button.isEnabled())
            window.tag_sort_combo.setCurrentIndex(window.tag_sort_combo.findData("count_desc"))
            self.app.processEvents()
            indoor_item = self._tag_item(window, "室内")
            self.assertEqual(indoor_item.data(Qt.ItemDataRole.UserRole + 2), 2)

            window.tag_search_input.setText("机")
            self.app.processEvents()
            tag_item = self._tag_item(window, "机械")
            self.assertEqual(tag_item.data(Qt.ItemDataRole.UserRole + 1), "机械")
            window.tag_list.setCurrentItem(tag_item)
            tag_item.setSelected(True)
            self.app.processEvents()

            self.assertTrue(window.rename_tag_button.isEnabled())
            self.assertTrue(window.delete_tag_button.isEnabled())
            self.assertTrue(window.merge_tag_button.isEnabled())
            self.assertEqual(window._selected_tag_context()[1], "机械")
            window.close()

    def test_topbar_tag_filter_guides_to_tag_page_when_no_tags_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            with patch("eidory.ui.main_window.QMessageBox.information") as information:
                window._choose_tag_filter()

            self.assertEqual(window.right_tab_widget.currentIndex(), 2)
            information.assert_called_once()
            window.close()

    def test_sidebar_tag_page_adds_tags_to_selected_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "image.jpg"),
                file_size=100,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            window = MainWindow(paths=paths, store=store)
            window.show()
            window.grid_view.set_images(
                store.images_by_ids([image_id]),
                selected_image_ids=[image_id],
            )
            self.app.processEvents()

            self.assertFalse(hasattr(window, "tag_panel_input"))
            self.assertEqual(window.tag_panel_pick_button.text(), "给选中图片添加标签")

            with patch("eidory.ui.main_window.TagPickerDialog") as picker_cls:
                picker = picker_cls.return_value
                picker.exec.return_value = QDialog.DialogCode.Accepted
                picker.selected_tags.return_value = ["参考", "机械"]
                window._tag_panel_pick_tags()

            self.assertEqual(store.get_image_tags(image_id), ["参考", "机械"])
            self.assertIn("已选择 1 张", window.tag_panel_selection_label.text())
            self.assertEqual(window.tag_panel_remove_combo.count(), 2)
            tag_names = {
                window.tag_panel_remove_combo.itemData(index)
                for index in range(window.tag_panel_remove_combo.count())
            }
            self.assertEqual(tag_names, {"参考", "机械"})
            self.assertGreaterEqual(window.tag_list.count(), 3)
            window.close()

    def test_tag_picker_filters_existing_tags_and_creates_new_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "image.jpg"),
                file_size=100,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            store.set_image_tags(image_id, ["室内", "机械", "夜晚"])

            dialog = TagPickerDialog(store.list_tags_with_counts())
            dialog.search_input.setText("机")
            self.app.processEvents()
            self.assertEqual(dialog.tag_list.count(), 1)
            item = dialog.tag_list.item(0)
            self.assertEqual(item.data(Qt.ItemDataRole.UserRole), "机械")
            item.setCheckState(Qt.CheckState.Checked)
            dialog.search_input.setText("新标签")
            self.app.processEvents()
            self.assertTrue(dialog.create_button.isEnabled())
            dialog.create_button.click()
            self.app.processEvents()

            self.assertEqual(dialog.selected_tags(), ["机械", "新标签"])
            dialog.close()

    def test_sidebar_tag_page_picker_adds_existing_and_new_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            first_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=100,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            second_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "second.jpg"),
                file_size=100,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=2,
            )
            store.set_image_tags(first_id, ["室内"])
            window = MainWindow(paths=paths, store=store)
            window.show()
            window.grid_view.set_images(
                store.images_by_ids([first_id, second_id]),
                selected_image_ids=[first_id, second_id],
            )
            self.app.processEvents()

            with patch("eidory.ui.main_window.TagPickerDialog") as picker_cls:
                picker = picker_cls.return_value
                picker.exec.return_value = QDialog.DialogCode.Accepted
                picker.selected_tags.return_value = ["室内", "新标签"]
                window._tag_panel_pick_tags()

            self.assertEqual(store.get_image_tags(first_id), ["室内", "新标签"])
            self.assertEqual(store.get_image_tags(second_id), ["室内", "新标签"])
            self.assertIn("已选择 2 张", window.tag_panel_selection_label.text())
            window.close()

    def test_tag_management_groups_and_moves_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            tag_id = store.create_tag("置换质感")
            group_id = store.create_tag_group("风格")
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            self.assertGreaterEqual(window.tag_group_combo.findData(group_id), 0)
            window._set_combo_to_data(window.tag_group_combo, group_id)
            window._select_only_tag_id(tag_id)
            window._move_selected_tags_to_group()
            self.app.processEvents()

            moved_tag = next(tag for tag in store.list_tags() if tag.id == tag_id)
            self.assertEqual(moved_tag.group_id, group_id)
            self.assertEqual(moved_tag.group_name, "风格")
            group_header_labels = [
                window.tag_list.item(index).text()
                for index in range(window.tag_list.count())
                if window.tag_list.item(index).data(TAG_GROUP_HEADER_ROLE)
            ]
            self.assertIn("风格", group_header_labels)
            window.close()

    def test_color_filter_uses_relative_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window.current_result_mode = "color"
            window.current_color_images = [
                self._image(1, score=0.04),
                self._image(2, score=0.02),
                self._image(3, score=0.005),
            ]
            window.score_threshold_slider.setValue(50)
            window._apply_color_result_filters()

            window.close()
            self.assertEqual([image.id for image in window.current_color_filtered_images], [1, 2])

    def test_search_stack_scope_uses_other_active_search_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window.current_result_mode = "semantic"
            window.current_semantic_query = "车"
            window.current_semantic_filtered_images = [
                self._image(1, score=0.4),
                self._image(2, score=0.3),
            ]
            self.assertEqual(window._stacked_search_scope_ids("color"), [1, 2])

            window.current_color_images = [self._image(3, score=0.2)]
            window.current_color_filtered_images = [self._image(3, score=0.2)]
            self.assertEqual(window._stacked_search_scope_ids("semantic"), [3])
            window.close()

    def test_search_filter_chain_can_replace_and_remove_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window._add_search_filter(SearchFilter("semantic", "车"))
            window._add_search_filter(SearchFilter("color", (255, 0, 0)))
            window._add_search_filter(SearchFilter("color", (0, 0, 255)))

            self.assertEqual(
                window.search_filters,
                [
                    SearchFilter("semantic", "车"),
                    SearchFilter("color", (0, 0, 255)),
                ],
            )
            self.assertIn("语义：车", window._format_filter_chain(window.search_filters))
            self.assertIn("颜色：#0000FF", window._format_filter_chain(window.search_filters))

            window._remove_search_filter(0)
            self.app.processEvents()

            window.close()
            self.assertEqual(window.search_filters, [SearchFilter("color", (0, 0, 255))])

    def test_temporary_project_load_preserves_intent_badges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            first_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=100,
                height=200,
                created_time_ns=None,
                modified_time_ns=1,
            )
            second_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "second.jpg"),
                file_size=456,
                width=200,
                height=100,
                created_time_ns=None,
                modified_time_ns=2,
            )
            project_id = store.create_temporary_project("机械参考", [first_id, second_id])
            store.add_images_to_temporary_project(
                project_id,
                [first_id],
                intent_labels={first_id: "引擎细节 +1"},
                intent_queries={first_id: "老旧引擎，机械结构"},
            )

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window._load_temporary_project(project_id)
            self.app.processEvents()

            self.assertEqual(window.current_result_mode, "temp_project")
            self.assertEqual(window.current_temp_project_badges, {first_id: ["引擎细节 +1"]})
            self.assertEqual(window.grid_view._badges_by_image_id, {first_id: ["引擎细节 +1"]})
            window._set_combo_to_data(window.sort_combo, "name")
            window._on_sort_changed()
            self.assertEqual(window.grid_view._badges_by_image_id, {first_id: ["引擎细节 +1"]})
            window.close()

    def test_clear_all_temporary_projects_resets_temporary_project_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=100,
                height=200,
                created_time_ns=None,
                modified_time_ns=1,
            )
            project_id = store.create_temporary_project("机械参考", [image_id])

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window._load_temporary_project(project_id)
            self.assertEqual(window.current_result_mode, "temp_project")
            window._clear_all_temporary_projects(confirm=False)
            self.app.processEvents()

            self.assertEqual(store.list_temporary_projects(), [])
            self.assertEqual(
                [
                    window.temp_project_list.item(index).data(PROJECT_LIST_KIND_ROLE)
                    for index in range(window.temp_project_list.count())
                    if window.temp_project_list.item(index).data(PROJECT_LIST_ID_ROLE) is not None
                ],
                [],
            )
            self.assertEqual(window.current_result_mode, "library")
            self.assertIsNotNone(store.get_image(image_id))
            window.close()

    def test_temporary_projects_can_be_manually_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_ids = []
            for index in range(3):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=123,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)
            first = store.create_temporary_project("一", [image_ids[0]])
            second = store.create_temporary_project("二", [image_ids[1]])
            third = store.create_temporary_project("三", [image_ids[2]])

            self.assertEqual([project.name for project in store.list_temporary_projects()], ["三", "二", "一"])
            self.assertTrue(store.move_temporary_project(third, 1))
            self.assertEqual([project.name for project in store.list_temporary_projects()], ["二", "三", "一"])
            self.assertTrue(store.move_temporary_project(first, -1))
            self.assertEqual([project.name for project in store.list_temporary_projects()], ["二", "一", "三"])
            self.assertFalse(store.move_temporary_project(second, -1))

    def test_search_chain_can_continue_within_loaded_temporary_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            keep_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "keep-machine.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            other_project_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "other.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=2,
            )
            outside_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "keep-outside.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=3,
            )
            project_id = store.create_temporary_project("机械暂存", [keep_id, other_project_id])
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window._load_temporary_project(project_id)
            base_ids, base_label = window._search_chain_base_context()
            result = window._compute_search_chain(
                filters=(SearchFilter("keyword", "keep"),),
                folder_path_prefix=None,
                collection_id=None,
                tag_ids=[],
                tag_match_mode="any",
                status_filter=None,
                base_image_ids=base_ids,
            )

            self.assertEqual(base_ids, {keep_id, other_project_id})
            self.assertEqual(base_label, "基于语义探针项目：机械暂存")
            self.assertEqual([image.id for image in result.images], [keep_id])
            self.assertNotIn(outside_id, [image.id for image in result.images])
            window.close()

    def test_search_operation_context_uses_visible_results_only_after_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window.grid_view.set_images([self._image(1), self._image(2)])
            window.search_filters = [SearchFilter("semantic", "晴天")]
            base_ids, base_label, merge_base = window._search_operation_context("refine")
            self.assertIsNone(base_ids)
            self.assertIsNone(base_label)
            self.assertIsNone(merge_base)

            window.current_result_mode = "search_chain"
            base_ids, base_label, merge_base = window._search_operation_context("refine")
            self.assertEqual(base_ids, {1, 2})
            self.assertEqual(base_label, "在当前结果中")
            self.assertIsNone(merge_base)

            base_ids, base_label, merge_base = window._search_operation_context("merge")
            self.assertIsNone(base_ids)
            self.assertEqual(base_label, "合并当前结果")
            self.assertEqual([image.id for image in merge_base or []], [1, 2])
            window.close()

    def test_refine_search_uses_unthresholded_source_result_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            source_images = [
                self._image(1, score=0.95),
                self._image(2, score=0.70),
                self._image(3, score=0.40),
            ]
            window.current_result_mode = "color"
            window.current_color_images = list(source_images)
            window.current_color_filtered_images = [source_images[0]]
            window.grid_view.set_images([source_images[0]])

            operation_context = window._capture_search_operation_context()
            base_ids, base_label, merge_base = window._search_operation_context(
                "refine",
                operation_context=operation_context,
            )

            self.assertEqual(base_ids, {1, 2, 3})
            self.assertEqual(base_label, "在当前结果中")
            self.assertIsNone(merge_base)

            base_ids, base_label, merge_base = window._search_operation_context(
                "merge",
                operation_context=operation_context,
            )
            self.assertIsNone(base_ids)
            self.assertEqual(base_label, "合并当前结果")
            self.assertEqual([image.id for image in merge_base or []], [1])
            window.close()

    def test_merge_search_operation_unions_current_and_new_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            sunny_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "sunny.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            house_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "house.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=2,
            )
            store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "tree.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=3,
            )
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            result = window._compute_search_chain(
                filters=(SearchFilter("keyword", "house"),),
                folder_path_prefix=None,
                collection_id=None,
                tag_ids=[],
                tag_match_mode="any",
                status_filter=None,
                merge_base_images=store.images_by_ids([sunny_id]),
            )

            self.assertEqual([image.id for image in result.images], [sunny_id, house_id])
            window.close()

    def test_collection_filter_matches_multiple_selected_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            root_id = store.create_collection("创作参考")
            indoor_id = store.create_collection("室内", root_id)
            outdoor_id = store.create_collection("室外", root_id)
            sketch_id = store.create_collection("线稿")
            image_ids = []
            for index in range(3):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=123,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)
            store.assign_images_to_collection([image_ids[0]], indoor_id)
            store.assign_images_to_collection([image_ids[1]], outdoor_id)
            store.assign_images_to_collection([image_ids[2]], sketch_id)
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            search_filter = SearchFilter(
                "collection",
                window._collection_filter_value([indoor_id, outdoor_id]),
            )
            result = window._compute_search_chain(
                filters=(search_filter,),
                folder_path_prefix=None,
                collection_id=None,
                tag_ids=[],
                tag_match_mode="any",
                status_filter=None,
            )

            self.assertEqual({image.id for image in result.images}, set(image_ids[:2]))
            self.assertEqual(search_filter_from_payload(search_filter_to_payload(search_filter)), search_filter)
            self.assertIn("创作参考 / 室内", window._filter_label(search_filter))
            window.close()

    def test_virtual_collection_filters_are_separate_and_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            untagged_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "untagged.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            tagged_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "tagged.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=2,
            )
            store.add_tags_to_images([tagged_id], ["已标签"])

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            top_labels = [
                window.collection_tree.topLevelItem(index).text(0)
                for index in range(window.collection_tree.topLevelItemCount())
            ]
            self.assertNotIn("未标签", top_labels)
            self.assertNotIn("未AI标签", top_labels)
            self.assertEqual(top_labels[-1], "未分类")
            tag_labels = [window.tag_list.item(index).text() for index in range(window.tag_list.count())]
            self.assertTrue(any(label.startswith("未标签") for label in tag_labels))
            ai_labels = [
                window.ai_vision_virtual_filter_list.item(index).text()
                for index in range(window.ai_vision_virtual_filter_list.count())
            ]
            self.assertTrue(any(label.startswith("未AI标签") for label in ai_labels))

            untagged_item = next(
                window.tag_list.item(index)
                for index in range(window.tag_list.count())
                if window.tag_list.item(index).text().startswith("未标签")
            )
            window.tag_list.setCurrentItem(untagged_item)
            self.app.processEvents()

            self.assertEqual(window._selected_virtual_filter(), "untagged")
            self.assertIsNone(window.collection_tree.currentItem())
            self.assertEqual({image.id for image in window.grid_view.images()}, {untagged_id})
            window.close()

    def test_startup_removes_previously_scanned_active_roots_missing_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            missing_root = Path(tmp) / "deleted-root"
            folder_id = store.add_folder(str(missing_root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(missing_root / "stale.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            store.update_thumbnail(image_id, str(paths.thumbnail_dir / "thumb_stale.webp"), "ready")
            collection_id = store.create_collection("失效目录")
            store.assign_images_to_collection([image_id], collection_id)
            store.finish_folder_scan(folder_id)

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            self.assertIsNone(store.get_folder(folder_id))
            self.assertIsNone(store.get_image(image_id))
            self.assertEqual(store.list_collections_with_counts()[0][1], 0)
            window.close()

    def test_refine_search_can_append_same_filter_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window._add_search_filter(SearchFilter("semantic", "晴天"))
            window._add_search_filter(SearchFilter("semantic", "房屋"), replace_same_kind=False)

            self.assertEqual(
                window.search_filters,
                [
                    SearchFilter("semantic", "晴天"),
                    SearchFilter("semantic", "房屋"),
                ],
            )
            window.close()

    def test_inspiration_matches_become_temporary_project_intents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.current_result_mode = "inspiration"
            window.current_inspiration_matches = {
                8: [
                    InspirationMatch("破旧工坊", "破旧工坊，昏暗灯光", "环境参考", 0.7),
                    InspirationMatch("引擎细节", "老旧引擎，机械结构", "机械参考", 0.6),
                ]
            }

            labels, queries = window._temporary_project_intents_for_images([self._image(8)])

            self.assertEqual(labels, {8: "破旧工坊 +1"})
            self.assertEqual(queries, {8: "破旧工坊，昏暗灯光"})
            window.close()

    def test_temporary_project_badges_are_preserved_when_re_saving_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.current_result_mode = "temp_project"
            window.current_temp_project_badges = {8: ["破旧工坊 +1"]}

            labels, queries = window._temporary_project_intents_for_images([self._image(8)])

            self.assertEqual(labels, {8: "破旧工坊 +1"})
            self.assertEqual(queries, {})
            window.close()

    def test_clicking_inspiration_term_does_not_run_single_probe_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window._show_inspiration_proposal(SimpleNamespace(
                terms=[
                    InspirationTerm(title=f"探针{i}", query=f"query {i}", reason="reason")
                    for i in range(6)
                ],
                questions=[],
                model_name="fake",
            ))
            calls: list[str] = []
            window._run_single_inspiration_term_search = lambda term: calls.append(term.title)

            sixth_item = window.inspiration_term_list.item(5)
            sixth_item.setCheckState(Qt.CheckState.Checked)
            window.inspiration_term_list.itemClicked.emit(sixth_item)
            self.app.processEvents()

            self.assertEqual(calls, [])
            self.assertEqual(
                [term.title for term in window._selected_inspiration_terms()],
                [f"探针{i}" for i in range(6)],
            )
            self.assertIn("保存并搜索会混排所有已选探针", window.inspiration_status_label.text())
            window.close()

    def test_inspiration_history_restores_brief_answers_questions_and_selected_terms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            project_id = store.create_inspiration_project(
                title="机械工程师",
                brief="落魄机械工程师研究摩托车",
                answers="雨夜，低饱和",
                questions=["更偏未来还是复古？"],
                provider_name="LM Studio",
                model_name="fake",
                terms=[
                    InspirationTerm(title="破旧工坊", query="破旧工坊", reason="环境"),
                    InspirationTerm(title="引擎细节", query="老旧引擎", reason="机械"),
                ],
                selected_titles={"引擎细节"},
            )

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            self.assertEqual(window.inspiration_history_list.count(), 1)
            window._load_inspiration_project(project_id)
            self.app.processEvents()

            self.assertEqual(window.current_inspiration_project_id, project_id)
            self.assertEqual(
                window.inspiration_brief_input.toPlainText(),
                "落魄机械工程师研究摩托车",
            )
            self.assertEqual(window.inspiration_answers_input.toPlainText(), "雨夜，低饱和")
            self.assertIn("更偏未来还是复古？", window.inspiration_questions_label.text())
            self.assertEqual(window.inspiration_term_list.count(), 2)
            self.assertEqual(
                window.inspiration_term_list.item(0).checkState(),
                Qt.CheckState.Unchecked,
            )
            self.assertEqual(
                window.inspiration_term_list.item(1).checkState(),
                Qt.CheckState.Checked,
            )
            self.assertTrue(window.search_inspiration_button.isEnabled())
            window.close()

    def test_reference_group_payload_creates_temporary_projects_with_badges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_ids: list[int] = []
            for index in range(4):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=123 + index,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window._create_reference_group_projects((
                [
                    ReferenceGroup(image_ids=image_ids[:2], representative_id=image_ids[0]),
                    ReferenceGroup(image_ids=image_ids[2:], representative_id=image_ids[2]),
                ],
                [
                    GroupNameSuggestion("破旧工坊", "工作台和昏暗室内参考。"),
                    GroupNameSuggestion("机械细节", "引擎和金属结构参考。"),
                ],
                "",
            ))

            projects = store.list_temporary_projects()
            self.assertEqual(len(projects), 2)
            names = {project.name for project in projects}
            self.assertEqual(names, {"破旧工坊", "机械细节"})
            colors = {project.color_hex for project in projects}
            self.assertEqual(colors, {TEMPORARY_PROJECT_COLORS[0]})
            self._expand_project_sidebar_section(window, "temporary")
            temporary_item = next(
                item
                for item in (window.temp_project_list.item(index) for index in range(window.temp_project_list.count()))
                if item.data(PROJECT_LIST_KIND_ROLE) == "temporary"
            )
            self.assertEqual(temporary_item.background().style(), Qt.BrushStyle.NoBrush)
            first_project = next(project for project in projects if project.name == "破旧工坊")
            self.assertEqual(store.temporary_project_image_badges(first_project.id), {
                image_ids[0]: ["破旧工坊"],
                image_ids[1]: ["破旧工坊"],
            })
            window.close()

    def test_reference_group_creation_can_be_cancelled_before_saving(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_ids: list[int] = []
            for index in range(4):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=123 + index,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            with patch(
                "eidory.ui.main_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.No,
            ):
                window._create_reference_group_projects(
                    (
                        [
                            ReferenceGroup(image_ids=image_ids[:2], representative_id=image_ids[0]),
                            ReferenceGroup(image_ids=image_ids[2:], representative_id=image_ids[2]),
                        ],
                        [
                            GroupNameSuggestion("破旧工坊", "工作台和昏暗室内参考。"),
                            GroupNameSuggestion("机械细节", "引擎和金属结构参考。"),
                        ],
                        "",
                    ),
                    confirm=True,
                )

            self.assertEqual(store.list_temporary_projects(), [])
            self.assertEqual(window.statusBar().currentMessage(), "已取消 AI 分组保存")
            window.close()

    def test_ai_project_detail_update_refreshes_loaded_temporary_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=100,
                height=200,
                created_time_ns=None,
                modified_time_ns=1,
            )
            project_id = store.create_temporary_project("临时项目", [image_id])

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window._load_temporary_project(project_id)
            window._apply_temporary_project_suggestion(
                (
                    project_id,
                    True,
                    ProjectSuggestion(
                        name="AI 命名项目",
                        summary="用于机械住处与旧设备参考。",
                        model_name="fake",
                    ),
                )
            )
            self.app.processEvents()

            updated = store.get_temporary_project(project_id)
            self.assertIsNotNone(updated)
            self.assertEqual(updated.name, "AI 命名项目")
            self.assertEqual(updated.summary, "用于机械住处与旧设备参考。")
            self.assertIn("AI 命名项目", window.result_state_label.text())
            self.assertIn("用于机械住处与旧设备参考。", window.result_state_label.text())
            window.close()

    def test_manual_project_detail_update_can_clear_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=100,
                height=200,
                created_time_ns=None,
                modified_time_ns=1,
            )
            project_id = store.create_temporary_project(
                "临时项目",
                [image_id],
                summary="旧摘要",
            )

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window._load_temporary_project(project_id)
            window._update_temporary_project_details_from_values(
                project_id,
                name="手动命名",
                summary="",
            )
            self.app.processEvents()

            updated = store.get_temporary_project(project_id)
            self.assertIsNotNone(updated)
            self.assertEqual(updated.name, "手动命名")
            self.assertEqual(updated.summary, "")
            self.assertIn("手动命名", window.result_state_label.text())
            self.assertNotIn("旧摘要", window.result_state_label.text())
            window.close()

    def test_saved_view_payload_restores_ui_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "image.jpg"),
                file_size=123,
                width=100,
                height=200,
                created_time_ns=None,
                modified_time_ns=1_700_000_000_000_000_000,
            )
            collection_id = store.create_collection("场景")
            store.assign_images_to_collection([image_id], collection_id)
            store.update_favorite(image_id, True)
            store.set_image_tags(image_id, ["室内", "夜晚"])

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            collection_item = self._collection_item(window.collection_tree, collection_id)
            self.assertIsNotNone(collection_item)
            window.collection_tree.setCurrentItem(collection_item)
            window._set_combo_to_data(window.status_filter_combo, "favorite")
            window.tag_list.clearSelection()
            self._tag_item(window, "室内").setSelected(True)
            window._set_combo_to_data(window.tag_match_combo, "any")
            window._set_combo_to_data(window.sort_combo, "name")
            window._set_combo_to_data(window.sort_order_combo, "asc")
            window.score_threshold_slider.setValue(56)
            window.search_filters = []

            payload = window._current_view_payload()
            self.assertEqual(payload["collection_id"], collection_id)
            self.assertEqual(payload["status_filter"], "favorite")
            self.assertEqual(payload["tag_match_mode"], "any")
            self.assertEqual(payload["sort_key"], "name")
            self.assertEqual(payload["sort_order"], "asc")
            self.assertEqual(payload["score_threshold"], 56)

            all_item = window.collection_tree.topLevelItem(0)
            window.collection_tree.setCurrentItem(all_item)
            window._set_combo_to_data(window.status_filter_combo, "all")
            window.tag_list.clearSelection()
            window.tag_list.item(0).setSelected(True)
            window._set_combo_to_data(window.tag_match_combo, "all")
            window._set_combo_to_data(window.sort_combo, "default")
            window._set_combo_to_data(window.sort_order_combo, "desc")
            window.score_threshold_slider.setValue(0)

            window._apply_view_payload(payload)
            self.app.processEvents()

            self.assertEqual(window._selected_collection_id(), collection_id)
            self.assertEqual(window._selected_status_filter(), "favorite")
            self.assertEqual(window._selected_tag_names(), ["室内"])
            self.assertEqual(window._selected_tag_match_mode(), "any")
            self.assertEqual(window.current_sort_key, "name")
            self.assertFalse(window.current_sort_desc)
            self.assertEqual(window.score_threshold_slider.value(), 56)
            window.close()

    def test_search_filter_payload_round_trip_validates_color(self) -> None:
        color_payload = search_filter_to_payload(
            SearchFilter("color", (12, 34, 56))
        )
        self.assertEqual(color_payload, {"kind": "color", "value": [12, 34, 56]})
        self.assertEqual(
            search_filter_from_payload(color_payload),
            SearchFilter("color", (12, 34, 56)),
        )
        self.assertEqual(
            search_filter_from_payload(
                {"kind": "file_type", "value": "media:image"}
            ),
            SearchFilter("file_type", "media:image"),
        )
        self.assertEqual(
            search_filter_from_payload({"kind": "similar", "value": 42}),
            SearchFilter("similar", 42),
        )
        self.assertIsNone(
            search_filter_from_payload({"kind": "similar", "value": 0})
        )
        self.assertIsNone(
            search_filter_from_payload(
                {"kind": "color", "value": [12, 34, 300]}
            )
        )

    def test_all_folders_can_run_import_tree_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            all_item = window.collection_tree.topLevelItem(0)
            _menu, actions = window._build_collection_context_menu(all_item)

            window.close()
            self.assertTrue(actions["import_tree"].isEnabled())
            self.assertFalse(actions["import_flat"].isEnabled())

    def test_collection_blank_area_can_import_folder_tree_to_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            _menu, actions = window._build_collection_context_menu(None)

            window.close()
            self.assertTrue(actions["import_tree"].isEnabled())
            self.assertFalse(actions["import_flat"].isEnabled())

    def test_collection_search_exclusion_marks_tree_menu_filters_search_not_browsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            root = Path(tmp) / "library"
            folder_id = store.add_folder(str(root))
            keep_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(root / "keep.jpg"),
                file_size=10,
                width=100,
                height=80,
                created_time_ns=None,
                modified_time_ns=1,
            )
            excluded_parent_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(root / "parent.jpg"),
                file_size=10,
                width=100,
                height=80,
                created_time_ns=None,
                modified_time_ns=2,
            )
            excluded_child_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(root / "child.jpg"),
                file_size=10,
                width=100,
                height=80,
                created_time_ns=None,
                modified_time_ns=3,
            )
            parent_id = store.create_collection("ML-09 线稿")
            child_id = store.create_collection("子线稿", parent_id=parent_id)
            store.assign_images_to_collection([excluded_parent_id], parent_id)
            store.assign_images_to_collection([excluded_child_id], child_id)
            store.add_search_excluded_collection_id(parent_id)

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._reload_images()
            self.app.processEvents()

            parent_item = self._collection_item(window.collection_tree, parent_id)
            child_item = self._collection_item(window.collection_tree, child_id)
            self.assertIsNotNone(parent_item)
            self.assertIsNotNone(child_item)
            assert parent_item is not None
            assert child_item is not None

            _menu, actions = window._build_collection_context_menu(parent_item)
            self.assertFalse(actions["exclude_from_search"].isEnabled())
            self.assertTrue(actions["include_in_search"].isEnabled())
            _child_menu, child_actions = window._build_collection_context_menu(child_item)
            self.assertEqual(child_actions["exclude_from_search"].text(), "已被上级文件夹排除")
            self.assertFalse(child_actions["exclude_from_search"].isEnabled())
            self.assertFalse(child_actions["include_in_search"].isEnabled())
            self.assertTrue(parent_item.text(0).endswith("⊘"))
            self.assertEqual(
                parent_item.foreground(0).color().name(),
                FolderTreeItemDelegate.EXCLUDED_NAME_COLOR.name(),
            )
            self.assertFalse(parent_item.icon(0).isNull())
            self.assertTrue(parent_item.data(0, COLLECTION_SEARCH_EXCLUDED_DIRECT_ROLE))
            self.assertEqual(parent_item.data(0, COLLECTION_SEARCH_EXCLUDED_MATCH_ROLE), parent_id)
            self.assertTrue(child_item.text(0).endswith("⊘"))
            self.assertFalse(child_item.data(0, COLLECTION_SEARCH_EXCLUDED_DIRECT_ROLE))
            self.assertEqual(child_item.data(0, COLLECTION_SEARCH_EXCLUDED_MATCH_ROLE), parent_id)
            self.assertCountEqual(
                [image.id for image in window.grid_view.images()],
                [keep_id, excluded_parent_id, excluded_child_id],
            )

            window.collection_tree.setCurrentItem(parent_item)
            self.app.processEvents()
            self.assertCountEqual(
                [image.id for image in window.grid_view.images()],
                [excluded_parent_id, excluded_child_id],
            )

            search_result = window._compute_search_chain(
                filters=(SearchFilter("keyword", "parent"),),
                folder_path_prefix=None,
                collection_id=None,
                tag_ids=[],
                tag_match_mode="any",
                status_filter=None,
                virtual_filter=None,
                excluded_folder_path_prefixes=window._search_excluded_folder_prefixes(),
                excluded_collection_ids=window._search_excluded_collection_ids(),
            )
            self.assertEqual(search_result.images, [])
            scoped_result = window._compute_search_chain(
                filters=(SearchFilter("keyword", "parent"),),
                folder_path_prefix=None,
                collection_id=None,
                tag_ids=[],
                tag_match_mode="any",
                status_filter=None,
                virtual_filter=None,
                excluded_folder_path_prefixes=window._search_excluded_folder_prefixes(),
                excluded_collection_ids=window._search_excluded_collection_ids(),
                base_image_ids={excluded_parent_id, excluded_child_id},
            )
            self.assertEqual(scoped_result.images, [])
            merge_result = window._compute_search_chain(
                filters=(SearchFilter("keyword", "keep"),),
                folder_path_prefix=None,
                collection_id=None,
                tag_ids=[],
                tag_match_mode="any",
                status_filter=None,
                virtual_filter=None,
                excluded_folder_path_prefixes=window._search_excluded_folder_prefixes(),
                excluded_collection_ids=window._search_excluded_collection_ids(),
                merge_base_images=store.images_by_ids([excluded_parent_id]),
            )
            self.assertEqual([image.id for image in merge_result.images], [keep_id])

            window._set_collection_search_excluded(parent_id, excluded=False)
            self.app.processEvents()

            refreshed_parent = self._collection_item(window.collection_tree, parent_id)
            self.assertIsNotNone(refreshed_parent)
            assert refreshed_parent is not None
            _menu, actions = window._build_collection_context_menu(refreshed_parent)
            self.assertTrue(actions["exclude_from_search"].isEnabled())
            self.assertFalse(actions["include_in_search"].isEnabled())
            self.assertCountEqual(
                [image.id for image in window.grid_view.images()],
                [excluded_parent_id, excluded_child_id],
            )
            window._set_collection_search_excluded(parent_id, excluded=True)
            self.app.processEvents()

            self.assertIn(parent_id, store.list_search_excluded_collection_ids())
            reexcluded_parent = self._collection_item(window.collection_tree, parent_id)
            self.assertIsNotNone(reexcluded_parent)
            assert reexcluded_parent is not None
            self.assertTrue(reexcluded_parent.text(0).endswith("⊘"))
            self.assertTrue(reexcluded_parent.data(0, COLLECTION_SEARCH_EXCLUDED_DIRECT_ROLE))
            window.close()

    def test_folder_search_exclusion_marks_tree_filters_search_not_browsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            root = Path(tmp) / "library"
            child = root / "ML-01"
            child.mkdir(parents=True)
            folder_id = store.add_folder(str(root))
            keep_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(root / "keep.jpg"),
                file_size=10,
                width=100,
                height=80,
                created_time_ns=None,
                modified_time_ns=1,
            )
            excluded_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(child / "excluded.jpg"),
                file_size=10,
                width=100,
                height=80,
                created_time_ns=None,
                modified_time_ns=2,
            )
            store.add_search_excluded_folder_prefix(str(child))
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window._reload_images()
            self.app.processEvents()

            child_item = None
            for top_index in range(window.folder_tree.topLevelItemCount()):
                top_item = window.folder_tree.topLevelItem(top_index)
                stack = [top_item]
                while stack:
                    item = stack.pop()
                    if item.data(0, Qt.ItemDataRole.UserRole + 2) == str(child):
                        child_item = item
                        break
                    stack.extend(item.child(index) for index in range(item.childCount()))
                if child_item is not None:
                    break

            self.assertIsNotNone(child_item)
            assert child_item is not None
            self.assertTrue(child_item.text(0).endswith("⊘"))
            self.assertTrue(child_item.data(0, FOLDER_SEARCH_EXCLUDED_DIRECT_ROLE))
            self.assertEqual(
                child_item.foreground(0).color().name(),
                FolderTreeItemDelegate.EXCLUDED_NAME_COLOR.name(),
            )
            self.assertFalse(child_item.icon(0).isNull())
            self.assertCountEqual(
                [image.id for image in window.grid_view.images()],
                [keep_id, excluded_id],
            )

            window.folder_tree.setCurrentItem(child_item)
            self.app.processEvents()
            self.assertEqual([image.id for image in window.grid_view.images()], [excluded_id])

            search_result = window._compute_search_chain(
                filters=(SearchFilter("keyword", "excluded"),),
                folder_path_prefix=None,
                collection_id=None,
                tag_ids=[],
                tag_match_mode="any",
                status_filter=None,
                virtual_filter=None,
                excluded_folder_path_prefixes=window._search_excluded_folder_prefixes(),
                excluded_collection_ids=window._search_excluded_collection_ids(),
            )
            self.assertEqual(search_result.images, [])
            scoped_result = window._compute_search_chain(
                filters=(SearchFilter("keyword", "excluded"),),
                folder_path_prefix=None,
                collection_id=None,
                tag_ids=[],
                tag_match_mode="any",
                status_filter=None,
                virtual_filter=None,
                excluded_folder_path_prefixes=window._search_excluded_folder_prefixes(),
                excluded_collection_ids=window._search_excluded_collection_ids(),
                base_image_ids={excluded_id},
            )
            self.assertEqual(scoped_result.images, [])
            merge_result = window._compute_search_chain(
                filters=(SearchFilter("keyword", "keep"),),
                folder_path_prefix=None,
                collection_id=None,
                tag_ids=[],
                tag_match_mode="any",
                status_filter=None,
                virtual_filter=None,
                excluded_folder_path_prefixes=window._search_excluded_folder_prefixes(),
                excluded_collection_ids=window._search_excluded_collection_ids(),
                merge_base_images=store.images_by_ids([excluded_id]),
            )
            self.assertEqual([image.id for image in merge_result.images], [keep_id])

            window._set_folder_search_excluded(str(child), excluded=False)
            self.app.processEvents()

            self.assertCountEqual(
                [image.id for image in window.grid_view.images()],
                [excluded_id],
            )
            window._set_folder_search_excluded(str(child), excluded=True)
            self.app.processEvents()

            self.assertIn(
                window._normalize_folder_path(str(child)),
                store.list_search_excluded_folder_prefixes(),
            )
            direct, matched_prefix = window._folder_search_exclusion_state(str(child))
            self.assertTrue(direct)
            self.assertEqual(matched_prefix, window._normalize_folder_path(str(child)))
            window.close()

    def test_folder_tree_import_preserves_disk_directory_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            disk_root = Path(tmp) / "精选练习素材 CC0 Eidory"
            child = disk_root / "ML-04 简单小景"
            nested = child / "子目录"
            nested.mkdir(parents=True)
            first_path = child / "a.jpg"
            second_path = nested / "b.jpg"
            first_path.write_bytes(b"fake image")
            second_path.write_bytes(b"fake image")
            folder_id = store.add_folder(str(disk_root))
            first_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(first_path),
                file_size=10,
                width=100,
                height=80,
                created_time_ns=None,
                modified_time_ns=1,
            )
            second_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(second_path),
                file_size=10,
                width=100,
                height=80,
                created_time_ns=None,
                modified_time_ns=2,
            )
            result = ScanResult(
                folder_id=folder_id,
                scanned_files=2,
                new_files=2,
                changed_files=0,
                unchanged_files=0,
                missing_marked=0,
                thumbnail_failures=0,
                image_ids=(first_id, second_id),
            )

            assigned = window._assign_import_result(
                result=result,
                folder_path=str(disk_root),
                collection_id=None,
                preserve_structure=True,
            )

            window.close()
            self.assertEqual(assigned, 2)
            self.assertEqual(
                store.collection_paths_for_image(first_id),
                ["精选练习素材 CC0 Eidory / ML-04 简单小景"],
            )
            self.assertEqual(
                store.collection_paths_for_image(second_id),
                ["精选练习素材 CC0 Eidory / ML-04 简单小景 / 子目录"],
            )

    def test_local_import_path_split_separates_files_and_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            folder = Path(tmp) / "ML-05 复杂场景"
            folder.mkdir()
            image = Path(tmp) / "a.jpg"
            image.write_bytes(b"fake image")
            text = Path(tmp) / "notes.txt"
            text.write_text("ignore", encoding="utf-8")

            file_paths, folder_paths = window._split_local_import_paths(
                [str(folder), str(image), str(text), str(folder)]
            )

            window.close()
            self.assertEqual(file_paths, [str(image)])
            self.assertEqual(folder_paths, [str(folder)])

    def test_video_selection_uses_embedded_video_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            video_path = Path(tmp) / "clip.mp4"
            video_path.write_bytes(b"fake mp4 bytes")
            folder_id = store.add_folder(str(Path(tmp)))
            video_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(video_path),
                file_size=video_path.stat().st_size,
                width=None,
                height=None,
                created_time_ns=None,
                modified_time_ns=video_path.stat().st_mtime_ns,
            )
            store.mark_embedding_not_required(video_id)
            store.update_thumbnail(video_id, None, "ready")
            video = store.get_image(video_id)
            self.assertIsNotNone(video)

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window._show_image_details(video)
            self.app.processEvents()

            self.assertIs(window.preview_stack.currentWidget(), window.video_widget)
            self.assertTrue(window.play_pause_button.isEnabled())
            self.assertEqual(window.embedding_label.text(), "无需语义索引")
            self.assertEqual(window.video_player.source().toLocalFile(), str(video_path))
            window.close()

    def test_embedded_preview_belongs_to_detail_tab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            detail_tab = window.right_tab_widget.widget(0)

            self.assertEqual(window.right_tab_widget.objectName(), "rightSidebarTabs")
            self.assertEqual(
                window.right_tab_widget.tabBar().tabSizeHint(0).height(),
                EqualWidthTabBar.BUTTON_MATCH_HEIGHT + EqualWidthTabBar.BOTTOM_GAP,
            )
            self.assertGreaterEqual(
                window.right_tab_widget.tabBar().tabSizeHint(0).width(),
                EqualWidthTabBar.MIN_TAB_WIDTH,
            )
            self.assertEqual(window.search_row.spacing(), TOP_TOOL_BUTTON_SPACING)
            self.assertEqual(window.right_tab_widget.parentWidget().minimumWidth(), 0)
            self.assertEqual(window.right_tab_widget.parentWidget().maximumWidth(), RIGHT_SIDEBAR_WIDTH)
            self.assertGreaterEqual(
                window.right_tab_widget.tabBar().minimumWidth(),
                EqualWidthTabBar.minimum_width_for_tab_count(window.right_tab_widget.count()),
            )
            for button in [
                window.reverse_exclusion_button,
                window.color_mode_button,
                window.keyword_mode_button,
                window.semantic_mode_button,
                window.collection_filter_button,
                window.similar_image_button,
                window.search_button,
                window.clear_search_button,
            ]:
                self.assertEqual(button.minimumWidth(), TOP_TOOL_BUTTON_MIN_WIDTH)
            self.assertEqual(window.color_mode_button.text(), "颜色")
            self.assertIn("background:", window.color_mode_button.styleSheet())
            parent = window.preview_stack.parentWidget()
            while parent is not None and parent is not detail_tab:
                parent = parent.parentWidget()
            self.assertIs(parent, detail_tab)
            self.assertEqual(window.right_tab_widget.tabText(0), "详情")
            window.close()

    def test_grid_context_menu_groups_actions_into_stable_submenus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            image = self._image(1)
            window.grid_view.set_images([image], selected_image_ids=[image.id])
            menu = window._build_grid_context_menu(
                selected_images=[image],
                context_image=image,
            )

            self.assertEqual(
                self._menu_texts(menu),
                [
                    "快速预览",
                    "对比查看",
                    "查找相似图片",
                    "文件与导出",
                    "收藏与标签",
                    "语义探针项目",
                    "创作节点项目",
                    "暂时收藏",
                    "文件夹归类",
                    "当前结果",
                ],
            )
            self.assertEqual(
                self._submenu_texts(menu, "文件与导出"),
                ["打开源文件", "在 Finder 中显示", "复制路径", "导出选中图片", "删除/移除图片..."],
            )
            self.assertEqual(
                self._submenu_texts(menu, "收藏与标签"),
                ["收藏选中 1 张", "取消收藏", "批量添加标签", "清除选中图片标签"],
            )
            self.assertEqual(
                self._submenu_texts(menu, "语义探针项目"),
                [
                    "保存选中为语义探针项目",
                    "加入已有语义探针项目",
                    "从当前语义探针项目移除",
                    "AI 分组选中图片",
                ],
            )
            self.assertEqual(
                self._submenu_texts(menu, "创作节点项目"),
                ["存入当前创作节点", "从当前节点移除"],
            )
            self.assertEqual(
                self._submenu_texts(menu, "暂时收藏"),
                ["新建暂时收藏", "加入已有暂时收藏", "从当前暂时收藏移除"],
            )
            self.assertEqual(
                self._submenu_texts(menu, "文件夹归类"),
                ["添加到文件夹", "移动到文件夹", "从当前文件夹移出"],
            )
            self.assertEqual(
                self._submenu_texts(menu, "当前结果"),
                ["保存当前搜索结果集", "从当前结果排除选中", "排除此图所在的文件夹"],
            )
            self.assertTrue(self._action_by_text(menu, "快速预览").isEnabled())
            self.assertFalse(self._action_by_text(menu, "对比查看").isEnabled())
            self.assertTrue(
                self._submenu_action_by_text(menu, "文件与导出", "导出选中图片").isEnabled()
            )
            self.assertEqual(
                self._submenu_action_by_text(menu, "文件与导出", "导出选中图片").data(),
                "export_selection",
            )
            self.assertEqual(
                self._submenu_action_by_text(menu, "文件与导出", "删除/移除图片...").data(),
                "delete_source",
            )
            self.assertEqual(window.export_selection_button.text(), "导出图片")
            self.assertTrue(window.rebuild_selected_thumbnails_button.isEnabled())
            self.assertTrue(window.remove_selected_index_button.isEnabled())
            window.close()

    def test_delete_selected_source_files_trashes_and_removes_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = AppPaths(
                data_dir=root / "data",
                thumbnail_dir=root / "data" / "thumbs",
                database_path=root / "data" / "eidory.sqlite3",
                log_dir=root / "data" / "logs",
            )
            paths.ensure()
            media_path = root / "source.jpg"
            media_path.write_bytes(b"fake")
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(media_path),
                file_size=media_path.stat().st_size,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            image = store.list_images()[0]
            window = MainWindow(paths=paths, store=store)
            window.show()
            window.grid_view.set_images([image], selected_image_ids=[image_id])
            self.app.processEvents()

            with (
                patch.object(window, "_ask_delete_or_remove_mode", return_value="source"),
                patch("eidory.ui.main_window.QFile.moveToTrash", return_value=True) as move_to_trash,
            ):
                window._delete_selected_source_files()

            move_to_trash.assert_called_once_with(str(media_path))
            self.assertEqual(store.list_images(), [])
            window.close()

    def test_delete_selected_index_can_be_undone_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = AppPaths(
                data_dir=root / "data",
                thumbnail_dir=root / "data" / "thumbs",
                database_path=root / "data" / "eidory.sqlite3",
                log_dir=root / "data" / "logs",
            )
            paths.ensure()
            media_path = root / "source.jpg"
            media_path.write_bytes(b"fake")
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(media_path),
                file_size=media_path.stat().st_size,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            collection_id = store.create_collection("测试文件夹")
            store.assign_images_to_collection([image_id], collection_id)
            store.set_image_tags(image_id, ["室内", "夜晚"])
            image = store.list_images()[0]
            window = MainWindow(paths=paths, store=store)
            window.show()
            window.grid_view.set_images([image], selected_image_ids=[image_id])
            self.app.processEvents()

            with patch.object(window, "_ask_delete_or_remove_mode", return_value="index"):
                window._delete_selected_source_files()

            self.assertTrue(media_path.exists())
            self.assertEqual(store.list_images(), [])
            self.assertTrue(window.undo_removal_action.isEnabled())

            window._undo_last_library_removal()

            restored_images = store.list_images()
            self.assertEqual([restored.id for restored in restored_images], [image_id])
            self.assertEqual(store.get_image_tags(image_id), ["夜晚", "室内"])
            self.assertEqual([item.id for item in store.list_images(collection_id=collection_id)], [image_id])
            self.assertFalse(window.undo_removal_action.isEnabled())
            window.close()

    def test_delete_selected_source_file_can_be_undone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = AppPaths(
                data_dir=root / "data",
                thumbnail_dir=root / "data" / "thumbs",
                database_path=root / "data" / "eidory.sqlite3",
                log_dir=root / "data" / "logs",
            )
            paths.ensure()
            media_path = root / "source.jpg"
            media_path.write_bytes(b"fake-source")
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(media_path),
                file_size=media_path.stat().st_size,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            image = store.list_images()[0]
            window = MainWindow(paths=paths, store=store)
            window.show()
            window.grid_view.set_images([image], selected_image_ids=[image_id])
            self.app.processEvents()

            def fake_trash(path: str) -> bool:
                Path(path).unlink()
                return True

            with (
                patch.object(window, "_ask_delete_or_remove_mode", return_value="source"),
                patch("eidory.ui.main_window.QFile.moveToTrash", side_effect=fake_trash),
            ):
                window._delete_selected_source_files()

            self.assertFalse(media_path.exists())
            self.assertEqual(store.list_images(), [])

            window._undo_last_library_removal()

            self.assertTrue(media_path.exists())
            self.assertEqual(media_path.read_bytes(), b"fake-source")
            self.assertEqual([image.id for image in store.list_images()], [image_id])
            window.close()

    def test_metadata_filter_matchers(self) -> None:
        landscape = self._image(1, file_ext=".jpg", width=1600, height=900)
        portrait = self._image(2, file_ext=".png", width=900, height=1600)
        square = self._image(3, file_ext=".webp", width=1200, height=1200)
        large = self._image(4, file_ext=".jpeg", width=3000, height=2000)
        small = self._image(5, file_ext=".jpg", width=640, height=480)
        video = self._image(6, file_ext=".mp4", width=None, height=None)

        self.assertTrue(MainWindow._image_matches_file_type(landscape, "media:image"))
        self.assertTrue(MainWindow._image_matches_file_type(video, "media:video"))
        self.assertTrue(MainWindow._image_matches_file_type(portrait, "ext:.png"))
        self.assertFalse(MainWindow._image_matches_file_type(video, "media:image"))
        self.assertTrue(MainWindow._image_matches_orientation(landscape, "landscape"))
        self.assertTrue(MainWindow._image_matches_orientation(portrait, "portrait"))
        self.assertTrue(MainWindow._image_matches_orientation(square, "square"))
        self.assertFalse(MainWindow._image_matches_orientation(video, "landscape"))
        self.assertTrue(MainWindow._image_matches_size(large, "large"))
        self.assertTrue(MainWindow._image_matches_size(small, "small"))
        self.assertFalse(MainWindow._image_matches_size(video, "small"))

    def test_score_threshold_uses_last_scored_filter_in_chain(self) -> None:
        self.assertEqual(
            last_score_filter_kind(
                [
                    SearchFilter("semantic", "车"),
                    SearchFilter("file_type", "media:image"),
                    SearchFilter("orientation", "landscape"),
                ]
            ),
            "semantic",
        )
        self.assertEqual(
            last_score_filter_kind(
                [
                    SearchFilter("semantic", "车"),
                    SearchFilter("color", (255, 0, 0)),
                    SearchFilter("size", "large"),
                ]
            ),
            "color",
        )
        self.assertEqual(
            last_score_filter_kind(
                [
                    SearchFilter("semantic", "车"),
                    SearchFilter("similar", 12),
                    SearchFilter("size", "large"),
                ]
            ),
            "similar",
        )
        ai_filter = SearchFilter("ai_vision", "scene_location:outdoor")
        self.assertEqual(search_filter_from_payload(search_filter_to_payload(ai_filter)), ai_filter)

    def test_semantic_threshold_uses_current_score_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window.current_semantic_images = [
                self._image(1, score=0.270),
                self._image(2, score=0.220),
                self._image(3, score=0.154),
            ]
            window.score_threshold_slider.setValue(43)
            window._apply_semantic_result_filters()

            self.assertEqual(
                [image.id for image in window.current_semantic_filtered_images],
                [1, 2],
            )
            self.assertIn("43%", window.score_threshold_label.text())
            window.close()

    def test_score_threshold_reexpands_from_unfiltered_source_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            source_images = [
                self._image(1, score=0.90),
                self._image(2, score=0.75),
                self._image(3, score=0.50),
                self._image(4, score=0.20),
            ]
            window.current_result_mode = "search_chain"
            window.search_filters = [SearchFilter("semantic", "水")]
            window.current_chain_images = list(source_images)
            window.grid_view.set_images(list(source_images))

            window.score_threshold_slider.setValue(100)
            self.app.processEvents()
            self.assertEqual([image.id for image in window.grid_view.images()], [1])

            window.score_threshold_slider.setValue(0)
            self.app.processEvents()
            self.assertEqual(
                [image.id for image in window.grid_view.images()],
                [1, 2, 3, 4],
            )
            self.assertEqual(
                [image.id for image in window.current_chain_images],
                [1, 2, 3, 4],
            )

            color_source_images = [
                self._image(5, score=1.00),
                self._image(6, score=0.60),
                self._image(7, score=0.25),
            ]
            window.search_filters = [SearchFilter("color", (240, 152, 196))]
            window.current_chain_images = list(color_source_images)
            window.grid_view.set_images(list(color_source_images))
            window.score_threshold_slider.setValue(100)
            self.app.processEvents()
            self.assertEqual([image.id for image in window.grid_view.images()], [5])
            window.score_threshold_slider.setValue(0)
            self.app.processEvents()
            self.assertEqual([image.id for image in window.grid_view.images()], [5, 6, 7])
            window.close()

    def test_search_chain_keeps_last_scored_filter_unthresholded_for_later_hard_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            color_images = [
                self._image(1, score=1.0, file_path="/tmp/bright-sky.jpg"),
                self._image(2, score=0.35, file_path="/tmp/soft-house.jpg"),
            ]
            window.search_service.color_search = lambda *args, **kwargs: SimpleNamespace(
                images=list(color_images),
                searchable_count=2,
                indexed_count=2,
                candidate_limit=2,
            )

            result = window._compute_search_chain(
                filters=(
                    SearchFilter("color", (240, 152, 196), 100),
                    SearchFilter("keyword", "house"),
                ),
                folder_path_prefix=None,
                collection_id=None,
                tag_ids=[],
                tag_match_mode="any",
                status_filter=None,
            )

            window.close()
            self.assertEqual([image.id for image in result.images], [2])

    def test_search_chain_applies_non_last_scored_filter_before_next_scored_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            captured_allowed_ids: list[set[int] | None] = []
            color_images = [
                self._image(1, score=1.0),
                self._image(2, score=0.55),
                self._image(3, score=0.20),
            ]
            window.search_service.color_search = lambda *args, **kwargs: SimpleNamespace(
                images=list(color_images),
                searchable_count=3,
                indexed_count=3,
                candidate_limit=3,
            )

            def semantic_search(*args, **kwargs):
                captured_allowed_ids.append(kwargs.get("allowed_image_ids"))
                allowed_ids = kwargs.get("allowed_image_ids") or set()
                return SimpleNamespace(
                    images=[self._image(image_id, score=0.8) for image_id in sorted(allowed_ids)],
                    searchable_count=len(allowed_ids),
                    candidate_limit=len(allowed_ids),
                )

            window.search_service.semantic_search = semantic_search

            window._compute_search_chain(
                filters=(
                    SearchFilter("color", (240, 152, 196), 50),
                    SearchFilter("semantic", "房屋", 0),
                ),
                folder_path_prefix=None,
                collection_id=None,
                tag_ids=[],
                tag_match_mode="any",
                status_filter=None,
            )

            window.close()
            self.assertEqual(captured_allowed_ids, [{1, 2}])

    def test_selected_search_filter_controls_score_threshold_slider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window.search_filters = [
                SearchFilter("color", (240, 152, 196), 78),
                SearchFilter("semantic", "水", 22),
            ]
            window.active_filter_index = 0
            window._refresh_filter_chain_ui()
            self.assertEqual(window.score_threshold_slider.value(), 78)
            self.assertIn("颜色相似度", window.score_threshold_label.text())

            window._select_search_filter_chip(1)
            self.assertEqual(window.score_threshold_slider.value(), 22)
            self.assertIn("语义相似度", window.score_threshold_label.text())

            window.close()

    def test_score_threshold_updates_recompute_only_for_non_last_scored_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            source_images = [
                self._image(1, score=0.9),
                self._image(2, score=0.4),
            ]
            window.current_result_mode = "search_chain"
            window.current_chain_images = list(source_images)
            window.grid_view.set_images(list(source_images))
            window.search_filters = [
                SearchFilter("color", (240, 152, 196), 80),
                SearchFilter("semantic", "水", 100),
            ]
            window.active_filter_index = 0
            with patch.object(window, "_execute_search_chain") as execute_search_chain:
                window.score_threshold_slider.setValue(30)
                self.app.processEvents()
            execute_search_chain.assert_called_once_with(operation_mode="recompute")
            self.assertEqual(window.search_filters[0].score_threshold, 30)

            window.active_filter_index = 1
            window._refresh_score_threshold_controls()
            with patch.object(window, "_execute_search_chain") as execute_search_chain:
                window.score_threshold_slider.setValue(0)
                self.app.processEvents()
            execute_search_chain.assert_not_called()
            self.assertEqual(window.search_filters[1].score_threshold, 0)
            self.assertEqual([image.id for image in window.grid_view.images()], [1, 2])
            window.close()

    def test_search_chain_refresh_recomputes_without_visible_result_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            source_images = [
                self._image(1, score=0.90),
                self._image(2, score=0.75),
                self._image(3, score=0.50),
                self._image(4, score=0.20),
            ]
            window.current_result_mode = "search_chain"
            window.search_filters = [SearchFilter("semantic", "水")]
            window.current_chain_images = list(source_images)
            window.current_chain_base_image_ids = {image.id for image in source_images}
            window.current_chain_base_label = "在当前结果中"
            window.grid_view.set_images([source_images[0]])

            with patch.object(window, "_execute_search_chain") as execute_search_chain:
                window._refresh_current_results_for_filters()

            execute_search_chain.assert_called_once_with(operation_mode="recompute")
            base_ids, base_label, merge_base_images = window._search_operation_context("recompute")
            self.assertEqual(base_ids, {1, 2, 3, 4})
            self.assertEqual(base_label, "在当前结果中")
            self.assertIsNone(merge_base_images)
            window.close()

    def test_score_threshold_reexpands_color_and_inspiration_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            color_images = [
                self._image(1, score=1.00),
                self._image(2, score=0.65),
                self._image(3, score=0.30),
            ]
            window.current_result_mode = "color"
            window.current_color_images = list(color_images)
            window.grid_view.set_images(list(color_images))
            window.score_threshold_slider.setValue(100)
            self.app.processEvents()
            self.assertEqual([image.id for image in window.grid_view.images()], [1])
            window.score_threshold_slider.setValue(0)
            self.app.processEvents()
            self.assertEqual([image.id for image in window.grid_view.images()], [1, 2, 3])

            term = InspirationTerm(
                id=None,
                title="水面",
                query="水面",
                axis="environment",
                reason="",
                selected=True,
            )
            inspiration_images = [
                self._image(4, score=0.80),
                self._image(5, score=0.55),
                self._image(6, score=0.10),
            ]
            window.current_result_mode = "inspiration"
            window.current_inspiration_terms = [term]
            window.current_inspiration_images = list(inspiration_images)
            window.current_inspiration_matches = {
                image.id: [InspirationMatch(term_title="水面", query="水面", score=image.score or 0, reason="")]
                for image in inspiration_images
            }
            window.grid_view.set_images(list(inspiration_images))
            window.score_threshold_slider.setValue(100)
            self.app.processEvents()
            self.assertEqual([image.id for image in window.grid_view.images()], [4])
            window.score_threshold_slider.setValue(0)
            self.app.processEvents()
            self.assertEqual([image.id for image in window.grid_view.images()], [4, 5, 6])
            window.close()

    def test_inspiration_plan_filters_use_or_inside_field_and_and_between_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_ids = []
            for index in range(4):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=123,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)
            analyses = [
                self._ai_vision_analysis(scene_location="indoor", time_of_day="night"),
                self._ai_vision_analysis(scene_location="threshold", time_of_day="night"),
                self._ai_vision_analysis(scene_location="outdoor", time_of_day="night"),
                self._ai_vision_analysis(scene_location="indoor", time_of_day="day"),
            ]
            for image_id, analysis in zip(image_ids, analyses, strict=True):
                store.upsert_ai_vision_success(
                    image_id=image_id,
                    provider_name="LM Studio",
                    model_name="vision-model",
                    prompt_version=AI_VISION_PROMPT_VERSION,
                    analysis=analysis,
                    source_modified_time_ns=image_id,
                )
            window = MainWindow(paths=paths, store=store)
            images = [self._image(image_id) for image_id in image_ids]
            plan_filters = [
                SearchPlanFilter("scene_location", "indoor"),
                SearchPlanFilter("scene_location", "threshold"),
                SearchPlanFilter("time_of_day", "night"),
            ]

            filtered = window._apply_inspiration_plan_filters_to_images(
                images,
                plan_filters,
            )

            self.assertEqual([image.id for image in filtered], image_ids[:2])
            scene_term = InspirationTerm(title="夜晚室内环境", query="夜晚室内环境", axis="environment")
            object_term = InspirationTerm(title="摩托车造型", query="摩托车结构细节", axis="object_detail")
            self.assertEqual(
                [
                    image.id
                    for image in window._images_for_inspiration_term_with_plan_filters(
                        scene_term,
                        images,
                        plan_filters,
                    )
                ],
                image_ids[:2],
            )
            self.assertEqual(
                [
                    image.id
                    for image in window._images_for_inspiration_term_with_plan_filters(
                        object_term,
                        images,
                        plan_filters,
                    )
                ],
                image_ids,
            )

            window.current_inspiration_terms = [scene_term]
            window.current_inspiration_raw_term_results = [(scene_term, images)]
            window.current_inspiration_plan_filters = plan_filters
            window._rebuild_current_inspiration_results_from_raw()
            self.assertEqual([image.id for image in window.current_inspiration_images], image_ids[:2])
            window.current_inspiration_plan_filters = plan_filters[:2]
            window._rebuild_current_inspiration_results_from_raw()
            self.assertEqual(
                [image.id for image in window.current_inspiration_images],
                [image_ids[0], image_ids[1], image_ids[3]],
            )
            window.close()

    def test_batch_tag_summary_and_tag_input_parsing(self) -> None:
        self.assertEqual(
            MainWindow._parse_tag_input(" 室内, 夜晚，室内\n人物 "),
            ["室内", "夜晚", "人物"],
        )
        self.assertEqual(
            MainWindow._format_batch_tag_summary(
                total=3,
                common=["室内"],
                partial=[("夜晚", 2), ("人物", 1)],
                no_tag_count=1,
            ),
            "共同标签：室内\n部分标签：夜晚 (2/3)、人物 (1/3)\n无标签：1",
        )

    def test_result_management_excludes_selected_items_in_temporary_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_ids = []
            for index in range(3):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=123,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)
            project_id = store.create_temporary_project("结果", image_ids)
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window._load_temporary_project(project_id)
            window.grid_view.set_images(window.grid_view.images(), selected_image_ids=[image_ids[1]])
            window._exclude_selection_from_results()
            self.app.processEvents()

            self.assertEqual([image.id for image in window.grid_view.images()], [image_ids[0], image_ids[2]])
            self.assertEqual(window.result_excluded_image_ids, {image_ids[1]})
            self.assertIn("排除 1", window.result_state_label.text())
            window._clear_result_exclusions()
            self.assertEqual([image.id for image in window.grid_view.images()], image_ids)
            window.close()

    def test_result_management_excludes_collection_chain_from_current_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            root_id = store.create_collection("具体造型")
            role_id = store.create_collection("角色", root_id)
            anatomy_id = store.create_collection("人体结构", role_id)
            other_id = store.create_collection("创作参考")
            image_ids = []
            for index in range(3):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=123,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)
            store.assign_images_to_collection([image_ids[0]], anatomy_id)
            store.assign_images_to_collection([image_ids[1]], role_id)
            store.assign_images_to_collection([image_ids[2]], other_id)
            chains = store.collection_chains_for_image(image_ids[0])
            self.assertEqual([[item.name for item in chain] for chain in chains], [["具体造型", "角色", "人体结构"]])
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            color_images = [self._image(image_id, score=1.0) for image_id in image_ids]
            window.current_result_mode = "color"
            window.current_color_images = color_images
            window.current_color_filtered_images = list(color_images)
            window.search_button.setEnabled(False)
            window.grid_view.set_images(color_images)
            window._exclude_collection_from_results(role_id)
            self.app.processEvents()

            self.assertEqual([image.id for image in window.grid_view.images()], [image_ids[2]])
            self.assertEqual(window.result_excluded_collection_ids, {role_id})
            self.assertIn("排除文件夹 1", window.result_state_label.text())
            filter_buttons = [button.text() for button in window.filter_chain_widget.findChildren(QPushButton)]
            self.assertIn("× 排除文件夹：具体造型 / 角色", filter_buttons)
            window.score_threshold_slider.setValue(80)
            self.app.processEvents()
            self.assertEqual([image.id for image in window.grid_view.images()], [image_ids[2]])
            window._remove_result_collection_exclusion(role_id)
            self.assertEqual([image.id for image in window.grid_view.images()], image_ids)
            window.close()

    def test_reverse_keyword_exclusion_subtracts_from_current_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            anatomy_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "anatomy-reference.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            landscape_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "pink-landscape.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=2,
            )
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            source_images = store.images_by_ids([anatomy_id, landscape_id])
            window.current_result_mode = "search_chain"
            window.search_filters = [SearchFilter("color", (240, 152, 196))]
            window.current_chain_images = source_images
            window.grid_view.set_images(source_images)
            exclusion_filter = SearchFilter("keyword", "anatomy")
            matches = window._compute_result_exclusion_filter_matches(
                exclusion_filter,
                {anatomy_id, landscape_id},
            )
            window._add_result_exclusion_filter_from_matches(exclusion_filter, matches)
            self.app.processEvents()

            self.assertEqual([image.id for image in window.grid_view.images()], [landscape_id])
            self.assertEqual(window.result_exclusion_filters, [exclusion_filter])
            filter_buttons = [button.text() for button in window.filter_chain_widget.findChildren(QPushButton)]
            self.assertIn("× 反向排除：关键词：anatomy", filter_buttons)
            self.assertIn("反向排除 1 项/1 张", window.result_state_label.text())
            window._remove_result_exclusion_filter(0)
            self.assertEqual([image.id for image in window.grid_view.images()], [anatomy_id, landscape_id])
            window.close()

    def test_reverse_color_exclusion_tracks_score_threshold_slider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            window.current_result_mode = "search_chain"
            window.search_filters = [SearchFilter("semantic", "粉色")]
            window.current_chain_images = [
                self._image(1, score=0.90),
                self._image(2, score=0.90),
                self._image(3, score=0.90),
            ]
            exclusion_filter = SearchFilter("color", (240, 152, 196))
            window.result_exclusion_filters = [exclusion_filter]
            window.result_exclusion_filter_matches = {
                exclusion_filter: [
                    self._image(1, score=0.90),
                    self._image(2, score=0.40),
                ]
            }
            window.score_threshold_slider.setValue(50)
            self.app.processEvents()
            self.assertEqual([image.id for image in window.grid_view.images()], [2, 3])

            window.score_threshold_slider.setValue(30)
            self.app.processEvents()
            self.assertEqual([image.id for image in window.grid_view.images()], [3])
            window.close()

    def test_tag_filter_search_chain_matches_selected_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            first, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            second, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "second.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=2,
            )
            third, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "third.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=3,
            )
            store.set_image_tags(first, ["室内", "夜晚"])
            store.set_image_tags(second, ["室内"])
            store.set_image_tags(third, ["机械"])
            indoor_id = self._tag_id(store, "室内")
            night_id = self._tag_id(store, "夜晚")
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            tag_filter = SearchFilter(
                "tag",
                window._tag_filter_value([indoor_id, night_id], "all"),
            )
            result = window._compute_search_chain(
                filters=(tag_filter,),
                folder_path_prefix=None,
                collection_id=None,
                tag_ids=[],
                tag_match_mode="any",
                status_filter=None,
            )

            self.assertEqual([image.id for image in result.images], [first])
            self.assertEqual(window._filter_label(tag_filter), "标签：全部：室内 + 夜晚")
            window.close()

    def test_reverse_tag_exclusion_subtracts_matching_tags_from_current_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            indoor, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "indoor.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            night, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "night.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=2,
            )
            mechanical, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "mechanical.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=3,
            )
            store.set_image_tags(indoor, ["室内"])
            store.set_image_tags(night, ["夜晚"])
            store.set_image_tags(mechanical, ["机械"])
            indoor_id = self._tag_id(store, "室内")
            night_id = self._tag_id(store, "夜晚")
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            source_images = store.images_by_ids([indoor, night, mechanical])
            window.current_result_mode = "search_chain"
            window.search_filters = [SearchFilter("keyword", "")]
            window.current_chain_images = source_images
            window.grid_view.set_images(source_images)
            exclusion_filter = SearchFilter(
                "tag",
                window._tag_filter_value([indoor_id, night_id], "any"),
            )
            matches = window._compute_result_exclusion_filter_matches(
                exclusion_filter,
                {indoor, night, mechanical},
            )
            window._add_result_exclusion_filter_from_matches(exclusion_filter, matches)
            self.app.processEvents()

            self.assertEqual([image.id for image in window.grid_view.images()], [mechanical])
            self.assertIn("反向排除 1 项/2 张", window.result_state_label.text())
            window.close()

    def test_visible_result_set_can_be_saved_as_temporary_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_ids = []
            for index in range(2):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=123,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window.current_result_mode = "search_chain"
            saved_order = list(reversed(image_ids))
            window.grid_view.set_images(store.images_by_ids(saved_order))
            window.score_threshold_slider.setValue(64)
            window.search_filters = [SearchFilter("color", (240, 152, 196), 64)]
            window.active_filter_index = 0

            with patch(
                "eidory.ui.main_window.QInputDialog.getText",
                return_value=("当前结果", True),
            ):
                window._save_current_visible_results_as_search_result_set()

            project = store.list_temporary_projects()[0]
            self.assertEqual(project.name, "当前结果")
            self.assertEqual(project.kind, "search")
            self.assertEqual(store.temporary_project_image_ids(project.id), saved_order)
            state_payload = json.loads(str(store.get_temporary_project_state(project.id)))
            self.assertEqual(state_payload["visible_image_ids"], saved_order)
            self.assertEqual(state_payload["view"]["score_threshold"], 64)
            self.assertEqual(
                state_payload["view"]["search_filters"],
                [{"kind": "color", "score_threshold": 64, "value": [240, 152, 196]}],
            )
            self.assertTrue(window.project_sidebar_expanded_sections["search"])
            self.assertFalse(window.project_sidebar_expanded_sections["creative"])
            self.assertFalse(window.project_sidebar_expanded_sections["temporary"])
            self.assertFalse(window.project_sidebar_expanded_sections["quick"])
            search_item = next(
                item
                for item in (window.temp_project_list.item(index) for index in range(window.temp_project_list.count()))
                if item.data(PROJECT_LIST_KIND_ROLE) == "search"
            )
            self.assertEqual(search_item.data(PROJECT_LIST_ID_ROLE), project.id)
            self.assertIsNone(window.current_temp_project_id)
            self.assertNotEqual(window.center_result_stack.currentWidget(), window.project_board_view)
            window.search_filters = []
            window.score_threshold_slider.setValue(0)
            window.center_result_stack.setCurrentWidget(window.project_board_view)
            window.temp_project_list.setCurrentItem(search_item)
            with patch.object(window, "_execute_search_chain") as execute_search:
                window._load_selected_temporary_project()
                execute_search.assert_called_once()
            self.assertEqual(window.current_temp_project_id, project.id)
            self.assertEqual(window.current_search_result_set_project_id, project.id)
            self.assertEqual(window.search_filters, [SearchFilter("color", (240, 152, 196), 64)])
            self.assertEqual(window.score_threshold_slider.value(), 64)
            self.assertEqual(window.center_result_stack.currentWidget(), window.grid_view)
            window._show_current_project_board()
            self.assertEqual(window.center_result_stack.currentWidget(), window.grid_view)
            self.assertIn("搜索结果集只使用图片墙", window.statusBar().currentMessage())
            window.close()

    def test_adding_selection_to_quick_project_expands_quick_section_without_opening_board(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            first_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            second_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "second.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=2,
            )
            project_id = store.create_temporary_project("临时收藏", [first_id], kind="quick")
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window.grid_view.set_images(store.images_by_ids([second_id]), selected_image_ids=[second_id])

            window._add_selection_to_temporary_project(project_id)

            self.assertEqual(store.temporary_project_image_ids(project_id), [first_id, second_id])
            self.assertTrue(window.project_sidebar_expanded_sections["quick"])
            self.assertFalse(window.project_sidebar_expanded_sections["creative"])
            self.assertFalse(window.project_sidebar_expanded_sections["temporary"])
            quick_item = next(
                item
                for item in (window.temp_project_list.item(index) for index in range(window.temp_project_list.count()))
                if item.data(PROJECT_LIST_KIND_ROLE) == "quick"
            )
            self.assertEqual(quick_item.data(PROJECT_LIST_ID_ROLE), project_id)
            self.assertNotEqual(window.center_result_stack.currentWidget(), window.project_board_view)
            window.close()

    def test_saving_selection_to_creative_node_expands_creative_section_without_opening_board(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            project_id = store.create_creative_project(
                title="故事项目",
                brief="一个室内场景",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=int(root_id), show_board=False)
            window.grid_view.set_images(store.images_by_ids([image_id]), selected_image_ids=[image_id])

            window._save_selection_to_current_creative_node()

            self.assertEqual(store.creative_node_image_ids(int(root_id)), [image_id])
            self.assertTrue(window.project_sidebar_expanded_sections["creative"])
            self.assertFalse(window.project_sidebar_expanded_sections["temporary"])
            self.assertFalse(window.project_sidebar_expanded_sections["quick"])
            creative_item = next(
                item
                for item in (window.temp_project_list.item(index) for index in range(window.temp_project_list.count()))
                if item.data(PROJECT_LIST_KIND_ROLE) == "creative"
            )
            self.assertEqual(creative_item.data(PROJECT_LIST_ID_ROLE), project_id)
            self.assertNotEqual(window.center_result_stack.currentWidget(), window.project_board_view)
            window.close()

    def test_creative_node_note_input_auto_saves_user_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            project_id = store.create_creative_project(
                title="故事项目",
                brief="空间站里的几个航天员在吃饭",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=int(root_id), show_board=False)

            window.creative_node_note_input.setPlainText("用户确认：晚餐时间，舷窗外有蓝色地球光。")
            window.creative_node_query_input.setText("空间站餐桌 晚餐 蓝色地球光")
            window._save_pending_creative_node_details()

            node = store.get_creative_node(int(root_id))
            self.assertIsNotNone(node)
            assert node is not None
            self.assertEqual(node.note, "用户确认：晚餐时间，舷窗外有蓝色地球光。")
            self.assertEqual(node.search_query, "空间站餐桌 晚餐 蓝色地球光")
            window.close()

    def test_creative_node_ai_completion_preserves_user_written_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            project_id = store.create_creative_project(
                title="故事项目",
                brief="空间站里的几个航天员在吃饭",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=int(root_id), show_board=False)

            user_note = "用户确认：晚餐时间，舷窗外有蓝色地球光。"
            suggestion = SimpleNamespace(
                note="空间站餐区有漂浮餐具、柔和舱内灯和窗外飞船剪影。",
                search_query="空间站餐区 漂浮餐具 舱内灯 飞船剪影",
            )
            window._handle_creative_node_note_done(int(root_id), user_note, suggestion, "fake")

            node = store.get_creative_node(int(root_id))
            self.assertIsNotNone(node)
            assert node is not None
            self.assertIn(user_note, node.note)
            self.assertIn("漂浮餐具", node.note)
            self.assertEqual(node.search_query, "空间站餐区 漂浮餐具 舱内灯 飞船剪影")
            window.close()

    def test_creative_node_ai_completion_rejects_placeholder_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            project_id = store.create_creative_project(
                title="故事项目",
                brief="空间站里的几个航天员在吃饭",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            user_note = "白天，但应该符合太空气氛，空间站外部应该是黑的"
            store.update_creative_node(int(root_id), note=user_note, search_query="太空白天 空间站外部黑色")
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=int(root_id), show_board=False)

            suggestion = SimpleNamespace(note="...", search_query="...")
            window._handle_creative_node_note_done(int(root_id), user_note, suggestion, "fake")

            node = store.get_creative_node(int(root_id))
            self.assertIsNotNone(node)
            assert node is not None
            self.assertEqual(node.note, user_note)
            self.assertNotIn("...", node.note)
            self.assertIn("AI没有返回可用节点补充内容", window.statusBar().currentMessage())
            window.close()

    def test_creative_node_ai_completion_rejects_reasoning_process_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            project_id = store.create_creative_project(
                title="故事项目",
                brief="空间站里的几个航天员在吃饭",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            user_note = "白天，但应该符合太空气氛，空间站外部应该是黑的"
            store.update_creative_node(int(root_id), note=user_note, search_query="太空白天 空间站外部黑色")
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=int(root_id), show_board=False)

            suggestion = SimpleNamespace(
                note=(
                    "Here's a thinking process: 1. **Analyze User Input:** "
                    "Project Theme: 空间站里的几个航天员在吃饭. "
                    "2. **Deconstruct Constraints:** Keep daytime and black space outside."
                ),
                search_query="Analyze User Input space station",
            )
            window._handle_creative_node_note_done(int(root_id), user_note, suggestion, "fake")

            node = store.get_creative_node(int(root_id))
            self.assertIsNotNone(node)
            assert node is not None
            self.assertEqual(node.note, user_note)
            self.assertNotIn("thinking process", node.note.lower())
            self.assertIn("AI没有返回可用节点补充内容", window.statusBar().currentMessage())
            window.close()

    def test_creative_node_tree_marks_nodes_with_non_template_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            project_id = store.create_creative_project(
                title="故事项目",
                brief="空间站里的几个航天员在吃饭",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            empty_child_id = store.create_creative_node(
                project_id=project_id,
                parent_id=int(root_id),
                title="空节点",
                note="",
                search_query="空节点",
            )
            template_child_id = store.create_creative_node(
                project_id=project_id,
                parent_id=int(root_id),
                title="模板提示节点",
                note="季节、昼夜、事件发生的具体时刻。",
                search_query="模板提示节点",
            )
            filled_child_id = store.create_creative_node(
                project_id=project_id,
                parent_id=int(root_id),
                title="已思考节点",
                note="用户确认：晚餐时间，舷窗外有蓝色地球光。",
                search_query="空间站晚餐 蓝色地球光",
            )
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=int(root_id), show_board=False)

            root_item = window._find_creative_node_tree_item(int(root_id))
            empty_item = window._find_creative_node_tree_item(empty_child_id)
            template_item = window._find_creative_node_tree_item(template_child_id)
            filled_item = window._find_creative_node_tree_item(filled_child_id)
            self.assertIsNotNone(root_item)
            self.assertIsNotNone(empty_item)
            self.assertIsNotNone(template_item)
            self.assertIsNotNone(filled_item)
            assert root_item is not None
            assert empty_item is not None
            assert template_item is not None
            assert filled_item is not None

            self.assertTrue(root_item.data(0, CREATIVE_NODE_HAS_NOTE_ROLE))
            self.assertFalse(empty_item.data(0, CREATIVE_NODE_HAS_NOTE_ROLE))
            self.assertFalse(template_item.data(0, CREATIVE_NODE_HAS_NOTE_ROLE))
            self.assertTrue(filled_item.data(0, CREATIVE_NODE_HAS_NOTE_ROLE))
            self.assertFalse(filled_item.icon(0).isNull())

            window.current_creative_node_id = empty_child_id
            window.creative_node_note_input.setPlainText("用户新增：餐具漂浮在半空。")
            window._save_pending_creative_node_details()
            self.assertTrue(empty_item.data(0, CREATIVE_NODE_HAS_NOTE_ROLE))
            self.assertFalse(empty_item.icon(0).isNull())
            window.close()

    def test_shuffle_current_grid_images_preserves_badges_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            images = [self._image(1), self._image(2), self._image(3)]
            window.grid_view.set_images(
                images,
                selected_image_ids=[2],
                badges_by_image_id={2: ["机械"], 3: ["室内"]},
            )

            with patch("eidory.ui.main_window.random.shuffle", side_effect=lambda values: values.reverse()):
                window._shuffle_current_grid_images()

            self.assertEqual([image.id for image in window.grid_view.images()], [3, 2, 1])
            self.assertEqual(window.manual_result_order_ids, [3, 2, 1])
            self.assertEqual(window.grid_view.selected_image_ids(), [2])
            self.assertEqual(window.grid_view._badges_by_image_id, {2: ["机械"], 3: ["室内"]})

            window.current_result_mode = "search_chain"
            window.search_filters = [SearchFilter("keyword", "x")]
            window.current_chain_images = list(images)
            window.current_chain_filtered_images = list(images)
            window._refresh_visible_results_after_result_management_change()

            self.assertEqual([image.id for image in window.grid_view.images()], [3, 2, 1])
            window.close()

    def test_background_embedding_progress_does_not_reset_current_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window.current_result_mode = "library"
            window.grid_view.set_images([self._image(1), self._image(2)])

            with patch.object(window, "_reload_images") as reload_images:
                window._handle_embedding_progress(EmbeddingProgress(None, None, "idle", "idle"))
                for _index in range(25):
                    window._handle_embedding_progress(EmbeddingProgress(1, "1.jpg", "ready", "ready"))

            reload_images.assert_not_called()
            self.assertEqual([image.id for image in window.grid_view.images()], [1, 2])
            window.close()

    def test_project_sidebar_groups_temporary_and_creative_projects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            temporary_id = store.create_temporary_project("临时参考", [image_id], kind="semantic")
            quick_id = store.create_temporary_project("临时收藏", [image_id], kind="quick")
            search_id = store.create_temporary_project("搜索结果", [image_id], kind="search")
            creative_id = store.create_creative_project(
                title="故事项目",
                brief="一个室内场景",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(creative_id)
            self.assertIsNotNone(root_id)
            store.add_images_to_creative_node(int(root_id), [image_id], intent_label="世界观")

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            texts = [window.temp_project_list.item(index).text() for index in range(window.temp_project_list.count())]
            self.assertEqual(texts, ["创作节点项目", "语义探针项目", "暂时收藏", "搜索结果集"])
            self.assertEqual(
                [
                    window.temp_project_list.item(index).data(PROJECT_LIST_SECTION_ID_ROLE)
                    for index in range(window.temp_project_list.count())
                ],
                ["creative", "temporary", "quick", "search"],
            )
            self.assertEqual(
                [
                    window.temp_project_list.item(index).data(PROJECT_LIST_KIND_ROLE)
                    for index in range(window.temp_project_list.count())
                    if window.temp_project_list.item(index).data(PROJECT_LIST_ID_ROLE) is not None
                ],
                [],
            )
            self._expand_project_sidebar_section(window, "creative")
            creative_item = next(
                item
                for item in (window.temp_project_list.item(index) for index in range(window.temp_project_list.count()))
                if item.data(PROJECT_LIST_KIND_ROLE) == "creative"
            )
            self.assertEqual(creative_item.data(PROJECT_LIST_ID_ROLE), creative_id)
            self.assertFalse(window.project_sidebar_expanded_sections["temporary"])
            self.assertFalse(window.project_sidebar_expanded_sections["quick"])
            self.assertFalse(window.project_sidebar_expanded_sections["search"])
            self._expand_project_sidebar_section(window, "temporary")
            temporary_item = next(
                item
                for item in (window.temp_project_list.item(index) for index in range(window.temp_project_list.count()))
                if item.data(PROJECT_LIST_KIND_ROLE) == "temporary"
            )
            self.assertEqual(temporary_item.data(PROJECT_LIST_ID_ROLE), temporary_id)
            self.assertFalse(window.project_sidebar_expanded_sections["creative"])
            self.assertFalse(window.project_sidebar_expanded_sections["quick"])
            self.assertFalse(window.project_sidebar_expanded_sections["search"])
            self._expand_project_sidebar_section(window, "quick")
            quick_item = next(
                item
                for item in (window.temp_project_list.item(index) for index in range(window.temp_project_list.count()))
                if item.data(PROJECT_LIST_KIND_ROLE) == "quick"
            )
            self.assertEqual(quick_item.data(PROJECT_LIST_ID_ROLE), quick_id)
            self.assertIn("临时收藏", quick_item.text())
            self.assertFalse(window.project_sidebar_expanded_sections["creative"])
            self.assertFalse(window.project_sidebar_expanded_sections["temporary"])
            self.assertFalse(window.project_sidebar_expanded_sections["search"])
            self._expand_project_sidebar_section(window, "search")
            search_item = next(
                item
                for item in (window.temp_project_list.item(index) for index in range(window.temp_project_list.count()))
                if item.data(PROJECT_LIST_KIND_ROLE) == "search"
            )
            self.assertEqual(search_item.data(PROJECT_LIST_ID_ROLE), search_id)
            self.assertIn("搜索结果", search_item.text())
            self.assertFalse(window.project_sidebar_expanded_sections["creative"])
            self.assertFalse(window.project_sidebar_expanded_sections["temporary"])
            self.assertFalse(window.project_sidebar_expanded_sections["quick"])
            window.close()

    def test_selecting_creative_project_from_project_sidebar_opens_board(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            project_id = store.create_creative_project(
                title="故事项目",
                brief="一个室内场景",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            store.add_images_to_creative_node(int(root_id), [image_id], intent_label="世界观")

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            self._expand_project_sidebar_section(window, "creative")
            creative_item = next(
                item
                for item in (window.temp_project_list.item(index) for index in range(window.temp_project_list.count()))
                if item.data(PROJECT_LIST_KIND_ROLE) == "creative"
            )

            window.temp_project_list.setCurrentItem(creative_item)
            self.app.processEvents()

            self.assertEqual(window.current_creative_project_id, project_id)
            self.assertEqual(window.center_result_stack.currentWidget(), window.project_board_view)
            self.assertIn("项目看板", window.result_state_label.text())
            window.close()

    def test_creative_project_sidebar_context_menu_matches_other_project_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            project_id = store.create_creative_project(
                title="故事项目",
                brief="一个室内场景",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            menu, _actions = window._build_creative_sidebar_project_context_menu()

            self.assertEqual(
                [
                    (action.text(), action.isEnabled())
                    for action in menu.actions()
                    if not action.isSeparator()
                ],
                [
                    ("打开创作节点项目", True),
                    ("编辑名称和摘要", True),
                    ("AI 重新命名和摘要", False),
                    ("上移", True),
                    ("下移", True),
                    ("删除创作节点项目", True),
                ],
            )
            window.close()

    def test_creative_project_copy_done_updates_copy_text_area(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            project_id = store.create_creative_project(
                title="雨夜维修场",
                brief="维修师在雨夜检查车辆",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=int(root_id), show_board=False)

            window._handle_creative_project_copy_done(
                project_id,
                int(root_id),
                False,
                SimpleNamespace(copy_text="雨夜维修棚里，灯箱和积水反光包围着沉默的维修师。", nodes=[]),
                "fake",
            )

            self.assertIn("雨夜维修棚", window.creative_project_copy_input.toPlainText())
            updated = store.get_creative_project(project_id)
            self.assertIsNotNone(updated)
            self.assertIn("雨夜维修棚", updated.copy_text)
            self.assertEqual(window.creative_content_tabs.currentIndex(), 1)
            window.close()

    def test_creative_node_ai_completion_requires_confirm_when_note_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            project_id = store.create_creative_project(
                title="飞行器项目",
                brief="飞行器驾驶员买饮料",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            node_id = store.create_creative_node(
                project_id=project_id,
                parent_id=int(root_id),
                title="事件",
                note="已经确认的人机交互动作。",
                search_query="人机交互动作",
            )
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=node_id, show_board=False)

            with (
                patch(
                    "eidory.ui.main_window.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.No,
                ) as question,
                patch.object(window, "_start_background_task") as start_task,
            ):
                window._generate_creative_children_for_selected_node()

            question.assert_called_once()
            start_task.assert_not_called()
            self.assertTrue(window.generate_creative_children_button.isEnabled())
            self.assertEqual(window.statusBar().currentMessage(), "已取消当前节点信息补全")
            window.close()

    def test_creative_node_completion_buttons_use_current_and_all_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            self.assertEqual(window.generate_creative_children_button.text(), "当前节点信息补全")
            self.assertEqual(window.generate_all_creative_nodes_button.text(), "补全所有节点信息")
            self.assertEqual(window.creative_template_label.text(), "新建项目模板")
            self.assertEqual(window.creative_new_project_button.text(), "按当前内容新建项目")
            self.assertEqual(window.creative_ai_project_button.text(), "AI 生成完整项目")
            self.assertEqual(window.open_creative_copy_button.text(), "文案")
            self.assertEqual(window.back_creative_nodes_button.text(), "返回节点树")
            self.assertFalse(hasattr(window, "open_creative_board_button"))
            self.assertTrue(window.generate_all_creative_nodes_button.isEnabled())
            self.assertTrue(window.creative_ai_project_button.isEnabled())
            self.assertFalse(window.open_creative_copy_button.isEnabled())
            window.close()

    def test_creative_copy_page_opens_from_bottom_node_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            project_id = store.create_creative_project(
                title="故事项目",
                brief="一个室内场景",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=int(root_id), show_board=False)

            self.assertEqual(window.creative_content_tabs.currentIndex(), 0)
            self.assertTrue(window.open_creative_copy_button.isEnabled())

            window.open_creative_copy_button.click()
            self.app.processEvents()

            self.assertEqual(window.creative_content_tabs.currentIndex(), 1)

            window.back_creative_nodes_button.click()
            self.app.processEvents()

            self.assertEqual(window.creative_content_tabs.currentIndex(), 0)
            window.close()

    def test_creative_template_preview_is_separate_from_selected_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=100,
                height=100,
                created_time_ns=None,
                modified_time_ns=1,
            )
            project_id = store.create_creative_project(
                title="旧项目",
                brief="飞行器驾驶员买饮料",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            store.add_images_to_creative_node(int(root_id), [image_id], intent_label="世界观")

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()

            self.assertIsNone(window.current_creative_project_id)
            self.assertEqual(window.creative_node_tree.topLevelItemCount(), 1)
            preview_root = window.creative_node_tree.topLevelItem(0)
            self.assertEqual(preview_root.text(0), "新创作项目")
            self.assertEqual(preview_root.child(0).text(0), "世界观")
            self.assertIsNone(window.creative_project_combo.currentData())
            self.assertIn("模板预览", window.creative_node_status_label.text())

            natural_index = window.creative_template_combo.findData("sceneNatural")
            self.assertGreaterEqual(natural_index, 0)
            window.creative_template_combo.setCurrentIndex(natural_index)
            self.app.processEvents()
            preview_root = window.creative_node_tree.topLevelItem(0)
            preview_children = [preview_root.child(index).text(0) for index in range(preview_root.childCount())]
            self.assertIn("地貌结构", preview_children)
            self.assertNotIn("人物", preview_children)

            window._load_creative_project(project_id, select_node_id=int(root_id), show_board=True)
            self.app.processEvents()
            self.assertEqual(window.current_creative_project_id, project_id)
            self.assertEqual(window.creative_node_tree.topLevelItem(0).text(0), "旧项目")
            self.assertEqual(window.center_result_stack.currentWidget(), window.project_board_view)

            object_index = window.creative_template_combo.findData("object")
            self.assertGreaterEqual(object_index, 0)
            window.creative_template_combo.setCurrentIndex(object_index)
            self.app.processEvents()
            self.assertEqual(window.creative_node_tree.topLevelItem(0).text(0), "旧项目")

            self._expand_project_sidebar_section(window, "creative")
            creative_item = next(
                item
                for item in (window.temp_project_list.item(index) for index in range(window.temp_project_list.count()))
                if item.data(PROJECT_LIST_KIND_ROLE) == "creative"
            )
            window.temp_project_list.setCurrentItem(creative_item)
            self.app.processEvents()

            self.assertEqual(window.current_creative_project_id, project_id)
            self.assertGreater(window.creative_node_tree.topLevelItemCount(), 0)
            self.assertEqual(window.center_result_stack.currentWidget(), window.project_board_view)
            window.close()

    def test_complete_all_creative_nodes_preserves_existing_user_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            project_id = store.create_creative_project(
                title="空间站晚餐",
                brief="空间站里的几个航天员在吃饭",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            user_note = "用户确认：白天作息，但窗外必须是黑色太空。"
            store.update_creative_node(int(root_id), note=user_note, search_query="空间站 白天 黑色太空")
            time_id = store.create_creative_node(
                project_id=project_id,
                parent_id=int(root_id),
                title="时间",
                note="季节、昼夜、事件发生的具体时刻。",
                search_query="时间",
            )

            class FakeProvider:
                def __init__(self) -> None:
                    self.calls: list[dict[str, str]] = []

                def generate_creative_node_note(self, **kwargs: str) -> tuple[SimpleNamespace, str]:
                    self.calls.append(dict(kwargs))
                    title = str(kwargs["node_title"])
                    current_note = str(kwargs["current_note"])
                    note = f"{title} AI补充说明，补足画面参考信息。"
                    return SimpleNamespace(note=note, search_query=f"{title} 搜索参考"), "fake"

            provider = FakeProvider()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=int(root_id), show_board=False)

            def run_immediately(target, **_kwargs):
                target()
                return True

            with patch.object(window, "_make_llm_provider", return_value=provider), patch.object(
                window,
                "_start_background_task",
                side_effect=run_immediately,
            ):
                window._generate_all_creative_node_notes()
                window._poll_events()

            self.assertGreaterEqual(len(provider.calls), 2)
            root_call = next(call for call in provider.calls if call["node_title"] == "空间站晚餐")
            time_call = next(call for call in provider.calls if call["node_title"] == "时间")
            self.assertIn(user_note, root_call["current_note"])
            self.assertEqual(time_call["current_note"], "")
            root = store.get_creative_node(int(root_id))
            time_node = store.get_creative_node(time_id)
            self.assertIsNotNone(root)
            self.assertIsNotNone(time_node)
            assert root is not None and time_node is not None
            self.assertIn(user_note, root.note)
            self.assertIn("空间站晚餐 AI补充说明", root.note)
            self.assertIn("时间 AI补充说明", time_node.note)
            self.assertFalse(window._creative_all_nodes_in_progress)
            window.close()

    def test_ai_generate_project_button_seeds_blank_project_and_completes_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()

            class FakeProvider:
                def __init__(self) -> None:
                    self.seed_calls = 0
                    self.note_calls = 0

                def generate_creative_project_seed(self, **_kwargs) -> tuple[SimpleNamespace, str]:
                    self.seed_calls += 1
                    return (
                        SimpleNamespace(
                            title="雨夜补给站",
                            brief="飞行器驾驶员在雨夜自动售货机旁买饮料",
                            extra="近未来，霓虹，潮湿街道，蓝紫色反光",
                        ),
                        "fake",
                    )

                def generate_creative_node_note(self, **kwargs: str) -> tuple[SimpleNamespace, str]:
                    self.note_calls += 1
                    title = str(kwargs["node_title"])
                    return (
                        SimpleNamespace(
                            note=f"{title} AI补全后的节点说明，提供可执行视觉参考。",
                            search_query=f"{title} 视觉参考",
                        ),
                        "fake",
                    )

            provider = FakeProvider()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            self.assertIsNone(window.current_creative_project_id)
            self.assertEqual(window.inspiration_brief_input.toPlainText(), "")
            self.assertEqual(window.inspiration_answers_input.toPlainText(), "")

            def run_immediately(target, **_kwargs):
                target()
                return True

            with patch.object(window, "_make_llm_provider", return_value=provider), patch.object(
                window,
                "_start_background_task",
                side_effect=run_immediately,
            ):
                window.creative_ai_project_button.click()
                window._poll_events()

            projects = store.list_creative_projects()
            self.assertEqual(len(projects), 1)
            self.assertEqual(projects[0].title, "雨夜补给站")
            self.assertIn("飞行器驾驶员", window.inspiration_brief_input.toPlainText())
            self.assertIn("蓝紫色反光", window.inspiration_answers_input.toPlainText())
            nodes = store.list_creative_nodes(projects[0].id)
            self.assertGreater(len(nodes), 3)
            self.assertGreaterEqual(provider.seed_calls, 1)
            self.assertEqual(provider.note_calls, len(nodes))
            self.assertTrue(any("AI补全后的节点说明" in node.note for node in nodes))
            self.assertFalse(window._creative_all_nodes_in_progress)
            window.close()

    def test_creative_node_search_marks_images_already_saved_in_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_ids: list[int] = []
            for index in range(3):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=123 + index,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)
            project_id = store.create_creative_project(
                title="故事项目",
                brief="飞行器驾驶员买饮料",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            world_id = store.create_creative_node(
                project_id=project_id,
                parent_id=int(root_id),
                title="世界观",
                note="自动售货机街区",
                search_query="自动售货机街区",
            )
            event_id = store.create_creative_node(
                project_id=project_id,
                parent_id=int(root_id),
                title="事件",
                note="买饮料动作",
                search_query="买饮料动作",
            )
            store.add_images_to_creative_node(world_id, [image_ids[0]], intent_label="世界观")

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=event_id, show_board=False)
            images = store.images_by_ids(image_ids[:2])
            window.semantic_search_revision = 12

            window._handle_creative_node_search_done(
                12,
                event_id,
                "买饮料动作",
                SemanticSearchResult(images=images, searchable_count=2, candidate_limit=2),
            )

            self.assertEqual(window.current_creative_node_badges[image_ids[0]][0], "已选：世界观")
            self.assertEqual(window.current_creative_node_badges[image_ids[1]], ["事件"])
            self.assertEqual(window.grid_view._badges_by_image_id[image_ids[0]][0], "已选：世界观")
            window.close()

    def test_selecting_temporary_project_from_project_sidebar_opens_board_with_badges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            library_dir = Path(tmp) / "library"
            library_dir.mkdir()
            image_path = library_dir / "first.jpg"
            pixmap = QPixmap(320, 180)
            pixmap.fill(QColor("#445566"))
            self.assertTrue(pixmap.save(str(image_path)))
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(library_dir))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=320,
                height=180,
                created_time_ns=None,
                modified_time_ns=image_path.stat().st_mtime_ns,
            )
            project_id = store.create_temporary_project("临时参考", [image_id])
            store.add_images_to_temporary_project(
                project_id,
                [image_id],
                intent_labels={image_id: "世界观"},
            )

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            self._expand_project_sidebar_section(window, "temporary")
            temporary_item = next(
                item
                for item in (window.temp_project_list.item(index) for index in range(window.temp_project_list.count()))
                if item.data(PROJECT_LIST_KIND_ROLE) == "temporary"
            )

            window.temp_project_list.setCurrentItem(temporary_item)
            self.app.processEvents()

            self.assertEqual(window.current_temp_project_id, project_id)
            self.assertEqual(window.center_result_stack.currentWidget(), window.project_board_view)
            self.assertIn("语义探针项目看板", window.result_state_label.text())
            item = window.project_board_view._image_items[image_id]
            self.assertEqual(getattr(item, "badge_text"), "世界观")

            window.project_board_view._select_image_id(image_id)
            self.app.processEvents()

            self.assertIsNotNone(window.selected_image)
            self.assertEqual(window.selected_image.id, image_id)
            self.assertEqual(window.file_name_input.text(), "first.jpg")
            self.assertIn(str(image_path), window.path_label.toPlainText())
            window.close()

    def test_temporary_project_board_removes_links_and_undo_restores_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_ids = []
            for index in range(2):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=123,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)
            project_id = store.create_temporary_project("临时参考", image_ids)
            store.add_images_to_temporary_project(
                project_id,
                [image_ids[0]],
                intent_labels={image_ids[0]: "世界观"},
            )

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._show_temporary_project_board(project_id)
            window.project_board_view._image_items[image_ids[0]].setPos(222, 333)

            window._remove_images_from_current_board([image_ids[0]])

            self.assertEqual(store.temporary_project_image_ids(project_id), [image_ids[1]])
            self.assertEqual({image.id for image in store.list_images()}, set(image_ids))

            window._undo_last_board_removal()

            self.assertEqual(store.temporary_project_image_ids(project_id), image_ids)
            self.assertEqual(store.temporary_project_image_badges(project_id), {image_ids[0]: ["世界观"]})
            restored = window.project_board_view._image_items[image_ids[0]]
            self.assertEqual(restored.pos().x(), 222)
            self.assertEqual(restored.pos().y(), 333)
            window.close()

    def test_temporary_project_board_autosaves_layout_when_switching_to_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            library_dir = Path(tmp) / "library"
            library_dir.mkdir()
            image_path = library_dir / "first.jpg"
            pixmap = QPixmap(320, 180)
            pixmap.fill(QColor("#445566"))
            self.assertTrue(pixmap.save(str(image_path)))
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(library_dir))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=320,
                height=180,
                created_time_ns=None,
                modified_time_ns=image_path.stat().st_mtime_ns,
            )
            project_id = store.create_temporary_project("临时参考", [image_id])
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._show_temporary_project_board(project_id)
            item = window.project_board_view._image_items[image_id]
            item.setPos(333, 444)
            getattr(item, "set_display_size")(260)
            getattr(item, "set_pinned")(True)
            getattr(item, "set_flipped")(True)
            getattr(item, "set_grayscale")(True)
            item.setVisible(False)

            window._show_gallery_view()
            window._show_temporary_project_board(project_id)

            restored = window.project_board_view._image_items[image_id]
            self.assertEqual(restored.pos().x(), 333)
            self.assertEqual(restored.pos().y(), 444)
            self.assertAlmostEqual(getattr(restored, "display_width"), 260)
            self.assertFalse(restored.isVisible())
            self.assertTrue(getattr(restored, "is_pinned")())
            self.assertTrue(getattr(restored, "is_flipped")())
            self.assertTrue(getattr(restored, "is_grayscale")())
            window.close()

    def test_temporary_project_board_autosave_timer_persists_dirty_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            library_dir = Path(tmp) / "library"
            library_dir.mkdir()
            image_path = library_dir / "first.jpg"
            pixmap = QPixmap(320, 180)
            pixmap.fill(QColor("#445566"))
            self.assertTrue(pixmap.save(str(image_path)))
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(library_dir))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=320,
                height=180,
                created_time_ns=None,
                modified_time_ns=image_path.stat().st_mtime_ns,
            )
            project_id = store.create_temporary_project("临时参考", [image_id])
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._show_temporary_project_board(project_id)
            window.project_board_view._image_items[image_id].setPos(123, 456)

            window._mark_current_board_layout_dirty()
            window._autosave_current_board_layout()

            payload_json = store.get_temporary_project_board_layout(project_id)
            self.assertIsNotNone(payload_json)
            payload = json.loads(str(payload_json))
            self.assertEqual(payload["items"][str(image_id)]["x"], 123.0)
            self.assertEqual(payload["items"][str(image_id)]["y"], 456.0)
            self.assertFalse(window._board_layout_dirty)
            window.close()

    def test_creative_project_board_autosaves_layout_when_switching_to_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            library_dir = Path(tmp) / "library"
            library_dir.mkdir()
            image_path = library_dir / "first.jpg"
            pixmap = QPixmap(320, 180)
            pixmap.fill(QColor("#445566"))
            self.assertTrue(pixmap.save(str(image_path)))
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(library_dir))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=320,
                height=180,
                created_time_ns=None,
                modified_time_ns=image_path.stat().st_mtime_ns,
            )
            project_id = store.create_creative_project(
                title="故事项目",
                brief="一个室内场景",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            store.add_images_to_creative_node(int(root_id), [image_id], intent_label="世界观")
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=int(root_id))
            window._show_current_creative_board()
            item = window.project_board_view._image_items[image_id]
            item.setPos(444, 555)
            getattr(item, "set_display_size")(280)
            getattr(item, "set_pinned")(True)
            getattr(item, "set_flipped")(True)

            window._show_gallery_view()
            window._show_current_creative_board()

            restored = window.project_board_view._image_items[image_id]
            self.assertEqual(restored.pos().x(), 444)
            self.assertEqual(restored.pos().y(), 555)
            self.assertAlmostEqual(getattr(restored, "display_width"), 280)
            self.assertTrue(getattr(restored, "is_pinned")())
            self.assertTrue(getattr(restored, "is_flipped")())
            window.close()

    def test_creative_project_board_removes_branch_links_and_undo_restores_original_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_ids = []
            for index in range(2):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=123,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)
            project_id = store.create_creative_project(
                title="故事项目",
                brief="一个室内场景",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            child_id = store.create_creative_node(
                project_id=project_id,
                parent_id=int(root_id),
                title="地点",
                note="旧公寓走廊",
                search_query="旧公寓走廊",
            )
            store.add_images_to_creative_node(child_id, image_ids, intent_label="地点")

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=int(root_id))
            window._show_current_creative_board()
            window.project_board_view._image_items[image_ids[0]].setPos(244, 355)

            window._remove_images_from_current_board([image_ids[0]])

            self.assertEqual(store.creative_node_image_ids(child_id), [image_ids[1]])
            self.assertEqual({image.id for image in store.list_images()}, set(image_ids))

            window._undo_last_board_removal()

            self.assertEqual(store.creative_node_image_ids(child_id), image_ids)
            self.assertEqual(store.creative_node_image_badges(project_id)[image_ids[0]], ["地点"])
            restored = window.project_board_view._image_items[image_ids[0]]
            self.assertEqual(restored.pos().x(), 244)
            self.assertEqual(restored.pos().y(), 355)
            window.close()

    def test_parent_creative_board_refreshes_when_child_gets_more_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_ids = []
            for index in range(2):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=123,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)
            project_id = store.create_creative_project(
                title="故事项目",
                brief="一个室内场景",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            child_id = store.create_creative_node(
                project_id=project_id,
                parent_id=int(root_id),
                title="事件",
                note="取饮料动作",
                search_query="取饮料动作",
            )
            store.add_images_to_creative_node(child_id, [image_ids[0]], intent_label="事件")

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=int(root_id), show_board=False)
            window._show_current_creative_board()
            self.assertEqual(set(window.project_board_view._image_items), {image_ids[0]})

            new_image = store.images_by_ids([image_ids[1]])[0]
            window.current_creative_node_id = child_id
            window.current_result_mode = "creative_node"
            window.current_creative_node_images = [new_image]
            window.current_creative_node_filtered_images = [new_image]
            window.grid_view.set_images([new_image], selected_image_ids=[image_ids[1]])

            window._save_selection_to_current_creative_node()

            self.assertEqual(store.creative_node_image_ids(child_id), image_ids)
            self.assertEqual(window._current_board_node_id, int(root_id))
            self.assertEqual(window._current_board_image_ids, tuple(image_ids))
            self.assertEqual(set(window.project_board_view._image_items), set(image_ids))
            window.close()

    def test_parent_creative_board_places_new_child_images_away_from_existing_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_ids = []
            for index in range(2):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=123,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)
            project_id = store.create_creative_project(
                title="故事项目",
                brief="一个室内场景",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            early_child_id = store.create_creative_node(
                project_id=project_id,
                parent_id=int(root_id),
                title="地点",
                note="街头",
                search_query="街头",
            )
            later_child_id = store.create_creative_node(
                project_id=project_id,
                parent_id=int(root_id),
                title="物件",
                note="道具",
                search_query="道具",
            )
            store.add_images_to_creative_node(later_child_id, [image_ids[1]], intent_label="物件")
            store.save_creative_node_board_layout(
                int(root_id),
                json.dumps(
                    {
                        "version": 1,
                        "items": {
                            str(image_ids[1]): {
                                "x": 80.0,
                                "y": 110.0,
                                "width": 160.0,
                                "height": 160.0,
                                "visible": True,
                            }
                        },
                    }
                ),
            )
            store.add_images_to_creative_node(early_child_id, [image_ids[0]], intent_label="地点")

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=int(root_id), show_board=False)
            window._show_current_creative_board()

            first_pos = window.project_board_view._image_items[image_ids[0]].pos()
            second_pos = window.project_board_view._image_items[image_ids[1]].pos()
            self.assertNotEqual((first_pos.x(), first_pos.y()), (second_pos.x(), second_pos.y()))
            self.assertGreater(first_pos.y(), second_pos.y())
            window.close()

    def test_parent_creative_board_repairs_duplicate_saved_layout_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_ids = []
            for index in range(2):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=123,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)
            project_id = store.create_creative_project(
                title="故事项目",
                brief="一个室内场景",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            child_id = store.create_creative_node(
                project_id=project_id,
                parent_id=int(root_id),
                title="地点",
                note="街头",
                search_query="街头",
            )
            store.add_images_to_creative_node(child_id, image_ids, intent_label="地点")
            duplicate_item = {
                "x": 80.0,
                "y": 110.0,
                "width": 160.0,
                "height": 160.0,
                "visible": True,
            }
            store.save_creative_node_board_layout(
                int(root_id),
                json.dumps(
                    {
                        "version": 1,
                        "items": {
                            str(image_ids[0]): duplicate_item,
                            str(image_ids[1]): dict(duplicate_item),
                        },
                    }
                ),
            )

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=int(root_id), show_board=False)
            window._show_current_creative_board()

            first_pos = window.project_board_view._image_items[image_ids[0]].pos()
            second_pos = window.project_board_view._image_items[image_ids[1]].pos()
            self.assertNotEqual((first_pos.x(), first_pos.y()), (second_pos.x(), second_pos.y()))
            self.assertGreater(second_pos.y(), first_pos.y())
            window.close()

    def test_board_layout_save_prunes_stale_items(self) -> None:
        current_payload = {
            "version": 1,
            "items": {
                "2": {"x": 240.0, "y": 120.0, "width": 160.0, "height": 90.0, "visible": True}
            },
        }
        existing_payload = {
            "version": 1,
            "items": {
                "1": {"x": 80.0, "y": 110.0, "width": 160.0, "height": 90.0, "visible": True},
                "2": {"x": 120.0, "y": 110.0, "width": 160.0, "height": 90.0, "visible": True},
            },
        }

        payload = MainWindow._merged_board_layout_payload(current_payload, existing_payload)

        self.assertEqual(payload["items"], current_payload["items"])

    def test_selecting_creative_node_opens_node_board_not_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_ids = []
            for index in range(2):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=123,
                    width=100,
                    height=100,
                    created_time_ns=None,
                    modified_time_ns=index + 1,
                )
                image_ids.append(image_id)
            project_id = store.create_creative_project(
                title="故事项目",
                brief="一个室内场景",
                language="zh",
                provider_name="LM Studio",
                model_name="fake",
            )
            root_id = store.creative_root_node_id(project_id)
            self.assertIsNotNone(root_id)
            child_id = store.create_creative_node(
                project_id=project_id,
                parent_id=int(root_id),
                title="地点",
                note="旧公寓走廊",
                search_query="旧公寓走廊",
            )
            store.add_images_to_creative_node(child_id, image_ids, intent_label="地点")

            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window._load_creative_project(project_id, select_node_id=int(root_id), show_board=True)
            self.app.processEvents()

            child_item = None
            root_item = window.creative_node_tree.topLevelItem(0)
            for index in range(root_item.childCount()):
                item = root_item.child(index)
                if item.data(0, Qt.ItemDataRole.UserRole) == child_id:
                    child_item = item
                    break
            self.assertIsNotNone(child_item)
            window.creative_node_tree.setCurrentItem(child_item)
            self.app.processEvents()

            self.assertEqual(window.current_creative_node_id, child_id)
            self.assertEqual(window.center_result_stack.currentWidget(), window.project_board_view)
            self.assertNotEqual(window.center_result_stack.currentWidget(), window.grid_view)
            self.assertIn("项目看板：地点", window.result_state_label.text())
            window.close()

    def test_preserve_current_view_scan_refresh_does_not_reset_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = AppPaths(
                data_dir=Path(tmp) / "data",
                thumbnail_dir=Path(tmp) / "data" / "thumbs",
                database_path=Path(tmp) / "data" / "eidory.sqlite3",
                log_dir=Path(tmp) / "data" / "logs",
            )
            paths.ensure()
            store = MetadataStore(paths.database_path)
            store.initialize()
            window = MainWindow(paths=paths, store=store)
            window.show()
            self.app.processEvents()
            window.current_result_mode = "search_chain"
            window.search_filters = [SearchFilter("semantic", "水")]
            window.grid_view.set_images([self._image(1), self._image(2)])

            with (
                patch.object(window, "_reload_images") as reload_images,
                patch.object(window, "_refresh_current_results_for_filters") as refresh_results,
            ):
                window._refresh_after_scan_database_change(preserve_current_view=True)

            reload_images.assert_not_called()
            refresh_results.assert_not_called()
            self.assertEqual([image.id for image in window.grid_view.images()], [1, 2])
            window.close()

    @staticmethod
    def _image(
        image_id: int,
        *,
        score: float | None = None,
        file_ext: str = ".jpg",
        file_path: str | None = None,
        width: int | None = 100,
        height: int | None = 100,
    ) -> ImageItem:
        path = file_path or f"/tmp/{image_id}{file_ext}"
        return ImageItem(
            id=image_id,
            folder_id=1,
            file_path=path,
            file_name=Path(path).name,
            file_ext=file_ext,
            file_size=100,
            width=width,
            height=height,
            created_at=None,
            modified_at=None,
            modified_time_ns=image_id,
            imported_at="2026-01-01T00:00:00+00:00",
            last_seen_at="2026-01-01T00:00:00+00:00",
            thumbnail_path=None,
            thumbnail_status="ready",
            embedding_status="ready",
            is_missing=False,
            is_favorite=False,
            note=None,
            score=score,
        )

    @staticmethod
    def _ai_vision_analysis(
        *,
        scene_location: str = "outdoor",
        environment_type: str = "built",
        time_of_day: str = "day",
        weather: str = "sunny",
        shot_scale: str = "long",
        view_angle: str = "eye_level",
        lighting: list[str] | None = None,
    ) -> AIVisionAnalysis:
        return AIVisionAnalysis(
            scene_location=scene_location,
            environment_type=environment_type,
            time_of_day=time_of_day,
            weather=weather,
            shot_scale=shot_scale,
            view_angle=view_angle,
            lighting=lighting or ["diffuse"],
            confidence={
                "scene_location": 0.9,
                "environment_type": 0.8,
                "time_of_day": 0.8,
                "weather": 0.7,
                "shot_scale": 0.8,
                "view_angle": 0.8,
                "lighting:diffuse": 0.7,
            },
            notes="",
            raw_json={},
        )

    @staticmethod
    def _collection_item(tree: QTreeWidget, collection_id: int) -> QTreeWidgetItem | None:
        def visit(item: QTreeWidgetItem) -> QTreeWidgetItem | None:
            if item.data(0, Qt.ItemDataRole.UserRole) == collection_id:
                return item
            for index in range(item.childCount()):
                found = visit(item.child(index))
                if found is not None:
                    return found
            return None

        for index in range(tree.topLevelItemCount()):
            found = visit(tree.topLevelItem(index))
            if found is not None:
                return found
        return None

    @staticmethod
    def _list_item_by_data(list_widget, value: str) -> QListWidgetItem:
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == value:
                return item
        raise AssertionError(f"list item not found: {value}")

    def _expand_project_sidebar_section(self, window: MainWindow, section_id: str) -> None:
        if window.project_sidebar_expanded_sections.get(section_id, False):
            return
        for row in range(window.temp_project_list.count()):
            item = window.temp_project_list.item(row)
            if item.data(PROJECT_LIST_SECTION_ID_ROLE) == section_id:
                window._handle_project_sidebar_item_clicked(item)
                self.app.processEvents()
                return
        raise AssertionError(f"project sidebar section not found: {section_id}")

    @staticmethod
    def _tag_item(window: MainWindow, tag_name: str) -> QListWidgetItem:
        for row in range(window.tag_list.count()):
            item = window.tag_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole + 1) == tag_name:
                return item
        raise AssertionError(f"tag item not found: {tag_name}")

    @staticmethod
    def _tag_id(store: MetadataStore, tag_name: str) -> int:
        for tag in store.list_tags():
            if tag.tag_name == tag_name:
                return tag.id
        raise AssertionError(f"tag not found: {tag_name}")

    @staticmethod
    def _menu_texts(menu) -> list[str]:
        return [action.text() for action in menu.actions() if not action.isSeparator()]

    @classmethod
    def _submenu_texts(cls, menu, title: str) -> list[str]:
        for action in menu.actions():
            if action.text() == title and action.menu() is not None:
                return cls._menu_texts(action.menu())
        raise AssertionError(f"submenu not found: {title}")

    @staticmethod
    def _action_by_text(menu, title: str):
        for action in menu.actions():
            if action.text() == title:
                return action
        raise AssertionError(f"action not found: {title}")

    @classmethod
    def _submenu_action_by_text(cls, menu, submenu_title: str, action_title: str):
        for action in menu.actions():
            if action.text() == submenu_title and action.menu() is not None:
                return cls._action_by_text(action.menu(), action_title)
        raise AssertionError(f"submenu not found: {submenu_title}")


if __name__ == "__main__":
    unittest.main()
