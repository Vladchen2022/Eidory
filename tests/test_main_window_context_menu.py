from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidgetItem, QMessageBox, QTreeWidget, QTreeWidgetItem

from eidory.config import AppPaths
from eidory.core.inspiration import InspirationMatch, InspirationTerm
from eidory.core.llm_provider import GroupNameSuggestion, ProjectSuggestion
from eidory.core.metadata_store import MetadataStore, TEMPORARY_PROJECT_COLORS
from eidory.core.reference_grouping import ReferenceGroup
from eidory.core.search_filters import (
    SearchFilter,
    last_score_filter_kind,
    search_filter_from_payload,
    search_filter_to_payload,
)
from eidory.models import ImageItem
from eidory.ui.main_window import MainWindow


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

            self.assertEqual(window.status_list.currentItem().data(Qt.ItemDataRole.UserRole), "favorite")
            self.assertEqual(window.tag_sort_combo.currentData(), "count_desc")
            self.assertEqual(window.tag_match_combo.currentData(), "any")
            self.assertEqual(window.right_tab_widget.currentIndex(), 1)
            self.assertEqual(set(window._selected_tag_ids()), {indoor_id, night_id})

            window.status_list.setCurrentItem(self._list_item_by_data(window.status_list, "missing"))
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
            self.assertEqual(window.tag_list.item(1).data(Qt.ItemDataRole.UserRole + 1), "室内")
            self.assertEqual(window.tag_list.item(1).data(Qt.ItemDataRole.UserRole + 2), 2)

            window.tag_search_input.setText("机")
            self.app.processEvents()
            self.assertEqual(window.tag_list.count(), 2)
            tag_item = window.tag_list.item(1)
            self.assertEqual(tag_item.data(Qt.ItemDataRole.UserRole + 1), "机械")
            window.tag_list.setCurrentItem(tag_item)
            self.app.processEvents()

            self.assertTrue(window.rename_tag_button.isEnabled())
            self.assertTrue(window.delete_tag_button.isEnabled())
            self.assertTrue(window.merge_tag_button.isEnabled())
            self.assertEqual(window._selected_tag_context()[1], "机械")
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
            self.assertEqual(window.temp_project_list.count(), 0)
            self.assertEqual(window.current_result_mode, "library")
            self.assertIsNotNone(store.get_image(image_id))
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
            self.assertEqual(window.temp_project_list.item(0).background().color().name().upper(), TEMPORARY_PROJECT_COLORS[0])
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
            window._set_list_current_by_data(window.status_list, "favorite")
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
            window._set_list_current_by_data(window.status_list, "all")
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
            self.assertIs(window.preview_stack.parentWidget(), detail_tab)
            self.assertEqual(window.right_tab_widget.tabText(0), "详情")
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


if __name__ == "__main__":
    unittest.main()
