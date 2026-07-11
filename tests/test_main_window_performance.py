from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication

from eidory.config import AppPaths
from eidory.core.metadata_store import MetadataStore
from eidory.models import ImageItem
from eidory.ui.main_window import MainWindow


class MainWindowPerformanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_show_image_details_defers_uncached_metadata_queries(self) -> None:
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
            image = self._image(1, file_path=str(Path(tmp) / "missing.jpg"), is_missing=True)

            with (
                patch.object(store, "collection_paths_for_image", side_effect=AssertionError("sync collection query")),
                patch.object(store, "get_image_tags", side_effect=AssertionError("sync tag query")),
                patch.object(store, "ai_vision_tags_for_image", side_effect=AssertionError("sync ai query")),
                patch.object(window, "_request_selection_metadata") as request_metadata,
            ):
                window._show_image_details(image)

            request_metadata.assert_called_once_with(image.id)
            self.assertEqual(window.image_collections_label.text(), "载入中...")
            self.assertEqual(window.ai_vision_detail_label.text(), "载入中...")
            self.assertEqual(window.tags_display.item(0).text(), "标签载入中...")
            window.close()

    def test_semantic_feedback_state_is_loaded_off_click_path(self) -> None:
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
            image = self._image(2, file_path=str(Path(tmp) / "second.jpg"), is_missing=True)
            window.current_result_mode = "semantic"
            window.current_semantic_query = "fog village"

            with (
                patch.object(store, "get_search_feedback", side_effect=AssertionError("sync feedback query")),
                patch.object(window, "_request_search_feedback") as request_feedback,
            ):
                window._refresh_feedback_buttons(image)

            request_feedback.assert_called_once()
            window.close()

    def test_detail_preview_cache_is_bounded_by_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(Path(tmp))
            window._detail_preview_cache_byte_limit = 40_000
            pixmap = QPixmap(100, 100)
            pixmap.fill(QColor("#446688"))

            first_key = ("/tmp/first.jpg", 1, 1, 100, 100)
            second_key = ("/tmp/second.jpg", 2, 2, 100, 100)
            window._cache_detail_preview_pixmap(first_key, pixmap)
            window._cache_detail_preview_pixmap(second_key, pixmap)

            self.assertNotIn(first_key, window._detail_preview_cache)
            self.assertIn(second_key, window._detail_preview_cache)
            self.assertLessEqual(window._detail_preview_cache_total_bytes, 40_000)
            window.close()

    def test_selection_detail_caches_evict_least_recently_used_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(Path(tmp))
            window._selection_detail_cache_limit = 2
            for image_id in (1, 2):
                window._image_collection_paths_cache[image_id] = [str(image_id)]
                window._image_tags_cache[image_id] = [str(image_id)]
                window._ai_vision_detail_cache[image_id] = str(image_id)
            window._touch_selection_detail_cache(1)
            window._image_collection_paths_cache[3] = ["3"]
            window._image_tags_cache[3] = ["3"]
            window._ai_vision_detail_cache[3] = "3"

            window._trim_selection_detail_caches()

            self.assertEqual(list(window._image_collection_paths_cache), [1, 3])
            self.assertEqual(list(window._image_tags_cache), [1, 3])
            self.assertEqual(list(window._ai_vision_detail_cache), [1, 3])
            window.close()

    def test_score_threshold_drag_throttles_updates_and_flushes_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(Path(tmp))
            window.score_threshold_slider.setSliderDown(True)

            with patch.object(window, "_apply_score_threshold_value") as apply_threshold:
                window._queue_score_threshold_update(25)
                window._queue_score_threshold_update(40)
                window._queue_score_threshold_update(65)

                apply_threshold.assert_not_called()
                self.assertTrue(window.score_threshold_update_timer.isActive())
                self.assertEqual(window._pending_score_threshold_value, 65)

                window._apply_pending_score_threshold_update()
                apply_threshold.assert_called_once_with(65, persist=False)

                window._queue_score_threshold_update(72)
                window.score_threshold_slider.setSliderDown(False)
                self.assertEqual(apply_threshold.call_count, 2)
                apply_threshold.assert_called_with(72, persist=True)

            window.close()

    def test_score_threshold_drag_updates_visible_images_before_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(Path(tmp))
            images = [
                self._image(1, file_path="/tmp/low.jpg", score=0.2),
                self._image(2, file_path="/tmp/mid.jpg", score=0.6),
                self._image(3, file_path="/tmp/high.jpg", score=0.9),
            ]
            window.current_result_mode = "semantic"
            window.current_semantic_images = images
            previous = window.score_threshold_slider.blockSignals(True)
            window.score_threshold_slider.setValue(0)
            window.score_threshold_slider.blockSignals(previous)
            window.grid_view.set_images(images)
            window.score_threshold_slider.setSliderDown(True)

            with patch.object(window.store, "set_setting") as set_setting:
                window.score_threshold_slider.setValue(50)
                window._apply_pending_score_threshold_update()

                self.assertEqual(
                    {image.id for image in window.grid_view.images()},
                    {2, 3},
                )
                set_setting.assert_not_called()

                window.score_threshold_slider.setValue(100)
                window._apply_pending_score_threshold_update()
                self.assertEqual([image.id for image in window.grid_view.images()], [3])

                window.score_threshold_slider.setSliderDown(False)
                set_setting.assert_called_with("ui.score_threshold", "100")

            window.close()

    def test_score_threshold_recompute_keeps_only_latest_busy_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(Path(tmp))
            window.search_button.setEnabled(False)
            initial_revision = window.semantic_search_revision

            with patch.object(window, "_execute_search_chain") as execute_search_chain:
                window._queue_score_threshold_recompute()
                window._queue_score_threshold_recompute()

                execute_search_chain.assert_not_called()
                self.assertTrue(window._pending_score_threshold_recompute)
                self.assertEqual(window.semantic_search_revision, initial_revision + 2)

                window.search_button.setEnabled(True)
                window._run_pending_score_threshold_recompute()
                self.app.processEvents()

                execute_search_chain.assert_called_once_with(operation_mode="recompute")
                self.assertFalse(window._pending_score_threshold_recompute)

            window.close()

    @staticmethod
    def _window(root: Path) -> MainWindow:
        paths = AppPaths(
            data_dir=root / "data",
            thumbnail_dir=root / "data" / "thumbs",
            database_path=root / "data" / "eidory.sqlite3",
            log_dir=root / "data" / "logs",
        )
        paths.ensure()
        store = MetadataStore(paths.database_path)
        store.initialize()
        return MainWindow(paths=paths, store=store)

    @staticmethod
    def _image(
        image_id: int,
        *,
        file_path: str,
        is_missing: bool = False,
        score: float | None = None,
    ) -> ImageItem:
        path = Path(file_path)
        return ImageItem(
            id=image_id,
            folder_id=1,
            file_path=str(path),
            file_name=path.name,
            file_ext=path.suffix or ".jpg",
            file_size=100,
            width=100,
            height=100,
            created_at=None,
            modified_at=None,
            modified_time_ns=image_id,
            imported_at="2026-01-01T00:00:00+00:00",
            last_seen_at="2026-01-01T00:00:00+00:00",
            thumbnail_path=None,
            thumbnail_status="missing",
            embedding_status="ready",
            is_missing=is_missing,
            is_favorite=False,
            note="",
            score=score,
        )


if __name__ == "__main__":
    unittest.main()
