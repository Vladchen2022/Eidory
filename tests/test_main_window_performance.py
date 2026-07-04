from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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

    @staticmethod
    def _image(
        image_id: int,
        *,
        file_path: str,
        is_missing: bool = False,
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
        )


if __name__ == "__main__":
    unittest.main()
