from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QMimeData, QPoint, QUrl
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication

from eidory.models import ImageItem
from eidory.ui.collection_tree import IMAGE_IDS_MIME, CollectionTreeWidget
from eidory.ui.justified_image_grid import JustifiedImageGridView


class JustifiedImageGridSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_multi_selection_tracks_selected_image_ids(self) -> None:
        grid = JustifiedImageGridView()
        grid.set_images([self._image(1), self._image(2), self._image(3)])

        grid._select_single(0)
        self.assertEqual(grid.selected_image_ids(), [1])
        self.assertEqual(grid.current_image().id, 1)

        grid._toggle_index(2)
        self.assertEqual(grid.selected_image_ids(), [1, 3])
        self.assertEqual(grid.current_image().id, 3)

        grid._select_range(1, additive=False)
        self.assertEqual(grid.selected_image_ids(), [2, 3])
        self.assertEqual(grid.current_image().id, 2)

        grid.select_image_id(1)
        self.assertEqual(grid.selected_image_ids(), [1])

    def test_set_images_preserves_selection_by_image_id(self) -> None:
        grid = JustifiedImageGridView()
        grid.set_images([self._image(1), self._image(2), self._image(3)])
        grid._select_single(1)

        grid.set_images([
            self._image(4),
            self._image(2, width=1920, height=1080),
            self._image(1),
        ])

        self.assertEqual(grid.selected_image_ids(), [2])
        self.assertEqual(grid.current_image().id, 2)
        self.assertEqual(grid.current_image().width, 1920)

    def test_set_images_drops_selection_when_image_is_absent(self) -> None:
        grid = JustifiedImageGridView()
        grid.set_images([self._image(1), self._image(2)])
        grid._select_single(1)

        grid.set_images([self._image(3)])

        self.assertEqual(grid.selected_image_ids(), [])
        self.assertIsNone(grid.current_image())

    def test_selected_file_urls_support_external_drag_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            existing_path = Path(tmp) / "existing.jpg"
            existing_path.write_bytes(b"image")
            missing_path = Path(tmp) / "missing.jpg"
            grid = JustifiedImageGridView()
            grid.set_images([
                self._image(1, file_path=str(existing_path)),
                self._image(2, file_path=str(missing_path)),
                self._image(3, file_path=str(Path(tmp) / "missing-flag.jpg"), is_missing=True),
            ])
            grid._select_single(0)
            grid._toggle_index(1)
            grid._toggle_index(2)

            urls = grid._selected_file_urls()

            self.assertEqual([url.toLocalFile() for url in urls], [str(existing_path)])

    def test_internal_image_drag_is_not_treated_as_external_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.jpg"
            path.write_bytes(b"image")
            mime = QMimeData()
            mime.setData(IMAGE_IDS_MIME, CollectionTreeWidget.encode_image_ids([1]))
            mime.setUrls([QUrl.fromLocalFile(str(path))])

            self.assertFalse(JustifiedImageGridView._supports_external_import_drop(mime))

    def test_hit_testing_uses_row_ranges_after_layout(self) -> None:
        grid = JustifiedImageGridView(thumbnail_size=90, spacing=4)
        grid.resize(360, 240)
        grid.show()
        self.app.processEvents()
        grid.set_images([
            self._image(image_id, width=160 + image_id % 5 * 20, height=90)
            for image_id in range(1, 220)
        ])

        self.assertGreater(len(grid._row_ranges), 1)
        for _row_top, _row_bottom, row_start, _row_end in grid._row_ranges[:5]:
            rect = grid._rects[row_start]
            point = QPoint(rect.center().x(), rect.center().y() - grid.verticalScrollBar().value())
            self.assertEqual(grid._index_at(point), row_start)

        grid.verticalScrollBar().setValue(grid.verticalScrollBar().maximum())
        self.app.processEvents()
        _row_top, _row_bottom, row_start, _row_end = grid._row_ranges[-1]
        last_rect = grid._rects[row_start]
        visible_point = QPoint(
            last_rect.center().x(),
            last_rect.center().y() - grid.verticalScrollBar().value(),
        )
        self.assertEqual(grid._index_at(visible_point), row_start)

    def test_missing_thumbnail_fallback_is_decoded_at_bounded_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "large.jpg"
            pixmap = QPixmap(1200, 420)
            pixmap.fill(QColor("#667788"))
            self.assertTrue(pixmap.save(str(image_path)))

            grid = JustifiedImageGridView(thumbnail_size=90, spacing=4)
            loaded = grid._pixmap_for(self._image(1, file_path=str(image_path), width=1200, height=420))

            self.assertFalse(loaded.isNull())
            self.assertLessEqual(max(loaded.width(), loaded.height()), 270)

    @staticmethod
    def _image(
        image_id: int,
        *,
        width: int | None = 100,
        height: int | None = 100,
        file_path: str | None = None,
        is_missing: bool = False,
    ) -> ImageItem:
        path = file_path or f"/tmp/{image_id}.jpg"
        return ImageItem(
            id=image_id,
            folder_id=1,
            file_path=path,
            file_name=Path(path).name,
            file_ext=".jpg",
            file_size=100,
            width=width,
            height=height,
            created_at=None,
            modified_at=None,
            modified_time_ns=image_id,
            imported_at="2026-01-01T00:00:00+00:00",
            last_seen_at="2026-01-01T00:00:00+00:00",
            thumbnail_path=None,
            thumbnail_status="pending",
            embedding_status="pending",
            is_missing=is_missing,
            is_favorite=False,
            note=None,
        )


if __name__ == "__main__":
    unittest.main()
