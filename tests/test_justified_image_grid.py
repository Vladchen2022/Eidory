from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from eidory.models import ImageItem
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

    @staticmethod
    def _image(image_id: int, *, width: int | None = 100, height: int | None = 100) -> ImageItem:
        return ImageItem(
            id=image_id,
            folder_id=1,
            file_path=f"/tmp/{image_id}.jpg",
            file_name=f"{image_id}.jpg",
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
            is_missing=False,
            is_favorite=False,
            note=None,
        )


if __name__ == "__main__":
    unittest.main()
