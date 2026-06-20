from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QKeyEvent, QPixmap
from PySide6.QtWidgets import QApplication

from eidory.models import ImageItem
from eidory.ui.project_board import ProjectBoardView, _cached_pixmap


class ProjectBoardViewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_space_toggle_restores_fit_all_not_previous_manual_view(self) -> None:
        board = ProjectBoardView()
        board.resize(900, 700)
        board.set_images([self._image(1), self._image(2)])

        board.reset_view()
        self.assertEqual(board._view_mode, "manual")

        board._select_image_id(1)
        board.toggle_fit_selection_or_restore()
        self.assertEqual(board._view_mode, "fit_selection")
        self.assertEqual(board._last_fit_selection_ids, (1,))

        board.toggle_fit_selection_or_restore()
        self.assertEqual(board._view_mode, "fit_all")
        self.assertEqual(board._last_fit_selection_ids, ())

    def test_selected_wide_image_can_fit_beyond_default_zoom_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "wide.jpg"
            pixmap = QPixmap(400, 60)
            pixmap.fill(QColor("#667788"))
            self.assertTrue(pixmap.save(str(image_path)))

            board = ProjectBoardView()
            board.resize(2400, 900)
            board.show()
            self.app.processEvents()
            board.set_images([self._image(1, file_path=str(image_path), width=400, height=60)])

            board._select_image_id(1)
            board.toggle_fit_selection_or_restore()

            item = board._image_items[1]
            viewport_width = board.viewport().width()
            displayed_width = item.sceneBoundingRect().width() * board._zoom
            self.assertEqual(board._view_mode, "fit_selection")
            self.assertGreater(board._zoom, 4.0)
            self.assertGreaterEqual(displayed_width, viewport_width - 30)

    def test_default_board_layout_uses_item_edges_to_avoid_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = []
            for image_id in range(1, 6):
                image_path = Path(tmp) / f"wide-{image_id}.jpg"
                pixmap = QPixmap(1200, 300)
                pixmap.fill(QColor("#667788"))
                self.assertTrue(pixmap.save(str(image_path)))
                images.append(self._image(image_id, file_path=str(image_path), width=1200, height=300))

            board = ProjectBoardView()
            board.resize(1400, 900)
            board.show()
            self.app.processEvents()
            board.set_images(images)

            rects = [
                board._image_items[image.id].sceneBoundingRect()
                for image in images
            ]
            for index, rect in enumerate(rects):
                for other in rects[index + 1:]:
                    self.assertFalse(rect.intersects(other), f"{rect} overlaps {other}")

    def test_board_image_items_keep_intent_badges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "first.jpg"
            pixmap = QPixmap(320, 180)
            pixmap.fill(QColor("#334455"))
            self.assertTrue(pixmap.save(str(image_path)))

            board = ProjectBoardView()
            board.set_images(
                [self._image(1, file_path=str(image_path), width=320, height=180)],
                badges_by_image_id={1: ["世界观", "地点"]},
            )

            item = board._image_items[1]
            self.assertEqual(getattr(item, "badge_text"), "世界观 +1")

    def test_layout_payload_restores_item_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "first.jpg"
            pixmap = QPixmap(320, 180)
            pixmap.fill(QColor("#334455"))
            self.assertTrue(pixmap.save(str(image_path)))

            image = self._image(1, file_path=str(image_path), width=320, height=180)
            board = ProjectBoardView()
            board.set_images([image])
            item = board._image_items[1]
            item.setPos(210, 320)
            getattr(item, "set_display_size")(260)
            getattr(item, "set_pinned")(True)
            getattr(item, "set_flipped")(True)
            getattr(item, "set_grayscale")(True)
            item.setVisible(False)

            payload = board.layout_payload()

            restored = ProjectBoardView()
            restored.set_images([image], layout_payload=payload)
            restored_item = restored._image_items[1]
            self.assertEqual(restored_item.pos().x(), 210)
            self.assertEqual(restored_item.pos().y(), 320)
            self.assertAlmostEqual(getattr(restored_item, "display_width"), 260)
            self.assertFalse(restored_item.isVisible())
            self.assertTrue(getattr(restored_item, "is_pinned")())
            self.assertTrue(getattr(restored_item, "is_flipped")())
            self.assertTrue(getattr(restored_item, "is_grayscale")())

    def test_delete_key_requests_selected_image_removal(self) -> None:
        board = ProjectBoardView()
        board.set_images([self._image(1), self._image(2)])
        emitted: list[list[int]] = []
        board.removeImagesRequested.connect(lambda image_ids: emitted.append(list(image_ids)))

        board._select_image_id(1)
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Delete,
            Qt.KeyboardModifier.NoModifier,
        )
        board.keyPressEvent(event)

        self.assertEqual(emitted, [[1]])
        self.assertTrue(event.isAccepted())

    def test_undo_shortcut_requests_board_removal_undo(self) -> None:
        board = ProjectBoardView()
        emitted: list[bool] = []
        board.undoRemovalRequested.connect(lambda: emitted.append(True))

        event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.MetaModifier,
        )
        board.keyPressEvent(event)

        self.assertEqual(emitted, [True])
        self.assertTrue(event.isAccepted())

    def test_cached_pixmap_respects_max_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "large.jpg"
            pixmap = QPixmap(600, 240)
            pixmap.fill(QColor("#445566"))
            self.assertTrue(pixmap.save(str(image_path)))

            cached = _cached_pixmap(image_path, max_side=160)

            self.assertFalse(cached.isNull())
            self.assertLessEqual(max(cached.width(), cached.height()), 160)

    @staticmethod
    def _image(
        image_id: int,
        *,
        file_path: str | None = None,
        width: int = 800,
        height: int = 450,
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
            thumbnail_status="ready",
            embedding_status="ready",
            is_missing=False,
            is_favorite=False,
            note=None,
        )
