from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent, QPixmap
from PySide6.QtWidgets import QApplication

from eidory.core.metadata_store import MetadataStore
from eidory.ui.image_preview_dialog import ImagePreviewDialog


class ImagePreviewDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_preview_pixmap_loads_when_pillow_pixel_limit_is_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.jpg"
            Image.new("RGB", (64, 48), color="red").save(image_path)

            original_limit = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = 1
            try:
                pixmap = ImagePreviewDialog._load_preview_pixmap(str(image_path), 240, 180)
            finally:
                Image.MAX_IMAGE_PIXELS = original_limit

            self.assertFalse(pixmap.isNull())

    def test_preview_writes_favorite_and_search_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            Image.new("RGB", (64, 48), color="blue").save(image_path)
            store = MetadataStore(root / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=64,
                height=48,
                created_time_ns=None,
                modified_time_ns=image_path.stat().st_mtime_ns,
            )
            image = replace(store.get_image(image_id), score=0.42)

            dialog = ImagePreviewDialog(
                images=[image],
                start_index=0,
                store=store,
                semantic_query="蓝色图片",
                model_name="fake-model",
                model_revision="test",
                embedding_dim=2,
            )
            dialog._save_favorite(True)
            dialog._save_feedback("relevant")
            dialog.close()

            self.assertTrue(store.get_image(image_id).is_favorite)
            self.assertEqual(
                store.get_search_feedback(
                    query="蓝色图片",
                    image_id=image_id,
                    model_name="fake-model",
                    model_revision="test",
                    embedding_dim=2,
                ),
                "relevant",
            )

    def test_preview_zoom_and_fit_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            Image.new("RGB", (800, 500), color="green").save(image_path)
            store = MetadataStore(root / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=800,
                height=500,
                created_time_ns=None,
                modified_time_ns=image_path.stat().st_mtime_ns,
            )

            dialog = ImagePreviewDialog(
                images=[store.get_image(image_id)],
                start_index=0,
                store=store,
                semantic_query=None,
                model_name="fake-model",
                model_revision="test",
                embedding_dim=2,
            )
            self.assertTrue(dialog.fit_to_window)
            dialog._zoom_by(120)
            self.assertFalse(dialog.fit_to_window)
            self.assertGreater(dialog.zoom_factor, 1.0)
            dialog._fit_image_to_window()
            self.assertTrue(dialog.fit_to_window)
            self.assertEqual(dialog.zoom_factor, 1.0)
            self.assertGreaterEqual(len(dialog._shortcuts), 8)
            dialog._panning = True
            dialog._fit_image_to_window()
            self.assertFalse(dialog._panning)
            dialog._zoom_by(120)
            self.assertFalse(dialog.fit_to_window)
            dialog._handle_space_pressed()
            self.assertTrue(dialog.fit_to_window)
            self.assertEqual(dialog.zoom_factor, 1.0)
            dialog.close()

    def test_fit_mode_scales_small_pixmap_up_to_preview_bounds(self) -> None:
        pixmap = QPixmap(64, 48)
        scaled = ImagePreviewDialog._scale_pixmap_to_bounds(pixmap, 640, 480)

        self.assertEqual(scaled.width(), 640)
        self.assertEqual(scaled.height(), 480)

    def test_preview_double_click_closes_image_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            Image.new("RGB", (64, 48), color="red").save(image_path)
            store = MetadataStore(root / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=64,
                height=48,
                created_time_ns=None,
                modified_time_ns=image_path.stat().st_mtime_ns,
            )
            dialog = ImagePreviewDialog(
                images=[store.get_image(image_id)],
                start_index=0,
                store=store,
                semantic_query=None,
                model_name="fake-model",
                model_revision="test",
                embedding_dim=2,
            )
            dialog.show()
            self.app.processEvents()

            event = QMouseEvent(
                QEvent.Type.MouseButtonDblClick,
                QPointF(10, 10),
                QPointF(10, 10),
                QPointF(10, 10),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )

            self.assertTrue(dialog.eventFilter(dialog.image_label, event))
            self.assertFalse(dialog.isVisible())

    def test_preview_double_click_closes_video_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "clip.mp4"
            video_path.write_bytes(b"fake mp4 bytes")
            store = MetadataStore(root / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(video_path),
                file_size=video_path.stat().st_size,
                width=None,
                height=None,
                created_time_ns=None,
                modified_time_ns=video_path.stat().st_mtime_ns,
            )
            store.mark_embedding_not_required(image_id)
            dialog = ImagePreviewDialog(
                images=[store.get_image(image_id)],
                start_index=0,
                store=store,
                semantic_query=None,
                model_name="fake-model",
                model_revision="test",
                embedding_dim=2,
            )
            dialog.show()
            self.app.processEvents()

            event = QMouseEvent(
                QEvent.Type.MouseButtonDblClick,
                QPointF(10, 10),
                QPointF(10, 10),
                QPointF(10, 10),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )

            self.assertTrue(dialog.eventFilter(dialog.video_widget, event))
            self.assertFalse(dialog.isVisible())

    def test_preview_close_shortcut_accepts_cmd_w(self) -> None:
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_W,
            Qt.KeyboardModifier.MetaModifier,
        )
        self.assertTrue(ImagePreviewDialog._is_close_shortcut(event))

    def test_preview_event_filter_closes_from_child_cmd_w(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            Image.new("RGB", (64, 48), color="red").save(image_path)
            store = MetadataStore(root / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=64,
                height=48,
                created_time_ns=None,
                modified_time_ns=image_path.stat().st_mtime_ns,
            )
            dialog = ImagePreviewDialog(
                images=[store.get_image(image_id)],
                start_index=0,
                store=store,
                semantic_query=None,
                model_name="fake-model",
                model_revision="test",
                embedding_dim=2,
            )
            dialog.show()
            self.app.processEvents()

            event = QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_W,
                Qt.KeyboardModifier.MetaModifier,
            )
            self.assertTrue(dialog.eventFilter(dialog.video_widget, event))
            self.assertTrue(event.isAccepted())
            self.assertFalse(dialog.isVisible())

    def test_preview_uses_video_page_for_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "clip.mp4"
            video_path.write_bytes(b"fake mp4 bytes")
            store = MetadataStore(root / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(video_path),
                file_size=video_path.stat().st_size,
                width=None,
                height=None,
                created_time_ns=None,
                modified_time_ns=video_path.stat().st_mtime_ns,
            )
            store.mark_embedding_not_required(image_id)

            dialog = ImagePreviewDialog(
                images=[store.get_image(image_id)],
                start_index=0,
                store=store,
                semantic_query=None,
                model_name="fake-model",
                model_revision="test",
                embedding_dim=2,
            )

            self.assertIs(dialog.preview_stack.currentWidget(), dialog.video_widget)
            self.assertFalse(dialog.fit_button.isEnabled())
            self.assertFalse(dialog.video_controls_widget.isHidden())
            self.assertEqual(dialog.video_player.source().toLocalFile(), str(video_path))
            dialog.close()


if __name__ == "__main__":
    unittest.main()
