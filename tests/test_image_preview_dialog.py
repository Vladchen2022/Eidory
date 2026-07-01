from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

from eidory.core.linetop_processor import LineTopSettings
from eidory.core.metadata_store import MetadataStore
from eidory.ui.image_preview_dialog import (
    INLINE_SOURCE_PREVIEW_MAX_BYTES,
    ImagePreviewDialog,
    SOURCE_PREVIEW_REFINE_DELAY_MS,
)


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

    def test_preview_bottom_bar_hides_file_action_buttons(self) -> None:
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

            dialog = ImagePreviewDialog(
                images=[store.get_image(image_id)],
                start_index=0,
                store=store,
                semantic_query=None,
                model_name="fake-model",
                model_revision="test",
                embedding_dim=2,
            )

            button_texts = {button.text() for button in dialog.findChildren(QPushButton)}
            self.assertFalse(
                {
                    "打开源文件",
                    "Finder 中显示",
                    "复制图片",
                    "复制路径",
                    "移除索引",
                }
                & button_texts
            )
            dialog.close()

    def test_source_preview_refine_timer_uses_fast_open_close_delay(self) -> None:
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

            dialog = ImagePreviewDialog(
                images=[store.get_image(image_id)],
                start_index=0,
                store=store,
                semantic_query=None,
                model_name="fake-model",
                model_revision="test",
                embedding_dim=2,
            )

            self.assertEqual(dialog._preview_refine_timer.interval(), SOURCE_PREVIEW_REFINE_DELAY_MS)
            self.assertGreaterEqual(dialog._preview_refine_timer.interval(), 300)
            dialog.close()

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
            fit_zoom = dialog._fit_zoom_factor()
            dialog._zoom_by(120)
            self.assertFalse(dialog.fit_to_window)
            self.assertGreater(dialog.zoom_factor, fit_zoom)
            dialog._fit_image_to_window()
            self.assertTrue(dialog.fit_to_window)
            self.assertEqual(dialog.zoom_factor, dialog._fit_zoom_factor())
            dialog._actual_size_image()
            self.assertFalse(dialog.fit_to_window)
            self.assertEqual(dialog.zoom_factor, 1.0)
            self.assertGreaterEqual(len(dialog._shortcuts), 8)
            dialog._panning = True
            dialog._fit_image_to_window()
            self.assertFalse(dialog._panning)
            dialog._zoom_by(120)
            self.assertFalse(dialog.fit_to_window)
            dialog._handle_space_pressed()
            self.assertFalse(dialog.isVisible())
            dialog.close()

    def test_preview_first_wheel_zoom_starts_from_fit_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "large.jpg"
            Image.new("RGB", (2400, 1600), color="green").save(image_path)
            store = MetadataStore(root / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=2400,
                height=1600,
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
            fit_zoom = dialog._fit_zoom_factor()
            dialog._zoom_by(120)

            self.assertFalse(dialog.fit_to_window)
            self.assertGreater(dialog.zoom_factor, fit_zoom)
            self.assertLess(dialog.zoom_factor, 1.0)
            dialog.close()

    def test_preview_wheel_zoom_reuses_loaded_pixmap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            Image.new("RGB", (1600, 1000), color="green").save(image_path)
            store = MetadataStore(root / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=1600,
                height=1000,
                created_time_ns=None,
                modified_time_ns=image_path.stat().st_mtime_ns,
            )

            with patch.object(
                ImagePreviewDialog,
                "_load_preview_pixmap",
                wraps=ImagePreviewDialog._load_preview_pixmap,
            ) as load_preview:
                dialog = ImagePreviewDialog(
                    images=[store.get_image(image_id)],
                    start_index=0,
                    store=store,
                    semantic_query=None,
                    model_name="fake-model",
                    model_revision="test",
                    embedding_dim=2,
                )
                initial_loads = load_preview.call_count
                dialog._zoom_by(120)

                self.assertEqual(load_preview.call_count, initial_loads)
                dialog._render_current_image()
                self.assertFalse(dialog.fit_to_window)
                dialog.close()

    def test_large_preview_source_refine_runs_in_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "large.jpg"
            Image.effect_noise((1600, 1200), 96).convert("RGB").save(image_path, quality=96)
            self.assertGreater(image_path.stat().st_size, INLINE_SOURCE_PREVIEW_MAX_BYTES)
            store = MetadataStore(root / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=1600,
                height=1200,
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
            with (
                patch.object(dialog, "_load_preview_pixmap", side_effect=AssertionError("main-thread source load")),
                patch.object(dialog._preview_source_thread_pool, "start") as start_task,
            ):
                dialog._render_current_image()

            start_task.assert_called_once()
            self.assertIsNotNone(dialog._preview_source_pending_token)
            dialog.close()

    def test_large_preview_source_load_keeps_only_latest_request_while_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_path = root / "first.jpg"
            second_path = root / "second.jpg"
            Image.new("RGB", (64, 48), color="red").save(first_path)
            Image.new("RGB", (64, 48), color="blue").save(second_path)
            store = MetadataStore(root / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            first_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(first_path),
                file_size=first_path.stat().st_size,
                width=64,
                height=48,
                created_time_ns=None,
                modified_time_ns=first_path.stat().st_mtime_ns,
            )
            second_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(second_path),
                file_size=second_path.stat().st_size,
                width=64,
                height=48,
                created_time_ns=None,
                modified_time_ns=second_path.stat().st_mtime_ns,
            )
            first = store.get_image(first_id)
            second = store.get_image(second_id)
            dialog = ImagePreviewDialog(
                images=[first, second],
                start_index=0,
                store=store,
                semantic_query=None,
                model_name="fake-model",
                model_revision="test",
                embedding_dim=2,
            )

            with patch.object(dialog._preview_source_thread_pool, "start") as start_task:
                dialog._request_preview_source_load(first, max_width=400, max_height=300)
                first_token = dialog._preview_source_running_token
                dialog.index = 1
                dialog._request_preview_source_load(second, max_width=400, max_height=300)

                self.assertEqual(start_task.call_count, 1)
                self.assertIsNotNone(dialog._preview_source_queued_request)

                dialog._handle_preview_source_loaded(first_token, QImage(), False)

            self.assertEqual(start_task.call_count, 2)
            self.assertEqual(dialog._preview_source_running_token[0], second.id)
            dialog.close()

    def test_linetop_render_keeps_only_latest_request_while_running(self) -> None:
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
            image = store.get_image(image_id)
            dialog = ImagePreviewDialog(
                images=[image],
                start_index=0,
                store=store,
                semantic_query=None,
                model_name="fake-model",
                model_revision="test",
                embedding_dim=2,
            )

            with (
                patch.object(dialog, "_linetop_preview_active", return_value=True),
                patch.object(dialog._linetop_thread_pool, "start") as start_task,
            ):
                dialog._request_linetop_render(image, 400, 300, use_thumbnail_first=False)
                first_token = dialog._linetop_render_running_token
                dialog._linetop_settings = LineTopSettings(mode="color_limit")
                dialog._request_linetop_render(image, 400, 300, use_thumbnail_first=False)

                self.assertEqual(start_task.call_count, 1)
                self.assertIsNotNone(dialog._linetop_queued_render_request)

                dialog._handle_linetop_render_loaded(first_token, QImage(), "")

            self.assertEqual(start_task.call_count, 2)
            self.assertEqual(dialog._linetop_render_running_token, dialog._linetop_render_pending_token)
            self.assertIn("color_limit", dialog._linetop_render_running_token)
            dialog.close()

    def test_preview_navigator_only_shows_when_image_can_pan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "large.jpg"
            Image.new("RGB", (2400, 1600), color="green").save(image_path)
            store = MetadataStore(root / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=2400,
                height=1600,
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

            self.assertTrue(dialog.fit_to_window)
            self.assertFalse(dialog.image_view._navigator.isVisible())

            dialog._actual_size_image()
            self.app.processEvents()

            self.assertFalse(dialog.fit_to_window)
            self.assertTrue(dialog.image_view._navigator.isVisible())
            self.assertGreater(dialog.image_view._navigator._visible_rect.width(), 0)
            self.assertGreater(dialog.image_view._navigator._visible_rect.height(), 0)

            dialog._fit_image_to_window()
            self.app.processEvents()

            self.assertTrue(dialog.fit_to_window)
            self.assertFalse(dialog.image_view._navigator.isVisible())
            dialog.close()

    def test_fit_mode_scales_small_pixmap_up_to_preview_bounds(self) -> None:
        pixmap = QPixmap(64, 48)
        scaled = ImagePreviewDialog._scale_pixmap_to_bounds(pixmap, 640, 480)

        self.assertEqual(scaled.width(), 640)
        self.assertEqual(scaled.height(), 480)

    def test_preview_transforms_grayscale_and_horizontal_mirror(self) -> None:
        image = QImage(2, 1, QImage.Format.Format_RGB32)
        image.setPixelColor(0, 0, QColor(255, 0, 0))
        image.setPixelColor(1, 0, QColor(0, 0, 255))
        pixmap = QPixmap.fromImage(image)

        transformed = ImagePreviewDialog._apply_preview_transforms(
            pixmap,
            grayscale=True,
            mirror_horizontal=True,
        ).toImage()

        left = transformed.pixelColor(0, 0)
        right = transformed.pixelColor(1, 0)
        self.assertEqual(left.red(), left.green())
        self.assertEqual(left.green(), left.blue())
        self.assertEqual(right.red(), right.green())
        self.assertEqual(right.green(), right.blue())
        self.assertLess(left.red(), right.red())

    def test_preview_transform_actions_use_icon_buttons(self) -> None:
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

            self.assertEqual(dialog.grayscale_button.text(), "")
            self.assertEqual(dialog.mirror_button.text(), "")
            self.assertFalse(dialog.grayscale_button.icon().isNull())
            self.assertFalse(dialog.mirror_button.icon().isNull())
            self.assertEqual(dialog.grayscale_button.accessibleName(), "黑白")
            self.assertEqual(dialog.mirror_button.accessibleName(), "左右翻转")
            dialog.close()

    def test_tab_toggles_linetop_panel_without_source_render_blocking_ui(self) -> None:
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

            with patch.object(dialog, "_request_linetop_render") as request_render:
                tab = QKeyEvent(
                    QEvent.Type.KeyPress,
                    Qt.Key.Key_Tab,
                    Qt.KeyboardModifier.NoModifier,
                )
                dialog.keyPressEvent(tab)

            self.assertFalse(dialog.advanced_panel.isHidden())
            self.assertTrue(dialog.advanced_toggle_button.isChecked())
            self.assertTrue(dialog.compare_toggle_button.isEnabled())
            self.assertTrue(dialog.save_render_button.isEnabled())
            self.assertTrue(request_render.called)

            dialog.keyPressEvent(
                QKeyEvent(
                    QEvent.Type.KeyPress,
                    Qt.Key.Key_Tab,
                    Qt.KeyboardModifier.NoModifier,
                )
            )
            self.assertTrue(dialog.advanced_panel.isHidden())
            self.assertFalse(dialog.advanced_toggle_button.isChecked())
            self.assertFalse(dialog.compare_toggle_button.isEnabled())
            self.assertFalse(dialog.compare_toggle_button.isChecked())
            self.assertFalse(dialog.save_render_button.isEnabled())
            dialog.close()

    def test_linetop_controls_build_expected_settings(self) -> None:
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

            dialog.linetop_color_limit_mode_button.setChecked(True)
            dialog.linetop_contrast_slider.setValue(180)
            dialog.linetop_brightness_slider.setValue(-12)
            dialog.linetop_color_limit_slider.setValue(5)
            dialog.linetop_color_grayscale_checkbox.setChecked(True)
            dialog.linetop_illustration_preset_button.setChecked(True)
            settings = dialog._current_linetop_settings_from_controls()

            self.assertEqual(
                settings,
                LineTopSettings(
                    mode="color_limit",
                    opacity=1.0,
                    edge_strength=2.0,
                    line_thickness=0.0,
                    overlay_contrast=1.8,
                    overlay_brightness=-0.12,
                    color_limit_steps=5,
                    color_limit_grayscale=True,
                    color_limit_shape_simplification=1,
                    smart_enhance=True,
                    smart_preset="illustration",
                    enhanced_line_engine=True,
                ),
            )
            self.assertFalse(hasattr(dialog, "linetop_opacity_slider"))
            self.assertFalse(hasattr(dialog, "linetop_shape_slider"))
            self.assertFalse(dialog.linetop_color_limit_row.isHidden())
            self.assertTrue(dialog.linetop_thickness_row.isHidden())
            dialog.close()

    def test_compare_toggle_switches_between_source_and_processed_preview(self) -> None:
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

            with patch.object(dialog, "_request_linetop_render") as request_render:
                dialog._set_linetop_panel_visible(True)
                self.assertTrue(request_render.called)
                request_render.reset_mock()
                dialog.compare_toggle_button.setChecked(True)
                self.assertFalse(request_render.called)
                dialog.compare_toggle_button.setChecked(False)
                self.assertTrue(request_render.called)

            dialog.close()

    def test_linetop_save_as_refuses_to_overwrite_source_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            Image.new("RGB", (64, 48), color="red").save(image_path)
            before = image_path.read_bytes()
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
            with patch.object(dialog, "_request_linetop_render"):
                dialog._set_linetop_panel_visible(True)

            with (
                patch(
                    "eidory.ui.image_preview_dialog.QFileDialog.getSaveFileName",
                    return_value=(str(image_path), "PNG Image (*.png)"),
                ),
                patch.object(QMessageBox, "warning") as warning,
                patch.object(dialog, "_render_linetop_export_image") as render_export,
            ):
                dialog._save_linetop_render_as()

            self.assertEqual(image_path.read_bytes(), before)
            self.assertFalse(render_export.called)
            self.assertTrue(warning.called)
            dialog.close()

    def test_preview_transform_actions_do_not_reload_source_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            Image.new("RGB", (1600, 1000), color="red").save(image_path)
            store = MetadataStore(root / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=1600,
                height=1000,
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

            with patch.object(dialog, "_load_preview_pixmap", wraps=dialog._load_preview_pixmap) as load_preview:
                dialog._set_grayscale_preview(True)
                dialog._set_mirrored_preview(True)
                dialog._set_grayscale_preview(False)
                dialog._set_mirrored_preview(False)

            self.assertEqual(load_preview.call_count, 0)
            dialog.close()

    def test_preview_transform_actions_reuse_thumbnail_before_source_refine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            thumbnail_path = root / "thumb.webp"
            Image.new("RGB", (1600, 1000), color="red").save(image_path)
            Image.new("RGB", (512, 320), color="red").save(thumbnail_path)
            store = MetadataStore(root / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(image_path),
                file_size=image_path.stat().st_size,
                width=1600,
                height=1000,
                created_time_ns=None,
                modified_time_ns=image_path.stat().st_mtime_ns,
            )
            store.update_thumbnail(image_id, str(thumbnail_path), "ready")

            dialog = ImagePreviewDialog(
                images=[store.get_image(image_id)],
                start_index=0,
                store=store,
                semantic_query=None,
                model_name="fake-model",
                model_revision="test",
                embedding_dim=2,
            )

            self.assertTrue(dialog._preview_base_pixmap.isNull())
            with patch.object(dialog, "_load_preview_pixmap", wraps=dialog._load_preview_pixmap) as load_preview:
                dialog._set_grayscale_preview(True)
                dialog._set_mirrored_preview(True)

            self.assertEqual(load_preview.call_count, 0)
            dialog.close()

    def test_image_preview_does_not_initialize_video_player(self) -> None:
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

            self.assertIsNone(dialog.video_player)
            self.assertIsNone(dialog.video_widget)
            self.assertIs(dialog.preview_stack.currentWidget(), dialog.image_view)
            dialog.close()

    def test_preview_double_click_fits_image_surface(self) -> None:
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
            dialog._actual_size_image()
            self.assertFalse(dialog.fit_to_window)

            event = QMouseEvent(
                QEvent.Type.MouseButtonDblClick,
                QPointF(10, 10),
                QPointF(10, 10),
                QPointF(10, 10),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )

            self.assertTrue(dialog.eventFilter(dialog.image_view.viewport(), event))
            self.assertTrue(dialog.isVisible())
            self.assertTrue(dialog.fit_to_window)
            self.assertEqual(dialog.zoom_factor, dialog._fit_zoom_factor())
            dialog.close()

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
        esc_event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Escape,
            Qt.KeyboardModifier.NoModifier,
        )
        self.assertTrue(ImagePreviewDialog._is_close_shortcut(esc_event))

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
            self.assertFalse(dialog.grayscale_button.isEnabled())
            self.assertFalse(dialog.mirror_button.isEnabled())
            self.assertFalse(dialog.video_controls_widget.isHidden())
            self.assertEqual(dialog.video_player.source().toLocalFile(), str(video_path))
            dialog.close()

    def test_preview_remove_index_keeps_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            Image.new("RGB", (64, 48), color="red").save(image_path)
            thumbnail_dir = root / "thumbs"
            thumbnail_dir.mkdir()
            thumbnail_path = thumbnail_dir / "thumb_000000001.webp"
            Image.new("RGB", (32, 24), color="red").save(thumbnail_path)
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
            store.update_thumbnail(image_id, str(thumbnail_path), "ready")
            dialog = ImagePreviewDialog(
                images=[store.get_image(image_id)],
                start_index=0,
                store=store,
                semantic_query=None,
                model_name="fake-model",
                model_revision="test",
                embedding_dim=2,
                thumbnail_dir=thumbnail_dir,
            )

            with patch("eidory.ui.image_preview_dialog.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
                dialog._remove_current_index()

            self.assertIsNone(store.get_image(image_id))
            self.assertTrue(image_path.exists())
            self.assertFalse(thumbnail_path.exists())
            dialog.close()


if __name__ == "__main__":
    unittest.main()
