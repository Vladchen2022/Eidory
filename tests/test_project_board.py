from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QMimeData, QPoint, Qt, QUrl, QTimer
from PySide6.QtGui import QColor, QImage, QKeyEvent, QPainter, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGraphicsScene

from eidory.models import ImageItem
from eidory.ui.project_board import BoardImageItem, ProjectBoardView, _cached_pixmap


class ProjectBoardViewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._boards = []

    def _board(self):
        board = ProjectBoardView()
        self._boards.append(board)
        return board

    def tearDown(self) -> None:
        # Standalone test views need the same decoder drain as MainWindow.close.
        # Retain Qt item wrappers until QApplication exits, as the shared window
        # fixture does; deleting them mid-suite is unsafe with PySide 6.11.
        for board in self._boards:
            board._image_loader.stop()
            for timer in board.findChildren(QTimer):
                timer.stop()
            self.assertTrue(board._image_loader.pool.waitForDone(3000))
            board.close()
        if not hasattr(self.app, '_board_test_views'):
            self.app._board_test_views = []
        self.app._board_test_views.extend(self._boards)
        self.app.processEvents()

    def wait_for_images(self, board):
        deadline = time.monotonic() + 5
        while board._pending_image_loads and time.monotonic() < deadline:
            QTest.qWait(2)
        self.assertFalse(board._pending_image_loads)

    def test_space_toggle_restores_fit_all_not_previous_manual_view(self) -> None:
        board = self._board()
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

    def test_board_accepts_external_file_drop_payload(self) -> None:
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile("/tmp/source.jpg")])

        self.assertTrue(ProjectBoardView._supports_external_import_drop(mime))

    def test_selected_wide_image_can_fit_beyond_default_zoom_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "wide.jpg"
            pixmap = QPixmap(400, 60)
            pixmap.fill(QColor("#667788"))
            self.assertTrue(pixmap.save(str(image_path)))

            board = self._board()
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

    def test_space_focus_upgrades_selected_image_from_thumbnail_to_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.png"
            thumbnail_path = Path(tmp) / "thumb.png"
            source = QPixmap(1800, 1200)
            source.fill(QColor("#667788"))
            thumbnail = QPixmap(180, 120)
            thumbnail.fill(QColor("#667788"))
            self.assertTrue(source.save(str(source_path)))
            self.assertTrue(thumbnail.save(str(thumbnail_path)))

            board = self._board()
            board.resize(1200, 800)
            board.show()
            self.app.processEvents()
            board.set_images(
                [
                    self._image(
                        1,
                        file_path=str(source_path),
                        thumbnail_path=str(thumbnail_path),
                        width=1800,
                        height=1200,
                    )
                ]
            )
            item = board._image_items[1]
            self.assertIsInstance(item, BoardImageItem)
            self.wait_for_images(board)
            self.assertEqual(max(item.source_pixmap().width(), item.source_pixmap().height()), 180)

            board._select_image_id(1)
            board.toggle_fit_selection_or_restore()

            self.wait_for_images(board)
            self.assertGreater(max(item.source_pixmap().width(), item.source_pixmap().height()), 384)
            self.assertEqual(board._view_mode, "fit_selection")

    def test_small_zoom_does_not_upgrade_selected_image_to_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.png"
            thumbnail_path = Path(tmp) / "thumb.png"
            source = QPixmap(1800, 1200)
            source.fill(QColor("#667788"))
            thumbnail = QPixmap(560, 373)
            thumbnail.fill(QColor("#667788"))
            self.assertTrue(source.save(str(source_path)))
            self.assertTrue(thumbnail.save(str(thumbnail_path)))

            board = self._board()
            board.resize(1200, 800)
            board.show()
            self.app.processEvents()
            board.set_images(
                [
                    self._image(
                        1,
                        file_path=str(source_path),
                        thumbnail_path=str(thumbnail_path),
                        width=1800,
                        height=1200,
                    )
                ]
            )
            item = board._image_items[1]
            self.assertIsInstance(item, BoardImageItem)
            # Establish the low zoom this test is about. Auto-fit can legitimately
            # request a larger image while color-managed decoding is in flight;
            # do not make the expectation depend on beating its 120 ms timer.
            board.reset_view()
            board.centerOn(item)
            self.wait_for_images(board)
            self.assertEqual(max(item.source_pixmap().width(), item.source_pixmap().height()), 384)

            board._select_image_id(1)
            board._zoom_by(1.2)

            self.assertEqual(max(item.source_pixmap().width(), item.source_pixmap().height()), 384)

            board.toggle_fit_selection_or_restore()

            self.wait_for_images(board)
            self.assertGreater(max(item.source_pixmap().width(), item.source_pixmap().height()), 384)

    def test_default_board_layout_uses_item_edges_to_avoid_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = []
            for image_id in range(1, 6):
                image_path = Path(tmp) / f"wide-{image_id}.jpg"
                pixmap = QPixmap(1200, 300)
                pixmap.fill(QColor("#667788"))
                self.assertTrue(pixmap.save(str(image_path)))
                images.append(self._image(image_id, file_path=str(image_path), width=1200, height=300))

            board = self._board()
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

            board = self._board()
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
            board = self._board()
            board.set_images([image])
            item = board._image_items[1]
            item.setPos(210, 320)
            getattr(item, "set_display_size")(260)
            getattr(item, "set_pinned")(True)
            getattr(item, "set_flipped")(True)
            getattr(item, "set_grayscale")(True)
            item.setVisible(False)

            payload = board.layout_payload()

            restored = self._board()
            restored.set_images([image], layout_payload=payload)
            restored_item = restored._image_items[1]
            self.assertEqual(restored_item.pos().x(), 210)
            self.assertEqual(restored_item.pos().y(), 320)
            self.assertAlmostEqual(getattr(restored_item, "display_width"), 260)
            self.assertFalse(restored_item.isVisible())
            self.assertTrue(getattr(restored_item, "is_pinned")())
            self.assertTrue(getattr(restored_item, "is_flipped")())
            self.assertTrue(getattr(restored_item, "is_grayscale")())

    def test_layout_payload_restores_local_display_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "first.png"
            pixmap = QPixmap(400, 300)
            pixmap.fill(QColor("#334455"))
            self.assertTrue(pixmap.save(str(image_path)))

            image = self._image(1, file_path=str(image_path), width=400, height=300)
            board = self._board()
            board.set_images([image])
            item = board._image_items[1]
            self.assertIsInstance(item, BoardImageItem)
            getattr(item, "set_local_display")(
                {
                    "enabled": True,
                    "type": "polygon",
                    "points": [[0.1, 0.1], [0.8, 0.2], [0.4, 0.75]],
                },
                shrink_to_bounds=True,
            )

            payload = board.layout_payload()
            restored = self._board()
            restored.set_images([image], layout_payload=payload)
            restored_item = restored._image_items[1]
            local_payload = getattr(restored_item, "local_display_payload")()

            self.assertEqual(local_payload["type"], "polygon")
            self.assertTrue(local_payload["enabled"])
            self.assertEqual(len(local_payload["points"]), 3)
            self.assertLess(getattr(restored_item, "display_width"), local_payload["full_width"])

    def test_local_display_toggle_preserves_saved_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "first.png"
            pixmap = QPixmap(400, 300)
            pixmap.fill(QColor("#334455"))
            self.assertTrue(pixmap.save(str(image_path)))

            image = self._image(1, file_path=str(image_path), width=400, height=300)
            board = self._board()
            board.set_images([image])
            item = board._image_items[1]
            self.assertIsInstance(item, BoardImageItem)
            getattr(item, "set_local_display")(
                {"enabled": True, "type": "rect", "bounds": {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.5}},
                shrink_to_bounds=True,
            )
            local_width = getattr(item, "display_width")

            board._select_image_id(1)
            self.assertTrue(board.selected_has_local_display())
            self.assertEqual(board.toggle_selected_local_display(), 1)
            self.assertFalse(getattr(item, "is_local_display_enabled")())
            self.assertGreater(getattr(item, "display_width"), local_width)
            self.assertIsNotNone(getattr(item, "local_display_payload")())

            self.assertEqual(board.toggle_selected_local_display(), 1)
            self.assertTrue(getattr(item, "is_local_display_enabled")())
            self.assertAlmostEqual(getattr(item, "display_width"), local_width, delta=0.01)

    def test_selected_has_local_display_requires_local_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "first.png"
            pixmap = QPixmap(400, 300)
            pixmap.fill(QColor("#334455"))
            self.assertTrue(pixmap.save(str(image_path)))

            image = self._image(1, file_path=str(image_path), width=400, height=300)
            board = self._board()
            board.set_images([image])
            board._select_image_id(1)
            self.assertFalse(board.selected_has_local_display())

            item = board._image_items[1]
            self.assertIsInstance(item, BoardImageItem)
            getattr(item, "set_local_display")(
                {"enabled": True, "type": "rect", "bounds": {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.5}},
                shrink_to_bounds=True,
            )

            self.assertTrue(board.selected_has_local_display())

    def test_polygon_local_display_renders_transparent_outside_polygon(self) -> None:
        pixmap = QPixmap(100, 100)
        pixmap.fill(QColor("#ff0000"))
        item = BoardImageItem(image_id=1, pixmap=pixmap, file_path="/tmp/source.png")

        item.set_local_display(
            {
                "enabled": True,
                "type": "polygon",
                "points": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            }
        )

        image = item.pixmap().toImage()
        self.assertEqual(image.pixelColor(5, 5).alpha(), 255)
        self.assertEqual(image.pixelColor(95, 95).alpha(), 0)

    def test_selected_board_image_renders_blue_outline(self) -> None:
        pixmap = QPixmap(80, 50)
        pixmap.fill(QColor("#f8f8f8"))
        item = BoardImageItem(image_id=1, pixmap=pixmap, file_path="/tmp/source.png")
        item.setPos(10, 10)

        scene = QGraphicsScene()
        scene.setSceneRect(0, 0, 110, 80)
        scene.addItem(item)
        item.setSelected(True)

        output = QImage(110, 80, QImage.Format.Format_ARGB32)
        output.fill(QColor("#101820"))
        painter = QPainter(output)
        scene.render(painter)
        painter.end()

        blue_pixels = 0
        for y in range(output.height()):
            for x in range(output.width()):
                color = output.pixelColor(x, y)
                if color.blue() > 210 and color.green() > 90 and color.red() < 90:
                    blue_pixels += 1

        self.assertGreater(blue_pixels, 120)

    def test_delete_key_requests_selected_image_removal(self) -> None:
        board = self._board()
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

    def test_context_menu_can_request_preview_for_board_image(self) -> None:
        board = self._board()
        board.resize(900, 700)
        board.set_images([self._image(1)])
        board.show()
        self.app.processEvents()
        emitted: list[int] = []
        board.imagePreviewRequested.connect(lambda image_id: emitted.append(int(image_id)))
        item_center = board._image_items[1].sceneBoundingRect().center()
        viewport_pos = board.mapFromScene(item_center)

        class FakeContextMenuEvent:
            def __init__(self, pos):
                self._pos = pos
                self.accepted = False

            def pos(self):
                return self._pos

            def globalPos(self):
                return board.viewport().mapToGlobal(self._pos)

            def accept(self):
                self.accepted = True

        fake_event = FakeContextMenuEvent(viewport_pos)

        class FakeAction:
            def __init__(self, text: str) -> None:
                self.text = text
                self.enabled = True

            def setEnabled(self, enabled: bool) -> None:
                self.enabled = enabled

        created_menus: list[object] = []

        class FakeMenu:
            def __init__(self, *_args, **_kwargs) -> None:
                self.actions: list[FakeAction] = []
                created_menus.append(self)

            def addAction(self, text: str) -> FakeAction:
                action = FakeAction(text)
                self.actions.append(action)
                return action

            def addSeparator(self) -> None:
                return

            def exec(self, *_args, **_kwargs):
                return self.actions[0]

        with patch("eidory.ui.project_board.QMenu", FakeMenu):
            board.contextMenuEvent(fake_event)

        self.assertEqual(emitted, [1])
        self.assertTrue(fake_event.accepted)
        menu_texts = [action.text for action in created_menus[0].actions]
        self.assertIn("通过预览显示图片", menu_texts)
        self.assertIn("从当前项目移除", menu_texts)
        self.assertNotIn("编辑局部展示", menu_texts)
        self.assertNotIn("切换局部/全图", menu_texts)
        self.assertNotIn("清除局部展示", menu_texts)

    def test_select_image_emits_selection_changed(self) -> None:
        board = self._board()
        board.set_images([self._image(1), self._image(2)])
        emitted: list[list[int]] = []
        board.selectionChanged.connect(lambda image_ids: emitted.append(list(image_ids)))

        board._select_image_id(2)

        self.assertEqual(emitted, [[2]])

    def test_select_image_uses_cached_selection_without_scene_scan(self) -> None:
        board = self._board()
        board.set_images([self._image(1), self._image(2)])
        emitted: list[list[int]] = []
        board.selectionChanged.connect(lambda image_ids: emitted.append(list(image_ids)))

        with patch.object(board._scene, "selectedItems", side_effect=AssertionError("scene selection scan")):
            board._select_image_id(2)

        self.assertEqual(board.selected_image_ids(), [2])
        self.assertEqual(emitted, [[2]])

    def test_select_image_does_not_emit_layout_changed(self) -> None:
        board = self._board()
        board.set_images([self._image(1), self._image(2)])
        self.app.processEvents()
        emitted: list[bool] = []
        board.layoutChanged.connect(lambda: emitted.append(True))

        board._select_image_id(2)
        self.app.processEvents()

        self.assertEqual(emitted, [])

    def test_item_position_change_emits_layout_changed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "first.jpg"
            pixmap = QPixmap(320, 180)
            pixmap.fill(QColor("#334455"))
            self.assertTrue(pixmap.save(str(image_path)))
            board = self._board()
            board.set_images([self._image(1, file_path=str(image_path), width=320, height=180)])
            self.app.processEvents()
            emitted: list[bool] = []
            board.layoutChanged.connect(lambda: emitted.append(True))

            board._image_items[1].setPos(123, 234)
            self.app.processEvents()
            time.sleep(0.55)
            self.app.processEvents()

            self.assertEqual(emitted, [True])

    def test_repeated_position_changes_emit_one_layout_changed_after_idle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "first.jpg"
            pixmap = QPixmap(320, 180)
            pixmap.fill(QColor("#334455"))
            self.assertTrue(pixmap.save(str(image_path)))
            board = self._board()
            board.set_images([self._image(1, file_path=str(image_path), width=320, height=180)])
            self.app.processEvents()
            emitted: list[bool] = []
            board.layoutChanged.connect(lambda: emitted.append(True))

            for index in range(40):
                board._image_items[1].setPos(123 + index, 234 + index)
                self.app.processEvents()
            time.sleep(0.55)
            self.app.processEvents()

            self.assertEqual(emitted, [True])

    def test_flush_pending_layout_change_emits_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "first.jpg"
            pixmap = QPixmap(320, 180)
            pixmap.fill(QColor("#334455"))
            self.assertTrue(pixmap.save(str(image_path)))
            board = self._board()
            board.set_images([self._image(1, file_path=str(image_path), width=320, height=180)])
            self.app.processEvents()
            emitted: list[bool] = []
            board.layoutChanged.connect(lambda: emitted.append(True))

            board._image_items[1].setPos(123, 234)
            self.app.processEvents()

            self.assertTrue(board.flush_pending_layout_change())
            self.assertEqual(emitted, [True])
            self.assertFalse(board.flush_pending_layout_change())

    def test_undo_shortcut_requests_board_removal_undo(self) -> None:
        board = self._board()
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

    def test_cached_pixmap_can_use_image_metadata_without_filesystem_stat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "large.jpg"
            pixmap = QPixmap(300, 180)
            pixmap.fill(QColor("#445566"))
            self.assertTrue(pixmap.save(str(image_path)))

            with patch.object(Path, "stat", side_effect=AssertionError("stat should not run")):
                cached = _cached_pixmap(
                    image_path,
                    max_side=160,
                    modified_time_ns=123,
                    file_size=456,
                )

            self.assertFalse(cached.isNull())

    def test_board_thumbnail_pixmap_uses_image_metadata_without_filesystem_stat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            thumbnail_path = Path(tmp) / "thumb.jpg"
            pixmap = QPixmap(300, 180)
            pixmap.fill(QColor("#445566"))
            self.assertTrue(pixmap.save(str(thumbnail_path)))
            image = self._image(1, file_path=str(Path(tmp) / "source.jpg"), thumbnail_path=str(thumbnail_path))

            with patch.object(Path, "stat", side_effect=AssertionError("stat should not run")):
                loaded = ProjectBoardView._pixmap_for(
                    image,
                    prefer_source=False,
                    max_side=160,
                )

            self.assertFalse(loaded.isNull())

    def test_board_item_batches_layout_notifications_while_dragging(self) -> None:
        pixmap = QPixmap(320, 180)
        pixmap.fill(QColor("#445566"))
        notifications: list[str] = []
        item = BoardImageItem(
            image_id=1,
            pixmap=pixmap,
            file_path="/tmp/source.jpg",
            on_layout_changed=lambda: notifications.append("changed"),
        )

        item._begin_layout_interaction()
        item.setPos(10, 10)
        item.setPos(24, 18)

        self.assertEqual(notifications, [])

        item._finish_layout_interaction()

        self.assertEqual(notifications, ["changed"])

    def test_board_view_batches_layout_notifications_for_multi_item_drag(self) -> None:
        board = self._board()
        board.set_images([self._image(1), self._image(2)])
        notifications: list[str] = []
        board.layoutChanged.connect(lambda: notifications.append("changed"))

        board._begin_item_drag_layout_batch()
        board._image_items[1].setPos(10, 10)
        board._image_items[2].setPos(24, 18)

        self.assertEqual(notifications, [])
        self.assertTrue(board._item_drag_layout_pending)

        board._finish_item_drag_layout_batch()
        self.assertTrue(board.flush_pending_layout_change())

        self.assertEqual(notifications, ["changed"])

    def test_board_mouse_drag_commits_position_once_on_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "source.jpg"
            pixmap = QPixmap(640, 360)
            pixmap.fill(QColor("#445566"))
            self.assertTrue(pixmap.save(str(image_path)))
            board = self._board()
            board.resize(1000, 760)
            board.show()
            self.app.processEvents()
            board.set_images([self._image(1, file_path=str(image_path), width=640, height=360)])
            self.app.processEvents()
            item = board._image_items[1]
            self.assertIsInstance(item, BoardImageItem)
            original_pos = QPoint(int(item.pos().x()), int(item.pos().y()))
            notifications: list[str] = []
            board.layoutChanged.connect(lambda: notifications.append("changed"))

            start = board.mapFromScene(item.sceneBoundingRect().center())
            QTest.mousePress(board.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
            QTest.mouseMove(board.viewport(), start + QPoint(90, 45), delay=0)

            self.assertEqual(QPoint(int(item.pos().x()), int(item.pos().y())), original_pos)

            QTest.mouseRelease(
                board.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                start + QPoint(90, 45),
            )
            self.app.processEvents()

            self.assertGreater(item.pos().x(), original_pos.x())
            self.assertGreater(item.pos().y(), original_pos.y())
            self.assertTrue(board.flush_pending_layout_change())
            self.assertEqual(notifications, ["changed"])

    @staticmethod
    def _image(
        image_id: int,
        *,
        file_path: str | None = None,
        thumbnail_path: str | None = None,
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
            thumbnail_path=thumbnail_path,
            thumbnail_status="ready",
            embedding_status="ready",
            is_missing=False,
            is_favorite=False,
            note=None,
        )
