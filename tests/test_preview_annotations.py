from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import sqlite3
import time
from unittest.mock import patch

import pytest
from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QInputDevice, QPointingDevice, QPixmap, QTabletEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

from eidory.core.metadata_store import MetadataStore
from eidory.core.preview_annotations import AnnotationStore, Stroke, decode, encode, erase
from eidory.ui.image_preview_dialog import ImagePreviewDialog
from eidory.ui.theme import apply_theme


@pytest.fixture
def preview(tmp_path):
    app = QApplication.instance() or QApplication([])
    store = MetadataStore(tmp_path / "library.sqlite3")
    store.initialize()
    folder = store.add_folder(str(tmp_path))
    images = []
    for i in range(2):
        path = tmp_path / f"ink-{i}.png"
        Image.new("RGB", (1000, 800), "white").save(path)
        iid, _ = store.upsert_image(folder_id=folder, file_path=str(path), file_size=path.stat().st_size,
            width=1000, height=800, created_time_ns=None, modified_time_ns=path.stat().st_mtime_ns)
        store.mark_embedding_not_required(iid)
        images.append(store.get_image(iid))
    dialog = ImagePreviewDialog(images=images, start_index=0, store=store, semantic_query=None,
                                model_name="test", model_revision="1", embedding_dim=2)
    dialog.show()
    deadline = time.monotonic()+3
    while not dialog.image_view._has_pixmap and time.monotonic() < deadline:
        QTest.qWait(10)
    assert dialog.image_view._has_pixmap
    yield dialog
    dialog.close()
    app.processEvents()


def draw(dialog, points=((.2, .5), (.5, .5), (.8, .5))):
    view = dialog.image_view
    dialog.annotation_controls.tool_buttons["pen"].click()
    positions = [view.mapFromScene(x*view._source_width, y*view._source_height) for x, y in points]
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=positions[0])
    for pos in positions[1:]:
        QTest.mouseMove(view.viewport(), pos)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=positions[-1])


def test_eraser_cuts_only_swept_part_and_preserves_style():
    stroke = Stroke(((.1, .5), (.9, .5)), .01, "#E53935")
    result = erase((stroke,), (.5, .2), (.5, .8), .05, (1, 1))
    assert len(result) == 2
    assert result[0].points[0] == stroke.points[0]
    assert result[0].points[-1][0] == pytest.approx(.445)
    assert result[1].points[0][0] == pytest.approx(.555)
    assert result[1].points[-1] == stroke.points[-1]
    assert all(s.color == stroke.color and s.width == stroke.width for s in result)
    assert erase((stroke,), (0, 0), (0, 0), .01, (1, 1)) == (stroke,)
    assert not erase((Stroke(((.5, .5),), .01, "#000000"),), (.5, .5), (.5, .5), .01, (1, 1))


def test_storage_rejects_corrupt_versions_and_nonfinite_geometry():
    strokes = (Stroke(((.3, .5), (.8, .9)), .005, "#FF0000"),)
    assert decode(encode(strokes)) == strokes
    for raw in ('{"version":2}', '{"version":1,"strokes":[{"points":[[NaN,.5]],"width":.1,"color":"#000000"}]}'):
        with pytest.raises(ValueError):
            decode(raw)


def test_draw_autosaves_reopen_and_source_metadata_untouched(preview):
    image = preview.current_image()
    source = Path(image.file_path).read_bytes()
    preview.store.add_tags_to_images([image.id], ["教学"])
    with preview.store.connect(readonly=True) as conn:
        before = {t: list(map(tuple, conn.execute(f"SELECT * FROM {t}")))
                  for t in ("images", "image_tags", "ai_vision_tags", "embeddings")}
    draw(preview)
    saved = AnnotationStore(preview.store.database_path).load(image.id)
    assert saved[0] == preview.image_view.annotations
    assert len(saved[0]) == 1
    preview.close()
    preview.annotation_controls.image = None
    preview._refresh()
    assert preview.image_view.annotations == saved[0]
    assert Path(image.file_path).read_bytes() == source
    with preview.store.connect(readonly=True) as conn:
        assert before == {t: list(map(tuple, conn.execute(f"SELECT * FROM {t}"))) for t in before}


def test_undo_redo_clear_and_image_switch_are_isolated(preview):
    draw(preview)
    ink = preview.image_view.annotations
    preview.annotation_controls.clear.click()
    assert not preview.image_view.annotations
    preview.annotation_controls.undo.click()
    assert preview.image_view.annotations == ink
    preview.annotation_controls.redo.click()
    assert not preview.image_view.annotations
    preview.annotation_controls.undo.click()
    QTest.keyClick(preview, Qt.Key.Key_Right)
    assert preview.index == 1 and not preview.image_view.annotations
    QTest.keyClick(preview, Qt.Key.Key_Left)
    assert preview.index == 0 and preview.image_view.annotations == ink
    assert not preview.image_view._ink_undo


def test_partial_erase_and_undo_through_real_pointer_input(preview):
    draw(preview)
    view = preview.image_view
    ink = view.annotations
    preview.annotation_controls.tool_buttons["eraser"].click()
    pos = view.mapFromScene(view._source_width*.5, view._source_height*.5)
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=pos)
    assert len(view.annotations) == 2
    preview.annotation_controls.undo.click()
    assert view.annotations == ink
    assert AnnotationStore(preview.store.database_path).load(preview.current_image().id)[0] == ink


def test_marks_follow_zoom_mirror_and_async_preview_resolution(preview):
    draw(preview)
    view = preview.image_view
    stroke = view.annotations[0]
    def mapped():
        x, y = stroke.points[0]
        w, h = view.ink_layer.aspect
        return view.ink_layer.mapToScene(QPointF(x*w, y*h))
    assert mapped().x()/view._source_width == pytest.approx(stroke.points[0][0])
    view.set_zoom_factor(3)
    view.set_pixmap(QPixmap(preview.current_image().file_path).scaled(500, 400),
                    original_width=1000, original_height=800, fit_to_window=False, zoom_factor=3)
    assert mapped().x()/view._source_width == pytest.approx(stroke.points[0][0])
    assert view.transform().m11()*view.ink_layer.transform().m11()*stroke.width == pytest.approx(3*800*stroke.width)
    preview.mirror_button.click()
    assert mapped().x()/view._source_width == pytest.approx(1-stroke.points[0][0])
    assert view.annotations[0] is stroke


def test_display_hide_and_copy_exclude_ink(preview):
    draw(preview)
    preview._copy_current_image()
    copied = QApplication.clipboard().image()
    assert not copied.isNull()
    assert copied.pixelColor(copied.width()//2, copied.height()//2).red() == 255
    assert copied.pixelColor(copied.width()//2, copied.height()//2).green() == 255
    preview.annotation_controls.visible.click()
    assert not preview.image_view.ink_layer.isVisible()
    preview.annotation_controls.tool_buttons["pen"].click()
    assert preview.image_view.ink_layer.isVisible()


def test_space_pans_in_drawing_mode_without_closing(preview):
    preview.annotation_controls.tool_buttons["pen"].click()
    QTest.keyPress(preview.image_view, Qt.Key.Key_Space)
    assert preview.isVisible() and preview.image_view._ink_space
    QTest.keyRelease(preview.image_view, Qt.Key.Key_Space)
    assert not preview.image_view._ink_space


def test_tablet_events_draw_once_and_ignore_pressure(preview):
    view = preview.image_view
    preview.annotation_controls.tool_buttons["pen"].click()
    device = QPointingDevice("Test pen", 12345, QInputDevice.DeviceType.Stylus,
        QPointingDevice.PointerType.Pen, QInputDevice.Capability.Position | QInputDevice.Capability.Pressure, 1, 2)
    for event_type, x, pressure in ((QEvent.Type.TabletPress, .2, .1),
                                    (QEvent.Type.TabletMove, .5, 1.), (QEvent.Type.TabletRelease, .8, 0.)):
        point = view.mapFromScene(x*view._source_width, .4*view._source_height)
        button = Qt.MouseButton.NoButton if event_type == QEvent.Type.TabletMove else Qt.MouseButton.LeftButton
        buttons = Qt.MouseButton.NoButton if event_type == QEvent.Type.TabletRelease else Qt.MouseButton.LeftButton
        event = QTabletEvent(event_type, device, QPointF(point), QPointF(view.viewport().mapToGlobal(point)),
            pressure, 0, 0, 0, 0, 0, Qt.KeyboardModifier.NoModifier, button, buttons)
        QApplication.sendEvent(view.viewport(), event)
        assert event.isAccepted()
    assert len(view.annotations) == 1
    assert len(view.annotations[0].points) == 3
    assert AnnotationStore(preview.store.database_path).load(preview.current_image().id)[0] == view.annotations


def test_overlay_actually_paints_above_image_and_scales(preview):
    draw(preview)
    view = preview.image_view
    for zoom in (1., 2.):
        view.set_zoom_factor(zoom)
        view.centerOn(view._source_width*.5, view._source_height*.5)
        QApplication.processEvents()
        point = view.mapFromScene(view._source_width*.5, view._source_height*.5)
        rendered = view.viewport().grab().toImage()
        ratio = rendered.devicePixelRatio()
        color = rendered.pixelColor(round(point.x()*ratio), round(point.y()*ratio))
        assert color.red() > 180 and color.green() < 120


def test_busy_database_keeps_unsaved_ink_and_blocks_navigation(preview):
    with sqlite3.connect(preview.store.database_path) as lock:
        lock.execute("BEGIN IMMEDIATE")
        start = time.monotonic()
        draw(preview)
        assert time.monotonic()-start < 1
        assert preview.annotation_controls.dirty
        assert not preview.annotation_controls.status.isHidden()
        with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Cancel):
            preview._move(1)
            preview.close()
        assert preview.index == 0 and preview.isVisible()
    assert preview.annotation_controls.flush()
    assert not preview.annotation_controls.dirty


def test_external_replacement_preserves_but_hides_old_marks(preview):
    draw(preview)
    image = preview.current_image()
    saved = preview.image_view.annotations
    Image.new("RGB", (500, 700), "blue").save(image.file_path)
    preview.annotation_controls.image = None
    preview._refresh()
    assert not preview.image_view.annotations and not preview.image_view._ink_editable
    assert preview.annotation_controls.blocked
    assert AnnotationStore(preview.store.database_path).load(image.id)[0] == saved


@pytest.mark.parametrize("width", [720, 1200, 1900])
def test_toolbar_layout_at_window_sizes_has_no_overlap(preview, width):
    apply_theme(QApplication.instance(), "paper", font_size=14)
    preview.resize(width, 820)
    QTest.qWait(70)
    assert not {"上一张", "下一张"} & {b.text() for b in preview.findChildren(QPushButton)}
    assert preview.width() == width
    assert preview.grayscale_button.height() == preview.fit_button.height() == preview.mirror_button.height()
    slider = preview.saturation_slider
    value = preview.saturation_value_label
    assert slider.mapToGlobal(slider.rect().topRight()).x() < value.mapToGlobal(value.rect().topLeft()).x()
    controls = preview.annotation_controls
    assert controls.isVisible() and controls.tool_buttons["pen"].isVisible()
    image_bottom = preview.image_view.mapTo(preview, preview.image_view.rect().bottomLeft()).y()
    assert preview.fit_button.mapTo(preview, preview.fit_button.rect().topLeft()).y() > image_bottom
    assert controls.mapTo(preview, controls.rect().topLeft()).y() > image_bottom
    for layout in (preview.preview_controls_layout, controls.layout()):
        rects = [layout.itemAt(i).geometry() for i in range(layout.count()) if not layout.itemAt(i).isEmpty()]
        assert all(not a.intersects(b) for i, a in enumerate(rects) for b in rects[i+1:])
        assert all(r.right() <= width for r in rects)
    apply_theme(QApplication.instance())


def test_annotation_options_only_open_for_selected_editing_tool(preview):
    controls = preview.annotation_controls
    assert controls.options.isHidden()
    assert controls.tool_buttons['pen'].isVisible()
    assert controls.tool_buttons['eraser'].isVisible()
    assert not controls.tool_buttons['pan'].isVisible()
    controls.tool_buttons['pen'].click()
    assert controls.options.isVisible() and controls.colors.isVisible()
    assert preview.image_view.annotation_tool == 'pen'
    controls.tool_buttons['eraser'].click()
    assert controls.options.isVisible() and not controls.colors.isVisible()
    assert not controls.tool_buttons['pen'].isChecked()
    assert preview.image_view.annotation_tool == 'eraser'
    controls.tool_buttons['eraser'].click()
    assert controls.options.isHidden()
    assert preview.image_view.annotation_tool == 'pan'
    controls.tool_buttons['pen'].setChecked(True)  # Native AX can toggle directly.
    assert controls.options.isVisible()
    controls.tool_buttons['pan'].click()
    assert controls.options.isHidden()
    assert not controls.tool_buttons['pen'].isChecked()


def test_collapsing_tools_keeps_ink_and_undo(preview):
    draw(preview)
    controls = preview.annotation_controls
    strokes = preview.image_view.annotations
    controls.tool_buttons['pen'].click()
    assert controls.options.isHidden()
    assert preview.image_view.annotations == strokes
    controls.tool_buttons['pen'].click()
    controls.undo.click()
    assert not preview.image_view.annotations
    controls.redo.click()
    assert preview.image_view.annotations == strokes


def test_advanced_output_actions_follow_processing_context(preview):
    actions = (preview.compare_toggle_button, preview.save_render_button, preview.copy_render_button)
    assert all(b.isHidden() for b in actions)
    preview.advanced_toggle_button.click()
    assert all(b.isVisible() for b in actions)
    preview.advanced_toggle_button.click()
    assert all(b.isHidden() for b in actions)


def test_delete_restore_retains_annotation_data(preview):
    draw(preview)
    image = preview.current_image()
    saved = AnnotationStore(preview.store.database_path).load(image.id)
    snapshot = preview.store.snapshot_images_for_restore([image.id])
    with preview.store.connect() as conn:
        conn.execute("DELETE FROM images WHERE id=?", (image.id,))
    assert AnnotationStore(preview.store.database_path).load(image.id) is None
    preview.store.restore_images_snapshot(snapshot)
    assert AnnotationStore(preview.store.database_path).load(image.id) == saved


def test_concurrent_preview_edits_cannot_silently_overwrite(preview):
    image = preview.current_image()
    notes = AnnotationStore(preview.store.database_path)
    first = (Stroke(((.3, .2),), .01, "#FF0000"),)
    second = (Stroke(((.6, .7),), .01, "#000000"),)
    signature = (image.file_size, image.modified_time_ns)
    notes.save(image.id, first, signature, expected=())
    with pytest.raises(ValueError, match="其他预览窗口"):
        notes.save(image.id, second, signature, expected=())
    assert notes.load(image.id)[0] == first
