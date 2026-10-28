from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from eidory.core.art_analysis import MAX_ANALYSIS_VIEW_EDGE, analyze_image_art, render_art_analysis_view
from eidory.core.metadata_store import MetadataStore
from eidory.ui.image_preview_dialog import ImagePreviewDialog


@pytest.fixture
def preview(tmp_path):
    app = QApplication.instance() or QApplication([])
    store = MetadataStore(tmp_path / "library.sqlite3")
    store.initialize()
    folder = store.add_folder(str(tmp_path))
    images = []
    for name, color in [("red", "red"), ("blue", "blue")]:
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (120, 80), color).save(path)
        image_id, _ = store.upsert_image(
            folder_id=folder, file_path=str(path), file_size=path.stat().st_size,
            width=120, height=80, created_time_ns=None,
            modified_time_ns=path.stat().st_mtime_ns,
        )
        images.append(store.get_image(image_id))
    dialog = ImagePreviewDialog(
        images=images, start_index=0, store=store, semantic_query=None,
        model_name="test", model_revision="test", embedding_dim=2,
    )
    with patch.object(dialog, "_start_art_analysis_task"), patch.object(dialog, "_request_linetop_render"):
        dialog._set_linetop_panel_visible(True)
        dialog.linetop_art_analysis_button.setChecked(True)
        yield dialog
        dialog.close()
        app.processEvents()


def test_late_analysis_does_not_overwrite_next_image(preview):
    first, second = preview.images
    first_result = analyze_image_art(first.file_path)
    second_result = analyze_image_art(second.file_path)
    first_token = preview._art_analysis_cache_key(first)
    preview._move(1)
    second_token = preview._art_analysis_cache_key(second)
    preview._handle_art_analysis_loaded(first_token, first_result, "")
    assert preview.art_analysis_widget._result is None
    assert second.file_name in preview.art_analysis_widget.status_label.text()
    preview._handle_art_analysis_loaded(second_token, second_result, "")
    assert preview.art_analysis_widget._result == second_result
    assert not preview.art_analysis_widget.content.isHidden()


def test_closed_preview_ignores_late_analysis_and_view(preview):
    image = preview.current_image()
    token = preview._art_analysis_cache_key(image)
    result = analyze_image_art(image.file_path)
    preview.close()
    with patch.object(preview.art_analysis_widget, "set_result") as update, \
            patch.object(preview.image_view, "set_pixmap") as paint:
        preview._handle_art_analysis_loaded(token, result, "")
        preview._handle_art_analysis_view_loaded((token, "detail"), QImage(), "")
    update.assert_not_called()
    paint.assert_not_called()


def test_analysis_error_can_retry_without_changing_source(preview):
    image = preview.current_image()
    original = Path(image.file_path).read_bytes()
    token = preview._art_analysis_cache_key(image)
    preview._handle_art_analysis_loaded(token, None, "read failed")
    assert "read failed" in preview.art_analysis_widget.status_label.text()
    preview._request_art_analysis(image)
    preview._handle_art_analysis_loaded(token, analyze_image_art(image.file_path), "")
    assert not preview.art_analysis_widget.content.isHidden()
    assert Path(image.file_path).read_bytes() == original


def test_analysis_view_keeps_zoom_and_pan_while_rendering(preview):
    from PySide6.QtGui import QPixmap

    image = preview.current_image()
    preview._art_analysis_cache[preview._art_analysis_cache_key(image)] = analyze_image_art(image.file_path)
    preview.fit_to_window = False
    preview.zoom_factor = 8.0
    source = QPixmap(120, 80)
    source.fill()
    preview.image_view.set_pixmap(
        source, original_width=120, original_height=80,
        fit_to_window=False, zoom_factor=8.0,
    )
    QApplication.processEvents()
    preview.image_view.centerOn(90, 50)
    before = preview.image_view._current_center_ratio()
    with patch.object(preview, "_start_art_analysis_view_task") as render, \
            patch.object(preview.image_view, "set_message") as clear:
        preview._set_art_analysis_view("detail")
        render.assert_called_once()
        clear.assert_not_called()
    assert preview.image_view.zoom_factor == 8.0
    output = QImage(120, 80, QImage.Format.Format_RGBA8888)
    output.fill(0xFFFFFFFF)
    preview._handle_art_analysis_view_loaded(preview._art_analysis_view_pending_token, output, "")
    after = preview.image_view._current_center_ratio()
    assert preview.image_view.zoom_factor == 8.0
    assert after == pytest.approx(before, abs=.002)


def test_analysis_visualizations_bound_processing_size(tmp_path):
    path = tmp_path / "large.png"
    Image.new("RGB", (2000, 1800), "red").save(path)
    result = analyze_image_art(str(path))
    rendered = render_art_analysis_view(
        str(path), result, mode="detail", max_width=16000, max_height=14400,
    )
    assert max(rendered.size) <= MAX_ANALYSIS_VIEW_EDGE
    assert abs(rendered.width / rendered.height - 2000 / 1800) < .002
