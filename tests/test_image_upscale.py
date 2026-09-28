import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import requests
from PIL import Image
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest

from test_filter_workflow import app, library, window
from eidory.core.comfy_reference import ComfyClient
from eidory.core.image_upscale import UpscaleJobs, ImageReplacement, digest, output_size, source_info
from eidory.core.scanner import ImageScanner
from eidory.core.thumbnailer import Thumbnailer
from eidory.ui.image_upscale import WipeComparison


@pytest.fixture
def upscale(library):
    paths, store, images = library
    repo = UpscaleJobs(store, paths)
    image = images[0]
    with patch.object(ComfyClient, 'request', return_value={'name': 'test.png', 'subfolder': 'EidoryUpscale'}):
        job = repo.create(image.id, 2, 'http://127.0.0.1:8188')
    candidate = repo.root / 'test-result.png'
    Image.new('RGB', (240, 160), '#87a36b').save(candidate)
    job.update(state='ready', file_path=str(candidate), width=240, height=160)
    repo.save(job)
    return repo, ImageReplacement(store, paths), image, job


def annotations(store, image_id):
    with store.connect(readonly=True) as conn:
        return {t: [tuple(r) for r in conn.execute(f'SELECT * FROM {t} WHERE image_id=?', (image_id,))]
                for t in ['image_tags', 'embeddings', 'ai_vision_tags', 'creative_node_images', 'creative_reference_notes', 'creative_reference_decisions', 'color_features']}


def annotate(store, image):
    store.add_tags_to_images([image.id], ['保留测试'])
    store.update_note(image.id, '不覆盖的备注')
    store.update_favorite(image.id, True)
    store.upsert_embedding_success(image_id=image.id, model_name='test', model_revision='1', vector=np.array([.3, .4, .5]))
    project = store.create_creative_project(title='保留关联')
    node = store.creative_root_node_id(project)
    with store.connect() as conn:
        conn.execute("INSERT INTO ai_vision_tags(image_id,analysis_type,provider_name,model_name,prompt_version,status,scene_location,created_at,updated_at) VALUES(?,'scene','test','test','1','ready','森林','now','now')", (image.id,))
        conn.execute("INSERT INTO creative_node_images(node_id,image_id,created_at) VALUES(?,?,'now')", (node, image.id))
        conn.execute("INSERT INTO creative_reference_notes(node_id,image_id,note) VALUES(?,?,'造型参考')", (node, image.id))
        conn.execute("INSERT INTO creative_reference_decisions(node_id,image_id,status,purpose) VALUES(?,?,'primary','shape')", (node, image.id))


def test_replace_scan_restore_preserves_every_annotation(upscale):
    repo, replacement, original, job = upscale
    store = repo.store
    annotate(store, original)
    before = annotations(store, original.id)
    old_hash = digest(original.file_path)
    result = replacement.replace(job)
    assert result.id == original.id and result.file_path == original.file_path
    assert (result.width, result.height) == (240, 160)
    assert result.is_favorite and result.note == '不覆盖的备注'
    assert annotations(store, result.id) == before
    assert digest(result.file_path) != old_hash
    # Actual scanner and a fresh store connection must not clear retained AI content.
    ImageScanner(store, Thumbnailer(repo.paths.thumbnail_dir)).scan_folder(str(Path(result.file_path).parent))
    store.initialize()
    assert annotations(store, result.id) == before
    restored = replacement.restore(original.id)
    assert digest(restored.file_path) == old_hash
    assert (restored.width, restored.height) == (120, 80)
    assert annotations(store, restored.id) == before
    assert repo.get(job['id'])['state'] == 'restored'
    assert not replacement.latest(original.id)


def test_scan_observation_taken_before_replace_cannot_erase_metadata(upscale):
    repo, replacement, original, job = upscale
    annotate(repo.store, original)
    before = annotations(repo.store, original.id)
    replacement.replace(job)
    iid, state = repo.store.upsert_image(folder_id=original.folder_id, file_path=original.file_path,
        file_size=original.file_size, width=120, height=80, created_time_ns=None, modified_time_ns=original.modified_time_ns)
    assert state == 'unchanged'
    assert repo.store.get_image(iid).width == 240
    assert annotations(repo.store, iid) == before


def test_unrelated_external_edit_still_invalidates_semantics(upscale):
    repo, replacement, original, job = upscale
    annotate(repo.store, original)
    replacement.replace(job)
    Image.new('RGB', (110, 90), 'purple').save(original.file_path)
    ImageScanner(repo.store, Thumbnailer(repo.paths.thumbnail_dir)).scan_folder(str(Path(original.file_path).parent))
    with repo.store.connect(readonly=True) as conn:
        assert conn.execute('SELECT vector_blob FROM embeddings WHERE image_id=?', (original.id,)).fetchone()[0] is None
    with pytest.raises(ValueError, match='外部'):
        replacement.restore(original.id)


def test_reject_source_changed_while_generating(upscale):
    repo, replacement, image, job = upscale
    Image.new('RGB', (120, 80), 'orange').save(image.file_path)
    changed = digest(image.file_path)
    with pytest.raises(ValueError, match='外部'):
        replacement.replace(job)
    assert digest(image.file_path) == changed
    assert repo.get(job['id'])['state'] == 'ready'


def test_discard_does_not_touch_source_or_image_count(upscale):
    repo, replacement, image, job = upscale
    old = digest(image.file_path)
    repo.discard(job['id'])
    assert digest(image.file_path) == old
    assert repo.get(job['id'])['state'] == 'discarded'
    assert repo.discard(job['id'], undo=True)['state'] == 'ready'
    assert len(repo.store.list_images()) == 4
    assert not replacement.latest(image.id)


def test_recover_process_crash_after_file_swap(upscale):
    repo, replacement, image, job = upscale
    annotate(repo.store, image)
    before = annotations(repo.store, image.id)
    # BaseException simulates immediate process death, bypassing error recovery.
    with patch.object(replacement, '_finish', side_effect=SystemExit):
        with pytest.raises(SystemExit):
            replacement.replace(job)
    assert repo.store.get_image(image.id).width == 120
    fresh = ImageReplacement(repo.store, repo.paths)
    fresh.recover()
    assert repo.store.get_image(image.id).width == 240
    assert repo.get(job['id'])['state'] == 'applied'
    assert annotations(repo.store, image.id) == before
    fresh.restore(image.id)
    assert digest(image.file_path) == job['source_digest']


def test_failed_swap_preserves_original_and_ready_result(upscale):
    repo, replacement, image, job = upscale
    with patch('eidory.core.image_upscale.os.replace', side_effect=OSError('disk full')):
        with pytest.raises(OSError):
            replacement.replace(job)
    assert digest(image.file_path) == job['source_digest']
    assert repo.get(job['id'])['state'] == 'ready'
    assert not replacement.latest(image.id)


def test_cannot_replace_while_ai_analysis_is_writing(upscale):
    repo, replacement, image, job = upscale
    annotate(repo.store, image)
    with repo.store.connect() as conn:
        conn.execute("UPDATE ai_vision_tags SET status='processing' WHERE image_id=?", (image.id,))
    with pytest.raises(ValueError, match='索引'):
        replacement.replace(job)
    assert digest(image.file_path) == job['source_digest']


@pytest.mark.parametrize('size,factor,expected', [((1024,1024),2,(2048,2048)), ((512,384),4,(2048,1536)), ((900,601),2048/900,(2048,1368))])
def test_dimensions(size, factor, expected):
    assert output_size(*size, factor) == expected


@pytest.mark.parametrize('factor', [0, 1, 10, float('nan'), float('inf'), 4])
def test_reject_invalid_or_excessive_size(factor):
    with pytest.raises(ValueError):
        output_size(1500, 1000, factor)


@pytest.mark.parametrize('fmt', ['JPEG', 'PNG', 'WEBP'])
def test_original_format_retained_with_transparency_and_orientation(library, tmp_path, fmt):
    paths, store, _ = library
    suffix = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'TIFF': '.tif', 'BMP': '.bmp'}[fmt]
    source = tmp_path / ('source' + suffix)
    im = Image.new('RGBA' if fmt in {'PNG','WEBP','TIFF'} else 'RGB', (80, 60), (80, 110, 150, 90) if fmt in {'PNG','WEBP','TIFF'} else (80,110,150))
    exif = Image.Exif()
    exif[274] = 6
    opts = {'exif': exif} if fmt in {'JPEG','PNG','WEBP','TIFF'} else {}
    im.save(source, fmt, **opts)
    iid = ImageScanner(store, Thumbnailer(paths.thumbnail_dir)).import_files([str(source)]).image_ids[0]
    repo = UpscaleJobs(store, paths)
    with patch.object(ComfyClient, 'request', return_value={'name':'input.png'}):
        job = repo.create(iid, 2, 'http://localhost:8188')
    target = tmp_path / 'result.png'
    Image.new('RGB', (job['target_width'],job['target_height']), 'green').save(target)
    job.update(state='ready', file_path=str(target))
    repo.save(job)
    replacement = ImageReplacement(store, paths)
    replacement.replace(job)
    with Image.open(source) as actual:
        assert actual.format == fmt
        assert actual.size == (job['target_width'], job['target_height'])
        assert actual.getexif().get(274,1) == 1
        if fmt in {'PNG','WEBP','TIFF'}:
            assert actual.getchannel('A').getextrema() == (90,90)
    replacement.restore(iid)
    assert digest(source) == job['source_digest']
    restored = store.get_image(iid)
    info = source_info(source)
    assert (restored.width, restored.height) == (info['width'], info['height'])


def test_submission_timeout_keeps_identity_no_duplicate(upscale):
    repo, _, _, job = upscale
    job['state']='submitting'
    repo.save(job)
    with patch.object(ComfyClient, 'validate'), patch.object(ComfyClient, 'request', side_effect=requests.Timeout) as post:
        assert repo.submit(job['id'])['state'] == 'unknown'
        repo.submit(job['id'])
        assert post.call_count == 1


def test_preview_single_entry_opens_source_and_closes_safely(window, app, library):
    image = library[2][0]
    window._open_image_preview(image)
    preview = next(iter(window._preview_dialogs))
    preview.upscale_button.click()
    app.processEvents()
    panel = window.image_upscaler.panel
    assert panel.image_id == image.id
    assert panel.info['width'] == 120
    assert panel.start.isEnabled()
    assert '240 × 160' in panel.dimensions.text()
    panel.close()
    app.processEvents()
    assert window.image_upscaler.panel is None
    preview.close()


def test_compare_uses_same_output_coordinates_and_can_drag_divider(app):
    view=WipeComparison()
    view.resize(800,600)
    view.show()
    view.set_image('before', QImage(120,80,QImage.Format.Format_RGB32))
    view.set_image('after', QImage(480,320,QImage.Format.Format_RGB32))
    view.actual()
    assert view.dimensions().width() == 480 and view.scale == 1
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(400,300))
    QTest.mouseMove(view,QPoint(600,300))
    QTest.mouseRelease(view,Qt.MouseButton.LeftButton,pos=QPoint(600,300))
    assert abs(view.divider-.75)<.01
    view.close()


def test_media_refresh_keeps_board_layout_and_search_badges(window, app, library, tmp_path):
    from dataclasses import replace
    paths, store, images = library
    image = images[0]
    board = window.project_board_view
    board.set_images(images)
    window._board_image_by_id = {x.id:x for x in images}
    window._current_board_image_ids = tuple(x.id for x in images)
    item = board._image_items[image.id]
    item.setPos(322, 244)
    item.setSelected(True)
    payload = board.layout_payload()
    window.grid_view.set_images(images, badges_by_image_id={image.id:['命中目标']})
    updated = replace(image,width=240,height=160,modified_time_ns=image.modified_time_ns+1)
    window.image_upscaler.media_changed(updated)
    app.processEvents()
    assert board.layout_payload() == payload
    assert board._image_items[image.id] is item
    assert window.grid_view._badges_by_image_id[image.id] == ['命中目标']
    assert board._images_by_id[image.id].width == 240
    assert '240 × 160' in window.creative_flow.board_inspector.info.text()


def test_result_can_be_reopened_and_discard_undone_without_new_job(window, app, library, upscale):
    repo, replacement, image, job = upscale
    # The fixtures share the same library; opening shows the durable result.
    window.image_upscaler.open(image)
    app.processEvents()
    panel = window.image_upscaler.panel
    assert panel.replace.isEnabled() and not panel.start.isEnabled()
    panel.close()
    app.processEvents()
    window.image_upscaler.open(image)
    app.processEvents()
    panel = window.image_upscaler.panel
    assert panel.job['id'] == job['id']
    panel.discard.click()
    app.processEvents()
    assert panel.discard.text() == '撤销丢弃'
    assert panel.start.isEnabled() and not panel.replace.isEnabled()
    panel.discard.click()
    app.processEvents()
    assert panel.replace.isEnabled()
    assert len(repo.all(image.id)) == 1
    panel.close()


def test_view_opened_during_poll_is_initialized_after_operation(window, app, library):
    c=window.image_upscaler
    c.working=True
    c.open(library[2][0])
    assert c.panel.info is None
    c.finished(lambda _: None, None)
    app.processEvents()
    assert c.panel.info['width'] == 120
    c.panel.close()


def test_uploaded_subfolder_not_rejected_by_top_level_file_combo():
    client=ComfyClient('http://127.0.0.1:8188')
    schema={'LoadImage':{'input':{'required':{'image':[['top-level.png'],{}]}}}}
    with patch.object(client,'request',side_effect=[{},schema]):
        assert '成功' in client.validate({'graph':{'1':{'class_type':'LoadImage','inputs':{'image':'EidoryUpscale/upload.png'}}}})


def test_escape_dismissed_panel_can_be_reopened(window, app, library):
    c=window.image_upscaler
    c.open(library[2][0]);app.processEvents()
    QTest.keyClick(c.panel,Qt.Key.Key_Escape);app.processEvents()
    assert c.panel is None
    c.open(library[2][0]);app.processEvents()
    assert c.panel.info and c.panel.start.isEnabled()
    c.panel.close()


def test_replacement_preserves_embedded_png_text(upscale):
    from PIL.PngImagePlugin import PngInfo
    repo,replacement,image,job=upscale
    info=PngInfo();info.add_text('workflow','original workflow');info.add_text('description','原始图片说明')
    with Image.open(image.file_path) as im:
        im.save(image.file_path,'PNG',pnginfo=info)
    job['source_digest']=digest(image.file_path)
    replacement.replace(job)
    with Image.open(image.file_path) as im:
        assert im.info['workflow']=='original workflow'
        assert im.info['description']=='原始图片说明'


@pytest.mark.parametrize('state,factor,mode,width,height', [('running',4,1,480,320), ('ready',3,2,360,240)])
def test_reopened_task_shows_submitted_dimensions(window, app, upscale, state, factor, mode, width, height):
    repo,_,image,job=upscale
    job.update(state=state,factor=factor,target_width=width,target_height=height)
    repo.save(job)
    c=window.image_upscaler
    c.open(image);app.processEvents()
    assert c.panel.mode.currentIndex()==mode
    assert f'{width} × {height}' in c.panel.dimensions.text()
    assert not c.panel.mode.isEnabled()
    if mode==2:
        assert c.panel.edge.value()==width
    c.panel.close()


def test_adopting_custom_size_offers_valid_next_scale(window, app, upscale):
    repo,replacement,image,job=upscale
    c=window.image_upscaler
    c.open(image);app.processEvents()
    c.panel.mode.setCurrentIndex(2)
    c.panel.edge.setValue(job['target_width'])
    updated=replacement.replace(job)
    c.media_changed(updated);app.processEvents()
    assert c.panel.mode.currentIndex()==0
    assert c.panel.start.isEnabled()
    assert '480 × 320' in c.panel.dimensions.text()
    c.panel.close()
