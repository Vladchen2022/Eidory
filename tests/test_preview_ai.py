import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest

from test_filter_workflow import app, library, window
from test_image_upscale import annotate, annotations
from eidory.core.comfy_reference import ComfyClient, GeneratedReferences, prepare_template, sampling_description
from eidory.core.comfy_workflows import official_turbo_graph
from eidory.core.image_upscale import ImageReplacement, UpscaleJobs, digest
from eidory.core.preview_ai import PreviewAIJobs, raster_mask
from eidory.ui.preview_ai import MaskCanvas

DRAWING = [dict(points=[[.35, .4], [.65, .6]], diameter=.3)]


@pytest.fixture(scope='module', autouse=True)
def suspend_qt_wrapper_cycle_collection():
    # PySide can collect a deleted widget wrapper during a subsequent Qt
    # constructor, returning NULL without an exception. Keep the UI tests in
    # this module deterministic; shared fixtures retain the QObject roots.
    import gc
    enabled = gc.isenabled()
    gc.disable()
    yield
    if enabled:
        gc.enable()


def candidate(library):
    paths, store, images = library
    repo = PreviewAIJobs(store, paths)
    job = repo.create(images[0].id, 'edit', '改成绿色', 'http://127.0.0.1:8188', drawing=DRAWING)
    target = repo.root/'candidates'/'0'/(job['id']+'.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (96, 64), 'green').save(target)
    job.update(state='ready', file_path=str(target), thumbnail_path=str(target.with_suffix('.webp')))
    repo.save(job)
    return repo, repo.composite(job), images[0]


def test_official_graph_uses_eight_step_schedule():
    template = prepare_template(official_turbo_graph())
    assert '8 步' in sampling_description(template)
    assert '11' not in template['graph']
    assert template['graph']['1']['inputs']['unet_name'] == 'qwen_image_2.1_turbo_bf16.safetensors'


def test_mask_erasing_and_inward_feather():
    strokes = DRAWING + [dict(points=[[.5, .5]], diameter=.1, erase=True)]
    hard = np.asarray(raster_mask(strokes, (120, 80)))
    feather = np.asarray(raster_mask(strokes, (120, 80), feather=8))
    assert hard[40, 60] == 0
    assert np.all(feather[hard==0]==0)
    assert np.any((feather > 0) & (feather < 255))
    with pytest.raises(ValueError):
        raster_mask([dict(points=[[float('nan'), .5]], diameter=.1)], (120, 80))


def test_edit_keeps_unselected_pixels_and_source_untouched(library):
    before = Path(library[2][0].file_path).read_bytes()
    repo, job, image = candidate(library)
    hard = np.asarray(raster_mask(DRAWING, (image.width, image.height)))
    original = np.asarray(Image.open(image.file_path).convert('RGB'))
    result = np.asarray(Image.open(job['file_path']).convert('RGB'))
    assert np.array_equal(result[hard==0], original[hard==0])
    assert np.any(result[hard==255] != original[hard==255])
    assert Path(image.file_path).read_bytes() == before
    assert result.shape == original.shape


def test_edit_replace_restore_recovers_original_and_semantics(library):
    repo, job, image = candidate(library)
    annotate(repo.store, image)
    before, sha = annotations(repo.store, image.id), digest(image.file_path)
    repo.replace(job['id'])
    assert repo.get(job['id'])['state']=='applied'
    with repo.store.connect(readonly=True) as conn:
        assert conn.execute('SELECT status FROM embeddings WHERE image_id=?', (image.id,)).fetchone()[0]=='stale'
    ImageReplacement(repo.store, repo.paths).restore(image.id)
    assert digest(image.file_path)==sha
    assert repo.get(job['id'])['state']=='restored'
    assert annotations(repo.store, image.id)==before


def test_external_changes_block_replacement(library):
    repo, job, image = candidate(library)
    Image.new('RGB', (120,80), 'yellow').save(image.file_path)
    sha = digest(image.file_path)
    with pytest.raises(ValueError, match='变化'):
        repo.replace(job['id'])
    assert digest(image.file_path)==sha
    assert repo.get(job['id'])['state']=='ready'


def test_edit_recovery_finishes_interrupted_file_database_commit(library):
    repo, job, image = candidate(library)
    annotate(repo.store, image)
    before, sha = annotations(repo.store, image.id), digest(image.file_path)
    replacement = ImageReplacement(repo.store, repo.paths)
    finish = replacement._finish
    calls = []
    def interrupted(record, conn):
        calls.append(record['id'])
        if len(calls)==1:
            raise RuntimeError('simulated failure after rename')
        return finish(record, conn)
    with patch.object(replacement, '_finish', side_effect=interrupted):
        with pytest.raises(RuntimeError):
            replacement.replace(job)
    assert repo.get(job['id'])['state']=='applied'
    replacement.restore(image.id)
    assert digest(image.file_path)==sha
    assert annotations(repo.store, image.id)==before


def test_save_copy_indexes_new_image_once_and_preserves_original(library, tmp_path):
    repo, job, source = candidate(library)
    sha = digest(source.file_path)
    collection = repo.store.create_collection('AI 测试')
    image = repo.adopt(job['id'], tmp_path/'saved', collection)
    assert image.id != source.id
    assert Path(image.file_path).is_file()
    assert digest(source.file_path)==sha
    assert repo.adopt(job['id'], tmp_path/'saved').id == image.id
    assert repo.store.count_images()==5
    with repo.store.connect(readonly=True) as conn:
        assert conn.execute('SELECT 1 FROM image_collections WHERE image_id=? AND collection_id=?', (image.id, collection)).fetchone()


def test_generation_needs_no_creative_project_and_locks_other_jobs(library):
    paths, store, images = library
    repo = PreviewAIJobs(store, paths)
    job = repo.create(images[0].id, 'generate', '一间厨房', 'http://127.0.0.1:8188', width=512, height=768)
    assert job['project_id']==0 and 'node_id' not in job
    assert job['workflow']['5']['inputs']['width']==512
    with pytest.raises(ValueError, match='等待'):
        UpscaleJobs(store, paths).create(images[1].id, 2, job['url'])
    with pytest.raises(ValueError, match='等待'):
        repo.create(images[1].id, 'generate', 'other', job['url'])
    reopened = PreviewAIJobs(store, paths)
    assert reopened.active()['id']==job['id']


def test_edit_graph_uploads_reference_and_mask_without_replacing_source(library):
    paths, store, images = library
    repo = PreviewAIJobs(store, paths)
    job = repo.create(images[0].id, 'edit', '去掉文字', 'http://127.0.0.1:8188', drawing=DRAWING)
    sha = digest(images[0].file_path)
    def request(client, method, route, **kw):
        if route=='/upload/image':
            return {'subfolder': 'EidoryEdit', 'name': kw['files']['image'][0]}
        assert route=='/prompt'
        graph = kw['json']['prompt']
        assert graph['6']['inputs']['latent_image']==['4',2]
        assert graph['4']['inputs']['images.image_2']==['16',0]
        assert graph['9']['inputs']['image'].startswith('EidoryEdit/')
        return {'prompt_id': job['id']}
    with patch.object(ComfyClient, 'validate'), patch.object(ComfyClient, 'request', request):
        assert repo.submit(job['id'])['state']=='queued'
    assert digest(images[0].file_path)==sha


def test_discard_is_reversible_and_does_not_touch_original(library):
    repo, job, image = candidate(library)
    sha = digest(image.file_path)
    assert repo.discard(job['id'])['state']=='discarded'
    restored = repo.discard(job['id'], undo=True)
    assert restored['state']=='ready'
    assert Path(restored['file_path']).is_file()
    assert digest(image.file_path)==sha


def test_composite_retry_is_idempotent_after_interrupted_commit(library):
    repo, job, _ = candidate(library)
    sha = digest(job['file_path'])
    job['composited'] = False
    repo.save(job)
    assert digest(repo.reconcile(job['id'])['file_path'])==sha


def test_lost_submit_response_preserves_identity_without_replay(library):
    import requests
    repo = PreviewAIJobs(library[1],library[0])
    job = repo.create(library[2][0].id, 'generate', 'one image', 'http://127.0.0.1:8188')
    with patch.object(ComfyClient,'validate'), patch.object(ComfyClient,'request',side_effect=requests.Timeout) as request:
        assert repo.submit(job['id'])['state']=='unknown'
        assert repo.submit(job['id'])['state']=='unknown'
    assert request.call_count==1
    assert repo.get(job['id'])['prompt_id']==job['id']


def test_project_controller_does_not_poll_preview_jobs(window, library):
    repo = window.preview_ai.repo
    job = repo.create(library[2][0].id, 'generate', 'image', 'http://127.0.0.1:8188')
    assert window.reference_generation.active() is None
    assert window.reference_generation.repo.active()['id']==job['id']
    with patch.object(window.reference_generation.repo,'reconcile') as reconcile:
        window.reference_generation.poll()
        reconcile.assert_not_called()


def test_grayscale_source_can_be_colorized_and_restored(library):
    paths, store, images = library
    image = images[0]
    Image.new('L',(120,80),128).save(image.file_path)
    repo, job, image = candidate(library)
    sha = digest(image.file_path)
    repo.replace(job['id'])
    assert Image.open(image.file_path).mode=='RGB'
    ImageReplacement(store,paths).restore(image.id)
    assert digest(image.file_path)==sha


def test_embedded_ready_result_keeps_new_candidates_after_return(window, library, app):
    repo, job, image = candidate(library)
    window._open_image_preview(image)
    preview = next(iter(window._preview_dialogs))
    preview.edit_button.click()
    app.processEvents()
    panel = window.preview_ai.panel
    assert panel.job_id==job['id']
    assert panel.save.isEnabled() and panel.replace.isEnabled()
    assert not panel.comparison.after.isNull()
    panel.back.click()
    app.processEvents()
    assert repo.get(job['id'])['state']=='ready'
    preview.edit_button.click()
    app.processEvents()
    assert window.preview_ai.panel.job_id==job['id']
    window.preview_ai.panel.back.click()
    preview.close()


def test_history_comparison_uses_snapshot_but_new_canvas_uses_current_source(window, library, app):
    repo, job, image = candidate(library)
    original = Image.open(image.file_path).convert('RGB').getpixel((0,0))
    Image.new('RGB', (120,80), 'yellow').save(image.file_path)
    window._open_image_preview(image)
    preview = next(iter(window._preview_dialogs))
    preview.edit_button.click()
    app.processEvents()
    panel = window.preview_ai.panel
    assert panel.canvas.before.pixelColor(0,0).name()=='#ffff00'
    assert panel.comparison.before.pixelColor(0,0).getRgb()[:3]==original
    panel.back.click()
    preview.close()


def test_mask_canvas_zoom_mapping_and_undo(app):
    canvas = MaskCanvas()
    canvas.resize(600,400)
    from PySide6.QtGui import QImage
    image = QImage(120,80,QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    canvas.set_image('before', image)
    canvas.show()
    app.processEvents()
    point = canvas.image_rect().center()
    assert canvas.point(point)==pytest.approx([.5,.5])
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=point.toPoint())
    assert len(canvas.strokes)==1
    canvas.undo()
    assert not canvas.strokes
    canvas.redo()
    assert len(canvas.strokes)==1
    canvas.scale *= 2
    canvas.pan = QPointF(45,-20)
    assert canvas.point(canvas.image_rect().center())==pytest.approx([.5,.5])
    canvas.close()


def test_preview_pages_and_drafts_survive_close(window, library, app):
    image = library[2][0]
    window._open_image_preview(image)
    preview = next(iter(window._preview_dialogs))
    assert preview.generate_button.text()=='AI 生图'
    assert preview.edit_button.isEnabled()
    preview.edit_button.click()
    app.processEvents()
    panel = window.preview_ai.panel
    assert preview.pages.currentWidget() is panel
    panel.prompt.setPlainText('只修改局部')
    panel.canvas.strokes = DRAWING.copy()
    panel.back.click()
    app.processEvents()
    assert preview.pages.currentWidget() is preview.preview_page
    preview.edit_button.click()
    app.processEvents()
    panel = window.preview_ai.panel
    assert panel.prompt.toPlainText()=='只修改局部'
    assert panel.canvas.strokes==DRAWING
    panel.back.click()
    preview.close()
