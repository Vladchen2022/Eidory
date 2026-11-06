from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from test_filter_workflow import app, library, window
from test_image_upscale import annotate, annotations
from eidory.core.comfy_reference import ComfyClient
from eidory.core.image_upscale import ImageReplacement, UpscaleJobs, digest, processing_input_size
from eidory.core.preview_annotations import AnnotationStore, Stroke
from eidory.ui.theme import apply_theme


def add_source(library, tmp_path, *, size=(5001, 1000), fmt='PNG', orientation=1):
    paths, store, _ = library
    suffix = {'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp'}[fmt]
    path = tmp_path / ('large-source' + suffix)
    rgba = fmt in {'PNG', 'WEBP'}
    with Image.new('RGBA' if rgba else 'RGB', size, (40, 80, 120, 96) if rgba else (40, 80, 120)) as im:
        exif = Image.Exif()
        exif[274] = orientation
        im.save(path, fmt, exif=exif)
    width, height = size[::-1] if orientation in {5, 6, 7, 8} else size
    stat = path.stat()
    iid, _ = store.upsert_image(folder_id=store.add_folder(str(tmp_path)), file_path=str(path),
                               file_size=stat.st_size, width=width, height=height,
                               created_time_ns=None, modified_time_ns=stat.st_mtime_ns)
    return store.get_image(iid)


@pytest.fixture
def large_source(library, tmp_path):
    return add_source(library, tmp_path)


@pytest.fixture
def reduced_job(library, large_source, tmp_path):
    repo = UpscaleJobs(library[1], library[0])
    with patch.object(ComfyClient, 'request', return_value={'name': 'reduced.png'}):
        job = repo.create(large_source.id, 2, 'http://127.0.0.1:8188', reduce_input=True)
    result = tmp_path / 'result.png'
    Image.new('RGB', (job['target_width'], job['target_height']), 'green').save(result)
    job.update(state='ready', file_path=str(result), width=job['target_width'], height=job['target_height'])
    repo.save(job)
    return repo, large_source, job


@pytest.mark.parametrize('size,expected', [
    ((2048, 1358), (2048, 1358)),
    ((2049, 1000), (2048, 1000)),
    ((5001, 1000), (2048, 410)),
    ((1000, 5001), (410, 2048)),
    ((5001, 1), (2048, 1)),
])
def test_input_reduction_limits_long_edge_and_keeps_orientation(size, expected):
    assert processing_input_size(*size, reduce_input=True) == expected
    assert processing_input_size(*size) == size


@pytest.mark.parametrize('fmt,orientation', [('PNG', 1), ('JPEG', 6), ('WEBP', 1)])
def test_uploaded_input_is_reduced_but_source_backup_is_full_size(library, tmp_path, fmt, orientation):
    source = add_source(library, tmp_path, fmt=fmt, orientation=orientation)
    before = digest(source.file_path)
    repo = UpscaleJobs(library[1], library[0])
    uploaded_sizes = []

    def upload(method, endpoint, **kwargs):
        assert (method, endpoint) == ('POST', '/upload/image')
        with Image.open(kwargs['files']['image'][1]) as actual:
            actual.load()
            uploaded_sizes.append(actual.size)
            assert max(actual.size) == 2048
            assert actual.getexif().get(274, 1) == 1
            if fmt in {'PNG', 'WEBP'}:
                assert actual.getchannel('A').getextrema() == (96, 96)
        return {'name': 'reduced.png', 'subfolder': 'EidoryUpscale'}

    with patch.object(ComfyClient, 'request', side_effect=upload):
        job = repo.create(source.id, 2, 'http://127.0.0.1:8188', reduce_input=True)
    assert digest(source.file_path) == digest(job['original_path']) == before
    assert job['source_info']['width'] == source.width
    assert job['source_info']['height'] == source.height
    with Image.open(job['comparison_path']) as comparison:
        assert comparison.size == (source.width, source.height)
        assert comparison.getexif().get(274, 1) == 1
    assert job['input_reduced']
    assert (job['target_width'], job['target_height']) == tuple(n * 2 for n in uploaded_sizes[0])
    assert job['workflow']['3']['inputs']['resize_type.multiplier'] == 2
    saved = repo.get(job['id'])
    assert saved['input_info'] == job['input_info'] and saved['input_reduced']


def test_reduction_is_opt_in_and_small_inputs_stay_unchanged(library, large_source):
    repo = UpscaleJobs(library[1], library[0])
    with patch.object(ComfyClient, 'request') as upload:
        with pytest.raises(ValueError, match='4096'):
            repo.create(large_source.id, 2, 'http://127.0.0.1:8188')
        upload.assert_not_called()
    small = library[2][0]
    with patch.object(ComfyClient, 'request', return_value={'name': 'small.png'}):
        job = repo.create(small.id, 2, 'http://127.0.0.1:8188', reduce_input=True)
    assert not job['input_reduced']
    with Image.open(job['input_path']) as im:
        assert im.size == (small.width, small.height)


def test_adopt_and_restore_reduced_input_result_preserves_original_and_annotations(reduced_job):
    repo, source, job = reduced_job
    store = repo.store
    annotate(store, source)
    before = annotations(store, source.id)
    original_hash = digest(source.file_path)
    ink = (Stroke(((.1, .2), (.8, .9)), .01, '#E53935'),)
    notes = AnnotationStore(store.database_path)
    notes.save(source.id, ink, (source.file_size, source.modified_time_ns))
    replacement = ImageReplacement(store, repo.paths)
    result = replacement.replace(job)
    assert result.id == source.id and result.file_path == source.file_path
    assert result.width == 4096 < source.width
    assert annotations(store, source.id) == before
    assert notes.load(source.id)[0] == ink
    restored = replacement.restore(source.id)
    assert (restored.width, restored.height) == (source.width, source.height)
    assert digest(source.file_path) == original_hash
    assert annotations(store, source.id) == before
    assert notes.load(source.id) == (ink, (restored.file_size, restored.modified_time_ns))


def test_reduce_and_undo_restore_parameters_without_modifying_source(window, app, library, large_source):
    c = window.image_upscaler
    c.timer.stop()
    original_hash = digest(large_source.file_path)
    c.open(large_source)
    app.processEvents()
    panel = c.panel
    panel.mode.setCurrentIndex(2)
    panel.edge.setValue(4096)
    assert panel.reduce_input_button.isVisible() and not panel.start.isEnabled()
    with patch.object(ComfyClient, 'request') as upload:
        panel.reduce_input_button.click()
        assert panel.mode.currentIndex() == 0 and panel.start.isEnabled()
        assert not panel.mode.model().item(1).isEnabled()
        assert '输入 2048 × 410 → 4096 × 820' in panel.dimensions.text()
        assert '低于原图' in panel.dimensions.text()
        assert panel.reduce_input_button.text() == '撤销缩小输入'
        panel.reduce_input_button.click()
        upload.assert_not_called()
    assert (panel.mode.currentIndex(), panel.edge.value()) == (2, 4096)
    assert not panel.start.isEnabled()
    assert digest(large_source.file_path) == original_hash
    assert library[1].get_image(large_source.id).width == large_source.width
    assert not c.repo.all(large_source.id)
    panel.close()


def test_reduced_input_choice_survives_returning_to_preview(window, app, large_source):
    window.grid_view.set_images([large_source])
    window._open_image_preview(large_source)
    preview = next(iter(window._preview_dialogs))
    preview.upscale_button.click()
    app.processEvents()
    window.image_upscaler.panel.mode.setCurrentIndex(2)
    window.image_upscaler.panel.edge.setValue(4096)
    window.image_upscaler.panel.reduce_input_button.click()
    window.image_upscaler.panel.back.click()
    app.processEvents()
    preview.upscale_button.click()
    app.processEvents()
    assert window.image_upscaler.panel.reduced_input()
    assert window.image_upscaler.panel.start.isEnabled()
    window.image_upscaler.panel.reduce_input_button.click()
    assert not window.image_upscaler.panel.reduced_input()
    assert window.image_upscaler.panel.mode.currentIndex() == 2
    assert window.image_upscaler.panel.edge.value() == 4096
    preview.close()


@pytest.mark.parametrize('state', ['running', 'ready', 'paused'])
def test_reopened_reduced_job_keeps_actual_input_and_compares_full_original(window, app, reduced_job, state):
    repo, source, job = reduced_job
    job['state'] = state
    repo.save(job)
    c = window.image_upscaler
    c.timer.stop()
    c.open(source)
    app.processEvents()
    panel = c.panel
    assert panel.reduced_input() and panel.mode.currentIndex() == 0
    assert '输入 2048 × 410 → 4096 × 820' in panel.dimensions.text()
    assert not panel.reduce_input_button.isEnabled()
    assert not panel.mode.isEnabled()
    assert panel.load_key[0] == job['comparison_path']
    assert panel.start.isEnabled() == (state == 'paused')
    if state == 'ready':
        panel.discard.click()
        app.processEvents()
        assert panel.reduce_input_button.isEnabled()
        panel.reduce_input_button.click()
        assert not panel.reduced_input()
    panel.close()


def test_start_submits_reduced_input_and_adoption_clears_preprocessing(window, app, large_source):
    c = window.image_upscaler
    c.timer.stop()
    c.open(large_source)
    app.processEvents()
    panel = c.panel
    panel.reduce_input_button.click()

    def submit(job_id):
        job = c.repo.get(job_id)
        job['state'] = 'running'
        c.repo.save(job)
        return job

    with patch.object(ComfyClient, 'request', return_value={'name': 'reduced.png'}), patch.object(c.repo, 'submit', side_effect=submit):
        panel.start.click()
    app.processEvents()
    job = c.repo.all(large_source.id)[0]
    with Image.open(job['input_path']) as actual:
        assert actual.size == (2048, 410)
    assert job['input_reduced'] and job['target_width'] == 4096
    assert not panel.reduce_input_button.isEnabled()
    # A source update invalidates a previous preparation choice for that source.
    job['state'] = 'cancelled'
    c.repo.save(job)
    c.media_changed(large_source)
    assert large_source.id not in c.input_reductions
    assert not panel.reduced_input()
    panel.close()


@pytest.mark.parametrize('edge,shown', [(2047, False), (2048, False), (2049, True)])
def test_reduction_button_only_appears_above_2048(window, app, library, tmp_path, edge, shown):
    source = add_source(library, tmp_path, size=(edge, 100))
    c = window.image_upscaler
    c.open(source)
    app.processEvents()
    assert c.panel.reduce_input_button.isVisible() == shown
    c.panel.close()


@pytest.mark.parametrize('width', [720, 1200])
def test_reduction_controls_fit_preview_window(window, app, large_source, width):
    from PySide6.QtWidgets import QPushButton
    apply_theme(app, density='comfortable', font_size=15)
    window.grid_view.set_images([large_source])
    window._open_image_preview(large_source)
    preview = next(iter(window._preview_dialogs))
    preview.resize(width, 600)
    preview.upscale_button.click()
    app.processEvents()
    panel = window.image_upscaler.panel
    panel.reduce_input_button.click()
    QTest.qWait(60)
    assert preview.width() == width
    for button in panel.findChildren(QPushButton):
        if button.isVisible():
            assert panel.rect().contains(button.mapTo(panel, button.rect().bottomRight())), button.text()
    preview.close()
    apply_theme(app)
