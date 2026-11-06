"""Queue recovery, source preservation, review identity and GUI race regressions."""
import json
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from PIL import Image, ImageCms
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImageReader
from PySide6.QtTest import QTest

from test_filter_workflow import app, library, window
from test_image_upscale import upscale
from eidory.core.comfy_reference import ComfyClient
from eidory.core.duplicate_detection import DuplicateGroup, DuplicateMember
from eidory.core.duplicate_review import DuplicateReview, fingerprint
from eidory.core.image_upscale import digest
from eidory.core.upscale_workflow import UpscaleWorkflow
from eidory.core.image_loader import srgb_pixels
from eidory.core.thumbnailer import Thumbnailer
from eidory.ui.display_image import read_display_image
from eidory.ui.duplicate_compare_dialog import DuplicateCompareDialog


def workflow(library):
    return UpscaleWorkflow(library[1], library[0])


def test_queue_persists_order_pause_and_cancellation(library):
    repo = workflow(library)
    ids = [i.id for i in library[2]]
    accepted, skipped = repo.enqueue(ids, 2, 'http://127.0.0.1:8188')
    assert len(accepted) == 4 and not skipped
    repo = workflow(library)
    repo.move(accepted[2]['id'], -1)
    assert [j['image_id'] for j in repo.waiting()] == [ids[0], ids[2], ids[1], ids[3]]
    repo.cancel(accepted[0]['id'])
    assert len(repo.waiting()) == 3
    repo.store.set_setting('upscale.queue_paused', '1')
    with patch.object(ComfyClient, 'request', side_effect=AssertionError('must not submit')):
        assert repo.start_next() is None


def test_queue_only_starts_one_and_preserves_originals(library):
    repo = workflow(library)
    images = library[2]
    hashes = [digest(i.file_path) for i in images]
    accepted, _ = repo.enqueue([i.id for i in images], 2, 'http://127.0.0.1:8188')
    def request(client, method, route, **kwargs):
        return {'name': 'input.png'} if route == '/upload/image' else {'prompt_id': accepted[0]['id']}
    with patch.object(ComfyClient, 'request', request), patch.object(ComfyClient, 'validate'):
        job = repo.start_next()
        assert job['state'] == 'queued'
        assert repo.start_next() is None
    assert len(repo.waiting()) == 3
    assert [digest(i.file_path) for i in images] == hashes


def test_preparation_failure_is_durable_and_retryable(library):
    repo = workflow(library)
    image = library[2][0]
    with patch.object(ComfyClient, 'request', side_effect=requests.ConnectionError('offline')):
        with pytest.raises(ValueError, match='上传输入失败'):
            repo.create(image.id, 2, 'http://127.0.0.1:8188')
    job = repo.all()[0]
    assert job['state'] == 'failed' and 'ConnectionError' in job['diagnostic']
    retried = repo.retry(job['id'])
    assert retried['state'] == 'waiting' and retried['retry_of'] == job['id']
    assert repo.get(job['id'])['state'] == 'failed'


def test_interrupted_preparation_is_not_submitted_again(library):
    repo = workflow(library)
    jobs, _ = repo.enqueue([library[2][0].id], 2, 'http://127.0.0.1:8188')
    job = jobs[0]
    job['state'] = 'submitting'
    repo.save(job)
    with patch.object(ComfyClient, 'request', side_effect=AssertionError('unsafe submit')):
        recovered = repo.reconcile(job['id'])
    assert recovered['state'] == 'failed'


def test_offline_preparation_pauses_remaining_queue(library):
    repo = workflow(library)
    repo.enqueue([i.id for i in library[2]], 2, 'http://127.0.0.1:8188')
    with patch.object(ComfyClient, 'request', side_effect=requests.ConnectionError('offline')):
        with pytest.raises(ValueError):
            repo.start_next()
    assert len(repo.waiting()) == 3
    assert repo.store.get_setting('upscale.queue_paused') == '1'


def test_queued_source_changes_require_review(library):
    repo = workflow(library)
    image = library[2][0]
    jobs, _ = repo.enqueue([image.id], 2, 'http://127.0.0.1:8188')
    with repo.store.connect() as conn:
        conn.execute('UPDATE images SET file_size=file_size+1 WHERE id=?', (image.id,))
    with pytest.raises(ValueError, match='排队后原图发生变化'):
        repo.start_next()
    assert repo.get(jobs[0]['id'])['state'] == 'failed'


def test_cleanup_preserves_ready_candidates_recovery_and_sources(upscale):
    base, replacement, image, job = upscale
    repo = UpscaleWorkflow(base.store, base.paths)
    original_hash = digest(image.file_path)
    assert repo.storage_report(0)['eligible'] == []
    repo.discard(job['id'])
    report = repo.storage_report(0)
    assert len(report['eligible']) == 1
    backup = repo.root/'replacements'/'protected'/'original.png'
    backup.parent.mkdir(parents=True)
    backup.write_bytes(b'protected')
    assert repo.cleanup(report) > 0
    assert backup.read_bytes() == b'protected'
    assert digest(image.file_path) == original_hash
    assert repo.get(job['id'])['purged_at']
    with pytest.raises(ValueError, match='清理'):
        repo.discard(job['id'], undo=True)


def test_cleanup_revalidates_state_after_report(upscale):
    base, replacement, image, job = upscale
    repo = UpscaleWorkflow(base.store, base.paths)
    repo.discard(job['id'])
    report = repo.storage_report(0)
    repo.discard(job['id'], undo=True)
    assert repo.cleanup(report) == 0
    assert Path(job['original_path']).exists()


def test_export_keeps_original_and_links_new_library_image(upscale, tmp_path):
    base, replacement, image, job = upscale
    repo = UpscaleWorkflow(base.store, base.paths)
    source_hash = digest(image.file_path)
    repo.store.add_tags_to_images([image.id], ['继承标签'])
    target = tmp_path/'export'/'new.png'
    copied = repo.export_copy(job['id'], target)
    assert copied.id != image.id and (copied.width, copied.height) == (240, 160)
    assert digest(image.file_path) == source_hash
    assert repo.get(job['id'])['copies'][0]['image_id'] == copied.id
    assert repo.get(job['id'])['state'] == 'exported'
    with repo.store.connect(readonly=True) as conn:
        assert conn.execute('SELECT COUNT(*) FROM image_tags WHERE image_id=?', (copied.id,)).fetchone()[0] == 1
    with pytest.raises(ValueError, match='已存在'):
        repo.export_copy(job['id'], target)
    assert Image.open(target).info['icc_profile']


def group(images):
    return DuplicateGroup('near', 'test', tuple(DuplicateMember(i, 'folder', f'hash-{i.id}', 1) for i in images))


def test_review_decisions_survive_restart_and_invalidate_changed_files(library):
    review = DuplicateReview(library[1])
    g = group(library[2][:2])
    review.mark(g, 'keep')
    review.save_cursor(g, 0)
    reopened = DuplicateReview(library[1])
    assert reopened.decisions()[fingerprint(g)] == 'keep'
    changed = group([replace(library[2][0], modified_time_ns=999), library[2][1]])
    assert fingerprint(changed) not in reopened.decisions()
    assert reopened.cursor() == [fingerprint(g), 0]


def test_review_navigation_is_explicit_and_preserves_group_order(window, library, app):
    groups = [group(library[2][:2]), group(library[2][1:3]), group(library[2][2:])]
    dialog = DuplicateCompareDialog(groups, parent=window)
    dialog.mark_review('keep')
    dialog.change_group(1)
    dialog.mark_review('different')
    dialog.change_group(-1)
    assert dialog.group_index == 0
    dialog.change_group(1)
    assert dialog.group_index == 1  # ordinary next does not skip reviewed groups
    dialog.next_unreviewed()
    assert dialog.group_index == 2
    dialog.loader.stop()
    dialog.loader.pool.waitForDone(3000)
    dialog.close()


def test_gray_icc_display_and_thumbnail_match_ai_pixels(tmp_path):
    source = tmp_path/'gray.jpg'
    profile = (Path(__file__).parent/'fixtures/gray-dot-gain20.icc').read_bytes()
    Image.new('L', (80, 60), 128).save(source, icc_profile=profile)
    with Image.open(source) as original:
        expected = srgb_pixels(original).getpixel((20, 20))[0]
    reader = QImageReader(str(source))
    displayed = read_display_image(reader, str(source))
    assert abs(displayed.pixelColor(20, 20).red() - expected) <= 2
    thumb = Thumbnailer(tmp_path/'thumb').generate(1, str(source))
    with Image.open(thumb) as image:
        assert abs(image.getpixel((20, 20))[0] - expected) <= 3
        assert image.info['icc_profile']


def test_gallery_old_query_cannot_replace_new_scope(window, library, app):
    pending = []
    with patch.object(window, '_start_background_task', side_effect=lambda work, **_: pending.append(work)):
        window._reload_images()
        window._reload_images()
    latest = pending[-1]
    latest()
    count = window.grid_view.rowCount()
    window.grid_view.set_images([library[2][-1]])
    pending[0]()
    assert count == 4 and window.grid_view.images() == [library[2][-1]]


def test_gallery_slow_query_keeps_gui_timer_running(window, library, app):
    entered, release = threading.Event(), threading.Event()
    original = window.store.list_images
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)
    threads = []
    def launch(work, **_):
        thread = threading.Thread(target=work)
        threads.append(thread)
        thread.start()
        return True
    def query(**kw):
        entered.set()
        release.wait(2)
        return original(**kw)
    with patch.object(window, '_start_background_task', launch), patch.object(window.store, 'list_images', query):
        window._reload_images()
        assert entered.wait(1)
        QTest.qWait(100)
        assert len(ticks) >= 3
        release.set()
        for thread in threads:
            thread.join(3)
        app.processEvents()
    timer.stop()
    assert window.grid_view.rowCount() == 4


def test_task_center_can_open_specific_historical_result(window, app, upscale):
    repo, replacement, image, job = upscale
    window.image_upscaler.open_tasks()
    dialog = window.image_upscaler.tasks
    assert dialog.table.rowCount() == 1
    dialog.table.selectRow(0)
    dialog.open_result()
    app.processEvents()
    assert window.image_upscaler.panel.job['id'] == job['id']
    assert window.image_upscaler.panel.preview is not None
    window.image_upscaler.tasks.close()


def test_original_changed_on_disk_after_enqueue_is_not_processed(library):
    repo = workflow(library)
    image = library[2][0]
    repo.enqueue([image.id], 2, 'http://127.0.0.1:8188')
    Image.new('RGB', (120, 80), 'purple').save(image.file_path)
    with pytest.raises(ValueError, match='源文件发生变化'):
        repo.start_next()


def test_updating_one_image_preserves_selection_badges_and_layout(window, library):
    grid = window.grid_view
    images = library[2]
    grid.set_images(images, selected_image_ids=[images[0].id], badges_by_image_id={images[0].id: ['命中']})
    rectangles = list(grid._rects)
    with patch.object(grid, '_rebuild_layout', side_effect=AssertionError('full rebuild')), patch.object(grid, '_emit_selection', side_effect=AssertionError('selection reset')):
        grid.refresh_image(replace(images[0], width=240, height=160, modified_time_ns=123456))
    assert grid._rects == rectangles
    assert grid.selected_image_ids() == [images[0].id]
    assert grid._badges_by_image_id[images[0].id] == ['命中']


def test_old_thumbnail_is_upgraded_lazily_without_modifying_source(tmp_path):
    from eidory.core.thumbnailer import upgrade_display_thumbnail
    source, thumb = tmp_path/'source.jpg', tmp_path/'thumb_000000001.webp'
    profile = (Path(__file__).parent/'fixtures/gray-dot-gain20.icc').read_bytes()
    Image.new('L', (80, 60), 128).save(source, icc_profile=profile)
    Image.new('RGB', (40, 30), (128, 128, 128)).save(thumb)
    before = digest(source)
    upgrade_display_thumbnail(str(thumb), str(source))
    assert digest(source) == before
    with Image.open(source) as original, Image.open(thumb) as preview:
        expected = srgb_pixels(original).getpixel((20, 20))[0]
        assert preview.info['icc_profile']
        assert abs(preview.getpixel((20, 20))[0] - expected) <= 3


def test_duplicate_original_pixels_load_and_sync(window, library, app):
    from PySide6.QtGui import QImage
    dialog = DuplicateCompareDialog([group(library[2][:2])], parent=window)
    dialog.show()
    app.processEvents()
    with patch.object(dialog.loader, 'request') as request:
        dialog.actual_images()
        assert request.call_count == 2
        assert all(c.args[2] == 0 for c in request.call_args_list)
    for image in library[2][:2]:
        dialog._loaded((dialog._generation, image.id), QImage(120, 80, QImage.Format.Format_RGB32), image.file_path)
    left, right = list(dialog.views.values())
    left.set_zoom_factor(2.)
    assert right.zoom_factor == 2.
    dialog.loader.stop()
    dialog.loader.pool.waitForDone(3000)
    dialog.close()


def test_invalid_icc_has_clear_ai_error_but_thumbnail_remains_readable(tmp_path):
    path = tmp_path/'invalid.jpg'
    Image.new('RGB', (60, 40), 'green').save(path, icc_profile=b'not-an-icc-profile')
    with Image.open(path) as source:
        with pytest.raises(ValueError, match='ICC'):
            srgb_pixels(source)
    thumb = Thumbnailer(tmp_path/'thumbs').generate(1, str(path))
    assert Image.open(thumb).size == (60, 40)


def test_save_copy_keeps_comparison_open_and_refreshes_gallery_on_close(window, app, upscale, tmp_path):
    repo, replacement, image, job = upscale
    c = window.image_upscaler
    c.timer.stop()
    c.open_job(job['id'])
    panel, preview = c.panel, c.panel.preview
    app.processEvents()
    target = tmp_path/'saved-copy.png'
    with patch('eidory.ui.image_upscale.QFileDialog.getSaveFileName', return_value=(str(target), 'PNG')) as save_dialog:
        c.export_copy()
    assert save_dialog.call_args.args[0] is preview
    app.processEvents()
    assert target.exists()
    assert preview.isVisible() and preview.pages.currentWidget() is panel
    assert c.panel is panel
    assert '已保存新图片' in panel.status.text()
    preview.close()
    app.processEvents()
    assert any(i.file_path == str(target) for i in window.grid_view.images())


def test_cancelling_a_filtered_task_does_not_select_the_next_target(window, library):
    c = window.image_upscaler
    c.timer.stop()
    jobs, _ = c.repo.enqueue([i.id for i in library[2][:2]], 2, 'http://127.0.0.1:8188')
    c.open_tasks()
    dialog = c.tasks
    dialog.filter.setCurrentIndex(1)
    dialog.table.selectRow(0)
    assert dialog.selected()['id'] == jobs[0]['id']
    dialog.cancel()
    assert dialog.selected() is None
    assert not dialog.buttons['cancel'].isEnabled()
    assert len(c.repo.waiting()) == 1
    dialog.close()
