"""Batch identity, completion prompts and explicit per-image review decisions."""
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest

from test_filter_workflow import app, library, window
from eidory.core.comfy_reference import ComfyClient
from eidory.core.image_upscale import digest
from eidory.core.upscale_workflow import UpscaleWorkflow
from eidory.core.scanner import ImageScanner
from eidory.core.thumbnailer import Thumbnailer


def ready_jobs(repo, images):
    jobs, skipped = repo.enqueue([i.id for i in images], 2, 'http://127.0.0.1:8188')
    assert not skipped
    for queued in jobs:
        with patch.object(ComfyClient, 'request', return_value={'name': 'input.png'}):
            job = repo.create(queued['image_id'], 2, queued['url'], queue_id=queued['id'])
        candidate = repo.root/'candidates'/'0'/(job['id']+'.png')
        candidate.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (240, 160), '#87a36b').save(candidate)
        job.update(state='ready', file_path=str(candidate), width=240, height=160)
        repo.save(job)
    return repo.batch(jobs[0]['batch_id'])['jobs']


def wait_loaded(dialog, app):
    for _ in range(100):
        app.processEvents()
        if dialog.loaded_sides == {'before', 'after'}:
            return
        QTest.qWait(10)
    pytest.fail('comparison images did not load')


@pytest.fixture
def review(window, library, app):
    c = window.image_upscaler
    c.timer.stop()
    jobs = ready_jobs(c.repo, library[2][:3])
    c.open_batch(jobs[0]['batch_id'])
    dialog = c.batch_review
    wait_loaded(dialog, app)
    yield c, dialog, jobs
    if not dialog.closed:
        dialog.close()
    dialog.loader.pool.waitForDone(3000)


def test_ten_selected_images_are_one_durable_ordered_batch(library):
    paths, store, images = library
    parent = Path(images[0].file_path).parent
    for i in range(6):
        Image.new('RGB', (120, 80), 'yellow').save(parent/f'extra-{i}.png')
    scan = ImageScanner(store, Thumbnailer(paths.thumbnail_dir)).scan_folder(str(parent))
    ids = list(reversed(scan.image_ids))
    assert len(ids) == 10
    repo = UpscaleWorkflow(store, paths)
    jobs, _ = repo.enqueue(ids, 2, 'http://127.0.0.1:8188')
    repo.move(jobs[5]['id'], -1)
    reopened = UpscaleWorkflow(store, paths)
    batch = reopened.batch(jobs[0]['batch_id'])
    assert len(reopened.batches()) == 1 and batch['total'] == 10
    assert batch['pending'] == 10 and not batch['complete']
    assert [j['image_id'] for j in batch['jobs']] == ids


def test_each_submission_has_its_own_batch_and_retry_keeps_its_slot(library):
    paths, store, images = library
    repo = UpscaleWorkflow(store, paths)
    first, _ = repo.enqueue([i.id for i in images[:2]], 2, 'http://127.0.0.1:8188')
    second, _ = repo.enqueue([i.id for i in images[2:]], 2, 'http://127.0.0.1:8188')
    assert first[0]['batch_id'] != second[0]['batch_id']
    failed = first[0]; failed['state'] = 'failed'; repo.save(failed)
    retry = repo.retry(failed['id'])
    assert retry['batch_id'] == failed['batch_id'] and retry['batch_index'] == 0
    batch = repo.batch(failed['batch_id'])
    assert batch['total'] == 2 and batch['jobs'][0]['id'] == retry['id']
    assert repo.get(failed['id'])['state'] == 'failed'


def test_batch_enqueue_is_atomic_if_persisting_one_task_fails(library):
    paths, store, images = library
    repo = UpscaleWorkflow(store, paths)
    save = repo.save
    count = 0
    def fail_second(job, conn=None):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError('test write failure')
        save(job, conn)
    with patch.object(repo, 'save', side_effect=fail_second):
        with pytest.raises(OSError):
            repo.enqueue([i.id for i in images], 2, 'http://127.0.0.1:8188')
    assert repo.all() == []


def test_completion_prompt_waits_for_entire_batch_and_survives_restart(window, library):
    c = window.image_upscaler; c.timer.stop()
    jobs, _ = c.repo.enqueue([i.id for i in library[2][:2]], 2, 'http://127.0.0.1:8188')
    jobs[0]['state'] = 'ready'; c.repo.save(jobs[0])
    c.notify_batches()
    assert c.batch_notice is None
    jobs[1]['state'] = 'failed'; c.repo.save(jobs[1])
    c.notify_batches()
    assert c.batch_notice is not None
    assert '1 张待审核' in c.batch_notice.text() and '1 张失败' in c.batch_notice.text()
    c.batch_notice.reject()
    c.notify_batches()
    assert c.batch_notice is None
    assert c.repo.get(jobs[0]['id'])['state'] == 'ready'
    assert c.w.store.get_setting('upscale.batch_notified.'+jobs[0]['batch_id'])


def test_paused_or_offline_jobs_are_not_announced_as_completed(library):
    paths, store, images = library
    repo = UpscaleWorkflow(store, paths)
    jobs, _ = repo.enqueue([i.id for i in images[:2]], 2, 'http://127.0.0.1:8188')
    for job, state in zip(jobs, ['paused', 'offline']):
        job['state'] = state; repo.save(job)
    batch = repo.batch(jobs[0]['batch_id'])
    assert batch['pending'] == 2 and not batch['complete']


def test_review_never_automatically_adopts_and_skip_leaves_pending(review, library):
    c, dialog, jobs = review
    hashes = [digest(i.file_path) for i in library[2]]
    assert dialog.job_id == jobs[0]['id']
    dialog.next_pending()
    assert dialog.job_id == jobs[1]['id']
    assert all(c.repo.get(j['id'])['state'] == 'ready' for j in jobs)
    assert [digest(i.file_path) for i in library[2]] == hashes
    dialog.close()
    c.open_batch(jobs[0]['batch_id'])
    assert c.batch_review.job_id == jobs[0]['id']
    c.batch_review.close()


def test_discard_advances_only_within_batch_and_can_be_undone(review, library, app):
    c, dialog, jobs = review
    hashes = [digest(i.file_path) for i in library[2]]
    dialog.decide('discard')
    assert c.repo.get(jobs[0]['id'])['state'] == 'discarded'
    assert dialog.job_id == jobs[1]['id']
    assert [digest(i.file_path) for i in library[2]] == hashes
    dialog.move(-1); wait_loaded(dialog, app)
    assert dialog.undo.isEnabled()
    dialog.undo_decision()
    assert c.repo.get(jobs[0]['id'])['state'] == 'ready'
    assert dialog.job_id == jobs[0]['id']


def test_adopt_changes_one_source_then_next_and_undo_restores(review, library, app):
    c, dialog, jobs = review
    hashes = [digest(i.file_path) for i in library[2]]
    dialog.decide('adopt')
    assert c.repo.get(jobs[0]['id'])['state'] == 'applied'
    assert dialog.job_id == jobs[1]['id']
    assert digest(library[2][0].file_path) != hashes[0]
    assert [digest(i.file_path) for i in library[2][1:]] == hashes[1:]
    dialog.move(-1); wait_loaded(dialog, app)
    assert dialog.undo.isEnabled()
    dialog.undo_decision()
    assert digest(library[2][0].file_path) == hashes[0]
    assert c.repo.get(jobs[0]['id'])['state'] == 'ready'


def test_external_change_prevents_adoption_and_does_not_advance(review, library):
    c, dialog, jobs = review
    Image.new('RGB', (120, 80), 'orange').save(library[2][0].file_path)
    changed = digest(library[2][0].file_path)
    dialog.decide('adopt')
    assert dialog.job_id == jobs[0]['id']
    assert c.repo.get(jobs[0]['id'])['state'] == 'ready'
    assert digest(library[2][0].file_path) == changed
    assert '外部' in dialog.status.text()


def test_export_copy_preserves_source_and_advances_only_after_success(review, library, tmp_path, app):
    c, dialog, jobs = review
    hashes = [digest(i.file_path) for i in library[2]]
    with patch('eidory.ui.upscale_batch_review.QFileDialog.getSaveFileName', return_value=('', '')):
        dialog.export_copy()
    assert dialog.job_id == jobs[0]['id']
    destination = tmp_path/'review-copy.png'
    with patch('eidory.ui.upscale_batch_review.QFileDialog.getSaveFileName', return_value=(str(destination), 'PNG')):
        dialog.export_copy()
    assert destination.exists() and c.repo.get(jobs[0]['id'])['state'] == 'exported'
    assert dialog.job_id == jobs[1]['id']
    assert [digest(i.file_path) for i in library[2]] == hashes
    dialog.close()
    for _ in range(50):
        app.processEvents()
        if any(i.file_path == str(destination) for i in c.w.grid_view.images()):
            break
        QTest.qWait(10)
    assert any(i.file_path == str(destination) for i in c.w.grid_view.images())


def test_escape_closes_review_loader_and_preserves_pending_results(review):
    c, dialog, jobs = review
    dialog.reject()
    assert dialog.closed and dialog.loader.stopped and c.batch_review is None
    assert c.repo.batch(jobs[0]['batch_id'])['ready'] == 3


def test_failed_export_does_not_mark_reviewed_or_advance(review, tmp_path):
    c, dialog, jobs = review
    destination = tmp_path/'already.png'; destination.write_bytes(b'keep')
    with patch('eidory.ui.upscale_batch_review.QFileDialog.getSaveFileName', return_value=(str(destination), 'PNG')):
        dialog.export_copy()
    assert destination.read_bytes() == b'keep'
    assert c.repo.get(jobs[0]['id'])['state'] == 'ready'
    assert dialog.job_id == jobs[0]['id']


def test_all_decisions_finish_with_explicit_summary_and_no_source_changes(review, library, app):
    c, dialog, jobs = review
    hashes = [digest(i.file_path) for i in library[2]]
    for job in jobs:
        wait_loaded(dialog, app)
        assert dialog.job_id == job['id']
        dialog.decide('discard')
    assert c.repo.batch(jobs[0]['batch_id'])['reviewed'] == 3
    assert '全部审核完成' in dialog.status.text()
    assert c.result_button.text() == 'AI 任务与结果'
    assert [digest(i.file_path) for i in library[2]] == hashes


def test_late_image_from_previous_job_cannot_enable_current_decision(review):
    c, dialog, jobs = review
    old = dialog.generation
    dialog.move(1)
    dialog.loaded((old, 'after'), QImage(30, 30, QImage.Format.Format_RGB32), 'old.png')
    assert dialog.view.after.isNull()
    assert not dialog.adopt.isEnabled() and not dialog.discard.isEnabled()


def test_task_center_keeps_selected_batch_and_opens_its_review(review, library):
    c, dialog, jobs = review
    extra, _ = c.repo.enqueue([library[2][3].id], 2, 'http://127.0.0.1:8188')
    c.open_tasks()
    tasks = c.tasks
    tasks.batch_filter.setCurrentIndex(tasks.batch_filter.findData(jobs[0]['batch_id']))
    tasks.refresh()
    assert tasks.table.rowCount() == 3 and tasks.review.isEnabled()
    tasks.review_batch()
    assert c.batch_review is dialog
    assert tasks.batch_filter.currentData() != extra[0]['batch_id']
    tasks.close()
