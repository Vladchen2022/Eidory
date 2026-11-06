from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from test_filter_workflow import app, library, window
from test_upscale_input_reduction import add_source
from eidory.core.comfy_reference import ComfyClient
from eidory.core.image_upscale import digest
from eidory.core.upscale_workflow import UpscaleWorkflow, batch_size_plan
from eidory.ui.upscale_tasks import BatchUpscaleDialog


@pytest.mark.parametrize('size,requested,actual,reduced,output', [
    ((1024, 512), 4, 4, False, [4096, 2048]),
    ((1025, 500), 4, 2, False, [2050, 1000]),
    ((2048, 1024), 4, 2, False, [4096, 2048]),
    ((2049, 1000), 4, 2, True, [4096, 2000]),
    ((5001, 1000), 4, 2, True, [4096, 820]),
    ((1000, 5001), 4, 2, True, [820, 4096]),
    ((1024, 512), 2, 2, False, [2048, 1024]),
    ((2048, 1024), 2, 2, False, [4096, 2048]),
    ((2049, 1000), 2, 2, True, [4096, 2000]),
    ((5001, 1), 4, 2, True, [4096, 2]),
])
def test_size_limits_choose_requested_then_two_then_reduced_copy(size, requested, actual, reduced, output):
    plan = batch_size_plan(*size, requested)
    assert plan['requested_factor'] == requested
    assert plan['factor'] == actual and plan['reduce_input'] == reduced
    assert plan['size_plan']['output'] == output
    assert max(output) <= 4096
    assert max(plan['size_plan']['input']) * actual <= 4096
    assert bool(plan['adjustment_reason']) == (reduced or actual != requested)


@pytest.mark.parametrize('size,factor', [((None, 40), 4), ((0, 40), 4), ((-1, 40), 4),
                                        ((40, 20), 1), ((40, 20), float('nan')), ((40, 20), 9)])
def test_invalid_dimensions_or_factors_do_not_become_automatic_fallback(size, factor):
    with pytest.raises(ValueError):
        batch_size_plan(*size, factor)


@pytest.fixture
def mixed_images(library, tmp_path):
    images = []
    for i, size in enumerate(((1024, 512), (1600, 900), (5001, 1000))):
        folder = tmp_path/f'input-{i}'; folder.mkdir()
        images.append(add_source(library, folder, size=size))
    return images


def test_mixed_batch_keeps_every_size_and_preserves_sources_and_image_rows(library, mixed_images):
    paths, store, _ = library
    before = [digest(i.file_path) for i in mixed_images]
    with store.connect(readonly=True) as conn:
        rows = list(conn.execute('SELECT * FROM images ORDER BY id'))
    repo = UpscaleWorkflow(store, paths)
    jobs, skipped = repo.enqueue([i.id for i in mixed_images], 4, 'http://127.0.0.1:8188')
    assert not skipped and len(jobs) == 3
    assert [j['factor'] for j in jobs] == [4, 2, 2]
    assert [j['reduce_input'] for j in jobs] == [False, False, True]
    assert all(j['requested_factor'] == 4 for j in jobs)
    reopened = UpscaleWorkflow(store, paths)
    batch = reopened.batch(jobs[0]['batch_id'])
    assert [j['image_id'] for j in batch['jobs']] == [i.id for i in mixed_images]
    assert batch['total'] == 3 and batch['pending'] == 3
    assert [digest(i.file_path) for i in mixed_images] == before
    with store.connect(readonly=True) as conn:
        assert list(conn.execute('SELECT * FROM images ORDER BY id')) == rows


@pytest.mark.parametrize('index,factor,reduced,input_size,output_size', [
    (0, 4, False, (1024, 512), (4096, 2048)),
    (1, 2, False, (1600, 900), (3200, 1800)),
    (2, 2, True, (2048, 410), (4096, 820)),
])
def test_real_preparation_uses_planned_scale_and_temporary_copy(library, mixed_images, index, factor, reduced, input_size, output_size):
    paths, store, _ = library
    image = mixed_images[index]
    original = digest(image.file_path)
    repo = UpscaleWorkflow(store, paths)
    jobs, _ = repo.enqueue([image.id], 4, 'http://127.0.0.1:8188')
    with patch.object(ComfyClient, 'request', return_value={'name': 'prepared.png'}):
        prepared = repo.create(image.id, jobs[0]['factor'], jobs[0]['url'],
                               reduce_input=jobs[0]['reduce_input'], queue_id=jobs[0]['id'])
    assert prepared['factor'] == factor and prepared['input_reduced'] == reduced
    assert prepared['requested_factor'] == 4
    assert (prepared['target_width'], prepared['target_height']) == output_size
    assert prepared['workflow']['3']['inputs']['resize_type.multiplier'] == factor
    with Image.open(prepared['input_path']) as inp:
        assert inp.size == input_size
    with Image.open(prepared['original_path']) as backup:
        assert backup.size == (image.width, image.height)
    assert digest(prepared['original_path']) == original == digest(image.file_path)
    assert store.get_image(image.id).file_size == image.file_size


def test_retry_preserves_requested_scale_actual_plan_and_batch_slot(library, mixed_images):
    repo = UpscaleWorkflow(library[1], library[0])
    jobs, _ = repo.enqueue([mixed_images[1].id], 4, 'http://127.0.0.1:8188')
    failed = jobs[0]; failed['state'] = 'failed'; repo.save(failed)
    retried = repo.retry(failed['id'])
    assert retried['factor'] == 2 and retried['requested_factor'] == 4
    assert retried['batch_id'] == failed['batch_id'] and retried['batch_index'] == failed['batch_index']
    assert retried['adjustment_reason'] == failed['adjustment_reason']
    assert repo.batch(retried['batch_id'])['total'] == 1


def test_preview_and_queue_report_the_same_per_image_plan(app, library, mixed_images):
    dialog = BatchUpscaleDialog(mixed_images)
    dialog.mode.setCurrentIndex(dialog.mode.findData(4))
    jobs, _ = UpscaleWorkflow(library[1], library[0]).enqueue([i.id for i in mixed_images], 4, 'http://127.0.0.1:8188')
    for row, job in enumerate(jobs):
        assert dialog.table.item(row, 3).text() == f"{job['factor']} 倍"
        assert dialog.table.item(row, 4).text() == ' × '.join(map(str, job['size_plan']['output']))
    assert '4 倍：1 张' in dialog.summary.text()
    assert '2 倍：2 张' in dialog.summary.text()
    assert '先缩小输入：1 张' in dialog.summary.text()
    dialog.close()


def test_menu_action_is_direct_and_preserves_selected_images(window, library):
    images = library[2][:3]
    window.grid_view.set_images(library[2], selected_image_ids=[i.id for i in images])
    menu = window._build_grid_context_menu(selected_images=images, context_image=images[1])
    action = next(a for a in menu.actions() if a.data() == 'batch_upscale')
    assert action.isEnabled() and '3 张' in action.text()
    with patch.object(window, '_build_grid_context_menu', return_value=SimpleNamespace(exec=lambda *_: action)), \
         patch.object(window.image_upscaler, 'open_selection') as opened:
        window._show_grid_context_menu(images[1], QPoint())
    assert [i.id for i in opened.call_args.args[0]] == [i.id for i in images]
    for empty, busy, maintenance in [(True, False, False), (False, True, False), (False, False, True)]:
        window.image_upscaler.working = busy
        window._database_maintenance_active = maintenance
        disabled = window._build_grid_context_menu(selected_images=[] if empty else images, context_image=None)
        assert not next(a for a in disabled.actions() if a.data() == 'batch_upscale').isEnabled()
    window.image_upscaler.working = False
    window._database_maintenance_active = False


def test_right_click_entry_enqueues_whole_snapshot_as_one_review_batch(window, library, mixed_images, app):
    c = window.image_upscaler
    c.timer.stop()
    window.store.set_setting('upscale.queue_paused', '1')
    # Opening the tasks window must not replace this snapshot with whatever
    # selection happens to remain in the gallery.
    window.grid_view.set_images(mixed_images, selected_image_ids=[mixed_images[0].id])
    def accept_plan():
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, BatchUpscaleDialog)
        assert [i.id for i in dialog.images] == [i.id for i in mixed_images]
        dialog.mode.setCurrentIndex(dialog.mode.findData(4))
        dialog.accept()
    QTimer.singleShot(0, accept_plan)
    c.open_selection(mixed_images)
    app.processEvents()
    batch = c.repo.batches()[0]
    assert [j['image_id'] for j in batch['jobs']] == [i.id for i in mixed_images]
    assert [j['factor'] for j in batch['jobs']] == [4, 2, 2]
    assert batch['ready'] == 0 and batch['pending'] == 3
    assert '已加入 3 张' in c.tasks.message.text() and '未入队 0 张' in c.tasks.message.text()
    assert c.tasks.batch_filter.currentData() == batch['id']


def test_cancelling_batch_dialog_does_not_create_tasks(window, library, app):
    c = window.image_upscaler
    c.timer.stop()
    QTimer.singleShot(0, lambda: QApplication.activeModalWidget().reject())
    c.open_selection(library[2][:2])
    assert not c.repo.all()


def test_real_viewport_right_click_opens_once_and_keeps_multi_selection(app, library):
    from eidory.ui.justified_image_grid import JustifiedImageGridView
    grid = JustifiedImageGridView()
    grid.resize(800, 500)
    images = library[2][:3]
    grid.set_images(images, selected_image_ids=[i.id for i in images])
    grid.show(); app.processEvents()
    captured = []
    grid.imageContextMenuRequested.connect(lambda image, point: captured.append(image.id))
    point = grid._rects[1].center()
    QTest.mouseClick(grid.viewport(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, point)
    app.processEvents()
    assert captured == [images[1].id]
    assert grid.selected_image_ids() == [i.id for i in images]
    native = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, point, grid.viewport().mapToGlobal(point))
    QApplication.sendEvent(grid.viewport(), native)
    assert captured == [images[1].id]
    grid.close()
