"""Fault regressions: temporary data only, no installed services or user files."""
from copy import deepcopy
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
from pathlib import Path
import shutil
import sqlite3
import threading
import time
from unittest.mock import patch

import pytest
import requests
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from PySide6.QtTest import QTest

from test_filter_workflow import app, library, window
from test_project_state import project
from test_comfy_reference import generation, graph
from test_creative_flow import make_project
from test_board_inspector import board_project
from eidory.core.comfy_reference import ComfyClient, prepare_template
from eidory.core.creative_workspace import CreativeWorkspaceStore
from eidory.core.database_restore import prepare_database, install_database
from eidory.core.metadata_store import MetadataStore
from eidory.core.project_package import export_project_package, import_project_package
from eidory.core.project_state import ProjectStateStore
from eidory.core.scanner import ImageScanner
from eidory.core.thumbnailer import Thumbnailer
from eidory.core.cancellable_http import post_json, RequestCancelled
from eidory.ui.justified_image_grid import JustifiedImageGridView


def annotated(store, pid, node, image):
    other = store.create_creative_node(project_id=pid, parent_id=store.creative_root_node_id(pid), title='其他用途')
    store.add_images_to_creative_node(other, [image])
    store.set_image_tags(image, ['人工标签'])
    decisions = ProjectStateStore(store)
    decisions.set_decision(node, image, 'primary', '造型')
    decisions.set_decision(other, image, 'unused', '光线')
    CreativeWorkspaceStore(store).save_reference_note(node, image, '保留原始用途说明')
    store.mark_ai_vision_processing(image_id=image, provider_name='fixture', model_name='test', prompt_version='scene-v1')
    return ['image_tags', 'creative_node_images', 'creative_reference_decisions', 'creative_reference_notes', 'ai_vision_tags']


def related(store, tables):
    with store.connect(readonly=True) as conn:
        return {table: [dict(row) for row in conn.execute(f'SELECT * FROM {table} ORDER BY rowid')] for table in tables}


@pytest.mark.parametrize('whole_root', [False, True])
def test_scan_missing_then_return_retains_all_annotations(project, tmp_path, whole_root):
    store, pid, node, image = project
    tables = annotated(store, pid, node, image)
    store.create_tag('尚未使用的标签')
    before = related(store, tables)
    path = Path(store.get_image(image).file_path)
    source = path.parent if whole_root else path
    moved = tmp_path / ('disconnected' if whole_root else 'moved.png')
    scanner = ImageScanner(store, Thumbnailer(tmp_path / 'thumbs'))
    source.rename(moved)
    assert scanner.scan_folder(str(path.parent)).missing_marked == 1
    assert not store.list_images(limit=None)
    assert store.get_image(image).is_missing
    assert related(store, tables) == before
    moved.rename(source)
    scanner.scan_folder(str(path.parent))
    assert store.list_images(limit=None)[0].id == image
    assert not store.get_image(image).is_missing
    assert related(store, tables) == before
    assert any(tag.tag_name == '尚未使用的标签' for tag in store.list_tags())


def test_delete_undo_restores_every_annotation_and_rejects_reused_id(project):
    store, pid, node, image = project
    tables = annotated(store, pid, node, image)
    before = related(store, tables)
    snapshot = store.snapshot_images_for_restore([image])
    store.remove_images_from_library([image])
    assert store.restore_images_snapshot(snapshot) == 1
    assert related(store, tables) == before
    with store.connect() as conn:
        conn.execute('UPDATE images SET file_size=file_size+1 WHERE id=?', (image,))
    with pytest.raises(ValueError, match='占用'):
        store.restore_images_snapshot(snapshot)


def test_old_preview_never_selects_wall_hidden_image(window, library):
    w = window
    images = library[2]
    w._show_gallery_view()
    w.grid_view.set_images(images[1:], selected_image_ids=[])
    w.selected_image = None
    w._sync_preview_selection(images[0])
    assert w.grid_view.selected_image_ids() == []
    assert w._selected_grid_images() == []
    assert w.selected_image is None
    with patch.object(w, '_ask_delete_or_remove_mode') as ask:
        w._delete_selected_source_files()
        ask.assert_not_called()


def test_preview_context_and_planning_do_not_select_hidden_wall(window, library):
    w = window
    make_project(w)
    w._show_gallery_view()
    w.grid_view.set_images(library[2], selected_image_ids=[])
    old_context = w._preview_selection_context()
    make_project(w)
    w._show_gallery_view()
    w.grid_view.set_images(library[2], selected_image_ids=[])
    w._sync_preview_selection(library[2][0], old_context)
    assert not w._selected_grid_images()
    w.grid_view.select_image_ids([library[2][0].id])
    w.creative_workspace.show_planning()
    assert not w._selected_grid_images()


def test_undo_conflicting_source_keeps_backup_and_existing_file(window, library, tmp_path):
    w = window
    image = library[2][0]
    source = Path(image.file_path)
    original = source.read_bytes()
    def trash(path):
        path.rename(tmp_path / 'trashed.png')
        return True, ''
    with patch.object(w, '_move_source_file_to_trash', side_effect=trash):
        w._delete_source_files_with_undo([image])
    backup = Path(w._last_removal_undo['source_backups'][0]['backup_path'])
    source.write_bytes(b'new occupant')
    with patch.object(QMessageBox, 'warning'):
        w._undo_last_library_removal()
    assert source.read_bytes() == b'new occupant'
    assert backup.read_bytes() == original
    assert w.store.get_image(image.id) is None
    source.unlink()
    w._undo_last_library_removal()
    assert source.read_bytes() == original
    assert w.store.get_image(image.id)
    assert w._last_removal_undo is None


def test_restore_reloads_projects_and_discards_old_undo(window, tmp_path):
    w = window
    pid, node, _ = make_project(w)
    original = w.store.get_creative_node(node.id).note
    backup = tmp_path / 'old.sqlite3'
    with w.store.connect(readonly=True) as src, sqlite3.connect(backup) as dst:
        src.backup(dst)
    row = w.creative_flow.targets()[0]
    w.creative_flow.change_target(row['id'], query='恢复后不能再次写入的旧命令')
    assert w.edit_history.past
    with patch.object(QFileDialog, 'getOpenFileName', return_value=(str(backup), '')), patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
        w._restore_database_from_file()
    assert w.current_creative_project_id is None
    assert not w.edit_history._histories.get((pid, None), ([], []))[0]
    w._load_creative_project(pid, select_node_id=node.id)
    assert w.store.get_creative_node(node.id).note == original
    assert not w.edit_history.past


def test_real_watcher_restores_disconnected_roots_sharing_parent(window, library, tmp_path):
    w = window
    store = w.store
    roots = []
    for n in range(2):
        root = tmp_path / f'disk-{n}'
        root.mkdir()
        shutil.copy2(library[2][0].file_path, root / 'one.png')
        result = w.scanner.scan_folder(str(root))
        col = store.create_collection(f'外部盘 {n}')
        store.assign_images_to_collection(result.image_ids, col)
        store.add_tags_to_images(result.image_ids, ['失联仍保留'])
        roots.append((root, result.image_ids[0]))
    w._set_file_watch_enabled(True)
    def wait(predicate):
        until = time.monotonic() + 7
        while not predicate() and time.monotonic() < until:
            w._poll_events()
            QTest.qWait(30)
        assert predicate(), (w._watch_path_roots, getattr(w, '_watch_path_extra_roots', {}), w._pending_watch_scan_roots,
                             [(str(root), store.get_image(i).is_missing) for root, i in roots], w.statusBar().currentMessage())
    wait(lambda: w._file_watch_configured_count > 0)
    for root, _ in roots:
        root.rename(root.with_name(root.name + '-disconnected'))
    wait(lambda: all(store.get_image(i).is_missing for _, i in roots))
    for root, _ in roots:
        root.with_name(root.name + '-disconnected').rename(root)
    wait(lambda: all(not store.get_image(i).is_missing for _, i in roots))
    assert all(store.get_image_tags(i) == ['失联仍保留'] for _, i in roots)


def test_project_undo_only_affects_current_project(window):
    w = window
    pid, node, _ = make_project(w)
    row = w.creative_flow.targets()[0]
    original = row['query']
    w.creative_flow.change_target(row['id'], query='项目 A 独有查询')
    a = w.creative_flow.targets()
    other, other_node, _ = make_project(w)
    other_before = w.creative_flow.targets()
    if QApplication.focusWidget():
        QApplication.focusWidget().clearFocus()
    w.edit_history.undo()
    assert w.current_creative_project_id == other
    assert w.creative_flow.targets() == other_before
    w._load_creative_project(pid, select_node_id=node.id)
    assert w.creative_flow.targets() == a
    if QApplication.focusWidget():
        QApplication.focusWidget().clearFocus()
    w.edit_history.undo()
    assert w.creative_flow.targets()[0]['query'] == original
    w.edit_history.redo()
    assert w.creative_flow.targets()[0]['query'] == '项目 A 独有查询'


def test_board_history_from_other_node_does_not_block_project_undo(window, library):
    w = window
    pid, a, b, _ = board_project(w, library[2])
    board = w.project_board_view
    origin = w._current_board_node_id
    before = deepcopy(board.layout_payload())
    board._image_items[library[2][0].id].setPos(QPointF(940, 760))
    board.flush_pending_layout_change()
    w._save_current_board_layout_if_needed(force=True)
    w._load_creative_project(pid, select_node_id=b.id, show_board=True)
    if QApplication.focusWidget():
        QApplication.focusWidget().clearFocus()
    w.edit_history.undo()
    assert json.loads(w.store.get_creative_node_board_layout(origin)) == before
    assert w.current_creative_node_id == b.id


def test_restore_validates_private_copy_and_rolls_back_failed_switch(project, tmp_path):
    store, pid, _, _ = project
    bad = tmp_path / 'bad.sqlite3'
    with sqlite3.connect(bad) as conn:
        for name in ('folders', 'images', 'app_settings'):
            conn.execute(f'CREATE TABLE {name}(id INTEGER)')
    with pytest.raises(ValueError, match='不兼容'):
        prepare_database(bad, tmp_path / 'invalid-staged.sqlite3')
    assert store.get_creative_project(pid).title == '研发室'
    backup = tmp_path / 'backup.sqlite3'
    with store.connect(readonly=True) as src, sqlite3.connect(backup) as dst:
        src.backup(dst)
    staged = tmp_path / 'staged.sqlite3'
    original = backup.read_bytes()
    prepare_database(backup, staged)
    assert backup.read_bytes() == original
    with patch.object(store, 'initialize', side_effect=RuntimeError('injected switch failure')):
        with pytest.raises(RuntimeError):
            install_database(store, staged, backup)
    assert store.get_creative_project(pid).title == '研发室'
    assert backup.read_bytes() == original


def test_grid_queue_stays_bounded_and_drops_old_responses(app, library):
    grid = JustifiedImageGridView()
    submitted = []
    with patch.object(grid._thread_pool, 'start', side_effect=lambda task, *args: submitted.append(task)):
        for generation in range(50):
            grid.set_images([replace(library[2][0], id=generation + 100)])
            for index in range(48):
                grid._schedule_pixmap_load((f'/tmp/{generation}-{index}.png', 1, 1, 256))
        assert len(submitted) == 8
        assert len(grid._image_loader.active) == 8
        assert len(grid._image_loader.queue) <= 48
        old = grid._load_generation - 1
        pix = QPixmap(8, 8)
        grid._handle_current_pixmap_loaded((old, ('old', 1, 1, 256)), pix.toImage(), '')
        assert not grid._loaded_pixmap_results
        for task in list(submitted):
            task.run()
        assert len(submitted) == 16
        assert all(token[0] == grid._load_generation for token in grid._image_loader.active)
        grid._image_loader.stop()


def test_board_accounts_for_transformed_pixels_and_keeps_layout(window, library):
    w = window
    board_project(w, library[2])
    board = w.project_board_view
    item = board._image_items[library[2][0].id]
    before = deepcopy(board.layout_payload())
    pix = QPixmap(4096, 4096)
    pix.fill()
    item.replace_source_pixmap(pix, allow_smaller=True)
    for gray, flipped in [(True, False), (True, True), (False, True), (False, False)]:
        item._grayscale, item._flipped = gray, flipped
        item._refresh_display_pixmap()
        assert len(item._display_variant_cache) <= 1
        assert board._item_pixel_cost(item) >= 4096 * 4096 * 4
    used = board.trim_image_memory(16 * 1024 * 1024)
    assert used <= 16 * 1024 * 1024
    assert board.layout_payload() == before


def test_comfy_offline_releases_wait_and_resumes_same_identity(generation, graph):
    repo, _, node = generation
    job = repo.create(node, 'robot', prepare_template(graph), 'http://localhost:8188')
    job['state'] = 'running'
    repo.save(job)
    with patch.object(ComfyClient, 'request', side_effect=requests.ConnectionError) as request:
        offline = repo.reconcile(job['id'])
        assert offline['state'] == 'offline'
        for _ in range(100):
            repo.reconcile(job['id'])
        assert request.call_count == 1
        stopped = repo.cancel(job['id'])
        assert stopped['state'] == 'paused'
        assert request.call_count == 1
        assert not repo.active()
    with patch.object(ComfyClient, 'request', side_effect=[{}, {'queue_running': [[1, job['prompt_id']]], 'queue_pending': []}]) as request:
        resumed = repo.resume(job['id'])
        assert resumed['state'] == 'running'
        assert resumed['prompt_id'] == job['prompt_id']
        assert all(call.args[0] == 'GET' for call in request.call_args_list)


def test_cancel_interrupts_real_slow_http_and_reaps_network_process():
    received, release = threading.Event(), threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            received.set()
            release.wait(10)
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    service = threading.Thread(target=server.serve_forever, daemon=True)
    service.start()
    cancel, errors = threading.Event(), []
    children = {p.pid for p in multiprocessing.active_children()}
    def request():
        try:
            post_json(f'http://127.0.0.1:{server.server_port}/slow', json={}, headers={}, timeout=90, cancel=cancel)
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=request)
    try:
        thread.start()
        assert received.wait(6)
        start = time.monotonic()
        cancel.set()
        thread.join(2)
        assert not thread.is_alive()
        assert time.monotonic() - start < 2
        assert isinstance(errors[0], RequestCancelled)
        assert {p.pid for p in multiprocessing.active_children()} <= children
    finally:
        cancel.set()
        release.set()
        thread.join(3)
        server.shutdown()
        server.server_close()


def test_import_copy_does_not_block_other_database_writes(project, tmp_path):
    store, pid, _, _ = project
    package = export_project_package(store, pid, tmp_path)
    copying, release, written = threading.Event(), threading.Event(), threading.Event()
    errors = []
    original = shutil.copy2
    def slow_copy(source, destination):
        copying.set()
        assert release.wait(5)
        return original(source, destination)
    def import_it():
        try:
            import_project_package(store, package, tmp_path / 'owned')
        except Exception as exc:
            errors.append(exc)
    def save_it():
        store.set_setting('parallel-write', 'saved')
        written.set()
    with patch('eidory.core.project_package.shutil.copy2', side_effect=slow_copy):
        importing = threading.Thread(target=import_it)
        importing.start()
        assert copying.wait(3)
        writer = threading.Thread(target=save_it)
        writer.start()
        try:
            assert written.wait(1), 'media copying still holds the database writer lock'
        finally:
            release.set()
            importing.join(5)
            writer.join(5)
    assert not errors
    assert store.get_setting('parallel-write') == 'saved'
