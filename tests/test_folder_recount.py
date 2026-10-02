from pathlib import Path
from unittest.mock import patch
import time

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox
from PySide6.QtTest import QTest

from test_filter_workflow import app, library, window
from eidory.core.folder_recount import recount_collection
from eidory.core.scanner import ImageScanner, ScanResult
from eidory.core.thumbnailer import Thumbnailer


def collection(store, images):
    parent = store.create_collection('数量核对')
    child = store.create_collection('子文件夹', parent_id=parent)
    store.assign_images_to_collection([im.id for im in images], child)
    store.assign_images_to_collection([images[0].id], parent)
    return parent, child


def test_recount_includes_children_deduplicates_and_restores_without_losing_data(library):
    _, store, images = library
    parent, child = collection(store, images)
    target = images[0]
    store.add_tags_to_images([target.id], ['保留标签'])
    store.update_note(target.id, '保留说明')
    path = Path(target.file_path)
    saved = path.read_bytes()
    path.unlink()
    result = recount_collection(store, parent)
    assert (result.before, result.available, result.missing, result.newly_missing) == (4, 3, 1, 1)
    assert store.collection_image_counts()[child] == 3
    assert store.get_image(target.id).is_missing
    assert store.get_image(target.id).note == '保留说明'
    with store.connect(readonly=True) as conn:
        assert conn.execute('select count(*) from image_tags where image_id=?', (target.id,)).fetchone()[0] == 1
        assert conn.execute('select count(*) from image_collections where image_id=?', (target.id,)).fetchone()[0] == 2
    path.write_bytes(saved)
    result = recount_collection(store, parent)
    assert (result.available, result.missing, result.recovered) == (4, 0, 1)
    assert not store.get_image(target.id).is_missing


def test_recount_does_not_reimport_removed_or_unlinked_files(library):
    _, store, images = library
    parent, _ = collection(store, images[:2])
    store.remove_images_from_library([images[0].id])
    assert Path(images[0].file_path).exists()
    result = recount_collection(store, parent)
    assert result.available == 1
    assert store.get_image(images[0].id) is None
    assert len(store.collection_file_records(parent)) == 1


def test_unavailable_disk_and_permission_errors_do_not_mark_missing(library):
    _, store, images = library
    parent, _ = collection(store, images)
    root = Path(images[0].file_path).parent
    moved = root.with_name('offline')
    root.rename(moved)
    result = recount_collection(store, parent)
    assert (result.available, result.missing, result.unchecked) == (4, 0, 4)
    assert len(result.errors) == 1
    moved.rename(root)
    import os
    real_stat = os.stat
    def stat_path(path, *args, **kwargs):
        if str(path) == images[0].file_path:
            raise PermissionError(13, 'Permission denied')
        return real_stat(path, *args, **kwargs)
    with patch('eidory.core.folder_recount.os.stat', side_effect=stat_path):
        result = recount_collection(store, parent)
    assert (result.available, result.missing, result.unchecked) == (4, 0, 1)


def test_recount_does_not_overwrite_a_concurrent_relink(library):
    _, store, images = library
    parent, _ = collection(store, images)
    path = Path(images[0].file_path)
    path.unlink()
    original_apply = store.apply_file_availability
    def relink(updates):
        with store.connect() as conn:
            conn.execute('update images set file_path=? where id=?', (str(path.with_name('replacement.png')), images[0].id))
        original_apply(updates)
    with patch.object(store, 'apply_file_availability', side_effect=relink):
        recount_collection(store, parent)
    assert not store.get_image(images[0].id).is_missing


def test_scan_survives_file_deleted_after_directory_listing(library):
    paths, store, images = library
    scanner = ImageScanner(store, Thumbnailer(paths.thumbnail_dir))
    root = str(Path(images[0].file_path).parent)
    listing = scanner._iter_image_files(root)
    Path(images[0].file_path).unlink()
    with patch.object(scanner, '_iter_image_files', return_value=listing):
        result = scanner.scan_folder(root)
    assert (result.scanned_files, result.missing_marked) == (3, 1)
    assert store.get_image(images[0].id).is_missing


def test_scan_does_not_remove_records_on_unreadable_subdirectory(library):
    paths, store, images = library
    scanner = ImageScanner(store, Thumbnailer(paths.thumbnail_dir))
    def failed_walk(root, **kwargs):
        kwargs['onerror'](PermissionError('unreadable subdirectory'))
        return []
    with patch('eidory.core.scanner.os.walk', side_effect=failed_walk):
        with pytest.raises(PermissionError):
            scanner.scan_folder(str(Path(images[0].file_path).parent))
    assert len(store.images_by_ids([im.id for im in images])) == 4


def test_refresh_menu_updates_counts_and_wall_without_duplicate_actions(window, library):
    _, store, images = library
    parent, _ = collection(store, images)
    window._refresh_collections(select_collection_id=parent)
    window._on_collection_selection_changed()
    assert window._collection_image_counts_for_status()[parent] == 4
    item = window.collection_tree.currentItem()
    menu, actions = window._build_collection_context_menu(item)
    assert sum(a.text() == '刷新文件夹数量' for a in menu.actions()) == 1
    assert actions['recount'].isEnabled()
    assert not window._build_collection_context_menu(None)[1]['recount'].isEnabled()
    Path(images[0].file_path).unlink()
    # Exercise background-result delivery and the real dialog contents.
    dialogs = []
    with patch.object(QMessageBox, 'exec', lambda dialog: dialogs.append((dialog.text(), dialog.detailedText())) or 0):
        window._recount_collection(parent)
        assert not window._build_collection_context_menu(item)[1]['recount'].isEnabled()
        window._poll_events()
    assert not window._folder_recount_running
    assert '4 → 3' in dialogs[0][0]
    assert str(Path(images[1].file_path).parent) in dialogs[0][1]
    assert window.collection_tree.currentItem().text(1) == '3'
    assert window._collection_image_counts_for_status()[parent] == 3
    assert images[0].id not in [im.id for im in window.grid_view.images()]
    assert window.collection_tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == parent


def test_recount_failure_allows_retry(window, library):
    _, store, images = library
    parent, _ = collection(store, images)
    with patch('eidory.ui.main_window.recount_collection', side_effect=RuntimeError('read failure')):
        window._recount_collection(parent)
    with patch.object(QMessageBox, 'warning') as warning:
        window._poll_events()
    assert not window._folder_recount_running
    assert warning.called


@pytest.mark.parametrize('removed,preserve', [(0, True), (1, False)])
def test_watch_scan_refreshes_stale_results_when_files_were_removed(window, removed, preserve):
    result = ScanResult(1, 3, 0, 0, 3, removed, 0, ())
    with patch.object(window, '_refresh_after_scan_database_change') as refresh:
        window._handle_watch_scan_done((['test'], [result], []))
    refresh.assert_called_once_with(preserve_current_view=preserve)


def test_real_file_watcher_updates_sidebar_and_wall_without_manual_refresh(window, library, app):
    _, store, images = library
    parent, _ = collection(store, images)
    window._refresh_collections(select_collection_id=parent)
    window._on_collection_selection_changed()
    window._set_file_watch_enabled(True)

    def wait_until(predicate):
        deadline = time.monotonic() + 6
        while not predicate() and time.monotonic() < deadline:
            window._poll_events()
            QTest.qWait(30)
        assert predicate()

    wait_until(lambda: window._file_watch_configured_count > 0)
    # Real filesystem event, through the Qt worker and its debounce timer.
    Path(images[0].file_path).unlink()
    Path(images[1].file_path).unlink()
    wait_until(lambda: window.collection_tree.currentItem().text(1) == '2')
    assert len(window.grid_view.images()) == 2
    assert window._collection_image_counts_for_status()[parent] == 2
    assert not window._pending_watch_scan_roots


def test_wall_delete_and_undo_update_sidebar_immediately(window, library, tmp_path):
    _, store, images = library
    parent, _ = collection(store, images)
    window._refresh_collections(select_collection_id=parent)
    window._on_collection_selection_changed()
    window._collection_image_counts_for_status()  # Prime the former stale cache.
    def trash(path):
        path.rename(tmp_path / ('trashed-' + path.name))
        return True, ''
    with patch.object(window, '_move_source_file_to_trash', side_effect=trash):
        window._delete_source_files_with_undo(images[:2])
    assert window.collection_tree.currentItem().text(1) == '2'
    assert window._collection_image_counts_for_status()[parent] == 2
    assert len(window.grid_view.images()) == 2
    window._undo_last_library_removal()
    assert window.collection_tree.currentItem().text(1) == '4'
    assert window._collection_image_counts_for_status()[parent] == 4
    assert len(window.grid_view.images()) == 4
