from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QMessageBox, QPushButton

from test_filter_workflow import app, library, window
from eidory.core.duplicate_detection import DuplicateGroup, DuplicateMember
from eidory.ui.duplicate_compare_dialog import DuplicateCompareDialog
from eidory.ui.main_window import DuplicateResultsDialog


def group(images, kind="exact"):
    return DuplicateGroup(kind, "文件内容相同" if kind == "exact" else "近似图片候选", tuple(
        DuplicateMember(image, f"参考图库 / 文件夹 {image.id}", "hash", image.id)
        for image in images
    ))


@pytest.fixture
def compare(app, library):
    images = library[2]
    dialog = DuplicateCompareDialog([group(images[:2]), group(images[2:], "near")])
    dialog.show()
    dialog.activateWindow()
    app.processEvents()
    yield dialog
    dialog.loader.stop()
    assert dialog.loader.pool.waitForDone(3000)
    dialog.close()
    app.processEvents()


def key(app, dialog, key, modifiers=Qt.KeyboardModifier.NoModifier):
    dialog.activateWindow()
    dialog.setFocus()
    app.processEvents()
    QTest.keyClick(dialog, key, modifiers)
    app.processEvents()


def test_metadata_is_next_to_each_image_and_plain_text(compare, library):
    for image in library[2][:2]:
        card = compare.cards[image.id]
        labels = {label.objectName(): label for label in card.findChildren(QLabel)}
        assert labels["duplicate完整路径"].text() == image.file_path
        assert "×" in labels["duplicate尺寸 / 大小"].text()
        assert str(image.id) in labels["duplicate所属文件夹"].text()
        assert labels["duplicate完整路径"].textFormat() == Qt.TextFormat.PlainText


def test_comparison_shows_complete_library_folder_hierarchy(window, library):
    images = library[2][:2]
    chains = [
        ["ML-05 复杂场景", "植物", "其他植物"],
        ["ML-04 简单小景", "自然场景", "其他自然景观"],
    ]
    members = []
    for image, chain in zip(images, chains):
        collection_id = window.store.ensure_collection_path(chain)
        window.store.assign_images_to_collection([image.id], collection_id)
        label = window._folder_label_for_image(image)
        assert label == " / ".join(chain)
        members.append(DuplicateMember(image, label, "hash", None))
    window._handle_duplicates_done([DuplicateGroup("exact", "same hash", tuple(members))])
    results = window._duplicate_results_dialog
    results.open_comparison()
    dialog = results.compare_dialog
    for image, chain in zip(images, chains):
        label = dialog.cards[image.id].findChild(QLabel, "duplicate所属文件夹")
        assert label.text() == " / ".join(chain)
        assert label.wordWrap()
        assert label.toolTip() == label.text()
    dialog.loader.stop()
    assert dialog.loader.pool.waitForDone(3000)
    results.close()


def test_detection_populates_complete_folder_labels_with_bulk_lookup(window, library):
    images = library[2][:2]
    expected = {}
    for image, root in zip(images, ("ML-05 复杂场景", "ML-04 简单小景")):
        chain = [root, "植物", "其他植物"]
        window.store.assign_images_to_collection([image.id], window.store.ensure_collection_path(chain))
        expected[image.id] = " / ".join(chain)
    with patch.object(window, "_folder_label_for_image", side_effect=AssertionError("must use bulk labels")):
        window._detect_duplicates()
    groups = None
    while not window.events.empty():
        kind, payload = window.events.get_nowait()
        if kind == "duplicates_done":
            groups = payload
    assert groups is not None
    actual = {m.image.id: m.folder_label for g in groups for m in g.members}
    assert {iid: actual[iid] for iid in expected} == expected


def test_default_window_keeps_information_and_actions_visible(app, compare):
    app.processEvents()
    viewport = compare.scroll.viewport()
    for card in compare.cards.values():
        for name in ("duplicate完整路径", "deleteDuplicate"):
            widget = card.findChild(QLabel if name.startswith("duplicate") else QPushButton, name)
            assert viewport.rect().contains(widget.mapTo(viewport, widget.rect().topLeft()))
            assert viewport.rect().contains(widget.mapTo(viewport, widget.rect().bottomRight()))


@pytest.mark.parametrize("dimensions", [(12285, 5177), (5177, 12285)])
def test_very_large_images_fit_completely_and_zoom_smoothly(compare, app, dimensions):
    view = next(iter(compare.views.values()))
    width, height = dimensions
    pixmap = QPixmap(2048 if width > height else 863, 863 if width > height else 2048)
    pixmap.fill(Qt.GlobalColor.gray)
    view.set_pixmap(pixmap, original_width=width, original_height=height,
                    fit_to_window=True, zoom_factor=1.)
    app.processEvents()
    fitted = view.zoom_factor
    assert fitted < .1
    assert not view.can_pan()
    visible = view._visible_source_rect()
    assert visible.width() >= pixmap.width() - 2
    assert visible.height() >= pixmap.height() - 2
    view.zoom_by(120)
    assert view.zoom_factor == pytest.approx(fitted * 1.15)


def test_keyboard_requires_explicit_selection_and_clears_on_navigation(app, compare, library):
    deleted = []
    compare.deleteRequested.connect(deleted.append)
    key(app, compare, Qt.Key.Key_Delete)
    assert deleted == []
    compare.select_image(library[2][1].id)
    key(app, compare, Qt.Key.Key_Right)
    assert compare.group_index == 1
    assert compare.selected_image_id is None
    key(app, compare, Qt.Key.Key_Delete)
    assert deleted == []
    key(app, compare, Qt.Key.Key_Left)
    assert compare.group_index == 0


@pytest.mark.parametrize("delete_key", [Qt.Key.Key_Delete, Qt.Key.Key_Backspace])
def test_delete_and_command_delete_only_emit_selected_id(app, compare, library, delete_key):
    removed, deleted = [], []
    compare.removeRequested.connect(removed.append)
    compare.deleteRequested.connect(deleted.append)
    target = library[2][1].id
    compare.select_image(target)
    key(app, compare, delete_key, Qt.KeyboardModifier.ControlModifier)
    assert removed == [target]
    assert deleted == []
    assert compare.selected_image_id is None
    compare.select_image(target)
    key(app, compare, delete_key)
    assert deleted == [target]
    assert compare.selected_image_id is None


def test_click_preview_and_per_image_buttons_target_their_own_card(app, compare, library):
    images = library[2]
    QTest.mouseClick(compare.views[images[1].id].viewport(), Qt.MouseButton.LeftButton)
    assert compare.selected_image_id == images[1].id
    removed = []
    compare.removeRequested.connect(removed.append)
    compare.cards[images[0].id].findChild(QPushButton, "removeDuplicate").click()
    assert removed == [images[0].id]
    assert compare.selected_image_id is None


def test_accessibility_selection_updates_action_target(compare, library):
    first, second = [image.id for image in library[2][:2]]
    compare.select_buttons[first].setChecked(True)
    assert compare.selected_image_id == first
    compare.select_buttons[second].setChecked(True)
    assert compare.selected_image_id == second
    assert not compare.select_buttons[first].isChecked()
    # The selected control stays consistent even if an AX action toggles it off.
    compare.select_buttons[second].setChecked(False)
    assert compare.selected_image_id == second
    assert compare.select_buttons[second].isChecked()


def test_auto_repeat_does_not_delete(compare, library, app):
    deleted = []
    compare.deleteRequested.connect(deleted.append)
    compare.select_image(library[2][0].id)
    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier, "", True)
    app.sendEvent(compare, event)
    assert not deleted


def test_group_with_three_images_and_stale_preview_is_ignored(compare, library):
    images = library[2]
    old_generation = compare._generation
    compare.set_groups([group(images[:3])])
    assert set(compare.cards) == {image.id for image in images[:2]}
    compare.change_candidate(1)
    assert set(compare.cards) == {images[0].id, images[2].id}
    with patch.object(compare.views[images[0].id], "set_pixmap") as set_pixmap:
        compare._loaded((old_generation, images[0].id), QImage(10, 10, QImage.Format.Format_RGB32), "")
    set_pixmap.assert_not_called()


def test_2799_exact_copies_create_only_two_cards_and_two_load_requests(compare, library):
    images = [replace(library[2][0], id=index + 100) for index in range(2799)]
    with patch.object(compare.loader, "request") as request:
        compare.set_groups([group(images)])
    assert set(compare.cards) == {images[0].id, images[1].id}
    assert len(compare.views) == 2
    assert request.call_count == 2
    assert "1 / 2798" in compare.pair_label.text()
    compare.select_image(images[1].id)
    compare.change_candidate(1)
    assert set(compare.cards) == {images[0].id, images[2].id}
    assert compare.selected_image_id is None
    assert compare.previous_candidate_button.isEnabled()


def test_legacy_near_component_cannot_expose_thousands_of_deletion_targets(compare, library):
    images = [replace(library[2][0], id=index + 100) for index in range(2799)]
    with patch.object(compare.loader, "request") as request:
        compare.set_groups([group(images, "near")])
    assert not compare.cards
    assert not compare.views
    request.assert_not_called()
    assert "重新检测" in compare.reason_label.text()
    removed = []
    compare.removeRequested.connect(removed.append)
    compare.request_action("remove", images[0].id)
    assert not removed


def test_hidden_copy_cannot_be_deleted_and_candidate_keys_clear_selection(app, compare, library):
    images = library[2]
    compare.set_groups([group(images)])
    deleted = []
    compare.deleteRequested.connect(deleted.append)
    compare.request_action("delete", images[3].id)
    assert not deleted
    compare.select_image(images[1].id)
    key(app, compare, Qt.Key.Key_Down)
    assert set(compare.cards) == {images[0].id, images[2].id}
    assert compare.selected_image_id is None
    key(app, compare, Qt.Key.Key_Delete)
    assert not deleted
    key(app, compare, Qt.Key.Key_Up)
    assert set(compare.cards) == {images[0].id, images[1].id}


def test_resolved_pair_does_not_jump_to_286th_pair_sharing_survivor(compare, library):
    base = library[2][0]
    image = lambda iid: replace(base, id=iid)
    groups = [group([image(1), image(2)], "near")]
    groups.extend(group([image(100 + 2*i), image(101 + 2*i)], "near") for i in range(416))
    groups[286] = group([image(1), image(10000)], "near")
    compare.set_groups(groups)
    assert compare.group_index == 0
    compare.set_groups(groups[1:])
    assert len(compare.groups) == 416
    assert compare.group_index == 0
    assert set(compare.cards) == {100, 101}


def test_result_update_keeps_exact_current_pair_instead_of_first_shared_image(compare, library):
    base = library[2][0]
    image = lambda iid: replace(base, id=iid)
    groups = [group([image(1), image(2)], "near"),
              group([image(1), image(3)], "near"),
              group([image(4), image(5)], "near")]
    compare.set_groups(groups)
    compare.change_group(1)
    compare.set_groups(groups)
    assert compare.group_index == 1
    assert set(compare.cards) == {1, 3}


def test_each_of_300_next_clicks_advances_exactly_one_group(compare, library):
    base = library[2][0]
    groups = [group([replace(base, id=1000 + 2*i), replace(base, id=1001 + 2*i)], "near")
              for i in range(305)]
    with patch.object(compare.loader, "request"):
        compare.set_groups(groups)
        assert not compare.next_button.autoRepeat()
        for expected in range(1, 301):
            compare.next_button.click()
            assert compare.group_index == expected
            assert set(compare.cards) == {1000 + 2*expected, 1001 + 2*expected}
        compare.previous_button.click()
        assert compare.group_index == 299


def test_refresh_cannot_reorder_queue_and_undo_keeps_visible_pair(compare, library):
    base = library[2][0]
    groups = [group([replace(base, id=1000 + 2*i), replace(base, id=1001 + 2*i)], "near")
              for i in range(5)]
    compare.set_groups(groups)
    compare.change_group(2)
    visible = set(compare.cards)
    compare.set_groups(list(reversed(groups)))
    assert compare.group_index == 2
    assert set(compare.cards) == visible
    compare.set_groups(groups[1:])
    assert compare.group_index == 1
    assert set(compare.cards) == visible
    compare.set_groups(groups)
    assert compare.group_index == 2
    assert set(compare.cards) == visible
    assert "当前图片组保持不变" in compare.order_label.text()


def test_remove_and_next_follow_queue_instead_of_a_distant_shared_image(window, library):
    images = library[2]
    groups = [group(images[:2], "near"), group(images[2:], "near"),
              group([images[0], images[2]], "near")]
    window._handle_duplicates_done(groups)
    results = window._duplicate_results_dialog
    results.open_comparison()
    dialog = results.compare_dialog
    dialog.request_action("remove", images[1].id)
    assert dialog.group_index == 0
    assert set(dialog.cards) == {images[2].id, images[3].id}
    assert "原顺序中的下一组" in dialog.order_label.text()
    assert "已解决 1 组" in dialog.order_label.text()
    dialog.next_button.click()
    assert dialog.group_index == 1
    assert set(dialog.cards) == {images[0].id, images[2].id}
    dialog.undoRequested.emit()
    assert dialog.group_index == 2
    assert set(dialog.cards) == {images[0].id, images[2].id}
    assert "已解决 0 组" in dialog.order_label.text()
    dialog.loader.stop()
    assert dialog.loader.pool.waitForDone(3000)
    results.close()


def test_unreadable_preview_keeps_metadata_and_actions(compare, library):
    target = library[2][0]
    compare._loaded((compare._generation, target.id), QImage(), "")
    assert compare.cards[target.id].findChild(QLabel, "duplicate完整路径").text() == target.file_path
    assert compare.cards[target.id].findChild(QPushButton, "deleteDuplicate").isEnabled()


def test_removing_prior_group_preserves_current_group_then_empty_state(compare, library):
    compare.change_group(1)
    compare.set_groups([group(library[2][2:], "near")])
    assert compare.group_index == 0
    assert set(compare.cards) == {image.id for image in library[2][2:]}
    compare.set_groups([])
    assert not compare.cards
    assert not compare.previous_button.isEnabled()
    assert not compare.next_button.isEnabled()
    assert "已处理完" in compare.group_label.text()


def test_results_entry_starts_from_selected_child(app, library):
    images = library[2]
    results = DuplicateResultsDialog([group(images[:2]), group(images[2:], "near")])
    results.show()
    app.processEvents()
    results.tree.setCurrentItem(results.tree.topLevelItem(1).child(0))
    results.compare_button.click()
    dialog = results.compare_dialog
    assert dialog.group_index == 1
    dialog.loader.stop()
    assert dialog.loader.pool.waitForDone(3000)
    results.close()
    assert results.compare_dialog is None
    app.processEvents()


def test_detection_opens_pair_without_replacing_picture_wall(window, library):
    images = library[2]
    before = window.grid_view.images()
    window._handle_duplicates_done([group(images[:2]), group(images[2:], "near")])
    results = window._duplicate_results_dialog
    compare = results.compare_dialog
    assert compare.isVisible() and not results.isVisible()
    assert window.grid_view.images() == before
    results.show_group_list()
    results.kind_filter.setCurrentIndex(2)
    assert results.tree.topLevelItem(0).isHidden()
    results.compare_button.click()
    assert set(compare.cards) == {im.id for im in images[2:]}
    compare.loader.stop()
    assert compare.loader.pool.waitForDone(3000)
    results.close()


def test_remove_preserves_source_and_undo_restores_comparison(window, library, app):
    images = library[2]
    window._handle_duplicates_done([group(images[:2])])
    results = window._duplicate_results_dialog
    results.open_comparison()
    dialog = results.compare_dialog
    target = images[1]
    dialog.request_action("remove", target.id)
    assert Path(target.file_path).is_file()
    assert window.store.get_image(target.id) is None
    assert window.store.get_image(images[0].id) is not None
    assert not dialog.groups
    key(app, dialog, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert window.store.get_image(target.id) is not None
    assert len(dialog.groups) == 1
    assert set(dialog.cards) == {image.id for image in images[:2]}
    assert dialog.selected_image_id is None
    dialog.loader.stop()
    assert dialog.loader.pool.waitForDone(3000)
    results.close()


def test_deleting_reference_rebases_only_the_visible_pair_and_undo_restores(window, library):
    images = library[2][:3]
    window._handle_duplicates_done([group(images)])
    results = window._duplicate_results_dialog
    results.open_comparison()
    dialog = results.compare_dialog
    dialog.change_candidate(1)
    assert set(dialog.cards) == {images[0].id, images[2].id}
    dialog.request_action("remove", images[0].id)
    assert set(dialog.cards) == {images[1].id, images[2].id}
    assert dialog.selected_image_id is None
    dialog.undoRequested.emit()
    assert len(dialog.groups[0].members) == 3
    assert set(dialog.cards) == {images[0].id, images[2].id}
    assert not dialog.selected_image_id
    dialog.loader.stop()
    assert dialog.loader.pool.waitForDone(3000)
    results.close()


def test_delete_backs_up_only_target_and_undo_restores_source(window, library, tmp_path):
    images = library[2]
    window._handle_duplicates_done([group(images[:2])])
    results = window._duplicate_results_dialog
    results.open_comparison()
    dialog = results.compare_dialog
    target = images[0]
    original = Path(target.file_path).read_bytes()

    def trash(path):
        path.rename(tmp_path / "trashed.png")
        return True, ""

    with patch.object(window, "_move_source_file_to_trash", side_effect=trash) as move:
        dialog.request_action("delete", target.id)
    move.assert_called_once_with(Path(target.file_path))
    assert not Path(target.file_path).exists()
    assert Path(images[1].file_path).is_file()
    assert window.store.get_image(target.id) is None
    dialog.undoRequested.emit()
    assert Path(target.file_path).read_bytes() == original
    assert window.store.get_image(target.id) is not None
    assert len(dialog.groups) == 1
    dialog.loader.stop()
    assert dialog.loader.pool.waitForDone(3000)
    results.close()


def test_failed_removal_does_not_drop_group_or_select_next_image(window, library):
    images = library[2]
    window._handle_duplicates_done([group(images[:3])])
    results = window._duplicate_results_dialog
    results.open_comparison()
    dialog = results.compare_dialog
    with patch.object(window, "_remove_images_from_library_with_undo", return_value=0):
        dialog.request_action("remove", images[1].id)
    assert len(dialog.groups[0].members) == 3
    assert len(dialog.cards) == 2
    assert dialog.selected_image_id is None
    assert all(window.store.get_image(image.id) for image in images)
    dialog.loader.stop()
    assert dialog.loader.pool.waitForDone(3000)
    results.close()


def test_failed_trash_keeps_source_index_and_comparison(window, library):
    images = library[2]
    window._handle_duplicates_done([group(images[:2])])
    results = window._duplicate_results_dialog
    results.open_comparison()
    dialog = results.compare_dialog
    with patch.object(window, "_move_source_file_to_trash", return_value=(False, "permission denied")), \
         patch.object(QMessageBox, "warning") as warning:
        dialog.request_action("delete", images[0].id)
    warning.assert_called_once()
    assert len(dialog.cards) == 2
    assert dialog.selected_image_id is None
    assert Path(images[0].file_path).is_file()
    assert window.store.get_image(images[0].id) is not None
    dialog.loader.stop()
    assert dialog.loader.pool.waitForDone(3000)
    results.close()
