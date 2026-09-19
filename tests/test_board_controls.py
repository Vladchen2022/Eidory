from copy import deepcopy
from unittest.mock import patch

from PySide6.QtCore import QPoint, QPointF, Qt, QEvent
from PySide6.QtGui import QWheelEvent, QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QVBoxLayout

from test_filter_workflow import app, library, window
from test_board_inspector import board_project
from test_creative_workspace import project


def wheel(board, delta=120, pixel=False):
    pos = QPoint(160, 160)
    event = QWheelEvent(QPointF(pos), QPointF(board.viewport().mapToGlobal(pos)),
                        QPoint(0, delta) if pixel else QPoint(),
                        QPoint() if pixel else QPoint(0, delta), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(board.viewport(), event)
    return pos


def test_wheel_zoom_is_anchored_bounded_and_defers_decode(window, library, app):
    w = window
    board_project(w, library[2])
    board = w.project_board_view
    QTest.qWait(150)
    board.select_image_id(library[2][0].id)
    before = deepcopy(board.layout_payload())
    zoom = board._zoom
    anchor = board.mapToScene(QPoint(160, 160))
    with patch.object(board, '_ensure_focus_quality_for_items') as decode:
        wheel(board)
        assert board._zoom > zoom
        assert (board.mapToScene(QPoint(160, 160)) - anchor).manhattanLength() < 8
        wheel(board, -120)
        assert abs(board._zoom - zoom) < 1e-6
        for _ in range(40):
            wheel(board, 1, pixel=True)
        assert board._zoom > zoom
        assert decode.call_count == 0
        assert board._quality_timer.isActive()
        for _ in range(100):
            wheel(board, 1200)
        assert board._zoom == 16
        for _ in range(100):
            wheel(board, -1200)
        assert board._zoom == .15
    assert board.layout_payload() == before


def test_black_white_uses_selection_or_all_and_is_one_undo_step(window, library, app):
    w = window
    board_project(w, library[2])
    board = w.project_board_view
    QTest.qWait(150)
    items = list(board._image_items.values())
    board.select_image_id(items[0].image_id)
    w.board_grayscale_button.click()
    assert items[0].is_grayscale()
    assert not any(i.is_grayscale() for i in items[1:])
    board.scene().clearSelection()
    history = len(w.edit_history.past)
    w.board_grayscale_button.click()
    assert all(i.is_grayscale() for i in items)
    assert len(w.edit_history.past) == history + 1
    w.edit_history.undo()
    assert board._image_items[items[0].image_id].is_grayscale()
    assert not any(board._image_items[i.image_id].is_grayscale() for i in items[1:])
    w.edit_history.redo()
    w.board_grayscale_button.click()
    assert not any(i.is_grayscale() for i in board._image_items.values())


def test_board_keys_preserve_text_inputs_and_work_when_overflowed(window, library, app):
    w = window
    _, _, _, inspector = board_project(w, library[2])
    board = w.project_board_view
    iid = library[2][0].id
    board.select_image_id(iid)
    w.activateWindow()
    board.setFocus()
    QTest.qWait(150)
    w.resize(1050, 900)
    QTest.qWait(30)
    assert not w.board_flip_button.isVisible()
    QTest.keyClick(board, Qt.Key.Key_F)
    assert board._image_items[iid].is_flipped()
    QTest.keyClick(board, Qt.Key.Key_Z)
    assert not board._image_items[iid].is_flipped()
    assert board.selected_image_ids() == [iid]
    QTest.keyClick(board, Qt.Key.Key_Y)
    assert board._image_items[iid].is_flipped()
    assert board.selected_image_ids() == [iid]
    QTest.keyClick(board, Qt.Key.Key_2)
    assert '主参考' in board._image_items[iid]._badges
    for key in (Qt.Key.Key_G, Qt.Key.Key_L):
        QTest.keyClick(board, key)
    assert next(s for s in board._section_items if iid in s.image_ids).locked
    assert '(G)' in w.creative_workspace.board_buttons[1].text()
    QTest.keyClick(board, Qt.Key.Key_H)
    assert w._board_focus_mode
    QTest.keyClick(board, Qt.Key.Key_H)
    assert not w._board_focus_mode
    inspector.note.setFocus()
    before = board.layout_payload()
    QTest.keyClicks(inspector.note, 'fbz123')
    assert inspector.note.toPlainText().endswith('fbz123')
    assert board.layout_payload() == before
    dialog = QDialog(w)
    layout = QVBoxLayout(dialog)
    text = QLineEdit(dialog)
    layout.addWidget(text)
    dialog.show()
    text.setFocus()
    QTest.qWait(10)
    QTest.keyClicks(text, 'fbg')
    assert text.text() == 'fbg'
    assert board.layout_payload() == before
    dialog.close()
    w.creative_flow.show_planning()
    with patch.object(w.board_flip_button, 'click') as click:
        QTest.keyClick(w, Qt.Key.Key_F)
        assert not click.called


def test_enter_collects_grid_selection_only_in_project_and_respects_dirty_targets(window, library, app):
    w = window
    pid, node = project(w)
    w._show_gallery_view()
    ids = [i.id for i in library[2][:2]]
    w.grid_view.set_images(library[2])
    w.grid_view.select_image_ids(ids)
    w.creative_workspace.refresh()
    w.activateWindow()
    w.grid_view.setFocus()
    QTest.qWait(20)
    QTest.keyClick(w.grid_view, Qt.Key.Key_Return)
    assert set(w.store.creative_node_image_ids(node.id)) == set(ids)
    QTest.keyClick(w.grid_view, Qt.Key.Key_Enter)
    assert len(w.store.creative_node_image_ids(node.id)) == 2
    w.grid_view.select_image_ids([library[2][2].id])
    w.creative_flow._search_dirty_node = node.id
    QTest.keyClick(w.grid_view, Qt.Key.Key_Return)
    assert len(w.store.creative_node_image_ids(node.id)) == 2
    w._load_creative_project(None)
    w.grid_view.setFocus()
    with patch.object(w.creative_workspace.collect_button, 'click') as click:
        QTest.keyClick(w.grid_view, Qt.Key.Key_Return)
        assert not click.called
