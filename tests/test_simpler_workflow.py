from copy import deepcopy
from unittest.mock import patch
from dataclasses import replace

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QColor

from test_filter_workflow import app, library, window, apply, PLANE
from test_creative_flow import make_project
from test_board_inspector import board_project
from eidory.core.search_filters import FilterPlan, FilterGroup, SearchFilter
from eidory.ui.compare_dialog import ImageCompareDialog


def test_filters_update_after_pause_without_apply(window):
    w = window
    w.filter_panel.put_condition('include', 0, PLANE)
    QTest.qWait(650); w._poll_events()
    assert {i.file_name for i in w.grid_view.images()} == {'red-plane.png', 'blue-plane.png'}
    assert w.filter_panel.apply_button.isHidden()
    w.filter_panel.clear_draft()
    QTest.qWait(650); w._poll_events()
    assert len(w.grid_view.images()) == 4


def test_search_target_auto_updates_and_retains_original_note(window):
    w = window
    _, node, _ = make_project(w)
    w.creative_flow.open_search(); w._poll_events()
    row = w.creative_flow.targets()[0]
    w.creative_flow.change_target(row['id'], query='robot joints')
    QTest.qWait(650); w._poll_events()
    assert any(f.value == 'robot joints' for f in w.filter_workflow.plan().positive_filters)
    assert w.store.get_creative_node(node.id).note == node.note
    assert w.creative_flow.search_refresh.isHidden()


def test_zero_results_explain_recoverable_condition(window):
    w = window
    apply(w, FilterPlan((FilterGroup((PLANE, SearchFilter('keyword','absent')), 'all'),)))
    assert not w.grid_view.images()
    choices = w.simple_search.rescue_options()
    assert choices[0][0] == 2
    w.result_excluded_collection_image_ids = {w.current_chain_result.ranked_scope_images[0].id}
    reduced = w.simple_search.rescue_options()
    assert all(option[0] <= original[0] for option, original in zip(reduced, choices))
    w.result_excluded_collection_image_ids.clear()
    w.filter_workflow.apply(choices[0][2]); w._poll_events()
    assert len(w.grid_view.images()) == 2


def test_feedback_changes_ranking_only_and_can_undo(window, library):
    w = window
    apply(w, FilterPlan((FilterGroup((PLANE,)),)))
    service = w.simple_search
    before = w.grid_view.images()
    service.context = service.signature()
    service.serial = 23
    service.accept_ranking(23, {'feedback': {before[-1].id:1}, 'scores': {before[-1].id: 2},
                               'before': ({}, {}, service.context)})
    assert w.grid_view.images()[0].id == before[-1].id
    assert {i.id for i in w.grid_view.images()} == {i.id for i in before}
    w.grid_view.setFocus(); w.edit_history.undo()
    assert [i.id for i in w.grid_view.images()] == [i.id for i in before]
    w.edit_history.redo()
    assert w.grid_view.images()[0].id == before[-1].id
    service.clear_feedback()
    assert [i.id for i in w.grid_view.images()] == [i.id for i in before]


def test_cancel_rejects_result_already_queued_and_older_generation(window):
    control = window.task_control
    old = control.begin('旧请求', lambda: None)
    old.emit('creative_node_note_done', (99, '', 'must not save', 'fake'))
    control.cancel()
    new = control.begin('新请求', lambda: None)
    with patch.object(window, '_handle_creative_node_note_done') as handler:
        window._poll_events()
        assert not handler.called
        assert control.current is new
        old.emit('creative_node_note_error', (99, RuntimeError('late')))
        window._poll_events()
        assert control.current is new
    control.cancel()


def test_batch_cancel_preserves_completed_suggestions(window):
    w = window
    pid, node, _ = make_project(w)
    job = w.task_control.begin('批量', lambda: None, batch=(pid, node.id))
    result = (node.id, node.note, 'test')
    job.checkpoint(result)
    with patch.object(w, '_handle_creative_all_node_notes_done') as save:
        w.task_control.cancel()
        assert save.call_args.args[2] == [result]
    job.checkpoint((node.id, '', 'late'))
    assert job.completed == [result]


def test_history_grade_remove_undo_redo_in_order(window, library):
    w = window
    pid, node, _, inspector = board_project(w, library[2])
    board = w.project_board_view
    iid = library[2][0].id
    board.select_image_id(iid)
    inspector.quick_grades['primary'].click()
    inspector.remove_button.click()
    board.setFocus()
    w.edit_history.undo()
    assert iid in board._image_items
    assert w.workflow_upgrade.state.decision(node.id, iid)['status'] == 'primary'
    w.edit_history.undo()
    assert w.workflow_upgrade.state.decision(node.id, iid)['status'] == 'candidate'
    w.edit_history.redo()
    assert w.workflow_upgrade.state.decision(node.id, iid)['status'] == 'primary'
    w.edit_history.redo()
    assert iid not in board._image_items


def test_layout_history_does_not_create_recursive_commands(window, library):
    w = window
    board_project(w, library[2]); board = w.project_board_view
    before = deepcopy(board.layout_payload())
    iid = library[2][0].id
    board._image_items[iid].setPos(QPointF(920,700)); board.flush_pending_layout_change()
    after = deepcopy(board.layout_payload())
    assert after != before
    count = len(w.edit_history.past)
    board.setFocus(); w.edit_history.undo()
    assert board.layout_payload() == before
    assert len(w.edit_history.past) == count-1
    w.edit_history.redo()
    assert board.layout_payload() == after


def test_history_does_not_overwrite_later_changes(window, library):
    w = window
    _, node, _, inspector = board_project(w, library[2])
    iid = library[2][0].id
    w.project_board_view.select_image_id(iid)
    inspector.quick_grades['primary'].click()
    w.workflow_upgrade.state.set_decision(node.id, iid, 'unused', '')
    w.project_board_view.setFocus(); w.edit_history.undo()
    assert w.workflow_upgrade.state.decision(node.id, iid)['status'] == 'unused'


def test_board_focus_does_not_edit_layout_and_grade_updates_highlight(window, library):
    w = window
    _, _, _, inspector = board_project(w, library[2]); b = w.project_board_view
    iid = library[2][0].id
    before = deepcopy(b.layout_payload())
    w.board_navigation.focus([iid])
    assert b.layout_payload() == before
    w.board_navigation.set_primary(True)
    b.select_image_id(iid); inspector.quick_grades['primary'].click()
    assert b._image_items[iid].opacity() == 1
    assert b._image_items[library[2][1].id].opacity() < .5


def test_compare_changes_grade_without_leaving_board_and_ignores_stale_load(window, library):
    w = window
    _, node, _, _ = board_project(w, library[2])
    d = ImageCompareDialog(library[2][:3], parent=w)
    d.show(); QApplication.processEvents()
    d.grade_buttons[0][0].setChecked(True)
    assert not d.grade_buttons[0][1].isChecked()
    assert w.workflow_upgrade.state.decision(node.id, library[2][0].id)['status'] == 'primary'
    assert w.center_result_stack.currentWidget() is w.project_board_view
    version = d.generation[1]
    d.choices[1].setCurrentIndex(2)
    image = QImage(40, 40, QImage.Format.Format_RGB32); image.fill(QColor('red'))
    with patch.object(d.views[1], 'set_pixmap') as setter:
        d.loaded((1, version), image, '')
        assert not setter.called
    d.done(0)
    QTest.qWait(250)
    from shiboken6 import isValid
    assert not isValid(d)


def test_retry_details_keeps_the_original_entry_target(window):
    w = window
    _, node, _ = make_project(w)
    with patch.object(w.creative_flow, 'entry_details') as details:
        w._retry_node_completion(node.id, 'details', 'original-entry')
        details.assert_called_once_with('original-entry')


def test_rejected_ai_job_releases_controls(window):
    w = window
    make_project(w)
    def reject(_run, **kw):
        kw['on_rejected']()
        return False
    with patch.object(w, '_start_background_task', side_effect=reject):
        w._generate_creative_children_for_selected_node()
    assert w.task_control.current is None
    assert not w._creative_node_note_in_progress
    assert not w._creative_all_nodes_in_progress


def test_hidden_board_memory_trim_preserves_layout(window, library):
    w = window
    board_project(w, library[2]); b = w.project_board_view
    QTest.qWait(150)
    before = deepcopy(b.layout_payload())
    w.creative_flow.show_planning()
    b.trim_image_memory(1)
    assert b.layout_payload() == before
    assert all(max(item.source_pixmap().width(),item.source_pixmap().height()) <= 128 for item in b._image_items.values())


def test_narrow_window_moves_sort_into_menu_without_hiding_sidebars(window):
    w = window
    w.resize(1000, 700); QApplication.processEvents(); w.responsive_chrome.refresh()
    assert not w.root_splitter.widget(0).isHidden()
    assert w.responsive_chrome.more.isVisible()
    assert w.sort_combo.isHidden()
    w.responsive_chrome.build_menu()
    menu = w.responsive_chrome.menu.actions()[0].menu()
    target = min(1, w.sort_combo.count()-1)
    menu.actions()[target].trigger()
    assert w.sort_combo.currentIndex() == target


def test_target_typing_is_one_undo_step(window):
    w = window
    _, node, _ = make_project(w)
    row = w.creative_flow.targets()[0]
    original = row['query']
    count = len(w.edit_history.past)
    for text in ('r', 'ro', 'robot'):
        w.creative_flow.change_target(row['id'], query=text)
    assert len(w.edit_history.past) == count + 1
    w.project_board_view.setFocus()
    QApplication.focusWidget().clearFocus()
    w.edit_history.undo()
    assert w.creative_flow.targets()[0]['query'] == original


def test_thumbnail_repair_runs_off_gui_and_cancel_preserves_finished_items(window, library):
    import threading
    w = window
    original = w.thumbnailer.generate
    thread_ids = []
    threads = []
    def generate(*args):
        thread_ids.append(threading.get_ident())
        path = original(*args)
        w.thumbnail_job.cancelled.set()
        return path
    def start(target, **kw):
        t = threading.Thread(target=target)
        threads.append(t); t.start()
    with patch.object(w, '_start_background_task', side_effect=start), patch.object(w.thumbnailer, 'generate', side_effect=generate):
        w.thumbnail_job.start(library[2])
        for _ in range(100):
            QTest.qWait(10)
            if not w.thumbnail_job.running:
                break
        for t in threads:
            t.join(3)
    assert thread_ids and all(t != threading.get_ident() for t in thread_ids)
    assert len(thread_ids) == 1
    assert not w.thumbnail_job.running
    assert '成功 1' in w.statusBar().currentMessage()


def test_navigation_map_stays_at_corner_after_scroll(window, library):
    w = window
    board_project(w, library[2]); b = w.project_board_view
    b._zoom_by(3)
    b.horizontalScrollBar().setValue(b.horizontalScrollBar().maximum()//2)
    b.verticalScrollBar().setValue(b.verticalScrollBar().maximum()//2)
    QTest.qWait(40)
    m = w.board_navigation.map
    assert m.x() == max(0, b.viewport().width()-m.width()-14)
    assert m.y() == max(0, b.viewport().height()-m.height()-14)
