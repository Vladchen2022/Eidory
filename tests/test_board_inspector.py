from copy import deepcopy

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtTest import QTest

from test_filter_workflow import app, library, window
from test_creative_flow import make_project
from eidory.ui.project_board import ProjectBoardView, BoardImageItem


def board_project(w, images):
    pid, a, nodes = make_project(w)
    b = next(n for n in nodes if n.title == '地点')
    w.store.add_images_to_creative_node(a.id, [im.id for im in images[:2]])
    w.store.add_images_to_creative_node(b.id, [images[2].id])
    w.creative_flow.show_project_board()
    return pid, a, b, w.creative_flow.board_inspector


def test_board_has_inline_inspector_and_grade_changes_preserve_selection_geometry_and_zoom(window, library):
    w = window
    pid, a, b, inspector = board_project(w, library[2])
    board = w.project_board_view
    iid = library[2][0].id
    board.select_image_id(iid)
    assert inspector.scope.currentText() == a.title
    board.reset_view()
    board._image_items[iid].setPos(QPointF(721, 419))
    before = deepcopy(board.layout_payload())
    transform = board.transform()
    inspector.quick_grades['primary'].click()
    assert w.workflow_upgrade.state.decision(a.id, iid)['status'] == 'primary'
    assert board.selected_image_ids() == [iid]
    assert board.transform() == transform
    assert board.layout_payload() == before
    assert '主参考' in board._image_items[iid]._badges
    assert w.creative_flow.reference_page.currentWidget() is inspector.page
    assert not inspector.actions.isHidden()
    assert w.creative_flow.tray.isVisible() is False
    inspector.purpose.setCurrentIndex(inspector.purpose.findData('造型'))
    inspector.note.setPlainText('只参考轮廓')
    QTest.qWait(350)
    assert w.workflow_upgrade.state.decision(a.id, iid)['purpose'] == '造型'
    assert w.creative_workspace.store.reference_note(a.id, iid) == '只参考轮廓'
    w._save_current_board_layout_if_needed(force=True)
    w.creative_flow.show_planning()
    assert inspector.actions.isHidden()
    w.creative_flow.show_project_board()
    board.select_image_id(iid)
    assert inspector.grade.currentData() == 'primary'
    assert inspector.note.toPlainText() == '只参考轮廓'
    assert board._image_items[iid].pos() == QPointF(721,419)


def test_shared_image_requires_explicit_node_and_keeps_other_purposes(window, library):
    w = window
    pid, a, b, inspector = board_project(w, library[2])
    iid = library[2][0].id
    w.store.add_images_to_creative_node(b.id, [iid])
    w.workflow_upgrade.state.set_decision(b.id, iid, 'candidate', '空间')
    w._show_creative_board_for_node(w._current_board_node_id, force=True)
    w.project_board_view.select_image_id(iid)
    assert inspector.scope.currentData() is None
    assert not inspector.quick_grades['primary'].isEnabled()
    inspector.scope.setCurrentIndex(inspector.scope.findData(a.id))
    inspector.quick_grades['primary'].click()
    assert w.workflow_upgrade.state.decision(a.id, iid)['status'] == 'primary'
    assert w.workflow_upgrade.state.decision(b.id, iid) == {'node_id':b.id, 'image_id':iid,'status':'candidate','purpose':'空间'}
    inspector.scope.setCurrentIndex(inspector.scope.findData('all'))
    assert inspector.grade.currentData() is None
    inspector.quick_grades['unused'].click()
    assert w.workflow_upgrade.state.decision(b.id, iid)['status'] == 'unused'
    assert w.workflow_upgrade.state.decision(b.id, iid)['purpose'] == '空间'
    assert '主参考' not in w.project_board_view._image_items[iid]._badges


def test_batch_grades_apply_only_selected_references(window, library):
    w = window
    pid, a, b, inspector = board_project(w, library[2])
    ids = [im.id for im in library[2][:3]]
    for iid in (ids[0],ids[2]): w.project_board_view._image_items[iid].setSelected(True)
    inspector.quick_grades['primary'].click()
    assert w.workflow_upgrade.state.decision(a.id, ids[0])['status']=='primary'
    assert w.workflow_upgrade.state.decision(b.id, ids[2])['status']=='primary'
    assert w.workflow_upgrade.state.decision(a.id, ids[1])['status']=='candidate'
    assert not inspector.note.isEnabled()


def test_board_remove_button_and_undo_keep_reference_grade_and_original_file(window, library):
    w = window
    pid, a, b, inspector = board_project(w, library[2])
    image = library[2][0]
    w.project_board_view.select_image_id(image.id)
    inspector.quick_grades['primary'].click()
    inspector.purpose.setCurrentIndex(inspector.purpose.findData('光线'))
    inspector.note.setPlainText('保留用途说明')
    inspector.remove_button.click()
    assert image.id not in w.project_board_view._image_items
    assert w.store.get_image(image.id) is not None
    w.creative_workspace.board_buttons[2].click()
    assert image.id in w.project_board_view._image_items
    assert w.workflow_upgrade.state.decision(a.id, image.id)['status'] == 'primary'
    assert w.workflow_upgrade.state.decision(a.id, image.id)['purpose'] == '光线'
    assert w.creative_workspace.store.reference_note(a.id, image.id) == '保留用途说明'


def test_cancel_sections_preserves_locked_frames_geometry_notes_and_roundtrip(window, library):
    w = window
    board_project(w, library[2])
    board = w.project_board_view
    button = w.creative_workspace.board_buttons[1]
    assert board._section_items and button.isChecked()
    board._section_items[0].locked=True
    board._add_text('自定义便签',987,654)
    before=deepcopy(board.layout_payload())
    button.click()
    assert not board._section_items and not button.isChecked()
    assert board.layout_payload()['items']==before['items']
    assert board.layout_payload()['notes']==before['notes']
    assert board.undo_layout()
    assert board.layout_payload()['sections']==before['sections'] and button.isChecked()
    assert board.redo_layout() and not button.isChecked()
    iid=library[2][0].id
    board._image_items[iid].setPos(QPointF(813,742))
    w._save_current_board_layout_if_needed(force=True)
    w.creative_flow.show_planning(); w.creative_flow.show_project_board()
    assert not board._section_items
    assert board._image_items[iid].pos()==QPointF(813,742)


def test_project_board_entry_ignores_stale_search_result_mode(window, library):
    w = window
    pid, a, b, inspector = board_project(w, library[2])
    w.creative_flow.select_node(a.id)
    w.current_search_result_set_project_id = 999
    w.creative_flow.show_project_board()
    assert w.center_result_stack.currentWidget() is w.project_board_view
    assert w._current_board_node_id == w.store.creative_root_node_id(pid)


def test_primary_border_remains_visible_without_selection(window, library):
    w = window
    board_project(w, library[2])
    board=w.project_board_view
    iid=library[2][0].id
    board.set_reference_badges({iid:['物件','主参考']})
    item=board._image_items[iid]
    assert isinstance(item, BoardImageItem)
    bounds=item.sceneBoundingRect()
    image=QImage(500,500,QImage.Format.Format_ARGB32)
    image.fill(QColor('white'))
    p=QPainter(image)
    board.scene().render(p, image.rect(),bounds)
    p.end()
    red=BoardImageItem.PRIMARY_OUTLINE_COLOR
    assert any(image.pixelColor(x,1).name()==red.name() for x in range(30,470))
