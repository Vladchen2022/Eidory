from dataclasses import replace
from unittest.mock import patch

from PySide6.QtTest import QTest

from test_filter_workflow import app, library, window, FakeSearch, visible_names
from eidory.core.filter_engine import FilterEngine, refresh_annotation_matches
from eidory.core.preview_annotations import AnnotationStore, Stroke
from eidory.core.search_filters import FilterGroup, FilterPlan, SearchFilter, filter_label, search_filter_from_payload
from eidory.ui.filter_panel import FilterEditorDialog


INK = Stroke(((.1, .2), (.7, .8)), .01, "#E53935")
MARKED = SearchFilter("annotation", "present")


def save_ink(store, image, strokes=(INK,)):
    AnnotationStore(store.database_path).save(image.id, strokes, (image.file_size, image.modified_time_ns))


def test_annotation_editor_and_saved_plan_roundtrip(app):
    dialog = FilterEditorDialog(condition=None, label_for=filter_label, pick_values=lambda *a: None)
    dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findText("标记"))
    assert dialog.value_combo.currentText() == "含有标记"
    assert dialog.strength_box.isHidden()
    dialog.accept()
    assert dialog.result_filter == MARKED
    assert filter_label(MARKED) == "标记：含有标记"
    plan = FilterPlan((FilterGroup((MARKED,)),), (FilterGroup((MARKED,)),))
    assert FilterPlan.from_payload(plan.to_payload()) == plan
    assert search_filter_from_payload({"kind": "annotation", "value": "unsupported"}) is None
    edit = FilterEditorDialog(condition=MARKED, label_for=filter_label, pick_values=lambda *a: None)
    assert edit.kind_combo.currentData() == "annotation"
    edit.reject()


def test_annotation_keep_exclude_and_combination_respect_scope(library):
    _, store, images = library
    save_ink(store, images[0])
    save_ink(store, images[2])
    engine = FilterEngine(store, FakeSearch(images))
    keep = FilterPlan((FilterGroup((MARKED,)),))
    assert [im.id for im in engine.evaluate(keep, images).images] == [images[0].id, images[2].id]
    assert [im.id for im in engine.evaluate(keep, images[1:]).images] == [images[2].id]
    exclude = FilterPlan(exclude=(FilterGroup((MARKED,)),))
    assert visible_names(engine.evaluate(exclude, images)) == {images[1].file_name, images[3].file_name}
    keyword = SearchFilter("keyword", images[0].file_name)
    assert [im.id for im in engine.evaluate(FilterPlan((FilterGroup((MARKED, keyword)),)), images).images] == [images[0].id]
    assert store.image_ids_with_preview_annotations() == {images[0].id, images[2].id}


def test_clear_and_cached_refresh_restore_original_scope_without_ai(library):
    _, store, images = library
    search = FakeSearch(images)
    engine = FilterEngine(store, search)
    semantic = SearchFilter("semantic", "anything", 0)
    plan = FilterPlan((FilterGroup((semantic, MARKED)),))
    result = engine.evaluate(plan, images)
    assert not result.images
    assert len(search.scopes) == 1
    save_ink(store, images[2])
    result = refresh_annotation_matches(result, store.image_ids_with_preview_annotations())
    assert [im.id for im in result.images] == [images[2].id]
    save_ink(store, images[2], ())
    assert not refresh_annotation_matches(result, store.image_ids_with_preview_annotations()).images
    assert len(search.scopes) == 1


def test_preview_membership_updates_wall_without_rerunning_semantics(window, library):
    _, store, images = library
    window._open_image_preview(images[0])
    preview = next(iter(window._preview_dialogs))
    semantic = SearchFilter("semantic", "anything", 0)
    window.filter_workflow.apply(FilterPlan((FilterGroup((semantic,)),), (FilterGroup((MARKED,)),)))
    window._poll_events()
    calls = len(window.search_service.scopes)
    assert len(window.grid_view.images()) == 4
    # Commit via the same editor signal used by mouse/tablet gestures.
    preview.image_view.annotations = (INK,)
    preview.image_view._ink_document_changed()
    QTest.qWait(120)
    assert preview.current_image().id not in {im.id for im in window.grid_view.images()}
    assert len(window.grid_view.images()) == 3
    preview.image_view.clear_annotations()
    QTest.qWait(120)
    assert len(window.grid_view.images()) == 4
    preview.image_view.undo_annotation()
    QTest.qWait(120)
    assert len(window.grid_view.images()) == 3
    preview.image_view.redo_annotation()
    QTest.qWait(120)
    assert len(window.grid_view.images()) == 4
    assert len(window.search_service.scopes) == calls
    preview.close()


def test_annotation_changes_during_pending_search_refresh_before_display(window, library):
    _, store, images = library
    plan = FilterPlan((FilterGroup((MARKED,)),))
    window.filter_workflow.apply(plan)
    window._poll_events()
    stale = window.current_chain_result
    assert not stale.images
    window.search_button.setEnabled(False)
    save_ink(store, images[0])
    window.filter_workflow.annotation_membership_changed(images[0].id)
    QTest.qWait(120)
    assert window.filter_workflow._annotation_dirty
    window.search_button.setEnabled(True)
    window._handle_search_chain_done(filters=plan.positive_filters, result=stale)
    assert [im.id for im in window.grid_view.images()] == [images[0].id]
    assert not window.filter_workflow._annotation_dirty


def test_membership_signal_only_fires_after_saved_first_or_last_stroke(window, library):
    _, store, images = library
    window._open_image_preview(images[0])
    preview = next(iter(window._preview_dialogs))
    changed = []
    preview.annotationMembershipChanged.connect(changed.append)
    view = preview.image_view
    view.annotations = (INK,)
    view._ink_document_changed()
    view.annotations = (INK, replace(INK, color="#000000"))
    view._ink_document_changed()
    preview.annotation_controls.visible.setChecked(False)
    assert changed == [images[0].id]
    view.clear_annotations()
    assert changed == [images[0].id]*2
    with patch.object(preview.annotation_controls.store, "save", side_effect=OSError("locked")):
        view.annotations = (INK,)
        view._ink_document_changed()
    assert changed == [images[0].id]*2
    preview.annotation_controls.flush()
    assert changed == [images[0].id]*3
    preview.close()


def test_annotation_change_does_not_hide_failed_search_or_reuse_stale_scope(window, library):
    _, _, images = library
    window.filter_workflow.apply(FilterPlan((FilterGroup((MARKED,)),)))
    window._poll_events()
    window.filter_panel.error = "筛选失败，仍显示上次结果"
    with patch.object(window, "_handle_search_chain_done") as show:
        window.filter_workflow.annotation_membership_changed(images[0].id)
        QTest.qWait(120)
    show.assert_not_called()
    assert window.filter_panel.error and window.filter_workflow._annotation_dirty
