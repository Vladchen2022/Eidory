from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from eidory.config import AppPaths
from eidory.core.filter_engine import FilterEngine
from eidory.core.metadata_store import MetadataStore
from eidory.core.scanner import ImageScanner
from eidory.core.search_filters import FilterGroup, FilterPlan, SearchFilter
from eidory.core.thumbnailer import Thumbnailer
from eidory.ui.filter_panel import FilterEditorDialog
from eidory.ui.main_window import MainWindow
from eidory.ui.accessibility import disable_qt_accessibility


@pytest.fixture(scope="session")
def app():
    disable_qt_accessibility()
    app = QApplication.instance() or QApplication([])
    disable_qt_accessibility()
    return app


@pytest.fixture
def library(tmp_path):
    paths = AppPaths(tmp_path / "data", tmp_path / "data/thumbs", tmp_path / "data/db.sqlite3", tmp_path / "data/logs")
    paths.ensure()
    store = MetadataStore(paths.database_path)
    store.initialize()
    store.set_setting("ui.file_watch_enabled", "0")
    directory = tmp_path / "images"
    directory.mkdir()
    for name, color in [("red-plane", "red"), ("red-car", "red"), ("blue-plane", "blue"), ("blue-boat", "blue")]:
        Image.new("RGB", (120, 80), color).save(directory / (name + ".png"))
    scan = ImageScanner(store, Thumbnailer(paths.thumbnail_dir)).scan_folder(str(directory))
    images = store.images_by_ids(scan.image_ids)
    return paths, store, images


class FakeSearch:
    def __init__(self, images):
        self.images = images
        self.scopes = []

    def color_search(self, rgb, **kwargs):
        self.scopes.append(set(kwargs["allowed_image_ids"]))
        images = [replace(im, score=.9 if im.file_name.startswith("red") else .1) for im in self.images if im.id in kwargs["allowed_image_ids"]]
        return SimpleNamespace(images=images, searchable_count=len(images), indexed_count=len(images), candidate_limit=len(images))

    def semantic_search(self, text, **kwargs):
        self.scopes.append(set(kwargs["allowed_image_ids"]))
        scores = {"red-plane.png": .9, "red-car.png": .7, "blue-plane.png": .6, "blue-boat.png": .2}
        images = [replace(im, score=scores[im.file_name]) for im in self.images if im.id in kwargs["allowed_image_ids"]]
        return SimpleNamespace(images=images, searchable_count=len(images), candidate_limit=len(images))


@pytest.fixture
def window(app, library):
    paths, store, images = library
    w = MainWindow(paths=paths, store=store)
    w.search_service = FakeSearch(images)
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    with patch.object(w, "_start_background_task", side_effect=lambda target, **kw: target()):
        yield w
    w.close()
    for pool in w.findChildren(QThreadPool):
        pool.clear()
        assert pool.waitForDone(3000)
    # Keep QObject roots alive for the QApplication lifetime. PySide 6.11 can
    # crash in its item-wrapper cleanup when these windows are repeatedly
    # deleted while pytest still holds references to their child widgets.
    if not hasattr(app, "_filter_test_windows"):
        app._filter_test_windows = []
    app._filter_test_windows.append(w)
    app.processEvents()


def visible_names(result):
    return {im.file_name for im in result.images if im.id not in result.excluded_image_ids}


RED = SearchFilter("color", (255, 0, 0), 50)
PLANE = SearchFilter("keyword", "plane")


@pytest.mark.parametrize("mode,expected", [
    ("all", {"red-car.png", "blue-plane.png", "blue-boat.png"}),
    ("any", {"blue-boat.png"}),
])
def test_exclusion_group_boolean_modes(library, mode, expected):
    _, store, images = library
    engine = FilterEngine(store, FakeSearch(images))
    result = engine.evaluate(FilterPlan(exclude=(FilterGroup((RED, PLANE), mode),)), images)
    assert visible_names(result) == expected


def test_separate_batches_union_and_empty_groups_do_not_exclude(library):
    _, store, images = library
    engine = FilterEngine(store, FakeSearch(images))
    result = engine.evaluate(FilterPlan(exclude=(FilterGroup((RED,)), FilterGroup((PLANE,)), FilterGroup((), "all"))), images)
    assert visible_names(result) == {"blue-boat.png"}
    assert visible_names(engine.evaluate(FilterPlan(exclude=(FilterGroup(),)), images)) == {im.file_name for im in images}


def test_scored_rules_are_order_independent_and_use_full_scope(library):
    _, store, images = library
    search = FakeSearch(images)
    engine = FilterEngine(store, search)
    semantic = SearchFilter("semantic", "planes", 50)
    first = engine.evaluate(FilterPlan((FilterGroup((semantic, RED)),)), images)
    second = engine.evaluate(FilterPlan((FilterGroup((RED, semantic)),)), images)
    assert visible_names(first) == visible_names(second) == {"red-plane.png", "red-car.png"}
    assert all(scope == {im.id for im in images} for scope in search.scopes)


def test_exclusion_strength_is_independent_of_positive_strength(library):
    _, store, images = library
    engine = FilterEngine(store, FakeSearch(images))
    plan = FilterPlan((FilterGroup((SearchFilter("semantic", "planes", 10),)),), (FilterGroup((RED,)),))
    first = engine.evaluate(plan, images)
    second = engine.evaluate(replace(plan, include=(FilterGroup((SearchFilter("semantic", "planes", 50),)),)), images)
    assert first.excluded_image_ids == second.excluded_image_ids


def test_filter_plan_roundtrip_preserves_groups_and_rejects_corruption():
    plan = FilterPlan((FilterGroup((PLANE,)),), (FilterGroup((RED, PLANE), "any"), FilterGroup((SearchFilter("size", "small"),))))
    assert FilterPlan.from_payload(json.loads(json.dumps(plan.to_payload()))) == plan
    broken = plan.to_payload()
    broken["exclude"][0]["match_mode"] = "oops"
    with pytest.raises(ValueError):
        FilterPlan.from_payload(broken)


def apply(window, plan):
    window.filter_panel.draft = plan
    QTest.mouseClick(window.filter_panel.apply_button, Qt.MouseButton.LeftButton)
    window._poll_events()
    QApplication.processEvents()


def test_edits_wait_for_debounce_and_manual_update_can_flush(window):
    panel = window.filter_panel
    panel.put_condition("include", 0, PLANE)
    assert len(window.grid_view.images()) == 4
    assert "正在准备更新" in panel.status_label.text()
    QTest.mouseClick(panel.apply_button, Qt.MouseButton.LeftButton)
    window._poll_events()
    assert {im.file_name for im in window.grid_view.images()} == {"red-plane.png", "blue-plane.png"}


def test_remove_earlier_condition_restores_original_scope(window):
    apply(window, FilterPlan((FilterGroup((SearchFilter("keyword", "red"),)),)))
    window.filter_panel.put_condition("include", 0, PLANE)
    QTest.mouseClick(window.filter_panel.apply_button, Qt.MouseButton.LeftButton)
    window._poll_events()
    assert [im.file_name for im in window.grid_view.images()] == ["red-plane.png"]
    window.filter_panel.remove_condition("include", 0, 0)
    QTest.mouseClick(window.filter_panel.apply_button, Qt.MouseButton.LeftButton)
    window._poll_events()
    assert {im.file_name for im in window.grid_view.images()} == {"red-plane.png", "blue-plane.png"}
    assert window.current_chain_base_image_ids is None


def test_switch_exclusion_mode_apply_then_undo(window):
    plan = FilterPlan(exclude=(FilterGroup((RED, PLANE), "all"),))
    apply(window, plan)
    assert len(window.grid_view.images()) == 3
    window.filter_panel.set_group_mode("exclude", 0, "any")
    assert len(window.grid_view.images()) == 3
    QTest.mouseClick(window.filter_panel.apply_button, Qt.MouseButton.LeftButton)
    window._poll_events()
    assert [im.file_name for im in window.grid_view.images()] == ["blue-boat.png"]
    QTest.mouseClick(window.filter_panel.undo_button, Qt.MouseButton.LeftButton)
    window._poll_events()
    assert len(window.grid_view.images()) == 3
    assert window.filter_panel.applied.exclude[0].match_mode == "all"


def test_preset_restore_replaces_exclusions_and_restores_strength(window):
    plan = FilterPlan(exclude=(FilterGroup((RED, PLANE), "all"),))
    apply(window, plan)
    payload = json.loads(json.dumps(window._current_view_payload()))
    apply(window, FilterPlan(exclude=(FilterGroup((PLANE,)),)))
    window._apply_view_payload(payload)
    window._poll_events()
    assert window.filter_workflow.plan() == plan
    assert len(window.grid_view.images()) == 3
    old_preset = {"version": 1, "search_filters": [{"kind": "keyword", "value": "plane"}]}
    window._apply_view_payload(old_preset)
    window._poll_events()
    assert not window.result_exclusion_filters
    assert {im.file_name for im in window.grid_view.images()} == {"red-plane.png", "blue-plane.png"}


def test_frozen_scope_stays_explicit_and_empty_filters_restore_it(window):
    apply(window, FilterPlan((FilterGroup((PLANE,)),)))
    window.filter_workflow.freeze_scope()
    window._poll_events()
    assert len(window.grid_view.images()) == 2
    assert "固定" in window.filter_panel.scope_button.text()
    apply(window, FilterPlan(exclude=(FilterGroup((RED,)),)))
    assert [im.file_name for im in window.grid_view.images()] == ["blue-plane.png"]
    window.filter_panel.clear_draft()
    QTest.mouseClick(window.filter_panel.apply_button, Qt.MouseButton.LeftButton)
    window._poll_events()
    assert len(window.grid_view.images()) == 2


def test_condition_editor_does_not_run_search_and_keeps_threshold(window):
    with patch.object(window.search_service, "color_search", side_effect=AssertionError("unexpected search")):
        dialog = FilterEditorDialog(condition=RED, label_for=window._filter_label, pick_values=window.filter_workflow.pick_values, parent=window)
        dialog.strength_slider.setValue(73)
        dialog.accept()
        assert dialog.result_filter == replace(RED, score_threshold=73)


def test_metadata_exclusion_uses_same_group_engine(window):
    plan = FilterPlan(exclude=(FilterGroup((SearchFilter("orientation", "landscape"), PLANE), "all"),))
    apply(window, plan)
    assert {im.file_name for im in window.grid_view.images()} == {"red-car.png", "blue-boat.png"}


def test_unknown_ai_tags_do_not_match_exclusion(window):
    plan = FilterPlan(exclude=(FilterGroup((SearchFilter("ai_vision", "weather:rain"),)),))
    apply(window, plan)
    assert len(window.grid_view.images()) == 4


def test_manual_exclusions_clear_only_on_apply_and_undo_restores(window):
    apply(window, FilterPlan((FilterGroup((PLANE,)),)))
    hidden = window.grid_view.images()[0].id
    window.result_excluded_image_ids.add(hidden)
    window._refresh_visible_results_after_result_management_change()
    assert len(window.grid_view.images()) == 1
    window.filter_panel.clear_draft()
    assert len(window.grid_view.images()) == 1
    QTest.mouseClick(window.filter_panel.apply_button, Qt.MouseButton.LeftButton)
    window._poll_events()
    assert len(window.grid_view.images()) == 4
    window.filter_workflow.undo()
    window._poll_events()
    assert len(window.grid_view.images()) == 1
    assert hidden in window.result_excluded_image_ids


def test_empty_results_can_recover_by_removing_exclusion(window):
    apply(window, FilterPlan(exclude=(FilterGroup((SearchFilter("file_type", "media:image"),)),)))
    assert not window.grid_view.images()
    window.filter_panel.remove_group("exclude", 0)
    QTest.mouseClick(window.filter_panel.apply_button, Qt.MouseButton.LeftButton)
    window._poll_events()
    assert len(window.grid_view.images()) == 4


def test_switch_from_project_to_whole_library_releases_fixed_scope(window):
    images = window.grid_view.images()
    project_id = window.store.create_temporary_project("测试范围", [images[0].id])
    window._load_temporary_project(project_id)
    assert window.filter_workflow.base_context()[0] == {images[0].id}
    window.filter_workflow.select_all_scope()
    window._poll_events()
    assert len(window.grid_view.images()) == 4
    assert window.filter_workflow.scope == {"kind": "sidebar"}


def test_disk_scope_restore_and_sidebar_selection_release_snapshot(window):
    window.folder_tree.setCurrentItem(window.folder_tree.topLevelItem(1))
    prefix = window._selected_folder_path_prefix()
    assert prefix
    apply(window, FilterPlan((FilterGroup((PLANE,)),)))
    payload = window._current_view_payload()
    window.filter_workflow.freeze_scope()
    window._poll_events()
    window.folder_tree.setCurrentItem(window.folder_tree.topLevelItem(0))
    window._poll_events()
    assert len(window.grid_view.images()) == 4
    assert window.filter_workflow.scope == {"kind": "sidebar"}
    window._apply_view_payload(payload)
    window._poll_events()
    assert window._selected_folder_path_prefix() == prefix
    assert len(window.grid_view.images()) == 2


def test_stale_query_cannot_replace_wall_or_unlock_new_query(window):
    pending = []
    with patch.object(window, "_start_background_task", side_effect=lambda target, **kw: pending.append(target)):
        window.filter_workflow.apply(FilterPlan((FilterGroup((PLANE,)),)))
        assert len(pending) == 1
        window.folder_tree.setCurrentItem(window.folder_tree.topLevelItem(1))
        assert len(pending) == 2
        pending[0]()
        window._poll_events()
        assert not window.filter_panel.apply_button.isEnabled()
        assert len(window.grid_view.images()) == 4
        pending[1]()
        window._poll_events()
        assert window.filter_panel.apply_button.isEnabled()
        assert len(window.grid_view.images()) == 2


def test_failed_query_keeps_completed_wall_and_can_retry(window):
    apply(window, FilterPlan((FilterGroup((PLANE,)),)))
    before = [im.id for im in window.grid_view.images()]
    plan = FilterPlan(exclude=(FilterGroup((RED,)),))
    with patch.object(window.search_service, "color_search", side_effect=RuntimeError("sample failure")):
        apply(window, plan)
    assert [im.id for im in window.grid_view.images()] == before
    assert "筛选失败" in window.filter_panel.status_label.text()
    assert window.filter_panel.apply_button.isEnabled()
    apply(window, plan)
    assert {im.file_name for im in window.grid_view.images()} == {"blue-plane.png", "blue-boat.png"}


def test_multiple_positive_and_negative_groups_survive_preset(window):
    plan = FilterPlan(
        (FilterGroup((PLANE,)), FilterGroup((SearchFilter("keyword", "boat"),))),
        (FilterGroup((RED, PLANE), "all"), FilterGroup((SearchFilter("keyword", "boat"),))),
    )
    apply(window, plan)
    payload = json.loads(json.dumps(window._current_view_payload()))
    window.filter_workflow.select_all_scope()
    window._apply_view_payload(payload)
    window._poll_events()
    assert window.filter_workflow.plan() == plan
    assert [im.file_name for im in window.grid_view.images()] == ["blue-plane.png"]


def test_real_color_features_use_same_boolean_exclusion(library):
    from eidory.core.search_service import SearchService
    paths, store, images = library
    service = SearchService(store=store, embedding_provider=None, vector_index=None)
    result = FilterEngine(store, service).evaluate(FilterPlan(exclude=(FilterGroup((RED, PLANE)),)), images)
    assert visible_names(result) == {"red-car.png", "blue-plane.png", "blue-boat.png"}
    assert result.color_indexed_count == 4


def test_deleted_preset_collection_stays_empty(window):
    cid = window.store.create_collection("plane references")
    ids = [im.id for im in window.grid_view.images() if "plane" in im.file_name]
    window.store.assign_images_to_collection(ids, cid)
    window._refresh_collections(select_collection_id=cid)
    apply(window, FilterPlan(exclude=(FilterGroup((RED,)),)))
    payload = window._current_view_payload()
    assert payload["filter_scope"] == {"kind": "collections", "collection_ids": [cid]}
    window.store.delete_collection(cid)
    window._apply_view_payload(payload)
    window._poll_events()
    assert not window.grid_view.images()
    assert window.filter_workflow.base_context()[0] == set()


def test_multiple_folder_scope_and_tag_exclusion(window):
    images = window.grid_view.images()
    red = window.store.create_collection("red")
    blue = window.store.create_collection("blue")
    window.store.assign_images_to_collection([im.id for im in images if "red" in im.file_name], red)
    window.store.assign_images_to_collection([im.id for im in images if "blue" in im.file_name], blue)
    for im in images:
        if "plane" in im.file_name:
            window.store.set_image_tags(im.id, ["飞机"])
    tag_id = next(t.id for t in window.store.list_tags() if t.tag_name == "飞机")
    window.filter_workflow.scope = {"kind": "collections", "collection_ids": [red, blue]}
    plan = FilterPlan(exclude=(FilterGroup((RED, SearchFilter("tag", f"all:{tag_id}")), "all"),))
    apply(window, plan)
    assert {im.file_name for im in window.grid_view.images()} == {"red-car.png", "blue-plane.png", "blue-boat.png"}


def test_pending_query_keeps_completed_exclusions_and_disables_sort_and_save(window):
    apply(window, FilterPlan(exclude=(FilterGroup((RED, PLANE), "all"),)))
    before = {im.id for im in window.grid_view.images()}
    pending = []
    with patch.object(window, "_start_background_task", side_effect=lambda target, **kw: pending.append(target)):
        window.filter_workflow.apply(FilterPlan(exclude=(FilterGroup((RED, PLANE), "any"),)))
        assert not window.sort_combo.isEnabled()
        assert not window.save_result_set_button.isEnabled()
        window._on_sort_changed()
        assert {im.id for im in window.grid_view.images()} == before
        pending[0]()
        window._poll_events()
    assert len(window.grid_view.images()) == 1
    assert window.sort_combo.isEnabled()


def test_live_strength_broadens_from_full_scope_without_search_and_undoes_one_drag(window):
    semantic = SearchFilter("semantic", "plane", 95)
    plan = FilterPlan((FilterGroup((semantic,)),))
    apply(window, plan)
    assert len(window.grid_view.images()) == 1
    panel = window.filter_panel
    slider = panel.strength_sliders["include", 0, 0]
    history_length = len(window.filter_workflow.history)
    with patch.object(window.search_service, "semantic_search", side_effect=AssertionError("live adjustment must reuse scores")):
        slider.setSliderDown(True)
        slider.setValue(50)
        for _ in range(20):
            if len(window.grid_view.images()) == 3:
                break
            QTest.qWait(50)
        assert len(window.grid_view.images()) == 3
        assert slider.isSliderDown()
        assert panel.strength_sliders["include", 0, 0] is slider
        slider.setValue(0)
        for _ in range(20):
            if len(window.grid_view.images()) == 4:
                break
            QTest.qWait(50)
        assert len(window.grid_view.images()) == 4
        assert len(window.filter_workflow.history) == history_length + 1
        slider.setValue(85)
        slider.setSliderDown(False)
        assert len(window.grid_view.images()) == 1  # release flushes the last value
        payload = window._current_view_payload()
        assert FilterPlan.from_payload(payload["filter_plan"]).positive_filters[0].score_threshold == 85
    window.filter_workflow.undo()
    window._poll_events()
    assert window.filter_workflow.plan() == plan
    assert len(window.grid_view.images()) == 1


@pytest.mark.parametrize("mode, expected", [("all", (3, 2, 3)), ("any", (1, 0, 1))])
def test_live_exclusion_strength_preserves_group_logic(window, mode, expected):
    apply(window, FilterPlan(exclude=(FilterGroup((RED, PLANE), mode),)))
    assert len(window.grid_view.images()) == expected[0]
    slider = window.filter_panel.strength_sliders["exclude", 0, 0]
    with patch.object(window.search_service, "color_search", side_effect=AssertionError("no new color search")):
        slider.setSliderDown(True)
        slider.setValue(0)
        QTest.qWait(80)
        assert len(window.grid_view.images()) == expected[1]
        slider.setValue(100)
        slider.setSliderDown(False)
        assert len(window.grid_view.images()) == expected[2]
        assert "排除更少" in slider.toolTip()


def test_live_strength_keeps_other_cutoffs_and_new_draft_staged(window):
    semantic = SearchFilter("semantic", "plane", 95)
    plan = FilterPlan((FilterGroup((semantic, RED)),))
    apply(window, plan)
    slider = window.filter_panel.strength_sliders["include", 0, 0]
    slider.setValue(0)
    window.filter_workflow.finish_strength()
    assert len(window.grid_view.images()) == 2  # red condition still applies
    assert window.filter_workflow.plan().positive_filters[1] == RED
    window.filter_panel.put_condition("include", 0, PLANE)
    slider = window.filter_panel.strength_sliders["include", 0, 0]
    slider.setValue(100)
    QTest.qWait(80)
    assert len(window.grid_view.images()) == 2
    assert "正在准备更新" in window.filter_panel.status_label.text()
    QTest.mouseClick(window.filter_panel.apply_button, Qt.MouseButton.LeftButton)
    window._poll_events()
    assert len(window.grid_view.images()) == 1
    assert window.filter_workflow.plan().positive_filters[0].score_threshold == 100


def test_pending_live_adjustment_cannot_replay_after_scope_query_starts(window):
    plan = FilterPlan((FilterGroup((SearchFilter("semantic", "plane", 95),)),))
    apply(window, plan)
    slider = window.filter_panel.strength_sliders["include", 0, 0]
    slider.setValue(0)
    # A background refresh supersedes the cached query before the 40 ms tick.
    jobs = []
    with patch.object(window, "_start_background_task", side_effect=lambda target, **kw: jobs.append(target)):
        window._execute_search_chain(operation_mode="recompute")
        QTest.qWait(80)
        assert len(window.grid_view.images()) == 1
        assert not slider.isEnabled()
        jobs[0]()
        window._poll_events()
    assert window.current_chain_result.evaluated_plan == plan
    assert len(window.grid_view.images()) == 1


def test_two_filter_boxes_keep_each_condition_on_its_own_row(window):
    window.resize(2352, 1243)
    apply(window, FilterPlan(
        (FilterGroup((SearchFilter("semantic", "破旧墙壁", 70), RED)),),
        (FilterGroup((SearchFilter("keyword", "黑白摄影"),)),),
    ))
    panel = window.filter_panel
    for width in (2352, 1400):
        window.resize(width, 950)
        QTest.qWait(50)
        left, right = panel.section_boxes["include"], panel.section_boxes["exclude"]
        assert left.isVisible() and right.isVisible()
        assert left.geometry().right() < right.x()
        assert abs(left.width() - right.width()) <= 1
        assert left.y() == right.y()
        first, second = panel.condition_rows["include", 0, 0], panel.condition_rows["include", 0, 1]
        assert first.geometry().bottom() < second.y()
        assert first.width() == second.width()
        assert left.isAncestorOf(first)
        assert right.isAncestorOf(panel.condition_rows["exclude", 0, 0])
        for scroll in panel.section_scrolls.values():
            assert scroll.verticalScrollBar().maximum() == 0
        for slider in panel.strength_sliders.values():
            assert slider.isVisible() and slider.width() >= 80
            assert slider.geometry().right() < slider.parentWidget().width()
        assert panel.height() <= 200


def test_five_condition_rows_grow_and_shrink_without_padding_empty_exclusion_box(window):
    panel = window.filter_panel
    window.resize(1400, 900)
    panel.expand_button.click()
    QTest.qWait(50)
    empty_right_height = panel.section_boxes["exclude"].height()
    conditions = tuple(SearchFilter("semantic", f"condition {index}", 50) for index in range(5))
    panel.set_plan(FilterPlan((FilterGroup(conditions),)), force=True)
    QTest.qWait(50)
    five_height = panel.section_boxes["include"].height()
    assert len(panel.condition_rows) == 5
    assert panel.section_scrolls["include"].verticalScrollBar().maximum() == 0
    assert panel.section_boxes["exclude"].height() == empty_right_height
    for _ in range(4):
        panel.remove_condition("include", 0, 1)
    QTest.qWait(50)
    assert five_height - panel.section_boxes["include"].height() == 4 * 34
    # The permanently visible empty right box can receive its first condition.
    panel.put_condition("exclude", 0, RED)
    assert panel.draft.exclude[0].filters == (RED,)
    assert panel.section_boxes["exclude"].isAncestorOf(panel.condition_rows["exclude", 0, 0])
    panel.remove_condition("exclude", 0, 0)
    QTest.qWait(50)
    assert panel.section_boxes["exclude"].isVisible()
    assert panel.section_boxes["exclude"].height() == empty_right_height
