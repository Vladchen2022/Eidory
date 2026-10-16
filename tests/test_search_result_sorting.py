"""Data-order regressions without starting an application or model service."""
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from eidory.models import ImageItem
from eidory.core.search_filters import SearchFilter
from eidory.ui.main_window import MainWindow
from eidory.ui.workflow_upgrade import WorkflowUpgrade


@pytest.fixture
def images():
    base = ImageItem(3, 1, '/c.jpg', 'c.jpg', '.jpg', 400, 400, 100,
                     None, None, 40, '2026-01-03', '', None, 'ready', 'ready',
                     False, False, None, 40, .2)
    return [base,
            replace(base, id=2, file_name='a.jpg', file_size=100, width=200,
                    height=300, modified_time_ns=10, imported_at='2026-01-01', duration_ms=10, score=.9),
            replace(base, id=1, file_name='b.jpg', file_size=300, width=200,
                    height=200, modified_time_ns=30, imported_at='2026-01-02', duration_ms=30, score=.4),
            replace(base, id=4, file_name='Z.jpg', file_size=200, width=None,
                    height=None, modified_time_ns=20, imported_at='2026-02-01', duration_ms=None, score=None)]


def presenter(images, key='width', desc=False, target=None, semantic=False, requested=False):
    grid = Mock()
    grid.selected_image_ids.return_value = [2]
    condition = SearchFilter('semantic', 'target')
    result = SimpleNamespace(ranked_scope_images=list(reversed(images)),
                             filter_matches={condition: images} if semantic else {})
    w = SimpleNamespace(current_result_mode='search_chain', grid_view=grid,
                        current_chain_filtered_images=images, current_chain_result=result,
                        current_sort_key=key, current_sort_desc=desc, manual_result_order_ids=None,
                        current_creative_project_id=None, store=Mock(), simple_search=Mock(),
                        _current_creative_node=lambda: None, _sort_value=MainWindow._sort_value)
    w._sort_images = partial(MainWindow._sort_images, w)
    w._apply_manual_result_order = partial(MainWindow._apply_manual_result_order, w)
    w.simple_search.sort.side_effect = lambda rows: list(reversed(rows))
    upgrade = SimpleNamespace(w=w, _toolbar_sort_requested=requested,
                              result_mode=SimpleNamespace(currentData=lambda: target),
                              match_ids={'target': {im.id for im in images}} if semantic else {},
                              match_badges={}, decorate_references=Mock(), result_hint=Mock())
    return upgrade


@pytest.mark.parametrize('key,asc,desc', [
    ('default', [3,2,1,4], [3,2,1,4]),
    ('score', [3,1,2,4], [2,1,3,4]),
    ('imported', [2,1,3,4], [4,3,1,2]),
    ('modified', [2,4,1,3], [3,1,4,2]),
    ('name', [2,1,3,4], [4,3,1,2]),
    ('file_size', [2,4,1,3], [3,1,4,2]),
    ('width', [1,2,3,4], [3,2,1,4]),
    ('height', [3,1,2,4], [2,1,3,4]),
    ('pixels', [1,3,2,4], [2,3,1,4]),
    ('duration', [2,1,3,4], [3,1,2,4]),
])
@pytest.mark.parametrize('descending', [False, True])
def test_metadata_sort_survives_nonsemantic_result_presentation(images, key, asc, desc, descending):
    upgrade = presenter(images, key, descending, requested=True)
    WorkflowUpgrade.display_results(upgrade)
    shown = upgrade.w.grid_view.set_images.call_args.args[0]
    assert [im.id for im in shown] == (desc if descending else asc)
    upgrade.w.simple_search.sort.assert_not_called()
    upgrade.w.grid_view.select_image_ids.assert_called_once_with([2])


@pytest.mark.parametrize('descending,expected', [(False, [2,3]), (True, [3,2])])
def test_sort_keeps_single_target_membership(images, descending, expected):
    upgrade = presenter(images, desc=descending, target='target', semantic=True, requested=True)
    upgrade.match_ids['target'] = {2,3}
    WorkflowUpgrade.display_results(upgrade)
    assert [im.id for im in upgrade.w.grid_view.set_images.call_args.args[0]] == expected
    upgrade.w.simple_search.sort.assert_not_called()


def test_target_similarity_sort_uses_that_targets_scores(images):
    upgrade = presenter(images, key='score', target='target', semantic=True, requested=True)
    condition = next(iter(upgrade.w.current_chain_result.filter_matches))
    upgrade.w.current_chain_result.filter_matches[condition] = [replace(im, score=1-im.score if im.score is not None else None) for im in images]
    WorkflowUpgrade.display_results(upgrade)
    assert [im.id for im in upgrade.w.grid_view.set_images.call_args.args[0]] == [2,1,3,4]


def test_manual_shuffle_is_not_overwritten_by_recommendations(images):
    upgrade = presenter(images, semantic=True)
    upgrade.w.manual_result_order_ids = [2,4,3,1]
    WorkflowUpgrade.display_results(upgrade)
    assert [im.id for im in upgrade.w.grid_view.set_images.call_args.args[0]] == [2,4,3,1]


def test_recommendation_and_feedback_remain_available(images):
    upgrade = presenter(images, semantic=True)
    WorkflowUpgrade.display_results(upgrade)
    upgrade.w.simple_search.sort.assert_called_once()
    assert [im.id for im in upgrade.w.simple_search.sort.call_args.args[0]] == [4,1,2,3]


def test_toolbar_order_ignores_old_preference_ranking(images):
    upgrade = presenter(images, target='__toolbar__', semantic=True)
    WorkflowUpgrade.display_results(upgrade)
    assert [im.id for im in upgrade.w.grid_view.set_images.call_args.args[0]] == [1,2,3,4]
    upgrade.w.simple_search.sort.assert_not_called()


def test_folder_only_search_selects_toolbar_order_on_completion(images):
    upgrade = presenter(images)
    entries = []
    selected = [0]
    upgrade.result_mode = SimpleNamespace(
        currentData=lambda: entries[selected[0]][1] if entries else None,
        blockSignals=lambda _: False, clear=entries.clear,
        addItem=lambda label, value: entries.append((label, value)),
        findData=lambda value: next((i for i, row in enumerate(entries) if row[1] == value), -1),
        setCurrentIndex=lambda index: selected.__setitem__(0, index))
    upgrade.w.current_chain_result.evaluated_plan = SimpleNamespace(positive_filters=[SearchFilter('collection', (1,2,3,4,5,6,7))])
    upgrade.display_results = partial(WorkflowUpgrade.display_results, upgrade)
    upgrade.refresh = Mock()
    WorkflowUpgrade.search_finished(upgrade)
    assert upgrade.result_mode.currentData() == '__toolbar__'
    assert [im.id for im in upgrade.w.grid_view.set_images.call_args.args[0]] == [1,2,3,4]


def test_folder_only_results_can_still_use_explicit_image_preferences(images):
    upgrade = presenter(images)
    WorkflowUpgrade.display_results(upgrade)
    upgrade.w.simple_search.sort.assert_called_once()


def test_toolbar_request_cancels_pending_preference_without_changing_target(images):
    upgrade = presenter(images, target='target', semantic=True)
    upgrade.w.simple_search.serial = 7
    WorkflowUpgrade.request_toolbar_sort(upgrade)
    assert upgrade._toolbar_sort_requested
    assert upgrade.w.simple_search.serial == 8
    assert upgrade.result_mode.currentData() == 'target'
