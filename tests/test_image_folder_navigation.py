from unittest.mock import patch

import pytest
from PySide6.QtCore import QPoint

from test_filter_workflow import app, library, window
from eidory.core.search_filters import FilterGroup, FilterPlan, SearchFilter


def file_action(menu):
    submenu = next(child for child in menu._eidory_submenus if child.title() == "文件与导出")
    return next(action for action in submenu.actions() if action.data() == "enter_collection")


def test_context_command_opens_folder_and_locates_later_page(window, library):
    _, store, images = library
    parent = store.create_collection("作品")
    child = store.create_collection("角色", parent_id=parent)
    store.assign_images_to_collection([im.id for im in images], child)
    target = sorted(images, key=lambda im: im.file_name)[-1]
    # Redundant parent membership must not require a folder chooser.
    store.assign_images_to_collection([target.id], parent)
    window._refresh_collections()
    window.page_size = 2
    window._set_combo_to_data(window.sort_combo, "name")
    window._set_combo_to_data(window.sort_order_combo, "asc")
    window.filter_workflow.set_plan(FilterPlan(
        (FilterGroup((SearchFilter("keyword", "not-in-this-folder"),)),),
        (FilterGroup((SearchFilter("keyword", "red"),)),),
    ))
    window.filter_workflow.scope = {"kind": "snapshot", "image_ids": [target.id], "label": "固定结果"}
    window.result_excluded_image_ids.add(target.id)
    window._set_combo_to_data(window.status_filter_combo, "favorite")
    window.grid_view.set_images([target], selected_image_ids=[target.id])
    menu = window._build_grid_context_menu(selected_images=[target], context_image=target)
    action = file_action(menu)
    with patch.object(window, "_build_grid_context_menu", return_value=menu), \
            patch.object(menu, "exec", return_value=action), \
            patch("eidory.ui.main_window.QInputDialog.getItem") as choose:
        window._show_grid_context_menu(target, QPoint())
    choose.assert_not_called()
    assert window._selected_collection_id() == child
    assert not window.search_filters and not window.result_exclusion_filters
    assert not window.result_excluded_image_ids
    assert window.filter_workflow.scope == {"kind": "sidebar"}
    assert window.grid_view.selected_image_ids() == [target.id]
    assert [im.file_name for im in window.grid_view.images()] == sorted(im.file_name for im in images)
    assert window.center_result_stack.currentWidget() is window.grid_view
    assert window.workspace_chrome.navigation_tabs.currentIndex() == 0
    assert window.collection_tree.currentItem().parent().isExpanded()


def test_uncategorized_and_multi_folder_choice_cancel(window, library):
    _, store, images = library
    target = images[0]
    window._enter_image_collection(target)
    assert window._selected_virtual_filter() == "uncategorized"
    assert window.grid_view.selected_image_ids() == [target.id]
    first = store.create_collection("参考一")
    second = store.create_collection("参考二")
    store.assign_images_to_collection([target.id], first)
    store.assign_images_to_collection([target.id], second)
    window._refresh_collections()
    before = window._current_view_payload()
    with patch("eidory.ui.main_window.QInputDialog.getItem", return_value=("", False)):
        window._enter_image_collection(target)
    assert window._current_view_payload() == before
    with patch("eidory.ui.main_window.QInputDialog.getItem", return_value=("参考二", True)):
        window._enter_image_collection(target)
    assert window._selected_collection_id() == second
    assert window.grid_view.selected_image_ids() == [target.id]


def test_folder_action_requires_a_single_image(window, library):
    images = library[2]
    for selection, context, enabled in [([], None, False), ([images[0]], images[0], True), (images[:2], images[0], False)]:
        menu = window._build_grid_context_menu(selected_images=selection, context_image=context)
        assert file_action(menu).isEnabled() == enabled


@pytest.mark.parametrize("sort_key", ["name", "imported", "modified", "width", "height", "pixels", "file_size", "duration", "default"])
@pytest.mark.parametrize("descending", [False, True])
def test_folder_position_matches_gallery_order(library, sort_key, descending):
    _, store, images = library
    folder = store.create_collection("测试目录")
    store.assign_images_to_collection([im.id for im in images[:3]], folder)
    for collection_id, virtual_filter in [(folder, None), (None, "uncategorized")]:
        ordered = store.list_images(collection_id=collection_id, virtual_filter=virtual_filter,
                                    sort_key=sort_key, sort_desc=descending, limit=None)
        for index, image in enumerate(ordered):
            assert store.image_position_in_collection(image.id, collection_id,
                                                      sort_key=sort_key, sort_desc=descending) == index
    assert store.image_position_in_collection(images[-1].id, folder) is None
