from PySide6.QtGui import QColor, QPalette
from PySide6.QtTest import QSignalSpy
import pytest
import time

from test_filter_workflow import app, library, window, apply
from eidory.core.search_filters import FilterGroup, FilterPlan, SearchFilter
from eidory.ui.theme import apply_theme
from eidory.ui.theme_colors import THEMES, DEFAULT_THEME, get_theme, theme_color


def contrast(a, b):
    def luminance(hex_color):
        rgb = QColor(hex_color).getRgbF()[:3]
        linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgb]
        return sum(v * w for v, w in zip(linear, (.2126, .7152, .0722)))
    light, dark = sorted((luminance(a), luminance(b)), reverse=True)
    return (light + .05) / (dark + .05)


@pytest.mark.parametrize("theme", THEMES, ids=lambda t: t.id)
def test_theme_text_and_controls_have_readable_contrast(theme):
    c = theme.colors
    for foreground, background in (("text", "window"), ("text", "base"), ("muted", "raised"),
                                   ("accent_text", "accent"), ("selection_text", "selection"),
                                   ("warning", "raised"), ("text", "canvas"), ("link", "raised")):
        assert contrast(c[foreground], c[background]) >= 4.5, (theme.id, foreground, background)


def test_theme_catalog_has_five_choices_and_safe_fallback():
    assert len(THEMES) == len({t.id for t in THEMES}) == 5
    assert sum(not t.dark for t in THEMES) == 2
    assert get_theme("removed-or-corrupt-theme").id == DEFAULT_THEME


def test_live_theme_changes_preserve_work_and_update_existing_content(window, app):
    w = window
    apply_theme(app, DEFAULT_THEME)
    w.store.add_tags_to_images([image.id for image in w.store.list_images(limit=None)], ["材质"])
    plan = FilterPlan(include=(FilterGroup((SearchFilter("color", (255, 0, 0), 50),)),))
    apply(w, plan)
    images = w.grid_view.images()
    w.grid_view.select_image_ids([images[0].id])
    w.project_board_view.set_images(images, title="主题切换测试", layout_payload={
        "notes": [{"text": "保留这个构图", "x": 75, "y": 95, "heading": True}],
    })
    w._set_tags_display(["材质"])
    app.processEvents()
    board = w.project_board_view
    layout = board.layout_payload()
    history = (list(board._undo_layouts), list(board._redo_layouts))
    selected = w.grid_view.selected_image_ids()
    width = w.root_splitter.sizes()
    changes = QSignalSpy(board.layoutChanged)
    try:
        for theme in THEMES:
            w.workspace_chrome.theme_combo.setCurrentIndex(w.workspace_chrome.theme_combo.findData(theme.id))
            app.processEvents()
            deadline = time.monotonic() + 3
            while w.tags_display.item(0).text() != "材质" and time.monotonic() < deadline:
                time.sleep(.01)  # Release Python's GIL for the metadata worker.
                app.processEvents()
            assert w.filter_workflow.plan() == plan
            assert w.grid_view.selected_image_ids() == selected
            assert w.root_splitter.sizes() == width
            assert board.layout_payload() == layout
            assert (board._undo_layouts, board._redo_layouts) == history
            assert board._note_items[0].defaultTextColor() == theme_color("warning")
            assert w.tags_display.item(0).text() == "材质"
            assert w.tags_display.item(0).foreground().color() == theme_color("text")
            assert board.backgroundBrush().color() == theme_color("canvas")
            assert app.palette().color(QPalette.ColorRole.Base) == theme_color("base")
            assert "@raised@" not in w.filter_panel.styleSheet()
            assert theme.colors["raised"] in w.filter_panel.styleSheet()
        assert changes.count() == 0
    finally:
        apply_theme(app, DEFAULT_THEME)


def test_theme_preference_survives_density_change_and_startup(window, app):
    chrome = window.workspace_chrome
    try:
        chrome.theme_combo.setCurrentIndex(chrome.theme_combo.findData("paper"))
        chrome.density_combo.setCurrentIndex(1)
        assert window.store.get_setting("ui.theme") == "paper"
        assert window.store.get_setting("ui.density") == "comfortable"
        assert get_theme().id == "paper"
        # Same persisted inputs as the application startup path.
        apply_theme(app, DEFAULT_THEME)
        apply_theme(app, window.store.get_setting("ui.theme"), density=window.store.get_setting("ui.density"))
        assert get_theme().id == "paper"
        assert app.palette().color(QPalette.ColorRole.Text) == QColor(get_theme("paper").colors["text"])
    finally:
        apply_theme(app, DEFAULT_THEME)
