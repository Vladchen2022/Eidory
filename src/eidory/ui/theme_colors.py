"""Shared UI colors. Image pixels and user-chosen swatches are never themed."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import QApplication, QListWidget, QTreeWidget, QTreeWidgetItemIterator, QWidget


@dataclass(frozen=True)
class Theme:
    id: str
    name: str
    english_name: str
    dark: bool
    colors: Mapping[str, str]


def _theme(id, name, english_name, dark, **colors):
    colors.setdefault("marker", colors["link"])
    return Theme(id, name, english_name, dark, MappingProxyType(colors))


THEMES = (
    _theme("graphite", "石墨深色", "Graphite", True,
           window="#242629", base="#202124", raised="#282a2e", button="#303237",
           hover="#3a3d43", border="#4e5159", text="#e5e5e7", muted="#adb0b9", disabled="#858892",
           accent="#4b76ba", accent_hover="#5884c8", accent_pressed="#3d649f", accent_text="#ffffff",
           selection="#384b68", selection_text="#ecf2fc", link="#a8c7ff", warning="#edbd89", warning_fill="#ad7840",
           canvas="#222427", grid="#292c31", grid_major="#353940", marker="#4f83ff"),
    _theme("light", "明亮浅色", "Daylight", False,
           window="#f3f5f8", base="#ffffff", raised="#e9edf3", button="#e7ecf3",
           hover="#dce4ee", border="#b1bdcd", text="#202b3a", muted="#566477", disabled="#728094",
           accent="#2d63bb", accent_hover="#2455a3", accent_pressed="#1c4487", accent_text="#ffffff",
           selection="#dbe8ff", selection_text="#173e75", link="#2455a3", warning="#865020", warning_fill="#a96525",
           canvas="#eef1f5", grid="#e4e8ee", grid_major="#ccd4df"),
    _theme("paper", "暖纸米白", "Warm Paper", False,
           window="#f3eee4", base="#fbf8f1", raised="#eae2d5", button="#e9dfd0",
           hover="#ded1bd", border="#bdb09b", text="#3a3228", muted="#6d6050", disabled="#8c7e6c",
           accent="#96582f", accent_hover="#7f4826", accent_pressed="#693b20", accent_text="#ffffff",
           selection="#e9d7bd", selection_text="#5c381f", link="#80502d", warning="#7c5220", warning_fill="#a16a28",
           canvas="#f2ecdf", grid="#e9e1d2", grid_major="#d8ccb8"),
    _theme("midnight", "午夜蓝", "Midnight Blue", True,
           window="#202a3b", base="#192232", raised="#29354a", button="#2e3c53",
           hover="#3a4c69", border="#536680", text="#e3eaf5", muted="#aebfd5", disabled="#8091aa",
           accent="#4e73b2", accent_hover="#5c82c2", accent_pressed="#3c5d95", accent_text="#ffffff",
           selection="#364e73", selection_text="#e4eeff", link="#a6c7ff", warning="#eac493", warning_fill="#ad8048",
           canvas="#1b2535", grid="#253247", grid_major="#36465f"),
    _theme("forest", "森林绿", "Forest", True,
           window="#24312c", base="#1c2723", raised="#2c3b34", button="#31453b",
           hover="#3d5648", border="#546f5f", text="#e3ede5", muted="#b0c4b5", disabled="#819c8b",
           accent="#32765a", accent_hover="#3b8968", accent_pressed="#275c46", accent_text="#ffffff",
           selection="#355744", selection_text="#e4f6e9", link="#a1d8b5", warning="#e3c493", warning_fill="#a37f47",
           canvas="#202c26", grid="#2a3a30", grid_major="#3d5143"),
)
DEFAULT_THEME = THEMES[0].id


def get_theme(theme_id: str | None = None) -> Theme:
    if theme_id is None:
        app = QApplication.instance()
        theme_id = app.property("eidoryTheme") if app else None
    return next((theme for theme in THEMES if theme.id == theme_id), THEMES[0])


def theme_color(token: str) -> QColor:
    theme = get_theme()
    if token.startswith("level_"):
        level = int(token.split("_")[1])
        start, end = QColor(theme.colors["raised"]), QColor(theme.colors["base"])
        ratio = max(0, min(5, level)) / 6
        return QColor(*(round(a + (b - a) * ratio) for a, b in zip(start.getRgb()[:3], end.getRgb()[:3])))
    return QColor(theme.colors[token])


def resolve_theme_style(template: str) -> str:
    for key, value in get_theme().colors.items():
        template = template.replace("@" + key + "@", value)
    return template


def set_theme_style(widget, template: str) -> None:
    """Retain tokens so local styles also update in already-open dialogs."""
    widget.setProperty("eidoryThemeStyle", template)
    widget.setStyleSheet(resolve_theme_style(template))


def set_theme_rich_text(label, template: str) -> None:
    label.setProperty("eidoryThemeRichText", template)
    label.setText(resolve_theme_style(template))


ITEM_THEME_ROLE = int(Qt.ItemDataRole.UserRole) + 9000


def set_item_theme(item, *, foreground=None, background=None, column=None):
    """Mark only UI-owned brushes; never recolor user-defined labels."""
    tokens = item.data(ITEM_THEME_ROLE) if column is None else item.data(column, ITEM_THEME_ROLE)
    tokens = dict(tokens or {})
    if foreground is not None:
        tokens["foreground"] = foreground
    if background is not None:
        tokens["background"] = background
    if column is None:
        item.setData(ITEM_THEME_ROLE, tokens)
    else:
        item.setData(column, ITEM_THEME_ROLE, tokens)
    _apply_item_theme(item, tokens, column)


def _apply_item_theme(item, tokens, column):
    for kind, token in (tokens or {}).items():
        setter = item.setForeground if kind == "foreground" else item.setBackground
        brush = QBrush(theme_color(token))
        setter(brush) if column is None else setter(column, brush)


def refresh_themed_widgets(app):
    for widget in app.allWidgets():
        template = widget.property("eidoryThemeStyle")
        if template:
            widget.setStyleSheet(resolve_theme_style(template))
        rich_text = widget.property("eidoryThemeRichText")
        if rich_text:
            widget.setText(resolve_theme_style(rich_text))
        if isinstance(widget, QListWidget):
            for row in range(widget.count()):
                item = widget.item(row)
                _apply_item_theme(item, item.data(ITEM_THEME_ROLE), None)
        elif isinstance(widget, QTreeWidget):
            iterator = QTreeWidgetItemIterator(widget)
            while iterator.value():
                item = iterator.value()
                for column in range(item.columnCount()):
                    _apply_item_theme(item, item.data(column, ITEM_THEME_ROLE), column)
                iterator += 1
        refresh = getattr(widget, "refresh_theme", None)
        if refresh:
            refresh()
        QWidget.update(widget)
