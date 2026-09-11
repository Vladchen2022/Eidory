from __future__ import annotations

import os
import sys
from typing import Callable

_item_view_factory = None
_item_view_interfaces = []


def configure_qt_accessibility(log_error: Callable[[str], None] | None = None) -> None:
    """Keep controls accessible while avoiding Cocoa's unsafe table selection scan.

    A deep AX hierarchy scan of Qt item views can return stale selected-cell
    interfaces on macOS. Expose these custom visual lists as a single group;
    buttons, editors, dialogs and top-level windows keep their normal interfaces.
    """
    # Headless render/test processes have no native accessibility consumer.
    # Activating Qt's table cache there leaves stale cell interfaces when test
    # windows are repeatedly cleared and destroyed.
    headless = os.environ.get("QT_QPA_PLATFORM", "").split(":", 1)[0] in {"offscreen", "minimal"}
    os.environ["QT_ACCESSIBILITY"] = "0" if headless else "1"
    _append_qt_logging_rule("qt.accessibility.*=false")
    try:
        from PySide6.QtGui import QAccessible
        if sys.platform == "darwin" and os.environ.get("QT_QPA_PLATFORM", "cocoa") == "cocoa":
            _install_item_view_factory()
        QAccessible.setActive(not headless)
    except Exception as exc:
        if log_error:
            log_error(f"配置 Qt Accessibility 失败：{exc}")


def _install_item_view_factory():
    global _item_view_factory
    if _item_view_factory is not None:
        return
    from PySide6.QtGui import QAccessible
    from PySide6.QtWidgets import QAbstractItemView, QAccessibleWidget

    class ItemViewGroup(QAccessibleWidget):
        def __init__(self, widget):
            super().__init__(widget, QAccessible.Role.Grouping)

        def childCount(self):
            return 0

        def child(self, index):
            return None

        def focusChild(self):
            return None

        def indexOfChild(self, child):
            return -1

    def factory(_class_name, obj):
        if isinstance(obj, QAbstractItemView) or type(obj).__name__ in {"JustifiedImageGridView", "ProjectBoardView"}:
            interface = ItemViewGroup(obj)
            # The Qt accessibility cache owns the C++ interface. Retain its
            # Python virtual-method wrapper for the same application lifetime.
            _item_view_interfaces.append(interface)
            return interface
        return None

    _item_view_factory = factory
    QAccessible.installFactory(factory)


def _append_qt_logging_rule(rule: str) -> None:
    existing = os.environ.get("QT_LOGGING_RULES", "").strip()
    rules = [part.strip() for part in existing.split(";") if part.strip()]
    if rule not in rules:
        os.environ["QT_LOGGING_RULES"] = ";".join([*rules, rule])


# Compatibility for extensions and older test helpers. Do not disable the whole
# backend: macOS may keep querying elements it has already discovered.
disable_qt_accessibility = configure_qt_accessibility


def hide_macos_accessibility_tree(widget: object, log_error=None) -> bool:
    return False
