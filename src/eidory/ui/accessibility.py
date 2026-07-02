from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from typing import Callable


def disable_qt_accessibility(log_error: Callable[[str], None] | None = None) -> None:
    # Keep the public helper for older call sites, but do not force-disable Qt AX.
    # Disabling/hiding Qt's Cocoa accessibility tree can leave AppKit serving
    # stale accessibility objects to external tools and has caused native crashes.
    os.environ.pop("QT_ACCESSIBILITY", None)


def hide_macos_accessibility_tree(
    widget: object,
    log_error: Callable[[str], None] | None = None,
) -> bool:
    """Hide the Cocoa/Qt accessibility subtree from deep external hierarchy scans."""

    if sys.platform != "darwin":
        return False
    objc_path = ctypes.util.find_library("objc")
    if not objc_path:
        return False
    try:
        native_view = int(widget.winId())  # type: ignore[attr-defined]
    except Exception as exc:
        if log_error is not None:
            log_error(f"获取 macOS 原生视图失败：{exc}")
        return False
    if not native_view:
        return False
    try:
        return _hide_macos_accessibility_tree_for_native_view(native_view, objc_path)
    except Exception as exc:
        if log_error is not None:
            log_error(f"隐藏 macOS Accessibility 子树失败：{exc}")
        return False


def _hide_macos_accessibility_tree_for_native_view(native_view: int, objc_path: str) -> bool:
    objc = ctypes.cdll.LoadLibrary(objc_path)
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]

    send_id = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc)
    )
    send_bool_with_id = ctypes.CFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )(("objc_msgSend", objc))
    send_void_bool = ctypes.CFUNCTYPE(
        None,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_bool,
    )(("objc_msgSend", objc))

    def sel(name: str) -> int:
        return int(objc.sel_registerName(name.encode("utf-8")) or 0)

    def cls(name: str) -> int:
        return int(objc.objc_getClass(name.encode("utf-8")) or 0)

    responds_to_selector = sel("respondsToSelector:")

    def responds(receiver: int, selector_name: str) -> bool:
        selector = sel(selector_name)
        return bool(
            receiver
            and selector
            and send_bool_with_id(
                ctypes.c_void_p(receiver),
                ctypes.c_void_p(responds_to_selector),
                ctypes.c_void_p(selector),
            )
        )

    def send_id_message(receiver: int, selector_name: str) -> int:
        if not receiver or not responds(receiver, selector_name):
            return 0
        return int(
            send_id(
                ctypes.c_void_p(receiver),
                ctypes.c_void_p(sel(selector_name)),
            )
            or 0
        )

    def send_bool_message(receiver: int, selector_name: str, value: bool) -> bool:
        if not receiver or not responds(receiver, selector_name):
            return False
        send_void_bool(
            ctypes.c_void_p(receiver),
            ctypes.c_void_p(sel(selector_name)),
            ctypes.c_bool(value),
        )
        return True

    window = send_id_message(native_view, "window")
    content_view = send_id_message(window, "contentView") if window else 0
    ns_application = send_id_message(cls("NSApplication"), "sharedApplication")
    targets = (ns_application, window, content_view, native_view)
    applied = False
    for target in targets:
        applied = send_bool_message(target, "setAccessibilityElementsHidden:", True) or applied
        applied = send_bool_message(target, "setAccessibilityElement:", False) or applied
    return applied
