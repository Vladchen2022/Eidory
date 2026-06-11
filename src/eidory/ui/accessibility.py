from __future__ import annotations

import os
from typing import Callable


def disable_qt_accessibility(log_error: Callable[[str], None] | None = None) -> None:
    os.environ["QT_ACCESSIBILITY"] = "0"
    try:
        from PySide6.QtGui import QAccessible

        QAccessible.setActive(False)
    except Exception as exc:
        if log_error is not None:
            log_error(f"禁用 Qt Accessibility 失败：{exc}")
