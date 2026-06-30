from __future__ import annotations

import multiprocessing
import os
import sys

os.environ["QT_ACCESSIBILITY"] = "0"

from eidory.config import AppPaths
from eidory.core.metadata_store import MetadataStore


def main() -> int:
    multiprocessing.freeze_support()

    from eidory.ui.accessibility import disable_qt_accessibility

    disable_qt_accessibility()

    from PySide6.QtWidgets import QApplication

    from eidory.ui.main_window import MainWindow
    from eidory.ui.theme import apply_dark_theme

    paths = AppPaths.default()
    paths.ensure()
    store = MetadataStore(paths.database_path)
    store.initialize()

    app = QApplication(sys.argv)
    disable_qt_accessibility()
    app.setApplicationName("Eidory")
    apply_dark_theme(app)
    window = MainWindow(paths=paths, store=store)
    window.resize(
        _setting_int(store, "ui.window_width", 1320, 900, 2400),
        _setting_int(store, "ui.window_height", 860, 600, 1800),
    )
    window.show()
    return app.exec()


def _setting_int(
    store: MetadataStore,
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = store.get_setting(key)
    try:
        value = int(raw) if raw is not None else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
