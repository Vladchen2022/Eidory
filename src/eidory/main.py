from __future__ import annotations

import multiprocessing
import sys
import time

from eidory.config import AppPaths
from eidory.core.metadata_store import MetadataStore


def main() -> int:
    multiprocessing.freeze_support()
    started = time.perf_counter()
    paths = AppPaths.default()
    paths.ensure()
    from eidory.core.diagnostics import setup_diagnostics, watch_event_loop
    logger = setup_diagnostics(paths.log_dir)

    from eidory.ui.accessibility import configure_qt_accessibility

    configure_qt_accessibility()

    from PySide6.QtWidgets import QApplication

    from eidory.ui.main_window import MainWindow
    from eidory.ui.branding import app_icon
    from eidory.ui.theme import apply_theme
    from eidory.ui.theme_colors import DEFAULT_THEME

    store = MetadataStore(paths.database_path)
    store.initialize()
    from eidory.core.image_upscale import ImageReplacement
    ImageReplacement(store, paths).recover()

    app = QApplication(sys.argv)
    configure_qt_accessibility()
    app.setApplicationName("Eidory")
    app.setWindowIcon(app_icon())
    apply_theme(app, store.get_setting("ui.theme") or DEFAULT_THEME, density=store.get_setting("ui.density") or "compact", font_size=_setting_int(store, "ui.font_size", 13, 11, 20))
    window = MainWindow(paths=paths, store=store)
    window.resize(
        _setting_int(store, "ui.window_width", 1320, 900, 2400),
        _setting_int(store, "ui.window_height", 860, 600, 1800),
    )
    window.show()
    watch_event_loop(window, logger)
    logger.info('Window shown in %.0f ms', (time.perf_counter()-started)*1000)
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
