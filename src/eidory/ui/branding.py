"""Application artwork shared by the window and packaged application."""
from pathlib import Path

from PySide6.QtGui import QIcon


APP_ICON_PATH = Path(__file__).with_name("icons") / "app_icon.png"


def app_icon() -> QIcon:
    return QIcon(str(APP_ICON_PATH))
