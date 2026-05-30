from __future__ import annotations

from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication


def apply_dark_theme(app: QApplication) -> None:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#34373d"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e6e8eb"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#2d3138"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#3a3f47"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#3a3f47"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#f0f2f5"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e6e8eb"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#444951"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#f0f2f5"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#4f7cff"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(
        """
        QWidget {
            background-color: #34373d;
            color: #e6e8eb;
            selection-background-color: #4f7cff;
            selection-color: #ffffff;
        }
        QLineEdit, QTextEdit, QListWidget, QTreeWidget, QScrollArea {
            background-color: #2d3138;
            color: #e6e8eb;
            border: 1px solid #555b65;
            border-radius: 4px;
        }
        QLineEdit:focus, QTextEdit:focus, QListWidget:focus, QTreeWidget:focus {
            border: 1px solid #5b8cff;
        }
        QPushButton {
            background-color: #444951;
            color: #f0f2f5;
            border: 1px solid #626976;
            border-radius: 4px;
            padding: 4px 10px;
        }
        QPushButton:hover {
            background-color: #4d535d;
        }
        QPushButton:pressed, QPushButton:checked {
            background-color: #4f7cff;
            border-color: #6d95ff;
            color: #ffffff;
        }
        QPushButton:disabled {
            background-color: #383d45;
            color: #7f858f;
            border-color: #444951;
        }
        QHeaderView::section {
            background-color: #3a3f47;
            color: #e6e8eb;
            border: 0;
            border-bottom: 1px solid #555b65;
            padding: 3px 4px;
        }
        QListWidget::item:selected, QTreeWidget::item:selected {
            background-color: #4f7cff;
            color: #ffffff;
        }
        QMenu {
            background-color: #3a3f47;
            color: #e6e8eb;
            border: 1px solid #5a616c;
        }
        QMenu::item:selected {
            background-color: #4f7cff;
            color: #ffffff;
        }
        QProgressBar {
            background-color: #2d3138;
            color: #e6e8eb;
            border: 1px solid #555b65;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4f7cff;
            border-radius: 3px;
        }
        QSlider::groove:horizontal {
            height: 5px;
            background: #555b65;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #d8dee9;
            border: 1px solid #6a7280;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QStatusBar {
            background-color: #2d3138;
            color: #cfd3da;
        }
        QTabWidget::pane {
            border: 1px solid #5b6370;
            border-radius: 5px;
            top: -1px;
        }
        QTabBar::tab {
            background-color: #424852;
            color: #dfe4ec;
            border: 1px solid #616a78;
            border-bottom-color: #5b6370;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            min-height: 26px;
            min-width: 0;
            padding: 6px 4px;
            margin-right: 1px;
        }
        QTabBar::tab:hover {
            background-color: #4d5662;
            color: #ffffff;
        }
        QTabBar::tab:selected {
            background-color: #5a6370;
            color: #ffffff;
            border-color: #7b8796;
            border-bottom-color: #5a6370;
        }
        """
    )
