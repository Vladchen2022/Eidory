from __future__ import annotations

from PySide6.QtGui import QFont, QFontDatabase, QPalette, QColor
from PySide6.QtWidgets import QApplication


def apply_dark_theme(app: QApplication) -> None:
    families = set(QFontDatabase.families())
    for family in (".AppleSystemUIFont", "Helvetica Neue", "Arial"):
        if family in families:
            app.setFont(QFont(family))
            break
    else:
        app.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont))
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#2f333a"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e8ecf2"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#242a31"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#2b323b"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#303842"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#f1f4f8"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e8ecf2"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#3a424d"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#f1f4f8"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#4f83ff"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(
        """
        QWidget {
            background-color: #2f333a;
            color: #e8ecf2;
            selection-background-color: #4f83ff;
            selection-color: #ffffff;
        }
        QMainWindow, QDialog {
            background-color: #2f333a;
        }
        QSplitter::handle {
            background-color: #222832;
        }
        QSplitter::handle:horizontal {
            width: 1px;
        }
        QSplitter::handle:vertical {
            height: 1px;
        }
        QLineEdit, QTextEdit, QListWidget, QTreeWidget, QTableWidget, QScrollArea {
            background-color: #242a31;
            border: 1px solid #47515f;
            border-radius: 5px;
            color: #e8ecf2;
        }
        QLineEdit, QTextEdit {
            padding: 4px 7px;
        }
        QListWidget, QTreeWidget, QTableWidget {
            alternate-background-color: #2a3139;
            outline: 0;
        }
        QLineEdit:focus, QTextEdit:focus, QListWidget:focus, QTreeWidget:focus {
            border: 1px solid #6a98ff;
        }
        QComboBox {
            background-color: #343b45;
            color: #f1f4f8;
            border: 1px solid #566171;
            border-radius: 5px;
            min-height: 24px;
            padding: 2px 24px 2px 8px;
        }
        QComboBox:hover {
            background-color: #3d4651;
            border-color: #6a7483;
        }
        QComboBox:focus {
            border-color: #6a98ff;
        }
        QComboBox:disabled {
            background-color: #30363f;
            color: #858d99;
            border-color: #424b57;
        }
        QComboBox::drop-down {
            border: 0;
            width: 22px;
        }
        QPushButton {
            background-color: #3a424d;
            color: #f1f4f8;
            border: 1px solid #566171;
            border-radius: 5px;
            min-height: 24px;
            padding: 4px 10px;
        }
        QPushButton:hover {
            background-color: #434c58;
            border-color: #687384;
        }
        QPushButton:pressed, QPushButton:checked {
            background-color: #4f83ff;
            border-color: #75a0ff;
            color: #ffffff;
        }
        QPushButton:disabled {
            background-color: #323941;
            color: #858d99;
            border-color: #434c57;
        }
        QHeaderView::section {
            background-color: #303741;
            color: #dfe5ee;
            border: 0;
            border-bottom: 1px solid #46505d;
            padding: 4px 5px;
        }
        QListWidget::item, QTreeWidget::item {
            min-height: 20px;
            padding: 2px 4px;
            border-radius: 3px;
        }
        QListWidget::item:hover, QTreeWidget::item:hover {
            background-color: #333c46;
        }
        QListWidget::item:selected, QTreeWidget::item:selected {
            background-color: #4f83ff;
            color: #ffffff;
        }
        QTableWidget::item:selected {
            background-color: #4f83ff;
            color: #ffffff;
        }
        QScrollBar:vertical {
            background: #242a31;
            border: 0;
            width: 12px;
            margin: 0;
        }
        QScrollBar:horizontal {
            background: #242a31;
            border: 0;
            height: 12px;
            margin: 0;
        }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background: #566272;
            border-radius: 5px;
            min-height: 28px;
            min-width: 28px;
        }
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
            background: #6a7687;
        }
        QScrollBar::add-line, QScrollBar::sub-line,
        QScrollBar::add-page, QScrollBar::sub-page {
            background: transparent;
            border: 0;
            width: 0;
            height: 0;
        }
        QMenu {
            background-color: #303741;
            color: #e8ecf2;
            border: 1px solid #566171;
            border-radius: 5px;
            padding: 4px;
        }
        QMenu::item {
            padding: 5px 24px 5px 10px;
            border-radius: 4px;
        }
        QMenu::item:selected {
            background-color: #4f83ff;
            color: #ffffff;
        }
        QMenu::separator {
            height: 1px;
            background: #48525f;
            margin: 4px 6px;
        }
        QProgressBar {
            background-color: #242a31;
            color: #e8ecf2;
            border: 1px solid #47515f;
            border-radius: 5px;
            text-align: center;
            min-height: 18px;
        }
        QProgressBar::chunk {
            background-color: #4f83ff;
            border-radius: 4px;
        }
        QSlider::groove:horizontal {
            height: 5px;
            background: #4a5563;
            border-radius: 2px;
        }
        QSlider::sub-page:horizontal {
            background: #4f83ff;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #edf2fb;
            border: 1px solid #7a8595;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:hover {
            border-color: #9bb7ff;
        }
        QCheckBox {
            spacing: 7px;
        }
        QCheckBox::indicator {
            width: 14px;
            height: 14px;
            border-radius: 4px;
            border: 1px solid #677386;
            background: #242a31;
        }
        QCheckBox::indicator:hover {
            border-color: #7d8ca3;
        }
        QCheckBox::indicator:checked {
            background: #4f83ff;
            border-color: #75a0ff;
        }
        QStatusBar {
            background-color: #242a31;
            color: #cfd6df;
            border-top: 1px solid #303843;
        }
        QTabWidget::pane {
            border: 1px solid #4f5a68;
            border-radius: 5px;
            top: -1px;
        }
        QTabBar::tab {
            background-color: #343b45;
            color: #dfe5ee;
            border: 1px solid #566171;
            border-bottom-color: #4f5a68;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            min-height: 26px;
            min-width: 0;
            padding: 6px 4px;
            margin-right: 1px;
        }
        QTabBar::tab:hover {
            background-color: #3e4854;
            color: #ffffff;
        }
        QTabBar::tab:selected {
            background-color: #46515e;
            color: #ffffff;
            border-color: #6e7a8b;
            border-bottom-color: #46515e;
        }
        QTabWidget#rightSidebarTabs::pane {
            border: 0;
            top: 0;
        }
        QTabWidget#rightSidebarTabs::tab-bar {
            top: 0;
        }
        QTabWidget#rightSidebarTabs QTabBar {
            background: transparent;
        }
        QTabWidget#rightSidebarTabs QTabBar::tab {
            background-color: #3a424d;
            color: #f1f4f8;
            border: 1px solid #566171;
            border-radius: 5px;
            min-height: 0;
            min-width: 0;
            padding: 4px 6px;
            margin-right: 5px;
            margin-bottom: 8px;
        }
        QTabWidget#rightSidebarTabs QTabBar::tab:hover {
            background-color: #434c58;
            border-color: #687384;
            color: #ffffff;
        }
        QTabWidget#rightSidebarTabs QTabBar::tab:selected {
            background-color: #4f83ff;
            border-color: #75a0ff;
            color: #ffffff;
        }
        QToolTip {
            background-color: #303842;
            color: #f1f4f8;
            border: 1px solid #566171;
            border-radius: 4px;
            padding: 4px 6px;
        }
        """
    )
