from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase, QPalette, QColor
from PySide6.QtWidgets import QApplication


def apply_dark_theme(app: QApplication, *, density: str = "compact") -> None:
    """Neutral surfaces keep reference colors intact; density never shrinks text."""
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    families = set(QFontDatabase.families())
    for family in (".AppleSystemUIFont", "Helvetica Neue", "Arial"):
        if family in families:
            font.setFamily(family)
            break
    app.setFont(font)
    palette = QPalette()
    for role, color in {
        QPalette.ColorRole.Window: "#242629",
        QPalette.ColorRole.WindowText: "#e5e5e7",
        QPalette.ColorRole.Base: "#202124",
        QPalette.ColorRole.AlternateBase: "#282a2e",
        QPalette.ColorRole.ToolTipBase: "#35373c",
        QPalette.ColorRole.ToolTipText: "#f4f4f5",
        QPalette.ColorRole.Text: "#e5e5e7",
        QPalette.ColorRole.Button: "#303237",
        QPalette.ColorRole.ButtonText: "#e5e5e7",
        QPalette.ColorRole.BrightText: "#ffffff",
        QPalette.ColorRole.Highlight: "#456da7",
        QPalette.ColorRole.HighlightedText: "#ffffff",
    }.items():
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#85878e"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#85878e"))
    app.setPalette(palette)
    spacing = {"button": 5, "item": 4} if density == "comfortable" else {"button": 3, "item": 2}
    sheet = """
        QWidget { background: #242629; color: #e5e5e7; selection-background-color: #456da7; selection-color: white; }
        QLabel, QCheckBox { background: transparent; }
        QLabel[role="muted"] { color: #a7a9b0; }
        QLabel[role="sectionTitle"] { color: #c6c8ce; font-weight: 600; padding: 5px 0; }
        QMainWindow, QDialog { background: #242629; }
        QSplitter::handle { background: #202124; }
        QSplitter::handle:horizontal { width: 6px; }
        QSplitter::handle:vertical { height: 6px; }
        QSplitter::handle:hover { background: #555c68; }
        QLineEdit, QTextEdit, QListWidget, QTreeWidget, QTableWidget, QScrollArea {
            background: #202124; color: #e5e5e7; border: 1px solid #42444a; border-radius: 5px;
        }
        QLineEdit, QTextEdit { padding: 5px 7px; }
        QLineEdit:focus, QTextEdit:focus, QListWidget:focus, QTreeWidget:focus { border-color: #789dde; }
        QListWidget, QTreeWidget, QTableWidget { alternate-background-color: #282a2e; outline: 0; }
        QScrollArea { border: 0; }
        QComboBox {
            background: #303237; color: #dedfe3; border: 1px solid #46484f; border-radius: 5px;
            min-height: 22px; padding: @BUTTON@px 22px @BUTTON@px 8px;
        }
        QComboBox:hover { background: #393b41; border-color: #60636c; }
        QComboBox:focus { border-color: #789dde; }
        QComboBox::drop-down { border: 0; width: 20px; }
        QComboBox::down-arrow { image: url("@CHEVRON@"); width: 12px; height: 12px; }
        QSpinBox, QDoubleSpinBox { background: #202124; color: #e5e5e7; border: 1px solid #42444a; border-radius: 5px; min-height: 24px; padding: 3px 6px; }
        QPushButton {
            background: #303237; color: #dedfe3; border: 1px solid #41434a; border-radius: 5px;
            min-height: 22px; padding: @BUTTON@px 10px;
        }
        QPushButton:hover { background: #3a3d43; border-color: #60636c; }
        QPushButton:pressed, QPushButton:checked { background: #394659; border-color: #637eaa; color: #f1f5fc; }
        QPushButton:focus { border-color: #789dde; }
        QPushButton[variant="primary"] { background: #4b76ba; color: white; border-color: #658ac6; }
        QPushButton[variant="primary"]:hover { background: #5884c8; border-color: #8aaade; }
        QPushButton[variant="primary"]:pressed { background: #3d649f; }
        QPushButton[variant="navigation"] { background: transparent; border-color: transparent; color: #b7bac2; padding: 5px 12px; }
        QPushButton[variant="navigation"]:hover { background: #30343b; color: #eef1f7; }
        QPushButton[variant="navigation"]:checked { background: #354358; border-color: #4d6387; color: #dfeaff; }
        QPushButton:disabled, QComboBox:disabled, QPushButton[variant="primary"]:disabled {
            background: #292b2f; color: #777a83; border-color: #35373c;
        }
        QHeaderView::section { background: #282a2e; color: #aeb1ba; border: 0; border-bottom: 1px solid #393b42; padding: 5px; }
        QListWidget::item, QTreeWidget::item { min-height: 21px; padding: @ITEM@px 5px; border-radius: 4px; }
        QListWidget::item:hover, QTreeWidget::item:hover { background: #30333a; }
        QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected { background: #384b68; color: #ecf2fc; }
        QScrollBar:vertical { background: #242629; border: 0; width: 10px; margin: 0; }
        QScrollBar:horizontal { background: #242629; border: 0; height: 10px; margin: 0; }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: #555961; border-radius: 4px; min-height: 28px; min-width: 28px; }
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: #737882; }
        QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: transparent; border: 0; width: 0; height: 0; }
        QMenu { background: #2e3035; color: #e5e5e7; border: 1px solid #525660; border-radius: 5px; padding: 5px; }
        QMenu::item { padding: 6px 24px 6px 10px; border-radius: 4px; }
        QMenu::item:selected { background: #3e5578; color: white; }
        QMenu::separator { height: 1px; background: #474a52; margin: 4px 6px; }
        QProgressBar { background: #202124; color: #e5e5e7; border: 1px solid #42444a; border-radius: 4px; text-align: center; min-height: 18px; }
        QProgressBar::chunk { background: #5d86c7; border-radius: 3px; }
        QSlider::groove:horizontal { height: 4px; background: #4a4e57; border-radius: 2px; }
        QSlider::sub-page:horizontal { background: #7799d2; border-radius: 2px; }
        QSlider::handle:horizontal { background: #e5e9ef; border: 1px solid #adb6c4; width: 12px; margin: -5px 0; border-radius: 6px; }
        QSlider::handle:horizontal:hover { border-color: white; background: white; }
        QCheckBox { spacing: 7px; }
        QCheckBox::indicator { width: 14px; height: 14px; border-radius: 3px; border: 1px solid #737882; background: #202124; }
        QCheckBox::indicator:hover { border-color: #a5b7d4; }
        QCheckBox::indicator:checked { background: #7799d2; border-color: #b6cdf1; }
        QStatusBar { background: #202124; color: #a7abb4; border-top: 1px solid #303238; }
        QTabWidget::pane { border: 0; top: 0; }
        QTabBar { background: transparent; }
        QTabBar::tab {
            background: transparent; color: #aeb1ba; border: 0; border-bottom: 2px solid transparent;
            min-height: 24px; min-width: 0; padding: 5px 12px; margin: 0 2px 6px 0;
        }
        QTabBar::tab:hover { background: #30333a; color: #eef1f7; }
        QTabBar::tab:selected { color: #d6e5ff; border-bottom-color: #7799d2; }
        QToolTip { background: #35373c; color: #f4f4f5; border: 1px solid #5c616d; border-radius: 4px; padding: 5px 7px; }
        QWidget#workspaceHeader { background: #202124; border-bottom: 1px solid #383b42; }
        QLabel#workspaceBrand { color: #f0f1f5; font-size: 16px; font-weight: 600; padding-right: 10px; }
        QWidget#workspaceHeader QPushButton { background: transparent; border-color: transparent; }
        QWidget#workspaceHeader QPushButton:hover { background: #363940; border-color: #555c68; }
        QWidget#workspaceHeader QPushButton:checked { background: #313844; color: #cbdcf7; }
        QWidget#navigationSidebar, QWidget#referenceSidebar { background: #242629; }
        QTabWidget#navigationTabs QTreeWidget, QTabWidget#navigationTabs QListWidget { background: #242629; border: 0; }
        QWidget#selectionActions { border-top: 1px solid #3b3e45; }
        QWidget#selectionActions QLabel { border: 0; }
        QWidget#boardTools { background: #242629; }
        QWidget#aiProposalPanel { background: #2d3440; border: 1px solid #536583; border-radius: 6px; }
        QWidget#aiProposalPanel QLabel { border: 0; }
    """
    app.setStyleSheet(sheet.replace("@BUTTON@", str(spacing["button"])).replace("@ITEM@", str(spacing["item"])).replace("@CHEVRON@", (Path(__file__).parent / "icons" / "chevron-down.svg").as_posix()))
