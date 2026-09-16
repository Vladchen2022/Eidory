from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFontDatabase, QPalette, QColor
from PySide6.QtWidgets import QApplication

from eidory.ui.theme_colors import DEFAULT_THEME, get_theme, refresh_themed_widgets, resolve_theme_style


def apply_theme(app: QApplication, theme_id: str = DEFAULT_THEME, *, density: str = "compact", font_size: int | None = None) -> None:
    """Switch presentation only; never rebuild library, filter or board content."""
    theme = get_theme(theme_id)
    app.setProperty("eidoryTheme", theme.id)
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    families = set(QFontDatabase.families())
    for family in (".AppleSystemUIFont", "Helvetica Neue", "Arial"):
        if family in families:
            font.setFamily(family)
            break
    if font_size is not None:
        app.setProperty("eidoryFontSize", max(11, min(20, int(font_size))))
    preferred = app.property("eidoryFontSize")
    if preferred:
        font.setPointSize(int(preferred))
    app.setFont(font)
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: "window", QPalette.ColorRole.WindowText: "text",
        QPalette.ColorRole.Base: "base", QPalette.ColorRole.AlternateBase: "raised",
        QPalette.ColorRole.ToolTipBase: "raised", QPalette.ColorRole.ToolTipText: "text",
        QPalette.ColorRole.Text: "text", QPalette.ColorRole.Button: "button",
        QPalette.ColorRole.ButtonText: "text", QPalette.ColorRole.BrightText: "accent_text",
        QPalette.ColorRole.Highlight: "selection", QPalette.ColorRole.HighlightedText: "selection_text",
        QPalette.ColorRole.PlaceholderText: "muted", QPalette.ColorRole.Link: "link",
    }
    for role, token in roles.items():
        palette.setColor(role, QColor(theme.colors[token]))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(theme.colors["disabled"]))
    app.setPalette(palette)
    spacing = {"button": 5, "item": 4} if density == "comfortable" else {"button": 3, "item": 2}
    chevron = "chevron-down.svg" if theme.dark else "chevron-down-light.svg"
    chevron_up = "chevron-up.svg" if theme.dark else "chevron-up-light.svg"
    sheet = resolve_theme_style(_STYLESHEET)
    sheet = sheet.replace("@BUTTON@", str(spacing["button"])).replace("@ITEM@", str(spacing["item"]))
    sheet = sheet.replace("@CHEVRON@", (Path(__file__).parent / "icons" / chevron).as_posix())
    sheet = sheet.replace("@CHEVRON_UP@", (Path(__file__).parent / "icons" / chevron_up).as_posix())
    app.setStyleSheet(sheet)
    refresh_themed_widgets(app)


def apply_dark_theme(app: QApplication, *, density: str = "compact") -> None:
    """Compatibility for callers that explicitly request the original theme."""
    apply_theme(app, DEFAULT_THEME, density=density)


_STYLESHEET = """
        QWidget { background: @window@; color: @text@; selection-background-color: @selection@; selection-color: @selection_text@; }
        QLabel, QCheckBox { background: transparent; }
        QWidget#confirmedEntriesPanel { border: 1px solid @border@; border-radius: 6px; background: @base@; }
        QWidget#creativeOverviewCard, QWidget#creativeIdeaCard { background: @base@; border: 1px solid @border@; border-radius: 8px; }
        QWidget#creativeSuggestionCard { background: @raised@; border: 1px solid @warning@; border-radius: 8px; }
        QWidget#creativeOverviewCard QLabel, QWidget#creativeIdeaCard QLabel, QWidget#creativeSuggestionCard QLabel { border: none; background: transparent; }
        QLabel[role="flowTitle"] { font-size: 19px; font-weight: 600; padding: 3px 0; }
        QLabel[role="flowSubtitle"] { font-weight: 600; padding: 3px 0; }
        QLabel#creativeBaseSummary { background: @base@; border: 1px solid @border@; border-radius: 6px; padding: 10px; }
        QWidget#pendingSuggestionPanel { border: 2px solid @warning@; border-radius: 6px; background: @raised@; }
        QLabel[role="muted"] { color: @muted@; }
        QLabel[role="sectionTitle"] { color: @muted@; font-weight: 600; padding: 5px 0; }
        QMainWindow, QDialog { background: @window@; }
        QSplitter::handle { background: @base@; }
        QSplitter::handle:horizontal { width: 6px; }
        QSplitter::handle:vertical { height: 6px; }
        QSplitter::handle:hover { background: @border@; }
        QLineEdit, QTextEdit, QListWidget, QTreeWidget, QTableWidget, QScrollArea {
            background: @base@; color: @text@; border: 1px solid @border@; border-radius: 5px;
        }
        QLineEdit, QTextEdit { padding: 5px 7px; }
        QLineEdit:focus, QTextEdit:focus, QListWidget:focus, QTreeWidget:focus { border-color: @link@; }
        QListWidget, QTreeWidget, QTableWidget { alternate-background-color: @raised@; outline: 0; }
        QScrollArea { border: 0; }
        QComboBox {
            background: @button@; color: @text@; border: 1px solid @border@; border-radius: 5px;
            min-height: 22px; padding: @BUTTON@px 22px @BUTTON@px 8px;
        }
        QComboBox:hover { background: @hover@; border-color: @border@; }
        QComboBox:focus { border-color: @link@; }
        QComboBox::drop-down { border: 0; width: 20px; }
        QComboBox::down-arrow { image: url("@CHEVRON@"); width: 12px; height: 12px; }
        QSpinBox, QDoubleSpinBox { background: @base@; color: @text@; border: 1px solid @border@; border-radius: 5px; min-height: 24px; padding: 3px 24px 3px 6px; }
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-origin: border; subcontrol-position: top right; width: 20px;
            background: @button@; border-left: 1px solid @border@; border-top-right-radius: 5px;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-origin: border; subcontrol-position: bottom right; width: 20px;
            background: @button@; border-left: 1px solid @border@; border-bottom-right-radius: 5px;
        }
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: @hover@; }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url("@CHEVRON_UP@"); width: 12px; height: 12px; }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url("@CHEVRON@"); width: 12px; height: 12px; }
        QPushButton {
            background: @button@; color: @text@; border: 1px solid @border@; border-radius: 5px;
            min-height: 22px; padding: @BUTTON@px 10px;
        }
        QPushButton:hover { background: @hover@; border-color: @border@; }
        QPushButton:pressed, QPushButton:checked { background: @selection@; border-color: @link@; color: @selection_text@; }
        QPushButton:focus { border-color: @link@; }
        QPushButton[variant="primary"] { background: @accent@; color: @accent_text@; border-color: @accent@; }
        QPushButton[variant="primary"]:hover { background: @accent_hover@; border-color: @accent_hover@; }
        QPushButton[variant="primary"]:pressed { background: @accent_pressed@; }
        QPushButton[variant="navigation"] { background: transparent; border-color: transparent; color: @muted@; padding: 5px 12px; }
        QPushButton[variant="navigation"]:hover { background: @button@; color: @text@; }
        QPushButton[variant="navigation"]:checked { background: @selection@; border-color: @selection@; color: @selection_text@; }
        QPushButton:disabled, QComboBox:disabled, QPushButton[variant="primary"]:disabled {
            background: @raised@; color: @disabled@; border-color: @hover@;
        }
        QHeaderView::section { background: @raised@; color: @muted@; border: 0; border-bottom: 1px solid @border@; padding: 5px; }
        QListWidget::item, QTreeWidget::item { min-height: 21px; padding: @ITEM@px 5px; border-radius: 4px; }
        QListWidget::item:hover, QTreeWidget::item:hover { background: @hover@; }
        QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected { background: @selection@; color: @selection_text@; }
        QScrollBar:vertical { background: @window@; border: 0; width: 10px; margin: 0; }
        QScrollBar:horizontal { background: @window@; border: 0; height: 10px; margin: 0; }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: @border@; border-radius: 4px; min-height: 28px; min-width: 28px; }
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: @disabled@; }
        QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: transparent; border: 0; width: 0; height: 0; }
        QMenu { background: @raised@; color: @text@; border: 1px solid @border@; border-radius: 5px; padding: 5px; }
        QMenu::item { padding: 6px 24px 6px 10px; border-radius: 4px; }
        QMenu::item:selected { background: @selection@; color: @selection_text@; }
        QMenu::separator { height: 1px; background: @border@; margin: 4px 6px; }
        QProgressBar { background: @base@; color: @text@; border: 1px solid @border@; border-radius: 4px; text-align: center; min-height: 18px; }
        QProgressBar::chunk { background: @accent@; border-radius: 3px; }
        QSlider::groove:horizontal { height: 4px; background: @border@; border-radius: 2px; }
        QSlider::sub-page:horizontal { background: @link@; border-radius: 2px; }
        QSlider::handle:horizontal { background: @link@; border: 1px solid @muted@; width: 12px; margin: -5px 0; border-radius: 6px; }
        QSlider::handle:horizontal:hover { border-color: @accent_text@; background: white; }
        QCheckBox { spacing: 7px; }
        QCheckBox::indicator { width: 14px; height: 14px; border-radius: 3px; border: 1px solid @disabled@; background: @base@; }
        QCheckBox::indicator:hover { border-color: @link@; }
        QCheckBox::indicator:checked { background: @link@; border-color: @link@; }
        QStatusBar { background: @base@; color: @muted@; border-top: 1px solid @border@; }
        QTabWidget::pane { border: 0; top: 0; }
        QTabBar { background: transparent; }
        QTabBar::tab {
            background: transparent; color: @muted@; border: 0; border-bottom: 2px solid transparent;
            min-height: 24px; min-width: 0; padding: 5px 12px; margin: 0 2px 6px 0;
        }
        QTabBar::tab:hover { background: @hover@; color: @text@; }
        QTabBar::tab:selected { color: @link@; border-bottom-color: @link@; }
        QToolTip { background: @hover@; color: @text@; border: 1px solid @border@; border-radius: 4px; padding: 5px 7px; }
        QWidget#workspaceHeader { background: @base@; border-bottom: 1px solid @border@; }
        QLabel#workspaceBrand { color: @text@; font-size: 16px; font-weight: 600; padding-right: 10px; }
        QWidget#workspaceHeader QPushButton { background: transparent; border-color: transparent; }
        QWidget#workspaceHeader QPushButton:hover { background: @hover@; border-color: @border@; }
        QWidget#workspaceHeader QPushButton:checked { background: @button@; color: @link@; }
        QWidget#navigationSidebar, QWidget#referenceSidebar { background: @window@; }
        QTabWidget#navigationTabs QTreeWidget, QTabWidget#navigationTabs QListWidget { background: @window@; border: 0; }
        QWidget#selectionActions { border-top: 1px solid @border@; }
        QWidget#selectionActions QLabel { border: 0; }
        QWidget#boardTools { background: @window@; }
        QWidget#aiProposalPanel { background: @selection@; border: 1px solid @selection@; border-radius: 6px; }
        QWidget#aiProposalPanel QLabel { border: 0; }

        QLabel[role="warning"], QLabel[role="filterExclude"] { color: @warning@; }
        QLabel[role="filterInclude"] { color: @link@; }
        QLabel[role="filterInclude"], QLabel[role="filterExclude"] { font-weight: bold; font-size: 14px; }
        QSlider[filterSection="exclude"]::sub-page:horizontal { background: @warning_fill@; }
        QSlider[filterSection="exclude"]::handle:horizontal { background: @warning@; }
        QLabel[role="preview"], QWidget[role="preview"] { background: @canvas@; color: @text@; border: 1px solid @border@; }
        DropImportBox { border: 1px dashed @border@; border-radius: 6px; background: @raised@; color: @text@; padding: 14px; }
        DropImportBox:disabled { color: @disabled@; }
        QComboBox QAbstractItemView { background: @base@; color: @text@; selection-background-color: @selection@; selection-color: @selection_text@; }

"""
