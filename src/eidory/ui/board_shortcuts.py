"""Page-scoped single keys; text inputs and dialogs always keep their keys."""
from weakref import ref

from PySide6.QtCore import QEvent, QObject, QPoint, Qt
from PySide6.QtWidgets import (QApplication, QAbstractSpinBox, QComboBox, QLineEdit,
                               QPlainTextEdit, QTextEdit, QWidget)


def set_board_button_text(button, text):
    key = button.property('boardShortcut')
    if key:
        text = text.removesuffix(' ▾')
        suffix = f' ({key})'
        if not text.endswith(suffix):
            text += suffix
        if button.menu():
            text += ' ▾'
    button.setText(text)


class BoardShortcuts(QObject):
    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        cw = window.creative_workspace
        inspector = window.creative_flow.board_inspector
        self.buttons = {
            Qt.Key.Key_0: (window.board_fit_all_button, '0'),
            Qt.Key.Key_N: (cw.board_buttons[0], 'N'),
            Qt.Key.Key_J: (window.board_navigation.button, 'J'),
            Qt.Key.Key_G: (cw.board_buttons[1], 'G'),
            Qt.Key.Key_Z: (cw.board_buttons[2], 'Z'),
            Qt.Key.Key_Y: (cw.board_buttons[3], 'Y'),
            Qt.Key.Key_P: (window.board_pin_button, 'P'),
            Qt.Key.Key_H: (window.board_hide_selected_button, 'H'),
            Qt.Key.Key_F: (window.board_flip_button, 'F'),
            Qt.Key.Key_B: (window.board_grayscale_button, 'B'),
            Qt.Key.Key_C: (window.board_edit_local_display_button, 'C'),
            Qt.Key.Key_V: (window.board_toggle_local_display_button, 'V'),
            Qt.Key.Key_L: (window.workspace_chrome.board_section_lock_button, 'L'),
            Qt.Key.Key_1: (inspector.quick_grades['candidate'], '1'),
            Qt.Key.Key_2: (inspector.quick_grades['primary'], '2'),
            Qt.Key.Key_3: (inspector.quick_grades['unused'], '3'),
        }
        for button, key in self.buttons.values():
            button.setProperty('boardShortcut', key)
        self.refresh_labels()
        inspector.remove_button.setText('移出看板 (Delete)')
        cw.collect_button.setToolTip('在图片墙选中图片后，按 Enter 收集到当前节点')
        # Filter only key receivers and the main window (for keys propagated
        # from toolbar buttons), not every paint/layout event in the application.
        for widget in (window, window.project_board_view, window.project_board_view.viewport(),
                       window.grid_view, window.grid_view.viewport()):
            widget.installEventFilter(self)

    def refresh_labels(self):
        fit = self._window().board_fit_all_button
        fit.setMinimumSize(0, 0)
        fit.setMaximumSize(16777215, 16777215)
        for button, _key in self.buttons.values():
            set_board_button_text(button, button.text() or button.accessibleName())
        self._window().board_grayscale_button.setAccessibleDescription(
            self._window().board_grayscale_button.toolTip())

    def eventFilter(self, watched, event):
        if event.type() not in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
            return False
        w = self._window()
        if w is None or not isinstance(watched, QWidget) or watched.window() is not w:
            return False
        if QApplication.activeModalWidget() or QApplication.activePopupWidget():
            return False
        if event.modifiers() != Qt.KeyboardModifier.NoModifier:
            return False
        focus = QApplication.focusWidget()
        # Also cover the viewport child of a text editor, and editable combos.
        parent = focus
        while parent is not None and parent is not w:
            if isinstance(parent, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox)):
                return False
            parent = parent.parentWidget()
        board_active = w.center_result_stack.currentWidget() is w.project_board_view
        button = None
        if board_active and event.key() in self.buttons:
            button = self.buttons[event.key()][0]
        elif (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
              and w.center_result_stack.currentWidget() is w.grid_view
              and focus in (w.grid_view, w.grid_view.viewport())
              and w.current_creative_project_id is not None
              and w.current_creative_node_id is not None):
            button = w.creative_workspace.collect_button
        if button is None:
            return False
        event.accept()
        if event.type() == QEvent.Type.ShortcutOverride or event.isAutoRepeat():
            return True
        if not button.isEnabled():
            return True
        if button.menu():
            anchor = button if button.isVisible() else w.workspace_chrome.board_tools.more
            button.menu().popup(anchor.mapToGlobal(QPoint(0, anchor.height())))
        else:
            button.click()
        return True
