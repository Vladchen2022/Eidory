"""A bounded inline editor; focus changes commit once, Escape discards."""
from PySide6.QtCore import QEvent, QTimer, Signal
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QTextEdit, QVBoxLayout, QWidget


class InlineIdeaEditor(QWidget):
    finished = Signal(object)

    def __init__(self, name='', visual='', *, fixed_name=False, parent=None):
        super().__init__(parent)
        self.done = False
        self.original = (name, visual)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.name = QLineEdit(name)
        self.name.setAccessibleName('想法名称 · 原地编辑')
        self.name.setPlaceholderText('想法名称')
        self.name.setReadOnly(fixed_name)
        self.visual = QTextEdit(visual)
        self.visual.setAcceptRichText(False)
        self.visual.setAccessibleName('想法细节 · 原地编辑')
        self.visual.setPlaceholderText('可选：造型、动作或材质细节')
        self.visual.setFixedHeight(95)
        self.hint = QLabel('离开此卡片自动保存 · Esc 取消')
        self.hint.setProperty('role', 'muted')
        layout.addWidget(self.name)
        layout.addWidget(self.visual)
        layout.addWidget(self.hint)
        for field in (self.name, self.visual, self.visual.viewport()):
            field.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and event.key() == 16777216:
            self.finish(cancel=True)
            return True
        if event.type() == QEvent.Type.FocusOut:
            QTimer.singleShot(0, self.commit_if_left)
        return False

    def commit_if_left(self):
        focus = QApplication.focusWidget()
        if focus is None or (focus is not self and not self.isAncestorOf(focus)):
            self.finish()

    def finish(self, *, cancel=False):
        if self.done:
            return
        name = self.name.text().strip()
        if not cancel and not name:
            if not any(self.original) and not self.visual.toPlainText().strip():
                cancel = True
            else:
                self.hint.setText('名称不能为空；填写名称或按 Esc 取消')
                self.name.setFocus()
                return
        self.done = True
        self.finished.emit(None if cancel else (name, self.visual.toPlainText().strip()))


def change_html(before, after):
    """Escape all authored content and mark only actual edits."""
    from difflib import SequenceMatcher
    from html import escape
    # Long prose is compared by lines to bound the UI cost on repetitive text.
    if len(before) + len(after) > 8000:
        return ('<del style="color:#aa3939">' + escape(before).replace('\n', '<br>') +
                '</del><br><b style="color:#25814a">' + escape(after).replace('\n', '<br>') + '</b>')
    parts = []
    for kind, i, j, a, b in SequenceMatcher(None, before, after, autojunk=True).get_opcodes():
        if kind in ('delete', 'replace'):
            parts.append('<del style="color:#aa3939">' + escape(before[i:j]) + '</del>')
        if kind in ('insert', 'replace'):
            parts.append('<b style="color:#25814a">' + escape(after[a:b]) + '</b>')
        elif kind == 'equal':
            parts.append(escape(after[a:b]))
    return ''.join(parts).replace('\n', '<br>') or '无变化'
