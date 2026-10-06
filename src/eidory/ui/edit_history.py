"""Bounded command history; database undo changes only fields the command touched."""
from contextlib import contextmanager
import time
import json
from functools import wraps
from weakref import ref
from PySide6.QtCore import QObject
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QLineEdit, QTextEdit, QPlainTextEdit


def undoable(label):
    def decorate(method):
        @wraps(method)
        def run(self, *args, **kwargs):
            window = getattr(self, 'w', self)
            history = getattr(window, 'edit_history', None)
            if history is None or (method.__name__ == "flush_note" and self._pending_note is None) or getattr(self, "refreshing", False) or getattr(self, "loading", False):
                return method(self, *args, **kwargs)
            with history.record(label):
                return method(self, *args, **kwargs)
        return run
    return decorate


class EditHistory(QObject):
    TABLES = ('creative_nodes', 'creative_node_workflow', 'creative_node_options',
              'creative_node_images', 'creative_reference_decisions', 'creative_reference_notes')

    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self._histories = {}
        self.busy = False
        self.depth = 0
        self._typing = None
        self.undo_action = QAction('撤销', window)
        self.redo_action = QAction('重做', window)
        self.undo_action.setShortcuts([QKeySequence(QKeySequence.StandardKey.Undo), QKeySequence("Meta+Z")])
        self.redo_action.setShortcuts([QKeySequence.StandardKey.Redo, QKeySequence('Meta+Shift+Z')])
        self.undo_action.triggered.connect(self.undo)
        self.redo_action.triggered.connect(self.redo)
        window.undo_removal_action.setShortcut(QKeySequence())
        window.addAction(self.undo_action)
        window.addAction(self.redo_action)
        menu = window.workflow_upgrade.tools_menu
        menu.addSeparator()
        menu.addAction(self.undo_action)
        menu.addAction(self.redo_action)
        for button, callback in ((window.creative_workspace.board_buttons[2], self.undo),
                                 (window.creative_workspace.board_buttons[3], self.redo)):
            button.clicked.disconnect()
            button.clicked.connect(callback)
        self.refresh()

    @property
    def w(self):
        return self._window()

    def context(self):
        w = self.w
        return (w.current_creative_project_id, w.current_temp_project_id)

    @property
    def past(self):
        return self._histories.setdefault(self.context(), ([], []))[0]

    @past.setter
    def past(self, value):
        self._histories[self.context()] = (value, self.future)

    @property
    def future(self):
        return self._histories.setdefault(self.context(), ([], []))[1]

    def snapshot(self, tables=None, *, context=None):
        pid = (context or self.context())[0]
        if pid is None:
            return {}
        result = {}
        with self.w.store.connect() as conn:
            for table in tables or self.TABLES:
                schema = conn.execute(f'PRAGMA table_info({table})').fetchall()
                if not schema:
                    continue
                keys = [r['name'] for r in sorted(schema, key=lambda r: r['pk']) if r['pk']]
                where = 'project_id=?' if table == 'creative_nodes' else 'node_id IN (SELECT id FROM creative_nodes WHERE project_id=?)'
                for row in conn.execute(f'SELECT * FROM {table} WHERE {where}', (pid,)):
                    data = {k: row[k] for k in row.keys() if k != 'updated_at'}
                    result[(table, tuple((key, row[key]) for key in keys))] = data
        return result

    @contextmanager
    def record(self, label):
        if self.busy or self.depth:
            yield
            return
        self.w.project_board_view.flush_pending_layout_change()
        context = self.context()
        past, _future = self._histories.setdefault(context, ([], []))
        self.depth += 1
        try:
            typing_key = (self.w.current_creative_project_id, self.w.current_creative_node_id) if label == '修改找图目标' else None
            tables = ('creative_node_options',) if typing_key else None
            before = self.snapshot(tables, context=context)
            previous = self._typing
            merge = (typing_key is not None and previous is not None and previous[0] == typing_key
                     and time.monotonic() - previous[1] < 1.2 and past
                     and past[-1] is previous[3] and before == previous[4])
            yield
            after = self.snapshot(tables, context=context)
            if merge:
                before = previous[2]
            changes = []
            for key in before.keys() | after.keys():
                old, new = before.get(key), after.get(key)
                if old != new:
                    if old is not None and new is not None:
                        fields = [f for f in old if old[f] != new[f]]
                        old, new = {f: old[f] for f in fields}, {f: new[f] for f in fields}
                    changes.append((key, old, new))
            if changes:
                if merge:
                    past.pop()
                self.push(label, lambda: self.restore(changes, backwards=True),
                          lambda: self.restore(changes, backwards=False), context=context)
                self._typing = (typing_key, time.monotonic(), before, past[-1], after) if typing_key else None
        finally:
            self.depth -= 1

    def restore(self, changes, *, backwards):
        # Validate the whole command before writing; later AI/user edits must
        # never be silently overwritten by an old undo operation.
        ordered = sorted(changes, key=lambda c: self.TABLES.index(c[0][0]))
        with self.w.store.connect() as conn:
            for (table, keys), old, new in ordered:
                expected = new if backwards else old
                where = ' AND '.join(f'{k}=?' for k, _ in keys)
                row = conn.execute(f'SELECT * FROM {table} WHERE {where}', tuple(v for _, v in keys)).fetchone()
                if expected is None:
                    if row is not None:
                        return False
                elif row is None or any(row[k] != v for k, v in expected.items()):
                    return False
            # Remove children before parents, insert/update parents before children.
            for (table, keys), old, new in reversed(ordered):
                target = old if backwards else new
                if target is None:
                    conn.execute(f'DELETE FROM {table} WHERE ' + ' AND '.join(f'{k}=?' for k, _ in keys), tuple(v for _, v in keys))
            for (table, keys), old, new in ordered:
                target, expected = (old, new) if backwards else (new, old)
                if target is None:
                    continue
                if expected is None:
                    conn.execute(f'INSERT INTO {table} (' + ','.join(target) + ') VALUES (' + ','.join('?' for _ in target) + ')', tuple(target.values()))
                else:
                    conn.execute(f'UPDATE {table} SET ' + ','.join(f'{k}=?' for k in target) + ' WHERE ' + ' AND '.join(f'{k}=?' for k, _ in keys), (*target.values(), *(v for _, v in keys)))
        w = self.w
        node_id = w.current_creative_node_id
        board = w.center_result_stack.currentWidget() is w.project_board_view
        w._invalidate_creative_node_direct_image_cache()
        w._clear_creative_node_editor_state()
        w._refresh_creative_projects(select_project_id=w.current_creative_project_id)
        if node_id and w.store.get_creative_node(node_id):
            w._load_creative_project(w.current_creative_project_id, select_node_id=node_id, show_board=board, save_current_editor=False)
        if w.current_creative_node_id:
            w._sync_creative_node_panel()
        if w.center_result_stack.currentWidget() is w.project_board_view:
            w.creative_flow.board_inspector.refresh_badges()
            w.creative_flow.board_inspector.refresh()
        w.creative_flow.refresh()
        return True

    def board_edit(self, before, after):
        if self.depth or self.busy:
            return
        from copy import deepcopy
        w = self.w
        key = (w.current_creative_project_id, w._current_board_node_id, w.current_temp_project_id)
        before, after = deepcopy(before), deepcopy(after)
        def restore(payload, expected):
            current = (w.current_creative_project_id, w._current_board_node_id, w.current_temp_project_id)
            if key != current:
                pid, node_id, temp_id = key
                if (pid, temp_id) != self.context():
                    return False
                getter, setter, target = ((w.store.get_creative_node_board_layout, w.store.save_creative_node_board_layout, node_id)
                    if node_id else (w.store.get_creative_board_layout, w.store.save_creative_board_layout, pid)
                    if pid else (w.store.get_temporary_project_board_layout, w.store.save_temporary_project_board_layout, temp_id))
                saved = getter(target)
                if not saved or json.loads(saved) != expected:
                    return False
                setter(target, json.dumps(payload, ensure_ascii=False))
                return True
            if set(payload.get('items', {})) != set(w.project_board_view.layout_payload().get('items', {})):
                return False
            w.project_board_view._restore_layout(payload)
            return True
        self.push('看板布局', lambda: restore(before, after), lambda: restore(after, before))

    def push(self, label, undo, redo, *, context=None):
        if self.busy:
            return
        key = self.context() if context is None else context
        past, future = self._histories.setdefault(key, ([], []))
        past.append((label, undo, redo))
        del past[:-40]
        future.clear()
        self.refresh()

    def refresh(self):
        self.undo_action.setText('撤销：' + self.past[-1][0] if self.past else '撤销')
        self.redo_action.setText('重做：' + self.future[-1][0] if self.future else '重做')
        # Enabled shortcuts route text edits to the focused text widget.
        self.redo_action.setEnabled(True)
        for button, stack in ((self.w.creative_workspace.board_buttons[2], self.past),
                              (self.w.creative_workspace.board_buttons[3], self.future)):
            button.setToolTip(('撤销：' if stack is self.past else '重做：') + (stack[-1][0] if stack else '没有操作'))

    def _run(self, backwards):
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QTextEdit, QPlainTextEdit)) and not focus.isReadOnly():
            (focus.undo if backwards else focus.redo)()
            return
        self.w.project_board_view.flush_pending_layout_change()
        source, destination = (self.past, self.future) if backwards else (self.future, self.past)
        if not source:
            if backwards:
                self.w._handle_undo_shortcut()
            return
        command = source[-1]
        self.busy = True
        try:
            ok = command[1 if backwards else 2]()
            if ok is False:
                self.w.statusBar().showMessage('该内容之后已有其他修改，未覆盖；可使用方案版本恢复。')
                return
            source.pop()
            destination.append(command)
            self.w.statusBar().showMessage(('已撤销：' if backwards else '已重做：') + command[0])
        finally:
            self.busy = False
            self.refresh()

    def undo(self):
        self._run(True)

    def redo(self):
        self._run(False)
