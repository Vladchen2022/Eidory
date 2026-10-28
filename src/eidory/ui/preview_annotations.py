"""Preview-only ink layer, input and compact, always-visible annotation controls."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QTransform
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QGraphicsItem, QGraphicsView, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QSlider, QWidget,
)

from eidory.core.preview_annotations import AnnotationStore, MAX_POINTS, Stroke, erase
from eidory.ui.wrapping_layout import WrappingLayout


class InkLayer(QGraphicsItem):
    def __init__(self, parent):
        super().__init__(parent)
        self.setZValue(10)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsToShape)
        self.aspect = (1., 1.)
        self.strokes = ()
        self.paths = {}
        self.live = None

    def boundingRect(self):
        return QRectF(0, 0, *self.aspect)

    def set_geometry(self, width, height, mirrored):
        aspect = (width/min(width, height), height/min(width, height))
        if aspect != self.aspect:
            self.prepareGeometryChange()
            self.aspect = aspect
            self.paths.clear()
        scale = min(width, height)
        self.setTransform(QTransform(-scale if mirrored else scale, 0, 0, scale,
                                     width if mirrored else 0, 0))
        self.update()

    def set_strokes(self, strokes):
        self.strokes = strokes
        # Reuse immutable path objects across edits; discard superseded paths.
        self.paths = {id(s): self.paths[id(s)] for s in strokes if id(s) in self.paths}
        self.update()

    def path(self, stroke):
        path = self.paths.get(id(stroke))
        if path is None:
            w, h = self.aspect
            path = QPainterPath(QPointF(stroke.points[0][0]*w, stroke.points[0][1]*h))
            for x, y in stroke.points[1:]:
                path.lineTo(x*w, y*h)
            self.paths[id(stroke)] = path
        return path

    def paint(self, painter, option, widget=None):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for stroke in self.strokes:
            self._paint_stroke(painter, stroke, self.path(stroke))
        if self.live is not None:
            stroke, path = self.live
            self._paint_stroke(painter, stroke, path)
        painter.restore()

    def _paint_stroke(self, painter, stroke, path):
        painter.setPen(QPen(QColor(stroke.color), stroke.width, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        if path.elementCount() == 1:
            e = path.elementAt(0)
            painter.drawPoint(QPointF(e.x, e.y))
        else:
            painter.drawPath(path)


class AnnotationViewMixin:
    """Scene coordinates may change during async decoding; ink stays image-relative."""
    def init_annotations(self):
        self.ink_layer = InkLayer(self._pixmap_item)
        self.annotation_tool = "pan"
        self.annotation_width = 4
        self.annotation_color = "#E53935"
        self.annotations = ()
        self._ink_undo, self._ink_redo = [], []
        self._ink_before = None
        self._ink_points = []
        self._ink_previous = None
        self._ink_space = False
        self._ink_mirrored = False
        self._ink_editable = False
        self._ink_visible = True

    def set_annotation_document(self, strokes, *, editable):
        self.finish_annotation_gesture()
        self.annotations = strokes
        self._ink_undo, self._ink_redo = [], []
        self._ink_editable = editable
        self.ink_layer.set_strokes(strokes)
        self.annotationHistoryChanged.emit()
        self._update_drag_mode()

    def set_annotation_tool(self, tool):
        self.finish_annotation_gesture()
        self.annotation_tool = tool
        self._ink_space = False
        if tool != "pan" and not self._ink_visible:
            self.set_annotations_visible(True)
        self._update_drag_mode()

    def set_annotations_visible(self, visible):
        self.finish_annotation_gesture()
        self._ink_visible = visible
        self.ink_layer.setVisible(visible)

    def sync_annotation_geometry(self):
        self.ink_layer.set_geometry(self._source_width, self._source_height, self._ink_mirrored)

    def annotation_editing(self):
        return self._ink_editable and self._has_pixmap and self._ink_visible and self.annotation_tool != "pan"

    def set_annotation_space(self, down):
        self.finish_annotation_gesture()
        self._ink_space = down
        self._update_drag_mode()

    def _ink_position(self, pos, *, clamp=False):
        point = self.mapToScene(pos.toPoint())
        x, y = point.x()/self._source_width, point.y()/self._source_height
        if not clamp and not (0 <= x <= 1 and 0 <= y <= 1):
            return None
        x, y = max(0., min(1., x)), max(0., min(1., y))
        return (1-x if self._ink_mirrored else x, y)

    def _ink_width_in_image(self, pixels):
        return min(2., pixels / max(1., min(self._source_width, self._source_height)*self.transform().m11()))

    def _begin_ink(self, pos):
        point = self._ink_position(pos)
        if point is None:
            return
        self._ink_before = self.annotations
        self._ink_previous = point
        self._ink_points = [point]
        self._ink_point_count = sum(len(s.points) for s in self.annotations)
        self._ink_stroke_width = self._ink_width_in_image(self.annotation_width)
        if self.annotation_tool == "pen":
            w, h = self.ink_layer.aspect
            self.ink_layer.live = (Stroke((point,), self._ink_stroke_width, self.annotation_color),
                                   QPainterPath(QPointF(point[0]*w, point[1]*h)))
            self.ink_layer.update()
        else:
            self._continue_ink(pos)

    def _continue_ink(self, pos):
        if self._ink_before is None:
            return
        point = self._ink_position(pos, clamp=True)
        if self.annotation_tool == "pen":
            w, h = self.ink_layer.aspect
            previous = self._ink_previous
            # Sub-pixel samples add no visible detail but inflate storage and undo.
            distance = ((point[0]-previous[0])*self._source_width)**2 + ((point[1]-previous[1])*self._source_height)**2
            if distance*self.transform().m11()**2 < 0.7**2 or self._ink_point_count+len(self._ink_points) >= MAX_POINTS:
                return
            self._ink_points.append(point)
            path = self.ink_layer.live[1]
            path.lineTo(point[0]*w, point[1]*h)
            pad = self._ink_stroke_width
            self.ink_layer.update(QRectF(QPointF(previous[0]*w, previous[1]*h),
                                        QPointF(point[0]*w, point[1]*h)).normalized().adjusted(-pad,-pad,pad,pad))
        else:
            candidate = erase(self.annotations, self._ink_previous, point,
                              self._ink_width_in_image(max(8, self.annotation_width*2)), self.ink_layer.aspect)
            if sum(len(s.points) for s in candidate) <= MAX_POINTS:
                self.annotations = candidate
                self.ink_layer.set_strokes(candidate)
        self._ink_previous = point

    def finish_annotation_gesture(self):
        if self._ink_before is None:
            return
        before = self._ink_before
        self._ink_before = None
        if self.annotation_tool == "pen" and self._ink_point_count < MAX_POINTS:
            self.annotations = before + (Stroke(tuple(self._ink_points), self._ink_stroke_width,
                                                self.ink_layer.live[0].color),)
        self.ink_layer.live = None
        self.ink_layer.set_strokes(self.annotations)
        if self.annotations != before:
            self._push_ink_undo(before)
            self._ink_redo.clear()
            self.annotationsChanged.emit()
        self.annotationHistoryChanged.emit()

    def _push_ink_undo(self, state):
        self._ink_undo.append(state)
        # Shared immutable strokes make ordinary drawing cheap. Bound erase history too.
        while len(self._ink_undo) > 40 or (len(self._ink_undo) > 1 and
                sum(len(s.points) for state in self._ink_undo for s in state) > 400_000):
            self._ink_undo.pop(0)

    def undo_annotation(self):
        self.finish_annotation_gesture()
        if self._ink_undo:
            self._ink_redo.append(self.annotations)
            self.annotations = self._ink_undo.pop()
            self._ink_document_changed()

    def redo_annotation(self):
        self.finish_annotation_gesture()
        if self._ink_redo:
            self._push_ink_undo(self.annotations)
            self.annotations = self._ink_redo.pop()
            self._ink_document_changed()

    def clear_annotations(self):
        self.finish_annotation_gesture()
        if self.annotations:
            self._push_ink_undo(self.annotations)
            self._ink_redo.clear()
            self.annotations = ()
            self._ink_document_changed()

    def _ink_document_changed(self):
        self.ink_layer.set_strokes(self.annotations)
        self.annotationsChanged.emit()
        self.annotationHistoryChanged.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.annotation_editing() and not self._ink_space:
            self._begin_ink(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._ink_before is not None:
            self._continue_ink(event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._ink_before is not None:
            self._continue_ink(event.position())
            self.finish_annotation_gesture()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def tabletEvent(self, event):
        from PySide6.QtCore import QEvent
        if self.annotation_editing() and not self._ink_space:
            if event.type() == QEvent.Type.TabletPress:
                self._begin_ink(event.position())
            elif event.type() == QEvent.Type.TabletMove:
                self._continue_ink(event.position())
            elif event.type() == QEvent.Type.TabletRelease:
                self._continue_ink(event.position())
                self.finish_annotation_gesture()
            event.accept()  # Suppress synthesized mouse strokes; pressure is intentionally ignored.
        else:
            super().tabletEvent(event)

    def viewportEvent(self, event):
        from PySide6.QtCore import QEvent
        if event.type() in {QEvent.Type.TabletPress, QEvent.Type.TabletMove, QEvent.Type.TabletRelease}:
            self.tabletEvent(event)
            return event.isAccepted()
        return super().viewportEvent(event)


class AnnotationControls(QWidget):
    membershipChanged = Signal(int)

    def __init__(self, view, store, parent=None):
        super().__init__(parent)
        self.view = view
        self.store = AnnotationStore(store.database_path)
        self.image = None
        self.signature = None
        self.saved_strokes = ()
        self.dirty = False
        self.blocked = False
        self.error = ""
        self.setObjectName("previewAnnotationControls")
        row = WrappingLayout(self, spacing=5)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)
        self.tools = QButtonGroup(self)
        self.tool_buttons = {}
        for tool, title in (("pan", "移动"), ("pen", "画笔"), ("eraser", "橡皮")):
            button = QPushButton(title)
            button.setCheckable(True)
            button.setAutoDefault(False)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda checked=False, t=tool: self.set_tool(t))
            self.tools.addButton(button)
            self.tool_buttons[tool] = button
            row.addWidget(button)
        self.tool_buttons["pan"].setChecked(True)
        self.tool_buttons["pan"].setToolTip("拖动平移图片；画笔或橡皮模式下也可按住空格平移")
        self.colors = QComboBox()
        for title, color in (("红色", "#E53935"), ("黄色", "#FFC107"), ("蓝色", "#2196F3"),
                             ("白色", "#FFFFFF"), ("黑色", "#212121")):
            self.colors.addItem(title, color)
        self.colors.setToolTip("标记颜色")
        self.colors.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.colors.currentIndexChanged.connect(lambda i: setattr(view, "annotation_color", self.colors.itemData(i)))
        row.addWidget(self.colors)
        row.addWidget(QLabel("粗细"))
        self.width = QSlider(Qt.Orientation.Horizontal)
        self.width.setRange(2, 20)
        self.width.setValue(4)
        self.width.setFixedWidth(72)
        self.width.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.width.setToolTip("画笔粗细；橡皮大小随之调整，不使用压感")
        self.width.valueChanged.connect(lambda n: setattr(view, "annotation_width", n))
        row.addWidget(self.width)
        self.undo = QPushButton("撤销")
        self.redo = QPushButton("重做")
        self.clear = QPushButton("清空标记")
        self.visible = QPushButton("显示标记")
        self.visible.setCheckable(True)
        self.visible.setChecked(True)
        self.visible.toggled.connect(view.set_annotations_visible)
        self.undo.clicked.connect(view.undo_annotation)
        self.redo.clicked.connect(view.redo_annotation)
        self.clear.clicked.connect(view.clear_annotations)
        for button in (self.undo, self.redo, self.clear, self.visible):
            button.setAutoDefault(False)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            row.addWidget(button)
        self.status = QPushButton()
        self.status.setAutoDefault(False)
        self.status.clicked.connect(self.retry)
        self.status.hide()
        view.annotationsChanged.connect(self.changed)
        view.annotationHistoryChanged.connect(self.update_actions)

    def set_tool(self, tool):
        if tool != "pan":
            self.visible.setChecked(True)
        self.view.set_annotation_tool(tool)

    def update_actions(self):
        editable = self.view._ink_editable
        for tool in ("pen", "eraser"):
            self.tool_buttons[tool].setEnabled(editable)
        self.undo.setEnabled(editable and bool(self.view._ink_undo))
        self.redo.setEnabled(editable and bool(self.view._ink_redo))
        self.clear.setEnabled(editable and bool(self.view.annotations))
        self.colors.setEnabled(editable)
        self.width.setEnabled(editable)

    def load_image(self, image, *, enabled=True):
        if self.image is not None and image is not None and self.image.id == image.id:
            try:
                stat = Path(image.file_path).stat()
                if self.signature == (stat.st_size, stat.st_mtime_ns):
                    return True
            except OSError:
                pass
        if self.dirty and not self.flush(interactive=True):
            return False
        self.image, self.signature = image, None
        self.dirty, self.blocked, self.error = False, False, ""
        self.saved_strokes = ()
        self.status.hide()
        strokes = ()
        if image is not None and enabled and not image.is_missing:
            try:
                stat = Path(image.file_path).stat()
                self.signature = (stat.st_size, stat.st_mtime_ns)
                saved = self.store.load(image.id)
                if saved:
                    strokes, signature = saved
                    self.saved_strokes = strokes
                    if signature != self.signature:
                        strokes = ()
                        self.blocked = True
                        self.error = "原图已变化，旧标记已保留（点击清除）"
            except Exception as exc:
                self.blocked = True
                self.error = f"标记读取失败（点击重试）：{exc}"
        self.view.set_annotation_document(strokes, editable=bool(self.signature and enabled and not self.blocked))
        if self.error:
            self.status.setText("原图已变化（清除旧标记）" if self.error.startswith("原图已变化")
                                else "标记读取失败（点击重试）")
            self.status.setToolTip(self.error)
            self.status.show()
        self.update_actions()
        return True

    def changed(self):
        self.dirty = True
        self.flush()

    def flush(self, *, interactive=False):
        self.view.finish_annotation_gesture()
        if not self.dirty or self.image is None:
            return True
        try:
            stat = Path(self.image.file_path).stat()
            if (stat.st_size, stat.st_mtime_ns) != self.signature:
                raise ValueError("原图已在外部改变，未覆盖标记")
            self.store.save(self.image.id, self.view.annotations, self.signature, expected=self.saved_strokes)
            membership_changed = bool(self.saved_strokes) != bool(self.view.annotations)
            self.saved_strokes = self.view.annotations
            self.dirty = False
            self.error = ""
            self.status.hide()
            if membership_changed:
                self.membershipChanged.emit(self.image.id)
            return True
        except Exception as exc:
            self.error = str(exc)
            self.status.setText("标记未保存（点击重试）")
            self.status.setToolTip(self.error)
            self.status.show()
            if interactive:
                answer = QMessageBox.warning(self, "标记尚未保存", f"{self.error}\n保留此窗口可继续重试。放弃会丢失本次未保存的修改。",
                    QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel)
                if answer == QMessageBox.StandardButton.Retry:
                    return self.flush(interactive=True)
                if answer == QMessageBox.StandardButton.Discard:
                    self.dirty = False
                    return True
            return False

    def retry(self):
        if self.dirty:
            self.flush(interactive=True)
        elif self.blocked:
            if self.error.startswith("原图已变化"):
                if QMessageBox.question(self, "清除旧标记", "原图内容或尺寸已改变。清除保留的旧标记后，可以重新标记。",
                                        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                                        QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Ok:
                    return
                try:
                    self.store.save(self.image.id, (), self.signature, expected=self.saved_strokes)
                    if self.saved_strokes:
                        self.membershipChanged.emit(self.image.id)
                except Exception as exc:
                    self.status.setToolTip(str(exc))
                    return
            image = self.image
            self.image = None
            self.load_image(image)
