from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, QSizeF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFontMetrics,
    QImage,
    QImageReader,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QTransform,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from eidory.models import ImageItem
from eidory.ui.drop_import_box import payload_from_mime_data, payload_supports_mime_data


BOARD_WIDTH = 5200
BOARD_HEIGHT = 3600
MIN_ZOOM = 0.15
MAX_ZOOM = 16.0
GRID_SIZE = 36
PIXMAP_CACHE_LIMIT = 512
MAX_SOURCE_PIXMAP_SIDE = 2200
MAX_BOARD_THUMBNAIL_SIDE = 560
LAYOUT_CHANGED_IDLE_MS = 450
DEFAULT_LAYOUT_LEFT = 80.0
DEFAULT_LAYOUT_TOP = 110.0
DEFAULT_LAYOUT_ROW_WIDTH = 1600.0
DEFAULT_LAYOUT_GAP_X = 48.0
DEFAULT_LAYOUT_GAP_Y = 64.0
LOCAL_DISPLAY_MIN_SIZE = 0.01
LOCAL_DISPLAY_MAX_POINTS = 240


_BOARD_PIXMAP_CACHE: OrderedDict[tuple[str, int, int, int], QPixmap] = OrderedDict()
_BOARD_PIXMAP_CACHE_BYTES: dict[tuple[str, int, int, int], int] = {}
_BOARD_PIXMAP_CACHE_TOTAL_BYTES = 0
_BOARD_PIXMAP_CACHE_BYTE_LIMIT = 220 * 1024 * 1024


def _cached_pixmap(
    path: Path,
    *,
    max_side: int = MAX_SOURCE_PIXMAP_SIDE,
    modified_time_ns: int | None = None,
    file_size: int | None = None,
) -> QPixmap:
    global _BOARD_PIXMAP_CACHE_TOTAL_BYTES
    max_side = max(128, int(max_side))
    if modified_time_ns is None or file_size is None:
        try:
            stat = path.stat()
        except OSError:
            return QPixmap()
        modified_time_ns = int(stat.st_mtime_ns)
        file_size = int(stat.st_size)
    key = (str(path), int(modified_time_ns), int(file_size), max_side)
    cached = _BOARD_PIXMAP_CACHE.get(key)
    if cached is not None:
        _BOARD_PIXMAP_CACHE.move_to_end(key)
        return QPixmap(cached)

    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid() and max(size.width(), size.height()) > max_side:
        scaled_size = QSize(size.width(), size.height())
        scaled_size.scale(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio)
        reader.setScaledSize(scaled_size)
    image = reader.read()
    pixmap = QPixmap.fromImage(image) if not image.isNull() else QPixmap()
    if pixmap.isNull():
        return pixmap
    existing_bytes = _BOARD_PIXMAP_CACHE_BYTES.pop(key, 0)
    _BOARD_PIXMAP_CACHE_TOTAL_BYTES = max(0, _BOARD_PIXMAP_CACHE_TOTAL_BYTES - existing_bytes)
    _BOARD_PIXMAP_CACHE[key] = QPixmap(pixmap)
    _BOARD_PIXMAP_CACHE.move_to_end(key)
    pixmap_bytes = _estimated_pixmap_bytes(pixmap)
    _BOARD_PIXMAP_CACHE_BYTES[key] = pixmap_bytes
    _BOARD_PIXMAP_CACHE_TOTAL_BYTES += pixmap_bytes
    while (
        len(_BOARD_PIXMAP_CACHE) > PIXMAP_CACHE_LIMIT
        or _BOARD_PIXMAP_CACHE_TOTAL_BYTES > _BOARD_PIXMAP_CACHE_BYTE_LIMIT
    ):
        old_key, _old_pixmap = _BOARD_PIXMAP_CACHE.popitem(last=False)
        old_bytes = _BOARD_PIXMAP_CACHE_BYTES.pop(old_key, 0)
        _BOARD_PIXMAP_CACHE_TOTAL_BYTES = max(0, _BOARD_PIXMAP_CACHE_TOTAL_BYTES - old_bytes)
    return pixmap


def _estimated_pixmap_bytes(pixmap: QPixmap) -> int:
    if pixmap.isNull():
        return 0
    return max(0, pixmap.width()) * max(0, pixmap.height()) * 4


def _clamp_float(value: object, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number != number:
        return fallback
    return number


def _clamp01(value: object, fallback: float = 0.0) -> float:
    return max(0.0, min(1.0, _clamp_float(value, fallback)))


def _normalize_local_bounds(raw_bounds: object) -> dict[str, float] | None:
    if not isinstance(raw_bounds, dict):
        return None
    x = _clamp01(raw_bounds.get("x"), 0.0)
    y = _clamp01(raw_bounds.get("y"), 0.0)
    w = max(0.0, _clamp_float(raw_bounds.get("w"), 0.0))
    h = max(0.0, _clamp_float(raw_bounds.get("h"), 0.0))
    if x + w > 1.0:
        w = max(0.0, 1.0 - x)
    if y + h > 1.0:
        h = max(0.0, 1.0 - y)
    if w < LOCAL_DISPLAY_MIN_SIZE or h < LOCAL_DISPLAY_MIN_SIZE:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _bounds_from_local_points(points: list[list[float]]) -> dict[str, float] | None:
    if len(points) < 3:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x = max(0.0, min(xs))
    y = max(0.0, min(ys))
    right = min(1.0, max(xs))
    bottom = min(1.0, max(ys))
    bounds = {"x": x, "y": y, "w": right - x, "h": bottom - y}
    return _normalize_local_bounds(bounds)


def _normalize_local_points(raw_points: object) -> list[list[float]]:
    if not isinstance(raw_points, list):
        return []
    points: list[list[float]] = []
    last_point: tuple[float, float] | None = None
    for raw_point in raw_points[:LOCAL_DISPLAY_MAX_POINTS]:
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
            continue
        point = (_clamp01(raw_point[0]), _clamp01(raw_point[1]))
        if (
            last_point is not None
            and abs(last_point[0] - point[0]) < 0.001
            and abs(last_point[1] - point[1]) < 0.001
        ):
            continue
        points.append([point[0], point[1]])
        last_point = point
    if len(points) >= 3 and points[0] == points[-1]:
        points.pop()
    return points


def _normalize_local_display(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    display_type = str(payload.get("type") or "rect").strip().lower()
    enabled = bool(payload.get("enabled", True))
    normalized: dict[str, Any]
    if display_type == "polygon":
        points = _normalize_local_points(payload.get("points"))
        bounds = _bounds_from_local_points(points)
        if bounds is None:
            return None
        normalized = {
            "enabled": enabled,
            "type": "polygon",
            "points": points,
            "bounds": bounds,
        }
    else:
        bounds = _normalize_local_bounds(payload.get("bounds", payload))
        if bounds is None:
            return None
        normalized = {
            "enabled": enabled,
            "type": "rect",
            "bounds": bounds,
        }
    full_width = _clamp_float(payload.get("full_width"), 0.0)
    if full_width > 1:
        normalized["full_width"] = float(full_width)
    return normalized


def _local_bounds_width(payload: dict[str, Any] | None) -> float:
    if not isinstance(payload, dict):
        return 1.0
    bounds = _normalize_local_bounds(payload.get("bounds"))
    if bounds is None:
        return 1.0
    return max(LOCAL_DISPLAY_MIN_SIZE, float(bounds["w"]))


def _copy_local_display_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    copied: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            copied[key] = dict(value)
        elif isinstance(value, list):
            copied[key] = [list(item) if isinstance(item, list) else item for item in value]
        else:
            copied[key] = value
    return copied


class BoardImageItem(QGraphicsPixmapItem):
    MIN_DISPLAY_WIDTH = 90.0
    RESIZE_MARGIN = 12.0

    def __init__(
        self,
        *,
        image_id: int,
        pixmap: QPixmap,
        file_path: str,
        badges: list[str] | None = None,
        on_layout_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(pixmap)
        self.image_id = image_id
        self._source_pixmap = pixmap
        self._badges = self._normalize_badges(badges)
        self._on_layout_changed = on_layout_changed
        self._display_width = float(max(1, pixmap.width()))
        self._display_height = float(max(1, pixmap.height()))
        self._resize_zone = ""
        self._resizing = False
        self._resize_start_scene = QPointF()
        self._resize_start_width = self._display_width
        self._aspect_ratio = pixmap.width() / max(1, pixmap.height())
        self._pinned = False
        self._flipped = False
        self._grayscale = False
        self._local_display: dict[str, Any] | None = None
        self._display_variant_cache: OrderedDict[tuple[object, ...], QPixmap] = OrderedDict()
        self.setToolTip(file_path)
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.setCacheMode(QGraphicsItem.CacheMode.ItemCoordinateCache)
        self.setAcceptHoverEvents(True)
        self._apply_flags()

    @property
    def display_width(self) -> float:
        return self._display_width

    @property
    def display_height(self) -> float:
        return self._display_height

    @property
    def badge_text(self) -> str:
        if not self._badges:
            return ""
        if len(self._badges) == 1:
            return self._badges[0]
        return f"{self._badges[0]} +{len(self._badges) - 1}"

    @staticmethod
    def _normalize_badges(badges: list[str] | None) -> list[str]:
        normalized: list[str] = []
        for badge in badges or []:
            text = str(badge).strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    def set_display_size(
        self,
        width: float,
        _height: float | None = None,
        *,
        sync_local_full_width: bool = True,
    ) -> None:
        pixmap = self.pixmap()
        if pixmap.isNull():
            return
        width = max(self.MIN_DISPLAY_WIDTH, float(width))
        previous_width = self._display_width
        previous_height = self._display_height
        self.prepareGeometryChange()
        self._display_width = width
        self._display_height = width / max(0.001, self._aspect_ratio)
        self.setScale(self._display_width / max(1, pixmap.width()))
        if sync_local_full_width:
            self._sync_local_full_width_from_display_width()
        if (
            abs(previous_width - self._display_width) > 0.001
            or abs(previous_height - self._display_height) > 0.001
        ):
            self._notify_layout_changed()

    def set_pinned(self, pinned: bool) -> None:
        pinned = bool(pinned)
        if self._pinned == pinned:
            return
        self._pinned = pinned
        self._apply_flags()
        self._notify_layout_changed()

    def toggle_pinned(self) -> bool:
        self.set_pinned(not self._pinned)
        return self._pinned

    def is_pinned(self) -> bool:
        return self._pinned

    def set_flipped(self, flipped: bool) -> None:
        flipped = bool(flipped)
        if self._flipped == flipped:
            return
        self._flipped = flipped
        self._refresh_display_pixmap()
        self._notify_layout_changed()

    def toggle_flipped(self) -> None:
        self.set_flipped(not self._flipped)

    def is_flipped(self) -> bool:
        return self._flipped

    def set_grayscale(self, grayscale: bool) -> None:
        grayscale = bool(grayscale)
        if self._grayscale == grayscale:
            return
        self._grayscale = grayscale
        self._refresh_display_pixmap()
        self._notify_layout_changed()

    def toggle_grayscale(self) -> None:
        self.set_grayscale(not self._grayscale)

    def is_grayscale(self) -> bool:
        return self._grayscale

    def local_display_payload(self) -> dict[str, Any] | None:
        return _copy_local_display_payload(self._local_display)

    def has_local_display(self) -> bool:
        return self._local_display is not None

    def is_local_display_enabled(self) -> bool:
        return bool(self._local_display and self._local_display.get("enabled", True))

    def set_local_display(
        self,
        payload: object,
        *,
        shrink_to_bounds: bool = False,
    ) -> bool:
        normalized = _normalize_local_display(payload)
        if normalized is None:
            return self.clear_local_display()
        full_width = _clamp_float(normalized.get("full_width"), 0.0)
        if full_width <= 1:
            full_width = self._full_display_width_guess()
        normalized["full_width"] = max(self.MIN_DISPLAY_WIDTH, float(full_width))
        self._local_display = normalized
        target_width = self._display_width
        if shrink_to_bounds:
            target_width = self._target_width_for_local_enabled(bool(normalized.get("enabled", True)))
        self._refresh_display_pixmap()
        if abs(self._display_width - target_width) > 0.001:
            self.set_display_size(target_width, sync_local_full_width=False)
        self._notify_layout_changed()
        return True

    def set_local_display_enabled(self, enabled: bool) -> bool:
        if self._local_display is None:
            return False
        enabled = bool(enabled)
        if bool(self._local_display.get("enabled", True)) == enabled:
            return False
        self._local_display["full_width"] = self._full_display_width_guess()
        self._local_display["enabled"] = enabled
        target_width = self._target_width_for_local_enabled(enabled)
        self._refresh_display_pixmap()
        self.set_display_size(target_width, sync_local_full_width=False)
        self._notify_layout_changed()
        return True

    def toggle_local_display_enabled(self) -> bool:
        if self._local_display is None:
            return False
        return self.set_local_display_enabled(not self.is_local_display_enabled())

    def clear_local_display(self) -> bool:
        if self._local_display is None:
            return False
        target_width = self._full_display_width_guess()
        self._local_display = None
        self._refresh_display_pixmap()
        self.set_display_size(target_width, sync_local_full_width=False)
        self._notify_layout_changed()
        return True

    def source_pixmap(self) -> QPixmap:
        return QPixmap(self._source_pixmap)

    def _apply_flags(self) -> None:
        flags = (
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        if not self._pinned:
            flags |= QGraphicsItem.GraphicsItemFlag.ItemIsMovable
        self.setFlags(flags)

    def _refresh_display_pixmap(self) -> None:
        pixmap = self._source_pixmap
        if pixmap.isNull():
            return
        cache_key = self._display_variant_cache_key()
        cached = self._display_variant_cache.get(cache_key)
        if cached is not None and not cached.isNull():
            self._display_variant_cache.move_to_end(cache_key)
            pixmap = QPixmap(cached)
        else:
            pixmap = self._local_display_pixmap(pixmap)
            if self._grayscale:
                image = pixmap.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
                pixmap = QPixmap.fromImage(image.convertToFormat(QImage.Format.Format_ARGB32))
            if self._flipped:
                pixmap = pixmap.transformed(QTransform().scale(-1, 1), Qt.TransformationMode.SmoothTransformation)
            self._display_variant_cache[cache_key] = QPixmap(pixmap)
            self._display_variant_cache.move_to_end(cache_key)
            while len(self._display_variant_cache) > 4:
                self._display_variant_cache.popitem(last=False)
        current_width = self._display_width
        self._aspect_ratio = pixmap.width() / max(1, pixmap.height())
        self.setPixmap(pixmap)
        self.set_display_size(current_width, sync_local_full_width=False)

    def _display_variant_cache_key(self) -> tuple[object, ...]:
        local_payload = _normalize_local_display(self._local_display)
        local_json = json.dumps(local_payload, sort_keys=True, separators=(",", ":")) if local_payload else ""
        return (
            int(self._source_pixmap.cacheKey()),
            bool(self._grayscale),
            bool(self._flipped),
            local_json,
        )

    def _local_display_pixmap(self, pixmap: QPixmap) -> QPixmap:
        if pixmap.isNull() or not self.is_local_display_enabled():
            return pixmap
        payload = _normalize_local_display(self._local_display)
        if payload is None:
            return pixmap
        bounds = payload.get("bounds")
        if not isinstance(bounds, dict):
            return pixmap
        source_width = max(1, pixmap.width())
        source_height = max(1, pixmap.height())
        left = max(0, min(source_width - 1, int(round(float(bounds["x"]) * source_width))))
        top = max(0, min(source_height - 1, int(round(float(bounds["y"]) * source_height))))
        right = max(
            left + 1,
            min(source_width, int(round((float(bounds["x"]) + float(bounds["w"])) * source_width))),
        )
        bottom = max(
            top + 1,
            min(source_height, int(round((float(bounds["y"]) + float(bounds["h"])) * source_height))),
        )
        width = max(1, right - left)
        height = max(1, bottom - top)
        if payload.get("type") != "polygon":
            return pixmap.copy(left, top, width, height)
        points = payload.get("points")
        if not isinstance(points, list) or len(points) < 3:
            return pixmap.copy(left, top, width, height)
        output = QPixmap(width, height)
        output.fill(Qt.GlobalColor.transparent)
        path = QPainterPath()
        first = True
        for raw_point in points:
            if not isinstance(raw_point, list) or len(raw_point) < 2:
                continue
            point = QPointF(
                _clamp01(raw_point[0]) * source_width - left,
                _clamp01(raw_point[1]) * source_height - top,
            )
            if first:
                path.moveTo(point)
                first = False
            else:
                path.lineTo(point)
        if first:
            return pixmap.copy(left, top, width, height)
        path.closeSubpath()
        painter = QPainter(output)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setClipPath(path)
        painter.drawPixmap(-left, -top, pixmap)
        painter.end()
        return output

    def _full_display_width_guess(self) -> float:
        if self._local_display is None:
            return max(self.MIN_DISPLAY_WIDTH, self._display_width)
        stored_width = _clamp_float(self._local_display.get("full_width"), 0.0)
        if stored_width > 1:
            return max(self.MIN_DISPLAY_WIDTH, stored_width)
        if self.is_local_display_enabled():
            return max(self.MIN_DISPLAY_WIDTH, self._display_width / _local_bounds_width(self._local_display))
        return max(self.MIN_DISPLAY_WIDTH, self._display_width)

    def _target_width_for_local_enabled(self, enabled: bool) -> float:
        full_width = self._full_display_width_guess()
        if not enabled:
            return full_width
        return max(self.MIN_DISPLAY_WIDTH, full_width * _local_bounds_width(self._local_display))

    def _sync_local_full_width_from_display_width(self) -> None:
        if self._local_display is None:
            return
        if self.is_local_display_enabled():
            full_width = self._display_width / _local_bounds_width(self._local_display)
        else:
            full_width = self._display_width
        self._local_display["full_width"] = max(self.MIN_DISPLAY_WIDTH, float(full_width))

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._notify_layout_changed()
        return result

    def _notify_layout_changed(self) -> None:
        if self._on_layout_changed is not None:
            self._on_layout_changed()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        text = self.badge_text
        if not text:
            return
        rect = self.boundingRect()
        transform = painter.worldTransform()
        bottom_left = transform.map(rect.bottomLeft())
        bottom_right = transform.map(rect.bottomRight())
        top_left = transform.map(rect.topLeft())
        image_width = abs(bottom_right.x() - bottom_left.x())
        image_height = abs(bottom_left.y() - top_left.y())
        if image_width < 36 or image_height < 18:
            return
        painter.save()
        painter.resetTransform()
        font = painter.font()
        font.setPointSizeF(10.0)
        font.setBold(True)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        max_text_width = max(28, int(min(image_width - 12, 220)))
        text = metrics.elidedText(text, Qt.TextElideMode.ElideRight, max_text_width)
        padding_x = 6.0
        padding_y = 3.0
        text_width = metrics.horizontalAdvance(text)
        text_height = metrics.height()
        badge_width = min(image_width, text_width + padding_x * 2)
        badge_height = text_height + padding_y * 2
        badge_rect = QRectF(
            bottom_left.x(),
            bottom_left.y() - badge_height,
            badge_width,
            badge_height,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(20, 24, 30, 220))
        painter.drawRect(badge_rect)
        painter.setPen(QColor("#eef2f8"))
        painter.drawText(
            QRectF(
                badge_rect.left() + padding_x,
                badge_rect.top() + padding_y,
                badge_rect.width() - padding_x * 2,
                text_height,
            ),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            text,
        )
        painter.restore()

    def hoverMoveEvent(self, event) -> None:
        if self._pinned:
            self.unsetCursor()
            super().hoverMoveEvent(event)
            return
        self._resize_zone = self._resize_zone_at(event.pos())
        if self._resize_zone in {"corner", "right", "bottom"}:
            if self._resize_zone == "right":
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif self._resize_zone == "bottom":
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            else:
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if not self._resizing:
            self._resize_zone = ""
            self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if self._pinned:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            zone = self._resize_zone_at(event.pos())
            if zone in {"corner", "right", "bottom"}:
                self._resize_zone = zone
                self._resizing = True
                self._resize_start_scene = event.scenePos()
                self._resize_start_width = self._display_width
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._resizing:
            delta = event.scenePos() - self._resize_start_scene
            if self._resize_zone == "bottom":
                new_width = self._resize_start_width + delta.y() * self._aspect_ratio
            elif self._resize_zone == "corner":
                new_width = self._resize_start_width + max(delta.x(), delta.y() * self._aspect_ratio)
            else:
                new_width = self._resize_start_width + delta.x()
            self.set_display_size(new_width)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._resizing:
            self._resizing = False
            self._resize_zone = ""
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _resize_zone_at(self, pos: QPointF) -> str:
        rect = self.boundingRect()
        margin = self.RESIZE_MARGIN / max(0.001, self.scale())
        near_right = rect.right() - pos.x() <= margin
        near_bottom = rect.bottom() - pos.y() <= margin
        if near_right and near_bottom:
            return "corner"
        if near_right:
            return "right"
        if near_bottom:
            return "bottom"
        return ""


class BoardPlaceholderItem(QGraphicsRectItem):
    def __init__(
        self,
        rect: QRectF,
        *,
        on_layout_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(rect)
        self._on_layout_changed = on_layout_changed

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._notify_layout_changed()
        return result

    def _notify_layout_changed(self) -> None:
        if self._on_layout_changed is not None:
            self._on_layout_changed()


class LocalDisplayEditorCanvas(QWidget):
    selectionCompleted = Signal()

    def __init__(self, pixmap: QPixmap, payload: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._pixmap = pixmap
        self._mode = "polygon"
        self._dragging_rect = False
        self._rect_start: QPointF | None = None
        self._rect_end: QPointF | None = None
        self._polygon_points: list[QPointF] = []
        self._dragging_polygon_point_index: int | None = None
        self._cursor_source_pos: QPointF | None = None
        self.setMinimumSize(720, 520)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.set_payload(payload)

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        mode = "rect" if mode == "rect" else "polygon"
        if self._mode == mode:
            return
        self._mode = mode
        self.update()

    def set_payload(self, payload: dict[str, Any] | None) -> None:
        normalized = _normalize_local_display(payload)
        if normalized is None or self._pixmap.isNull():
            return
        self._mode = "rect" if normalized.get("type") == "rect" else "polygon"
        bounds = normalized.get("bounds", {})
        if isinstance(bounds, dict):
            self._rect_start = QPointF(
                float(bounds["x"]) * self._pixmap.width(),
                float(bounds["y"]) * self._pixmap.height(),
            )
            self._rect_end = QPointF(
                (float(bounds["x"]) + float(bounds["w"])) * self._pixmap.width(),
                (float(bounds["y"]) + float(bounds["h"])) * self._pixmap.height(),
            )
        if normalized.get("type") == "polygon":
            points = normalized.get("points", [])
            if isinstance(points, list):
                self._polygon_points = [
                    QPointF(
                        _clamp01(point[0]) * self._pixmap.width(),
                        _clamp01(point[1]) * self._pixmap.height(),
                    )
                    for point in points
                    if isinstance(point, list) and len(point) >= 2
                ]
        self.update()

    def clear_selection(self) -> None:
        self._rect_start = None
        self._rect_end = None
        self._polygon_points = []
        self._dragging_polygon_point_index = None
        self._cursor_source_pos = None
        self.update()

    def remove_last_polygon_point(self) -> bool:
        if self._mode != "polygon" or not self._polygon_points:
            return False
        self._polygon_points.pop()
        self.update()
        return True

    def has_selection(self) -> bool:
        return self.selection_payload() is not None

    def selection_payload(self) -> dict[str, Any] | None:
        if self._pixmap.isNull():
            return None
        source_width = max(1.0, float(self._pixmap.width()))
        source_height = max(1.0, float(self._pixmap.height()))
        if self._mode == "rect":
            rect = self._normalized_rect()
            if rect is None:
                return None
            return _normalize_local_display(
                {
                    "enabled": True,
                    "type": "rect",
                    "bounds": {
                        "x": rect.left() / source_width,
                        "y": rect.top() / source_height,
                        "w": rect.width() / source_width,
                        "h": rect.height() / source_height,
                    },
                }
            )
        if len(self._polygon_points) < 3:
            return None
        points = [
            [point.x() / source_width, point.y() / source_height]
            for point in self._polygon_points[:LOCAL_DISPLAY_MAX_POINTS]
        ]
        return _normalize_local_display({"enabled": True, "type": "polygon", "points": points})

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#242a31"))
        display_rect = self._display_rect()
        if display_rect.isNull() or self._pixmap.isNull():
            painter.setPen(QColor("#dfe6ef"))
            painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), "无法载入图片")
            painter.end()
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(
            display_rect,
            self._pixmap,
            QRectF(0, 0, self._pixmap.width(), self._pixmap.height()),
        )
        selection_path = self._selection_path(display_rect)
        if selection_path is not None:
            outer = QPainterPath()
            outer.addRect(display_rect)
            overlay = outer.subtracted(selection_path)
            painter.fillPath(overlay, QColor(12, 16, 22, 154))
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(QColor("#4f83ff"), 2))
            painter.drawPath(selection_path)
        if self._mode == "polygon":
            self._paint_polygon_points(painter, display_rect)
        painter.setPen(QPen(QColor("#657386"), 1))
        painter.drawRect(display_rect)
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        source_pos = self._source_pos_for_widget(event.position())
        if source_pos is None:
            super().mousePressEvent(event)
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if self._mode == "rect":
            self._dragging_rect = True
            self._rect_start = source_pos
            self._rect_end = source_pos
        else:
            point_index = self._nearest_polygon_point_index(source_pos)
            if point_index is not None:
                self._dragging_polygon_point_index = point_index
                self._polygon_points[point_index] = source_pos
            elif len(self._polygon_points) < LOCAL_DISPLAY_MAX_POINTS:
                self._polygon_points.append(source_pos)
        self.update()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        source_pos = self._source_pos_for_widget(event.position())
        self._cursor_source_pos = source_pos
        if self._mode == "rect" and self._dragging_rect and source_pos is not None:
            self._rect_end = source_pos
            self.update()
            event.accept()
            return
        if self._mode == "polygon" and self._dragging_polygon_point_index is not None and source_pos is not None:
            self._polygon_points[self._dragging_polygon_point_index] = source_pos
            self.update()
            event.accept()
            return
        if self._mode == "polygon":
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._mode == "rect" and self._dragging_rect and event.button() == Qt.MouseButton.LeftButton:
            source_pos = self._source_pos_for_widget(event.position())
            if source_pos is not None:
                self._rect_end = source_pos
            self._dragging_rect = False
            self.update()
            event.accept()
            return
        if self._mode == "polygon" and self._dragging_polygon_point_index is not None:
            source_pos = self._source_pos_for_widget(event.position())
            if source_pos is not None:
                self._polygon_points[self._dragging_polygon_point_index] = source_pos
            self._dragging_polygon_point_index = None
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if (
            self._mode == "polygon"
            and event.button() == Qt.MouseButton.LeftButton
            and len(self._polygon_points) >= 3
        ):
            self.selectionCompleted.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter} and self.has_selection():
            self.selectionCompleted.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Backspace and self.remove_last_polygon_point():
            event.accept()
            return
        super().keyPressEvent(event)

    def _display_rect(self) -> QRectF:
        if self._pixmap.isNull():
            return QRectF()
        available = QRectF(self.rect()).adjusted(14, 14, -14, -14)
        target = QSizeF(self._pixmap.size())
        target.scale(available.size(), Qt.AspectRatioMode.KeepAspectRatio)
        left = available.left() + (available.width() - target.width()) / 2
        top = available.top() + (available.height() - target.height()) / 2
        return QRectF(left, top, target.width(), target.height())

    def _source_pos_for_widget(self, widget_pos: QPointF) -> QPointF | None:
        display_rect = self._display_rect()
        if display_rect.isNull() or not display_rect.contains(widget_pos):
            return None
        x = (widget_pos.x() - display_rect.left()) / max(1.0, display_rect.width()) * self._pixmap.width()
        y = (widget_pos.y() - display_rect.top()) / max(1.0, display_rect.height()) * self._pixmap.height()
        return QPointF(
            max(0.0, min(float(self._pixmap.width()), x)),
            max(0.0, min(float(self._pixmap.height()), y)),
        )

    def _widget_pos_for_source(self, source_pos: QPointF, display_rect: QRectF) -> QPointF:
        x = display_rect.left() + source_pos.x() / max(1.0, self._pixmap.width()) * display_rect.width()
        y = display_rect.top() + source_pos.y() / max(1.0, self._pixmap.height()) * display_rect.height()
        return QPointF(x, y)

    def _nearest_polygon_point_index(self, source_pos: QPointF) -> int | None:
        if self._mode != "polygon" or not self._polygon_points:
            return None
        tolerance = max(10.0, min(self._pixmap.width(), self._pixmap.height()) * 0.018)
        best_index: int | None = None
        best_distance = tolerance
        for index, point in enumerate(self._polygon_points):
            distance = ((point.x() - source_pos.x()) ** 2 + (point.y() - source_pos.y()) ** 2) ** 0.5
            if distance <= best_distance:
                best_index = index
                best_distance = distance
        return best_index

    def _normalized_rect(self) -> QRectF | None:
        if self._rect_start is None or self._rect_end is None:
            return None
        left = max(0.0, min(self._rect_start.x(), self._rect_end.x()))
        top = max(0.0, min(self._rect_start.y(), self._rect_end.y()))
        right = min(float(self._pixmap.width()), max(self._rect_start.x(), self._rect_end.x()))
        bottom = min(float(self._pixmap.height()), max(self._rect_start.y(), self._rect_end.y()))
        rect = QRectF(left, top, right - left, bottom - top)
        if rect.width() < 2 or rect.height() < 2:
            return None
        return rect

    def _selection_path(self, display_rect: QRectF) -> QPainterPath | None:
        if self._mode == "rect":
            rect = self._normalized_rect()
            if rect is None:
                return None
            top_left = self._widget_pos_for_source(rect.topLeft(), display_rect)
            bottom_right = self._widget_pos_for_source(rect.bottomRight(), display_rect)
            path = QPainterPath()
            path.addRect(QRectF(top_left, bottom_right).normalized())
            return path
        points = list(self._polygon_points)
        if self._cursor_source_pos is not None and points:
            points.append(self._cursor_source_pos)
        if len(points) < 2:
            return None
        path = QPainterPath()
        path.moveTo(self._widget_pos_for_source(points[0], display_rect))
        for point in points[1:]:
            path.lineTo(self._widget_pos_for_source(point, display_rect))
        if len(self._polygon_points) >= 3:
            path.closeSubpath()
        return path

    def _paint_polygon_points(self, painter: QPainter, display_rect: QRectF) -> None:
        if not self._polygon_points:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("#d8e5ff"), 1))
        painter.setBrush(QColor("#4f83ff"))
        for point in self._polygon_points:
            center = self._widget_pos_for_source(point, display_rect)
            painter.drawEllipse(center, 4.0, 4.0)
        painter.restore()


class LocalDisplayEditorDialog(QDialog):
    def __init__(
        self,
        pixmap: QPixmap,
        payload: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑局部展示")
        self.setModal(True)
        self._canvas = LocalDisplayEditorCanvas(pixmap, payload)
        self._selected_payload: dict[str, Any] | None = None
        self._clear_requested = False
        has_existing_payload = _normalize_local_display(payload) is not None

        self._rect_button = QPushButton("矩形选区")
        self._polygon_button = QPushButton("多边形选区")
        for button in [self._rect_button, self._polygon_button]:
            button.setCheckable(True)
            button.setMinimumWidth(120)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self._rect_button)
        self._mode_group.addButton(self._polygon_button)
        self._rect_button.clicked.connect(lambda: self._set_mode("rect"))
        self._polygon_button.clicked.connect(lambda: self._set_mode("polygon"))
        if self._canvas.mode() == "rect":
            self._rect_button.setChecked(True)
        else:
            self._polygon_button.setChecked(True)

        self._hint_label = QLabel()
        self._hint_label.setWordWrap(True)
        self._refresh_hint()

        reset_button = QPushButton("重画")
        clear_button = QPushButton("清除局部展示")
        cancel_button = QPushButton("取消")
        apply_button = QPushButton("应用")
        reset_button.clicked.connect(self._canvas.clear_selection)
        clear_button.clicked.connect(self._request_clear)
        clear_button.setEnabled(has_existing_payload)
        cancel_button.clicked.connect(self.reject)
        apply_button.clicked.connect(self._accept_selection)
        self._canvas.selectionCompleted.connect(self._accept_selection)

        mode_row = QHBoxLayout()
        mode_row.addWidget(self._rect_button)
        mode_row.addWidget(self._polygon_button)
        mode_row.addStretch(1)

        action_row = QHBoxLayout()
        action_row.addWidget(reset_button)
        action_row.addWidget(clear_button)
        action_row.addStretch(1)
        action_row.addWidget(cancel_button)
        action_row.addWidget(apply_button)

        layout = QVBoxLayout(self)
        layout.addLayout(mode_row)
        layout.addWidget(self._hint_label)
        layout.addWidget(self._canvas, 1)
        layout.addLayout(action_row)
        self.resize(900, 680)

    def selected_payload(self) -> dict[str, Any] | None:
        return _copy_local_display_payload(self._selected_payload)

    def clear_requested(self) -> bool:
        return self._clear_requested

    def keyPressEvent(self, event) -> None:
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            self._accept_selection()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Backspace:
            if self._canvas.remove_last_polygon_point():
                event.accept()
                return
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            event.accept()
            return
        super().keyPressEvent(event)

    def _set_mode(self, mode: str) -> None:
        self._canvas.set_mode(mode)
        if mode == "rect":
            self._rect_button.setChecked(True)
        else:
            self._polygon_button.setChecked(True)
        self._refresh_hint()

    def _refresh_hint(self) -> None:
        if self._canvas.mode() == "rect":
            text = "拖拽出矩形区域，点击应用。"
        else:
            text = "单击添加多边形点；Enter 或双击应用；Backspace 删除上一个点。"
        self._hint_label.setText(text)

    def _accept_selection(self) -> None:
        payload = self._canvas.selection_payload()
        if payload is None:
            self._hint_label.setText("还没有有效选区。")
            return
        self._selected_payload = payload
        self.accept()

    def _request_clear(self) -> None:
        self._selected_payload = None
        self._clear_requested = True
        self.accept()


class ProjectBoardView(QGraphicsView):
    imageDoubleClicked = Signal(int)
    imagePreviewRequested = Signal(int)
    selectionChanged = Signal(list)
    layoutChanged = Signal()
    removeImagesRequested = Signal(list)
    undoRemovalRequested = Signal()
    filesDropped = Signal(object)
    dropPayloadDropped = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#202830"))
        self._image_items: dict[int, QGraphicsItem] = {}
        self._title_item: QGraphicsTextItem | None = None
        self._zoom = 1.0
        self._space_pan_active = False
        self._space_pan_moved = False
        self._panning = False
        self._last_pan_pos = QPoint()
        self._view_mode = "fit_all"
        self._last_fit_selection_ids: tuple[int, ...] = ()
        self._selected_image_ids_cache: list[int] = []
        self._refit_timer_pending = False
        self._suppress_layout_changed = False
        self._layout_change_pending = False
        self._layout_change_timer = QTimer(self)
        self._layout_change_timer.setSingleShot(True)
        self._layout_change_timer.setInterval(LAYOUT_CHANGED_IDLE_MS)
        self._layout_change_timer.timeout.connect(self._emit_pending_layout_changed)
        self._scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self._scene.selectionChanged.connect(self._emit_selection_changed)

    def set_images(
        self,
        images: list[ImageItem],
        *,
        title: str = "",
        layout_payload: dict[str, Any] | None = None,
        badges_by_image_id: dict[int, list[str]] | None = None,
    ) -> None:
        normalized_badges = {
            int(image_id): [str(badge) for badge in badges]
            for image_id, badges in (badges_by_image_id or {}).items()
        }
        self.setUpdatesEnabled(False)
        previous_scene_signal_state = self._scene.blockSignals(True)
        previous_selected_ids = list(self._selected_image_ids_cache)
        self._suppress_layout_changed = True
        try:
            self._scene.clear()
            self._image_items.clear()
            self._selected_image_ids_cache = []
            self._scene.setSceneRect(0, 0, BOARD_WIDTH, BOARD_HEIGHT)
            self.resetTransform()
            self._zoom = 1.0
            self._view_mode = "fit_all"
            self._last_fit_selection_ids = ()

            stored_items = {}
            if isinstance(layout_payload, dict):
                raw_items = layout_payload.get("items", {})
                if isinstance(raw_items, dict):
                    stored_items = raw_items

            self._title_item = None
            if title:
                self._title_item = self._scene.addText(title)
                self._title_item.setDefaultTextColor(QColor("#eef2f8"))
                font = self._title_item.font()
                font.setPointSize(18)
                font.setBold(True)
                self._title_item.setFont(font)
                self._title_item.setPos(56, 36)

            default_x = DEFAULT_LAYOUT_LEFT
            default_y = DEFAULT_LAYOUT_TOP
            row_height = 0.0
            row_right = DEFAULT_LAYOUT_LEFT + DEFAULT_LAYOUT_ROW_WIDTH
            stored_positions: set[tuple[int, int]] = set()
            has_duplicate_stored_position = False
            for image in images:
                payload = stored_items.get(str(image.id), {})
                if not isinstance(payload, dict):
                    continue
                position = (
                    int(round(_safe_float(payload.get("x"), DEFAULT_LAYOUT_LEFT))),
                    int(round(_safe_float(payload.get("y"), DEFAULT_LAYOUT_TOP))),
                )
                if position in stored_positions:
                    has_duplicate_stored_position = True
                    break
                stored_positions.add(position)
            if (
                has_duplicate_stored_position
                or any(not isinstance(stored_items.get(str(image.id)), dict) for image in images)
            ):
                stored_bottom = DEFAULT_LAYOUT_TOP
                for image in images:
                    payload = stored_items.get(str(image.id), {})
                    if not isinstance(payload, dict):
                        continue
                    y = _safe_float(payload.get("y"), DEFAULT_LAYOUT_TOP)
                    height = max(80.0, _safe_float(payload.get("height"), 160.0))
                    stored_bottom = max(stored_bottom, y + height)
                if stored_bottom > DEFAULT_LAYOUT_TOP:
                    default_y = stored_bottom + DEFAULT_LAYOUT_GAP_Y
            placed_positions: set[tuple[int, int]] = set()
            for index, image in enumerate(images):
                item_payload = stored_items.get(str(image.id), {})
                pixmap = self._pixmap_for(
                    image,
                    prefer_source=False,
                    max_side=MAX_BOARD_THUMBNAIL_SIDE,
                )
                item_width, item_height = self._default_item_size(pixmap)
                if default_x > DEFAULT_LAYOUT_LEFT and default_x + item_width > row_right:
                    default_x = DEFAULT_LAYOUT_LEFT
                    default_y += row_height + DEFAULT_LAYOUT_GAP_Y
                    row_height = 0.0
                if isinstance(item_payload, dict):
                    x = _safe_float(item_payload.get("x"), default_x)
                    y = _safe_float(item_payload.get("y"), default_y)
                    item_width = max(80.0, _safe_float(item_payload.get("width"), item_width))
                    item_height = max(80.0, _safe_float(item_payload.get("height"), item_height))
                    position = (int(round(x)), int(round(y)))
                    if position in placed_positions:
                        item_payload = {}
                        x = float(default_x)
                        y = float(default_y)
                else:
                    x = float(default_x)
                    y = float(default_y)
                self._add_image_item(
                    image,
                    pixmap,
                    x,
                    y,
                    item_width,
                    item_height,
                    normalized_badges.get(image.id, []),
                )
                self._apply_item_payload_state(image.id, item_payload)
                placed_positions.add((int(round(x)), int(round(y))))
                row_height = max(row_height, item_height)
                default_x += item_width + DEFAULT_LAYOUT_GAP_X

            if not images:
                text = self._scene.addText("当前节点没有保存图片")
                text.setDefaultTextColor(QColor("#d8dee9"))
                text.setPos(80, 110)
        finally:
            self.setUpdatesEnabled(True)
            self._scene.blockSignals(previous_scene_signal_state)
            self._finish_layout_reset()

        if previous_selected_ids:
            self.selectionChanged.emit([])
        if images:
            self._schedule_refit_current_view_mode()

    def fit_all_images(self) -> None:
        self._view_mode = "fit_all"
        self._last_fit_selection_ids = ()
        self._fit_items_to_view(self._visible_image_items(), margin=80.0)

    def hide_selected_items(self) -> int:
        items = self._selected_image_items()
        for item in items:
            item.setVisible(False)
            item.setSelected(False)
        if items:
            self._schedule_layout_changed()
        return len(items)

    def show_all_items(self) -> None:
        changed = any(not item.isVisible() for item in self._image_items.values())
        for item in self._image_items.values():
            item.setVisible(True)
        if changed:
            self._schedule_layout_changed()
        self.fit_all_images()

    def toggle_selected_pinned(self) -> int:
        changed = 0
        for item in self._selected_image_items():
            if isinstance(item, BoardImageItem):
                item.toggle_pinned()
                changed += 1
        return changed

    def toggle_selected_flipped(self) -> int:
        changed = 0
        for item in self._selected_image_items():
            if isinstance(item, BoardImageItem):
                item.toggle_flipped()
                changed += 1
        return changed

    def toggle_selected_grayscale(self) -> int:
        changed = 0
        for item in self._selected_image_items():
            if isinstance(item, BoardImageItem):
                item.toggle_grayscale()
                changed += 1
        return changed

    def edit_selected_local_display(self) -> bool:
        selected = [item for item in self._selected_image_items() if isinstance(item, BoardImageItem)]
        if not selected:
            return False
        return self._edit_local_display_item(selected[0])

    def toggle_selected_local_display(self) -> int:
        changed = 0
        for item in self._selected_image_items():
            if isinstance(item, BoardImageItem) and item.toggle_local_display_enabled():
                changed += 1
        return changed

    def clear_selected_local_display(self) -> int:
        changed = 0
        for item in self._selected_image_items():
            if isinstance(item, BoardImageItem) and item.clear_local_display():
                changed += 1
        return changed

    def set_title_visible(self, visible: bool) -> None:
        if self._title_item is not None:
            self._title_item.setVisible(visible)

    def layout_payload(self) -> dict[str, Any]:
        self.flush_pending_layout_change()
        items: dict[str, dict[str, Any]] = {}
        for image_id, item in self._image_items.items():
            if isinstance(item, BoardImageItem):
                width = item.display_width
                height = item.display_height
            else:
                rect = item.mapToScene(item.boundingRect()).boundingRect()
                width = rect.width()
                height = rect.height()
            items[str(image_id)] = {
                "x": float(item.pos().x()),
                "y": float(item.pos().y()),
                "width": float(width),
                "height": float(height),
                "visible": bool(item.isVisible()),
            }
            if isinstance(item, BoardImageItem):
                items[str(image_id)].update(
                    {
                        "pinned": item.is_pinned(),
                        "flipped": item.is_flipped(),
                        "grayscale": item.is_grayscale(),
                    }
                )
                local_display = item.local_display_payload()
                if local_display is not None:
                    items[str(image_id)]["local_display"] = local_display
        return {"version": 1, "items": items}

    def fit_visible_images(self) -> None:
        selected_items = self._selected_image_items()
        if selected_items:
            self._fit_selected_items(selected_items)
        else:
            self.fit_all_images()

    def toggle_fit_selection_or_restore(self) -> None:
        selected_ids = tuple(sorted(self.selected_image_ids()))
        if (
            selected_ids
            and self._view_mode == "fit_selection"
            and selected_ids == self._last_fit_selection_ids
        ):
            self.fit_all_images()
            return
        selected_items = self._selected_image_items()
        if selected_items:
            self._fit_selected_items(selected_items)
            return
        self.fit_all_images()

    def reset_view(self) -> None:
        self.resetTransform()
        self._zoom = 1.0
        self._view_mode = "manual"
        self._last_fit_selection_ids = ()
        self.centerOn(BOARD_WIDTH / 2, BOARD_HEIGHT / 2)

    def wheelEvent(self, event) -> None:
        modifiers = event.modifiers()
        zoom_modifier = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        if modifiers & zoom_modifier:
            raw_delta = event.angleDelta().y()
            if raw_delta == 0:
                raw_delta = event.pixelDelta().y()
            delta = max(-160, min(160, raw_delta))
            if delta == 0:
                event.accept()
                return
            factor = 1.12 if delta > 0 else 1 / 1.12
            self._view_mode = "manual"
            self._last_fit_selection_ids = ()
            self._zoom_by(factor)
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event) -> None:
        undo_modifier = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        if event.matches(QKeySequence.StandardKey.Undo) or (
            event.key() == Qt.Key.Key_Z and bool(event.modifiers() & undo_modifier)
        ):
            self.undoRemovalRequested.emit()
            event.accept()
            return
        if event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}:
            image_ids = self.selected_image_ids()
            if image_ids:
                self.removeImagesRequested.emit(image_ids)
                event.accept()
                return
        if event.key() == Qt.Key.Key_C and not event.isAutoRepeat():
            modifiers = event.modifiers()
            blocked_modifiers = (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.MetaModifier
                | Qt.KeyboardModifier.AltModifier
            )
            if not bool(modifiers & blocked_modifiers):
                if modifiers & Qt.KeyboardModifier.ShiftModifier:
                    if self.toggle_selected_local_display():
                        event.accept()
                        return
                elif self.edit_selected_local_display():
                    event.accept()
                    return
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan_active = True
            self._space_pan_moved = False
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            if event.key() in {Qt.Key.Key_Equal, Qt.Key.Key_Plus}:
                self._zoom_by(1.2)
                event.accept()
                return
            if event.key() in {Qt.Key.Key_Minus, Qt.Key.Key_Underscore}:
                self._zoom_by(1 / 1.2)
                event.accept()
                return
            if event.key() == Qt.Key.Key_0:
                self.reset_view()
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            should_fit = self._space_pan_active and not self._space_pan_moved
            self._space_pan_active = False
            self._panning = False
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self.viewport().unsetCursor()
            if should_fit:
                self.toggle_fit_selection_or_restore()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if self._space_pan_active and event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._last_pan_pos = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            current_pos = event.position().toPoint()
            delta = current_pos - self._last_pan_pos
            if abs(delta.x()) > 1 or abs(delta.y()) > 1:
                self._space_pan_moved = True
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._last_pan_pos = current_pos
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._panning and event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            if self._space_pan_active:
                self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def dragEnterEvent(self, event) -> None:
        if self._supports_external_import_drop(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if self._supports_external_import_drop(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        if not self._supports_external_import_drop(event.mimeData()):
            super().dropEvent(event)
            return
        payload = payload_from_mime_data(event.mimeData())
        local_paths = [
            str(path)
            for path in payload.get("local_paths", [])
            if str(path)
        ]
        if local_paths and not payload.get("urls") and not payload.get("html") and not payload.get("text") and payload.get("image") is None:
            self.filesDropped.emit(local_paths)
        else:
            self.dropPayloadDropped.emit(payload)
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()

    @staticmethod
    def _supports_external_import_drop(mime_data) -> bool:
        return payload_supports_mime_data(mime_data)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        painter.fillRect(rect, QColor("#202830"))
        minor_pen = QPen(QColor("#2b3440"))
        minor_pen.setWidth(0)
        major_pen = QPen(QColor("#364250"))
        major_pen.setWidth(0)

        left = int(rect.left()) - (int(rect.left()) % GRID_SIZE)
        top = int(rect.top()) - (int(rect.top()) % GRID_SIZE)
        right = int(rect.right())
        bottom = int(rect.bottom())

        x = left
        while x <= right:
            painter.setPen(major_pen if (x // GRID_SIZE) % 4 == 0 else minor_pen)
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
            x += GRID_SIZE

        y = top
        while y <= bottom:
            painter.setPen(major_pen if (y // GRID_SIZE) % 4 == 0 else minor_pen)
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)
            y += GRID_SIZE

    def mouseDoubleClickEvent(self, event) -> None:
        image_id = self._image_id_at(event.position().toPoint())
        if image_id is not None:
            self._select_image_id(image_id)
            self.toggle_fit_selection_or_restore()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:
        image_id = self._image_id_at(event.pos())
        if image_id is None:
            super().contextMenuEvent(event)
            return
        if image_id not in self.selected_image_ids():
            self._select_image_id(image_id)
        menu = QMenu(self)
        preview_action = menu.addAction("通过预览显示图片")
        menu.addSeparator()
        edit_local_action = menu.addAction("编辑局部展示")
        toggle_local_action = menu.addAction("切换局部/全图")
        selected_board_items = [item for item in self._selected_image_items() if isinstance(item, BoardImageItem)]
        edit_local_action.setEnabled(len(selected_board_items) == 1)
        toggle_local_action.setEnabled(any(item.has_local_display() for item in selected_board_items))
        menu.addSeparator()
        remove_action = menu.addAction("从当前项目移除")
        remove_action.setEnabled(bool(self.selected_image_ids()))
        chosen = menu.exec(event.globalPos())
        if chosen == preview_action:
            self.imagePreviewRequested.emit(image_id)
            event.accept()
            return
        if chosen == edit_local_action:
            self._edit_local_display_for_image(image_id)
            event.accept()
            return
        if chosen == toggle_local_action:
            self.toggle_selected_local_display()
            event.accept()
            return
        if chosen == remove_action:
            image_ids = self.selected_image_ids()
            if image_ids:
                self.removeImagesRequested.emit(image_ids)
            event.accept()
            return
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._view_mode in {"fit_all", "fit_selection"}:
            self._schedule_refit_current_view_mode()

    def selected_image_ids(self) -> list[int]:
        return list(self._selected_image_ids_cache)

    def _collect_selected_image_ids(self) -> list[int]:
        image_ids: list[int] = []
        seen: set[int] = set()
        for item in self._scene.selectedItems():
            image_id = self._image_id_for_item(item)
            if image_id is not None and image_id not in seen:
                image_ids.append(image_id)
                seen.add(image_id)
        return image_ids

    def select_image_id(self, image_id: int) -> None:
        self._select_image_id(image_id)

    def _zoom_by(self, factor: float) -> None:
        next_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self._zoom * factor))
        if abs(next_zoom - self._zoom) < 0.001:
            return
        scale_factor = next_zoom / max(0.001, self._zoom)
        self.scale(scale_factor, scale_factor)
        self._zoom = next_zoom

    def _fit_selected_items(self, items: list[QGraphicsItem]) -> None:
        if not items:
            return
        self._view_mode = "fit_selection"
        self._last_fit_selection_ids = tuple(sorted(self.selected_image_ids()))
        self._fit_items_to_view(items, margin=10.0)

    def _refit_current_view_mode(self) -> None:
        if self._view_mode == "fit_selection":
            selected_items = self._selected_image_items()
            if selected_items:
                self._fit_items_to_view(selected_items, margin=10.0)
                return
        if self._view_mode == "fit_all":
            self._fit_items_to_view(self._visible_image_items(), margin=80.0)

    def _schedule_refit_current_view_mode(self) -> None:
        if self._refit_timer_pending:
            return
        self._refit_timer_pending = True
        QTimer.singleShot(0, self._run_scheduled_refit)

    def _run_scheduled_refit(self) -> None:
        self._refit_timer_pending = False
        self._refit_current_view_mode()

    def _finish_layout_reset(self) -> None:
        self._suppress_layout_changed = False
        self._layout_change_pending = False
        self._layout_change_timer.stop()

    def _schedule_layout_changed(self) -> None:
        if self._suppress_layout_changed or not self._image_items:
            return
        self._layout_change_pending = True
        self._layout_change_timer.start()

    def flush_pending_layout_change(self) -> bool:
        if not self._layout_change_pending:
            return False
        self._layout_change_timer.stop()
        self._emit_pending_layout_changed()
        return True

    def _emit_pending_layout_changed(self) -> None:
        if not self._layout_change_pending:
            return
        self._layout_change_pending = False
        if self._suppress_layout_changed or not self._image_items:
            return
        self.layoutChanged.emit()

    def _add_image_item(
        self,
        image: ImageItem,
        pixmap: QPixmap,
        x: float,
        y: float,
        width: float,
        height: float,
        badges: list[str],
    ) -> None:
        if pixmap.isNull():
            frame = BoardPlaceholderItem(
                QRectF(0, 0, width, height),
                on_layout_changed=self._schedule_layout_changed,
            )
            frame.setBrush(QColor("#202830"))
            frame.setPen(QPen(QColor("#536071"), 1))
            item: QGraphicsItem = frame
            label = QGraphicsTextItem(image.file_name, frame)
            label.setDefaultTextColor(QColor("#dfe6ef"))
            label.setTextWidth(width - 18)
            label.setPos(9, height / 2 - 20)
        else:
            item = BoardImageItem(
                image_id=image.id,
                pixmap=pixmap,
                file_path=image.file_path,
                badges=badges,
                on_layout_changed=self._schedule_layout_changed,
            )
            item.set_display_size(width, height)
        item.setPos(x, y)
        if not isinstance(item, BoardImageItem):
            item.setFlags(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            )
        item.setData(0, int(image.id))
        item.setZValue(len(self._image_items) + 1)
        self._scene.addItem(item)
        self._image_items[image.id] = item

    def _apply_item_payload_state(self, image_id: int, item_payload: object) -> None:
        if not isinstance(item_payload, dict):
            return
        item = self._image_items.get(image_id)
        if item is None:
            return
        if "visible" in item_payload:
            item.setVisible(bool(item_payload.get("visible")))
        if isinstance(item, BoardImageItem):
            if "pinned" in item_payload:
                item.set_pinned(bool(item_payload.get("pinned")))
            if "flipped" in item_payload:
                item.set_flipped(bool(item_payload.get("flipped")))
            if "grayscale" in item_payload:
                item.set_grayscale(bool(item_payload.get("grayscale")))
            if "local_display" in item_payload:
                item.set_local_display(item_payload.get("local_display"), shrink_to_bounds=False)

    def _selected_image_items(self) -> list[QGraphicsItem]:
        items: list[QGraphicsItem] = []
        for image_id in self._selected_image_ids_cache:
            item = self._image_items.get(int(image_id))
            if item is not None and item.isSelected():
                items.append(item)
        return items

    def _select_image_id(self, image_id: int) -> None:
        previous_ids = list(self._selected_image_ids_cache)
        if previous_ids == [image_id]:
            return
        self._scene.blockSignals(True)
        try:
            for item in self._selected_image_items():
                item.setSelected(False)
            target = self._image_items.get(image_id)
            if target is not None:
                target.setSelected(True)
        finally:
            self._scene.blockSignals(False)
        selected_ids = [image_id] if image_id in self._image_items else []
        if selected_ids != previous_ids:
            self._selected_image_ids_cache = selected_ids
            self.selectionChanged.emit(selected_ids)

    def _edit_local_display_for_image(self, image_id: int) -> bool:
        item = self._image_items.get(image_id)
        if not isinstance(item, BoardImageItem):
            return False
        return self._edit_local_display_item(item)

    def _edit_local_display_item(self, item: BoardImageItem) -> bool:
        source_pixmap = item.source_pixmap()
        if source_pixmap.isNull():
            return False
        dialog = LocalDisplayEditorDialog(source_pixmap, item.local_display_payload(), self.window())
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        if dialog.clear_requested():
            return item.clear_local_display()
        payload = dialog.selected_payload()
        if payload is None:
            return False
        item.set_local_display(payload, shrink_to_bounds=True)
        return True

    def _emit_selection_changed(self) -> None:
        selected_ids = self._collect_selected_image_ids()
        if selected_ids == self._selected_image_ids_cache:
            return
        self._selected_image_ids_cache = selected_ids
        self.selectionChanged.emit(selected_ids)

    def _visible_image_items(self) -> list[QGraphicsItem]:
        return [item for item in self._image_items.values() if item.isVisible()]

    def _fit_items_to_view(self, items: list[QGraphicsItem], *, margin: float) -> None:
        if not items:
            return
        rect = self._items_rect(items)
        if not rect.isValid() or rect.width() <= 1 or rect.height() <= 1:
            return
        viewport_rect = self.viewport().rect()
        available_width = max(1.0, float(viewport_rect.width()) - margin * 2)
        available_height = max(1.0, float(viewport_rect.height()) - margin * 2)
        next_zoom = min(available_width / rect.width(), available_height / rect.height())
        next_zoom = max(MIN_ZOOM, min(MAX_ZOOM, next_zoom))
        self.resetTransform()
        self.scale(next_zoom, next_zoom)
        self._zoom = next_zoom
        self.centerOn(rect.center())

    def _image_id_at(self, viewport_pos) -> int | None:
        item = self.itemAt(viewport_pos)
        return self._image_id_for_item(item)

    def _image_id_for_item(self, item: QGraphicsItem | None) -> int | None:
        current: QGraphicsItem | None = item
        while current is not None:
            image_id = current.data(0)
            if image_id is not None:
                return int(image_id)
            current = current.parentItem()
        return None

    @staticmethod
    def _items_rect(items: list[QGraphicsItem]) -> QRectF:
        rect = QRectF()
        for item in items:
            item_rect = item.mapToScene(item.boundingRect()).boundingRect()
            rect = item_rect if rect.isNull() else rect.united(item_rect)
        return rect

    @staticmethod
    def _pixmap_for(
        image: ImageItem,
        *,
        prefer_source: bool = False,
        max_side: int = MAX_SOURCE_PIXMAP_SIDE,
    ) -> QPixmap:
        candidates = (
            [image.file_path, image.thumbnail_path]
            if prefer_source
            else [image.thumbnail_path, image.file_path]
        )
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if candidate == image.file_path:
                pixmap = _cached_pixmap(
                    path,
                    max_side=max_side,
                    modified_time_ns=image.modified_time_ns,
                    file_size=image.file_size,
                )
            else:
                pixmap = _cached_pixmap(
                    path,
                    max_side=max_side,
                    modified_time_ns=image.modified_time_ns,
                    file_size=image.file_size,
                )
            if not pixmap.isNull():
                return pixmap
        return QPixmap()

    @staticmethod
    def _default_item_size(pixmap: QPixmap) -> tuple[float, float]:
        target_height = 160.0
        if pixmap.isNull():
            return 220.0, target_height
        ratio = pixmap.width() / max(1, pixmap.height())
        width = max(120.0, min(360.0, target_height * ratio))
        return width, target_height


def _safe_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
