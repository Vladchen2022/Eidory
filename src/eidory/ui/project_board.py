from __future__ import annotations

import math

from eidory.ui.theme_colors import theme_color

import json
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, QSizeF, Qt, QTimer, Signal, QSignalBlocker
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
    QInputDialog,
    QLabel,
    QMenu,
    QApplication,
    QPushButton,
    QStyle,
    QStyleOptionGraphicsItem,
    QVBoxLayout,
    QWidget,
)

from eidory.models import ImageItem
from eidory.ui.drop_import_box import payload_from_mime_data, payload_supports_mime_data
from eidory.ui.board_image_loader import BoardImageLoader


BOARD_WIDTH = 5200
BOARD_HEIGHT = 3600
MIN_ZOOM = 0.15
MAX_ZOOM = 16.0
GRID_SIZE = 36
PIXMAP_CACHE_LIMIT = 512
MAX_SOURCE_PIXMAP_SIDE = 2200
MAX_BOARD_THUMBNAIL_SIDE = 384
_BOARD_IMAGE_SIZES = OrderedDict()
MAX_BOARD_FOCUS_PIXMAP_SIDE = 4096
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
    _store_cached_pixmap(key, pixmap)
    return pixmap


def _store_cached_pixmap(key, pixmap):
    global _BOARD_PIXMAP_CACHE_TOTAL_BYTES
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


def trim_pixmap_cache(budget):
    global _BOARD_PIXMAP_CACHE_TOTAL_BYTES, _BOARD_PIXMAP_CACHE_BYTE_LIMIT
    _BOARD_PIXMAP_CACHE_BYTE_LIMIT = max(0, budget)
    while _BOARD_PIXMAP_CACHE and _BOARD_PIXMAP_CACHE_TOTAL_BYTES > budget:
        key, _ = _BOARD_PIXMAP_CACHE.popitem(last=False)
        _BOARD_PIXMAP_CACHE_TOTAL_BYTES -= _BOARD_PIXMAP_CACHE_BYTES.pop(key, 0)


def _estimated_pixmap_bytes(pixmap: QPixmap) -> int:
    if pixmap.isNull():
        return 0
    return max(0, pixmap.width()) * max(0, pixmap.height()) * 4


def _image_aspect_ratio(image: ImageItem, pixmap: QPixmap) -> float:
    decoded = pixmap.width() / max(1, pixmap.height())
    if not image.width or not image.height:
        return decoded
    metadata = image.width / image.height
    # The scanner stores raw dimensions; EXIF-oriented thumbnails may have
    # their axes swapped. Keep exact metadata proportions despite pixel rounding.
    tolerance = 2 / max(1, min(pixmap.width(), pixmap.height()))
    if abs(decoded / metadata - 1) <= tolerance:
        return metadata
    if abs(decoded * metadata - 1) <= tolerance:
        return 1 / metadata
    return decoded


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
    SELECTION_OUTLINE_COLOR = QColor("#2f80ff")
    SELECTION_OUTLINE_WIDTH = 3
    PRIMARY_OUTLINE_COLOR = QColor('#d64b40')

    def __init__(
        self,
        *,
        image_id: int,
        pixmap: QPixmap,
        file_path: str,
        badges: list[str] | None = None,
        on_layout_changed: Callable[[], None] | None = None,
        source_aspect_ratio: float | None = None,
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
        self._source_aspect_ratio = source_aspect_ratio or pixmap.width() / max(1, pixmap.height())
        self._aspect_ratio = self._source_aspect_ratio
        self._pinned = False
        self._flipped = False
        self._grayscale = False
        self._layout_interaction_active = False
        self._layout_interaction_dirty = False
        self._local_display: dict[str, Any] | None = None
        self._display_variant_cache: OrderedDict[tuple[object, ...], QPixmap] = OrderedDict()
        self.setToolTip(file_path)
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        # Cosmetic outlines and badges must stay readable at the current zoom.
        # ItemCoordinateCache rasterizes these at item scale, shrinking them
        # together with the image on a zoomed-out board.
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)
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
        from eidory.core.project_state import reference_badge_text
        return reference_badge_text(self._badges)

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
        # Rounded thumbnail dimensions must not change the logical canvas
        # geometry when higher-resolution pixels arrive.
        pixel_ratio = pixmap.width() / max(1, pixmap.height())
        self.setTransform(QTransform().scale(1, pixel_ratio / self._aspect_ratio))
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

    def can_start_drag_at_scene_pos(self, scene_pos: QPointF) -> bool:
        if self._pinned:
            return False
        local_pos = self.mapFromScene(scene_pos)
        return self._resize_zone_at(local_pos) not in {"corner", "right", "bottom"}

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

    def replace_source_pixmap(self, pixmap: QPixmap, *, allow_smaller: bool = False) -> bool:
        if pixmap.isNull():
            return False
        if not allow_smaller and (
            pixmap.width() * pixmap.height()
            < self._source_pixmap.width() * self._source_pixmap.height()
        ):
            return False
        self._source_pixmap = QPixmap(pixmap)
        self._display_variant_cache.clear()
        self._refresh_display_pixmap()
        return True

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
        self._aspect_ratio = self._source_aspect_ratio
        if self.is_local_display_enabled():
            bounds = self._local_display['bounds']
            self._aspect_ratio *= bounds['w'] / max(0.001, bounds['h'])
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

    def _begin_layout_interaction(self) -> None:
        if self._layout_interaction_active:
            return
        self._layout_interaction_active = True
        self._layout_interaction_dirty = False
        self.setTransformationMode(Qt.TransformationMode.FastTransformation)

    def _finish_layout_interaction(self) -> None:
        if not self._layout_interaction_active:
            return
        should_notify = self._layout_interaction_dirty
        self._layout_interaction_active = False
        self._layout_interaction_dirty = False
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        if should_notify:
            self._notify_layout_changed()

    def _notify_layout_changed(self) -> None:
        if self._layout_interaction_active:
            self._layout_interaction_dirty = True
            return
        if self._on_layout_changed is not None:
            self._on_layout_changed()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        clean_option = QStyleOptionGraphicsItem(option)
        if selected:
            clean_option.state &= ~QStyle.StateFlag.State_Selected
        super().paint(painter, clean_option, widget)
        primary = '主参考' in self._badges
        if primary:
            painter.save()
            pen = QPen(self.PRIMARY_OUTLINE_COLOR, 4)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.boundingRect().adjusted(0.5, 0.5, -0.5, -0.5))
            painter.restore()
        if selected:
            painter.save()
            pen = QPen(self.SELECTION_OUTLINE_COLOR, self.SELECTION_OUTLINE_WIDTH)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            inset = 5.0 / max(abs(painter.worldTransform().m11()), 0.01) if primary else 0.5
            painter.drawRect(self.boundingRect().adjusted(inset, inset, -inset, -inset))
            painter.restore()
        text = ('★ ' if primary else '') + self.badge_text
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
        painter.setBrush(self.PRIMARY_OUTLINE_COLOR if primary else QColor(20, 24, 30, 220))
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
                self._begin_layout_interaction()
                event.accept()
                return
            self._begin_layout_interaction()
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
            self._finish_layout_interaction()
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._finish_layout_interaction()

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


class BoardTextItem(QGraphicsTextItem):
    def __init__(self, text, changed, *, heading=False):
        super().__init__(text)
        self.heading = bool(heading)
        self.changed = changed
        self.setTextWidth(360)
        self.setDefaultTextColor(theme_color("warning" if heading else "text"))
        font = self.font()
        font.setPointSize(22 if heading else 15)
        font.setBold(heading)
        self.setFont(font)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setToolTip("拖动调整位置，双击编辑文字，选中后按 Delete 移除")
        self.setZValue(5)

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.changed()
        return result

    def mouseDoubleClickEvent(self, event):
        text, ok = QInputDialog.getMultiLineText(None, "编辑看板文字", "内容", self.toPlainText())
        if ok:
            self.setPlainText(text)
            self.changed()
        event.accept()


class BoardSectionItem(QGraphicsRectItem):
    """A persisted section; moving its frame translates its member images once."""
    def __init__(self, view, payload):
        super().__init__(0, 0, payload.get("width", 1120), payload.get("height", 420))
        self.view = view
        self.label = str(payload.get("label", "参考图"))
        self.image_ids = [int(i) for i in payload.get("image_ids", []) if int(i) in view._image_items]
        self.locked = bool(payload.get("locked", False))
        self._ready = False
        self._position = QPointF(payload.get("x", 80), payload.get("y", 110))
        self.setPos(self._position)
        self.setZValue(-1)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, not self.locked)
        self.heading = QGraphicsTextItem(self)
        self.heading.setPos(20, 12)
        self.heading.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        font = self.heading.font(); font.setPointSize(22); font.setBold(True); self.heading.setFont(font)
        self.refresh_theme()
        self._ready = True

    def refresh_theme(self):
        self.setPen(QPen(theme_color("border"), 2))
        color = theme_color("raised"); color.setAlpha(100); self.setBrush(color)
        self.heading.setDefaultTextColor(theme_color("text"))
        self.heading.setPlainText(self.label + (" · 已锁定" if self.locked else ""))
        self.setToolTip("双击标题锁定／解锁分区布局；锁定后自动整理会保留本区位置与大小")

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self._ready:
            delta = self.pos() - self._position
            self._position = QPointF(self.pos())
            for image_id in self.image_ids:
                item = self.view._image_items.get(image_id)
                if item:
                    item.setPos(item.pos() + delta)
            self.view._schedule_layout_changed()
        return result

    def mousePressEvent(self, event):
        # Images are scene siblings, so Qt must not move selected members again.
        for item in self.view.scene().selectedItems():
            if item is not self:
                item.setSelected(False)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.locked = not self.locked
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, not self.locked)
        self.refresh_theme()
        self.view._schedule_layout_changed()
        event.accept()

    def payload(self):
        return dict(label=self.label, image_ids=self.image_ids, locked=self.locked,
                    x=self.x(), y=self.y(), width=self.rect().width(), height=self.rect().height())


class ProjectBoardView(QGraphicsView):
    imageLoaded = Signal(int)
    imageDoubleClicked = Signal(int)
    imagePreviewRequested = Signal(int)
    selectionChanged = Signal(list)
    layoutChanged = Signal()
    sectionsChanged = Signal(bool)
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
        self.setBackgroundBrush(theme_color("canvas"))
        self._image_items: dict[int, QGraphicsItem] = {}
        self._note_items: list[BoardTextItem] = []
        self._section_items: list[BoardSectionItem] = []
        self._undo_layouts = []
        self._redo_layouts = []
        self._last_layout = None
        self._board_title = ""
        self._board_badges = {}
        self._images_by_id: dict[int, ImageItem] = {}
        self._focus_quality_image_ids: set[int] = set()
        self._image_generation = 0
        self._pending_image_loads: set[tuple[int, int, int]] = set()
        self._thumbnail_pixmaps: dict[int, QPixmap] = {}
        self._loaded_image_sides: dict[int, int] = {}
        self._failed_image_ids: set[int] = set()
        self._pending_local_edit = None
        self._image_loader = BoardImageLoader(self)
        self._image_loader.loaded.connect(self._image_loaded)
        self._quality_timer = QTimer(self)
        self._quality_timer.setSingleShot(True)
        self._quality_timer.setInterval(120)
        self._quality_timer.timeout.connect(self._refresh_visible_quality)
        self.horizontalScrollBar().valueChanged.connect(self._schedule_visible_quality)
        self.verticalScrollBar().valueChanged.connect(self._schedule_visible_quality)
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
        self._item_drag_layout_batch_active = False
        self._item_drag_layout_pending = False
        self._manual_item_drag_pending = False
        self._manual_item_drag_active = False
        self._manual_item_drag_start_viewport = QPoint()
        self._manual_item_drag_start_scene = QPointF()
        self._manual_item_drag_delta = QPointF()
        self._manual_item_drag_anchor_item: BoardImageItem | None = None
        self._manual_item_drag_items: list[tuple[BoardImageItem, QPointF, QRectF, float]] = []
        self._suppress_layout_changed = False
        self._layout_change_pending = False
        self._layout_change_timer = QTimer(self)
        self._layout_change_timer.setSingleShot(True)
        self._layout_change_timer.setInterval(LAYOUT_CHANGED_IDLE_MS)
        self._layout_change_timer.timeout.connect(self._emit_pending_layout_changed)
        self._scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self._scene.selectionChanged.connect(self._emit_selection_changed)

    def refresh_theme(self) -> None:
        self.setBackgroundBrush(theme_color("canvas"))
        for item in self._scene.items():
            if isinstance(item, BoardSectionItem):
                item.refresh_theme()
            elif isinstance(item, QGraphicsTextItem):
                heading = isinstance(item, BoardTextItem) and item.heading
                item.setDefaultTextColor(theme_color("warning" if heading else "text"))
            elif isinstance(item, BoardPlaceholderItem):
                item.setBrush(theme_color("canvas"))
                item.setPen(QPen(theme_color("border"), 1))
        self.resetCachedContent()
        self.viewport().update()

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
        self._board_title = title
        self._board_badges = normalized_badges
        self._image_generation += 1
        self._image_loader.reset()
        self._pending_local_edit = None
        self._pending_image_loads.clear()
        self._thumbnail_pixmaps.clear()
        self._loaded_image_sides.clear()
        self._failed_image_ids.clear()
        self.setUpdatesEnabled(False)
        previous_scene_signal_state = self._scene.blockSignals(True)
        previous_selected_ids = list(self._selected_image_ids_cache)
        self._suppress_layout_changed = True
        try:
            self._scene.clear()
            self._image_items.clear()
            self._note_items.clear()
            self._section_items.clear()
            self._images_by_id = {int(image.id): image for image in images}
            self._focus_quality_image_ids.clear()
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
                self._title_item.setDefaultTextColor(theme_color("text"))
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
                pixmap = self._cached_thumbnail(image)
                loaded = not pixmap.isNull()
                if not loaded:
                    # Metadata gives the final geometry immediately; decoding
                    # must never block set_images or alter saved positions.
                    size = QSize(*_BOARD_IMAGE_SIZES.get((image.file_path, image.modified_time_ns, image.file_size),
                                 (max(1, image.width or 220), max(1, image.height or 160))))
                    size.scale(128, 128, Qt.AspectRatioMode.KeepAspectRatio)
                    pixmap = QPixmap(size)
                    pixmap.fill(theme_color('canvas'))
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
                if loaded:
                    self._thumbnail_pixmaps[image.id] = pixmap
                    self._loaded_image_sides[image.id] = MAX_BOARD_THUMBNAIL_SIDE
                else:
                    self._request_image(image.id, MAX_BOARD_THUMBNAIL_SIDE)
                placed_positions.add((int(round(x)), int(round(y))))
                row_height = max(row_height, item_height)
                default_x += item_width + DEFAULT_LAYOUT_GAP_X

            if isinstance(layout_payload, dict):
                for raw_note in layout_payload.get("notes", []):
                    if isinstance(raw_note, dict):
                        self._add_text(str(raw_note.get("text", "")), _safe_float(raw_note.get("x"), 80), _safe_float(raw_note.get("y"), 80), heading=raw_note.get("heading", False))
            if isinstance(layout_payload, dict):
                for section in layout_payload.get("sections", []):
                    if isinstance(section, dict):
                        self._add_section(section)
                # Existing groups never jump when new references are collected.
                if self._section_items:
                    assigned = {i for section in self._section_items for i in section.image_ids}
                    fresh = [i for i in self._image_items if i not in assigned]
                    if fresh:
                        bottom = max(s.sceneBoundingRect().bottom() for s in self._section_items) + 60
                        self._layout_sections(fresh, bottom)
            if not images and not self._note_items:
                text = self._scene.addText("当前节点没有保存图片")
                text.setDefaultTextColor(theme_color("text"))
                text.setPos(80, 110)
        finally:
            self.setUpdatesEnabled(True)
            self._scene.blockSignals(previous_scene_signal_state)
            self._finish_layout_reset()

        self._undo_layouts.clear()
        self._redo_layouts.clear()
        self._last_layout = self._capture_layout_payload()
        self.sectionsChanged.emit(bool(self._section_items))

        if previous_selected_ids:
            self.selectionChanged.emit([])
        if images:
            self._schedule_refit_current_view_mode()
            self._schedule_visible_quality()

    @staticmethod
    def _cached_thumbnail(image):
        for path in (image.thumbnail_path, image.file_path):
            key = (path, image.modified_time_ns, image.file_size, MAX_BOARD_THUMBNAIL_SIDE)
            cached = _BOARD_PIXMAP_CACHE.get(key)
            if cached is not None:
                _BOARD_PIXMAP_CACHE.move_to_end(key)
                return QPixmap(cached)
        return QPixmap()

    def _request_image(self, image_id, side):
        image = self._images_by_id.get(image_id)
        if image is None:
            return
        token = (self._image_generation, image_id, side)
        if token in self._pending_image_loads:
            return
        paths = (image.file_path, image.thumbnail_path) if side > MAX_BOARD_THUMBNAIL_SIDE else (image.thumbnail_path, image.file_path)
        self._pending_image_loads.add(token)
        self._image_loader.request(token, [str(p) for p in paths if p], side, urgent=side > MAX_BOARD_THUMBNAIL_SIDE)

    def _image_loaded(self, token, image, path):
        self._pending_image_loads.discard(token)
        generation, iid, side = token
        if generation != self._image_generation:
            return
        item = self._image_items.get(iid)
        if not isinstance(item, BoardImageItem):
            return
        if image.isNull():
            self._focus_quality_image_ids.discard(iid)
            item.setToolTip(self._images_by_id[iid].file_path + '\n无法读取图片')
            if iid not in self._failed_image_ids and iid not in self._thumbnail_pixmaps:
                label = QGraphicsTextItem(self._images_by_id[iid].file_name, item)
                item._unreadable_label = label
                label.setDefaultTextColor(theme_color('text'))
                label.setTextWidth(max(40, item.display_width - 18))
                label.setScale(1 / max(.001, item.scale()))
                label.setPos(9 / item.scale(), max(0, item.display_height / 2 - 20) / item.scale())
            self._failed_image_ids.add(iid)
            if self._pending_local_edit == (generation, iid):
                self._pending_local_edit = None
            return
        metadata = self._images_by_id[iid]
        size_key = (metadata.file_path, metadata.modified_time_ns, metadata.file_size)
        _BOARD_IMAGE_SIZES[size_key] = (image.width(), image.height())
        _BOARD_IMAGE_SIZES.move_to_end(size_key)
        while len(_BOARD_IMAGE_SIZES) > 4096:
            _BOARD_IMAGE_SIZES.popitem(last=False)
        pixmap = QPixmap.fromImage(image)
        if side == MAX_BOARD_THUMBNAIL_SIDE:
            self._thumbnail_pixmaps[iid] = pixmap
            metadata = self._images_by_id[iid]
            _store_cached_pixmap((path, metadata.modified_time_ns, metadata.file_size, side), pixmap)
        elif self._pending_local_edit != (generation, iid) and (not self.isVisible() or not self._needs_focus_quality_upgrade_from_thumbnail(item)):
            self._focus_quality_image_ids.discard(iid)
            return
        self._failed_image_ids.discard(iid)
        label = getattr(item, '_unreadable_label', None)
        if label is not None:
            self._scene.removeItem(label)
            item._unreadable_label = None
        item.setToolTip(self._images_by_id[iid].file_path)
        # Loading pixels is not a layout edit (nor an undo step).
        previous = self._suppress_layout_changed
        self._suppress_layout_changed = True
        aspect_changed = False
        try:
            if side >= self._loaded_image_sides.get(iid, 0):
                ratio = _image_aspect_ratio(self._images_by_id[iid], pixmap)
                aspect_changed = abs(item._source_aspect_ratio - ratio) > 1e-9
                item._source_aspect_ratio = ratio
                item.replace_source_pixmap(pixmap, allow_smaller=True)
                self._loaded_image_sides[iid] = side
        finally:
            self._suppress_layout_changed = previous
        self._schedule_visible_quality()
        if aspect_changed:
            self._schedule_refit_current_view_mode()
        self.imageLoaded.emit(iid)
        if side > MAX_BOARD_THUMBNAIL_SIDE and self._pending_local_edit == (generation, iid):
            self._pending_local_edit = None
            if iid in self.selected_image_ids():
                self._edit_local_display_item(item)

    def _schedule_visible_quality(self, *_):
        if hasattr(self, '_quality_timer'):
            self._quality_timer.start()

    def _needs_focus_quality_upgrade_from_thumbnail(self, item):
        thumbnail = self._thumbnail_pixmaps.get(item.image_id)
        if thumbnail is None:
            return True
        rendered = max(item.sceneBoundingRect().width(), item.sceneBoundingRect().height()) * self._zoom * self.devicePixelRatioF()
        return rendered > max(thumbnail.width(), thumbnail.height()) * .92

    def trim_image_memory(self, budget):
        # Downgrade offscreen pixel buffers, retaining positions, edits and a
        # small preview. Visible images regain resolution through the loader.
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        def cost(item):
            pix = item.source_pixmap()
            return pix.width() * pix.height() * 4
        items = [item for item in self._image_items.values() if isinstance(item, BoardImageItem)]
        total = sum(cost(item) for item in items)
        candidates = sorted(items, key=lambda item: (self.isVisible() and visible.intersects(item.sceneBoundingRect()), -cost(item)))
        previous = self._suppress_layout_changed
        self._suppress_layout_changed = True
        try:
            for item in candidates:
                if total <= budget:
                    break
                if self.isVisible() and visible.intersects(item.sceneBoundingRect()):
                    continue
                old = cost(item)
                pix = item.source_pixmap().scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                item.replace_source_pixmap(pix, allow_smaller=True)
                self._thumbnail_pixmaps[item.image_id] = pix
                self._loaded_image_sides[item.image_id] = 128
                self._focus_quality_image_ids.discard(item.image_id)
                total -= old - cost(item)
        finally:
            self._suppress_layout_changed = previous
        return total

    def _refresh_visible_quality(self):
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        for iid, item in self._image_items.items():
            if not isinstance(item, BoardImageItem):
                continue
            needs_high = iid not in self._failed_image_ids and self.isVisible() and item.isVisible() and visible.intersects(item.sceneBoundingRect()) and self._needs_focus_quality_upgrade_from_thumbnail(item)
            if needs_high:
                self._ensure_focus_quality_for_items([item], force=True)
            elif iid in self._focus_quality_image_ids and iid in self._thumbnail_pixmaps:
                self._focus_quality_image_ids.discard(iid)
                previous = self._suppress_layout_changed
                self._suppress_layout_changed = True
                try:
                    item.replace_source_pixmap(self._thumbnail_pixmaps[iid], allow_smaller=True)
                    self._loaded_image_sides[iid] = MAX_BOARD_THUMBNAIL_SIDE
                finally:
                    self._suppress_layout_changed = previous

    def hideEvent(self, event):
        super().hideEvent(event)
        self._refresh_visible_quality()

    def showEvent(self, event):
        super().showEvent(event)
        self._schedule_visible_quality()

    def fit_all_images(self) -> None:
        self._view_mode = "fit_all"
        self._last_fit_selection_ids = ()
        self._fit_items_to_view(self._visible_image_items() + list(self._note_items) + list(self._section_items), margin=80.0)

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
        items = [item for item in (self._selected_image_items() or list(self._image_items.values()))
                 if isinstance(item, BoardImageItem)]
        if not items:
            return 0
        # A mixed selection becomes uniformly monochrome, then uniformly color.
        grayscale = not all(item.is_grayscale() for item in items)
        changed = [item for item in items if item.is_grayscale() != grayscale]
        self.flush_pending_layout_change()
        previous = self._suppress_layout_changed
        self._suppress_layout_changed = True
        try:
            for item in changed:
                item.set_grayscale(grayscale)
        finally:
            self._suppress_layout_changed = previous
        if changed:
            self._schedule_layout_changed()
            self.flush_pending_layout_change()
        return len(changed)

    def edit_selected_local_display(self) -> bool:
        selected = [item for item in self._selected_image_items() if isinstance(item, BoardImageItem)]
        if not selected:
            return False
        return self._edit_local_display_for_image(selected[0].image_id)

    def toggle_selected_local_display(self) -> int:
        changed = 0
        for item in self._selected_image_items():
            if isinstance(item, BoardImageItem) and item.toggle_local_display_enabled():
                changed += 1
        return changed

    def selected_has_local_display(self) -> bool:
        return any(
            isinstance(item, BoardImageItem) and item.has_local_display()
            for item in self._selected_image_items()
        )

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
        return self._capture_layout_payload()

    def _capture_layout_payload(self) -> dict[str, Any]:
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
        notes = [{"text": item.toPlainText(), "x": item.x(), "y": item.y(), "heading": item.heading} for item in self._note_items]
        return {"version": 1, "items": items, "notes": notes, "sections": [s.payload() for s in self._section_items]}

    def _add_text(self, text, x, y, *, heading=False):
        item = BoardTextItem(text, self._schedule_layout_changed, heading=heading)
        self._scene.addItem(item)
        self._note_items.append(item)
        item.setPos(x, y)
        return item

    def add_note(self, *, heading=False):
        text, ok = QInputDialog.getMultiLineText(self, "添加区域标题" if heading else "添加看板便签", "内容")
        if not ok or not text.strip():
            return
        existing = self._visible_image_items() + list(self._note_items)
        bounds = self._items_rect(existing) if existing else QRectF(DEFAULT_LAYOUT_LEFT, DEFAULT_LAYOUT_TOP, 0, 0)
        self._add_text(text.strip(), bounds.left(), bounds.bottom() + 40, heading=heading)
        self._schedule_layout_changed()
        self.flush_pending_layout_change()
        self.fit_all_images()

    def _add_section(self, payload):
        item = BoardSectionItem(self, payload)
        self._scene.addItem(item)
        self._section_items.append(item)
        return item

    def set_reference_badges(self, badges):
        """Refresh reference decisions without rebuilding items or resetting the view."""
        self._board_badges = {int(k): list(v) for k, v in badges.items()}
        for image_id, item in self._image_items.items():
            if isinstance(item, BoardImageItem):
                item._badges = item._normalize_badges(self._board_badges.get(image_id, []))
                item.update()
        self.viewport().update()

    def toggle_node_sections(self):
        if not self._section_items:
            self.arrange_by_labels()
            return
        self.flush_pending_layout_change()
        # Remove only generated frames. Images, custom notes, local display and
        # their geometry remain untouched; undo can restore even locked frames.
        for section in self._section_items:
            self._scene.removeItem(section)
        self._section_items.clear()
        self._schedule_layout_changed()
        self.flush_pending_layout_change()
        self.sectionsChanged.emit(False)

    def toggle_section_lock(self):
        selected = [s for s in self._section_items if s.isSelected()]
        if not selected:
            ids = set(self.selected_image_ids())
            selected = [s for s in self._section_items if ids.intersection(s.image_ids)]
        for section in selected:
            section.locked = not section.locked
            section.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, not section.locked)
            section.refresh_theme()
        self._schedule_layout_changed()

    def _layout_sections(self, image_ids, top):
        groups = {}
        for image_id in image_ids:
            labels = self._board_badges.get(image_id, [])
            groups.setdefault(labels[0] if labels else "参考图", []).append(image_id)
        # Two columns for ordinary windows; three for wide canvases. Each column
        # fills independently so uneven groups do not leave a whole empty row.
        columns = min(len(groups), 3 if self.viewport().width() > 1700 else 2)
        if not columns:
            return
        bottoms = [top] * columns
        width, gap = 1120, 60
        for label, ids in groups.items():
            col = min(range(columns), key=lambda c: bottoms[c])
            left, y = DEFAULT_LAYOUT_LEFT + col * (width + gap), bottoms[col]
            x, row_y, row_height = left + 24, y + 72, 0
            ids.sort(key=lambda i: 0 if "主参考" in self._board_badges.get(i, []) else 1)
            for image_id in ids:
                item = self._image_items[image_id]
                ratio = max(.35, min(3, item.display_width / max(item.display_height, 1)))
                height = 320 if "主参考" in self._board_badges.get(image_id, []) else 210
                item_width = min(width - 48, height * ratio)
                if x > left + 24 and x + item_width > left + width - 24:
                    x = left + 24; row_y += row_height + 35; row_height = 0
                if isinstance(item, BoardImageItem):
                    item.set_display_size(item_width, height)
                item.setPos(x, row_y)
                x += item_width + 24; row_height = max(row_height, height)
            section_height = row_y + row_height + 24 - y
            self._add_section(dict(label=label, image_ids=ids, x=left, y=y, width=width, height=section_height))
            bottoms[col] = y + section_height + gap

    def arrange_by_labels(self):
        self.flush_pending_layout_change()
        self._suppress_layout_changed = True
        try:
            labels = {badges[0] for badges in self._board_badges.values() if badges}
            for note in list(self._note_items):
                if note.heading and note.toPlainText() in labels:
                    self._scene.removeItem(note)
                    self._note_items.remove(note)
            locked = [s for s in self._section_items if s.locked]
            retained = {i for s in locked for i in s.image_ids}
            for section in list(self._section_items):
                if not section.locked:
                    self._scene.removeItem(section)
                    self._section_items.remove(section)
            top = max((s.sceneBoundingRect().bottom() + 60 for s in locked), default=DEFAULT_LAYOUT_TOP)
            self._layout_sections([i for i in self._image_items if i not in retained], top)
        finally:
            self._suppress_layout_changed = False
        self._schedule_layout_changed()
        self.flush_pending_layout_change()
        self.fit_all_images()
        self.sectionsChanged.emit(bool(self._section_items))

    def undo_layout(self):
        self.flush_pending_layout_change()
        if not self._undo_layouts:
            return False
        previous = self._undo_layouts.pop()
        self._redo_layouts.append(self._capture_layout_payload())
        self._restore_layout(previous)
        return True

    def redo_layout(self):
        self.flush_pending_layout_change()
        if not self._redo_layouts:
            return False
        following = self._redo_layouts.pop()
        self._undo_layouts.append(self._capture_layout_payload())
        self._restore_layout(following)
        return True

    def _restore_layout(self, payload):
        undo, redo = list(self._undo_layouts), list(self._redo_layouts)
        selected_ids = self.selected_image_ids()
        transform = self.transform()
        center = self.mapToScene(self.viewport().rect().center())
        self.set_images(list(self._images_by_id.values()), title=self._board_title, layout_payload=payload, badges_by_image_id=self._board_badges)
        with QSignalBlocker(self._scene):
            for image_id in selected_ids:
                item = self._image_items.get(image_id)
                if item is not None and item.isVisible():
                    item.setSelected(True)
        self._emit_selection_changed()
        self._undo_layouts, self._redo_layouts = undo, redo
        self.setTransform(transform)
        self.centerOn(center)
        self._view_mode = "manual"
        self.layoutChanged.emit()

    def render_to_png(self, path):
        bounds = self._scene.itemsBoundingRect().adjusted(-30, -30, 30, 30)
        scale = min(1.0, 2400 / max(bounds.width(), bounds.height(), 1))
        image = QImage(max(1, int(bounds.width()*scale)), max(1, int(bounds.height()*scale)), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(theme_color("canvas"))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self._scene.render(painter, QRectF(0, 0, image.width(), image.height()), bounds)
        painter.end()
        if not image.save(str(path), "PNG"):
            raise OSError("无法保存看板图片")

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
        self._schedule_visible_quality()

    def wheelEvent(self, event) -> None:
        # Plain wheel zooms the canvas and its images together. Small trackpad
        # deltas stay proportional instead of jumping a full step per event.
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta:
            factor = 1.12 ** (max(-240, min(240, delta)) / 120)
            self._last_fit_selection_ids = ()
            self._zoom_by(factor, anchor_pos=event.position().toPoint(), defer_quality=True)
        event.accept()

    def keyPressEvent(self, event) -> None:
        undo_modifier = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        if event.matches(QKeySequence.StandardKey.Redo):
            history = getattr(self.window(), "edit_history", None)
            if history is not None and history.future:
                history.redo()
            else:
                self.redo_layout()
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Undo) or (
            event.key() == Qt.Key.Key_Z and bool(event.modifiers() & undo_modifier)
        ):
            history = getattr(self.window(), "edit_history", None)
            if history is not None and history.past:
                history.undo()
            elif not self.undo_layout():
                self.undoRemovalRequested.emit()
            event.accept()
            return
        if event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}:
            selected_notes = [item for item in self._note_items if item.isSelected()]
            if selected_notes:
                for item in selected_notes:
                    self._scene.removeItem(item)
                    self._note_items.remove(item)
                self._schedule_layout_changed()
                self.flush_pending_layout_change()
                event.accept()
                return
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
        scene_press_pos = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton:
            image_id = self._image_id_at(event.position().toPoint())
            item = self._image_items.get(image_id) if image_id is not None else None
            if isinstance(item, BoardImageItem) and item.can_start_drag_at_scene_pos(scene_press_pos):
                self._manual_item_drag_pending = True
                self._manual_item_drag_active = False
                self._manual_item_drag_anchor_item = item
                self._manual_item_drag_start_viewport = event.position().toPoint()
                self._manual_item_drag_start_scene = scene_press_pos
                self._manual_item_drag_delta = QPointF()
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
        if self._manual_item_drag_pending or self._manual_item_drag_active:
            current_viewport_pos = event.position().toPoint()
            if not self._manual_item_drag_active:
                drag_delta = current_viewport_pos - self._manual_item_drag_start_viewport
                if drag_delta.manhattanLength() < QApplication.startDragDistance():
                    event.accept()
                    return
                self._begin_manual_item_drag()
            if self._manual_item_drag_active:
                scene_pos = self.mapToScene(current_viewport_pos)
                self._update_manual_item_drag(scene_pos - self._manual_item_drag_start_scene)
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
        if event.button() == Qt.MouseButton.LeftButton and self._manual_item_drag_active:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._update_manual_item_drag(scene_pos - self._manual_item_drag_start_scene)
            self._finish_manual_item_drag(commit=True)
            super().mouseReleaseEvent(event)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._manual_item_drag_pending:
            self._reset_manual_item_drag()
            super().mouseReleaseEvent(event)
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

    def _begin_manual_item_drag(self) -> None:
        items = [
            item
            for item in self._selected_image_items()
            if isinstance(item, BoardImageItem) and not item.is_pinned()
        ]
        anchor = self._manual_item_drag_anchor_item
        if anchor is not None and not anchor.is_pinned() and anchor not in items:
            items.insert(0, anchor)
        if not items:
            self._reset_manual_item_drag()
            return
        self._manual_item_drag_items = [
            (item, QPointF(item.pos()), QRectF(item.sceneBoundingRect()), float(item.opacity()))
            for item in items
        ]
        self._manual_item_drag_active = True
        self._manual_item_drag_pending = False
        self._manual_item_drag_delta = QPointF()
        self._begin_item_drag_layout_batch()
        for item, _pos, rect, _opacity in self._manual_item_drag_items:
            item._begin_layout_interaction()
            item.setOpacity(0.0)
            self._update_scene_rect(rect.adjusted(-8, -8, 8, 8))

    def _update_manual_item_drag(self, delta: QPointF) -> None:
        if not self._manual_item_drag_active:
            return
        previous_delta = QPointF(self._manual_item_drag_delta)
        self._manual_item_drag_delta = QPointF(delta)
        dirty_rects: list[QRectF] = []
        for _item, _start_pos, rect, _opacity in self._manual_item_drag_items:
            dirty_rects.append(rect.translated(previous_delta).adjusted(-8, -8, 8, 8))
            dirty_rects.append(rect.translated(self._manual_item_drag_delta).adjusted(-8, -8, 8, 8))
        self._update_scene_rects(dirty_rects)

    def _finish_manual_item_drag(self, *, commit: bool) -> None:
        if not self._manual_item_drag_active and not self._manual_item_drag_pending:
            return
        final_delta = QPointF(self._manual_item_drag_delta)
        items = list(self._manual_item_drag_items)
        dirty_rects: list[QRectF] = []
        for item, start_pos, rect, opacity in items:
            if commit:
                item.setPos(start_pos + final_delta)
            item.setOpacity(opacity)
            item._finish_layout_interaction()
            dirty_rects.append(rect.translated(final_delta).adjusted(-8, -8, 8, 8))
            dirty_rects.append(item.sceneBoundingRect().adjusted(-8, -8, 8, 8))
        self._reset_manual_item_drag()
        self._finish_item_drag_layout_batch()
        self._update_scene_rects(dirty_rects)

    def _reset_manual_item_drag(self) -> None:
        self._manual_item_drag_pending = False
        self._manual_item_drag_active = False
        self._manual_item_drag_start_viewport = QPoint()
        self._manual_item_drag_start_scene = QPointF()
        self._manual_item_drag_delta = QPointF()
        self._manual_item_drag_anchor_item = None
        self._manual_item_drag_items = []

    def _update_scene_rect(self, rect: QRectF) -> None:
        self._update_scene_rects([rect])

    def _update_scene_rects(self, rects: list[QRectF]) -> None:
        dirty_rect = QRect()
        has_dirty_rect = False
        viewport_bounds = self.viewport().rect()
        for rect in rects:
            if rect.isNull() or not rect.isValid():
                continue
            polygon = self.mapFromScene(rect)
            viewport_rect = polygon.boundingRect().adjusted(-4, -4, 4, 4)
            viewport_rect = viewport_rect.intersected(viewport_bounds)
            if viewport_rect.isNull() or not viewport_rect.isValid():
                continue
            dirty_rect = viewport_rect if not has_dirty_rect else dirty_rect.united(viewport_rect)
            has_dirty_rect = True
        if has_dirty_rect:
            self.viewport().update(dirty_rect)

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawForeground(painter, rect)
        if not self._manual_item_drag_active or not self._manual_item_drag_items:
            return
        painter.save()
        painter.setOpacity(0.92)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        for item, _start_pos, source_rect, _opacity in self._manual_item_drag_items:
            target_rect = source_rect.translated(self._manual_item_drag_delta)
            if not target_rect.intersects(rect):
                continue
            pixmap = item.pixmap()
            if pixmap.isNull():
                continue
            painter.drawPixmap(
                target_rect,
                pixmap,
                QRectF(0, 0, pixmap.width(), pixmap.height()),
            )
        painter.restore()

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        painter.fillRect(rect, theme_color("canvas"))
        if self._manual_item_drag_active:
            return
        minor_pen = QPen(theme_color("grid"))
        minor_pen.setWidth(0)
        major_pen = QPen(theme_color("grid_major"))
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
        if isinstance(self.itemAt(event.position().toPoint()), BoardTextItem):
            super().mouseDoubleClickEvent(event)
            return
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
            menu = QMenu(self)
            menu.addAction("添加区域标题…", lambda: self.add_note(heading=True))
            menu.addAction("添加便签…", self.add_note)
            menu.addAction("按节点 / 标签分区排列", self.arrange_by_labels)
            menu.addSeparator()
            menu.addAction("撤销看板编辑", self.undo_layout)
            menu.addAction("重做看板编辑", self.redo_layout)
            menu.exec(event.globalPos())
            return
        if image_id not in self.selected_image_ids():
            self._select_image_id(image_id)
        menu = QMenu(self)
        preview_action = menu.addAction("通过预览显示图片")
        compare_action = menu.addAction("对比选中图片…")
        compare_action.setEnabled(len(self.selected_image_ids()) >= 2)
        menu.addSeparator()
        remove_action = menu.addAction("从当前项目移除")
        remove_action.setEnabled(bool(self.selected_image_ids()))
        chosen = menu.exec(event.globalPos())
        if chosen == compare_action:
            self.window()._compare_selected_images()
            event.accept()
            return
        if chosen == preview_action:
            self.imagePreviewRequested.emit(image_id)
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

    def _zoom_by(self, factor: float, *, anchor_pos=None, defer_quality=False) -> None:
        next_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self._zoom * factor))
        if abs(next_zoom - self._zoom) < 1e-9:
            return
        if factor > 1 and not defer_quality:
            self._ensure_focus_quality_for_items(self._selected_image_items(), target_zoom=next_zoom)
        scale_factor = next_zoom / max(0.001, self._zoom)
        center = self.mapToScene(self.viewport().rect().center())
        under_pointer = self.mapToScene(anchor_pos) if anchor_pos is not None else None
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.scale(scale_factor, scale_factor)
        self.centerOn(center)
        if under_pointer is not None:
            self.centerOn(self.mapToScene(self.viewport().rect().center())
                          + under_pointer - self.mapToScene(anchor_pos))
        self.setTransformationAnchor(anchor)
        self._view_mode = "manual"
        self._zoom = next_zoom
        self._schedule_visible_quality()

    def _fit_selected_items(self, items: list[QGraphicsItem]) -> None:
        if not items:
            return
        target = self._fit_target_for_items(items, margin=10.0)
        if target is None:
            return
        target_zoom, rect = target
        self._ensure_focus_quality_for_items(items, target_zoom=target_zoom)
        self._view_mode = "fit_selection"
        self._last_fit_selection_ids = tuple(sorted(self.selected_image_ids()))
        self._apply_fit_target(target_zoom, rect)

    def _refit_current_view_mode(self) -> None:
        if self._view_mode == "fit_selection":
            selected_items = self._selected_image_items()
            if selected_items:
                self._fit_items_to_view(selected_items, margin=10.0)
                return
        if self._view_mode == "fit_all":
            self._fit_items_to_view(self._visible_image_items() + list(self._note_items) + list(self._section_items), margin=80.0)

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
        if self._suppress_layout_changed:
            return
        if self._item_drag_layout_batch_active:
            self._item_drag_layout_pending = True
            return
        self._layout_change_pending = True
        self._layout_change_timer.start()

    def flush_pending_layout_change(self) -> bool:
        if self._item_drag_layout_pending:
            self._item_drag_layout_pending = False
            self._layout_change_pending = True
        if not self._layout_change_pending:
            return False
        self._layout_change_timer.stop()
        self._emit_pending_layout_changed()
        return True

    def _emit_pending_layout_changed(self) -> None:
        if not self._layout_change_pending:
            return
        self._layout_change_pending = False
        if self._suppress_layout_changed:
            return
        payload = self._capture_layout_payload()
        if self._last_layout is not None and payload != self._last_layout:
            history = getattr(self.window(), "edit_history", None)
            if history is not None:
                history.board_edit(self._last_layout, payload)
            self._undo_layouts.append(deepcopy(self._last_layout))
            self._undo_layouts = self._undo_layouts[-40:]
            self._redo_layouts.clear()
        self._last_layout = payload
        self.layoutChanged.emit()

    def _begin_item_drag_layout_batch(self) -> None:
        self._item_drag_layout_batch_active = True
        self._item_drag_layout_pending = False

    def _finish_item_drag_layout_batch(self) -> None:
        if not self._item_drag_layout_batch_active:
            return
        should_schedule = self._item_drag_layout_pending
        self._item_drag_layout_batch_active = False
        self._item_drag_layout_pending = False
        if should_schedule:
            self._schedule_layout_changed()

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
            frame.setBrush(theme_color("canvas"))
            frame.setPen(QPen(theme_color("border"), 1))
            item: QGraphicsItem = frame
            label = QGraphicsTextItem(image.file_name, frame)
            label.setDefaultTextColor(theme_color("text"))
            label.setTextWidth(width - 18)
            label.setPos(9, height / 2 - 20)
        else:
            item = BoardImageItem(
                image_id=image.id,
                pixmap=pixmap,
                file_path=image.file_path,
                badges=badges,
                on_layout_changed=self._schedule_layout_changed,
                source_aspect_ratio=_image_aspect_ratio(image, pixmap),
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
        token = (self._image_generation, image_id, MAX_BOARD_FOCUS_PIXMAP_SIDE)
        if image_id in self._focus_quality_image_ids and token not in self._pending_image_loads:
            return self._edit_local_display_item(item)
        self._pending_local_edit = (self._image_generation, image_id)
        self._ensure_focus_quality_for_items([item], force=True)
        return True

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

    def _ensure_focus_quality_for_items(
        self,
        items: list[QGraphicsItem],
        *,
        target_zoom: float | None = None,
        force: bool = False,
    ) -> None:
        for item in items:
            if not isinstance(item, BoardImageItem):
                continue
            image_id = int(item.image_id)
            rendered = max(item.sceneBoundingRect().width(), item.sceneBoundingRect().height()) * (target_zoom or self._zoom) * self.devicePixelRatioF()
            wanted = min(MAX_BOARD_FOCUS_PIXMAP_SIDE, max(768, 2 ** int(math.ceil(math.log2(max(1, rendered))))))
            if image_id in self._focus_quality_image_ids and self._loaded_image_sides.get(image_id, 0) >= wanted:
                continue
            if not force and target_zoom is not None and not self._needs_focus_quality_upgrade(item, target_zoom):
                continue
            self._focus_quality_image_ids.add(image_id)
            image = self._images_by_id.get(image_id)
            if image is None:
                continue
            self._request_image(image_id, wanted)

    @staticmethod
    def _needs_focus_quality_upgrade(item: BoardImageItem, target_zoom: float) -> bool:
        current_pixmap = item.source_pixmap()
        if current_pixmap.isNull():
            return True
        rendered_side = max(item.sceneBoundingRect().width(), item.sceneBoundingRect().height()) * target_zoom
        available_side = max(current_pixmap.width(), current_pixmap.height())
        return rendered_side > available_side * 0.92

    def _emit_selection_changed(self) -> None:
        selected_ids = self._collect_selected_image_ids()
        if selected_ids == self._selected_image_ids_cache:
            return
        self._selected_image_ids_cache = selected_ids
        self.selectionChanged.emit(selected_ids)

    def _visible_image_items(self) -> list[QGraphicsItem]:
        return [item for item in self._image_items.values() if item.isVisible()]

    def _fit_items_to_view(self, items: list[QGraphicsItem], *, margin: float) -> None:
        target = self._fit_target_for_items(items, margin=margin)
        if target is None:
            return
        next_zoom, rect = target
        self._apply_fit_target(next_zoom, rect)

    def _apply_fit_target(self, next_zoom: float, rect: QRectF) -> None:
        self.resetTransform()
        self.scale(next_zoom, next_zoom)
        self._zoom = next_zoom
        self.centerOn(rect.center())
        self._schedule_visible_quality()

    def _fit_target_for_items(self, items: list[QGraphicsItem], *, margin: float) -> tuple[float, QRectF] | None:
        if not items:
            return None
        rect = self._items_rect(items)
        if not rect.isValid() or rect.width() <= 1 or rect.height() <= 1:
            return None
        viewport_rect = self.viewport().rect()
        available_width = max(1.0, float(viewport_rect.width()) - margin * 2)
        available_height = max(1.0, float(viewport_rect.height()) - margin * 2)
        next_zoom = min(available_width / rect.width(), available_height / rect.height())
        return max(MIN_ZOOM, min(MAX_ZOOM, next_zoom)), rect

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
