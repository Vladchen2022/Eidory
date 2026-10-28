"""Non-destructive preview ink. Coordinates and widths are relative to the image."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import closing
from functools import cached_property
import json
import math
from pathlib import Path
import re
import sqlite3


MAX_POINTS = 100_000
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
Point = tuple[float, float]


@dataclass(frozen=True)
class Stroke:
    points: tuple[Point, ...]
    width: float  # Fraction of the shorter image edge, including when zoomed.
    color: str

    @cached_property
    def bounds(self):
        return (min(p[0] for p in self.points), min(p[1] for p in self.points),
                max(p[0] for p in self.points), max(p[1] for p in self.points))


def encode(strokes: tuple[Stroke, ...]) -> str:
    if sum(len(s.points) for s in strokes) > MAX_POINTS:
        raise ValueError("此图标记过多，请先清理部分标记")
    payload = json.dumps({"version": 1, "strokes": [
        {"points": s.points, "width": s.width, "color": s.color} for s in strokes
    ]}, separators=(",", ":"), allow_nan=False)
    if len(payload.encode()) > MAX_PAYLOAD_BYTES:
        raise ValueError("此图标记过多，请先清理部分标记")
    return payload


def decode(payload: str) -> tuple[Stroke, ...]:
    if len(payload.encode()) > MAX_PAYLOAD_BYTES:
        raise ValueError("标记数据超过大小限制，未覆盖原数据")
    doc = json.loads(payload)
    if doc.get("version") != 1:
        raise ValueError("无法读取此版本的标记，未覆盖原数据")
    strokes = []
    count = 0
    for raw in doc["strokes"]:
        width = float(raw["width"])
        color = raw["color"]
        points = tuple((float(x), float(y)) for x, y in raw["points"])
        count += len(points)
        if (not points or count > MAX_POINTS or not math.isfinite(width) or not 0 < width <= 2
                or not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color)
                or any(not math.isfinite(v) or not 0 <= v <= 1 for p in points for v in p)):
            raise ValueError("标记数据无效，未覆盖原数据")
        strokes.append(Stroke(points, width, color))
    return tuple(strokes)


class AnnotationStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path

    def load(self, image_id: int):
        # Do not use the library's long busy timeout on the preview's UI thread.
        with closing(sqlite3.connect(self.database_path, timeout=0.15)) as conn:
            row = conn.execute("SELECT payload,file_size,modified_time_ns FROM image_preview_annotations WHERE image_id=?",
                               (image_id,)).fetchone()
        return None if row is None else (decode(row[0]), (row[1], row[2]))

    def save(self, image_id: int, strokes: tuple[Stroke, ...], signature: tuple[int, int], *,
             expected: tuple[Stroke, ...] | None = None) -> None:
        payload = encode(strokes)
        with closing(sqlite3.connect(self.database_path, timeout=0.15)) as conn, conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN IMMEDIATE")
            if expected is not None:
                current = conn.execute("SELECT payload FROM image_preview_annotations WHERE image_id=?", (image_id,)).fetchone()
                if (decode(current[0]) if current else ()) != expected:
                    raise ValueError("其他预览窗口已修改此图标记。未覆盖其修改，请关闭本预览后重新打开。")
            if not strokes:
                conn.execute("DELETE FROM image_preview_annotations WHERE image_id=?", (image_id,))
            else:
                conn.execute("""INSERT INTO image_preview_annotations(image_id,payload,file_size,modified_time_ns)
                    VALUES(?,?,?,?) ON CONFLICT(image_id) DO UPDATE SET payload=excluded.payload,
                    file_size=excluded.file_size,modified_time_ns=excluded.modified_time_ns""",
                             (image_id, payload, *signature))


def _inside_interval(p: Point, q: Point, a: Point, b: Point, radius: float):
    """Parameter interval of segment p..q inside the capsule a..b (convex)."""
    vx, vy = q[0] - p[0], q[1] - p[1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    intervals = []
    for center in (a, b):
        x, y = p[0] - center[0], p[1] - center[1]
        aa, bb, cc = vx * vx + vy * vy, 2 * (x * vx + y * vy), x*x + y*y - radius*radius
        if aa < 1e-24:
            if cc <= 0:
                intervals.append((0., 1.))
        else:
            disc = bb*bb - 4*aa*cc
            if disc >= 0:
                root = math.sqrt(disc)
                lo, hi = max(0., (-bb-root)/(2*aa)), min(1., (-bb+root)/(2*aa))
                if lo <= hi:
                    intervals.append((lo, hi))
    if length > 1e-12:
        ux, uy = dx / length, dy / length
        x, y = p[0]-a[0], p[1]-a[1]
        lo, hi = 0., 1.
        for start, delta, minimum, maximum in (
            (x*ux+y*uy, vx*ux+vy*uy, 0., length),
            (-x*uy+y*ux, -vx*uy+vy*ux, -radius, radius),
        ):
            if abs(delta) < 1e-12:
                if not minimum <= start <= maximum:
                    hi = -1.
            else:
                t0, t1 = sorted(((minimum-start)/delta, (maximum-start)/delta))
                lo, hi = max(lo, t0), min(hi, t1)
        if lo <= hi:
            intervals.append((lo, hi))
    return (min(x[0] for x in intervals), max(x[1] for x in intervals)) if intervals else None


def erase(strokes: tuple[Stroke, ...], start: Point, end: Point, radius: float,
          aspect: tuple[float, float]) -> tuple[Stroke, ...]:
    """Cut only the portion swept by the eraser, including fast pointer movements."""
    w, h = aspect
    a, b = (start[0]*w, start[1]*h), (end[0]*w, end[1]*h)
    result = []
    for stroke in strokes:
        r = radius + stroke.width / 2
        left, top, right, bottom = min(a[0], b[0])-r, min(a[1], b[1])-r, max(a[0], b[0])+r, max(a[1], b[1])+r
        x0, y0, x1, y1 = stroke.bounds
        if x1*w < left or x0*w > right or y1*h < top or y0*h > bottom:
            result.append(stroke)
            continue
        points = tuple((x*w, y*h) for x, y in stroke.points)
        pieces, current, changed = [], [], False
        pairs = zip(points, points[1:]) if len(points) > 1 else [(points[0], points[0])]
        for p, q in pairs:
            interval = (None if (max(p[0], q[0]) < left or min(p[0], q[0]) > right
                        or max(p[1], q[1]) < top or min(p[1], q[1]) > bottom)
                        else _inside_interval(p, q, a, b, r))
            if interval is None:
                if not current:
                    current.append(p)
                current.append(q)
                continue
            changed = True
            lo, hi = interval
            at = lambda t: (p[0]+(q[0]-p[0])*t, p[1]+(q[1]-p[1])*t)
            if lo > 1e-9:
                if not current:
                    current.append(p)
                current.append(at(lo))
            if current:
                pieces.append(current)
                current = []
            if hi < 1-1e-9:
                current = [at(hi), q]
        if not changed:
            result.append(stroke)
        else:
            if current:
                pieces.append(current)
            result.extend(Stroke(tuple((x/w, y/h) for x, y in piece), stroke.width, stroke.color)
                          for piece in pieces)
    return tuple(result)
