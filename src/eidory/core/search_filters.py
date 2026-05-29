from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from eidory.models import ImageItem


SearchFilterValue = str | tuple[int, int, int] | int


@dataclass(frozen=True)
class SearchFilter:
    kind: str
    value: SearchFilterValue


@dataclass(frozen=True)
class SearchChainResult:
    images: list[ImageItem]
    semantic_searchable_count: int = 0
    semantic_candidate_limit: int = 0
    similar_searchable_count: int = 0
    similar_candidate_limit: int = 0
    color_searchable_count: int = 0
    color_indexed_count: int = 0
    color_candidate_limit: int = 0


VALID_FILTER_KINDS = {
    "semantic",
    "similar",
    "keyword",
    "color",
    "file_type",
    "orientation",
    "size",
}
SCORED_FILTER_KINDS = {"semantic", "similar", "color"}


def search_filter_to_payload(search_filter: SearchFilter) -> dict[str, object]:
    value = search_filter.value
    if isinstance(value, tuple):
        value = list(value)
    return {"kind": search_filter.kind, "value": value}


def search_filter_from_payload(payload: object) -> SearchFilter | None:
    if not isinstance(payload, dict):
        return None
    kind = payload.get("kind")
    value = payload.get("value")
    if kind not in VALID_FILTER_KINDS:
        return None
    if kind == "similar":
        try:
            image_id = int(value)
        except (TypeError, ValueError):
            return None
        if image_id <= 0:
            return None
        return SearchFilter("similar", image_id)
    if kind == "color":
        rgb = _rgb_from_payload(value)
        return SearchFilter("color", rgb) if rgb is not None else None
    if not isinstance(value, str):
        return None
    return SearchFilter(str(kind), value)


def _rgb_from_payload(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        rgb = tuple(int(part) for part in value)
    except (TypeError, ValueError):
        return None
    if any(part < 0 or part > 255 for part in rgb):
        return None
    return rgb


def last_score_filter_kind(filters: Sequence[SearchFilter]) -> str | None:
    for search_filter in reversed(filters):
        if search_filter.kind in SCORED_FILTER_KINDS:
            return search_filter.kind
    return None


def format_filter_chain(
    filters: Sequence[SearchFilter],
    *,
    image_label_for_id: Callable[[int], str] | None = None,
) -> str:
    if not filters:
        return "无"
    return " > ".join(
        filter_label(search_filter, image_label_for_id=image_label_for_id)
        for search_filter in filters
    )


def filter_label(
    search_filter: SearchFilter,
    *,
    image_label_for_id: Callable[[int], str] | None = None,
) -> str:
    if search_filter.kind == "semantic":
        return f"语义：{search_filter.value}"
    if search_filter.kind == "similar":
        image_id = int(search_filter.value)
        image_label = image_label_for_id(image_id) if image_label_for_id else f"#{image_id}"
        return f"相似：{image_label}"
    if search_filter.kind == "keyword":
        return f"关键词：{search_filter.value}"
    if search_filter.kind == "color":
        return f"颜色：{format_color_hex(search_filter.value)}"  # type: ignore[arg-type]
    if search_filter.kind == "file_type":
        return f"类型：{file_type_filter_label(str(search_filter.value))}"
    if search_filter.kind == "orientation":
        return f"方向：{orientation_filter_label(str(search_filter.value))}"
    if search_filter.kind == "size":
        return f"尺寸：{size_filter_label(str(search_filter.value))}"
    return str(search_filter.value)


def format_color_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def file_type_filter_label(value: str) -> str:
    labels = {
        "media:image": "图片",
        "media:video": "视频",
        "ext:.jpg": "JPG",
        "ext:.jpeg": "JPEG",
        "ext:.png": "PNG",
        "ext:.webp": "WebP",
        "ext:.mp4": "MP4",
        "ext:.mov": "MOV",
        "ext:.m4v": "M4V",
        "ext:.avi": "AVI",
        "ext:.mkv": "MKV",
        "ext:.webm": "WebM",
    }
    return labels.get(value, value)


def orientation_filter_label(value: str) -> str:
    return {
        "landscape": "横图",
        "portrait": "竖图",
        "square": "正方形",
    }.get(value, value)


def size_filter_label(value: str) -> str:
    return {
        "large": "大图 >= 2MP",
        "small": "小图 <= 0.5MP",
    }.get(value, value)
