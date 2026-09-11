from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from eidory.core.ai_vision import ai_vision_label
from eidory.models import ImageItem


SearchFilterValue = str | tuple[int, int, int] | int


@dataclass(frozen=True)
class SearchFilter:
    kind: str
    value: SearchFilterValue
    score_threshold: int | None = None


@dataclass(frozen=True)
class FilterGroup:
    filters: tuple[SearchFilter, ...] = ()
    match_mode: str = "all"

    def __post_init__(self) -> None:
        if self.match_mode not in {"all", "any"}:
            raise ValueError("filter group mode must be all or any")


@dataclass(frozen=True)
class FilterPlan:
    # Groups are OR-ed; conditions within a group use its explicit mode.
    include: tuple[FilterGroup, ...] = ()
    exclude: tuple[FilterGroup, ...] = ()

    @property
    def positive_filters(self) -> tuple[SearchFilter, ...]:
        return tuple(f for group in self.include for f in group.filters)

    @property
    def negative_filters(self) -> tuple[SearchFilter, ...]:
        return tuple(f for group in self.exclude for f in group.filters)

    @property
    def has_conditions(self) -> bool:
        return bool(self.positive_filters or self.negative_filters)

    def to_payload(self) -> dict[str, object]:
        def groups(values):
            return [{"match_mode": g.match_mode, "filters": [search_filter_to_payload(f) for f in g.filters]} for g in values]
        return {"include": groups(self.include), "exclude": groups(self.exclude)}

    @classmethod
    def from_payload(cls, payload: object) -> "FilterPlan":
        if not isinstance(payload, dict):
            raise ValueError("invalid filter plan")
        def groups(key):
            raw_groups = payload.get(key, [])
            if not isinstance(raw_groups, list):
                raise ValueError("invalid filter groups")
            result = []
            for raw in raw_groups:
                if not isinstance(raw, dict) or not isinstance(raw.get("filters"), list):
                    raise ValueError("invalid filter group")
                filters = tuple(search_filter_from_payload(f) for f in raw["filters"])
                if any(f is None for f in filters):
                    raise ValueError("invalid filter condition")
                result.append(FilterGroup(filters, str(raw.get("match_mode", "all"))))
            return tuple(result)
        return cls(groups("include"), groups("exclude"))


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
    filters_applied: bool = False
    excluded_image_ids: frozenset[int] = frozenset()
    filter_matches: dict[SearchFilter, tuple[ImageItem, ...]] = field(default_factory=dict)
    scope_count: int = 0
    evaluated_plan: FilterPlan | None = None
    # Full original scope in display rank order, before any strength cutoff.
    # Keep it so loosening a slider can restore previously hidden pictures.
    ranked_scope_images: tuple[ImageItem, ...] | None = None


VALID_FILTER_KINDS = {
    "semantic",
    "similar",
    "keyword",
    "color",
    "tag",
    "file_type",
    "orientation",
    "size",
    "ai_vision",
    "collection",
    "status",
}
SCORED_FILTER_KINDS = {"semantic", "similar", "color"}


def search_filter_to_payload(search_filter: SearchFilter) -> dict[str, object]:
    value = search_filter.value
    if isinstance(value, tuple):
        value = list(value)
    payload: dict[str, object] = {"kind": search_filter.kind, "value": value}
    if search_filter.score_threshold is not None:
        payload["score_threshold"] = search_filter.score_threshold
    return payload


def search_filter_from_payload(payload: object) -> SearchFilter | None:
    if not isinstance(payload, dict):
        return None
    kind = payload.get("kind")
    value = payload.get("value")
    if kind not in VALID_FILTER_KINDS:
        return None
    score_threshold = _score_threshold_from_payload(payload.get("score_threshold"))
    if kind == "similar":
        try:
            image_id = int(value)
        except (TypeError, ValueError):
            return None
        if image_id <= 0:
            return None
        return SearchFilter("similar", image_id, score_threshold)
    if kind == "color":
        rgb = _rgb_from_payload(value)
        return SearchFilter("color", rgb, score_threshold) if rgb is not None else None
    if not isinstance(value, str):
        return None
    return SearchFilter(str(kind), value, score_threshold if kind in SCORED_FILTER_KINDS else None)


def _score_threshold_from_payload(value: object) -> int | None:
    if value is None:
        return None
    try:
        threshold = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, min(100, threshold))


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
    if search_filter.kind == "tag":
        return f"标签：{search_filter.value}"
    if search_filter.kind == "collection":
        return f"文件夹：{search_filter.value}"
    if search_filter.kind == "file_type":
        return f"类型：{file_type_filter_label(str(search_filter.value))}"
    if search_filter.kind == "status":
        return "状态：" + {"favorite": "收藏", "unindexed": "未索引", "missing": "源文件丢失", "available": "文件可用"}.get(str(search_filter.value), str(search_filter.value))
    if search_filter.kind == "orientation":
        return f"方向：{orientation_filter_label(str(search_filter.value))}"
    if search_filter.kind == "size":
        return f"尺寸：{size_filter_label(str(search_filter.value))}"
    if search_filter.kind == "ai_vision":
        field, value = ai_vision_filter_parts(str(search_filter.value))
        return f"AI：{ai_vision_label(field, value)}" if field and value else f"AI：{search_filter.value}"
    return str(search_filter.value)


def ai_vision_filter_parts(value: str) -> tuple[str, str]:
    if ":" not in value:
        return "", ""
    field, raw_value = value.split(":", 1)
    field = field.strip()
    raw_value = raw_value.strip()
    return field, raw_value


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
