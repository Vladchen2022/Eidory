from __future__ import annotations

from dataclasses import replace
from collections.abc import Sequence

from eidory.core.media_types import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS
from eidory.core.metadata_store import MetadataStore
from eidory.core.search_filters import FilterGroup, FilterPlan, SearchFilter, SearchChainResult
from eidory.core.search_service import SearchService
from eidory.models import ImageItem


def balanced_rank(base, conditions, raw_matches):
    """Deterministic round-robin rank: every target gets a chance on the first screen."""
    conditions = sorted(set(conditions), key=lambda f: (f.kind, str(f.value)))
    if len(conditions) < 2:
        return list(base)
    queues = [iter(sorted(raw_matches[f], key=lambda im: (-(im.score or 0), im.id))) for f in conditions]
    seen, result = set(), []
    while queues:
        active = []
        for queue in queues:
            for im in queue:
                if im.id not in seen:
                    seen.add(im.id)
                    result.append(im)
                    active.append(queue)
                    break
        queues = active
    result.extend(im for im in base if im.id not in seen)
    return result


def matching_group_ids(group: FilterGroup, matches: dict[SearchFilter, set[int]]) -> set[int]:
    """Empty groups never match; an empty include section is handled as all."""
    if not group.filters:
        return set()
    values = [matches[f] for f in group.filters]
    return set.union(*values) if group.match_mode == "any" else set.intersection(*values)


def threshold_matches(condition: SearchFilter, images: Sequence[ImageItem]) -> list[ImageItem]:
    strength = condition.score_threshold or 0
    if strength <= 0 or condition.kind not in {"semantic", "similar", "color"}:
        return list(images)
    scores = [float(image.score) for image in images if image.score is not None]
    if not scores:
        return []
    # Evaluate relative strength against this condition's ENTIRE base scope.
    # Other positive/negative conditions never change this reference range.
    high = max(scores)
    low = 0.0 if condition.kind == "color" else min(scores)
    threshold = low + (high - low) * max(0, min(100, strength)) / 100.0
    return [image for image in images if image.score is not None and image.score >= threshold]


def same_filter_criteria(first: FilterPlan | None, second: FilterPlan) -> bool:
    """Strength changes are safe to replay; scope/rule changes require a query."""
    def criteria(plan):
        return tuple(tuple((g.match_mode, tuple((f.kind, f.value) for f in g.filters))
                           for g in getattr(plan, section)) for section in ("include", "exclude"))
    return first is not None and criteria(first) == criteria(second)


def rethreshold_result(result: SearchChainResult, plan: FilterPlan) -> SearchChainResult:
    """Reapply cutoffs to cached full-scope scores without SQL or model calls."""
    if result.ranked_scope_images is None or not same_filter_criteria(result.evaluated_plan, plan):
        raise ValueError("Live strength adjustment requires the same evaluated criteria")
    raw = {(f.kind, f.value): images for f, images in result.filter_matches.items()}
    raw_matches = {f: raw[(f.kind, f.value)] for f in (*plan.positive_filters, *plan.negative_filters)}
    matches = {f: {im.id for im in threshold_matches(f, images)} for f, images in raw_matches.items()}
    includes = [group for group in plan.include if group.filters]
    positive_ids = (set().union(*(matching_group_ids(group, matches) for group in includes))
                    if includes else {im.id for im in result.ranked_scope_images})
    excluded_ids = set().union(*(matching_group_ids(group, matches) for group in plan.exclude)) & positive_ids
    return replace(result, images=[im for im in result.ranked_scope_images if im.id in positive_ids],
                   excluded_image_ids=frozenset(excluded_ids), filter_matches=raw_matches, evaluated_plan=plan)


def refresh_annotation_matches(result: SearchChainResult, annotated_ids: set[int]) -> SearchChainResult:
    """Refresh ink membership against the full cached scope without rerunning AI searches."""
    plan = result.evaluated_plan
    if plan is None or result.ranked_scope_images is None:
        raise ValueError("Annotation refresh requires a complete evaluated scope")
    raw = dict(result.filter_matches)
    for condition in (*plan.positive_filters, *plan.negative_filters):
        if condition.kind == "annotation":
            raw[condition] = tuple(im for im in result.ranked_scope_images if im.id in annotated_ids)
    return rethreshold_result(replace(result, filter_matches=raw), plan)


class FilterEngine:
    """Pure rule orchestration over a stable scope, with no Qt or UI state."""

    def __init__(self, store: MetadataStore, search_service: SearchService):
        self.store = store
        self.search_service = search_service

    def evaluate(self, plan: FilterPlan, base_images: Sequence[ImageItem]) -> SearchChainResult:
        base = list(base_images)
        base_ids = {image.id for image in base}
        raw_cache: dict[tuple[str, object], list[ImageItem]] = {}
        raw_matches: dict[SearchFilter, tuple[ImageItem, ...]] = {}
        matches: dict[SearchFilter, set[int]] = {}
        stats: dict[str, int] = {}
        tags: dict[int, list[str]] | None = None

        for condition in (*plan.positive_filters, *plan.negative_filters):
            if condition in matches:
                continue
            key = (condition.kind, condition.value)
            if key not in raw_cache:
                if not base_ids:
                    found = []
                elif condition.kind in {"semantic", "similar", "color"}:
                    kwargs = {"allowed_image_ids": base_ids}
                    if condition.kind == "semantic":
                        result = self.search_service.semantic_search(str(condition.value), **kwargs)
                    elif condition.kind == "similar":
                        result = self.search_service.similar_image_search(int(condition.value), **kwargs)
                    else:
                        result = self.search_service.color_search(condition.value, generate_missing_features=True, **kwargs)
                    found = [im for im in result.images if im.id in base_ids]
                    stats[condition.kind + "_searchable_count"] = result.searchable_count
                    stats[condition.kind + "_candidate_limit"] = result.candidate_limit
                    if condition.kind == "color":
                        stats["color_indexed_count"] = result.indexed_count
                elif condition.kind == "keyword":
                    if tags is None:
                        tags = self.store.image_tags_for_image_ids(list(base_ids))
                    text = str(condition.value).strip().casefold()
                    found = [im for im in base if any(text in value.casefold() for value in (im.file_name, im.file_path, im.note or "", *tags.get(im.id, [])))]
                elif condition.kind == "collection":
                    ids: set[int] = set()
                    for raw_id in str(condition.value).split(","):
                        if raw_id.strip().isdigit():
                            ids.update(self.store.image_ids_for_collection(int(raw_id)))
                    found = [im for im in base if im.id in ids]
                elif condition.kind == "tag":
                    mode, _, values = str(condition.value).partition(":")
                    tag_ids = [int(v) for v in values.split(",") if v.strip().isdigit()]
                    ids = self.store.image_ids_matching_tags(tag_ids, tag_match_mode="any" if mode == "any" else "all") if tag_ids else set()
                    found = [im for im in base if im.id in ids]
                elif condition.kind == "ai_vision":
                    field, _, value = str(condition.value).partition(":")
                    ids = self.store.image_ids_matching_ai_vision(field, value)
                    found = [im for im in base if im.id in ids]
                elif condition.kind == "annotation":
                    ids = self.store.image_ids_with_preview_annotations() if condition.value == "present" else set()
                    found = [im for im in base if im.id in ids]
                else:
                    found = [im for im in base if self._matches_metadata(im, condition)]
                raw_cache[key] = found
            found = raw_cache[key]
            raw_matches[condition] = tuple(found)
            matches[condition] = {im.id for im in threshold_matches(condition, found)}

        includes = [group for group in plan.include if group.filters]
        positive_ids = set().union(*(matching_group_ids(group, matches) for group in includes)) if includes else base_ids
        excluded_ids = set().union(*(matching_group_ids(group, matches) for group in plan.exclude)) & positive_ids
        ranked = base
        # Exclusion scores never replace positive ranking. Metadata-only filters
        # retain the base order, including an explicitly frozen result set.
        for condition in reversed(plan.positive_filters):
            if condition.kind in {"semantic", "similar", "color"}:
                ranked_matches = raw_cache[(condition.kind, condition.value)]
                scored_ids = {im.id for im in ranked_matches}
                ranked = [*ranked_matches, *(replace(im, score=None) for im in base if im.id not in scored_ids)]
                break
        scored = [f for f in plan.positive_filters if f.kind in {'semantic', 'similar', 'color'}]
        if len(set(scored)) > 1:
            ranked = balanced_rank(base, scored, raw_matches)
        return SearchChainResult(
            images=[im for im in ranked if im.id in positive_ids],
            excluded_image_ids=frozenset(excluded_ids),
            filters_applied=True,
            filter_matches=raw_matches,
            scope_count=len(base_ids),
            evaluated_plan=plan,
            ranked_scope_images=tuple(ranked),
            **stats,
        )

    @staticmethod
    def _matches_metadata(image: ImageItem, condition: SearchFilter) -> bool:
        value = str(condition.value)
        if condition.kind == "file_type":
            if value == "media:image":
                return image.file_ext in SUPPORTED_IMAGE_EXTENSIONS
            if value == "media:video":
                return image.file_ext in SUPPORTED_VIDEO_EXTENSIONS
            return value.startswith("ext:") and image.file_ext == value[4:]
        if condition.kind == "status":
            return {"favorite": image.is_favorite, "unindexed": image.embedding_status != "ready", "missing": image.is_missing, "available": not image.is_missing}.get(value, False)
        if not image.width or not image.height:
            return False
        if condition.kind == "orientation":
            ratio = image.width / image.height
            return {"landscape": ratio > 1.08, "portrait": ratio < .92, "square": .92 <= ratio <= 1.08}.get(value, False)
        if condition.kind == "size":
            pixels = image.width * image.height
            return (value == "large" and pixels >= 2_000_000) or (value == "small" and pixels <= 500_000)
        return False
