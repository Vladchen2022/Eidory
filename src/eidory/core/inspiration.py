from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from eidory.models import ImageItem


@dataclass(frozen=True)
class InspirationTerm:
    title: str
    query: str
    axis: str = "visual"
    reason: str = ""
    id: int | None = None
    selected: bool = False


@dataclass(frozen=True)
class InspirationMatch:
    term_title: str
    query: str
    reason: str
    score: float | None


@dataclass(frozen=True)
class InspirationMixResult:
    images: list[ImageItem]
    matches_by_image_id: dict[int, list[InspirationMatch]]


def normalize_inspiration_terms(raw_terms: Sequence[object]) -> list[InspirationTerm]:
    terms: list[InspirationTerm] = []
    seen_titles: set[str] = set()
    for raw_term in raw_terms:
        if not isinstance(raw_term, dict):
            continue
        title = _clean_text(raw_term.get("title"), max_length=40)
        query = _clean_text(raw_term.get("query"), max_length=120)
        if not title or not query:
            continue
        key = title.casefold()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        terms.append(
            InspirationTerm(
                title=title,
                query=query,
                axis=_clean_text(raw_term.get("axis"), max_length=40) or "visual",
                reason=_clean_text(raw_term.get("reason"), max_length=160),
            )
        )
    return terms


def mix_inspiration_search_results(
    term_results: Sequence[tuple[InspirationTerm, Sequence[ImageItem]]],
    *,
    limit: int = 500,
) -> InspirationMixResult:
    matches_by_image_id: dict[int, list[InspirationMatch]] = {}
    best_images: dict[int, ImageItem] = {}
    for term, images in term_results:
        for image in images:
            matches_by_image_id.setdefault(image.id, []).append(
                InspirationMatch(
                    term_title=term.title,
                    query=term.query,
                    reason=term.reason,
                    score=image.score,
                )
            )
            current = best_images.get(image.id)
            if current is None or _score_value(image.score) > _score_value(current.score):
                best_images[image.id] = image

    mixed_ids: list[int] = []
    seen: set[int] = set()
    max_length = max((len(images) for _term, images in term_results), default=0)
    for rank in range(max_length):
        for _term, images in term_results:
            if rank >= len(images):
                continue
            image_id = images[rank].id
            if image_id in seen:
                continue
            seen.add(image_id)
            mixed_ids.append(image_id)
            if len(mixed_ids) >= limit:
                break
        if len(mixed_ids) >= limit:
            break

    images = [
        replace(best_images[image_id], score=_best_score(matches_by_image_id[image_id]))
        for image_id in mixed_ids
        if image_id in best_images
    ]
    return InspirationMixResult(images=images, matches_by_image_id=matches_by_image_id)


def _clean_text(value: object, *, max_length: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())[:max_length]


def _score_value(score: float | None) -> float:
    return float(score) if score is not None else float("-inf")


def _best_score(matches: Sequence[InspirationMatch]) -> float | None:
    scores = [match.score for match in matches if match.score is not None]
    return max(scores) if scores else None
