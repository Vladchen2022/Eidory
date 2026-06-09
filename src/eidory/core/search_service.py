from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eidory.core.color_features import encode_image_color, encode_query_color
from eidory.core.embedding_provider import EmbeddingProvider
from eidory.core.metadata_store import MetadataStore
from eidory.core.vector_index import VectorIndex
from eidory.models import ImageItem


@dataclass(frozen=True)
class SemanticSearchResult:
    images: list[ImageItem]
    searchable_count: int
    candidate_limit: int


@dataclass(frozen=True)
class ColorSearchResult:
    images: list[ImageItem]
    searchable_count: int
    indexed_count: int
    candidate_limit: int


def adaptive_candidate_limit(searchable_count: int) -> int:
    if searchable_count <= 0:
        return 0
    if searchable_count <= 500:
        return searchable_count
    if searchable_count <= 5_000:
        return min(searchable_count, 1_000)
    if searchable_count <= 50_000:
        return min(searchable_count, 2_000)
    return min(searchable_count, 5_000)


class SearchService:
    def __init__(
        self,
        *,
        store: MetadataStore,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex,
    ):
        self.store = store
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index

    def keyword_search(self, query: str, *, limit: int = 500) -> list[ImageItem]:
        return self.store.list_images(text_query=query, limit=limit)

    def semantic_search(
        self,
        query: str,
        *,
        folder_id: int | None = None,
        folder_path_prefix: str | None = None,
        collection_id: int | None = None,
        tag_id: int | None = None,
        tag_ids: list[int] | None = None,
        tag_match_mode: str = "any",
        status_filter: str | None = None,
        virtual_filter: str | None = None,
        allowed_image_ids: set[int] | None = None,
    ) -> SemanticSearchResult:
        if not query.strip():
            images = self.store.list_images(
                folder_id=folder_id,
                folder_path_prefix=folder_path_prefix,
                collection_id=collection_id,
                tag_id=tag_id,
                tag_ids=tag_ids,
                tag_match_mode=tag_match_mode,
                status_filter=status_filter,
                virtual_filter=virtual_filter,
                limit=500,
            )
            if allowed_image_ids is not None:
                images = [image for image in images if image.id in allowed_image_ids]
            return SemanticSearchResult(
                images=images,
                searchable_count=len(images),
                candidate_limit=len(images),
            )

        candidate_ids = self.store.searchable_image_ids_for_model(
            model_name=self.embedding_provider.model_name,
            model_revision=self.embedding_provider.model_revision,
            embedding_dim=self.embedding_provider.dim,
            folder_id=folder_id,
            folder_path_prefix=folder_path_prefix,
            collection_id=collection_id,
            tag_id=tag_id,
            tag_ids=tag_ids,
            tag_match_mode=tag_match_mode,
            status_filter=status_filter,
            virtual_filter=virtual_filter,
        )
        if allowed_image_ids is not None:
            candidate_ids = [image_id for image_id in candidate_ids if image_id in allowed_image_ids]
        candidate_limit = adaptive_candidate_limit(len(candidate_ids))
        if candidate_limit == 0:
            return SemanticSearchResult(
                images=[],
                searchable_count=len(candidate_ids),
                candidate_limit=0,
            )

        query_vector = self.embedding_provider.encode_text(query)
        matches = self.vector_index.search(
            query_vector,
            candidate_limit,
            allowed_image_ids=set(candidate_ids),
        )
        image_ids = [image_id for image_id, _score in matches]
        scores = {image_id: score for image_id, score in matches}
        return SemanticSearchResult(
            images=self.store.images_by_ids(image_ids, scores),
            searchable_count=len(candidate_ids),
            candidate_limit=candidate_limit,
        )

    def similar_image_search(
        self,
        image_id: int,
        *,
        folder_id: int | None = None,
        folder_path_prefix: str | None = None,
        collection_id: int | None = None,
        tag_id: int | None = None,
        tag_ids: list[int] | None = None,
        tag_match_mode: str = "any",
        status_filter: str | None = None,
        virtual_filter: str | None = None,
        allowed_image_ids: set[int] | None = None,
    ) -> SemanticSearchResult:
        query_vector = self.store.embedding_vector_for_image(
            image_id,
            model_name=self.embedding_provider.model_name,
            model_revision=self.embedding_provider.model_revision,
            embedding_dim=self.embedding_provider.dim,
        )
        if query_vector is None:
            return SemanticSearchResult(images=[], searchable_count=0, candidate_limit=0)

        candidate_ids = self.store.searchable_image_ids_for_model(
            model_name=self.embedding_provider.model_name,
            model_revision=self.embedding_provider.model_revision,
            embedding_dim=self.embedding_provider.dim,
            folder_id=folder_id,
            folder_path_prefix=folder_path_prefix,
            collection_id=collection_id,
            tag_id=tag_id,
            tag_ids=tag_ids,
            tag_match_mode=tag_match_mode,
            status_filter=status_filter,
            virtual_filter=virtual_filter,
        )
        candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id != image_id]
        if allowed_image_ids is not None:
            candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id in allowed_image_ids]

        candidate_limit = adaptive_candidate_limit(len(candidate_ids))
        if candidate_limit == 0:
            return SemanticSearchResult(
                images=[],
                searchable_count=len(candidate_ids),
                candidate_limit=0,
            )

        matches = self.vector_index.search(
            query_vector,
            candidate_limit,
            allowed_image_ids=set(candidate_ids),
        )
        image_ids = [matched_image_id for matched_image_id, _score in matches]
        scores = {matched_image_id: score for matched_image_id, score in matches}
        return SemanticSearchResult(
            images=self.store.images_by_ids(image_ids, scores),
            searchable_count=len(candidate_ids),
            candidate_limit=candidate_limit,
        )

    def color_search(
        self,
        rgb: tuple[int, int, int],
        *,
        folder_id: int | None = None,
        folder_path_prefix: str | None = None,
        collection_id: int | None = None,
        tag_id: int | None = None,
        tag_ids: list[int] | None = None,
        tag_match_mode: str = "any",
        status_filter: str | None = None,
        virtual_filter: str | None = None,
        allowed_image_ids: set[int] | None = None,
        limit: int = 5_000,
    ) -> ColorSearchResult:
        candidates = self.store.color_search_candidates(
            folder_id=folder_id,
            folder_path_prefix=folder_path_prefix,
            collection_id=collection_id,
            tag_id=tag_id,
            tag_ids=tag_ids,
            tag_match_mode=tag_match_mode,
            status_filter=status_filter,
            virtual_filter=virtual_filter,
        )
        if allowed_image_ids is not None:
            candidates = [image for image in candidates if image.id in allowed_image_ids]
        if not candidates:
            return ColorSearchResult(images=[], searchable_count=0, indexed_count=0, candidate_limit=0)

        candidate_ids = [image.id for image in candidates]
        self._ensure_color_features(candidates)
        features = self.store.color_features_by_image_ids(candidate_ids)
        if not features:
            return ColorSearchResult(
                images=[],
                searchable_count=len(candidates),
                indexed_count=0,
                candidate_limit=0,
            )

        query_vector = encode_query_color(rgb)
        scored: list[tuple[int, float]] = []
        for image_id in candidate_ids:
            vector = features.get(image_id)
            if vector is None:
                continue
            score = float(np.dot(vector, query_vector))
            if score > 0:
                scored.append((image_id, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        limited = scored[: max(0, limit)]
        image_ids = [image_id for image_id, _score in limited]
        scores = {image_id: score for image_id, score in limited}
        return ColorSearchResult(
            images=self.store.images_by_ids(image_ids, scores),
            searchable_count=len(candidates),
            indexed_count=len(features),
            candidate_limit=min(len(scored), max(0, limit)),
        )

    def _ensure_color_features(self, images: list[ImageItem]) -> None:
        image_ids = [image.id for image in images]
        ready_ids = set(self.store.color_features_by_image_ids(image_ids))
        failed_ids = self.store.color_feature_ids_by_status(image_ids, ["failed"])
        for image in images:
            if image.id in ready_ids or image.id in failed_ids:
                continue
            path = Path(image.file_path)
            if image.is_missing or not path.exists():
                continue
            try:
                self.store.upsert_color_feature_success(
                    image_id=image.id,
                    vector=encode_image_color(image.file_path),
                )
            except Exception as exc:
                self.store.mark_color_feature_failed(image.id, str(exc))
