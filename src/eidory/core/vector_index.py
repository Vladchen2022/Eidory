from __future__ import annotations

import threading

import numpy as np

from eidory.core.metadata_store import MetadataStore


class VectorIndex:
    def __init__(
        self,
        store: MetadataStore,
        *,
        model_name: str,
        model_revision: str,
        embedding_dim: int,
    ):
        self.store = store
        self.model_name = model_name
        self.model_revision = model_revision
        self.embedding_dim = embedding_dim
        self._image_ids = np.empty((0,), dtype=np.int64)
        self._matrix = np.empty((0, embedding_dim), dtype=np.float32)
        self._id_to_index: dict[int, int] = {}
        self._dirty = True
        self._lock = threading.RLock()

    def invalidate(self) -> None:
        with self._lock:
            self._dirty = True

    def reload(self) -> None:
        image_ids, matrix = self.store.embeddings_for_model(
            model_name=self.model_name,
            model_revision=self.model_revision,
            embedding_dim=self.embedding_dim,
        )
        matrix = _normalize_matrix(matrix)
        with self._lock:
            self._image_ids = image_ids
            self._matrix = matrix
            self._id_to_index = {
                int(image_id): index
                for index, image_id in enumerate(image_ids)
            }
            self._dirty = False

    def upsert(self, image_id: int, vector: np.ndarray) -> None:
        vector = _normalize_vector(vector, self.embedding_dim)
        with self._lock:
            if self._dirty:
                return
            existing_index = self._id_to_index.get(int(image_id))
            if existing_index is None:
                self._dirty = True
            else:
                self._matrix[existing_index] = vector

    def remove(self, image_id: int) -> None:
        with self._lock:
            index = self._id_to_index.get(int(image_id))
            if index is None:
                return
            self._dirty = True

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        allowed_image_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        with self._lock:
            dirty = self._dirty
        if dirty:
            self.reload()

        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query vector must be one-dimensional")
        if query.shape[0] != self.embedding_dim:
            raise ValueError(
                f"query vector dim {query.shape[0]} does not match index dim {self.embedding_dim}"
            )
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm

        with self._lock:
            image_ids = self._image_ids
            matrix = self._matrix
            id_to_index = dict(self._id_to_index)
        if matrix.shape[0] == 0:
            return []
        scores = matrix @ query

        if allowed_image_ids is None:
            candidate_indexes = np.arange(matrix.shape[0])
        else:
            if not allowed_image_ids:
                return []
            candidate_indexes = np.fromiter(
                (
                    index
                    for image_id in allowed_image_ids
                    if (index := id_to_index.get(int(image_id))) is not None
                ),
                dtype=np.int64,
            )
            if candidate_indexes.shape[0] == 0:
                return []

        k = min(top_k, candidate_indexes.shape[0])
        candidate_scores = scores[candidate_indexes]
        if k == candidate_scores.shape[0]:
            order = candidate_indexes[np.argsort(-candidate_scores)]
        else:
            local_candidates = np.argpartition(-candidate_scores, k - 1)[:k]
            ordered_local = local_candidates[np.argsort(-candidate_scores[local_candidates])]
            order = candidate_indexes[ordered_local]
        return [
            (int(image_ids[index]), float(scores[index]))
            for index in order
        ]


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def _normalize_vector(vector: np.ndarray, embedding_dim: int) -> np.ndarray:
    query = np.asarray(vector, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError("embedding vector must be one-dimensional")
    if query.shape[0] != embedding_dim:
        raise ValueError(
            f"embedding vector dim {query.shape[0]} does not match index dim {embedding_dim}"
        )
    norm = np.linalg.norm(query)
    if norm == 0:
        raise ValueError("embedding vector must not be zero")
    return query / norm
