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
        self._updates: dict[int, np.ndarray] = {}
        self._removed: set[int] = set()

    def invalidate(self) -> None:
        with self._lock:
            self._dirty = True

    def reload(self) -> None:
        # Serialize publication with upsert/invalidate. An indexer may commit
        # during the DB read; its following upsert must not be lost on publish.
        with self._lock:
            image_ids, matrix = self.store.embeddings_for_model(
                model_name=self.model_name,
                model_revision=self.model_revision,
                embedding_dim=self.embedding_dim,
            )
            matrix = _normalize_matrix(matrix)
            matrix.flags.writeable = False
            self._image_ids = image_ids
            self._matrix = matrix
            self._id_to_index = {
                int(image_id): index
                for index, image_id in enumerate(image_ids)
            }
            self._updates.clear()
            self._removed.clear()
            self._dirty = False

    def upsert(self, image_id: int, vector: np.ndarray) -> None:
        vector = _normalize_vector(vector, self.embedding_dim)
        with self._lock:
            if self._dirty:
                return
            self._updates[int(image_id)] = vector
            self._removed.discard(int(image_id))
            if len(self._updates) + len(self._removed) >= 256:
                self._compact_updates()

    def remove(self, image_id: int) -> None:
        with self._lock:
            self._updates.pop(int(image_id), None)
            self._removed.add(int(image_id))
            if len(self._updates) + len(self._removed) >= 256:
                self._compact_updates()

    def _compact_updates(self) -> None:
        """Publish a new immutable snapshot without rereading the database."""
        ids = sorted((set(self._id_to_index) | set(self._updates)) - self._removed)
        matrix = np.empty((len(ids), self.embedding_dim), dtype=np.float32)
        for row, image_id in enumerate(ids):
            vector = self._updates.get(image_id)
            matrix[row] = vector if vector is not None else self._matrix[self._id_to_index[image_id]]
        matrix.flags.writeable = False
        self._image_ids = np.asarray(ids, dtype=np.int64)
        self._matrix = matrix
        self._id_to_index = {image_id: row for row, image_id in enumerate(ids)}
        self._updates.clear()
        self._removed.clear()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        allowed_image_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
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
            if self._dirty:
                self.reload()
            image_ids = self._image_ids
            matrix = self._matrix
            id_to_index = self._id_to_index
            updates = dict(self._updates)
            removed = set(self._removed)

        if allowed_image_ids is None:
            candidate_ids = [int(i) for i in image_ids if i not in removed]
            candidate_ids.extend(i for i in updates if i not in id_to_index)
            if updates:
                candidate_ids.sort()
        else:
            candidate_ids = [int(i) for i in allowed_image_ids
                             if i not in removed and (i in id_to_index or i in updates)]
        if not candidate_ids:
            return []
        indexes = np.asarray([id_to_index.get(i, -1) for i in candidate_ids], dtype=np.int64)
        in_base = indexes >= 0
        candidate_scores = np.empty(len(candidate_ids), dtype=np.float32)
        # Small node/folder scopes need not multiply the entire library matrix.
        if len(candidate_ids) < matrix.shape[0] // 4:
            candidate_scores[in_base] = matrix[indexes[in_base]] @ query
        else:
            scores = matrix @ query
            candidate_scores[in_base] = scores[indexes[in_base]]
        if updates:
            positions = [j for j, i in enumerate(candidate_ids) if i in updates]
            if positions:
                candidate_scores[positions] = np.vstack([updates[candidate_ids[j]] for j in positions]) @ query

        k = min(top_k, len(candidate_ids))
        if k == candidate_scores.shape[0]:
            order = np.argsort(-candidate_scores)
        else:
            local_candidates = np.argpartition(-candidate_scores, k - 1)[:k]
            order = local_candidates[np.argsort(-candidate_scores[local_candidates])]
        return [
            (candidate_ids[index], float(candidate_scores[index]))
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
