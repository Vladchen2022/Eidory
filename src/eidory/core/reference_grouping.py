from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferenceGroup:
    image_ids: list[int]
    representative_id: int


def cluster_reference_vectors(
    vectors_by_image_id: dict[int, np.ndarray],
    *,
    max_groups: int = 6,
) -> list[ReferenceGroup]:
    clean_vectors: list[np.ndarray] = []
    image_ids: list[int] = []
    for image_id, vector in vectors_by_image_id.items():
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1 or array.size == 0:
            continue
        norm = np.linalg.norm(array)
        if norm == 0:
            continue
        image_ids.append(int(image_id))
        clean_vectors.append(array / norm)
    if not clean_vectors:
        return []
    if len(clean_vectors) <= 3:
        return [ReferenceGroup(image_ids=image_ids, representative_id=image_ids[0])]

    matrix = np.vstack(clean_vectors).astype(np.float32, copy=False)
    group_count = _target_group_count(len(image_ids), max_groups=max_groups)
    centers = _initial_centers(matrix, group_count)
    labels = np.zeros(matrix.shape[0], dtype=np.int64)
    for _iteration in range(16):
        scores = matrix @ centers.T
        next_labels = np.argmax(scores, axis=1)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        centers = _recompute_centers(matrix, labels, group_count, centers)

    groups: list[ReferenceGroup] = []
    for label in range(group_count):
        indexes = np.flatnonzero(labels == label)
        if indexes.size == 0:
            continue
        center = centers[label]
        best_local = int(indexes[np.argmax(matrix[indexes] @ center)])
        groups.append(
            ReferenceGroup(
                image_ids=[image_ids[int(index)] for index in indexes],
                representative_id=image_ids[best_local],
            )
        )
    return sorted(groups, key=lambda group: (-len(group.image_ids), min(group.image_ids)))


def _target_group_count(count: int, *, max_groups: int) -> int:
    if count <= 3:
        return 1
    if count <= 8:
        return 2
    if count <= 18:
        return 3
    if count <= 36:
        return 4
    return min(max_groups, 6)


def _initial_centers(matrix: np.ndarray, group_count: int) -> np.ndarray:
    mean = matrix.mean(axis=0)
    mean_norm = np.linalg.norm(mean)
    if mean_norm == 0:
        first_index = 0
    else:
        mean = mean / mean_norm
        first_index = int(np.argmax(matrix @ mean))
    center_indexes = [first_index]
    while len(center_indexes) < group_count:
        selected = matrix[np.array(center_indexes)]
        nearest_scores = np.max(matrix @ selected.T, axis=1)
        for index in center_indexes:
            nearest_scores[index] = 1
        center_indexes.append(int(np.argmin(nearest_scores)))
    return matrix[np.array(center_indexes)].copy()


def _recompute_centers(
    matrix: np.ndarray,
    labels: np.ndarray,
    group_count: int,
    previous_centers: np.ndarray,
) -> np.ndarray:
    centers = previous_centers.copy()
    for label in range(group_count):
        members = matrix[labels == label]
        if members.size == 0:
            continue
        center = members.mean(axis=0)
        norm = np.linalg.norm(center)
        centers[label] = center / norm if norm else previous_centers[label]
    return centers
