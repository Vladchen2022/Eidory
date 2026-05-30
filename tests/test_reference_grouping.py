from __future__ import annotations

import unittest

import numpy as np

from eidory.core.reference_grouping import cluster_reference_vectors


class ReferenceGroupingTest(unittest.TestCase):
    def test_cluster_reference_vectors_splits_clear_visual_groups(self) -> None:
        vectors = {
            1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1, 0.0], dtype=np.float32),
            3: np.array([0.0, 1.0, 0.0], dtype=np.float32),
            4: np.array([0.1, 0.9, 0.0], dtype=np.float32),
        }

        groups = cluster_reference_vectors(vectors, max_groups=3)
        grouped_ids = [set(group.image_ids) for group in groups]

        self.assertEqual(len(groups), 2)
        self.assertIn({1, 2}, grouped_ids)
        self.assertIn({3, 4}, grouped_ids)

    def test_cluster_reference_vectors_ignores_invalid_vectors(self) -> None:
        groups = cluster_reference_vectors(
            {
                1: np.array([1.0, 0.0], dtype=np.float32),
                2: np.array([0.0, 0.0], dtype=np.float32),
                3: np.array([[1.0, 0.0]], dtype=np.float32),
            }
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].image_ids, [1])


if __name__ == "__main__":
    unittest.main()
