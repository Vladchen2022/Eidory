from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from eidory.core.embedding_provider import EmbeddingProvider
from eidory.core.metadata_store import MetadataStore
from eidory.core.search_service import SearchService, adaptive_candidate_limit
from eidory.core.vector_index import VectorIndex


class FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def model_revision(self) -> str:
        return "test"

    @property
    def dim(self) -> int:
        return 2

    def encode_image(self, image_path: str) -> np.ndarray:
        raise NotImplementedError

    def encode_text(self, text: str) -> np.ndarray:
        if "blue" in text:
            return np.asarray([0.0, 1.0], dtype=np.float32)
        return np.asarray([1.0, 0.0], dtype=np.float32)


class VectorSearchTest(unittest.TestCase):
    def test_numpy_vector_index_orders_by_cosine_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            other_folder_id = store.add_folder(str(Path(tmp) / "other"))
            red_id = self._insert_image(store, folder_id, Path(tmp) / "library" / "red.jpg", 1)
            blue_id = self._insert_image(store, folder_id, Path(tmp) / "library" / "blue.jpg", 2)
            other_id = self._insert_image(store, other_folder_id, Path(tmp) / "other" / "other.jpg", 3)
            video_id = self._insert_image(store, folder_id, Path(tmp) / "library" / "clip.mp4", 4)
            excluded_collection_id = store.create_collection("排除分类")
            store.assign_images_to_collection([red_id, blue_id], excluded_collection_id)

            provider = FakeEmbeddingProvider()
            store.upsert_embedding_success(
                image_id=red_id,
                model_name=provider.model_name,
                model_revision=provider.model_revision,
                vector=np.asarray([1.0, 0.0], dtype=np.float32),
            )
            store.upsert_embedding_success(
                image_id=blue_id,
                model_name=provider.model_name,
                model_revision=provider.model_revision,
                vector=np.asarray([0.0, 1.0], dtype=np.float32),
            )
            store.upsert_embedding_success(
                image_id=other_id,
                model_name=provider.model_name,
                model_revision=provider.model_revision,
                vector=np.asarray([0.8, 0.2], dtype=np.float32),
            )
            store.upsert_embedding_success(
                image_id=video_id,
                model_name=provider.model_name,
                model_revision=provider.model_revision,
                vector=np.asarray([1.0, 0.0], dtype=np.float32),
            )
            stats = store.embedding_stats(
                model_name=provider.model_name,
                model_revision=provider.model_revision,
                embedding_dim=provider.dim,
            )
            self.assertEqual(stats["total"], 3)
            self.assertEqual(stats["ready"], 3)
            self.assertEqual(stats["pending"], 0)

            index = VectorIndex(
                store,
                model_name=provider.model_name,
                model_revision=provider.model_revision,
                embedding_dim=provider.dim,
            )
            service = SearchService(store=store, embedding_provider=provider, vector_index=index)
            raw_ids = [
                image_id
                for image_id, _score in index.search(
                    np.asarray([1.0, 0.0], dtype=np.float32),
                    top_k=10,
                )
            ]
            self.assertNotIn(video_id, raw_ids)
            self.assertIsNone(
                store.embedding_vector_for_image(
                    video_id,
                    model_name=provider.model_name,
                    model_revision=provider.model_revision,
                    embedding_dim=provider.dim,
                )
            )

            red_results = service.semantic_search("red subject").images
            self.assertEqual(red_results[0].id, red_id)
            self.assertGreater(red_results[0].score or 0, red_results[1].score or 0)

            blue_results = service.semantic_search("blue subject").images
            self.assertEqual(blue_results[0].id, blue_id)

            scoped_results = service.semantic_search(
                "red subject",
                folder_id=other_folder_id,
            )
            self.assertEqual(scoped_results.searchable_count, 1)
            self.assertEqual(scoped_results.images[0].id, other_id)

            nested_results = service.semantic_search(
                "red subject",
                folder_path_prefix=str(Path(tmp) / "library"),
            )
            self.assertEqual(nested_results.searchable_count, 2)
            self.assertEqual(nested_results.images[0].id, red_id)

            excluded_results = service.semantic_search(
                "red subject",
                excluded_folder_path_prefixes=[str(Path(tmp) / "library")],
            )
            self.assertEqual(excluded_results.searchable_count, 1)
            self.assertEqual([image.id for image in excluded_results.images], [other_id])

            excluded_collection_results = service.semantic_search(
                "red subject",
                excluded_collection_ids=[excluded_collection_id],
            )
            self.assertEqual(excluded_collection_results.searchable_count, 1)
            self.assertEqual([image.id for image in excluded_collection_results.images], [other_id])

            scoped_to_blue = service.semantic_search(
                "red subject",
                allowed_image_ids={blue_id},
            )
            self.assertEqual(scoped_to_blue.searchable_count, 1)
            self.assertEqual([image.id for image in scoped_to_blue.images], [blue_id])

            similar_results = service.similar_image_search(red_id)
            self.assertEqual(similar_results.searchable_count, 2)
            self.assertEqual(similar_results.images[0].id, other_id)
            self.assertEqual(similar_results.images[1].id, blue_id)
            self.assertNotIn(red_id, [image.id for image in similar_results.images])
            self.assertGreater(similar_results.images[0].score or 0, similar_results.images[1].score or 0)

            similar_scoped_to_blue = service.similar_image_search(
                red_id,
                allowed_image_ids={blue_id},
            )
            self.assertEqual(similar_scoped_to_blue.searchable_count, 1)
            self.assertEqual([image.id for image in similar_scoped_to_blue.images], [blue_id])

            store.mark_embedding_failed(
                image_id=red_id,
                model_name=provider.model_name,
                model_revision=provider.model_revision,
                embedding_dim=provider.dim,
                error_message="boom",
            )
            store.mark_embedding_processing(
                image_id=blue_id,
                model_name=provider.model_name,
                model_revision=provider.model_revision,
                embedding_dim=provider.dim,
            )
            self.assertEqual(
                store.retry_failed_embeddings(
                    model_name=provider.model_name,
                    model_revision=provider.model_revision,
                    embedding_dim=provider.dim,
                ),
                2,
            )
            stats = store.embedding_stats(
                model_name=provider.model_name,
                model_revision=provider.model_revision,
                embedding_dim=provider.dim,
            )
            self.assertEqual(stats["pending"], 2)

    def test_color_search_orders_by_selected_color_without_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            red_path = root / "red.jpg"
            blue_path = root / "blue.jpg"
            video_path = root / "clip.mp4"
            self._make_image(red_path, "red")
            self._make_image(blue_path, "blue")
            video_path.write_bytes(b"fake mp4 bytes")

            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            red_id = self._insert_image(store, folder_id, red_path, 1)
            blue_id = self._insert_image(store, folder_id, blue_path, 2)
            video_id = self._insert_image(store, folder_id, video_path, 3)
            excluded_collection_id = store.create_collection("排除分类")
            store.assign_images_to_collection([red_id], excluded_collection_id)

            provider = FakeEmbeddingProvider()
            service = SearchService(
                store=store,
                embedding_provider=provider,
                vector_index=VectorIndex(
                    store,
                    model_name=provider.model_name,
                    model_revision=provider.model_revision,
                    embedding_dim=provider.dim,
                ),
            )

            result = service.color_search((255, 0, 0))
            self.assertEqual(result.searchable_count, 2)
            self.assertEqual(result.images[0].id, red_id)
            self.assertIn(blue_id, [image.id for image in result.images])
            self.assertNotIn(video_id, [image.id for image in result.images])
            self.assertGreater(result.images[0].score or 0, result.images[-1].score or 0)
            self.assertEqual(set(store.color_features_by_image_ids([red_id, blue_id])), {red_id, blue_id})

            scoped_result = service.color_search((255, 0, 0), allowed_image_ids={blue_id})
            self.assertEqual(scoped_result.searchable_count, 1)
            self.assertEqual([image.id for image in scoped_result.images], [blue_id])

            excluded_result = service.color_search(
                (255, 0, 0),
                excluded_folder_path_prefixes=[str(root)],
            )
            self.assertEqual(excluded_result.searchable_count, 0)
            self.assertEqual(excluded_result.images, [])

            excluded_collection_result = service.color_search(
                (255, 0, 0),
                excluded_collection_ids=[excluded_collection_id],
            )
            self.assertEqual(excluded_collection_result.searchable_count, 1)
            self.assertEqual([image.id for image in excluded_collection_result.images], [blue_id])

    def test_vector_index_upsert_and_remove_keep_loaded_index_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            first_id = self._insert_image(store, folder_id, Path(tmp) / "library" / "first.jpg", 1)
            second_id = self._insert_image(store, folder_id, Path(tmp) / "library" / "second.jpg", 2)
            third_id = self._insert_image(store, folder_id, Path(tmp) / "library" / "third.jpg", 3)
            provider = FakeEmbeddingProvider()
            for image_id, vector in (
                (first_id, np.asarray([1.0, 0.0], dtype=np.float32)),
                (second_id, np.asarray([0.0, 1.0], dtype=np.float32)),
            ):
                store.upsert_embedding_success(
                    image_id=image_id,
                    model_name=provider.model_name,
                    model_revision=provider.model_revision,
                    vector=vector,
                )
            index = VectorIndex(
                store,
                model_name=provider.model_name,
                model_revision=provider.model_revision,
                embedding_dim=provider.dim,
            )

            initial_scores = dict(index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=2))
            self.assertAlmostEqual(initial_scores[first_id], 1.0, places=6)

            updated_vector = np.asarray([0.0, 1.0], dtype=np.float32)
            store.upsert_embedding_success(
                image_id=first_id,
                model_name=provider.model_name,
                model_revision=provider.model_revision,
                vector=updated_vector,
            )
            index.upsert(first_id, updated_vector)
            updated_scores = dict(index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=2))
            self.assertAlmostEqual(updated_scores[first_id], 0.0, places=6)

            third_vector = np.asarray([1.0, 0.0], dtype=np.float32)
            store.upsert_embedding_success(
                image_id=third_id,
                model_name=provider.model_name,
                model_revision=provider.model_revision,
                vector=third_vector,
            )
            index.upsert(third_id, third_vector)
            self.assertIn(
                third_id,
                [image_id for image_id, _score in index.search(third_vector, top_k=3)],
            )

            store.mark_embedding_failed(
                image_id=third_id,
                model_name=provider.model_name,
                model_revision=provider.model_revision,
                embedding_dim=provider.dim,
                error_message="boom",
            )
            index.remove(third_id)
            self.assertNotIn(
                third_id,
                [image_id for image_id, _score in index.search(third_vector, top_k=3)],
            )

    def test_adaptive_candidate_limit(self) -> None:
        self.assertEqual(adaptive_candidate_limit(0), 0)
        self.assertEqual(adaptive_candidate_limit(300), 300)
        self.assertEqual(adaptive_candidate_limit(2_000), 1_000)
        self.assertEqual(adaptive_candidate_limit(20_000), 2_000)
        self.assertEqual(adaptive_candidate_limit(100_000), 5_000)

    @staticmethod
    def _insert_image(store: MetadataStore, folder_id: int, path: Path, mtime_ns: int) -> int:
        image_id, _state = store.upsert_image(
            folder_id=folder_id,
            file_path=str(path),
            file_size=123,
            width=10,
            height=10,
            created_time_ns=None,
            modified_time_ns=mtime_ns,
        )
        return image_id

    @staticmethod
    def _make_image(path: Path, color: str) -> None:
        image = Image.new("RGB", (64, 48), color=color)
        image.save(path)


if __name__ == "__main__":
    unittest.main()
