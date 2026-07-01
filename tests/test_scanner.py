from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from eidory.core.metadata_store import MetadataStore
from eidory.core.scanner import ImageScanner
from eidory.core.thumbnailer import Thumbnailer


class FakeVideoThumbnailer(Thumbnailer):
    def generate_video(self, image_id: int, video_path: str) -> Path:
        output_path = self.thumbnail_path_for(image_id)
        Image.new("RGB", (64, 36), color="black").save(output_path, "WEBP")
        return output_path


class ScannerTest(unittest.TestCase):
    def test_scan_folder_uses_batched_database_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            self._make_image(root / "a.jpg", "red")
            self._make_image(root / "b.jpg", "blue")

            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            scanner = ImageScanner(store, Thumbnailer(Path(tmp) / "thumbs"))

            with (
                patch.object(store, "upsert_image", side_effect=AssertionError("single-row upsert should not run")),
                patch.object(store, "upsert_images", wraps=store.upsert_images) as upsert_images,
                patch.object(store, "update_thumbnail", side_effect=AssertionError("single thumbnail update should not run")),
                patch.object(store, "update_thumbnails", wraps=store.update_thumbnails) as update_thumbnails,
                patch.object(store, "upsert_color_feature_success", side_effect=AssertionError("single color update should not run")),
                patch.object(store, "upsert_color_feature_successes", wraps=store.upsert_color_feature_successes) as color_successes,
            ):
                result = scanner.scan_folder(str(root))

            self.assertEqual(result.scanned_files, 2)
            self.assertEqual(result.new_files, 2)
            upsert_images.assert_called_once()
            update_thumbnails.assert_called_once()
            color_successes.assert_called_once()

    def test_scan_folder_generates_thumbnails_and_removes_deleted_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            hidden = root / ".hidden"
            nested = root / "nested"
            hidden.mkdir(parents=True)
            nested.mkdir()
            self._make_image(root / "a.jpg", "red")
            self._make_image(nested / "b.png", "blue")
            self._make_image(hidden / "ignored.jpg", "green")

            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            scanner = ImageScanner(store, Thumbnailer(Path(tmp) / "thumbs"))

            result = scanner.scan_folder(str(root))
            self.assertEqual(result.scanned_files, 2)
            self.assertEqual(result.new_files, 2)
            self.assertEqual(result.thumbnail_failures, 0)
            self.assertEqual(len(result.image_ids), 2)

            images = store.list_images(limit=10)
            self.assertEqual(len(images), 2)
            self.assertTrue(all(image.thumbnail_path for image in images))
            self.assertTrue(all(Path(image.thumbnail_path or "").exists() for image in images))

            os.remove(root / "a.jpg")
            result = scanner.scan_folder(str(root))
            self.assertEqual(result.scanned_files, 1)
            self.assertEqual(result.missing_marked, 1)
            self.assertTrue(result.removed_thumbnail_paths)

            self.assertEqual(store.count_missing_images(), 0)
            self.assertEqual(
                [image.file_name for image in store.list_images(limit=10)],
                ["b.png"],
            )

    def test_scan_missing_root_removes_folder_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            self._make_image(root / "a.jpg", "red")

            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            scanner = ImageScanner(store, Thumbnailer(Path(tmp) / "thumbs"))

            scanner.scan_folder(str(root))
            self.assertEqual(len(store.list_folders()), 1)

            os.remove(root / "a.jpg")
            root.rmdir()
            result = scanner.scan_folder(str(root))

            self.assertEqual(result.scanned_files, 0)
            self.assertEqual(result.missing_marked, 1)
            self.assertEqual(store.count_images(), 0)
            self.assertEqual(store.list_folders(include_inactive=True), [])

    def test_scan_folder_skip_paths_do_not_import_or_mark_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            first_path = root / "first.jpg"
            skipped_path = root / "skipped.jpg"
            self._make_image(first_path, "red")
            self._make_image(skipped_path, "blue")

            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            scanner = ImageScanner(store, Thumbnailer(Path(tmp) / "thumbs"))

            result = scanner.scan_folder(str(root), skip_paths={str(skipped_path)})
            self.assertEqual(result.scanned_files, 1)
            self.assertEqual([image.file_name for image in store.list_images(limit=10)], ["first.jpg"])

            result = scanner.scan_folder(str(root), skip_paths={str(first_path), str(skipped_path)})
            self.assertEqual(result.scanned_files, 0)
            self.assertEqual(result.missing_marked, 0)
            self.assertEqual(
                [image.file_name for image in store.list_images(limit=10)],
                ["first.jpg"],
            )

    def test_scan_regenerates_failed_thumbnail_without_file_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            self._make_image(root / "a.jpg", "red")

            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            scanner = ImageScanner(store, Thumbnailer(Path(tmp) / "thumbs"))

            scanner.scan_folder(str(root))
            image = store.list_images(limit=1)[0]
            Path(image.thumbnail_path or "").unlink()
            store.update_thumbnail(image.id, None, "failed")

            result = scanner.scan_folder(str(root))
            self.assertEqual(result.unchanged_files, 1)
            self.assertEqual(result.thumbnail_failures, 0)
            repaired = store.get_image(image.id)
            self.assertIsNotNone(repaired)
            self.assertEqual(repaired.thumbnail_status, "ready")
            self.assertTrue(Path(repaired.thumbnail_path or "").exists())

    def test_scan_folder_new_only_does_not_mark_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            first_path = root / "first.jpg"
            second_path = root / "second.jpg"
            self._make_image(first_path, "red")

            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            scanner = ImageScanner(store, Thumbnailer(Path(tmp) / "thumbs"))

            scanner.scan_folder(str(root))
            first_path.unlink()
            self._make_image(second_path, "blue")

            result = scanner.scan_folder_new_only(str(root))

            self.assertEqual(result.scanned_files, 1)
            self.assertEqual(result.new_files, 1)
            self.assertEqual(result.missing_marked, 0)
            self.assertEqual(store.count_missing_images(), 0)
            self.assertEqual(
                sorted(image.file_name for image in store.list_images(limit=10)),
                ["first.jpg", "second.jpg"],
            )

    def test_scan_allows_local_images_above_pillow_pixel_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            self._make_image(root / "a.jpg", "red")

            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            scanner = ImageScanner(store, Thumbnailer(Path(tmp) / "thumbs"))

            original_limit = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = 1
            try:
                result = scanner.scan_folder(str(root))
            finally:
                Image.MAX_IMAGE_PIXELS = original_limit

            self.assertEqual(result.scanned_files, 1)
            self.assertEqual(result.thumbnail_failures, 0)
            image = store.list_images(limit=1)[0]
            self.assertEqual((image.width, image.height), (64, 48))
            self.assertEqual(image.thumbnail_status, "ready")

    def test_scan_tolerates_lightly_truncated_local_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            image_path = root / "truncated.jpg"
            self._make_image(image_path, "red")
            data = image_path.read_bytes()
            image_path.write_bytes(data[:-2])

            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            scanner = ImageScanner(store, Thumbnailer(Path(tmp) / "thumbs"))

            result = scanner.scan_folder(str(root))
            self.assertEqual(result.scanned_files, 1)
            self.assertEqual(result.thumbnail_failures, 0)
            image = store.list_images(limit=1)[0]
            self.assertEqual(image.thumbnail_status, "ready")
            self.assertTrue(Path(image.thumbnail_path or "").exists())

    def test_import_files_imports_explicit_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "finder"
            root.mkdir()
            image_path = root / "dragged.jpg"
            ignored_path = root / "ignored.txt"
            self._make_image(image_path, "red")
            ignored_path.write_text("not image", encoding="utf-8")

            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            scanner = ImageScanner(store, Thumbnailer(Path(tmp) / "thumbs"))

            result = scanner.import_files([str(image_path), str(ignored_path)])
            self.assertEqual(result.scanned_files, 1)
            self.assertEqual(result.new_files, 1)
            self.assertEqual(len(result.image_ids), 1)
            image = store.get_image(result.image_ids[0])
            self.assertIsNotNone(image)
            self.assertEqual(image.file_name, "dragged.jpg")

    def test_scan_folder_generates_video_thumbnails_without_embedding_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "library"
            root.mkdir()
            self._make_image(root / "image.jpg", "red")
            video_path = root / "clip.mp4"
            video_path.write_bytes(b"fake mp4 bytes")

            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            scanner = ImageScanner(store, FakeVideoThumbnailer(Path(tmp) / "thumbs"))

            result = scanner.scan_folder(str(root))
            self.assertEqual(result.scanned_files, 2)
            self.assertEqual(result.new_files, 2)
            self.assertEqual(result.thumbnail_failures, 0)

            images = store.list_images(limit=10)
            video = next(image for image in images if image.file_name == "clip.mp4")
            self.assertEqual(video.file_ext, ".mp4")
            self.assertIsNone(video.width)
            self.assertIsNone(video.height)
            self.assertTrue(video.thumbnail_path)
            self.assertTrue(Path(video.thumbnail_path or "").exists())
            self.assertEqual(video.thumbnail_status, "ready")
            self.assertEqual(video.embedding_status, "ready")

            jobs = store.next_embedding_jobs(
                model_name="fake-model",
                model_revision="test",
                embedding_dim=2,
                limit=10,
            )
            self.assertEqual([job.file_name for job in jobs], ["image.jpg"])

            stats = store.embedding_stats(
                model_name="fake-model",
                model_revision="test",
                embedding_dim=2,
            )
            self.assertEqual(stats["total"], 1)
            self.assertEqual(stats["pending"], 1)
            self.assertEqual(
                [image.file_name for image in store.list_images(status_filter="unindexed", limit=10)],
                ["image.jpg"],
            )

    def test_import_files_accepts_explicit_video_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "finder"
            root.mkdir()
            video_path = root / "dragged.mp4"
            video_path.write_bytes(b"fake mp4 bytes")

            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            scanner = ImageScanner(store, Thumbnailer(Path(tmp) / "thumbs"))

            result = scanner.import_files([str(video_path)])
            self.assertEqual(result.scanned_files, 1)
            self.assertEqual(result.new_files, 1)
            image = store.get_image(result.image_ids[0])
            self.assertIsNotNone(image)
            self.assertEqual(image.file_name, "dragged.mp4")
            self.assertEqual(image.embedding_status, "ready")

    @staticmethod
    def _make_image(path: Path, color: str) -> None:
        image = Image.new("RGB", (64, 48), color=color)
        image.save(path)


if __name__ == "__main__":
    unittest.main()
