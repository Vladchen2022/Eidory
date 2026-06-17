from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from eidory.core.duplicate_detection import (
    build_image_dhash_records,
    find_duplicate_groups,
    find_near_duplicate_candidates,
    hamming_distance,
    image_dhash,
)
from eidory.models import ImageItem


class DuplicateDetectionTest(unittest.TestCase):
    def test_exact_duplicate_groups_by_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.jpg"
            second = root / "second.jpg"
            third = root / "third.jpg"
            Image.new("RGB", (64, 48), color="red").save(first)
            shutil.copy2(first, second)
            Image.new("RGB", (64, 48), color="blue").save(third)

            groups = find_duplicate_groups(
                [
                    self._image(1, first),
                    self._image(2, second),
                    self._image(3, third),
                ],
                folder_label_for_image={1: "A", 2: "B", 3: "C"},
            )

            exact = [group for group in groups if group.kind == "exact"]
            self.assertEqual(len(exact), 1)
            self.assertEqual({member.image.id for member in exact[0].members}, {1, 2})
            self.assertEqual({member.folder_label for member in exact[0].members}, {"A", "B"})

    def test_near_duplicate_groups_by_perceptual_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            large = root / "large.jpg"
            small = root / "small.jpg"
            other = root / "other.jpg"
            image = Image.new("RGB", (180, 100), color="white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 20, 150, 80), fill="black")
            draw.line((20, 20, 150, 80), fill="red", width=4)
            image.save(large)
            image.resize((90, 50)).save(small)
            Image.new("RGB", (180, 100), color="green").save(other)

            self.assertLessEqual(hamming_distance(image_dhash(large), image_dhash(small)), 8)
            groups = find_duplicate_groups(
                [
                    self._image(1, large, width=180, height=100),
                    self._image(2, small, width=90, height=50),
                    self._image(3, other, width=180, height=100),
                ],
                near_distance=8,
            )

            near = [group for group in groups if group.kind == "near"]
            self.assertEqual(len(near), 1)
            self.assertEqual({member.image.id for member in near[0].members}, {1, 2})

    def test_find_near_duplicate_candidates_uses_cached_library_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "library.jpg"
            imported = root / "imported.jpg"
            other = root / "other.jpg"
            image = Image.new("RGB", (200, 120), color="white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((30, 25, 170, 95), fill="black")
            draw.line((30, 95, 170, 25), fill="blue", width=5)
            image.save(library)
            image.resize((100, 60)).save(imported)
            Image.new("RGB", (200, 120), color="orange").save(other)

            records = build_image_dhash_records(
                [
                    self._image(1, library, width=200, height=120),
                    self._image(2, other, width=200, height=120),
                ]
            )
            candidates = find_near_duplicate_candidates(
                imported,
                hash_records=records,
                near_distance=8,
            )

            self.assertEqual([candidate.image.id for candidate in candidates], [1])
            self.assertLessEqual(candidates[0].distance, 8)

    def test_find_near_duplicate_candidates_can_include_same_path_for_import_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "existing.jpg"
            image = Image.new("RGB", (160, 90), color="white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 20, 130, 70), fill="black")
            image.save(existing)

            records = build_image_dhash_records([
                self._image(1, existing, width=160, height=90)
            ])

            default_candidates = find_near_duplicate_candidates(
                existing,
                hash_records=records,
                near_distance=8,
            )
            import_candidates = find_near_duplicate_candidates(
                existing,
                hash_records=records,
                near_distance=8,
                include_same_path=True,
            )

            self.assertEqual(default_candidates, [])
            self.assertEqual([candidate.image.id for candidate in import_candidates], [1])
            self.assertEqual(import_candidates[0].distance, 0)
            self.assertEqual(import_candidates[0].similarity, 1.0)

    @staticmethod
    def _image(
        image_id: int,
        path: Path,
        *,
        width: int = 64,
        height: int = 48,
    ) -> ImageItem:
        stat = path.stat()
        return ImageItem(
            id=image_id,
            folder_id=1,
            file_path=str(path),
            file_name=path.name,
            file_ext=path.suffix.lower(),
            file_size=stat.st_size,
            width=width,
            height=height,
            created_at=None,
            modified_at=None,
            modified_time_ns=stat.st_mtime_ns,
            imported_at="2026-01-01T00:00:00+00:00",
            last_seen_at="2026-01-01T00:00:00+00:00",
            thumbnail_path=None,
            thumbnail_status="ready",
            embedding_status="ready",
            is_missing=False,
            is_favorite=False,
            note=None,
        )


if __name__ == "__main__":
    unittest.main()
