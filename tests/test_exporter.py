from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eidory.core.exporter import export_images_to_directory, export_library_to_directory
from eidory.core.metadata_store import MetadataStore


class ExporterTest(unittest.TestCase):
    def test_export_library_preserves_collection_tree_and_unassigned_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            first_path = root / "first.jpg"
            second_path = root / "second.jpg"
            third_path = root / "third.jpg"
            first_path.write_bytes(b"first")
            second_path.write_bytes(b"second")
            third_path.write_bytes(b"third")

            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            first_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(first_path),
                file_size=first_path.stat().st_size,
                width=1,
                height=1,
                created_time_ns=None,
                modified_time_ns=first_path.stat().st_mtime_ns,
            )
            second_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(second_path),
                file_size=second_path.stat().st_size,
                width=1,
                height=1,
                created_time_ns=None,
                modified_time_ns=second_path.stat().st_mtime_ns,
            )
            store.upsert_image(
                folder_id=folder_id,
                file_path=str(third_path),
                file_size=third_path.stat().st_size,
                width=1,
                height=1,
                created_time_ns=None,
                modified_time_ns=third_path.stat().st_mtime_ns,
            )
            parent = store.create_collection("参考")
            child = store.create_collection("室内/机械", parent)
            store.assign_images_to_collection([first_id], parent)
            store.assign_images_to_collection([second_id], child)

            target = Path(tmp) / "export"
            result = export_library_to_directory(store, target)

            self.assertEqual(result.copied, 3)
            self.assertEqual(result.skipped_missing, 0)
            self.assertTrue((target / "参考" / "first.jpg").is_file())
            self.assertTrue((target / "参考" / "室内_机械" / "second.jpg").is_file())
            self.assertTrue((target / "未归类" / "third.jpg").is_file())

    def test_export_selected_images_copies_unique_names_and_skips_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            first_path = root / "same.jpg"
            second_path = root / "other.jpg"
            first_path.write_bytes(b"first")
            second_path.write_bytes(b"second")

            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(root))
            first_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(first_path),
                file_size=first_path.stat().st_size,
                width=1,
                height=1,
                created_time_ns=None,
                modified_time_ns=first_path.stat().st_mtime_ns,
            )
            second_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(second_path),
                file_size=second_path.stat().st_size,
                width=1,
                height=1,
                created_time_ns=None,
                modified_time_ns=second_path.stat().st_mtime_ns,
            )
            store.mark_missing_for_folder(folder_id, [str(second_path)])

            first = store.get_image(first_id)
            second = store.get_image(second_id)
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            target = Path(tmp) / "export"
            (target).mkdir()
            (target / "other.jpg").write_bytes(b"existing")

            result = export_images_to_directory([first, second], target)

            self.assertEqual(result.copied, 1)
            self.assertEqual(result.skipped_missing, 1)
            self.assertTrue((target / "other-2.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
