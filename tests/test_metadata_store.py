from __future__ import annotations

import tempfile
import threading
import unittest
import sqlite3
from pathlib import Path

from eidory.core.ai_vision import AIVisionAnalysis, AI_VISION_PROMPT_VERSION
from eidory.core.metadata_store import MetadataStore, TEMPORARY_PROJECT_COLORS


class MetadataStoreTest(unittest.TestCase):
    def test_connections_use_busy_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()

            with store.connect() as conn:
                timeout = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])

            self.assertGreaterEqual(timeout, 30_000)

    def test_concurrent_upsert_same_path_is_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetadataStore(Path(tmp) / "eidory.sqlite3")
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_path = str(Path(tmp) / "library" / "same.jpg")
            barrier = threading.Barrier(8)
            errors: list[BaseException] = []
            states: list[str] = []

            def worker() -> None:
                try:
                    barrier.wait(timeout=3)
                    _image_id, state = store.upsert_image(
                        folder_id=folder_id,
                        file_path=image_path,
                        file_size=123,
                        width=10,
                        height=20,
                        created_time_ns=None,
                        modified_time_ns=1_700_000_000_000_000_000,
                    )
                    states.append(state)
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            self.assertEqual(store.count_images(), 1)
            self.assertEqual(states.count("new"), 1)
            self.assertEqual(states.count("unchanged"), 7)

    def test_tags_note_and_favorite_persist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "image.jpg"),
                file_size=123,
                width=10,
                height=20,
                created_time_ns=None,
                modified_time_ns=1_700_000_000_000_000_000,
            )
            self.assertEqual(state, "new")

            store.set_image_tags(image_id, ["赛博朋克", "灯光", "赛博朋克"])
            store.update_note(image_id, "warm light reference")
            store.update_favorite(image_id, True)

            reopened = MetadataStore(db_path)
            reopened.initialize()
            image = reopened.get_image(image_id)
            self.assertIsNotNone(image)
            self.assertTrue(image.is_favorite)
            self.assertEqual(image.note, "warm light reference")
            self.assertEqual(reopened.get_image_tags(image_id), ["灯光", "赛博朋克"])

    def test_path_prefix_remap_updates_folders_and_recovers_missing_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            old_root = Path(tmp) / "old-library"
            new_root = Path(tmp) / "new-library"
            new_root.mkdir()
            new_file = new_root / "image.jpg"
            new_file.write_bytes(b"new image bytes")

            store = MetadataStore(db_path)
            store.initialize()
            folder_id = store.add_folder(str(old_root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(old_root / "image.jpg"),
                file_size=1,
                width=10,
                height=20,
                created_time_ns=None,
                modified_time_ns=1,
            )
            self.assertEqual(store.mark_missing_for_folder(folder_id, []), 1)

            self.assertEqual(
                store.path_prefix_match_counts(str(old_root)),
                {"folders": 1, "images": 1, "missing": 1},
            )
            result = store.remap_path_prefix(str(old_root), str(new_root))

            self.assertEqual(result["folders_updated"], 1)
            self.assertEqual(result["images_updated"], 1)
            self.assertEqual(result["relinked"], 1)
            self.assertEqual(result["still_missing"], 0)
            self.assertEqual(result["conflicts"], 0)
            folder = store.get_folder(folder_id)
            self.assertIsNotNone(folder)
            self.assertEqual(folder.folder_path, str(new_root))
            image = store.get_image(image_id)
            self.assertIsNotNone(image)
            self.assertEqual(image.file_path, str(new_file))
            self.assertFalse(image.is_missing)
            self.assertEqual(image.thumbnail_status, "pending")
            self.assertEqual(image.embedding_status, "pending")

    def test_repair_missing_image_path_relinks_single_file_and_marks_media_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            root = Path(tmp) / "library"
            root.mkdir()
            new_path = root / "relinked.jpg"
            new_path.write_bytes(b"new image bytes")
            stat = new_path.stat()

            store = MetadataStore(db_path)
            store.initialize()
            folder_id = store.add_folder(str(root))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(root / "missing.jpg"),
                file_size=1,
                width=10,
                height=20,
                created_time_ns=None,
                modified_time_ns=1,
            )
            self.assertEqual(store.mark_missing_for_folder(folder_id, []), 1)

            store.repair_missing_image_path(
                image_id,
                file_path=str(new_path),
                file_size=stat.st_size,
                width=640,
                height=480,
                modified_time_ns=stat.st_mtime_ns,
            )

            image = store.get_image(image_id)
            self.assertIsNotNone(image)
            self.assertFalse(image.is_missing)
            self.assertEqual(image.file_path, str(new_path))
            self.assertEqual(image.file_name, "relinked.jpg")
            self.assertEqual((image.width, image.height), (640, 480))
            self.assertEqual(image.thumbnail_status, "pending")
            self.assertEqual(image.embedding_status, "pending")

    def test_remove_missing_images_from_library_keeps_existing_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            root = Path(tmp) / "library"
            store = MetadataStore(db_path)
            store.initialize()
            folder_id = store.add_folder(str(root))
            missing_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(root / "missing.jpg"),
                file_size=1,
                width=10,
                height=20,
                created_time_ns=None,
                modified_time_ns=1,
            )
            existing_path = str(root / "existing.jpg")
            existing_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=existing_path,
                file_size=2,
                width=30,
                height=40,
                created_time_ns=None,
                modified_time_ns=2,
            )
            self.assertEqual(store.mark_missing_for_folder(folder_id, [existing_path]), 1)

            thumbnail_paths, removed = store.remove_missing_images_from_library()

            self.assertEqual(thumbnail_paths, [])
            self.assertEqual(removed, 1)
            self.assertIsNone(store.get_image(missing_id))
            self.assertIsNotNone(store.get_image(existing_id))
            self.assertEqual(store.count_missing_images(), 0)

    def test_remove_unclassified_active_roots_removes_stale_scan_roots_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()

            stale_folder_id = store.add_folder(str(Path(tmp) / "old-root"))
            stale_image_id, _state = store.upsert_image(
                folder_id=stale_folder_id,
                file_path=str(Path(tmp) / "old-root" / "orphan.jpg"),
                file_size=1,
                width=10,
                height=20,
                created_time_ns=None,
                modified_time_ns=1,
            )
            store.update_thumbnail(stale_image_id, str(Path(tmp) / "thumb_old.webp"), "ready")

            managed_folder_id = store.add_folder(str(Path(tmp) / "managed-root"))
            managed_image_id, _state = store.upsert_image(
                folder_id=managed_folder_id,
                file_path=str(Path(tmp) / "managed-root" / "kept.jpg"),
                file_size=2,
                width=30,
                height=40,
                created_time_ns=None,
                modified_time_ns=2,
            )
            collection_id = store.create_collection("参考")
            store.assign_images_to_collection([managed_image_id], collection_id)
            self.assertEqual(
                [folder.id for folder in store.list_folders_with_collection_images()],
                [managed_folder_id],
            )

            thumbnail_paths, removed_roots, removed_images = store.remove_unclassified_active_roots()

            self.assertEqual(thumbnail_paths, [str(Path(tmp) / "thumb_old.webp")])
            self.assertEqual(removed_roots, 1)
            self.assertEqual(removed_images, 1)
            self.assertIsNone(store.get_folder(stale_folder_id))
            self.assertIsNone(store.get_image(stale_image_id))
            self.assertIsNotNone(store.get_folder(managed_folder_id))
            self.assertIsNotNone(store.get_image(managed_image_id))

    def test_search_feedback_is_query_and_model_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "image.jpg"),
                file_size=123,
                width=10,
                height=20,
                created_time_ns=None,
                modified_time_ns=1_700_000_000_000_000_000,
            )

            model = {
                "model_name": "fake-model",
                "model_revision": "test",
                "embedding_dim": 2,
            }
            store.upsert_search_feedback(
                query="机械设备细节",
                image_id=image_id,
                score=0.42,
                label="relevant",
                **model,
            )
            self.assertEqual(
                store.get_search_feedback(
                    query="机械设备细节",
                    image_id=image_id,
                    **model,
                ),
                "relevant",
            )
            self.assertIsNone(
                store.get_search_feedback(
                    query="温暖阳光室内",
                    image_id=image_id,
                    **model,
                )
            )
            self.assertEqual(
                store.search_feedback_counts(query="机械设备细节", **model),
                {"relevant": 1, "irrelevant": 0, "ignored": 0},
            )

            store.upsert_search_feedback(
                query="机械设备细节",
                image_id=image_id,
                score=0.38,
                label="irrelevant",
                **model,
            )
            self.assertEqual(
                store.get_search_feedback(
                    query="机械设备细节",
                    image_id=image_id,
                    **model,
                ),
                "irrelevant",
            )
            self.assertEqual(
                store.search_feedback_counts(query="机械设备细节", **model),
                {"relevant": 0, "irrelevant": 1, "ignored": 0},
            )

    def test_saved_views_crud_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()

            view_id = store.upsert_saved_view("室内参考", '{"status_filter":"favorite"}')
            view = store.get_saved_view(view_id)
            self.assertIsNotNone(view)
            self.assertEqual(view.name, "室内参考")
            self.assertEqual(view.payload_json, '{"status_filter":"favorite"}')

            same_id = store.upsert_saved_view("室内参考", '{"status_filter":"all"}')
            self.assertEqual(same_id, view_id)
            self.assertEqual(store.get_saved_view(view_id).payload_json, '{"status_filter":"all"}')

            self.assertTrue(store.rename_saved_view(view_id, "室内收藏"))
            self.assertEqual([saved_view.name for saved_view in store.list_saved_views()], ["室内收藏"])

            reopened = MetadataStore(db_path)
            reopened.initialize()
            self.assertEqual(reopened.get_saved_view(view_id).name, "室内收藏")
            self.assertTrue(reopened.delete_saved_view(view_id))
            self.assertEqual(reopened.list_saved_views(), [])

    def test_temporary_projects_persist_image_order_and_delete_without_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            first_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=10,
                height=20,
                created_time_ns=None,
                modified_time_ns=1_700_000_000_000_000_001,
            )
            second_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "second.jpg"),
                file_size=456,
                width=30,
                height=40,
                created_time_ns=None,
                modified_time_ns=1_700_000_000_000_000_002,
            )

            project_id = store.create_temporary_project(
                "飞行器参考",
                [second_id, first_id],
                summary="飞行器和引擎结构参考",
            )
            changed = store.add_images_to_temporary_project(
                project_id,
                [second_id],
                intent_labels={second_id: "引擎细节 +1"},
                intent_queries={second_id: "老旧引擎，机械结构"},
            )
            self.assertGreaterEqual(changed, 1)
            projects = store.list_temporary_projects()
            self.assertEqual(len(projects), 1)
            self.assertEqual(projects[0].name, "飞行器参考")
            self.assertEqual(projects[0].image_count, 2)
            self.assertEqual(projects[0].summary, "飞行器和引擎结构参考")
            self.assertEqual(projects[0].color_hex, TEMPORARY_PROJECT_COLORS[0])
            self.assertEqual(store.temporary_project_image_ids(project_id), [second_id, first_id])
            self.assertEqual(
                store.temporary_project_image_badges(project_id),
                {second_id: ["引擎细节 +1"]},
            )

            self.assertTrue(store.delete_temporary_project(project_id))
            self.assertEqual(store.list_temporary_projects(), [])
            self.assertIsNotNone(store.get_image(first_id))
            self.assertIsNotNone(store.get_image(second_id))

    def test_temporary_projects_persist_after_reopening_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=10,
                height=20,
                created_time_ns=None,
                modified_time_ns=1,
            )
            project_id = store.create_temporary_project(
                "长期项目",
                [image_id],
                summary="关闭软件后仍应保留。",
                color_hex="#756742",
            )
            store.add_images_to_temporary_project(
                project_id,
                [image_id],
                intent_labels={image_id: "机械住处"},
            )

            reopened = MetadataStore(db_path)
            reopened.initialize()

            projects = reopened.list_temporary_projects()
            self.assertEqual(len(projects), 1)
            self.assertEqual(projects[0].name, "长期项目")
            self.assertEqual(projects[0].summary, "关闭软件后仍应保留。")
            self.assertEqual(projects[0].color_hex, "#756742")
            self.assertEqual(reopened.temporary_project_image_ids(project_id), [image_id])
            self.assertEqual(
                reopened.temporary_project_image_badges(project_id),
                {image_id: ["机械住处"]},
            )

    def test_temporary_project_details_can_be_updated_with_unique_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=10,
                height=20,
                created_time_ns=None,
                modified_time_ns=1,
            )

            existing_id = store.create_temporary_project("机械参考", [image_id])
            project_id = store.create_temporary_project("未命名", [image_id])

            updated = store.update_temporary_project_details(
                project_id,
                name="机械参考",
                summary="用于寻找机械细节。",
            )

            self.assertIsNotNone(updated)
            self.assertEqual(updated.name, "机械参考 2")
            self.assertEqual(updated.summary, "用于寻找机械细节。")
            self.assertEqual(store.get_temporary_project(existing_id).name, "机械参考")

            cleared = store.update_temporary_project_details(
                project_id,
                summary="",
            )
            self.assertIsNotNone(cleared)
            self.assertEqual(cleared.summary, "")
            with self.assertRaises(ValueError):
                store.update_temporary_project_details(project_id, name="   ")

    def test_temporary_project_color_can_be_set_and_next_color_cycles_from_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=10,
                height=20,
                created_time_ns=None,
                modified_time_ns=1,
            )

            first_id = store.create_temporary_project(
                "第一批",
                [image_id],
                color_hex="#756742",
            )
            second_id = store.create_temporary_project("第二批", [image_id])

            self.assertEqual(store.get_temporary_project(first_id).color_hex, "#756742")
            self.assertEqual(
                store.get_temporary_project(second_id).color_hex,
                TEMPORARY_PROJECT_COLORS[2],
            )

    def test_temporary_project_sort_order_migrates_existing_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE temporary_projects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        summary TEXT NOT NULL DEFAULT '',
                        color_hex TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO temporary_projects(name, summary, color_hex, created_at, updated_at)
                    VALUES ('旧项目', '', '', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
                    """
                )

            store = MetadataStore(db_path)
            store.initialize()

            with sqlite3.connect(db_path) as conn:
                columns = {row[1] for row in conn.execute("PRAGMA table_info(temporary_projects)")}
                indexes = {row[1] for row in conn.execute("PRAGMA index_list(temporary_projects)")}
            self.assertIn("sort_order", columns)
            self.assertIn("idx_temporary_projects_sort_order", indexes)
            self.assertEqual([project.name for project in store.list_temporary_projects()], ["旧项目"])

    def test_temporary_project_add_and_remove_only_changes_project_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_ids: list[int] = []
            for index in range(3):
                image_id, _state = store.upsert_image(
                    folder_id=folder_id,
                    file_path=str(Path(tmp) / "library" / f"{index}.jpg"),
                    file_size=100 + index,
                    width=10,
                    height=20,
                    created_time_ns=None,
                    modified_time_ns=1_700_000_000_000_000_010 + index,
                )
                image_ids.append(image_id)

            project_id = store.create_temporary_project("机械参考", [image_ids[0]])
            store.add_images_to_temporary_project(
                project_id,
                [image_ids[1], image_ids[2]],
                intent_labels={image_ids[1]: "破旧工坊"},
                intent_queries={image_ids[1]: "破旧工坊，昏暗灯光"},
            )

            self.assertEqual(store.temporary_project_image_ids(project_id), image_ids)
            self.assertEqual(
                store.temporary_project_image_badges(project_id),
                {image_ids[1]: ["破旧工坊"]},
            )

            removed = store.remove_images_from_temporary_project(project_id, [image_ids[1]])
            self.assertEqual(removed, 1)
            self.assertEqual(store.temporary_project_image_ids(project_id), [image_ids[0], image_ids[2]])
            self.assertEqual(store.temporary_project_image_badges(project_id), {})
            self.assertIsNotNone(store.get_image(image_ids[1]))

    def test_clear_temporary_projects_removes_only_project_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            image_id, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "first.jpg"),
                file_size=123,
                width=10,
                height=20,
                created_time_ns=None,
                modified_time_ns=1,
            )
            first_project = store.create_temporary_project("第一组", [image_id])
            store.create_temporary_project("第二组", [image_id])
            store.add_images_to_temporary_project(
                first_project,
                [image_id],
                intent_labels={image_id: "破旧工坊"},
            )

            cleared = store.clear_temporary_projects()

            self.assertEqual(cleared, 2)
            self.assertEqual(store.list_temporary_projects(), [])
            self.assertEqual(store.temporary_project_image_ids(first_project), [])
            self.assertIsNotNone(store.get_image(image_id))

    def test_duration_persists_and_images_can_sort_by_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            folder_id = store.add_folder(str(Path(tmp) / "library"))
            first, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "short.mp4"),
                file_size=300,
                width=640,
                height=360,
                created_time_ns=None,
                modified_time_ns=3,
                duration_ms=5_000,
            )
            second, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "long.mp4"),
                file_size=100,
                width=1920,
                height=1080,
                created_time_ns=None,
                modified_time_ns=2,
                duration_ms=20_000,
            )
            third, _state = store.upsert_image(
                folder_id=folder_id,
                file_path=str(Path(tmp) / "library" / "still.jpg"),
                file_size=200,
                width=1000,
                height=1000,
                created_time_ns=None,
                modified_time_ns=1,
            )

            self.assertEqual(store.get_image(second).duration_ms, 20_000)
            self.assertEqual(
                [image.id for image in store.list_images(limit=10, sort_key="duration")],
                [second, first, third],
            )
            self.assertEqual(
                [image.id for image in store.list_images(limit=10, sort_key="pixels")],
                [second, third, first],
            )
            self.assertEqual(
                [
                    image.id
                    for image in store.list_images(
                        limit=10,
                        sort_key="file_size",
                        sort_desc=False,
                    )
                ],
                [second, third, first],
            )

    def test_nested_folder_prefix_filters_images_and_counts_subtrees(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            root = Path(tmp) / "library"
            folder_id = store.add_folder(str(root))

            root_image = self._insert_image(store, folder_id, root / "cover.jpg", 1)
            scene_image = self._insert_image(
                store,
                folder_id,
                root / "场景" / "自然场景" / "天空和云.jpg",
                2,
            )
            human_image = self._insert_image(
                store,
                folder_id,
                root / "场景" / "人文场景" / "现代.jpg",
                3,
            )

            tree = store.folder_subtree_counts()
            self.assertEqual(len(tree), 1)
            _folder, counts = tree[0]
            self.assertEqual(counts[str(root)], 3)
            self.assertEqual(counts[str(root / "场景")], 2)
            self.assertEqual(counts[str(root / "场景" / "自然场景")], 1)

            scene_ids = [
                image.id
                for image in store.list_images(
                    folder_path_prefix=str(root / "场景"),
                    limit=10,
                )
            ]
            self.assertCountEqual(scene_ids, [scene_image, human_image])

            nature_ids = [
                image.id
                for image in store.list_images(
                    folder_path_prefix=str(root / "场景" / "自然场景"),
                    limit=10,
                )
            ]
            self.assertEqual(nature_ids, [scene_image])
            self.assertNotIn(root_image, scene_ids)

    def test_batch_favorite_tags_and_remove_from_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            root = Path(tmp) / "library"
            folder_id = store.add_folder(str(root))
            first = self._insert_image(store, folder_id, root / "first.jpg", 1)
            second = self._insert_image(store, folder_id, root / "second.jpg", 2)

            store.update_thumbnail(first, str(Path(tmp) / "thumb_1.webp"), "ready")
            store.update_thumbnail(second, str(Path(tmp) / "thumb_2.webp"), "ready")

            self.assertEqual(store.update_favorites([first, second], True), 2)
            self.assertTrue(store.get_image(first).is_favorite)
            self.assertTrue(store.get_image(second).is_favorite)

            self.assertEqual(
                store.add_tags_to_images([first, second], ["机械", "参考", "机械"]),
                4,
            )
            self.assertEqual(store.get_image_tags(first), ["参考", "机械"])
            self.assertEqual(store.get_image_tags(second), ["参考", "机械"])
            self.assertEqual(
                store.add_tags_to_images([first], ["机械"]),
                0,
            )

            self.assertEqual(store.clear_tags_for_images([first]), 2)
            self.assertEqual(store.get_image_tags(first), [])
            self.assertEqual(store.get_image_tags(second), ["参考", "机械"])
            self.assertEqual([tag.tag_name for tag in store.list_tags()], ["参考", "机械"])

            store.set_image_tags(second, [])
            self.assertEqual(store.get_image_tags(second), [])
            self.assertEqual(store.list_tags(), [])

            thumbnails = store.remove_images_from_library([first])
            self.assertEqual(thumbnails, [str(Path(tmp) / "thumb_1.webp")])
            self.assertIsNone(store.get_image(first))
            self.assertIsNotNone(store.get_image(second))

    def test_batch_tag_counts_and_remove_specific_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            root = Path(tmp) / "library"
            folder_id = store.add_folder(str(root))
            first = self._insert_image(store, folder_id, root / "first.jpg", 1)
            second = self._insert_image(store, folder_id, root / "second.jpg", 2)
            third = self._insert_image(store, folder_id, root / "third.jpg", 3)
            store.set_image_tags(first, ["室内", "夜晚"])
            store.set_image_tags(second, ["室内", "人物"])

            self.assertEqual(
                store.tag_counts_for_images([first, second, third]),
                {"人物": 1, "夜晚": 1, "室内": 2},
            )
            self.assertEqual(store.count_images_with_tags([first, second, third]), 2)

            self.assertEqual(
                store.remove_tags_from_images([first, second, third], ["室内"]),
                2,
            )
            self.assertEqual(store.get_image_tags(first), ["夜晚"])
            self.assertEqual(store.get_image_tags(second), ["人物"])
            self.assertNotIn("室内", [tag.tag_name for tag in store.list_tags()])

    def test_list_images_can_filter_multiple_tags_with_any_or_all_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            root = Path(tmp) / "library"
            folder_id = store.add_folder(str(root))
            first = self._insert_image(store, folder_id, root / "first.jpg", 1)
            second = self._insert_image(store, folder_id, root / "second.jpg", 2)
            third = self._insert_image(store, folder_id, root / "third.jpg", 3)
            store.set_image_tags(first, ["室内", "夜晚"])
            store.set_image_tags(second, ["室内", "人物"])
            store.set_image_tags(third, ["室外"])
            indoor_id = self._tag_id(store, "室内")
            night_id = self._tag_id(store, "夜晚")
            outdoor_id = self._tag_id(store, "室外")

            self.assertEqual(
                [
                    image.id
                    for image in store.list_images(
                        tag_ids=[indoor_id, outdoor_id],
                        tag_match_mode="any",
                        limit=10,
                    )
                ],
                [third, second, first],
            )
            self.assertEqual(
                [
                    image.id
                    for image in store.list_images(
                        tag_ids=[indoor_id, night_id],
                        tag_match_mode="all",
                        limit=10,
                    )
                ],
                [first],
            )

    def test_tag_management_rename_delete_merge_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            root = Path(tmp) / "library"
            folder_id = store.add_folder(str(root))
            first = self._insert_image(store, folder_id, root / "first.jpg", 1)
            second = self._insert_image(store, folder_id, root / "second.jpg", 2)
            third = self._insert_image(store, folder_id, root / "third.jpg", 3)

            store.set_image_tags(first, ["室内", "参考"])
            store.set_image_tags(second, ["室内"])
            store.set_image_tags(third, ["室外"])

            self.assertEqual(
                {
                    tag.tag_name: count
                    for tag, count in store.list_tags_with_counts()
                },
                {"参考": 1, "室内": 2, "室外": 1},
            )

            indoor_id = self._tag_id(store, "室内")
            reference_id = self._tag_id(store, "参考")
            self.assertTrue(store.rename_tag(indoor_id, "室内场景"))
            self.assertEqual(store.get_image_tags(first), ["参考", "室内场景"])
            with self.assertRaises(ValueError):
                store.rename_tag(indoor_id, "参考")

            outdoor_id = self._tag_id(store, "室外")
            self.assertEqual(store.merge_tag(outdoor_id, indoor_id), 1)
            self.assertEqual(store.get_image_tags(third), ["室内场景"])
            self.assertNotIn("室外", [tag.tag_name for tag in store.list_tags()])
            self.assertEqual(
                {
                    tag.tag_name: count
                    for tag, count in store.list_tags_with_counts()
                },
                {"参考": 1, "室内场景": 3},
            )

            self.assertEqual(store.delete_tag(reference_id), 1)
            self.assertEqual(store.get_image_tags(first), ["室内场景"])
            self.assertNotIn("参考", [tag.tag_name for tag in store.list_tags()])

    def test_remove_folder_index_removes_root_or_subtree_without_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            root = Path(tmp) / "library"
            folder_id = store.add_folder(str(root))
            root_image = self._insert_image(store, folder_id, root / "cover.jpg", 1)
            subtree_image = self._insert_image(
                store,
                folder_id,
                root / "场景" / "自然" / "sky.jpg",
                2,
            )
            sibling_image = self._insert_image(
                store,
                folder_id,
                root / "角色" / "person.jpg",
                3,
            )
            store.update_thumbnail(root_image, str(Path(tmp) / "root.webp"), "ready")
            store.update_thumbnail(subtree_image, str(Path(tmp) / "subtree.webp"), "ready")
            store.update_thumbnail(sibling_image, str(Path(tmp) / "sibling.webp"), "ready")
            store.set_image_tags(subtree_image, ["只在子树"])
            store.set_image_tags(sibling_image, ["保留"])

            thumbnails, removed = store.remove_images_by_folder_path_prefix(root / "场景")
            self.assertEqual(removed, 1)
            self.assertEqual(thumbnails, [str(Path(tmp) / "subtree.webp")])
            self.assertIsNone(store.get_image(subtree_image))
            self.assertIsNotNone(store.get_image(root_image))
            self.assertIsNotNone(store.get_image(sibling_image))
            self.assertEqual([tag.tag_name for tag in store.list_tags()], ["保留"])

            thumbnails, removed = store.remove_folder_from_library(folder_id)
            self.assertEqual(removed, 2)
            self.assertCountEqual(
                thumbnails,
                [str(Path(tmp) / "root.webp"), str(Path(tmp) / "sibling.webp")],
            )
            self.assertEqual(store.count_images(), 0)
            self.assertEqual(store.list_folders(include_inactive=True), [])
            self.assertEqual(store.list_tags(), [])

    def test_restore_snapshot_recreates_folder_after_last_image_removal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            root = Path(tmp) / "library"
            folder_id = store.add_folder(str(root))
            image_id = self._insert_image(store, folder_id, root / "first.jpg", 1)
            snapshot = store.snapshot_images_for_restore([image_id])

            store.remove_images_from_library([image_id])
            self.assertEqual(store.count_images(), 0)
            self.assertEqual(store.list_folders(include_inactive=True), [])

            restored = store.restore_images_snapshot(snapshot)

            self.assertEqual(restored, 1)
            self.assertIsNotNone(store.get_folder(folder_id))
            self.assertIsNotNone(store.get_image(image_id))

    def test_collections_support_tree_moves_assignment_and_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            root = Path(tmp) / "library"
            folder_id = store.add_folder(str(root))
            first = self._insert_image(store, folder_id, root / "first.jpg", 1)
            second = self._insert_image(store, folder_id, root / "second.jpg", 2)
            third = self._insert_image(store, folder_id, root / "third.jpg", 3)

            references = store.create_collection("参考")
            indoor = store.create_collection("室内", parent_id=references)
            outdoor = store.create_collection("室外", parent_id=references)

            self.assertEqual(store.assign_images_to_collection([first, second], indoor), 2)
            self.assertEqual(store.assign_images_to_collection([third], outdoor), 1)
            self.assertEqual(store.assign_images_to_collection([first], indoor), 0)
            self.assertEqual(store.collection_paths_for_image(first), ["参考 / 室内"])
            self.assertEqual(store.collection_paths_for_image(third), ["参考 / 室外"])
            self.assertEqual(
                store.collection_image_counts(),
                {references: 3, indoor: 2, outdoor: 1},
            )

            reference_ids = [
                image.id for image in store.list_images(collection_id=references, limit=10)
            ]
            self.assertCountEqual(reference_ids, [first, second, third])
            indoor_ids = [
                image.id for image in store.list_images(collection_id=indoor, limit=10)
            ]
            self.assertCountEqual(indoor_ids, [first, second])

            store.update_collection_tree(
                [
                    (references, None, 0),
                    (indoor, None, 1),
                    (outdoor, indoor, 0),
                ]
            )
            moved = {collection.id: collection for collection in store.list_collections()}
            self.assertIsNone(moved[indoor].parent_id)
            self.assertEqual(moved[outdoor].parent_id, indoor)
            self.assertEqual(store.collection_descendant_ids(indoor), [indoor, outdoor])

            with self.assertRaises(ValueError):
                store.create_collection("室外", parent_id=indoor)

    def test_update_image_path_after_rename_refreshes_file_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            root = Path(tmp) / "library"
            folder_id = store.add_folder(str(root))
            image_id = self._insert_image(store, folder_id, root / "old-name.jpg", 1)

            new_path = root / "new-name.jpg"
            store.update_image_path_after_rename(
                image_id,
                file_path=str(new_path),
                file_size=1234,
                modified_time_ns=99,
            )

            image = store.get_image(image_id)
            self.assertIsNotNone(image)
            assert image is not None
            self.assertEqual(image.file_path, str(new_path))
            self.assertEqual(image.file_name, "new-name.jpg")
            self.assertEqual(image.file_ext, ".jpg")
            self.assertEqual(image.file_size, 1234)
            self.assertEqual(image.modified_time_ns, 99)

    def test_delete_collection_removes_exclusive_image_indexes_but_keeps_shared_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            root = Path(tmp) / "library"
            folder_id = store.add_folder(str(root))
            first = self._insert_image(store, folder_id, root / "first.jpg", 1)
            second = self._insert_image(store, folder_id, root / "second.jpg", 2)
            third = self._insert_image(store, folder_id, root / "third.jpg", 3)
            store.update_thumbnail(first, str(Path(tmp) / "first.webp"), "ready")
            store.update_thumbnail(second, str(Path(tmp) / "second.webp"), "ready")

            folder = store.create_collection("要删")
            child = store.create_collection("子文件夹", parent_id=folder)
            other = store.create_collection("保留")
            store.assign_images_to_collection([first], folder)
            store.assign_images_to_collection([second], child)
            store.assign_images_to_collection([second, third], other)

            affected, deleted, thumbnails = store.delete_collection(folder)
            self.assertEqual(affected, 2)
            self.assertEqual(deleted, 1)
            self.assertEqual(thumbnails, [str(Path(tmp) / "first.webp")])
            self.assertIsNone(store.get_image(first))
            self.assertIsNotNone(store.get_image(second))
            self.assertIsNotNone(store.get_image(third))
            self.assertEqual(store.image_ids_for_collection(other), {second, third})
            self.assertNotIn(folder, [collection.id for collection in store.list_collections()])
            self.assertNotIn(child, [collection.id for collection in store.list_collections()])

    def test_remove_images_from_collection_subtree_deletes_orphans_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            root = Path(tmp) / "library"
            folder_id = store.add_folder(str(root))
            folder = store.create_collection("folder")
            child = store.create_collection("child", folder)
            other = store.create_collection("other")
            first = self._insert_image(store, folder_id, root / "first.jpg", 1)
            second = self._insert_image(store, folder_id, root / "second.jpg", 2)
            third = self._insert_image(store, folder_id, root / "third.jpg", 3)
            store.update_thumbnail(first, str(Path(tmp) / "first.webp"), "ready")
            store.update_thumbnail(second, str(Path(tmp) / "second.webp"), "ready")
            store.update_thumbnail(third, str(Path(tmp) / "third.webp"), "ready")
            store.assign_images_to_collection([first], folder)
            store.assign_images_to_collection([second, third], child)
            store.assign_images_to_collection([third], other)

            removed_links, deleted_images, thumbnails = store.remove_images_from_collection_subtree(
                [first, second, third],
                folder,
            )

            self.assertEqual(removed_links, 3)
            self.assertEqual(deleted_images, 2)
            self.assertCountEqual(
                thumbnails,
                [str(Path(tmp) / "first.webp"), str(Path(tmp) / "second.webp")],
            )
            self.assertIsNone(store.get_image(first))
            self.assertIsNone(store.get_image(second))
            self.assertIsNotNone(store.get_image(third))
            self.assertEqual(store.image_ids_for_collection(folder), set())
            self.assertEqual(store.image_ids_for_collection(other), {third})

    def test_move_images_to_collection_removes_source_links_and_preserves_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            root = Path(tmp) / "library"
            folder_id = store.add_folder(str(root))
            source = store.create_collection("source")
            child = store.create_collection("child", source)
            target = store.create_collection("target")
            first = self._insert_image(store, folder_id, root / "first.jpg", 1)
            second = self._insert_image(store, folder_id, root / "second.jpg", 2)
            third = self._insert_image(store, folder_id, root / "third.jpg", 3)
            store.assign_images_to_collection([first, third], source)
            store.assign_images_to_collection([second], child)

            inserted, removed_links, deleted_images, thumbnails = store.move_images_to_collection(
                [first, second, third],
                source_collection_id=source,
                target_collection_id=target,
            )

            self.assertEqual(inserted, 3)
            self.assertEqual(removed_links, 3)
            self.assertEqual(deleted_images, 0)
            self.assertEqual(thumbnails, [])
            self.assertEqual(store.image_ids_for_collection(source), set())
            self.assertEqual(store.image_ids_for_collection(target), {first, second, third})
            self.assertIsNotNone(store.get_image(first))
            self.assertIsNotNone(store.get_image(second))
            self.assertIsNotNone(store.get_image(third))

    def test_ensure_collection_path_and_folder_prefix_image_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            root = Path(tmp) / "library"
            folder_id = store.add_folder(str(root))
            root_image = self._insert_image(store, folder_id, root / "cover.jpg", 1)
            sky = self._insert_image(store, folder_id, root / "场景" / "自然" / "sky.jpg", 2)
            person = self._insert_image(store, folder_id, root / "角色" / "person.jpg", 3)

            self.assertEqual(
                [
                    image.id
                    for image in store.list_images_for_folder_path_prefix(root / "场景")
                ],
                [sky],
            )

            leaf = store.ensure_collection_path(["library", "场景", "自然"])
            self.assertIsNotNone(leaf)
            self.assertEqual(store.ensure_collection_path(["library", "场景", "自然"]), leaf)
            self.assertEqual(store.assign_images_to_collection([sky], leaf), 1)
            self.assertEqual(
                [
                    image.id
                    for image in store.list_images(collection_id=leaf, limit=10)
                ],
                [sky],
            )
            self.assertNotIn(root_image, store.image_ids_for_collection(leaf))
            self.assertNotIn(person, store.image_ids_for_collection(leaf))

    def test_ai_vision_default_rules_jobs_stats_filtering_and_stale_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            store = MetadataStore(db_path)
            store.initialize()
            root = Path(tmp) / "library"
            folder_id = store.add_folder(str(root))
            black = store.create_collection("黑白摄影")
            references = store.create_collection("创作参考")
            global_ref = store.create_collection("全局参考", references)
            still_life = store.create_collection("具体造型")

            first = self._insert_image(store, folder_id, root / "first.jpg", 1)
            second = self._insert_image(store, folder_id, root / "second.jpg", 2)
            excluded = self._insert_image(store, folder_id, root / "excluded.jpg", 3)
            store.assign_images_to_collection([first], black)
            store.assign_images_to_collection([second], global_ref)
            store.assign_images_to_collection([excluded], still_life)

            self.assertEqual(store.seed_default_ai_vision_collection_rules(), 2)
            stats = store.ai_vision_stats(
                provider_name="LM Studio",
                model_name="vision-model",
                prompt_version=AI_VISION_PROMPT_VERSION,
            )
            self.assertEqual(stats["total"], 2)
            self.assertEqual(stats["pending"], 2)
            self.assertEqual(
                [image.id for image in store.next_ai_vision_jobs(
                    provider_name="LM Studio",
                    model_name="vision-model",
                    prompt_version=AI_VISION_PROMPT_VERSION,
                    limit=10,
                )],
                [first, second],
            )
            self.assertTrue(store.mark_image_missing(second))
            stats = store.ai_vision_stats(
                provider_name="LM Studio",
                model_name="vision-model",
                prompt_version=AI_VISION_PROMPT_VERSION,
            )
            self.assertEqual(stats["total"], 1)
            self.assertEqual(
                [image.id for image in store.next_ai_vision_jobs(
                    provider_name="LM Studio",
                    model_name="vision-model",
                    prompt_version=AI_VISION_PROMPT_VERSION,
                    limit=10,
                )],
                [first],
            )
            store.upsert_image(
                folder_id=folder_id,
                file_path=str(root / "second.jpg"),
                file_size=2,
                width=10,
                height=20,
                created_time_ns=None,
                modified_time_ns=2,
            )

            store.upsert_ai_vision_success(
                image_id=first,
                provider_name="LM Studio",
                model_name="vision-model",
                prompt_version=AI_VISION_PROMPT_VERSION,
                analysis=self._ai_vision_analysis(),
                source_modified_time_ns=1,
            )
            self.assertEqual(
                store.image_ids_matching_ai_vision("scene_location", "outdoor"),
                {first},
            )
            self.assertEqual(
                store.image_ids_matching_ai_vision("lighting", "back_light"),
                {first},
            )
            stats = store.ai_vision_stats(
                provider_name="LM Studio",
                model_name="vision-model",
                prompt_version=AI_VISION_PROMPT_VERSION,
            )
            self.assertEqual(stats["ready"], 1)
            self.assertEqual(stats["pending"], 1)

            store.mark_ai_vision_failed(
                image_id=second,
                provider_name="LM Studio",
                model_name="vision-model",
                prompt_version=AI_VISION_PROMPT_VERSION,
                error_message="model failed",
            )
            self.assertEqual(store.retry_failed_ai_vision(), 1)
            self.assertEqual(store.ai_vision_tags_for_image(second)["status"], "pending")

            store.upsert_image(
                folder_id=folder_id,
                file_path=str(root / "first.jpg"),
                file_size=999,
                width=10,
                height=20,
                created_time_ns=None,
                modified_time_ns=99,
            )
            self.assertEqual(store.ai_vision_tags_for_image(first)["status"], "stale")

            store.set_ai_vision_collection_rule(global_ref, mode="exclude")
            stats = store.ai_vision_stats(
                provider_name="LM Studio",
                model_name="vision-model",
                prompt_version=AI_VISION_PROMPT_VERSION,
            )
            self.assertEqual(stats["total"], 1)

    def test_virtual_image_filters_find_untagged_un_ai_tagged_and_uncategorized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eidory.sqlite3"
            root = Path(tmp) / "library"
            store = MetadataStore(db_path)
            store.initialize()
            folder_id = store.add_folder(str(root))
            collection_id = store.create_collection("参考")

            uncategorized = self._insert_image(store, folder_id, root / "uncategorized.jpg", 1)
            tagged_ready = self._insert_image(store, folder_id, root / "tagged-ready.jpg", 2)
            tagged_no_ai = self._insert_image(store, folder_id, root / "tagged-no-ai.jpg", 3)
            untagged_no_ai = self._insert_image(store, folder_id, root / "untagged-no-ai.jpg", 4)
            video_id = self._insert_image(store, folder_id, root / "clip.mp4", 5)

            store.add_tags_to_images([tagged_ready, tagged_no_ai], ["已标签"])
            store.assign_images_to_collection(
                [tagged_ready, tagged_no_ai, untagged_no_ai],
                collection_id,
            )
            store.upsert_ai_vision_success(
                image_id=tagged_ready,
                provider_name="LM Studio",
                model_name="vision-model",
                prompt_version=AI_VISION_PROMPT_VERSION,
                analysis=self._ai_vision_analysis(),
                source_modified_time_ns=2,
            )

            self.assertEqual(
                {image.id for image in store.list_images(virtual_filter="untagged", limit=50)},
                {uncategorized, untagged_no_ai, video_id},
            )
            self.assertEqual(
                {image.id for image in store.list_images(virtual_filter="un_ai_tagged", limit=50)},
                {uncategorized, tagged_no_ai, untagged_no_ai},
            )
            self.assertEqual(
                {image.id for image in store.list_images(virtual_filter="uncategorized", limit=50)},
                {uncategorized, video_id},
            )

            self.assertEqual(store.count_images_for_virtual_filter("untagged"), 3)
            self.assertEqual(store.count_images_for_virtual_filter("un_ai_tagged"), 3)
            self.assertEqual(store.count_images_for_virtual_filter("uncategorized"), 2)

    @staticmethod
    def _ai_vision_analysis() -> AIVisionAnalysis:
        return AIVisionAnalysis(
            scene_location="outdoor",
            environment_type="natural",
            time_of_day="day",
            weather="sunny",
            shot_scale="long",
            view_angle="eye_level",
            lighting=["back_light", "diffuse"],
            confidence={
                "scene_location": 0.9,
                "environment_type": 0.8,
                "time_of_day": 0.8,
                "weather": 0.7,
                "shot_scale": 0.8,
                "view_angle": 0.8,
                "lighting:back_light": 0.7,
                "lighting:diffuse": 0.6,
            },
            notes="test",
            raw_json={},
        )

    @staticmethod
    def _insert_image(
        store: MetadataStore,
        folder_id: int,
        path: Path,
        modified_time_ns: int,
    ) -> int:
        image_id, _state = store.upsert_image(
            folder_id=folder_id,
            file_path=str(path),
            file_size=123,
            width=10,
            height=20,
            created_time_ns=None,
            modified_time_ns=modified_time_ns,
        )
        return image_id

    @staticmethod
    def _tag_id(store: MetadataStore, tag_name: str) -> int:
        for tag in store.list_tags():
            if tag.tag_name == tag_name:
                return tag.id
        raise AssertionError(f"tag not found: {tag_name}")


if __name__ == "__main__":
    unittest.main()
