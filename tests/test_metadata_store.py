from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eidory.core.metadata_store import MetadataStore, TEMPORARY_PROJECT_COLORS


class MetadataStoreTest(unittest.TestCase):
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
