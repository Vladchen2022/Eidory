from __future__ import annotations

import os
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from eidory.core.color_features import COLOR_VECTOR_DIM, COLOR_VECTOR_VERSION
from eidory.core.inspiration import InspirationTerm
from eidory.core.media_types import SUPPORTED_IMAGE_EXTENSIONS
from eidory.core.time_utils import timestamp_ns_to_iso, utc_now_iso
from eidory.models import (
    CollectionItem,
    FolderItem,
    ImageItem,
    SavedViewItem,
    TagItem,
    TemporaryProjectItem,
)


def _clean_optional_text(value: object, *, max_length: int) -> str | None:
    if not isinstance(value, str):
        return None
    clean = " ".join(value.strip().split())[:max_length]
    return clean or None


class MetadataStore:
    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
        with self.connect() as conn:
            conn.executescript(schema_path.read_text(encoding="utf-8"))
            self._ensure_schema_migrations(conn)

    @staticmethod
    def _ensure_schema_migrations(conn: sqlite3.Connection) -> None:
        image_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(images)").fetchall()
        }
        if "duration_ms" not in image_columns:
            conn.execute("ALTER TABLE images ADD COLUMN duration_ms INTEGER")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_views (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_saved_views_updated_at
            ON saved_views(updated_at)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS temporary_projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                summary TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        temporary_project_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(temporary_projects)").fetchall()
        }
        if "summary" not in temporary_project_columns:
            conn.execute("ALTER TABLE temporary_projects ADD COLUMN summary TEXT NOT NULL DEFAULT ''")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_temporary_projects_updated_at
            ON temporary_projects(updated_at)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS temporary_project_images (
                project_id INTEGER NOT NULL,
                image_id INTEGER NOT NULL,
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                intent_label TEXT,
                intent_query TEXT,
                PRIMARY KEY(project_id, image_id),
                FOREIGN KEY(project_id) REFERENCES temporary_projects(id) ON DELETE CASCADE,
                FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
            )
            """
        )
        temporary_image_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(temporary_project_images)").fetchall()
        }
        if "intent_label" not in temporary_image_columns:
            conn.execute("ALTER TABLE temporary_project_images ADD COLUMN intent_label TEXT")
        if "intent_query" not in temporary_image_columns:
            conn.execute("ALTER TABLE temporary_project_images ADD COLUMN intent_query TEXT")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_temporary_project_images_project_order
            ON temporary_project_images(project_id, sort_order)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inspiration_projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                brief TEXT NOT NULL,
                answers TEXT,
                questions_json TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_inspiration_projects_updated_at
            ON inspiration_projects(updated_at)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inspiration_terms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                query TEXT NOT NULL,
                axis TEXT NOT NULL,
                reason TEXT,
                selected INTEGER NOT NULL DEFAULT 0,
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES inspiration_projects(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_inspiration_terms_project_order
            ON inspiration_terms(project_id, sort_order)
            """
        )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def add_folder(self, folder_path: str) -> int:
        normalized = os.path.abspath(os.path.expanduser(folder_path))
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO folders(folder_path, import_mode, added_at, is_active)
                VALUES (?, 'indexed', ?, 1)
                ON CONFLICT(folder_path) DO UPDATE SET is_active = 1
                """,
                (normalized, now),
            )
            row = conn.execute(
                "SELECT id FROM folders WHERE folder_path = ?", (normalized,)
            ).fetchone()
            return int(row["id"])

    def list_folders(self, include_inactive: bool = False) -> list[FolderItem]:
        sql = "SELECT * FROM folders"
        params: tuple[object, ...] = ()
        if not include_inactive:
            sql += " WHERE is_active = 1"
        sql += " ORDER BY folder_path"
        with self.connect() as conn:
            return [self._folder_from_row(row) for row in conn.execute(sql, params)]

    def folder_subtree_counts(self) -> list[tuple[FolderItem, dict[str, int]]]:
        with self.connect() as conn:
            folder_rows = conn.execute(
                "SELECT * FROM folders WHERE is_active = 1 ORDER BY folder_path"
            ).fetchall()
            image_rows = conn.execute(
                """
                SELECT i.folder_id, i.file_path
                FROM images i
                JOIN folders f ON f.id = i.folder_id
                WHERE f.is_active = 1
                  AND i.is_missing = 0
                """
            ).fetchall()

        folders = [self._folder_from_row(row) for row in folder_rows]
        roots = {folder.id: self._normalize_folder_prefix(folder.folder_path) for folder in folders}
        counts: dict[int, dict[str, int]] = {
            folder.id: {roots[folder.id]: 0} for folder in folders
        }

        for row in image_rows:
            folder_id = int(row["folder_id"])
            root = roots.get(folder_id)
            if root is None:
                continue
            image_dir = self._normalize_folder_prefix(os.path.dirname(str(row["file_path"])))
            if not self._path_is_under_prefix(image_dir, root, include_self=True):
                continue

            current = image_dir
            while True:
                counts[folder_id][current] = counts[folder_id].get(current, 0) + 1
                if current == root:
                    break
                parent = self._normalize_folder_prefix(os.path.dirname(current))
                if parent == current or not self._path_is_under_prefix(parent, root, include_self=True):
                    break
                current = parent

        return [(folder, counts[folder.id]) for folder in folders]

    def get_folder(self, folder_id: int) -> FolderItem | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM folders WHERE id = ?", (folder_id,)).fetchone()
            return self._folder_from_row(row) if row else None

    def list_collections(self) -> list[CollectionItem]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM collections
                ORDER BY parent_id IS NOT NULL, parent_id, sort_order, name COLLATE NOCASE
                """
            ).fetchall()
        return [self._collection_from_row(row) for row in rows]

    def list_collections_with_counts(self) -> list[tuple[CollectionItem, int]]:
        collections = self.list_collections()
        counts = self.collection_image_counts()
        return [(collection, counts.get(collection.id, 0)) for collection in collections]

    def collection_image_counts(self) -> dict[int, int]:
        collections = self.list_collections()
        parent_by_id = {collection.id: collection.parent_id for collection in collections}
        image_ids_by_collection = {collection.id: set() for collection in collections}
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT ic.collection_id, ic.image_id
                FROM image_collections ic
                JOIN images i ON i.id = ic.image_id
                WHERE i.is_missing = 0
                """
            ).fetchall()

        for row in rows:
            collection_id = int(row["collection_id"])
            image_id = int(row["image_id"])
            current: int | None = collection_id
            seen: set[int] = set()
            while current is not None and current in image_ids_by_collection and current not in seen:
                seen.add(current)
                image_ids_by_collection[current].add(image_id)
                current = parent_by_id.get(current)
        return {
            collection_id: len(image_ids)
            for collection_id, image_ids in image_ids_by_collection.items()
        }

    def collection_descendant_ids(self, collection_id: int) -> list[int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                WITH RECURSIVE subtree(id) AS (
                    SELECT id FROM collections WHERE id = ?
                    UNION ALL
                    SELECT c.id
                    FROM collections c
                    JOIN subtree s ON c.parent_id = s.id
                )
                SELECT id FROM subtree
                """,
                (collection_id,),
            ).fetchall()
        return [int(row["id"]) for row in rows]

    def image_ids_for_collection(self, collection_id: int) -> set[int]:
        collection_ids = self.collection_descendant_ids(collection_id)
        if not collection_ids:
            return set()
        placeholders = ",".join("?" for _ in collection_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT ic.image_id
                FROM image_collections ic
                JOIN images i ON i.id = ic.image_id
                WHERE ic.collection_id IN ({placeholders})
                  AND i.is_missing = 0
                """,
                tuple(collection_ids),
            ).fetchall()
        return {int(row["image_id"]) for row in rows}

    def create_collection(self, name: str, parent_id: int | None = None) -> int:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("collection name must not be empty")
        now = utc_now_iso()
        with self.connect() as conn:
            if parent_id is not None:
                parent = conn.execute(
                    "SELECT id FROM collections WHERE id = ?", (parent_id,)
                ).fetchone()
                if parent is None:
                    raise ValueError("parent collection not found")
            order_row = conn.execute(
                """
                SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_order
                FROM collections
                WHERE (parent_id IS NULL AND ? IS NULL) OR parent_id = ?
                """,
                (parent_id, parent_id),
            ).fetchone()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO collections(parent_id, name, sort_order, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (parent_id, clean_name, int(order_row["next_order"]), now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("collection name already exists under the same parent") from exc
            return int(cur.lastrowid)

    def ensure_collection_path(
        self,
        names: Sequence[str],
        parent_id: int | None = None,
    ) -> int | None:
        current_parent = parent_id
        last_id: int | None = None
        for name in names:
            clean_name = name.strip()
            if not clean_name:
                continue
            with self.connect() as conn:
                row = conn.execute(
                    """
                    SELECT id
                    FROM collections
                    WHERE name = ?
                      AND ((parent_id IS NULL AND ? IS NULL) OR parent_id = ?)
                    """,
                    (clean_name, current_parent, current_parent),
                ).fetchone()
            if row is None:
                last_id = self.create_collection(clean_name, current_parent)
            else:
                last_id = int(row["id"])
            current_parent = last_id
        return last_id

    def rename_collection(self, collection_id: int, new_name: str) -> bool:
        clean_name = new_name.strip()
        if not clean_name:
            raise ValueError("collection name must not be empty")
        now = utc_now_iso()
        with self.connect() as conn:
            try:
                cur = conn.execute(
                    "UPDATE collections SET name = ?, updated_at = ? WHERE id = ?",
                    (clean_name, now, collection_id),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("collection name already exists under the same parent") from exc
            return int(cur.rowcount) > 0

    def delete_collection(self, collection_id: int) -> tuple[int, int, list[str]]:
        with self.connect() as conn:
            image_rows = conn.execute(
                """
                WITH RECURSIVE subtree(id) AS (
                    SELECT id FROM collections WHERE id = ?
                    UNION ALL
                    SELECT c.id
                    FROM collections c
                    JOIN subtree s ON c.parent_id = s.id
                )
                SELECT DISTINCT ic.image_id
                FROM image_collections ic
                WHERE ic.collection_id IN (SELECT id FROM subtree)
                """,
                (collection_id,),
            ).fetchall()
            image_ids = [int(row["image_id"]) for row in image_rows]
            image_count = len(image_ids)

            thumbnail_paths: list[str] = []
            deleted_images = 0
            if image_ids:
                placeholders = ",".join("?" for _ in image_ids)
                keep_rows = conn.execute(
                    f"""
                    WITH RECURSIVE subtree(id) AS (
                        SELECT id FROM collections WHERE id = ?
                        UNION ALL
                        SELECT c.id
                        FROM collections c
                        JOIN subtree s ON c.parent_id = s.id
                    )
                    SELECT DISTINCT image_id
                    FROM image_collections
                    WHERE image_id IN ({placeholders})
                      AND collection_id NOT IN (SELECT id FROM subtree)
                    """,
                    (collection_id, *image_ids),
                ).fetchall()
                keep_ids = {int(row["image_id"]) for row in keep_rows}
                delete_ids = [image_id for image_id in image_ids if image_id not in keep_ids]
                if delete_ids:
                    delete_placeholders = ",".join("?" for _ in delete_ids)
                    thumb_rows = conn.execute(
                        f"""
                        SELECT thumbnail_path
                        FROM images
                        WHERE id IN ({delete_placeholders})
                          AND thumbnail_path IS NOT NULL
                        """,
                        tuple(delete_ids),
                    ).fetchall()
                    thumbnail_paths = [str(row["thumbnail_path"]) for row in thumb_rows]
                    cur = conn.execute(
                        f"DELETE FROM images WHERE id IN ({delete_placeholders})",
                        tuple(delete_ids),
                    )
                    deleted_images = int(cur.rowcount)

            conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
            self._delete_unused_tags(conn)
            return image_count, deleted_images, thumbnail_paths

    def update_collection_tree(
        self,
        updates: Sequence[tuple[int, int | None, int]],
    ) -> None:
        parent_by_id = {collection_id: parent_id for collection_id, parent_id, _order in updates}
        for collection_id in parent_by_id:
            seen: set[int] = set()
            current: int | None = collection_id
            while current is not None:
                if current in seen:
                    raise ValueError("collection tree contains a cycle")
                seen.add(current)
                current = parent_by_id.get(current)

        now = utc_now_iso()
        with self.connect() as conn:
            try:
                conn.executemany(
                    """
                    UPDATE collections
                    SET parent_id = ?, sort_order = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        (parent_id, sort_order, now, collection_id)
                        for collection_id, parent_id, sort_order in updates
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("collection name already exists under the same parent") from exc

    def assign_images_to_collection(
        self,
        image_ids: Sequence[int],
        collection_id: int,
    ) -> int:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return 0
        now = utc_now_iso()
        inserted = 0
        with self.connect() as conn:
            collection = conn.execute(
                "SELECT id FROM collections WHERE id = ?", (collection_id,)
            ).fetchone()
            if collection is None:
                raise ValueError("collection not found")
            for image_id in clean_ids:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO image_collections(image_id, collection_id, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (image_id, collection_id, now),
                )
                inserted += int(cur.rowcount)
        return inserted

    def remove_images_from_collection_subtree(
        self,
        image_ids: Sequence[int],
        collection_id: int,
        *,
        preserve_collection_ids: Sequence[int] = (),
    ) -> tuple[int, int, list[str]]:
        clean_ids = sorted({int(image_id) for image_id in image_ids})
        if not clean_ids:
            return 0, 0, []
        with self.connect() as conn:
            subtree_ids = self._collection_subtree_ids(conn, collection_id)
            remove_collection_ids = sorted(set(subtree_ids) - {int(value) for value in preserve_collection_ids})
            if not remove_collection_ids:
                return 0, 0, []

            image_placeholders = ",".join("?" for _ in clean_ids)
            collection_placeholders = ",".join("?" for _ in remove_collection_ids)
            cur = conn.execute(
                f"""
                DELETE FROM image_collections
                WHERE image_id IN ({image_placeholders})
                  AND collection_id IN ({collection_placeholders})
                """,
                (*clean_ids, *remove_collection_ids),
            )
            removed_links = int(cur.rowcount)
            deleted_images, thumbnail_paths = self._delete_orphan_images(conn, clean_ids)
            self._delete_unused_tags(conn)
            return removed_links, deleted_images, thumbnail_paths

    def move_images_to_collection(
        self,
        image_ids: Sequence[int],
        *,
        source_collection_id: int,
        target_collection_id: int,
    ) -> tuple[int, int, int, list[str]]:
        clean_ids = sorted({int(image_id) for image_id in image_ids})
        if not clean_ids:
            return 0, 0, 0, []
        now = utc_now_iso()
        with self.connect() as conn:
            target = conn.execute(
                "SELECT id FROM collections WHERE id = ?", (target_collection_id,)
            ).fetchone()
            source = conn.execute(
                "SELECT id FROM collections WHERE id = ?", (source_collection_id,)
            ).fetchone()
            if target is None or source is None:
                raise ValueError("collection not found")

            inserted = 0
            for image_id in clean_ids:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO image_collections(image_id, collection_id, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (image_id, target_collection_id, now),
                )
                inserted += int(cur.rowcount)

            subtree_ids = self._collection_subtree_ids(conn, source_collection_id)
            remove_collection_ids = sorted(set(subtree_ids) - {target_collection_id})
            removed_links = 0
            if remove_collection_ids:
                image_placeholders = ",".join("?" for _ in clean_ids)
                collection_placeholders = ",".join("?" for _ in remove_collection_ids)
                cur = conn.execute(
                    f"""
                    DELETE FROM image_collections
                    WHERE image_id IN ({image_placeholders})
                      AND collection_id IN ({collection_placeholders})
                    """,
                    (*clean_ids, *remove_collection_ids),
                )
                removed_links = int(cur.rowcount)

            deleted_images, thumbnail_paths = self._delete_orphan_images(conn, clean_ids)
            self._delete_unused_tags(conn)
            return inserted, removed_links, deleted_images, thumbnail_paths

    @staticmethod
    def _collection_subtree_ids(conn: sqlite3.Connection, collection_id: int) -> list[int]:
        rows = conn.execute(
            """
            WITH RECURSIVE subtree(id) AS (
                SELECT id FROM collections WHERE id = ?
                UNION ALL
                SELECT c.id
                FROM collections c
                JOIN subtree s ON c.parent_id = s.id
            )
            SELECT id FROM subtree
            """,
            (collection_id,),
        ).fetchall()
        return [int(row["id"]) for row in rows]

    @staticmethod
    def _delete_orphan_images(
        conn: sqlite3.Connection,
        image_ids: Sequence[int],
    ) -> tuple[int, list[str]]:
        clean_ids = sorted({int(image_id) for image_id in image_ids})
        if not clean_ids:
            return 0, []
        image_placeholders = ",".join("?" for _ in clean_ids)
        orphan_rows = conn.execute(
            f"""
            SELECT i.id, i.thumbnail_path
            FROM images i
            WHERE i.id IN ({image_placeholders})
              AND NOT EXISTS (
                  SELECT 1
                  FROM image_collections ic
                  WHERE ic.image_id = i.id
              )
            """,
            tuple(clean_ids),
        ).fetchall()
        orphan_ids = [int(row["id"]) for row in orphan_rows]
        thumbnail_paths = [
            str(row["thumbnail_path"])
            for row in orphan_rows
            if row["thumbnail_path"] is not None
        ]
        if not orphan_ids:
            return 0, []
        orphan_placeholders = ",".join("?" for _ in orphan_ids)
        cur = conn.execute(
            f"DELETE FROM images WHERE id IN ({orphan_placeholders})",
            tuple(orphan_ids),
        )
        return int(cur.rowcount), thumbnail_paths

    def list_images_for_folder_path_prefix(self, folder_path_prefix: str) -> list[ImageItem]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM images
                WHERE file_path LIKE ? ESCAPE '\\'
                  AND is_missing = 0
                ORDER BY file_path COLLATE NOCASE
                """,
                (self._folder_path_like(folder_path_prefix),),
            ).fetchall()
        return [self._image_from_row(row) for row in rows]

    def upsert_image(
        self,
        *,
        folder_id: int,
        file_path: str,
        file_size: int,
        width: int | None,
        height: int | None,
        created_time_ns: int | None,
        modified_time_ns: int,
        duration_ms: int | None = None,
    ) -> tuple[int, str]:
        now = utc_now_iso()
        file_name = os.path.basename(file_path)
        file_ext = Path(file_name).suffix.lower()
        created_at = timestamp_ns_to_iso(created_time_ns) if created_time_ns else None
        modified_at = timestamp_ns_to_iso(modified_time_ns)

        with self.connect() as conn:
            existing = conn.execute(
                "SELECT * FROM images WHERE file_path = ?", (file_path,)
            ).fetchone()
            if existing is None:
                cur = conn.execute(
                    """
                    INSERT INTO images(
                        folder_id, file_path, file_name, file_ext, file_size, width, height,
                        duration_ms,
                        created_at, modified_at, modified_time_ns, imported_at, last_seen_at,
                        thumbnail_status, embedding_status, is_missing
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 'pending', 0)
                    """,
                    (
                        folder_id,
                        file_path,
                        file_name,
                        file_ext,
                        file_size,
                        width,
                        height,
                        duration_ms,
                        created_at,
                        modified_at,
                        modified_time_ns,
                        now,
                        now,
                    ),
                )
                return int(cur.lastrowid), "new"

            changed = (
                int(existing["file_size"]) != file_size
                or int(existing["modified_time_ns"]) != modified_time_ns
                or bool(existing["is_missing"])
            )
            conn.execute(
                """
                UPDATE images
                SET folder_id = ?, file_name = ?, file_ext = ?, file_size = ?, width = ?,
                    height = ?, duration_ms = ?, created_at = ?, modified_at = ?, modified_time_ns = ?,
                    last_seen_at = ?, is_missing = 0
                WHERE id = ?
                """,
                (
                    folder_id,
                    file_name,
                    file_ext,
                    file_size,
                    width,
                    height,
                    duration_ms,
                    created_at,
                    modified_at,
                    modified_time_ns,
                    now,
                    existing["id"],
                ),
            )
            if changed:
                conn.execute(
                    """
                    UPDATE images
                    SET thumbnail_status = 'pending',
                        thumbnail_path = NULL,
                        embedding_status = 'pending'
                    WHERE id = ?
                    """,
                    (existing["id"],),
                )
                conn.execute(
                    """
                    UPDATE embeddings
                    SET status = 'stale', vector_blob = NULL, updated_at = ?
                    WHERE image_id = ?
                    """,
                    (now, existing["id"]),
                )
                conn.execute(
                    """
                    UPDATE color_features
                    SET status = 'stale', hist_blob = NULL, updated_at = ?
                    WHERE image_id = ?
                    """,
                    (now, existing["id"]),
                )
                return int(existing["id"]), "changed"
            return int(existing["id"]), "unchanged"

    def update_thumbnail(self, image_id: int, thumbnail_path: str | None, status: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE images
                SET thumbnail_path = ?, thumbnail_status = ?
                WHERE id = ?
                """,
                (thumbnail_path, status, image_id),
            )

    def mark_embedding_not_required(self, image_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE images SET embedding_status = 'ready' WHERE id = ?",
                (image_id,),
            )

    def thumbnail_needs_generation(self, image_id: int) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT thumbnail_status, thumbnail_path
                FROM images
                WHERE id = ?
                """,
                (image_id,),
            ).fetchone()
        if row is None:
            return False
        if str(row["thumbnail_status"]) != "ready":
            return True
        thumbnail_path = row["thumbnail_path"]
        return not thumbnail_path or not Path(str(thumbnail_path)).exists()

    def mark_missing_for_folder(self, folder_id: int, seen_paths: Iterable[str]) -> int:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS seen_scan_paths(path TEXT PRIMARY KEY)")
            conn.execute("DELETE FROM seen_scan_paths")
            conn.executemany(
                "INSERT OR IGNORE INTO seen_scan_paths(path) VALUES (?)",
                ((path,) for path in seen_paths),
            )
            cur = conn.execute(
                """
                UPDATE images
                SET is_missing = 1, last_seen_at = ?
                WHERE folder_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM seen_scan_paths s WHERE s.path = images.file_path
                  )
                  AND is_missing = 0
                """,
                (now, folder_id),
            )
            conn.execute("DELETE FROM seen_scan_paths")
            return int(cur.rowcount)

    def finish_folder_scan(self, folder_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE folders SET last_scanned_at = ? WHERE id = ?",
                (utc_now_iso(), folder_id),
            )

    def list_images(
        self,
        *,
        limit: int = 500,
        offset: int = 0,
        folder_id: int | None = None,
        folder_path_prefix: str | None = None,
        collection_id: int | None = None,
        tag_id: int | None = None,
        tag_ids: Sequence[int] | None = None,
        tag_match_mode: str = "any",
        status_filter: str | None = None,
        text_query: str | None = None,
        include_missing: bool = False,
        sort_key: str = "default",
        sort_desc: bool = True,
    ) -> list[ImageItem]:
        clauses: list[str] = []
        params: list[object] = []
        if folder_id is not None:
            clauses.append("images.folder_id = ?")
            params.append(folder_id)
        if folder_path_prefix:
            clauses.append("images.file_path LIKE ? ESCAPE '\\'")
            params.append(self._folder_path_like(folder_path_prefix))
        if collection_id is not None:
            collection_ids = self.collection_descendant_ids(collection_id)
            if collection_ids:
                placeholders = ",".join("?" for _ in collection_ids)
                clauses.append(
                    f"""
                    EXISTS (
                        SELECT 1
                        FROM image_collections ic
                        WHERE ic.image_id = images.id
                          AND ic.collection_id IN ({placeholders})
                    )
                    """
                )
                params.extend(collection_ids)
            else:
                clauses.append("0 = 1")
        if tag_id is not None:
            tag_ids = [*list(tag_ids or []), tag_id]
        tag_clause, tag_params = self._tag_filter_clause(
            "images.id",
            tag_ids=tag_ids,
            tag_match_mode=tag_match_mode,
        )
        if tag_clause:
            clauses.append(tag_clause)
            params.extend(tag_params)
        if status_filter == "favorite":
            clauses.append("images.is_favorite = 1")
        elif status_filter == "unindexed":
            image_placeholders = ",".join("?" for _ in SUPPORTED_IMAGE_EXTENSIONS)
            clauses.append(
                f"images.file_ext IN ({image_placeholders}) AND images.embedding_status != 'ready'"
            )
            params.extend(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        elif status_filter == "missing":
            clauses.append("images.is_missing = 1")
            include_missing = True
        if not include_missing:
            clauses.append("images.is_missing = 0")
        if text_query:
            like = f"%{text_query.strip()}%"
            clauses.append(
                """
                (
                    images.file_name LIKE ?
                    OR images.file_path LIKE ?
                    OR COALESCE(images.note, '') LIKE ?
                    OR EXISTS (
                        SELECT 1
                        FROM image_tags it
                        JOIN tags t ON t.id = it.tag_id
                        WHERE it.image_id = images.id AND t.tag_name LIKE ?
                    )
                )
                """
            )
            params.extend([like, like, like, like])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_by = self._image_order_by(sort_key, sort_desc)
        params.extend([limit, offset])
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT images.*
                FROM images
                {where}
                ORDER BY {order_by}
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
            return [self._image_from_row(row) for row in rows]

    @staticmethod
    def _image_order_by(sort_key: str, sort_desc: bool) -> str:
        direction = "DESC" if sort_desc else "ASC"
        id_direction = "DESC" if sort_desc else "ASC"
        if sort_key == "name":
            return f"images.file_name COLLATE NOCASE {direction}, images.id {id_direction}"
        if sort_key == "modified":
            return f"images.modified_time_ns {direction}, images.id {id_direction}"
        if sort_key == "file_size":
            return f"images.file_size {direction}, images.id {id_direction}"
        if sort_key == "width":
            return f"images.width IS NULL ASC, images.width {direction}, images.id {id_direction}"
        if sort_key == "height":
            return f"images.height IS NULL ASC, images.height {direction}, images.id {id_direction}"
        if sort_key == "pixels":
            return (
                "(images.width IS NULL OR images.height IS NULL) ASC, "
                f"(images.width * images.height) {direction}, images.id {id_direction}"
            )
        if sort_key == "duration":
            return f"images.duration_ms IS NULL ASC, images.duration_ms {direction}, images.id {id_direction}"
        if sort_key == "imported":
            return f"images.imported_at {direction}, images.id {id_direction}"
        return "images.id DESC"

    @staticmethod
    def _tag_filter_clause(
        image_id_expr: str,
        *,
        tag_ids: Sequence[int] | None,
        tag_match_mode: str,
    ) -> tuple[str | None, list[object]]:
        clean_ids = sorted({int(tag_id) for tag_id in tag_ids or []})
        if not clean_ids:
            return None, []
        placeholders = ",".join("?" for _ in clean_ids)
        if tag_match_mode == "all":
            return (
                f"""
                {image_id_expr} IN (
                    SELECT it.image_id
                    FROM image_tags it
                    WHERE it.tag_id IN ({placeholders})
                    GROUP BY it.image_id
                    HAVING COUNT(DISTINCT it.tag_id) = ?
                )
                """,
                [*clean_ids, len(clean_ids)],
            )
        return (
            f"""
            EXISTS (
                SELECT 1
                FROM image_tags it
                WHERE it.image_id = {image_id_expr}
                  AND it.tag_id IN ({placeholders})
            )
            """,
            list(clean_ids),
        )

    def count_images(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM images").fetchone()
            return int(row["c"])

    def get_image(self, image_id: int) -> ImageItem | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
            return self._image_from_row(row) if row else None

    def images_by_ids(self, image_ids: Sequence[int], scores: dict[int, float] | None = None) -> list[ImageItem]:
        if not image_ids:
            return []
        placeholders = ",".join("?" for _ in image_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM images WHERE id IN ({placeholders})", tuple(image_ids)
            ).fetchall()
        by_id = {int(row["id"]): self._image_from_row(row, scores.get(int(row["id"])) if scores else None) for row in rows}
        return [by_id[image_id] for image_id in image_ids if image_id in by_id]

    def color_search_candidates(
        self,
        *,
        folder_id: int | None = None,
        folder_path_prefix: str | None = None,
        collection_id: int | None = None,
        tag_id: int | None = None,
        tag_ids: Sequence[int] | None = None,
        tag_match_mode: str = "any",
        status_filter: str | None = None,
    ) -> list[ImageItem]:
        if status_filter == "missing":
            return []

        image_placeholders = ",".join("?" for _ in SUPPORTED_IMAGE_EXTENSIONS)
        clauses = [
            "images.is_missing = 0",
            f"images.file_ext IN ({image_placeholders})",
        ]
        params: list[object] = list(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        if folder_id is not None:
            clauses.append("images.folder_id = ?")
            params.append(folder_id)
        if folder_path_prefix:
            clauses.append("images.file_path LIKE ? ESCAPE '\\'")
            params.append(self._folder_path_like(folder_path_prefix))
        if collection_id is not None:
            collection_ids = self.collection_descendant_ids(collection_id)
            if not collection_ids:
                return []
            placeholders = ",".join("?" for _ in collection_ids)
            clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM image_collections ic
                    WHERE ic.image_id = images.id
                      AND ic.collection_id IN ({placeholders})
                )
                """
            )
            params.extend(collection_ids)
        if tag_id is not None:
            tag_ids = [*list(tag_ids or []), tag_id]
        tag_clause, tag_params = self._tag_filter_clause(
            "images.id",
            tag_ids=tag_ids,
            tag_match_mode=tag_match_mode,
        )
        if tag_clause:
            clauses.append(tag_clause)
            params.extend(tag_params)
        if status_filter == "favorite":
            clauses.append("images.is_favorite = 1")
        elif status_filter == "unindexed":
            clauses.append("images.embedding_status != 'ready'")

        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT images.*
                FROM images
                WHERE {' AND '.join(clauses)}
                ORDER BY images.id DESC
                """,
                params,
            ).fetchall()
        return [self._image_from_row(row) for row in rows]

    def color_feature_needs_generation(
        self,
        image_id: int,
        *,
        vector_version: str = COLOR_VECTOR_VERSION,
        vector_dim: int = COLOR_VECTOR_DIM,
    ) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT vector_version, vector_dim, hist_blob, status
                FROM color_features
                WHERE image_id = ?
                """,
                (image_id,),
            ).fetchone()
        if row is None:
            return True
        return (
            str(row["vector_version"]) != vector_version
            or int(row["vector_dim"]) != vector_dim
            or str(row["status"]) != "ready"
            or row["hist_blob"] is None
        )

    def upsert_color_feature_success(
        self,
        *,
        image_id: int,
        vector: np.ndarray,
        vector_version: str = COLOR_VECTOR_VERSION,
    ) -> None:
        normalized = np.asarray(vector, dtype=np.float32)
        if normalized.ndim != 1:
            raise ValueError("color vector must be one-dimensional")
        dim = int(normalized.shape[0])
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO color_features(
                    image_id, vector_version, vector_dim, hist_blob,
                    status, error_message, updated_at
                )
                VALUES (?, ?, ?, ?, 'ready', NULL, ?)
                ON CONFLICT(image_id)
                DO UPDATE SET
                    vector_version = excluded.vector_version,
                    vector_dim = excluded.vector_dim,
                    hist_blob = excluded.hist_blob,
                    status = 'ready',
                    error_message = NULL,
                    updated_at = excluded.updated_at
                """,
                (image_id, vector_version, dim, normalized.tobytes(), now),
            )

    def mark_color_feature_failed(self, image_id: int, error_message: str) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO color_features(
                    image_id, vector_version, vector_dim, hist_blob,
                    status, error_message, updated_at
                )
                VALUES (?, ?, ?, NULL, 'failed', ?, ?)
                ON CONFLICT(image_id)
                DO UPDATE SET
                    vector_version = excluded.vector_version,
                    vector_dim = excluded.vector_dim,
                    hist_blob = NULL,
                    status = 'failed',
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    image_id,
                    COLOR_VECTOR_VERSION,
                    COLOR_VECTOR_DIM,
                    error_message[:2000],
                    now,
                ),
            )

    def color_features_by_image_ids(
        self,
        image_ids: Sequence[int],
        *,
        vector_version: str = COLOR_VECTOR_VERSION,
        vector_dim: int = COLOR_VECTOR_DIM,
    ) -> dict[int, np.ndarray]:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return {}
        placeholders = ",".join("?" for _ in clean_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT image_id, hist_blob
                FROM color_features
                WHERE image_id IN ({placeholders})
                  AND vector_version = ?
                  AND vector_dim = ?
                  AND status = 'ready'
                  AND hist_blob IS NOT NULL
                """,
                (*clean_ids, vector_version, vector_dim),
            ).fetchall()
        features: dict[int, np.ndarray] = {}
        for row in rows:
            vector = np.frombuffer(row["hist_blob"], dtype=np.float32)
            if vector.shape[0] == vector_dim:
                features[int(row["image_id"])] = vector.copy()
        return features

    def color_feature_ids_by_status(
        self,
        image_ids: Sequence[int],
        statuses: Sequence[str],
        *,
        vector_version: str = COLOR_VECTOR_VERSION,
        vector_dim: int = COLOR_VECTOR_DIM,
    ) -> set[int]:
        clean_ids = self._clean_ids(image_ids)
        clean_statuses = [status for status in statuses if status]
        if not clean_ids or not clean_statuses:
            return set()
        id_placeholders = ",".join("?" for _ in clean_ids)
        status_placeholders = ",".join("?" for _ in clean_statuses)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT image_id
                FROM color_features
                WHERE image_id IN ({id_placeholders})
                  AND status IN ({status_placeholders})
                  AND vector_version = ?
                  AND vector_dim = ?
                """,
                (*clean_ids, *clean_statuses, vector_version, vector_dim),
            ).fetchall()
        return {int(row["image_id"]) for row in rows}

    def update_note(self, image_id: int, note: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE images SET note = ? WHERE id = ?", (note, image_id))

    def update_favorite(self, image_id: int, is_favorite: bool) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE images SET is_favorite = ? WHERE id = ?",
                (1 if is_favorite else 0, image_id),
            )

    def update_favorites(self, image_ids: Sequence[int], is_favorite: bool) -> int:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return 0
        placeholders = ",".join("?" for _ in clean_ids)
        with self.connect() as conn:
            cur = conn.execute(
                f"UPDATE images SET is_favorite = ? WHERE id IN ({placeholders})",
                (1 if is_favorite else 0, *clean_ids),
            )
            return int(cur.rowcount)

    def list_tags(self) -> list[TagItem]:
        with self.connect() as conn:
            return [
                self._tag_from_row(row)
                for row in conn.execute("SELECT * FROM tags ORDER BY tag_name COLLATE NOCASE")
            ]

    def list_tags_with_counts(self) -> list[tuple[TagItem, int]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT t.*, COUNT(it.image_id) AS image_count
                FROM tags t
                LEFT JOIN image_tags it ON it.tag_id = t.id
                GROUP BY t.id
                ORDER BY t.tag_name COLLATE NOCASE
                """
            ).fetchall()
        return [(self._tag_from_row(row), int(row["image_count"])) for row in rows]

    def rename_tag(self, tag_id: int, new_name: str) -> bool:
        clean_name = new_name.strip()
        if not clean_name:
            raise ValueError("tag name must not be empty")
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT id FROM tags WHERE tag_name = ?", (clean_name,)
            ).fetchone()
            if existing is not None and int(existing["id"]) != tag_id:
                raise ValueError("tag name already exists")
            cur = conn.execute(
                "UPDATE tags SET tag_name = ? WHERE id = ?",
                (clean_name, tag_id),
            )
            return int(cur.rowcount) > 0

    def delete_tag(self, tag_id: int) -> int:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM image_tags WHERE tag_id = ?", (tag_id,))
            conn.execute("DELETE FROM tags WHERE id = ?", (tag_id,))
            return int(cur.rowcount)

    def merge_tag(self, source_tag_id: int, target_tag_id: int) -> int:
        if source_tag_id == target_tag_id:
            return 0
        now = utc_now_iso()
        with self.connect() as conn:
            source = conn.execute(
                "SELECT id FROM tags WHERE id = ?", (source_tag_id,)
            ).fetchone()
            target = conn.execute(
                "SELECT id FROM tags WHERE id = ?", (target_tag_id,)
            ).fetchone()
            if source is None or target is None:
                raise ValueError("tag not found")
            source_count = conn.execute(
                "SELECT COUNT(*) AS count FROM image_tags WHERE tag_id = ?",
                (source_tag_id,),
            ).fetchone()
            conn.execute(
                """
                INSERT OR IGNORE INTO image_tags(
                    image_id, tag_id, source, confirmed_by_user, created_at
                )
                SELECT image_id, ?, 'manual', 1, ?
                FROM image_tags
                WHERE tag_id = ?
                """,
                (target_tag_id, now, source_tag_id),
            )
            conn.execute("DELETE FROM tags WHERE id = ?", (source_tag_id,))
            return int(source_count["count"])

    def get_image_tags(self, image_id: int) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT t.tag_name
                FROM tags t
                JOIN image_tags it ON it.tag_id = t.id
                WHERE it.image_id = ?
                ORDER BY t.tag_name COLLATE NOCASE
                """,
                (image_id,),
            ).fetchall()
            return [str(row["tag_name"]) for row in rows]

    def tag_counts_for_images(self, image_ids: Sequence[int]) -> dict[str, int]:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return {}
        placeholders = ",".join("?" for _ in clean_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT t.tag_name, COUNT(DISTINCT it.image_id) AS image_count
                FROM image_tags it
                JOIN tags t ON t.id = it.tag_id
                WHERE it.image_id IN ({placeholders})
                GROUP BY t.id
                ORDER BY t.tag_name COLLATE NOCASE
                """,
                tuple(clean_ids),
            ).fetchall()
        return {str(row["tag_name"]): int(row["image_count"]) for row in rows}

    def count_images_with_tags(self, image_ids: Sequence[int]) -> int:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return 0
        placeholders = ",".join("?" for _ in clean_ids)
        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(DISTINCT image_id) AS image_count
                FROM image_tags
                WHERE image_id IN ({placeholders})
                """,
                tuple(clean_ids),
            ).fetchone()
        return int(row["image_count"])

    def set_image_tags(self, image_id: int, tag_names: Sequence[str]) -> None:
        clean_names = self._clean_tag_names(tag_names)

        now = utc_now_iso()
        with self.connect() as conn:
            tag_ids = []
            for name in clean_names:
                conn.execute(
                    """
                    INSERT INTO tags(tag_name, tag_type, created_at)
                    VALUES (?, 'user', ?)
                    ON CONFLICT(tag_name) DO NOTHING
                    """,
                    (name, now),
                )
                row = conn.execute("SELECT id FROM tags WHERE tag_name = ?", (name,)).fetchone()
                tag_ids.append(int(row["id"]))

            conn.execute("DELETE FROM image_tags WHERE image_id = ?", (image_id,))
            conn.executemany(
                """
                INSERT INTO image_tags(image_id, tag_id, source, confirmed_by_user, created_at)
                VALUES (?, ?, 'manual', 1, ?)
                """,
                ((image_id, tag_id, now) for tag_id in tag_ids),
            )
            self._delete_unused_tags(conn)

    def add_tags_to_images(self, image_ids: Sequence[int], tag_names: Sequence[str]) -> int:
        clean_ids = self._clean_ids(image_ids)
        clean_names = self._clean_tag_names(tag_names)
        if not clean_ids or not clean_names:
            return 0

        now = utc_now_iso()
        inserted = 0
        with self.connect() as conn:
            tag_ids = []
            for name in clean_names:
                conn.execute(
                    """
                    INSERT INTO tags(tag_name, tag_type, created_at)
                    VALUES (?, 'user', ?)
                    ON CONFLICT(tag_name) DO NOTHING
                    """,
                    (name, now),
                )
                row = conn.execute("SELECT id FROM tags WHERE tag_name = ?", (name,)).fetchone()
                tag_ids.append(int(row["id"]))

            for image_id in clean_ids:
                for tag_id in tag_ids:
                    cur = conn.execute(
                        """
                        INSERT OR IGNORE INTO image_tags(
                            image_id, tag_id, source, confirmed_by_user, created_at
                        )
                        VALUES (?, ?, 'manual', 1, ?)
                        """,
                        (image_id, tag_id, now),
                    )
                    inserted += int(cur.rowcount)
        return inserted

    def clear_tags_for_images(self, image_ids: Sequence[int]) -> int:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return 0
        placeholders = ",".join("?" for _ in clean_ids)
        with self.connect() as conn:
            cur = conn.execute(
                f"DELETE FROM image_tags WHERE image_id IN ({placeholders})",
                tuple(clean_ids),
            )
            self._delete_unused_tags(conn)
            return int(cur.rowcount)

    def remove_tags_from_images(
        self,
        image_ids: Sequence[int],
        tag_names: Sequence[str],
    ) -> int:
        clean_ids = self._clean_ids(image_ids)
        clean_names = self._clean_tag_names(tag_names)
        if not clean_ids or not clean_names:
            return 0
        image_placeholders = ",".join("?" for _ in clean_ids)
        tag_placeholders = ",".join("?" for _ in clean_names)
        with self.connect() as conn:
            cur = conn.execute(
                f"""
                DELETE FROM image_tags
                WHERE image_id IN ({image_placeholders})
                  AND tag_id IN (
                      SELECT id
                      FROM tags
                      WHERE tag_name IN ({tag_placeholders})
                  )
                """,
                (*clean_ids, *clean_names),
            )
            self._delete_unused_tags(conn)
            return int(cur.rowcount)

    def remove_images_from_library(self, image_ids: Sequence[int]) -> list[str]:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return []
        placeholders = ",".join("?" for _ in clean_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT thumbnail_path
                FROM images
                WHERE id IN ({placeholders})
                  AND thumbnail_path IS NOT NULL
                """,
                tuple(clean_ids),
            ).fetchall()
            thumbnail_paths = [str(row["thumbnail_path"]) for row in rows]
            conn.execute(f"DELETE FROM images WHERE id IN ({placeholders})", tuple(clean_ids))
            self._delete_unused_tags(conn)
            return thumbnail_paths

    def remove_folder_from_library(self, folder_id: int) -> tuple[list[str], int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT thumbnail_path
                FROM images
                WHERE folder_id = ?
                  AND thumbnail_path IS NOT NULL
                """,
                (folder_id,),
            ).fetchall()
            count_row = conn.execute(
                "SELECT COUNT(*) AS count FROM images WHERE folder_id = ?",
                (folder_id,),
            ).fetchone()
            thumbnail_paths = [str(row["thumbnail_path"]) for row in rows]
            conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
            self._delete_unused_tags(conn)
            return thumbnail_paths, int(count_row["count"])

    def remove_images_by_folder_path_prefix(self, folder_path_prefix: str) -> tuple[list[str], int]:
        like = self._folder_path_like(folder_path_prefix)
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT thumbnail_path
                FROM images
                WHERE file_path LIKE ? ESCAPE '\\'
                  AND thumbnail_path IS NOT NULL
                """,
                (like,),
            ).fetchall()
            thumbnail_paths = [str(row["thumbnail_path"]) for row in rows]
            cur = conn.execute(
                "DELETE FROM images WHERE file_path LIKE ? ESCAPE '\\'",
                (like,),
            )
            self._delete_unused_tags(conn)
            return thumbnail_paths, int(cur.rowcount)

    def upsert_search_feedback(
        self,
        *,
        query: str,
        image_id: int,
        model_name: str,
        model_revision: str,
        embedding_dim: int,
        score: float | None,
        label: str,
    ) -> None:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("search feedback query must not be empty")
        if label not in {"relevant", "irrelevant", "ignored"}:
            raise ValueError(f"unsupported search feedback label: {label}")

        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO search_feedback(
                    query, image_id, model_name, model_revision, embedding_dim,
                    score, label, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(query, image_id, model_name, model_revision, embedding_dim)
                DO UPDATE SET
                    score = excluded.score,
                    label = excluded.label,
                    updated_at = excluded.updated_at
                """,
                (
                    clean_query,
                    image_id,
                    model_name,
                    model_revision,
                    embedding_dim,
                    score,
                    label,
                    now,
                    now,
                ),
            )

    def get_search_feedback(
        self,
        *,
        query: str,
        image_id: int,
        model_name: str,
        model_revision: str,
        embedding_dim: int,
    ) -> str | None:
        clean_query = query.strip()
        if not clean_query:
            return None
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT label
                FROM search_feedback
                WHERE query = ?
                  AND image_id = ?
                  AND model_name = ?
                  AND model_revision = ?
                  AND embedding_dim = ?
                """,
                (clean_query, image_id, model_name, model_revision, embedding_dim),
            ).fetchone()
            return str(row["label"]) if row else None

    def search_feedback_counts(
        self,
        *,
        query: str,
        model_name: str,
        model_revision: str,
        embedding_dim: int,
    ) -> dict[str, int]:
        counts = {"relevant": 0, "irrelevant": 0, "ignored": 0}
        clean_query = query.strip()
        if not clean_query:
            return counts
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT label, COUNT(*) AS count
                FROM search_feedback
                WHERE query = ?
                  AND model_name = ?
                  AND model_revision = ?
                  AND embedding_dim = ?
                GROUP BY label
                """,
                (clean_query, model_name, model_revision, embedding_dim),
            ).fetchall()
        for row in rows:
            label = str(row["label"])
            if label in counts:
                counts[label] = int(row["count"])
        return counts

    def list_saved_views(self) -> list[SavedViewItem]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM saved_views
                ORDER BY name COLLATE NOCASE
                """
            ).fetchall()
        return [self._saved_view_from_row(row) for row in rows]

    def get_saved_view(self, saved_view_id: int) -> SavedViewItem | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM saved_views WHERE id = ?",
                (saved_view_id,),
            ).fetchone()
        return self._saved_view_from_row(row) if row else None

    def upsert_saved_view(self, name: str, payload_json: str) -> int:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("saved view name must not be empty")
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO saved_views(name, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (clean_name, payload_json, now, now),
            )
            row = conn.execute(
                "SELECT id FROM saved_views WHERE name = ?",
                (clean_name,),
            ).fetchone()
            return int(row["id"])

    def rename_saved_view(self, saved_view_id: int, new_name: str) -> bool:
        clean_name = new_name.strip()
        if not clean_name:
            raise ValueError("saved view name must not be empty")
        now = utc_now_iso()
        with self.connect() as conn:
            try:
                cur = conn.execute(
                    "UPDATE saved_views SET name = ?, updated_at = ? WHERE id = ?",
                    (clean_name, now, saved_view_id),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("saved view name already exists") from exc
            return int(cur.rowcount) > 0

    def delete_saved_view(self, saved_view_id: int) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM saved_views WHERE id = ?", (saved_view_id,))
            return int(cur.rowcount) > 0

    def create_temporary_project(
        self,
        name: str,
        image_ids: Sequence[int],
        *,
        summary: str = "",
    ) -> int:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("temporary project name must not be empty")
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            raise ValueError("temporary project must contain at least one image")
        clean_summary = _clean_optional_text(summary, max_length=600) or ""
        now = utc_now_iso()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT id FROM temporary_projects WHERE name = ?",
                (clean_name,),
            ).fetchone()
            if existing is not None:
                suffix = 2
                base_name = clean_name
                while True:
                    candidate = f"{base_name} {suffix}"
                    row = conn.execute(
                        "SELECT id FROM temporary_projects WHERE name = ?",
                        (candidate,),
                    ).fetchone()
                    if row is None:
                        clean_name = candidate
                        break
                    suffix += 1

            cur = conn.execute(
                """
                INSERT INTO temporary_projects(name, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (clean_name, clean_summary, now, now),
            )
            project_id = int(cur.lastrowid)
            for index, image_id in enumerate(clean_ids):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO temporary_project_images(
                        project_id, image_id, sort_order, created_at
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (project_id, image_id, index, now),
                )
            return project_id

    def list_temporary_projects(self) -> list[TemporaryProjectItem]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT p.*, COUNT(tpi.image_id) AS image_count
                FROM temporary_projects p
                LEFT JOIN temporary_project_images tpi ON tpi.project_id = p.id
                GROUP BY p.id
                ORDER BY p.updated_at DESC, p.id DESC
                """
            ).fetchall()
        return [
            TemporaryProjectItem(
                id=int(row["id"]),
                name=str(row["name"]),
                image_count=int(row["image_count"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                summary=str(row["summary"] or ""),
            )
            for row in rows
        ]

    def temporary_project_image_ids(self, project_id: int) -> list[int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT image_id
                FROM temporary_project_images
                WHERE project_id = ?
                ORDER BY sort_order, image_id
                """,
                (project_id,),
            ).fetchall()
        return [int(row["image_id"]) for row in rows]

    def temporary_project_image_badges(self, project_id: int) -> dict[int, list[str]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT image_id, intent_label
                FROM temporary_project_images
                WHERE project_id = ?
                  AND intent_label IS NOT NULL
                  AND TRIM(intent_label) != ''
                ORDER BY sort_order, image_id
                """,
                (project_id,),
            ).fetchall()
        return {
            int(row["image_id"]): [str(row["intent_label"])]
            for row in rows
        }

    def get_temporary_project(self, project_id: int) -> TemporaryProjectItem | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT p.*, COUNT(tpi.image_id) AS image_count
                FROM temporary_projects p
                LEFT JOIN temporary_project_images tpi ON tpi.project_id = p.id
                WHERE p.id = ?
                GROUP BY p.id
                """,
                (project_id,),
            ).fetchone()
        if row is None:
            return None
        return TemporaryProjectItem(
            id=int(row["id"]),
            name=str(row["name"]),
            image_count=int(row["image_count"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            summary=str(row["summary"] or ""),
        )

    def delete_temporary_project(self, project_id: int) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM temporary_projects WHERE id = ?", (project_id,))
            return int(cur.rowcount) > 0

    def update_temporary_project_details(
        self,
        project_id: int,
        *,
        name: str | None = None,
        summary: str | None = None,
    ) -> TemporaryProjectItem | None:
        clean_name = _clean_optional_text(name, max_length=80) if name is not None else None
        clean_summary = _clean_optional_text(summary, max_length=600) if summary is not None else None
        if clean_name is None and clean_summary is None:
            return self.get_temporary_project(project_id)
        now = utc_now_iso()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id, name FROM temporary_projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            if row is None:
                return None
            if clean_name is not None:
                clean_name = self._unique_temporary_project_name(
                    conn,
                    clean_name,
                    exclude_project_id=project_id,
                )
            assignments: list[str] = ["updated_at = ?"]
            params: list[object] = [now]
            if clean_name is not None:
                assignments.append("name = ?")
                params.append(clean_name)
            if clean_summary is not None:
                assignments.append("summary = ?")
                params.append(clean_summary)
            params.append(project_id)
            conn.execute(
                f"UPDATE temporary_projects SET {', '.join(assignments)} WHERE id = ?",
                params,
            )
        return self.get_temporary_project(project_id)

    @staticmethod
    def _unique_temporary_project_name(
        conn: sqlite3.Connection,
        name: str,
        *,
        exclude_project_id: int | None = None,
    ) -> str:
        base_name = name
        suffix = 2
        candidate = base_name
        while True:
            row = conn.execute(
                "SELECT id FROM temporary_projects WHERE name = ?",
                (candidate,),
            ).fetchone()
            if row is None or (
                exclude_project_id is not None and int(row["id"]) == exclude_project_id
            ):
                return candidate
            candidate = f"{base_name} {suffix}"
            suffix += 1

    def add_images_to_temporary_project(
        self,
        project_id: int,
        image_ids: Sequence[int],
        *,
        intent_labels: Mapping[int, str] | None = None,
        intent_queries: Mapping[int, str] | None = None,
    ) -> int:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return 0
        intent_labels = intent_labels or {}
        intent_queries = intent_queries or {}
        now = utc_now_iso()
        changed = 0
        with self.connect() as conn:
            project = conn.execute(
                "SELECT id FROM temporary_projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            if project is None:
                raise ValueError("temporary project does not exist")
            row = conn.execute(
                """
                SELECT COALESCE(MAX(sort_order), -1) AS max_sort_order
                FROM temporary_project_images
                WHERE project_id = ?
                """,
                (project_id,),
            ).fetchone()
            next_order = int(row["max_sort_order"]) + 1 if row is not None else 0
            for offset, image_id in enumerate(clean_ids):
                label = _clean_optional_text(intent_labels.get(image_id), max_length=80)
                query = _clean_optional_text(intent_queries.get(image_id), max_length=160)
                cur = conn.execute(
                    """
                    INSERT INTO temporary_project_images(
                        project_id, image_id, sort_order, created_at, intent_label, intent_query
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, image_id) DO UPDATE SET
                        intent_label = COALESCE(excluded.intent_label, temporary_project_images.intent_label),
                        intent_query = COALESCE(excluded.intent_query, temporary_project_images.intent_query)
                    """,
                    (project_id, image_id, next_order + offset, now, label, query),
                )
                changed += int(cur.rowcount)
            conn.execute(
                "UPDATE temporary_projects SET updated_at = ? WHERE id = ?",
                (now, project_id),
            )
        return changed

    def remove_images_from_temporary_project(
        self,
        project_id: int,
        image_ids: Sequence[int],
    ) -> int:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return 0
        placeholders = ",".join("?" for _ in clean_ids)
        now = utc_now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                f"""
                DELETE FROM temporary_project_images
                WHERE project_id = ?
                  AND image_id IN ({placeholders})
                """,
                [project_id, *clean_ids],
            )
            removed = int(cur.rowcount)
            if removed:
                conn.execute(
                    "UPDATE temporary_projects SET updated_at = ? WHERE id = ?",
                    (now, project_id),
                )
            return removed

    def create_inspiration_project(
        self,
        *,
        title: str,
        brief: str,
        answers: str,
        questions: Sequence[str],
        provider_name: str,
        model_name: str,
        terms: Sequence[InspirationTerm],
        selected_titles: set[str],
    ) -> int:
        clean_title = title.strip() or "未命名灵感项目"
        clean_brief = brief.strip()
        if not clean_brief:
            raise ValueError("inspiration project brief must not be empty")
        now = utc_now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO inspiration_projects(
                    title, brief, answers, questions_json, provider_name, model_name,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_title,
                    clean_brief,
                    answers.strip(),
                    json.dumps(list(questions), ensure_ascii=False),
                    provider_name,
                    model_name,
                    now,
                    now,
                ),
            )
            project_id = int(cur.lastrowid)
            for index, term in enumerate(terms):
                conn.execute(
                    """
                    INSERT INTO inspiration_terms(
                        project_id, title, query, axis, reason, selected, sort_order, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        term.title,
                        term.query,
                        term.axis,
                        term.reason,
                        1 if term.title in selected_titles else 0,
                        index,
                        now,
                    ),
                )
            return project_id

    def inspiration_terms_for_project(
        self,
        project_id: int,
        *,
        selected_only: bool = False,
    ) -> list[InspirationTerm]:
        clauses = ["project_id = ?"]
        params: list[object] = [project_id]
        if selected_only:
            clauses.append("selected = 1")
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM inspiration_terms
                WHERE {' AND '.join(clauses)}
                ORDER BY sort_order, id
                """,
                params,
            ).fetchall()
        return [
            InspirationTerm(
                id=int(row["id"]),
                title=str(row["title"]),
                query=str(row["query"]),
                axis=str(row["axis"]),
                reason=str(row["reason"] or ""),
                selected=bool(row["selected"]),
            )
            for row in rows
        ]

    def next_embedding_jobs(
        self,
        *,
        model_name: str,
        model_revision: str,
        embedding_dim: int,
        limit: int,
    ) -> list[ImageItem]:
        image_placeholders = ",".join("?" for _ in SUPPORTED_IMAGE_EXTENSIONS)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT i.*
                FROM images i
                LEFT JOIN embeddings e
                  ON e.image_id = i.id
                 AND e.model_name = ?
                 AND e.model_revision = ?
                 AND e.embedding_dim = ?
                WHERE i.is_missing = 0
                  AND i.file_ext IN ({image_placeholders})
                  AND (e.id IS NULL OR e.status IN ('pending', 'stale'))
                ORDER BY i.id
                LIMIT ?
                """,
                (
                    model_name,
                    model_revision,
                    embedding_dim,
                    *sorted(SUPPORTED_IMAGE_EXTENSIONS),
                    limit,
                ),
            ).fetchall()
            return [self._image_from_row(row) for row in rows]

    def mark_embedding_processing(
        self,
        *,
        image_id: int,
        model_name: str,
        model_revision: str,
        embedding_dim: int,
    ) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO embeddings(
                    image_id, model_name, model_revision, embedding_dim,
                    vector_blob, status, error_message, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, NULL, 'processing', NULL, ?, ?)
                ON CONFLICT(image_id, model_name, model_revision, embedding_dim)
                DO UPDATE SET status = 'processing', error_message = NULL, updated_at = excluded.updated_at
                """,
                (image_id, model_name, model_revision, embedding_dim, now, now),
            )
            conn.execute(
                "UPDATE images SET embedding_status = 'processing' WHERE id = ?",
                (image_id,),
            )

    def upsert_embedding_success(
        self,
        *,
        image_id: int,
        model_name: str,
        model_revision: str,
        vector: np.ndarray,
    ) -> None:
        normalized = np.asarray(vector, dtype=np.float32)
        if normalized.ndim != 1:
            raise ValueError("embedding vector must be one-dimensional")
        dim = int(normalized.shape[0])
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO embeddings(
                    image_id, model_name, model_revision, embedding_dim,
                    vector_blob, status, error_message, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'ready', NULL, ?, ?)
                ON CONFLICT(image_id, model_name, model_revision, embedding_dim)
                DO UPDATE SET
                    vector_blob = excluded.vector_blob,
                    status = 'ready',
                    error_message = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    image_id,
                    model_name,
                    model_revision,
                    dim,
                    normalized.tobytes(),
                    now,
                    now,
                ),
            )
            conn.execute(
                "UPDATE images SET embedding_status = 'ready' WHERE id = ?",
                (image_id,),
            )

    def mark_embedding_failed(
        self,
        *,
        image_id: int,
        model_name: str,
        model_revision: str,
        embedding_dim: int,
        error_message: str,
    ) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO embeddings(
                    image_id, model_name, model_revision, embedding_dim,
                    vector_blob, status, error_message, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, NULL, 'failed', ?, ?, ?)
                ON CONFLICT(image_id, model_name, model_revision, embedding_dim)
                DO UPDATE SET status = 'failed', error_message = excluded.error_message,
                              vector_blob = NULL, updated_at = excluded.updated_at
                """,
                (
                    image_id,
                    model_name,
                    model_revision,
                    embedding_dim,
                    error_message[:2000],
                    now,
                    now,
                ),
            )
            conn.execute(
                "UPDATE images SET embedding_status = 'failed' WHERE id = ?",
                (image_id,),
            )

    def retry_failed_embeddings(
        self,
        *,
        model_name: str,
        model_revision: str,
        embedding_dim: int,
    ) -> int:
        now = utc_now_iso()
        image_placeholders = ",".join("?" for _ in SUPPORTED_IMAGE_EXTENSIONS)
        with self.connect() as conn:
            cur = conn.execute(
                f"""
                UPDATE embeddings
                SET status = 'pending', error_message = NULL, updated_at = ?
                WHERE model_name = ? AND model_revision = ? AND embedding_dim = ?
                  AND status IN ('failed', 'processing')
                  AND EXISTS (
                      SELECT 1
                      FROM images i
                      WHERE i.id = embeddings.image_id
                        AND i.is_missing = 0
                        AND i.file_ext IN ({image_placeholders})
                  )
                """,
                (
                    now,
                    model_name,
                    model_revision,
                    embedding_dim,
                    *sorted(SUPPORTED_IMAGE_EXTENSIONS),
                ),
            )
            conn.execute(
                f"""
                UPDATE images
                SET embedding_status = 'pending'
                WHERE file_ext IN ({image_placeholders})
                  AND id IN (
                    SELECT image_id FROM embeddings
                    WHERE model_name = ? AND model_revision = ? AND embedding_dim = ?
                      AND status = 'pending'
                )
                """,
                (
                    *sorted(SUPPORTED_IMAGE_EXTENSIONS),
                    model_name,
                    model_revision,
                    embedding_dim,
                ),
            )
            return int(cur.rowcount)

    def embeddings_for_model(
        self,
        *,
        model_name: str,
        model_revision: str,
        embedding_dim: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_placeholders = ",".join("?" for _ in SUPPORTED_IMAGE_EXTENSIONS)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT e.image_id, e.vector_blob
                FROM embeddings e
                JOIN images i ON i.id = e.image_id
                WHERE e.model_name = ?
                  AND e.model_revision = ?
                  AND e.embedding_dim = ?
                  AND e.status = 'ready'
                  AND e.vector_blob IS NOT NULL
                  AND i.is_missing = 0
                  AND i.file_ext IN ({image_placeholders})
                ORDER BY e.image_id
                """,
                (
                    model_name,
                    model_revision,
                    embedding_dim,
                    *sorted(SUPPORTED_IMAGE_EXTENSIONS),
                ),
            ).fetchall()

        ids: list[int] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            vector = np.frombuffer(row["vector_blob"], dtype=np.float32)
            if vector.shape[0] != embedding_dim:
                continue
            ids.append(int(row["image_id"]))
            vectors.append(vector.copy())

        if not vectors:
            return np.empty((0,), dtype=np.int64), np.empty((0, embedding_dim), dtype=np.float32)
        return np.asarray(ids, dtype=np.int64), np.vstack(vectors).astype(np.float32, copy=False)

    def embedding_vector_for_image(
        self,
        image_id: int,
        *,
        model_name: str,
        model_revision: str,
        embedding_dim: int,
    ) -> np.ndarray | None:
        image_placeholders = ",".join("?" for _ in SUPPORTED_IMAGE_EXTENSIONS)
        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT e.vector_blob
                FROM embeddings e
                JOIN images i ON i.id = e.image_id
                WHERE e.image_id = ?
                  AND e.model_name = ?
                  AND e.model_revision = ?
                  AND e.embedding_dim = ?
                  AND e.status = 'ready'
                  AND e.vector_blob IS NOT NULL
                  AND i.is_missing = 0
                  AND i.file_ext IN ({image_placeholders})
                """,
                (
                    image_id,
                    model_name,
                    model_revision,
                    embedding_dim,
                    *sorted(SUPPORTED_IMAGE_EXTENSIONS),
                ),
            ).fetchone()
        if row is None:
            return None
        vector = np.frombuffer(row["vector_blob"], dtype=np.float32)
        if vector.shape[0] != embedding_dim:
            return None
        return vector.copy()

    def searchable_image_ids_for_model(
        self,
        *,
        model_name: str,
        model_revision: str,
        embedding_dim: int,
        folder_id: int | None = None,
        folder_path_prefix: str | None = None,
        collection_id: int | None = None,
        tag_id: int | None = None,
        tag_ids: Sequence[int] | None = None,
        tag_match_mode: str = "any",
        status_filter: str | None = None,
    ) -> list[int]:
        if status_filter in {"missing", "unindexed"}:
            return []

        clauses = [
            "i.is_missing = 0",
            self._image_extension_clause("i.file_ext"),
            "e.model_name = ?",
            "e.model_revision = ?",
            "e.embedding_dim = ?",
            "e.status = 'ready'",
            "e.vector_blob IS NOT NULL",
        ]
        params: list[object] = [model_name, model_revision, embedding_dim]
        params[:0] = sorted(SUPPORTED_IMAGE_EXTENSIONS)

        if folder_id is not None:
            clauses.append("i.folder_id = ?")
            params.append(folder_id)
        if folder_path_prefix:
            clauses.append("i.file_path LIKE ? ESCAPE '\\'")
            params.append(self._folder_path_like(folder_path_prefix))
        if collection_id is not None:
            collection_ids = self.collection_descendant_ids(collection_id)
            if not collection_ids:
                return []
            placeholders = ",".join("?" for _ in collection_ids)
            clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM image_collections ic
                    WHERE ic.image_id = i.id
                      AND ic.collection_id IN ({placeholders})
                )
                """
            )
            params.extend(collection_ids)
        if tag_id is not None:
            tag_ids = [*list(tag_ids or []), tag_id]
        tag_clause, tag_params = self._tag_filter_clause(
            "i.id",
            tag_ids=tag_ids,
            tag_match_mode=tag_match_mode,
        )
        if tag_clause:
            clauses.append(tag_clause)
            params.extend(tag_params)
        if status_filter == "favorite":
            clauses.append("i.is_favorite = 1")

        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT i.id
                FROM images i
                JOIN embeddings e ON e.image_id = i.id
                WHERE {' AND '.join(clauses)}
                ORDER BY i.id
                """,
                params,
            ).fetchall()
            return [int(row["id"]) for row in rows]

    def embedding_stats(
        self,
        *,
        model_name: str,
        model_revision: str,
        embedding_dim: int,
    ) -> dict[str, int]:
        with self.connect() as conn:
            total_row = conn.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM images
                WHERE is_missing = 0
                  AND {self._image_extension_clause("file_ext")}
                """,
                tuple(sorted(SUPPORTED_IMAGE_EXTENSIONS)),
            ).fetchone()
            status_rows = conn.execute(
                """
                SELECT e.status, COUNT(*) AS count
                FROM embeddings e
                JOIN images i ON i.id = e.image_id
                WHERE e.model_name = ?
                  AND e.model_revision = ?
                  AND e.embedding_dim = ?
                  AND i.is_missing = 0
                  AND i.file_ext IN ({})
                GROUP BY e.status
                """.format(",".join("?" for _ in SUPPORTED_IMAGE_EXTENSIONS)),
                (
                    model_name,
                    model_revision,
                    embedding_dim,
                    *sorted(SUPPORTED_IMAGE_EXTENSIONS),
                ),
            ).fetchall()

        stats = {
            "total": int(total_row["count"]),
            "ready": 0,
            "failed": 0,
            "processing": 0,
            "stale": 0,
            "pending": 0,
        }
        for row in status_rows:
            status = str(row["status"])
            if status in stats:
                stats[status] = int(row["count"])
        known = stats["ready"] + stats["failed"] + stats["processing"] + stats["stale"]
        stats["pending"] = max(0, stats["total"] - known)
        return stats

    def set_setting(self, key: str, value: str) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO app_settings(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, value, now),
            )

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
            return str(row["value"]) if row else default

    @staticmethod
    def _folder_from_row(row: sqlite3.Row) -> FolderItem:
        return FolderItem(
            id=int(row["id"]),
            folder_path=str(row["folder_path"]),
            import_mode=str(row["import_mode"]),
            added_at=str(row["added_at"]),
            last_scanned_at=row["last_scanned_at"],
            is_active=bool(row["is_active"]),
        )

    @staticmethod
    def _collection_from_row(row: sqlite3.Row) -> CollectionItem:
        return CollectionItem(
            id=int(row["id"]),
            parent_id=int(row["parent_id"]) if row["parent_id"] is not None else None,
            name=str(row["name"]),
            sort_order=int(row["sort_order"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _image_from_row(row: sqlite3.Row, score: float | None = None) -> ImageItem:
        return ImageItem(
            id=int(row["id"]),
            folder_id=int(row["folder_id"]),
            file_path=str(row["file_path"]),
            file_name=str(row["file_name"]),
            file_ext=str(row["file_ext"]),
            file_size=int(row["file_size"]),
            width=row["width"],
            height=row["height"],
            created_at=row["created_at"],
            modified_at=row["modified_at"],
            modified_time_ns=int(row["modified_time_ns"]),
            imported_at=str(row["imported_at"]),
            last_seen_at=str(row["last_seen_at"]),
            thumbnail_path=row["thumbnail_path"],
            thumbnail_status=str(row["thumbnail_status"]),
            embedding_status=str(row["embedding_status"]),
            is_missing=bool(row["is_missing"]),
            is_favorite=bool(row["is_favorite"]),
            note=row["note"],
            duration_ms=row["duration_ms"],
            score=score,
        )

    @staticmethod
    def _tag_from_row(row: sqlite3.Row) -> TagItem:
        return TagItem(
            id=int(row["id"]),
            tag_name=str(row["tag_name"]),
            tag_type=str(row["tag_type"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _saved_view_from_row(row: sqlite3.Row) -> SavedViewItem:
        return SavedViewItem(
            id=int(row["id"]),
            name=str(row["name"]),
            payload_json=str(row["payload_json"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _clean_tag_names(tag_names: Sequence[str]) -> list[str]:
        clean_names: list[str] = []
        seen: set[str] = set()
        for name in tag_names:
            clean = name.strip()
            key = clean.casefold()
            if clean and key not in seen:
                clean_names.append(clean)
                seen.add(key)
        return clean_names

    @classmethod
    def _folder_path_like(cls, folder_path_prefix: str) -> str:
        normalized = cls._normalize_folder_prefix(folder_path_prefix)
        escaped = (
            normalized.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        return f"{escaped}{os.sep}%"

    @staticmethod
    def _normalize_folder_prefix(folder_path: str) -> str:
        normalized = os.path.abspath(os.path.expanduser(folder_path))
        return normalized.rstrip(os.sep) or os.sep

    @staticmethod
    def _path_is_under_prefix(path: str, folder_path_prefix: str, *, include_self: bool) -> bool:
        normalized_path = os.path.abspath(os.path.expanduser(path)).rstrip(os.sep) or os.sep
        normalized_prefix = os.path.abspath(os.path.expanduser(folder_path_prefix)).rstrip(os.sep) or os.sep
        if include_self and normalized_path == normalized_prefix:
            return True
        if normalized_prefix == os.sep:
            return normalized_path.startswith(os.sep)
        return normalized_path.startswith(f"{normalized_prefix}{os.sep}")

    @staticmethod
    def _image_extension_clause(column: str) -> str:
        placeholders = ",".join("?" for _ in SUPPORTED_IMAGE_EXTENSIONS)
        return f"{column} IN ({placeholders})"

    @staticmethod
    def _clean_ids(image_ids: Sequence[int]) -> list[int]:
        clean_ids: list[int] = []
        seen = set()
        for image_id in image_ids:
            clean = int(image_id)
            if clean not in seen:
                clean_ids.append(clean)
                seen.add(clean)
        return clean_ids

    @staticmethod
    def _delete_unused_tags(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            DELETE FROM tags
            WHERE NOT EXISTS (
                SELECT 1
                FROM image_tags it
                WHERE it.tag_id = tags.id
            )
            """
        )
