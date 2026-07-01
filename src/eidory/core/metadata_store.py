from __future__ import annotations

import os
import json
import re
import sqlite3
import threading
from itertools import islice
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from eidory.core.ai_vision import AIVisionAnalysis, AI_VISION_PROMPT_VERSION
from eidory.core.color_features import COLOR_VECTOR_DIM, COLOR_VECTOR_VERSION
from eidory.core.duplicate_detection import ImageHashCacheRecord
from eidory.core.inspiration import InspirationTerm
from eidory.core.media_types import SUPPORTED_IMAGE_EXTENSIONS
from eidory.core.time_utils import timestamp_ns_to_iso, utc_now_iso
from eidory.models import (
    CollectionItem,
    CreativeNodeItem,
    CreativeProjectItem,
    FolderItem,
    ImageItem,
    InspirationProjectItem,
    SavedViewItem,
    TagGroupItem,
    TagItem,
    TemporaryProjectItem,
)


def _clean_optional_text(value: object, *, max_length: int) -> str | None:
    if not isinstance(value, str):
        return None
    clean = " ".join(value.strip().split())[:max_length]
    return clean or None


SEARCH_EXCLUDED_FOLDER_PREFIXES_SETTING = "search.excluded_folder_prefixes"
SEARCH_EXCLUDED_COLLECTION_IDS_SETTING = "search.excluded_collection_ids"

TEMPORARY_PROJECT_COLORS = (
    "#7A4E56",
    "#756742",
    "#4F6E5B",
    "#4C6078",
    "#67577D",
    "#735846",
    "#586E78",
    "#6F5970",
)
TEMPORARY_PROJECT_KINDS = {"semantic", "quick", "search"}

DEFAULT_AI_VISION_COLLECTION_PATHS = (
    ("黑白摄影",),
    ("创作参考",),
    ("ML-04 简单小景",),
    ("ML-05 复杂场景",),
    ("ML-06 场景带角色",),
    ("ML-07 黑白摄影",),
)

VIRTUAL_IMAGE_FILTERS = {"untagged", "un_ai_tagged", "uncategorized"}


def _clean_color_hex(value: object) -> str:
    if not isinstance(value, str):
        return ""
    clean = value.strip()
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", clean):
        return clean.upper()
    return ""


def _clean_temporary_project_kind(kind: str | None) -> str:
    clean_kind = (kind or "semantic").strip().lower()
    return clean_kind if clean_kind in TEMPORARY_PROJECT_KINDS else "semantic"


class MetadataStore:
    _connection_lock = threading.RLock()
    _busy_timeout_ms = 30_000
    _cache_size_kib = 64 * 1024
    _mmap_size_bytes = 256 * 1024 * 1024
    _sqlite_chunk_size = 900
    _wal_configured_databases: set[str] = set()

    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
        with self.connect() as conn:
            self._prepare_existing_schema_for_schema_script(conn)
            conn.executescript(schema_path.read_text(encoding="utf-8"))
            self._ensure_schema_migrations(conn)

    @staticmethod
    def _prepare_existing_schema_for_schema_script(conn: sqlite3.Connection) -> None:
        table_names = {
            str(row["name"])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "temporary_projects" in table_names:
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(temporary_projects)").fetchall()
            }
            if "summary" not in columns:
                conn.execute("ALTER TABLE temporary_projects ADD COLUMN summary TEXT NOT NULL DEFAULT ''")
            if "color_hex" not in columns:
                conn.execute("ALTER TABLE temporary_projects ADD COLUMN color_hex TEXT NOT NULL DEFAULT ''")
            if "kind" not in columns:
                conn.execute("ALTER TABLE temporary_projects ADD COLUMN kind TEXT NOT NULL DEFAULT 'semantic'")
            if "sort_order" not in columns:
                conn.execute("ALTER TABLE temporary_projects ADD COLUMN sort_order INTEGER NOT NULL DEFAULT 0")
                MetadataStore._backfill_temporary_project_sort_order(conn)
        if "creative_projects" in table_names:
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(creative_projects)").fetchall()
            }
            if "is_pinned" not in columns:
                conn.execute("ALTER TABLE creative_projects ADD COLUMN is_pinned INTEGER NOT NULL DEFAULT 0")
            if "copy_text" not in columns:
                conn.execute("ALTER TABLE creative_projects ADD COLUMN copy_text TEXT NOT NULL DEFAULT ''")
            if "sort_order" not in columns:
                conn.execute("ALTER TABLE creative_projects ADD COLUMN sort_order INTEGER NOT NULL DEFAULT 0")
                MetadataStore._backfill_creative_project_sort_order(conn)
        MetadataStore._ensure_tag_group_schema(conn)

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
        MetadataStore._ensure_tag_group_schema(conn)
        MetadataStore._ensure_remaining_schema_migrations(conn)

    @staticmethod
    def _ensure_tag_group_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tag_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        tag_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'tags'"
        ).fetchone()
        if tag_table is None:
            return
        tag_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(tags)").fetchall()
        }
        if "group_id" not in tag_columns:
            conn.execute("ALTER TABLE tags ADD COLUMN group_id INTEGER REFERENCES tag_groups(id) ON DELETE SET NULL")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_group_id ON tags(group_id)")

    @staticmethod
    def _ensure_remaining_schema_migrations(conn: sqlite3.Connection) -> None:
        for statement in (
            """
            CREATE INDEX IF NOT EXISTS idx_images_missing_name
            ON images(is_missing, file_name COLLATE NOCASE, id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_images_missing_modified
            ON images(is_missing, modified_time_ns, id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_images_missing_imported
            ON images(is_missing, imported_at, id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_images_missing_file_size
            ON images(is_missing, file_size, id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_images_missing_width
            ON images(is_missing, width, id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_images_missing_height
            ON images(is_missing, height, id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_images_missing_duration
            ON images(is_missing, duration_ms, id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_images_missing_pixels
            ON images(is_missing, (width * height), id)
            """,
        ):
            conn.execute(statement)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS image_hashes (
                image_id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                modified_time_ns INTEGER NOT NULL,
                file_sha256 TEXT NOT NULL,
                dhash TEXT NOT NULL,
                hash_source TEXT NOT NULL,
                hash_source_size INTEGER NOT NULL,
                hash_source_modified_time_ns INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_image_hashes_file_sha256
            ON image_hashes(file_sha256)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_image_hashes_dhash
            ON image_hashes(dhash)
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
                color_hex TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL DEFAULT 'semantic',
                sort_order INTEGER NOT NULL DEFAULT 0,
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
        if "color_hex" not in temporary_project_columns:
            conn.execute("ALTER TABLE temporary_projects ADD COLUMN color_hex TEXT NOT NULL DEFAULT ''")
        if "kind" not in temporary_project_columns:
            conn.execute("ALTER TABLE temporary_projects ADD COLUMN kind TEXT NOT NULL DEFAULT 'semantic'")
        if "sort_order" not in temporary_project_columns:
            conn.execute("ALTER TABLE temporary_projects ADD COLUMN sort_order INTEGER NOT NULL DEFAULT 0")
            MetadataStore._backfill_temporary_project_sort_order(conn)
        MetadataStore._backfill_temporary_project_colors(conn)
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_temporary_projects_updated_at
            ON temporary_projects(updated_at)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_temporary_projects_sort_order
            ON temporary_projects(sort_order)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_temporary_projects_kind_sort
            ON temporary_projects(kind, sort_order)
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
            CREATE TABLE IF NOT EXISTS temporary_project_board_layouts (
                project_id INTEGER PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES temporary_projects(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS temporary_project_states (
                project_id INTEGER PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES temporary_projects(id) ON DELETE CASCADE
            )
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
        MetadataStore._ensure_creative_project_schema(conn)
        MetadataStore._ensure_ai_vision_schema(conn)

    @staticmethod
    def _ensure_creative_project_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS creative_projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                brief TEXT NOT NULL DEFAULT '',
                language TEXT NOT NULL DEFAULT 'zh',
                provider_name TEXT NOT NULL DEFAULT '',
                model_name TEXT NOT NULL DEFAULT '',
                is_pinned INTEGER NOT NULL DEFAULT 0,
                copy_text TEXT NOT NULL DEFAULT '',
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        project_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(creative_projects)").fetchall()
        }
        if "is_pinned" not in project_columns:
            conn.execute("ALTER TABLE creative_projects ADD COLUMN is_pinned INTEGER NOT NULL DEFAULT 0")
        if "copy_text" not in project_columns:
            conn.execute("ALTER TABLE creative_projects ADD COLUMN copy_text TEXT NOT NULL DEFAULT ''")
        if "sort_order" not in project_columns:
            conn.execute("ALTER TABLE creative_projects ADD COLUMN sort_order INTEGER NOT NULL DEFAULT 0")
            MetadataStore._backfill_creative_project_sort_order(conn)
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_creative_projects_updated_at
            ON creative_projects(updated_at)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_creative_projects_sort_order
            ON creative_projects(sort_order)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS creative_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                parent_id INTEGER,
                title TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                search_query TEXT NOT NULL DEFAULT '',
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES creative_projects(id) ON DELETE CASCADE,
                FOREIGN KEY(parent_id) REFERENCES creative_nodes(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_creative_nodes_project_parent_order
            ON creative_nodes(project_id, parent_id, sort_order, id)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS creative_node_images (
                node_id INTEGER NOT NULL,
                image_id INTEGER NOT NULL,
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                intent_label TEXT,
                intent_query TEXT,
                PRIMARY KEY(node_id, image_id),
                FOREIGN KEY(node_id) REFERENCES creative_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_creative_node_images_node_order
            ON creative_node_images(node_id, sort_order)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS creative_board_layouts (
                project_id INTEGER PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES creative_projects(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS creative_node_board_layouts (
                node_id INTEGER PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(node_id) REFERENCES creative_nodes(id) ON DELETE CASCADE
            )
            """
        )

    @staticmethod
    def _ensure_ai_vision_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_vision_collection_rules (
                collection_id INTEGER PRIMARY KEY,
                mode TEXT NOT NULL,
                include_descendants INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(collection_id) REFERENCES collections(id) ON DELETE CASCADE,
                CHECK(mode IN ('include', 'exclude'))
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ai_vision_collection_rules_mode
            ON ai_vision_collection_rules(mode)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_vision_tags (
                image_id INTEGER PRIMARY KEY,
                provider_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                scene_location TEXT,
                environment_type TEXT,
                time_of_day TEXT,
                weather TEXT,
                shot_scale TEXT,
                view_angle TEXT,
                lighting_json TEXT NOT NULL DEFAULT '[]',
                confidence_json TEXT NOT NULL DEFAULT '{}',
                notes TEXT,
                error_message TEXT,
                source_modified_time_ns INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
                CHECK(status IN ('pending', 'processing', 'ready', 'failed', 'stale', 'skipped'))
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ai_vision_tags_status
            ON ai_vision_tags(status)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ai_vision_tags_fields
            ON ai_vision_tags(scene_location, environment_type, time_of_day, weather, shot_scale, view_angle)
            """
        )

    @contextmanager
    def connect(self, *, readonly: bool = False) -> Iterator[sqlite3.Connection]:
        with self._connection_lock:
            database_key = str(self.database_path.resolve())
            if readonly:
                database = f"{self.database_path.resolve().as_uri()}?mode=ro"
                conn = sqlite3.connect(
                    database,
                    timeout=self._busy_timeout_ms / 1000,
                    uri=True,
                )
            else:
                conn = sqlite3.connect(
                    self.database_path,
                    timeout=self._busy_timeout_ms / 1000,
                )
            conn.row_factory = sqlite3.Row
            self.configure_connection(conn, readonly=readonly, configure_wal=False)
            if not readonly:
                self._ensure_wal_configured(conn, database_key)
        try:
            yield conn
            if not readonly:
                conn.commit()
        except Exception:
            if not readonly:
                conn.rollback()
            raise
        finally:
            conn.close()

    @classmethod
    def configure_connection(
        cls,
        conn: sqlite3.Connection,
        *,
        readonly: bool = False,
        configure_wal: bool = True,
    ) -> None:
        conn.execute(f"PRAGMA busy_timeout = {cls._busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys = ON")
        if not readonly and configure_wal:
            conn.execute("PRAGMA journal_mode = WAL")
        if not readonly:
            conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute(f"PRAGMA cache_size = -{cls._cache_size_kib}")
        conn.execute(f"PRAGMA mmap_size = {cls._mmap_size_bytes}")

    @classmethod
    def _ensure_wal_configured(cls, conn: sqlite3.Connection, database_key: str) -> None:
        if database_key in cls._wal_configured_databases:
            return
        conn.execute("PRAGMA journal_mode = WAL")
        cls._wal_configured_databases.add(database_key)

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
        with self.connect(readonly=True) as conn:
            return [self._folder_from_row(row) for row in conn.execute(sql, params)]

    def list_folders_with_collection_images(self) -> list[FolderItem]:
        with self.connect(readonly=True) as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT f.*
                FROM folders f
                JOIN images i ON i.folder_id = f.id
                JOIN image_collections ic ON ic.image_id = i.id
                WHERE f.is_active = 1
                  AND i.is_missing = 0
                ORDER BY f.folder_path COLLATE NOCASE
                """
            ).fetchall()
        return [self._folder_from_row(row) for row in rows]

    def folder_subtree_counts(self) -> list[tuple[FolderItem, dict[str, int]]]:
        with self.connect(readonly=True) as conn:
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

    def get_folder_by_path(self, folder_path: str) -> FolderItem | None:
        normalized = self._normalize_folder_prefix(folder_path)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM folders WHERE folder_path = ? AND is_active = 1",
                (normalized,),
            ).fetchone()
            return self._folder_from_row(row) if row else None

    def remove_unclassified_active_roots(self) -> tuple[list[str], int, int]:
        """Remove stale physical scan roots that no longer map to Eidory folders.

        Imported files are only part of the managed library after they are linked
        through image_collections. Old active scan roots with no such links can be
        resurrected by file watching or manual rescans, so they must not remain as
        active roots.
        """
        with self.connect() as conn:
            folder_rows = conn.execute(
                """
                SELECT f.id
                FROM folders f
                WHERE f.is_active = 1
                  AND NOT EXISTS (
                      SELECT 1
                      FROM images i
                      JOIN image_collections ic ON ic.image_id = i.id
                      WHERE i.folder_id = f.id
                  )
                """
            ).fetchall()
            folder_ids = [int(row["id"]) for row in folder_rows]
            if not folder_ids:
                return [], 0, 0

            placeholders = ",".join("?" for _ in folder_ids)
            image_rows = conn.execute(
                f"""
                SELECT thumbnail_path
                FROM images
                WHERE folder_id IN ({placeholders})
                  AND thumbnail_path IS NOT NULL
                """,
                tuple(folder_ids),
            ).fetchall()
            count_row = conn.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM images
                WHERE folder_id IN ({placeholders})
                """,
                tuple(folder_ids),
            ).fetchone()
            thumbnail_paths = [str(row["thumbnail_path"]) for row in image_rows]
            cur = conn.execute(
                f"DELETE FROM folders WHERE id IN ({placeholders})",
                tuple(folder_ids),
            )
            self._delete_unused_tags(conn)
            return thumbnail_paths, int(cur.rowcount), int(count_row["count"])

    def path_remap_candidates(self) -> list[str]:
        candidates: set[str] = set()
        with self.connect() as conn:
            folder_rows = conn.execute(
                "SELECT folder_path FROM folders WHERE is_active = 1"
            ).fetchall()
            missing_rows = conn.execute(
                "SELECT file_path FROM images WHERE is_missing = 1"
            ).fetchall()

        for row in folder_rows:
            candidates.add(self._normalize_folder_prefix(str(row["folder_path"])))
        for row in missing_rows:
            candidates.add(
                self._normalize_folder_prefix(os.path.dirname(str(row["file_path"])))
            )
        return sorted(candidates, key=str.casefold)

    def path_prefix_match_counts(self, old_prefix: str) -> dict[str, int]:
        normalized = self._normalize_folder_prefix(old_prefix)
        like = self._folder_path_like(normalized)
        with self.connect() as conn:
            folder_row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM folders
                WHERE folder_path = ?
                   OR folder_path LIKE ? ESCAPE '\\'
                """,
                (normalized, like),
            ).fetchone()
            image_row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM images
                WHERE file_path LIKE ? ESCAPE '\\'
                """,
                (like,),
            ).fetchone()
            missing_row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM images
                WHERE file_path LIKE ? ESCAPE '\\'
                  AND is_missing = 1
                """,
                (like,),
            ).fetchone()
        return {
            "folders": int(folder_row["count"]),
            "images": int(image_row["count"]),
            "missing": int(missing_row["count"]),
        }

    def remap_path_prefix(self, old_prefix: str, new_prefix: str) -> dict[str, int]:
        old_normalized = self._normalize_folder_prefix(old_prefix)
        new_normalized = self._normalize_folder_prefix(new_prefix)
        if old_normalized == new_normalized:
            raise ValueError("old and new paths are the same")
        if not os.path.isdir(new_normalized):
            raise FileNotFoundError(f"new path does not exist: {new_normalized}")

        now = utc_now_iso()
        folder_like = self._folder_path_like(old_normalized)
        image_like = self._folder_path_like(old_normalized)
        counts = {
            "folders_updated": 0,
            "folders_merged": 0,
            "images_updated": 0,
            "relinked": 0,
            "still_missing": 0,
            "conflicts": 0,
        }

        with self.connect() as conn:
            folder_rows = conn.execute(
                """
                SELECT id, folder_path
                FROM folders
                WHERE folder_path = ?
                   OR folder_path LIKE ? ESCAPE '\\'
                ORDER BY LENGTH(folder_path), folder_path
                """,
                (old_normalized, folder_like),
            ).fetchall()
            for row in folder_rows:
                folder_id = int(row["id"])
                old_folder_path = str(row["folder_path"])
                new_folder_path = self._replace_path_prefix(
                    old_folder_path,
                    old_normalized,
                    new_normalized,
                )
                if new_folder_path is None or new_folder_path == old_folder_path:
                    continue
                existing = conn.execute(
                    "SELECT id FROM folders WHERE folder_path = ? AND id != ?",
                    (new_folder_path, folder_id),
                ).fetchone()
                if existing is not None:
                    target_id = int(existing["id"])
                    conn.execute(
                        "UPDATE images SET folder_id = ? WHERE folder_id = ?",
                        (target_id, folder_id),
                    )
                    conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
                    counts["folders_merged"] += 1
                    continue
                conn.execute(
                    """
                    UPDATE folders
                    SET folder_path = ?, is_active = 1
                    WHERE id = ?
                    """,
                    (new_folder_path, folder_id),
                )
                counts["folders_updated"] += 1

            image_rows = conn.execute(
                """
                SELECT id, file_path, file_size, modified_time_ns, is_missing
                FROM images
                WHERE file_path LIKE ? ESCAPE '\\'
                ORDER BY file_path
                """,
                (image_like,),
            ).fetchall()
            for row in image_rows:
                image_id = int(row["id"])
                old_file_path = str(row["file_path"])
                new_file_path = self._replace_path_prefix(
                    old_file_path,
                    old_normalized,
                    new_normalized,
                )
                if new_file_path is None or new_file_path == old_file_path:
                    continue
                existing = conn.execute(
                    "SELECT id FROM images WHERE file_path = ? AND id != ?",
                    (new_file_path, image_id),
                ).fetchone()
                if existing is not None:
                    counts["conflicts"] += 1
                    continue

                file_name = os.path.basename(new_file_path)
                file_ext = Path(file_name).suffix.lower()
                try:
                    stat = os.stat(new_file_path, follow_symlinks=False)
                    exists = os.path.isfile(new_file_path)
                except OSError:
                    stat = None
                    exists = False

                if exists and stat is not None:
                    file_size = int(stat.st_size)
                    modified_time_ns = int(stat.st_mtime_ns)
                    modified_at = timestamp_ns_to_iso(modified_time_ns)
                    changed = (
                        int(row["file_size"]) != file_size
                        or int(row["modified_time_ns"]) != modified_time_ns
                        or bool(row["is_missing"])
                    )
                    conn.execute(
                        """
                        UPDATE images
                        SET file_path = ?, file_name = ?, file_ext = ?, file_size = ?,
                            modified_at = ?, modified_time_ns = ?, last_seen_at = ?,
                            is_missing = 0
                        WHERE id = ?
                        """,
                        (
                            new_file_path,
                            file_name,
                            file_ext,
                            file_size,
                            modified_at,
                            modified_time_ns,
                            now,
                            image_id,
                        ),
                    )
                    counts["relinked"] += 1
                    if changed:
                        self._mark_image_media_stale(conn, image_id, now)
                else:
                    conn.execute(
                        """
                        UPDATE images
                        SET file_path = ?, file_name = ?, file_ext = ?,
                            last_seen_at = ?, is_missing = 1
                        WHERE id = ?
                        """,
                        (new_file_path, file_name, file_ext, now, image_id),
                    )
                    counts["still_missing"] += 1
                counts["images_updated"] += 1

        return counts

    def list_collections(self) -> list[CollectionItem]:
        with self.connect(readonly=True) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM collections
                ORDER BY parent_id IS NOT NULL, parent_id, sort_order, name COLLATE NOCASE
                """
            ).fetchall()
        return [self._collection_from_row(row) for row in rows]

    def collection_export_paths(self) -> list[tuple[CollectionItem, tuple[str, ...]]]:
        collections = self.list_collections()
        by_id = {collection.id: collection for collection in collections}
        path_cache: dict[int, tuple[str, ...]] = {}

        def path_for(collection_id: int) -> tuple[str, ...]:
            cached = path_cache.get(collection_id)
            if cached is not None:
                return cached
            collection = by_id[collection_id]
            if collection.parent_id is None or collection.parent_id not in by_id:
                path = (collection.name,)
            else:
                path = (*path_for(collection.parent_id), collection.name)
            path_cache[collection_id] = path
            return path

        return [(collection, path_for(collection.id)) for collection in collections]

    def list_images_for_collection_direct(self, collection_id: int) -> list[ImageItem]:
        with self.connect(readonly=True) as conn:
            rows = conn.execute(
                """
                SELECT i.*
                FROM images i
                JOIN image_collections ic ON ic.image_id = i.id
                WHERE ic.collection_id = ?
                  AND i.is_missing = 0
                ORDER BY i.file_name COLLATE NOCASE, i.id
                """,
                (collection_id,),
            ).fetchall()
        return [self._image_from_row(row) for row in rows]

    def list_images_without_collections(self) -> list[ImageItem]:
        with self.connect(readonly=True) as conn:
            rows = conn.execute(
                """
                SELECT i.*
                FROM images i
                WHERE i.is_missing = 0
                  AND NOT EXISTS (
                      SELECT 1
                      FROM image_collections ic
                      WHERE ic.image_id = i.id
                  )
                ORDER BY i.file_name COLLATE NOCASE, i.id
                """
            ).fetchall()
        return [self._image_from_row(row) for row in rows]

    def list_collections_with_counts(self) -> list[tuple[CollectionItem, int]]:
        collections = self.list_collections()
        counts = self.collection_image_counts()
        return [(collection, counts.get(collection.id, 0)) for collection in collections]

    def collection_image_counts(self) -> dict[int, int]:
        collections = self.list_collections()
        counts = {collection.id: 0 for collection in collections}
        with self.connect(readonly=True) as conn:
            rows = conn.execute(
                """
                WITH RECURSIVE collection_ancestors(collection_id, ancestor_id) AS (
                    SELECT id, id
                    FROM collections
                    UNION ALL
                    SELECT ca.collection_id, c.parent_id
                    FROM collection_ancestors ca
                    JOIN collections c ON c.id = ca.ancestor_id
                    WHERE c.parent_id IS NOT NULL
                )
                SELECT ca.ancestor_id AS collection_id, COUNT(DISTINCT ic.image_id) AS count
                FROM collection_ancestors ca
                JOIN image_collections ic ON ic.collection_id = ca.collection_id
                JOIN images i ON i.id = ic.image_id
                WHERE i.is_missing = 0
                GROUP BY ca.ancestor_id
                """
            ).fetchall()
        for row in rows:
            collection_id = int(row["collection_id"])
            if collection_id in counts:
                counts[collection_id] = int(row["count"])
        return counts

    def collection_paths_for_image(self, image_id: int) -> list[str]:
        collections = self.list_collections()
        by_id = {collection.id: collection for collection in collections}

        def path_for(collection_id: int) -> str:
            parts: list[str] = []
            current = by_id.get(collection_id)
            seen: set[int] = set()
            while current is not None and current.id not in seen:
                seen.add(current.id)
                parts.append(current.name)
                current = by_id.get(current.parent_id) if current.parent_id is not None else None
            return " / ".join(reversed(parts))

        with self.connect(readonly=True) as conn:
            rows = conn.execute(
                """
                SELECT collection_id
                FROM image_collections
                WHERE image_id = ?
                ORDER BY collection_id
                """,
                (image_id,),
            ).fetchall()
        return [
            path_for(int(row["collection_id"]))
            for row in rows
            if int(row["collection_id"]) in by_id
        ]

    def collection_chains_for_image(self, image_id: int) -> list[list[CollectionItem]]:
        collections = self.list_collections()
        by_id = {collection.id: collection for collection in collections}

        def chain_for(collection_id: int) -> list[CollectionItem]:
            chain: list[CollectionItem] = []
            current = by_id.get(collection_id)
            seen: set[int] = set()
            while current is not None and current.id not in seen:
                seen.add(current.id)
                chain.append(current)
                current = by_id.get(current.parent_id) if current.parent_id is not None else None
            return list(reversed(chain))

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT collection_id
                FROM image_collections
                WHERE image_id = ?
                ORDER BY collection_id
                """,
                (image_id,),
            ).fetchall()
        return [
            chain
            for row in rows
            if (chain := chain_for(int(row["collection_id"])))
        ]

    def collection_descendant_ids(self, collection_id: int) -> list[int]:
        with self.connect(readonly=True) as conn:
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
            self._delete_empty_folders(conn)
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
            self._delete_empty_folders(conn)
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
            self._delete_empty_folders(conn)
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

    @staticmethod
    def _delete_empty_folders(conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            """
            DELETE FROM folders
            WHERE NOT EXISTS (
                SELECT 1
                FROM images i
                WHERE i.folder_id = folders.id
            )
            """
        )
        return int(cur.rowcount)

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
        with self.connect() as conn:
            return self._upsert_image_on_connection(
                conn,
                now=now,
                folder_id=folder_id,
                file_path=file_path,
                file_size=file_size,
                width=width,
                height=height,
                created_time_ns=created_time_ns,
                modified_time_ns=modified_time_ns,
                duration_ms=duration_ms,
            )

    def upsert_images(self, records: Sequence[Mapping[str, object]]) -> list[tuple[int, str]]:
        if not records:
            return []
        now = utc_now_iso()
        results: list[tuple[int, str]] = []
        with self.connect() as conn:
            for record in records:
                results.append(
                    self._upsert_image_on_connection(
                        conn,
                        now=now,
                        folder_id=int(record["folder_id"]),
                        file_path=str(record["file_path"]),
                        file_size=int(record["file_size"]),
                        width=self._optional_int(record.get("width")),
                        height=self._optional_int(record.get("height")),
                        created_time_ns=self._optional_int(record.get("created_time_ns")),
                        modified_time_ns=int(record["modified_time_ns"]),
                        duration_ms=self._optional_int(record.get("duration_ms")),
                    )
                )
        return results

    def _upsert_image_on_connection(
        self,
        conn: sqlite3.Connection,
        *,
        now: str,
        folder_id: int,
        file_path: str,
        file_size: int,
        width: int | None,
        height: int | None,
        created_time_ns: int | None,
        modified_time_ns: int,
        duration_ms: int | None,
    ) -> tuple[int, str]:
        file_name = os.path.basename(file_path)
        file_ext = Path(file_name).suffix.lower()
        created_at = timestamp_ns_to_iso(created_time_ns) if created_time_ns else None
        modified_at = timestamp_ns_to_iso(modified_time_ns)

        cur = conn.execute(
            """
            INSERT INTO images(
                folder_id, file_path, file_name, file_ext, file_size, width, height,
                duration_ms,
                created_at, modified_at, modified_time_ns, imported_at, last_seen_at,
                thumbnail_status, embedding_status, is_missing
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 'pending', 0)
            ON CONFLICT(file_path) DO NOTHING
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
        if int(cur.rowcount) > 0:
            return int(cur.lastrowid), "new"

        existing = conn.execute(
            "SELECT * FROM images WHERE file_path = ?", (file_path,)
        ).fetchone()
        if existing is None:
            raise RuntimeError("image upsert conflict could not be resolved")

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
            self._mark_image_media_stale(conn, int(existing["id"]), now)
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

    def update_thumbnails(self, records: Sequence[tuple[int, str | None, str]]) -> None:
        if not records:
            return
        with self.connect() as conn:
            conn.executemany(
                """
                UPDATE images
                SET thumbnail_path = ?, thumbnail_status = ?
                WHERE id = ?
                """,
                ((thumbnail_path, status, image_id) for image_id, thumbnail_path, status in records),
            )

    def update_image_path_after_rename(
        self,
        image_id: int,
        *,
        file_path: str,
        file_size: int,
        modified_time_ns: int,
    ) -> None:
        normalized = os.path.abspath(os.path.expanduser(file_path))
        file_name = os.path.basename(normalized)
        file_ext = Path(file_name).suffix.lower()
        modified_at = timestamp_ns_to_iso(modified_time_ns)
        now = utc_now_iso()
        with self.connect() as conn:
            conflict = conn.execute(
                """
                SELECT id
                FROM images
                WHERE file_path = ?
                  AND id != ?
                """,
                (normalized, image_id),
            ).fetchone()
            if conflict is not None:
                raise ValueError("another image already uses this path")
            cur = conn.execute(
                """
                UPDATE images
                SET file_path = ?,
                    file_name = ?,
                    file_ext = ?,
                    file_size = ?,
                    modified_at = ?,
                    modified_time_ns = ?,
                    last_seen_at = ?,
                    is_missing = 0
                WHERE id = ?
                """,
                (
                    normalized,
                    file_name,
                    file_ext,
                    file_size,
                    modified_at,
                    modified_time_ns,
                    now,
                    image_id,
                ),
            )
            if int(cur.rowcount) == 0:
                raise ValueError("image not found")

    def repair_missing_image_path(
        self,
        image_id: int,
        *,
        file_path: str,
        file_size: int,
        width: int | None,
        height: int | None,
        modified_time_ns: int,
        duration_ms: int | None = None,
    ) -> None:
        normalized = os.path.abspath(os.path.expanduser(file_path))
        file_name = os.path.basename(normalized)
        file_ext = Path(file_name).suffix.lower()
        modified_at = timestamp_ns_to_iso(modified_time_ns)
        now = utc_now_iso()
        with self.connect() as conn:
            current = conn.execute(
                """
                SELECT id, file_size, modified_time_ns, is_missing
                FROM images
                WHERE id = ?
                """,
                (image_id,),
            ).fetchone()
            if current is None:
                raise ValueError("image not found")
            conflict = conn.execute(
                """
                SELECT id
                FROM images
                WHERE file_path = ?
                  AND id != ?
                """,
                (normalized, image_id),
            ).fetchone()
            if conflict is not None:
                raise ValueError("another image already uses this path")

            changed = (
                int(current["file_size"]) != int(file_size)
                or int(current["modified_time_ns"]) != int(modified_time_ns)
                or bool(current["is_missing"])
            )
            conn.execute(
                """
                UPDATE images
                SET file_path = ?,
                    file_name = ?,
                    file_ext = ?,
                    file_size = ?,
                    width = ?,
                    height = ?,
                    duration_ms = ?,
                    modified_at = ?,
                    modified_time_ns = ?,
                    last_seen_at = ?,
                    is_missing = 0
                WHERE id = ?
                """,
                (
                    normalized,
                    file_name,
                    file_ext,
                    file_size,
                    width,
                    height,
                    duration_ms,
                    modified_at,
                    modified_time_ns,
                    now,
                    image_id,
                ),
            )
            if changed:
                self._mark_image_media_stale(conn, image_id, now)

    def mark_embedding_not_required(self, image_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE images SET embedding_status = 'ready' WHERE id = ?",
                (image_id,),
            )

    def mark_embeddings_not_required(self, image_ids: Sequence[int]) -> None:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return
        with self.connect() as conn:
            conn.executemany(
                "UPDATE images SET embedding_status = 'ready' WHERE id = ?",
                ((image_id,) for image_id in clean_ids),
            )

    def thumbnail_needs_generation(self, image_id: int) -> bool:
        return image_id in self.thumbnail_ids_needing_generation([image_id])

    def thumbnail_ids_needing_generation(self, image_ids: Sequence[int]) -> set[int]:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return set()
        rows: list[sqlite3.Row] = []
        with self.connect(readonly=True) as conn:
            for chunk in self._chunks(clean_ids, self._sqlite_chunk_size):
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    conn.execute(
                        f"""
                        SELECT id, thumbnail_status, thumbnail_path
                        FROM images
                        WHERE id IN ({placeholders})
                        """,
                        tuple(chunk),
                    ).fetchall()
                )
        needed: set[int] = set()
        for row in rows:
            image_id = int(row["id"])
            if str(row["thumbnail_status"]) != "ready":
                needed.add(image_id)
                continue
            thumbnail_path = row["thumbnail_path"]
            if not thumbnail_path or not Path(str(thumbnail_path)).exists():
                needed.add(image_id)
        return needed

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

    def remove_unseen_images_for_folder(
        self,
        folder_id: int,
        seen_paths: Iterable[str],
    ) -> tuple[list[str], int]:
        with self.connect() as conn:
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS seen_scan_paths(path TEXT PRIMARY KEY)")
            conn.execute("DELETE FROM seen_scan_paths")
            conn.executemany(
                "INSERT OR IGNORE INTO seen_scan_paths(path) VALUES (?)",
                ((path,) for path in seen_paths),
            )
            rows = conn.execute(
                """
                SELECT id, thumbnail_path
                FROM images
                WHERE folder_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM seen_scan_paths s WHERE s.path = images.file_path
                  )
                """,
                (folder_id,),
            ).fetchall()
            image_ids = [int(row["id"]) for row in rows]
            thumbnail_paths = [
                str(row["thumbnail_path"])
                for row in rows
                if row["thumbnail_path"] is not None
            ]
            if image_ids:
                placeholders = ",".join("?" for _ in image_ids)
                conn.execute(
                    f"DELETE FROM images WHERE id IN ({placeholders})",
                    tuple(image_ids),
                )
            conn.execute("DELETE FROM seen_scan_paths")
            self._delete_unused_tags(conn)
            return thumbnail_paths, len(image_ids)

    def mark_image_missing(self, image_id: int) -> bool:
        now = utc_now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                """
                UPDATE images
                SET is_missing = 1, last_seen_at = ?
                WHERE id = ?
                  AND is_missing = 0
                """,
                (now, image_id),
            )
            return int(cur.rowcount) > 0

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
        virtual_filter: str | None = None,
        sort_key: str = "default",
        sort_desc: bool = True,
        excluded_folder_path_prefixes: Sequence[str] | None = None,
        excluded_collection_ids: Sequence[int] | None = None,
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
        virtual_clause, virtual_params = self._virtual_image_filter_clause(
            "images.id",
            "images.file_ext",
            virtual_filter,
        )
        if virtual_clause:
            clauses.append(virtual_clause)
            params.extend(virtual_params)
        self._append_excluded_folder_path_clauses(
            clauses,
            params,
            "images.file_path",
            excluded_folder_path_prefixes,
        )
        self._append_excluded_collection_clauses(
            clauses,
            params,
            "images.id",
            excluded_collection_ids,
        )
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
        with self.connect(readonly=True) as conn:
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

    @staticmethod
    def _virtual_image_filter_clause(
        image_id_expr: str,
        file_ext_expr: str,
        virtual_filter: str | None,
    ) -> tuple[str | None, list[object]]:
        if not virtual_filter:
            return None, []
        if virtual_filter not in VIRTUAL_IMAGE_FILTERS:
            raise ValueError(f"unknown virtual image filter: {virtual_filter}")
        if virtual_filter == "untagged":
            return (
                f"""
                NOT EXISTS (
                    SELECT 1
                    FROM image_tags it
                    WHERE it.image_id = {image_id_expr}
                )
                """,
                [],
            )
        if virtual_filter == "un_ai_tagged":
            image_placeholders = ",".join("?" for _ in SUPPORTED_IMAGE_EXTENSIONS)
            return (
                f"""
                {file_ext_expr} IN ({image_placeholders})
                AND NOT EXISTS (
                    SELECT 1
                    FROM ai_vision_tags avt
                    WHERE avt.image_id = {image_id_expr}
                      AND avt.status = 'ready'
                )
                """,
                list(sorted(SUPPORTED_IMAGE_EXTENSIONS)),
            )
        return (
            f"""
            NOT EXISTS (
                SELECT 1
                FROM image_collections ic
                WHERE ic.image_id = {image_id_expr}
            )
            """,
            [],
        )

    def count_images(self) -> int:
        with self.connect(readonly=True) as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM images").fetchone()
            return int(row["c"])

    def near_duplicate_metadata_candidates(
        self,
        *,
        width: int | None,
        height: int | None,
        file_size: int,
        limit: int = 400,
    ) -> list[ImageItem]:
        clauses = ["is_missing = 0"]
        params: list[object] = []
        image_placeholders = ",".join("?" for _ in SUPPORTED_IMAGE_EXTENSIONS)
        clauses.append(f"file_ext IN ({image_placeholders})")
        params.extend(sorted(SUPPORTED_IMAGE_EXTENSIONS))

        source_area = int(width or 0) * int(height or 0)
        size_low = max(0, int(file_size * 0.45)) if file_size > 0 else 0
        size_high = int(file_size * 2.2) if file_size > 0 else 0
        if width and height:
            aspect_tolerance = max(int(source_area * 0.04), 1)
            clauses.append(
                """
                (
                    (width = ? AND height = ?)
                    OR (
                        width IS NOT NULL
                        AND height IS NOT NULL
                        AND ABS((width * ?) - (height * ?)) <= ?
                    )
                    OR (file_size BETWEEN ? AND ?)
                )
                """
            )
            params.extend([width, height, height, width, aspect_tolerance, size_low, size_high])
        elif file_size > 0:
            clauses.append("file_size BETWEEN ? AND ?")
            params.extend([size_low, size_high])
        else:
            return []

        params.extend([
            width or -1,
            height or -1,
            file_size,
            source_area,
            max(1, int(limit)),
        ])
        with self.connect(readonly=True) as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM images
                WHERE {' AND '.join(clauses)}
                ORDER BY
                    CASE WHEN width = ? AND height = ? THEN 0 ELSE 1 END,
                    ABS(file_size - ?),
                    ABS((COALESCE(width, 0) * COALESCE(height, 0)) - ?),
                    id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._image_from_row(row) for row in rows]

    def count_missing_images(self) -> int:
        with self.connect(readonly=True) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM images WHERE is_missing = 1"
            ).fetchone()
            return int(row["c"])

    def count_images_for_virtual_filter(self, virtual_filter: str) -> int:
        virtual_clause, virtual_params = self._virtual_image_filter_clause(
            "images.id",
            "images.file_ext",
            virtual_filter,
        )
        clauses = ["images.is_missing = 0"]
        params: list[object] = []
        if virtual_clause:
            clauses.append(virtual_clause)
            params.extend(virtual_params)
        with self.connect(readonly=True) as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS c
                FROM images
                WHERE {' AND '.join(clauses)}
                """,
                params,
            ).fetchone()
        return int(row["c"])

    def virtual_image_filter_counts(self) -> dict[str, int]:
        return {
            virtual_filter: self.count_images_for_virtual_filter(virtual_filter)
            for virtual_filter in VIRTUAL_IMAGE_FILTERS
        }

    def image_ids_for_virtual_filter(self, virtual_filter: str) -> set[int]:
        virtual_clause, virtual_params = self._virtual_image_filter_clause(
            "images.id",
            "images.file_ext",
            virtual_filter,
        )
        clauses = ["images.is_missing = 0"]
        params: list[object] = []
        if virtual_clause:
            clauses.append(virtual_clause)
            params.extend(virtual_params)
        with self.connect(readonly=True) as conn:
            rows = conn.execute(
                f"""
                SELECT images.id
                FROM images
                WHERE {' AND '.join(clauses)}
                """,
                params,
            ).fetchall()
        return {int(row["id"]) for row in rows}

    def get_image(self, image_id: int) -> ImageItem | None:
        with self.connect(readonly=True) as conn:
            row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
            return self._image_from_row(row) if row else None

    def get_image_by_path(self, file_path: str | Path) -> ImageItem | None:
        normalized = os.path.abspath(os.path.expanduser(str(file_path)))
        with self.connect(readonly=True) as conn:
            row = conn.execute(
                "SELECT * FROM images WHERE file_path = ? AND is_missing = 0",
                (normalized,),
            ).fetchone()
            return self._image_from_row(row) if row else None

    def images_by_ids(self, image_ids: Sequence[int], scores: dict[int, float] | None = None) -> list[ImageItem]:
        if not image_ids:
            return []
        unique_ids = list(dict.fromkeys(int(image_id) for image_id in image_ids))
        rows: list[sqlite3.Row] = []
        with self.connect(readonly=True) as conn:
            for chunk in self._chunks(unique_ids, self._sqlite_chunk_size):
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    conn.execute(
                        f"SELECT * FROM images WHERE id IN ({placeholders})",
                        tuple(chunk),
                    ).fetchall()
                )
        by_id = {int(row["id"]): self._image_from_row(row, scores.get(int(row["id"])) if scores else None) for row in rows}
        return [by_id[image_id] for image_id in image_ids if image_id in by_id]

    def image_hash_records(self, image_ids: Sequence[int]) -> dict[int, ImageHashCacheRecord]:
        clean_ids = sorted({int(image_id) for image_id in image_ids})
        if not clean_ids:
            return {}
        rows: list[sqlite3.Row] = []
        with self.connect(readonly=True) as conn:
            for chunk in self._chunks(clean_ids, self._sqlite_chunk_size):
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    conn.execute(
                        f"""
                        SELECT *
                        FROM image_hashes
                        WHERE image_id IN ({placeholders})
                        """,
                        tuple(chunk),
                    ).fetchall()
                )
        records: dict[int, ImageHashCacheRecord] = {}
        for row in rows:
            try:
                dhash = int(str(row["dhash"]), 16)
            except (TypeError, ValueError):
                continue
            image_id = int(row["image_id"])
            records[image_id] = ImageHashCacheRecord(
                image_id=image_id,
                file_path=str(row["file_path"]),
                file_size=int(row["file_size"]),
                modified_time_ns=int(row["modified_time_ns"]),
                file_sha256=str(row["file_sha256"]),
                dhash=dhash,
                hash_source=str(row["hash_source"]),
                hash_source_size=int(row["hash_source_size"]),
                hash_source_modified_time_ns=int(row["hash_source_modified_time_ns"]),
            )
        return records

    def upsert_image_hash_record(self, record: ImageHashCacheRecord) -> None:
        self.upsert_image_hash_records([record])

    def upsert_image_hash_records(self, records: Sequence[ImageHashCacheRecord]) -> None:
        clean_records = list(records)
        if not clean_records:
            return
        now = utc_now_iso()
        rows = [
            (
                int(record.image_id),
                record.file_path,
                int(record.file_size),
                int(record.modified_time_ns),
                record.file_sha256,
                f"{int(record.dhash):016x}",
                record.hash_source,
                int(record.hash_source_size),
                int(record.hash_source_modified_time_ns),
                now,
            )
            for record in clean_records
        ]
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO image_hashes(
                    image_id, file_path, file_size, modified_time_ns,
                    file_sha256, dhash, hash_source, hash_source_size,
                    hash_source_modified_time_ns, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    file_size = excluded.file_size,
                    modified_time_ns = excluded.modified_time_ns,
                    file_sha256 = excluded.file_sha256,
                    dhash = excluded.dhash,
                    hash_source = excluded.hash_source,
                    hash_source_size = excluded.hash_source_size,
                    hash_source_modified_time_ns = excluded.hash_source_modified_time_ns,
                    updated_at = excluded.updated_at
                """,
                rows,
            )

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
        virtual_filter: str | None = None,
        excluded_folder_path_prefixes: Sequence[str] | None = None,
        excluded_collection_ids: Sequence[int] | None = None,
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
        virtual_clause, virtual_params = self._virtual_image_filter_clause(
            "images.id",
            "images.file_ext",
            virtual_filter,
        )
        if virtual_clause:
            clauses.append(virtual_clause)
            params.extend(virtual_params)
        self._append_excluded_folder_path_clauses(
            clauses,
            params,
            "images.file_path",
            excluded_folder_path_prefixes,
        )
        self._append_excluded_collection_clauses(
            clauses,
            params,
            "images.id",
            excluded_collection_ids,
        )
        if status_filter == "favorite":
            clauses.append("images.is_favorite = 1")
        elif status_filter == "unindexed":
            clauses.append("images.embedding_status != 'ready'")

        with self.connect(readonly=True) as conn:
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
        return image_id in self.color_feature_ids_needing_generation(
            [image_id],
            vector_version=vector_version,
            vector_dim=vector_dim,
        )

    def color_feature_ids_needing_generation(
        self,
        image_ids: Sequence[int],
        *,
        vector_version: str = COLOR_VECTOR_VERSION,
        vector_dim: int = COLOR_VECTOR_DIM,
    ) -> set[int]:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return set()
        rows: list[sqlite3.Row] = []
        with self.connect(readonly=True) as conn:
            for chunk in self._chunks(clean_ids, self._sqlite_chunk_size):
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    conn.execute(
                        f"""
                        SELECT image_id, vector_version, vector_dim, hist_blob, status
                        FROM color_features
                        WHERE image_id IN ({placeholders})
                        """,
                        tuple(chunk),
                    ).fetchall()
                )
        rows_by_id = {int(row["image_id"]): row for row in rows}
        needed: set[int] = set()
        for image_id in clean_ids:
            row = rows_by_id.get(image_id)
            if row is None:
                needed.add(image_id)
                continue
            if (
                str(row["vector_version"]) != vector_version
                or int(row["vector_dim"]) != vector_dim
                or str(row["status"]) != "ready"
                or row["hist_blob"] is None
            ):
                needed.add(image_id)
        return needed

    def upsert_color_feature_success(
        self,
        *,
        image_id: int,
        vector: np.ndarray,
        vector_version: str = COLOR_VECTOR_VERSION,
    ) -> None:
        self.upsert_color_feature_successes(
            [(image_id, vector)],
            vector_version=vector_version,
        )

    def upsert_color_feature_successes(
        self,
        records: Sequence[tuple[int, np.ndarray]],
        *,
        vector_version: str = COLOR_VECTOR_VERSION,
    ) -> None:
        if not records:
            return
        rows: list[tuple[int, str, int, bytes, str]] = []
        now = utc_now_iso()
        for image_id, vector in records:
            normalized = np.asarray(vector, dtype=np.float32)
            if normalized.ndim != 1:
                raise ValueError("color vector must be one-dimensional")
            dim = int(normalized.shape[0])
            rows.append((int(image_id), vector_version, dim, normalized.tobytes(), now))
        with self.connect() as conn:
            conn.executemany(
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
                rows,
            )

    def mark_color_features_failed(self, records: Sequence[tuple[int, str]]) -> None:
        if not records:
            return
        now = utc_now_iso()
        with self.connect() as conn:
            conn.executemany(
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
                    (
                        int(image_id),
                        COLOR_VECTOR_VERSION,
                        COLOR_VECTOR_DIM,
                        str(error_message)[:2000],
                        now,
                    )
                    for image_id, error_message in records
                ),
            )

    def mark_color_feature_failed(self, image_id: int, error_message: str) -> None:
        self.mark_color_features_failed([(image_id, error_message)])

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
        rows: list[sqlite3.Row] = []
        with self.connect(readonly=True) as conn:
            for chunk in self._chunks(clean_ids, self._sqlite_chunk_size):
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    conn.execute(
                        f"""
                        SELECT image_id, hist_blob
                        FROM color_features
                        WHERE image_id IN ({placeholders})
                          AND vector_version = ?
                          AND vector_dim = ?
                          AND status = 'ready'
                          AND hist_blob IS NOT NULL
                        """,
                        (*chunk, vector_version, vector_dim),
                    ).fetchall()
                )
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
        status_placeholders = ",".join("?" for _ in clean_statuses)
        rows: list[sqlite3.Row] = []
        with self.connect(readonly=True) as conn:
            for chunk in self._chunks(clean_ids, self._sqlite_chunk_size):
                id_placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    conn.execute(
                        f"""
                        SELECT image_id
                        FROM color_features
                        WHERE image_id IN ({id_placeholders})
                          AND status IN ({status_placeholders})
                          AND vector_version = ?
                          AND vector_dim = ?
                        """,
                        (*chunk, *clean_statuses, vector_version, vector_dim),
                    ).fetchall()
                )
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

    def list_tag_groups(self) -> list[TagGroupItem]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM tag_groups
                ORDER BY sort_order, name COLLATE NOCASE
                """
            ).fetchall()
        return [self._tag_group_from_row(row) for row in rows]

    def create_tag_group(self, name: str) -> int:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("tag group name must not be empty")
        now = utc_now_iso()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT id FROM tag_groups WHERE name = ?", (clean_name,)
            ).fetchone()
            if existing is not None:
                raise ValueError("tag group name already exists")
            row = conn.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_order FROM tag_groups"
            ).fetchone()
            sort_order = int(row["next_order"] or 0)
            cur = conn.execute(
                """
                INSERT INTO tag_groups(name, sort_order, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (clean_name, sort_order, now, now),
            )
            return int(cur.lastrowid)

    def rename_tag_group(self, group_id: int, new_name: str) -> bool:
        clean_name = new_name.strip()
        if not clean_name:
            raise ValueError("tag group name must not be empty")
        now = utc_now_iso()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT id FROM tag_groups WHERE name = ?", (clean_name,)
            ).fetchone()
            if existing is not None and int(existing["id"]) != group_id:
                raise ValueError("tag group name already exists")
            cur = conn.execute(
                """
                UPDATE tag_groups
                SET name = ?, updated_at = ?
                WHERE id = ?
                """,
                (clean_name, now, group_id),
            )
            return int(cur.rowcount) > 0

    def delete_tag_group(self, group_id: int) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "UPDATE tags SET group_id = NULL WHERE group_id = ?",
                (group_id,),
            )
            conn.execute("DELETE FROM tag_groups WHERE id = ?", (group_id,))
            return int(cur.rowcount)

    def move_tags_to_group(self, tag_ids: Sequence[int], group_id: int | None) -> int:
        clean_ids = self._clean_ids(tag_ids)
        if not clean_ids:
            return 0
        placeholders = ",".join("?" for _ in clean_ids)
        with self.connect() as conn:
            if group_id is not None:
                group = conn.execute(
                    "SELECT id FROM tag_groups WHERE id = ?", (group_id,)
                ).fetchone()
                if group is None:
                    raise ValueError("tag group not found")
            cur = conn.execute(
                f"UPDATE tags SET group_id = ? WHERE id IN ({placeholders})",
                (group_id, *clean_ids),
            )
            return int(cur.rowcount)

    def create_tag(self, tag_name: str, group_id: int | None = None) -> int:
        clean_name = tag_name.strip()
        if not clean_name:
            raise ValueError("tag name must not be empty")
        now = utc_now_iso()
        with self.connect() as conn:
            if group_id is not None:
                group = conn.execute(
                    "SELECT id FROM tag_groups WHERE id = ?", (group_id,)
                ).fetchone()
                if group is None:
                    raise ValueError("tag group not found")
            existing = conn.execute(
                "SELECT id FROM tags WHERE tag_name = ?", (clean_name,)
            ).fetchone()
            if existing is not None:
                tag_id = int(existing["id"])
                if group_id is not None:
                    conn.execute(
                        "UPDATE tags SET group_id = ? WHERE id = ?",
                        (group_id, tag_id),
                    )
                return tag_id
            cur = conn.execute(
                """
                INSERT INTO tags(tag_name, tag_type, group_id, created_at)
                VALUES (?, 'user', ?, ?)
                """,
                (clean_name, group_id, now),
            )
            return int(cur.lastrowid)

    def list_tags(self) -> list[TagItem]:
        with self.connect() as conn:
            return [
                self._tag_from_row(row)
                for row in conn.execute(
                    """
                    SELECT t.*, g.name AS group_name
                    FROM tags t
                    LEFT JOIN tag_groups g ON g.id = t.group_id
                    ORDER BY g.sort_order, g.name COLLATE NOCASE, t.tag_name COLLATE NOCASE
                    """
                )
            ]

    def list_tags_with_counts(self) -> list[tuple[TagItem, int]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT t.*, g.name AS group_name, COUNT(it.image_id) AS image_count
                FROM tags t
                LEFT JOIN tag_groups g ON g.id = t.group_id
                LEFT JOIN image_tags it ON it.tag_id = t.id
                GROUP BY t.id
                ORDER BY g.sort_order, g.name COLLATE NOCASE, t.tag_name COLLATE NOCASE
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
        with self.connect(readonly=True) as conn:
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
        with self.connect(readonly=True) as conn:
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
        with self.connect(readonly=True) as conn:
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
            self._delete_empty_folders(conn)
            return thumbnail_paths

    def snapshot_images_for_restore(self, image_ids: Sequence[int]) -> dict[str, object]:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return {"image_ids": []}
        placeholders = ",".join("?" for _ in clean_ids)

        def rows(conn: sqlite3.Connection, sql: str, params: Sequence[object]) -> list[dict[str, object]]:
            return [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]

        with self.connect() as conn:
            image_rows = rows(
                conn,
                f"SELECT * FROM images WHERE id IN ({placeholders})",
                clean_ids,
            )
            folder_ids = sorted({int(row["folder_id"]) for row in image_rows})
            folder_rows: list[dict[str, object]] = []
            if folder_ids:
                folder_placeholders = ",".join("?" for _ in folder_ids)
                folder_rows = rows(
                    conn,
                    f"SELECT * FROM folders WHERE id IN ({folder_placeholders})",
                    folder_ids,
                )
            image_tags = rows(
                conn,
                f"SELECT * FROM image_tags WHERE image_id IN ({placeholders})",
                clean_ids,
            )
            tag_ids = sorted({int(row["tag_id"]) for row in image_tags})
            tag_rows: list[dict[str, object]] = []
            if tag_ids:
                tag_placeholders = ",".join("?" for _ in tag_ids)
                tag_rows = rows(
                    conn,
                    f"SELECT * FROM tags WHERE id IN ({tag_placeholders})",
                    tag_ids,
                )
            return {
                "image_ids": clean_ids,
                "folders": folder_rows,
                "images": image_rows,
                "tags": tag_rows,
                "image_tags": image_tags,
                "embeddings": rows(
                    conn,
                    f"SELECT * FROM embeddings WHERE image_id IN ({placeholders})",
                    clean_ids,
                ),
                "color_features": rows(
                    conn,
                    f"SELECT * FROM color_features WHERE image_id IN ({placeholders})",
                    clean_ids,
                ),
                "image_collections": rows(
                    conn,
                    f"SELECT * FROM image_collections WHERE image_id IN ({placeholders})",
                    clean_ids,
                ),
                "search_feedback": rows(
                    conn,
                    f"SELECT * FROM search_feedback WHERE image_id IN ({placeholders})",
                    clean_ids,
                ),
                "temporary_project_images": rows(
                    conn,
                    f"SELECT * FROM temporary_project_images WHERE image_id IN ({placeholders})",
                    clean_ids,
                ),
            }

    def restore_images_snapshot(
        self,
        snapshot: Mapping[str, object],
        *,
        image_ids: Sequence[int] | None = None,
    ) -> int:
        snapshot_ids = self._clean_ids(snapshot.get("image_ids", []))  # type: ignore[arg-type]
        clean_ids = self._clean_ids(image_ids if image_ids is not None else snapshot_ids)
        if not clean_ids:
            return 0
        id_set = set(clean_ids)

        def table_rows(table: str, *, id_key: str = "image_id") -> list[dict[str, object]]:
            raw_rows = snapshot.get(table, [])
            if not isinstance(raw_rows, list):
                return []
            clean_rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)]
            if table in {"folders", "tags"}:
                return clean_rows
            return [row for row in clean_rows if int(row.get(id_key, 0) or 0) in id_set]

        def insert_rows(conn: sqlite3.Connection, table: str, rows_to_insert: list[dict[str, object]]) -> None:
            if not rows_to_insert:
                return
            columns = list(rows_to_insert[0].keys())
            column_sql = ", ".join(columns)
            placeholders = ", ".join("?" for _ in columns)
            conn.executemany(
                f"INSERT OR IGNORE INTO {table} ({column_sql}) VALUES ({placeholders})",
                [tuple(row.get(column) for column in columns) for row in rows_to_insert],
            )

        with self.connect() as conn:
            insert_rows(conn, "folders", table_rows("folders", id_key="id"))
            insert_rows(conn, "tags", table_rows("tags"))
            image_rows = table_rows("images", id_key="id")
            insert_rows(conn, "images", image_rows)
            insert_rows(conn, "embeddings", table_rows("embeddings"))
            insert_rows(conn, "color_features", table_rows("color_features"))
            insert_rows(conn, "image_tags", table_rows("image_tags"))
            insert_rows(conn, "image_collections", table_rows("image_collections"))
            insert_rows(conn, "search_feedback", table_rows("search_feedback"))
            insert_rows(conn, "temporary_project_images", table_rows("temporary_project_images"))
            self._delete_unused_tags(conn)
            placeholders = ",".join("?" for _ in clean_ids)
            row = conn.execute(
                f"SELECT COUNT(*) AS count FROM images WHERE id IN ({placeholders})",
                tuple(clean_ids),
            ).fetchone()
            return int(row["count"] if row is not None else 0)

    def remove_missing_images_from_library(self) -> tuple[list[str], int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT thumbnail_path
                FROM images
                WHERE is_missing = 1
                  AND thumbnail_path IS NOT NULL
                """
            ).fetchall()
            count_row = conn.execute(
                "SELECT COUNT(*) AS count FROM images WHERE is_missing = 1"
            ).fetchone()
            thumbnail_paths = [str(row["thumbnail_path"]) for row in rows]
            conn.execute("DELETE FROM images WHERE is_missing = 1")
            self._delete_unused_tags(conn)
            self._delete_empty_folders(conn)
            return thumbnail_paths, int(count_row["count"])

    def folders_with_missing_images(self) -> list[FolderItem]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT f.*
                FROM folders f
                JOIN images i ON i.folder_id = f.id
                WHERE f.is_active = 1
                  AND i.is_missing = 1
                ORDER BY f.folder_path COLLATE NOCASE
                """
            ).fetchall()
        return [self._folder_from_row(row) for row in rows]

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
            self._delete_empty_folders(conn)
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
            self._delete_empty_folders(conn)
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
        color_hex: str = "",
        kind: str = "semantic",
    ) -> int:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("temporary project name must not be empty")
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            raise ValueError("temporary project must contain at least one image")
        clean_summary = _clean_optional_text(summary, max_length=600) or ""
        clean_kind = _clean_temporary_project_kind(kind)
        now = utc_now_iso()
        with self.connect() as conn:
            clean_color = _clean_color_hex(color_hex) or self._next_temporary_project_color(conn)
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
                INSERT INTO temporary_projects(
                    name, summary, color_hex, kind, sort_order, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_name,
                    clean_summary,
                    clean_color,
                    clean_kind,
                    self._next_temporary_project_sort_order(conn),
                    now,
                    now,
                ),
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

    def list_temporary_projects(self, *, kind: str | None = None) -> list[TemporaryProjectItem]:
        clean_kind = _clean_temporary_project_kind(kind) if kind is not None else None
        with self.connect() as conn:
            where = "WHERE p.kind = ?" if clean_kind is not None else ""
            params: tuple[object, ...] = (clean_kind,) if clean_kind is not None else ()
            rows = conn.execute(
                f"""
                SELECT p.*, COUNT(tpi.image_id) AS image_count
                FROM temporary_projects p
                LEFT JOIN temporary_project_images tpi ON tpi.project_id = p.id
                {where}
                GROUP BY p.id
                ORDER BY p.sort_order DESC, p.updated_at DESC, p.id DESC
                """,
                params,
            ).fetchall()
        return [
            TemporaryProjectItem(
                id=int(row["id"]),
                name=str(row["name"]),
                image_count=int(row["image_count"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                summary=str(row["summary"] or ""),
                color_hex=str(row["color_hex"] or ""),
                kind=_clean_temporary_project_kind(str(row["kind"] or "semantic")),
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

    def temporary_project_image_links(
        self,
        project_id: int,
        image_ids: Sequence[int],
    ) -> list[dict[str, object]]:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return []
        placeholders = ",".join("?" for _ in clean_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT image_id, sort_order, created_at, intent_label, intent_query
                FROM temporary_project_images
                WHERE project_id = ?
                  AND image_id IN ({placeholders})
                ORDER BY sort_order, image_id
                """,
                [project_id, *clean_ids],
            ).fetchall()
        return [
            {
                "project_id": int(project_id),
                "image_id": int(row["image_id"]),
                "sort_order": int(row["sort_order"]),
                "created_at": str(row["created_at"]),
                "intent_label": row["intent_label"],
                "intent_query": row["intent_query"],
            }
            for row in rows
        ]

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
            color_hex=str(row["color_hex"] or ""),
            kind=_clean_temporary_project_kind(str(row["kind"] or "semantic")),
        )

    def next_temporary_project_color(self) -> str:
        with self.connect() as conn:
            return self._next_temporary_project_color(conn)

    def delete_temporary_project(self, project_id: int) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM temporary_projects WHERE id = ?", (project_id,))
            return int(cur.rowcount) > 0

    def clear_temporary_projects(self, *, kind: str | None = None) -> int:
        clean_kind = _clean_temporary_project_kind(kind) if kind is not None else None
        with self.connect() as conn:
            if clean_kind is None:
                row = conn.execute("SELECT COUNT(*) AS count FROM temporary_projects").fetchone()
                count = int(row["count"]) if row is not None else 0
                conn.execute("DELETE FROM temporary_projects")
                return count
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM temporary_projects WHERE kind = ?",
                (clean_kind,),
            ).fetchone()
            count = int(row["count"]) if row is not None else 0
            conn.execute("DELETE FROM temporary_projects WHERE kind = ?", (clean_kind,))
            return count

    def move_temporary_project(
        self,
        project_id: int,
        direction: int,
        *,
        kind: str | None = None,
    ) -> bool:
        if direction == 0:
            return False
        clean_kind = _clean_temporary_project_kind(kind) if kind is not None else None
        with self.connect() as conn:
            self._normalize_temporary_project_sort_order(conn)
            where = "WHERE kind = ?" if clean_kind is not None else ""
            params: tuple[object, ...] = (clean_kind,) if clean_kind is not None else ()
            rows = conn.execute(
                f"""
                SELECT id, sort_order
                FROM temporary_projects
                {where}
                ORDER BY sort_order DESC, updated_at DESC, id DESC
                """,
                params,
            ).fetchall()
            ids = [int(row["id"]) for row in rows]
            try:
                index = ids.index(int(project_id))
            except ValueError:
                return False
            target_index = index + (-1 if direction < 0 else 1)
            if target_index < 0 or target_index >= len(rows):
                return False
            current = rows[index]
            target = rows[target_index]
            conn.execute(
                "UPDATE temporary_projects SET sort_order = ? WHERE id = ?",
                (int(target["sort_order"]), int(current["id"])),
            )
            conn.execute(
                "UPDATE temporary_projects SET sort_order = ? WHERE id = ?",
                (int(current["sort_order"]), int(target["id"])),
            )
            return True

    def update_temporary_project_details(
        self,
        project_id: int,
        *,
        name: str | None = None,
        summary: str | None = None,
    ) -> TemporaryProjectItem | None:
        clean_name: str | None = None
        if name is not None:
            clean_name = " ".join(name.strip().split())[:80]
            if not clean_name:
                raise ValueError("temporary project name must not be empty")
        clean_summary = (
            " ".join(summary.strip().split())[:600]
            if summary is not None
            else None
        )
        if name is None and summary is None:
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

    @staticmethod
    def _backfill_temporary_project_colors(conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id
            FROM temporary_projects
            WHERE color_hex IS NULL OR TRIM(color_hex) = ''
            ORDER BY created_at, id
            """
        ).fetchall()
        for index, row in enumerate(rows):
            conn.execute(
                "UPDATE temporary_projects SET color_hex = ? WHERE id = ?",
                (TEMPORARY_PROJECT_COLORS[index % len(TEMPORARY_PROJECT_COLORS)], int(row["id"])),
            )

    @staticmethod
    def _next_temporary_project_color(conn: sqlite3.Connection) -> str:
        row = conn.execute(
            """
            SELECT color_hex
            FROM temporary_projects
            WHERE color_hex IS NOT NULL AND TRIM(color_hex) != ''
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return TEMPORARY_PROJECT_COLORS[0]
        current = _clean_color_hex(row["color_hex"])
        if current in TEMPORARY_PROJECT_COLORS:
            index = TEMPORARY_PROJECT_COLORS.index(current)
            return TEMPORARY_PROJECT_COLORS[(index + 1) % len(TEMPORARY_PROJECT_COLORS)]
        count_row = conn.execute("SELECT COUNT(*) AS count FROM temporary_projects").fetchone()
        count = int(count_row["count"]) if count_row is not None else 0
        return TEMPORARY_PROJECT_COLORS[count % len(TEMPORARY_PROJECT_COLORS)]

    @staticmethod
    def _backfill_temporary_project_sort_order(conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id
            FROM temporary_projects
            ORDER BY updated_at ASC, id ASC
            """
        ).fetchall()
        for index, row in enumerate(rows):
            conn.execute(
                "UPDATE temporary_projects SET sort_order = ? WHERE id = ?",
                (index, int(row["id"])),
            )

    @staticmethod
    def _normalize_temporary_project_sort_order(conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id
            FROM temporary_projects
            ORDER BY sort_order ASC, updated_at ASC, id ASC
            """
        ).fetchall()
        for index, row in enumerate(rows):
            conn.execute(
                "UPDATE temporary_projects SET sort_order = ? WHERE id = ?",
                (index, int(row["id"])),
            )

    @staticmethod
    def _next_temporary_project_sort_order(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_order FROM temporary_projects"
        ).fetchone()
        return int(row["next_order"]) if row is not None else 0

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

    def restore_temporary_project_image_links(
        self,
        project_id: int,
        links: Sequence[Mapping[str, object]],
    ) -> int:
        if not links:
            return 0
        now = utc_now_iso()
        restored = 0
        with self.connect() as conn:
            project = conn.execute(
                "SELECT id FROM temporary_projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            if project is None:
                return 0
            for link in links:
                try:
                    image_id = int(link.get("image_id", 0) or 0)
                    sort_order = int(link.get("sort_order", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if image_id <= 0:
                    continue
                image = conn.execute("SELECT id FROM images WHERE id = ?", (image_id,)).fetchone()
                if image is None:
                    continue
                created_at = str(link.get("created_at") or now)
                label = _clean_optional_text(link.get("intent_label"), max_length=80)
                query = _clean_optional_text(link.get("intent_query"), max_length=160)
                cur = conn.execute(
                    """
                    INSERT INTO temporary_project_images(
                        project_id, image_id, sort_order, created_at, intent_label, intent_query
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, image_id) DO UPDATE SET
                        sort_order = excluded.sort_order,
                        intent_label = excluded.intent_label,
                        intent_query = excluded.intent_query
                    """,
                    (project_id, image_id, sort_order, created_at, label, query),
                )
                restored += int(cur.rowcount)
            if restored:
                conn.execute(
                    "UPDATE temporary_projects SET updated_at = ? WHERE id = ?",
                    (now, project_id),
                )
        return restored

    def save_temporary_project_board_layout(self, project_id: int, payload_json: str) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            project = conn.execute(
                "SELECT id FROM temporary_projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            if project is None:
                return
            conn.execute(
                """
                INSERT INTO temporary_project_board_layouts(project_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (project_id, payload_json, now),
            )
            conn.execute(
                "UPDATE temporary_projects SET updated_at = ? WHERE id = ?",
                (now, project_id),
            )

    def get_temporary_project_board_layout(self, project_id: int) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM temporary_project_board_layouts WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        return str(row["payload_json"]) if row is not None else None

    def save_temporary_project_state(self, project_id: int, payload_json: str) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            project = conn.execute(
                "SELECT id FROM temporary_projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            if project is None:
                return
            conn.execute(
                """
                INSERT INTO temporary_project_states(project_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (project_id, payload_json, now),
            )
            conn.execute(
                "UPDATE temporary_projects SET updated_at = ? WHERE id = ?",
                (now, project_id),
            )

    def get_temporary_project_state(self, project_id: int) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM temporary_project_states WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        return str(row["payload_json"]) if row is not None else None

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

    def list_inspiration_projects(self, *, limit: int = 80) -> list[InspirationProjectItem]:
        safe_limit = max(1, min(int(limit), 500))
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    p.*,
                    COUNT(t.id) AS term_count,
                    COALESCE(SUM(CASE WHEN t.selected = 1 THEN 1 ELSE 0 END), 0) AS selected_count
                FROM inspiration_projects p
                LEFT JOIN inspiration_terms t ON t.project_id = p.id
                GROUP BY p.id
                ORDER BY p.updated_at DESC, p.id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [self._inspiration_project_from_row(row) for row in rows]

    def get_inspiration_project(self, project_id: int) -> InspirationProjectItem | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    p.*,
                    COUNT(t.id) AS term_count,
                    COALESCE(SUM(CASE WHEN t.selected = 1 THEN 1 ELSE 0 END), 0) AS selected_count
                FROM inspiration_projects p
                LEFT JOIN inspiration_terms t ON t.project_id = p.id
                WHERE p.id = ?
                GROUP BY p.id
                """,
                (project_id,),
            ).fetchone()
        return self._inspiration_project_from_row(row) if row is not None else None

    def update_inspiration_project_selection(
        self,
        project_id: int,
        *,
        selected_titles: set[str],
    ) -> bool:
        now = utc_now_iso()
        with self.connect() as conn:
            project = conn.execute(
                "SELECT id FROM inspiration_projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            if project is None:
                return False
            conn.execute(
                "UPDATE inspiration_terms SET selected = 0 WHERE project_id = ?",
                (project_id,),
            )
            if selected_titles:
                placeholders = ",".join("?" for _ in selected_titles)
                conn.execute(
                    f"""
                    UPDATE inspiration_terms
                    SET selected = 1
                    WHERE project_id = ?
                      AND title IN ({placeholders})
                    """,
                    [project_id, *sorted(selected_titles)],
                )
            conn.execute(
                "UPDATE inspiration_projects SET updated_at = ? WHERE id = ?",
                (now, project_id),
            )
            return True

    def delete_inspiration_project(self, project_id: int) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM inspiration_projects WHERE id = ?", (project_id,))
            return int(cur.rowcount) > 0

    def create_creative_project(
        self,
        *,
        title: str,
        brief: str = "",
        language: str = "zh",
        provider_name: str = "",
        model_name: str = "",
    ) -> int:
        clean_title = " ".join(title.strip().split())[:120] or "未命名创作项目"
        clean_brief = brief.strip()[:1200]
        clean_language = language if language in {"zh", "en"} else "zh"
        now = utc_now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO creative_projects(
                    title, brief, language, provider_name, model_name, sort_order, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_title,
                    clean_brief,
                    clean_language,
                    provider_name.strip()[:80],
                    model_name.strip()[:120],
                    self._next_creative_project_sort_order(conn),
                    now,
                    now,
                ),
            )
            project_id = int(cur.lastrowid)
            conn.execute(
                """
                INSERT INTO creative_nodes(
                    project_id, parent_id, title, note, search_query, sort_order, created_at, updated_at
                )
                VALUES (?, NULL, ?, ?, ?, 0, ?, ?)
                """,
                (project_id, clean_title, clean_brief, clean_brief or clean_title, now, now),
            )
            return project_id

    def list_creative_projects(self) -> list[CreativeProjectItem]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    p.*,
                    COUNT(DISTINCT n.id) AS node_count,
                    COUNT(DISTINCT cni.image_id) AS image_count
                FROM creative_projects p
                LEFT JOIN creative_nodes n ON n.project_id = p.id
                LEFT JOIN creative_node_images cni ON cni.node_id = n.id
                GROUP BY p.id
                ORDER BY p.is_pinned DESC, p.sort_order DESC, p.updated_at DESC, p.id DESC
                """
            ).fetchall()
        return [self._creative_project_from_row(row) for row in rows]

    def get_creative_project(self, project_id: int) -> CreativeProjectItem | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    p.*,
                    COUNT(DISTINCT n.id) AS node_count,
                    COUNT(DISTINCT cni.image_id) AS image_count
                FROM creative_projects p
                LEFT JOIN creative_nodes n ON n.project_id = p.id
                LEFT JOIN creative_node_images cni ON cni.node_id = n.id
                WHERE p.id = ?
                GROUP BY p.id
                """,
                (project_id,),
            ).fetchone()
        return self._creative_project_from_row(row) if row is not None else None

    def delete_creative_project(self, project_id: int) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM creative_projects WHERE id = ?", (project_id,))
            return int(cur.rowcount) > 0

    def set_creative_project_pinned(self, project_id: int, pinned: bool) -> bool:
        now = utc_now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                """
                UPDATE creative_projects
                SET is_pinned = ?, updated_at = ?
                WHERE id = ?
                """,
                (1 if pinned else 0, now, project_id),
            )
            return int(cur.rowcount) > 0

    def move_creative_project(self, project_id: int, direction: int) -> bool:
        if direction == 0:
            return False
        with self.connect() as conn:
            self._normalize_creative_project_sort_order(conn)
            current_row = conn.execute(
                "SELECT is_pinned FROM creative_projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            if current_row is None:
                return False
            rows = conn.execute(
                """
                SELECT id, sort_order
                FROM creative_projects
                WHERE is_pinned = ?
                ORDER BY sort_order DESC, updated_at DESC, id DESC
                """,
                (int(current_row["is_pinned"] or 0),),
            ).fetchall()
            ids = [int(row["id"]) for row in rows]
            try:
                index = ids.index(int(project_id))
            except ValueError:
                return False
            target_index = index + (-1 if direction < 0 else 1)
            if target_index < 0 or target_index >= len(rows):
                return False
            current = rows[index]
            target = rows[target_index]
            conn.execute(
                "UPDATE creative_projects SET sort_order = ? WHERE id = ?",
                (int(target["sort_order"]), int(current["id"])),
            )
            conn.execute(
                "UPDATE creative_projects SET sort_order = ? WHERE id = ?",
                (int(current["sort_order"]), int(target["id"])),
            )
            return True

    def update_creative_project_details(
        self,
        project_id: int,
        *,
        title: str | None = None,
        brief: str | None = None,
    ) -> CreativeProjectItem | None:
        clean_title: str | None = None
        if title is not None:
            clean_title = " ".join(title.strip().split())[:120]
            if not clean_title:
                raise ValueError("creative project title must not be empty")
        clean_brief = brief.strip()[:1200] if brief is not None else None
        if title is None and brief is None:
            return self.get_creative_project(project_id)
        now = utc_now_iso()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM creative_projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            if row is None:
                return None
            assignments: list[str] = ["updated_at = ?"]
            params: list[object] = [now]
            if clean_title is not None:
                assignments.append("title = ?")
                params.append(clean_title)
            if clean_brief is not None:
                assignments.append("brief = ?")
                params.append(clean_brief)
            params.append(project_id)
            conn.execute(
                f"UPDATE creative_projects SET {', '.join(assignments)} WHERE id = ?",
                params,
            )
            if clean_title is not None:
                root = conn.execute(
                    """
                    SELECT id
                    FROM creative_nodes
                    WHERE project_id = ? AND parent_id IS NULL
                    ORDER BY sort_order, id
                    LIMIT 1
                    """,
                    (project_id,),
                ).fetchone()
                if root is not None:
                    conn.execute(
                        "UPDATE creative_nodes SET title = ?, updated_at = ? WHERE id = ?",
                        (clean_title, now, int(root["id"])),
                    )
        return self.get_creative_project(project_id)

    def update_creative_project_copy(self, project_id: int, copy_text: str) -> bool:
        clean_copy = copy_text.strip()[:4000]
        now = utc_now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                """
                UPDATE creative_projects
                SET copy_text = ?, updated_at = ?
                WHERE id = ?
                """,
                (clean_copy, now, project_id),
            )
            return int(cur.rowcount) > 0

    def creative_root_node_id(self, project_id: int) -> int | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id
                FROM creative_nodes
                WHERE project_id = ? AND parent_id IS NULL
                ORDER BY sort_order, id
                LIMIT 1
                """,
                (project_id,),
            ).fetchone()
        return int(row["id"]) if row is not None else None

    def create_creative_node(
        self,
        *,
        project_id: int,
        parent_id: int | None,
        title: str,
        note: str = "",
        search_query: str = "",
    ) -> int:
        clean_title = " ".join(title.strip().split())[:120] or "未命名节点"
        clean_note = note.strip()[:1600]
        clean_query = " ".join((search_query or clean_note or clean_title).strip().split())[:400]
        now = utc_now_iso()
        with self.connect() as conn:
            project = conn.execute(
                "SELECT id FROM creative_projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            if project is None:
                raise ValueError("creative project does not exist")
            if parent_id is not None:
                parent = conn.execute(
                    "SELECT id FROM creative_nodes WHERE id = ? AND project_id = ?",
                    (parent_id, project_id),
                ).fetchone()
                if parent is None:
                    raise ValueError("parent creative node does not exist")
            row = conn.execute(
                """
                SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_order
                FROM creative_nodes
                WHERE project_id = ? AND parent_id IS ?
                """,
                (project_id, parent_id),
            ).fetchone()
            sort_order = int(row["next_order"]) if row is not None else 0
            cur = conn.execute(
                """
                INSERT INTO creative_nodes(
                    project_id, parent_id, title, note, search_query, sort_order, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (project_id, parent_id, clean_title, clean_note, clean_query, sort_order, now, now),
            )
            conn.execute(
                "UPDATE creative_projects SET updated_at = ? WHERE id = ?",
                (now, project_id),
            )
            return int(cur.lastrowid)

    def update_creative_node(
        self,
        node_id: int,
        *,
        title: str | None = None,
        note: str | None = None,
        search_query: str | None = None,
    ) -> CreativeNodeItem | None:
        assignments: list[str] = ["updated_at = ?"]
        now = utc_now_iso()
        params: list[object] = [now]
        if title is not None:
            clean_title = " ".join(title.strip().split())[:120]
            if not clean_title:
                raise ValueError("creative node title must not be empty")
            assignments.append("title = ?")
            params.append(clean_title)
        if note is not None:
            assignments.append("note = ?")
            params.append(note.strip()[:1600])
        if search_query is not None:
            assignments.append("search_query = ?")
            params.append(" ".join(search_query.strip().split())[:400])
        if len(assignments) == 1:
            return self.get_creative_node(node_id)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT project_id FROM creative_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
            if row is None:
                return None
            params.append(node_id)
            conn.execute(
                f"UPDATE creative_nodes SET {', '.join(assignments)} WHERE id = ?",
                params,
            )
            conn.execute(
                "UPDATE creative_projects SET updated_at = ? WHERE id = ?",
                (now, int(row["project_id"])),
            )
        return self.get_creative_node(node_id)

    def delete_creative_node(self, node_id: int) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT project_id, parent_id FROM creative_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
            if row is None or row["parent_id"] is None:
                return False
            now = utc_now_iso()
            cur = conn.execute("DELETE FROM creative_nodes WHERE id = ?", (node_id,))
            if cur.rowcount:
                conn.execute(
                    "UPDATE creative_projects SET updated_at = ? WHERE id = ?",
                    (now, int(row["project_id"])),
                )
            return int(cur.rowcount) > 0

    def get_creative_node(self, node_id: int) -> CreativeNodeItem | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT n.*, COUNT(cni.image_id) AS image_count
                FROM creative_nodes n
                LEFT JOIN creative_node_images cni ON cni.node_id = n.id
                WHERE n.id = ?
                GROUP BY n.id
                """,
                (node_id,),
            ).fetchone()
        return self._creative_node_from_row(row) if row is not None else None

    def list_creative_nodes(self, project_id: int) -> list[CreativeNodeItem]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT n.*, COUNT(cni.image_id) AS image_count
                FROM creative_nodes n
                LEFT JOIN creative_node_images cni ON cni.node_id = n.id
                WHERE n.project_id = ?
                GROUP BY n.id
                ORDER BY
                    CASE WHEN n.parent_id IS NULL THEN 0 ELSE 1 END,
                    n.parent_id,
                    n.sort_order,
                    n.id
                """,
                (project_id,),
            ).fetchall()
        return [self._creative_node_from_row(row) for row in rows]

    def creative_nodes_with_branch_image_counts(
        self,
        project_id: int,
    ) -> tuple[list[CreativeNodeItem], dict[int, int]]:
        with self.connect() as conn:
            node_rows = conn.execute(
                """
                SELECT n.*, COUNT(cni.image_id) AS image_count
                FROM creative_nodes n
                LEFT JOIN creative_node_images cni ON cni.node_id = n.id
                WHERE n.project_id = ?
                GROUP BY n.id
                ORDER BY
                    CASE WHEN n.parent_id IS NULL THEN 0 ELSE 1 END,
                    n.parent_id,
                    n.sort_order,
                    n.id
                """,
                (project_id,),
            ).fetchall()
            image_rows = conn.execute(
                """
                SELECT cni.node_id, cni.image_id
                FROM creative_node_images cni
                JOIN creative_nodes n ON n.id = cni.node_id
                WHERE n.project_id = ?
                """,
                (project_id,),
            ).fetchall()
        nodes = [self._creative_node_from_row(row) for row in node_rows]
        branch_counts = self._branch_image_counts_from_rows(node_rows, image_rows)
        return nodes, branch_counts

    def add_images_to_creative_node(
        self,
        node_id: int,
        image_ids: Sequence[int],
        *,
        intent_label: str = "",
        intent_query: str = "",
    ) -> int:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return 0
        label = _clean_optional_text(intent_label, max_length=80)
        query = _clean_optional_text(intent_query, max_length=200)
        now = utc_now_iso()
        changed = 0
        with self.connect() as conn:
            node = conn.execute(
                "SELECT project_id FROM creative_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
            if node is None:
                raise ValueError("creative node does not exist")
            row = conn.execute(
                """
                SELECT COALESCE(MAX(sort_order), -1) AS max_sort_order
                FROM creative_node_images
                WHERE node_id = ?
                """,
                (node_id,),
            ).fetchone()
            next_order = int(row["max_sort_order"]) + 1 if row is not None else 0
            for offset, image_id in enumerate(clean_ids):
                cur = conn.execute(
                    """
                    INSERT INTO creative_node_images(
                        node_id, image_id, sort_order, created_at, intent_label, intent_query
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id, image_id) DO UPDATE SET
                        intent_label = COALESCE(excluded.intent_label, creative_node_images.intent_label),
                        intent_query = COALESCE(excluded.intent_query, creative_node_images.intent_query)
                    """,
                    (node_id, image_id, next_order + offset, now, label, query),
                )
                changed += int(cur.rowcount)
            conn.execute(
                "UPDATE creative_projects SET updated_at = ? WHERE id = ?",
                (now, int(node["project_id"])),
            )
        return changed

    def remove_images_from_creative_node(self, node_id: int, image_ids: Sequence[int]) -> int:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return 0
        placeholders = ",".join("?" for _ in clean_ids)
        now = utc_now_iso()
        with self.connect() as conn:
            node = conn.execute(
                "SELECT project_id FROM creative_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
            if node is None:
                return 0
            cur = conn.execute(
                f"""
                DELETE FROM creative_node_images
                WHERE node_id = ? AND image_id IN ({placeholders})
                """,
                [node_id, *clean_ids],
            )
            removed = int(cur.rowcount)
            if removed:
                conn.execute(
                    "UPDATE creative_projects SET updated_at = ? WHERE id = ?",
                    (now, int(node["project_id"])),
                )
            return removed

    def remove_images_from_creative_node_branch(self, node_id: int, image_ids: Sequence[int]) -> int:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return 0
        node = self.get_creative_node(node_id)
        if node is None:
            return 0
        node_ids = self._creative_descendant_node_ids(node.project_id, node_id)
        if not node_ids:
            return 0
        image_placeholders = ",".join("?" for _ in clean_ids)
        node_placeholders = ",".join("?" for _ in node_ids)
        now = utc_now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                f"""
                DELETE FROM creative_node_images
                WHERE node_id IN ({node_placeholders})
                  AND image_id IN ({image_placeholders})
                """,
                [*node_ids, *clean_ids],
            )
            removed = int(cur.rowcount)
            if removed:
                conn.execute(
                    "UPDATE creative_projects SET updated_at = ? WHERE id = ?",
                    (now, node.project_id),
                )
            return removed

    def creative_node_image_links_for_branch(
        self,
        node_id: int,
        image_ids: Sequence[int],
    ) -> list[dict[str, object]]:
        clean_ids = self._clean_ids(image_ids)
        if not clean_ids:
            return []
        node = self.get_creative_node(node_id)
        if node is None:
            return []
        node_ids = self._creative_descendant_node_ids(node.project_id, node_id)
        if not node_ids:
            return []
        image_placeholders = ",".join("?" for _ in clean_ids)
        node_placeholders = ",".join("?" for _ in node_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT node_id, image_id, sort_order, created_at, intent_label, intent_query
                FROM creative_node_images
                WHERE node_id IN ({node_placeholders})
                  AND image_id IN ({image_placeholders})
                ORDER BY node_id, sort_order, image_id
                """,
                [*node_ids, *clean_ids],
            ).fetchall()
        return [
            {
                "node_id": int(row["node_id"]),
                "image_id": int(row["image_id"]),
                "sort_order": int(row["sort_order"]),
                "created_at": str(row["created_at"]),
                "intent_label": row["intent_label"],
                "intent_query": row["intent_query"],
            }
            for row in rows
        ]

    def restore_creative_node_image_links(
        self,
        links: Sequence[Mapping[str, object]],
    ) -> int:
        if not links:
            return 0
        now = utc_now_iso()
        restored = 0
        project_ids: set[int] = set()
        with self.connect() as conn:
            for link in links:
                try:
                    node_id = int(link.get("node_id", 0) or 0)
                    image_id = int(link.get("image_id", 0) or 0)
                    sort_order = int(link.get("sort_order", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if node_id <= 0 or image_id <= 0:
                    continue
                node = conn.execute(
                    "SELECT project_id FROM creative_nodes WHERE id = ?",
                    (node_id,),
                ).fetchone()
                if node is None:
                    continue
                image = conn.execute("SELECT id FROM images WHERE id = ?", (image_id,)).fetchone()
                if image is None:
                    continue
                project_ids.add(int(node["project_id"]))
                created_at = str(link.get("created_at") or now)
                label = _clean_optional_text(link.get("intent_label"), max_length=80)
                query = _clean_optional_text(link.get("intent_query"), max_length=200)
                cur = conn.execute(
                    """
                    INSERT INTO creative_node_images(
                        node_id, image_id, sort_order, created_at, intent_label, intent_query
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id, image_id) DO UPDATE SET
                        sort_order = excluded.sort_order,
                        intent_label = excluded.intent_label,
                        intent_query = excluded.intent_query
                    """,
                    (node_id, image_id, sort_order, created_at, label, query),
                )
                restored += int(cur.rowcount)
            if restored:
                for project_id in project_ids:
                    conn.execute(
                        "UPDATE creative_projects SET updated_at = ? WHERE id = ?",
                        (now, project_id),
                    )
        return restored

    def creative_node_image_ids(self, node_id: int, *, include_descendants: bool = False) -> list[int]:
        node_ids = [node_id]
        if include_descendants:
            node = self.get_creative_node(node_id)
            if node is not None:
                node_ids = self._creative_descendant_node_ids(node.project_id, node_id)
        if not node_ids:
            return []
        placeholders = ",".join("?" for _ in node_ids)
        node_order = {node_id: index for index, node_id in enumerate(node_ids)}
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT node_id, image_id, sort_order
                FROM creative_node_images
                WHERE node_id IN ({placeholders})
                """,
                node_ids,
            ).fetchall()
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                node_order.get(int(row["node_id"]), len(node_order)),
                int(row["sort_order"]),
                int(row["image_id"]),
            ),
        )
        image_ids: list[int] = []
        seen: set[int] = set()
        for row in ordered_rows:
            image_id = int(row["image_id"])
            if image_id in seen:
                continue
            seen.add(image_id)
            image_ids.append(image_id)
        return image_ids

    def creative_node_branch_board_data(self, node_id: int) -> dict[str, object] | None:
        with self.connect() as conn:
            node_row = conn.execute(
                """
                SELECT n.*, COUNT(cni.image_id) AS image_count
                FROM creative_nodes n
                LEFT JOIN creative_node_images cni ON cni.node_id = n.id
                WHERE n.id = ?
                GROUP BY n.id
                """,
                (node_id,),
            ).fetchone()
            if node_row is None:
                return None
            project_id = int(node_row["project_id"])
            node_rows = conn.execute(
                """
                SELECT n.*, COUNT(cni.image_id) AS image_count
                FROM creative_nodes n
                LEFT JOIN creative_node_images cni ON cni.node_id = n.id
                WHERE n.project_id = ?
                GROUP BY n.id
                ORDER BY
                    CASE WHEN n.parent_id IS NULL THEN 0 ELSE 1 END,
                    n.parent_id,
                    n.sort_order,
                    n.id
                """,
                (project_id,),
            ).fetchall()
            branch_node_ids = self._ordered_descendant_ids_from_node_rows(node_rows, node_id)
            if not branch_node_ids:
                return {
                    "node": self._creative_node_from_row(node_row),
                    "nodes": [self._creative_node_from_row(row) for row in node_rows],
                    "image_ids": [],
                    "images": [],
                    "badges": {},
                }
            node_placeholders = ",".join("?" for _ in branch_node_ids)
            link_rows = conn.execute(
                f"""
                SELECT
                    cni.node_id,
                    cni.image_id,
                    cni.sort_order,
                    COALESCE(NULLIF(TRIM(cni.intent_label), ''), n.title) AS badge
                FROM creative_node_images cni
                JOIN creative_nodes n ON n.id = cni.node_id
                WHERE cni.node_id IN ({node_placeholders})
                """,
                branch_node_ids,
            ).fetchall()
            node_order = {branch_id: index for index, branch_id in enumerate(branch_node_ids)}
            ordered_links = sorted(
                link_rows,
                key=lambda row: (
                    node_order.get(int(row["node_id"]), len(node_order)),
                    int(row["sort_order"]),
                    int(row["image_id"]),
                ),
            )
            image_ids: list[int] = []
            seen_image_ids: set[int] = set()
            badges: dict[int, list[str]] = {}
            for row in ordered_links:
                image_id = int(row["image_id"])
                if image_id not in seen_image_ids:
                    seen_image_ids.add(image_id)
                    image_ids.append(image_id)
                badge = str(row["badge"] or "").strip()
                if badge:
                    badges.setdefault(image_id, [])
                    if badge not in badges[image_id]:
                        badges[image_id].append(badge)
            image_rows = []
            if image_ids:
                image_placeholders = ",".join("?" for _ in image_ids)
                image_rows = conn.execute(
                    f"SELECT * FROM images WHERE id IN ({image_placeholders})",
                    image_ids,
                ).fetchall()
        images_by_id = {int(row["id"]): self._image_from_row(row) for row in image_rows}
        return {
            "node": self._creative_node_from_row(node_row),
            "nodes": [self._creative_node_from_row(row) for row in node_rows],
            "image_ids": [image_id for image_id in image_ids if image_id in images_by_id],
            "images": [images_by_id[image_id] for image_id in image_ids if image_id in images_by_id],
            "badges": badges,
        }

    def creative_node_branch_image_counts(self, project_id: int) -> dict[int, int]:
        with self.connect() as conn:
            node_rows = conn.execute(
                """
                SELECT id, parent_id
                FROM creative_nodes
                WHERE project_id = ?
                ORDER BY sort_order, id
                """,
                (project_id,),
            ).fetchall()
            image_rows = conn.execute(
                """
                SELECT cni.node_id, cni.image_id
                FROM creative_node_images cni
                JOIN creative_nodes n ON n.id = cni.node_id
                WHERE n.project_id = ?
                """,
                (project_id,),
            ).fetchall()
        return self._branch_image_counts_from_rows(node_rows, image_rows)

    @staticmethod
    def _branch_image_counts_from_rows(
        node_rows: Sequence[sqlite3.Row],
        image_rows: Sequence[sqlite3.Row],
    ) -> dict[int, int]:
        node_ids = [int(row["id"]) for row in node_rows]
        children_by_parent: dict[int | None, list[int]] = {}
        for row in node_rows:
            node_id = int(row["id"])
            parent_id = row["parent_id"]
            children_by_parent.setdefault(
                int(parent_id) if parent_id is not None else None,
                [],
            ).append(node_id)
        direct_images: dict[int, set[int]] = {node_id: set() for node_id in node_ids}
        for row in image_rows:
            node_id = int(row["node_id"])
            if node_id in direct_images:
                direct_images[node_id].add(int(row["image_id"]))

        branch_images_by_node: dict[int, set[int]] = {}
        visiting: set[int] = set()

        def collect(current_id: int) -> set[int]:
            if current_id in branch_images_by_node:
                return set(branch_images_by_node[current_id])
            if current_id in visiting:
                return set(direct_images.get(current_id, set()))
            visiting.add(current_id)
            images = set(direct_images.get(current_id, set()))
            for child_id in children_by_parent.get(current_id, []):
                images.update(collect(child_id))
            visiting.discard(current_id)
            branch_images_by_node[current_id] = images
            return set(images)

        for node_id in node_ids:
            collect(node_id)
        return {node_id: len(images) for node_id, images in branch_images_by_node.items()}

    def creative_node_image_badges(
        self,
        project_id: int,
        image_ids: Sequence[int] | None = None,
    ) -> dict[int, list[str]]:
        filtered_image_ids: list[int] = []
        if image_ids is not None:
            seen: set[int] = set()
            for image_id in image_ids:
                clean_id = int(image_id)
                if clean_id in seen:
                    continue
                filtered_image_ids.append(clean_id)
                seen.add(clean_id)
            if not filtered_image_ids:
                return {}
        image_filter_sql = ""
        params: list[object] = [project_id]
        if filtered_image_ids:
            placeholders = ",".join("?" for _ in filtered_image_ids)
            image_filter_sql = f" AND cni.image_id IN ({placeholders})"
            params.extend(filtered_image_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT cni.image_id, COALESCE(NULLIF(TRIM(cni.intent_label), ''), n.title) AS badge
                FROM creative_node_images cni
                JOIN creative_nodes n ON n.id = cni.node_id
                WHERE n.project_id = ?
                {image_filter_sql}
                ORDER BY n.sort_order, cni.sort_order, cni.image_id
                """,
                params,
            ).fetchall()
        badges: dict[int, list[str]] = {}
        for row in rows:
            image_id = int(row["image_id"])
            badge = str(row["badge"] or "").strip()
            if not badge:
                continue
            badges.setdefault(image_id, [])
            if badge not in badges[image_id]:
                badges[image_id].append(badge)
        return badges

    def save_creative_board_layout(self, project_id: int, payload_json: str) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO creative_board_layouts(project_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (project_id, payload_json, now),
            )
            conn.execute(
                "UPDATE creative_projects SET updated_at = ? WHERE id = ?",
                (now, project_id),
            )

    def get_creative_board_layout(self, project_id: int) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM creative_board_layouts WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        return str(row["payload_json"]) if row is not None else None

    def save_creative_node_board_layout(self, node_id: int, payload_json: str) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT project_id FROM creative_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
            if row is None:
                return
            conn.execute(
                """
                INSERT INTO creative_node_board_layouts(node_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (node_id, payload_json, now),
            )
            conn.execute(
                "UPDATE creative_projects SET updated_at = ? WHERE id = ?",
                (now, int(row["project_id"])),
            )

    def get_creative_node_board_layout(self, node_id: int) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM creative_node_board_layouts WHERE node_id = ?",
                (node_id,),
            ).fetchone()
        return str(row["payload_json"]) if row is not None else None

    def _creative_descendant_node_ids(self, project_id: int, node_id: int) -> list[int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, parent_id
                FROM creative_nodes
                WHERE project_id = ?
                ORDER BY sort_order, id
                """,
                (project_id,),
            ).fetchall()
        return self._ordered_descendant_ids_from_node_rows(rows, node_id)

    @staticmethod
    def _ordered_descendant_ids_from_node_rows(rows: Sequence[sqlite3.Row], node_id: int) -> list[int]:
        children_by_parent: dict[int | None, list[int]] = {}
        existing_ids: set[int] = set()
        for row in rows:
            current_id = int(row["id"])
            parent_id = row["parent_id"]
            existing_ids.add(current_id)
            children_by_parent.setdefault(
                int(parent_id) if parent_id is not None else None,
                [],
            ).append(current_id)
        if node_id not in existing_ids:
            return []
        ordered: list[int] = []

        def visit(current_id: int) -> None:
            ordered.append(current_id)
            for child_id in children_by_parent.get(current_id, []):
                visit(child_id)

        visit(node_id)
        return ordered

    @staticmethod
    def _backfill_creative_project_sort_order(conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id
            FROM creative_projects
            ORDER BY updated_at ASC, id ASC
            """
        ).fetchall()
        for index, row in enumerate(rows):
            conn.execute(
                "UPDATE creative_projects SET sort_order = ? WHERE id = ?",
                (index, int(row["id"])),
            )

    @staticmethod
    def _normalize_creative_project_sort_order(conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id
            FROM creative_projects
            ORDER BY sort_order ASC, updated_at ASC, id ASC
            """
        ).fetchall()
        for index, row in enumerate(rows):
            conn.execute(
                "UPDATE creative_projects SET sort_order = ? WHERE id = ?",
                (index, int(row["id"])),
            )

    @staticmethod
    def _next_creative_project_sort_order(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_order FROM creative_projects"
        ).fetchone()
        return int(row["next_order"]) if row is not None else 0

    @staticmethod
    def _creative_project_from_row(row: sqlite3.Row) -> CreativeProjectItem:
        return CreativeProjectItem(
            id=int(row["id"]),
            title=str(row["title"]),
            brief=str(row["brief"] or ""),
            language=str(row["language"] or "zh"),
            provider_name=str(row["provider_name"] or ""),
            model_name=str(row["model_name"] or ""),
            node_count=int(row["node_count"]),
            image_count=int(row["image_count"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            is_pinned=bool(int(row["is_pinned"] or 0)),
            copy_text=str(row["copy_text"] or ""),
        )

    @staticmethod
    def _creative_node_from_row(row: sqlite3.Row) -> CreativeNodeItem:
        parent_id = row["parent_id"]
        return CreativeNodeItem(
            id=int(row["id"]),
            project_id=int(row["project_id"]),
            parent_id=int(parent_id) if parent_id is not None else None,
            title=str(row["title"]),
            note=str(row["note"] or ""),
            search_query=str(row["search_query"] or ""),
            sort_order=int(row["sort_order"]),
            image_count=int(row["image_count"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _inspiration_project_from_row(row: sqlite3.Row) -> InspirationProjectItem:
        try:
            questions = json.loads(str(row["questions_json"] or "[]"))
        except json.JSONDecodeError:
            questions = []
        if not isinstance(questions, list):
            questions = []
        clean_questions = [
            str(question)
            for question in questions
            if str(question).strip()
        ]
        return InspirationProjectItem(
            id=int(row["id"]),
            title=str(row["title"]),
            brief=str(row["brief"]),
            answers=str(row["answers"] or ""),
            questions=clean_questions,
            provider_name=str(row["provider_name"]),
            model_name=str(row["model_name"]),
            term_count=int(row["term_count"]),
            selected_count=int(row["selected_count"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

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
        virtual_filter: str | None = None,
        excluded_folder_path_prefixes: Sequence[str] | None = None,
        excluded_collection_ids: Sequence[int] | None = None,
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
        virtual_clause, virtual_params = self._virtual_image_filter_clause(
            "i.id",
            "i.file_ext",
            virtual_filter,
        )
        if virtual_clause:
            clauses.append(virtual_clause)
            params.extend(virtual_params)
        self._append_excluded_folder_path_clauses(
            clauses,
            params,
            "i.file_path",
            excluded_folder_path_prefixes,
        )
        self._append_excluded_collection_clauses(
            clauses,
            params,
            "i.id",
            excluded_collection_ids,
        )
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
        with self.connect(readonly=True) as conn:
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

    def seed_default_ai_vision_collection_rules(
        self,
        default_paths: Sequence[Sequence[str]] = DEFAULT_AI_VISION_COLLECTION_PATHS,
    ) -> int:
        setting_key = "ai_vision.default_rules_seeded"
        with self.connect() as conn:
            seeded_row = conn.execute(
                "SELECT value FROM app_settings WHERE key = ?",
                (setting_key,),
            ).fetchone()
            if seeded_row is not None and str(seeded_row["value"]) == "1":
                return 0
            row = conn.execute("SELECT COUNT(*) AS count FROM ai_vision_collection_rules").fetchone()
            if int(row["count"]) > 0:
                now = utc_now_iso()
                conn.execute(
                    """
                    INSERT INTO app_settings(key, value, updated_at)
                    VALUES (?, '1', ?)
                    ON CONFLICT(key) DO UPDATE SET value = '1', updated_at = excluded.updated_at
                    """,
                    (setting_key, now),
                )
                return 0

        collection_ids_by_path = {
            tuple(path): collection.id
            for collection, path in self.collection_export_paths()
        }
        now = utc_now_iso()
        inserted = 0
        with self.connect() as conn:
            for path in default_paths:
                collection_id = collection_ids_by_path.get(tuple(path))
                if collection_id is None:
                    continue
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO ai_vision_collection_rules(
                        collection_id, mode, include_descendants, created_at, updated_at
                    )
                    VALUES (?, 'include', 1, ?, ?)
                    """,
                    (collection_id, now, now),
                )
                inserted += int(cur.rowcount)
            if inserted:
                conn.execute(
                    """
                    INSERT INTO app_settings(key, value, updated_at)
                    VALUES (?, '1', ?)
                    ON CONFLICT(key) DO UPDATE SET value = '1', updated_at = excluded.updated_at
                    """,
                    (setting_key, now),
                )
        return inserted

    def set_ai_vision_collection_rule(
        self,
        collection_id: int,
        *,
        mode: str,
        include_descendants: bool = True,
    ) -> None:
        if mode not in {"include", "exclude"}:
            raise ValueError("AI vision rule mode must be include or exclude")
        now = utc_now_iso()
        with self.connect() as conn:
            collection = conn.execute(
                "SELECT id FROM collections WHERE id = ?", (collection_id,)
            ).fetchone()
            if collection is None:
                raise ValueError("collection not found")
            conn.execute(
                """
                INSERT INTO ai_vision_collection_rules(
                    collection_id, mode, include_descendants, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(collection_id) DO UPDATE SET
                    mode = excluded.mode,
                    include_descendants = excluded.include_descendants,
                    updated_at = excluded.updated_at
                """,
                (collection_id, mode, 1 if include_descendants else 0, now, now),
            )

    def remove_ai_vision_collection_rule(self, collection_id: int) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM ai_vision_collection_rules WHERE collection_id = ?",
                (collection_id,),
            )
            return int(cur.rowcount) > 0

    def list_ai_vision_collection_rules_with_stats(
        self,
        *,
        provider_name: str,
        model_name: str,
        prompt_version: str = AI_VISION_PROMPT_VERSION,
    ) -> list[dict[str, object]]:
        path_by_id = {
            collection.id: " / ".join(path)
            for collection, path in self.collection_export_paths()
        }
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT r.collection_id, r.mode, r.include_descendants, c.name
                FROM ai_vision_collection_rules r
                JOIN collections c ON c.id = r.collection_id
                WHERE r.mode = 'include'
                ORDER BY r.mode DESC, c.name COLLATE NOCASE
                """
            ).fetchall()
        rules: list[dict[str, object]] = []
        for row in rows:
            collection_id = int(row["collection_id"])
            stats = self.ai_vision_rule_stats(
                collection_id=collection_id,
                include_descendants=bool(row["include_descendants"]),
                provider_name=provider_name,
                model_name=model_name,
                prompt_version=prompt_version,
            )
            rules.append(
                {
                    "collection_id": collection_id,
                    "name": str(row["name"]),
                    "path": path_by_id.get(collection_id, str(row["name"])),
                    "mode": str(row["mode"]),
                    "include_descendants": bool(row["include_descendants"]),
                    "stats": stats,
                }
            )
        return rules

    def ai_vision_rule_stats(
        self,
        *,
        collection_id: int,
        include_descendants: bool,
        provider_name: str,
        model_name: str,
        prompt_version: str = AI_VISION_PROMPT_VERSION,
    ) -> dict[str, int]:
        with self.connect() as conn:
            image_ids = self._ai_vision_rule_image_ids(
                conn,
                collection_id,
                include_descendants=include_descendants,
            )
            return self._ai_vision_stats_for_image_ids(
                conn,
                image_ids,
                provider_name=provider_name,
                model_name=model_name,
                prompt_version=prompt_version,
            )

    def ai_vision_stats(
        self,
        *,
        provider_name: str,
        model_name: str,
        prompt_version: str = AI_VISION_PROMPT_VERSION,
    ) -> dict[str, int]:
        with self.connect() as conn:
            image_ids = self._ai_vision_effective_image_ids(conn)
            return self._ai_vision_stats_for_image_ids(
                conn,
                image_ids,
                provider_name=provider_name,
                model_name=model_name,
                prompt_version=prompt_version,
            )

    def next_ai_vision_jobs(
        self,
        *,
        provider_name: str,
        model_name: str,
        prompt_version: str,
        limit: int,
    ) -> list[ImageItem]:
        with self.connect(readonly=True) as conn:
            image_ids = sorted(self._ai_vision_effective_image_ids(conn))
            if not image_ids:
                return []
            rows: list[sqlite3.Row] = []
            remaining = max(0, int(limit))
            for chunk in self._chunks(image_ids, self._sqlite_chunk_size):
                if remaining <= 0:
                    break
                placeholders = ",".join("?" for _ in chunk)
                chunk_rows = conn.execute(
                    f"""
                    SELECT i.*
                    FROM images i
                    LEFT JOIN ai_vision_tags t ON t.image_id = i.id
                    WHERE i.id IN ({placeholders})
                      AND (
                        t.image_id IS NULL
                        OR t.status IN ('pending', 'stale')
                        OR t.provider_name != ?
                        OR t.model_name != ?
                        OR t.prompt_version != ?
                      )
                    ORDER BY i.id
                    LIMIT ?
                    """,
                    (*chunk, provider_name, model_name, prompt_version, remaining),
                ).fetchall()
                rows.extend(chunk_rows)
                remaining = max(0, int(limit) - len(rows))
            return [self._image_from_row(row) for row in rows[: max(0, int(limit))]]

    def mark_ai_vision_processing(
        self,
        *,
        image_id: int,
        provider_name: str,
        model_name: str,
        prompt_version: str,
    ) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_vision_tags(
                    image_id, provider_name, model_name, prompt_version, status,
                    lighting_json, confidence_json, error_message, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 'processing', '[]', '{}', NULL, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    provider_name = excluded.provider_name,
                    model_name = excluded.model_name,
                    prompt_version = excluded.prompt_version,
                    status = 'processing',
                    error_message = NULL,
                    updated_at = excluded.updated_at
                """,
                (image_id, provider_name, model_name, prompt_version, now, now),
            )

    def upsert_ai_vision_success(
        self,
        *,
        image_id: int,
        provider_name: str,
        model_name: str,
        prompt_version: str,
        analysis: AIVisionAnalysis,
        source_modified_time_ns: int,
    ) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_vision_tags(
                    image_id, provider_name, model_name, prompt_version, status,
                    scene_location, environment_type, time_of_day, weather,
                    shot_scale, view_angle, lighting_json, confidence_json, notes,
                    error_message, source_modified_time_ns, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    provider_name = excluded.provider_name,
                    model_name = excluded.model_name,
                    prompt_version = excluded.prompt_version,
                    status = 'ready',
                    scene_location = excluded.scene_location,
                    environment_type = excluded.environment_type,
                    time_of_day = excluded.time_of_day,
                    weather = excluded.weather,
                    shot_scale = excluded.shot_scale,
                    view_angle = excluded.view_angle,
                    lighting_json = excluded.lighting_json,
                    confidence_json = excluded.confidence_json,
                    notes = excluded.notes,
                    error_message = NULL,
                    source_modified_time_ns = excluded.source_modified_time_ns,
                    updated_at = excluded.updated_at
                """,
                (
                    image_id,
                    provider_name,
                    model_name,
                    prompt_version,
                    analysis.scene_location,
                    analysis.environment_type,
                    analysis.time_of_day,
                    analysis.weather,
                    analysis.shot_scale,
                    analysis.view_angle,
                    json.dumps(analysis.lighting, ensure_ascii=False),
                    json.dumps(analysis.confidence, ensure_ascii=False),
                    analysis.notes,
                    source_modified_time_ns,
                    now,
                    now,
                ),
            )

    def mark_ai_vision_failed(
        self,
        *,
        image_id: int,
        provider_name: str,
        model_name: str,
        prompt_version: str,
        error_message: str,
    ) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_vision_tags(
                    image_id, provider_name, model_name, prompt_version, status,
                    lighting_json, confidence_json, error_message, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 'failed', '[]', '{}', ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    provider_name = excluded.provider_name,
                    model_name = excluded.model_name,
                    prompt_version = excluded.prompt_version,
                    status = 'failed',
                    scene_location = NULL,
                    environment_type = NULL,
                    time_of_day = NULL,
                    weather = NULL,
                    shot_scale = NULL,
                    view_angle = NULL,
                    lighting_json = '[]',
                    confidence_json = '{}',
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    image_id,
                    provider_name,
                    model_name,
                    prompt_version,
                    error_message[:2000],
                    now,
                    now,
                ),
            )

    def retry_failed_ai_vision(self) -> int:
        now = utc_now_iso()
        with self.connect() as conn:
            image_ids = sorted(self._ai_vision_effective_image_ids(conn))
            if not image_ids:
                return 0
            placeholders = ",".join("?" for _ in image_ids)
            cur = conn.execute(
                f"""
                UPDATE ai_vision_tags
                SET status = 'pending', error_message = NULL, updated_at = ?
                WHERE image_id IN ({placeholders})
                  AND status IN ('failed', 'processing')
                """,
                (now, *image_ids),
            )
            return int(cur.rowcount)

    def ai_vision_tags_for_image(self, image_id: int) -> dict[str, object] | None:
        with self.connect(readonly=True) as conn:
            row = conn.execute(
                "SELECT * FROM ai_vision_tags WHERE image_id = ?",
                (image_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            lighting = json.loads(str(row["lighting_json"] or "[]"))
        except json.JSONDecodeError:
            lighting = []
        try:
            confidence = json.loads(str(row["confidence_json"] or "{}"))
        except json.JSONDecodeError:
            confidence = {}
        return {
            "image_id": int(row["image_id"]),
            "provider_name": str(row["provider_name"]),
            "model_name": str(row["model_name"]),
            "prompt_version": str(row["prompt_version"]),
            "status": str(row["status"]),
            "scene_location": row["scene_location"],
            "environment_type": row["environment_type"],
            "time_of_day": row["time_of_day"],
            "weather": row["weather"],
            "shot_scale": row["shot_scale"],
            "view_angle": row["view_angle"],
            "lighting": lighting if isinstance(lighting, list) else [],
            "confidence": confidence if isinstance(confidence, dict) else {},
            "notes": str(row["notes"] or ""),
            "error_message": str(row["error_message"] or ""),
            "updated_at": str(row["updated_at"]),
        }

    def image_ids_matching_ai_vision(self, field: str, value: str) -> set[int]:
        if field == "lighting":
            with self.connect() as conn:
                rows = conn.execute(
                    """
                    SELECT image_id
                    FROM ai_vision_tags
                    WHERE status = 'ready'
                      AND lighting_json LIKE ?
                    """,
                    (f'%"{value}"%',),
                ).fetchall()
            return {int(row["image_id"]) for row in rows}
        if field not in {
            "scene_location",
            "environment_type",
            "time_of_day",
            "weather",
            "shot_scale",
            "view_angle",
        }:
            return set()
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT image_id
                FROM ai_vision_tags
                WHERE status = 'ready'
                  AND {field} = ?
                """,
                (value,),
            ).fetchall()
        return {int(row["image_id"]) for row in rows}

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

    def list_search_excluded_folder_prefixes(self) -> list[str]:
        raw = self.get_setting(SEARCH_EXCLUDED_FOLDER_PREFIXES_SETTING, "[]")
        try:
            payload = json.loads(raw or "[]")
        except json.JSONDecodeError:
            payload = []
        if not isinstance(payload, list):
            payload = []
        return self._compact_folder_prefixes(
            str(value)
            for value in payload
            if isinstance(value, str) and value.strip()
        )

    def set_search_excluded_folder_prefixes(self, prefixes: Iterable[str]) -> None:
        compacted = self._compact_folder_prefixes(prefixes)
        self.set_setting(
            SEARCH_EXCLUDED_FOLDER_PREFIXES_SETTING,
            json.dumps(compacted, ensure_ascii=False),
        )

    def add_search_excluded_folder_prefix(self, folder_path_prefix: str) -> bool:
        normalized = self._normalize_folder_prefix(folder_path_prefix)
        prefixes = self.list_search_excluded_folder_prefixes()
        if any(
            self._path_is_under_prefix(normalized, prefix, include_self=True)
            for prefix in prefixes
        ):
            return False
        prefixes.append(normalized)
        self.set_search_excluded_folder_prefixes(prefixes)
        return True

    def remove_search_excluded_folder_prefix(self, folder_path_prefix: str) -> bool:
        normalized = self._normalize_folder_prefix(folder_path_prefix)
        prefixes = self.list_search_excluded_folder_prefixes()
        remaining = [prefix for prefix in prefixes if prefix != normalized]
        if len(remaining) == len(prefixes):
            return False
        self.set_search_excluded_folder_prefixes(remaining)
        return True

    def list_search_excluded_collection_ids(self) -> list[int]:
        raw = self.get_setting(SEARCH_EXCLUDED_COLLECTION_IDS_SETTING, "[]")
        try:
            payload = json.loads(raw or "[]")
        except json.JSONDecodeError:
            payload = []
        if not isinstance(payload, list):
            payload = []
        ids: list[int] = []
        for value in payload:
            try:
                collection_id = int(value)
            except (TypeError, ValueError):
                continue
            if collection_id > 0:
                ids.append(collection_id)
        return self._compact_collection_ids(ids)

    def set_search_excluded_collection_ids(self, collection_ids: Iterable[int]) -> None:
        compacted = self._compact_collection_ids(collection_ids)
        self.set_setting(
            SEARCH_EXCLUDED_COLLECTION_IDS_SETTING,
            json.dumps(compacted, ensure_ascii=False),
        )

    def add_search_excluded_collection_id(self, collection_id: int) -> bool:
        collection_id = int(collection_id)
        if collection_id <= 0:
            return False
        ids = self.list_search_excluded_collection_ids()
        if self._collection_is_under_any(collection_id, ids, include_self=True):
            return False
        ids.append(collection_id)
        self.set_search_excluded_collection_ids(ids)
        return True

    def remove_search_excluded_collection_id(self, collection_id: int) -> bool:
        collection_id = int(collection_id)
        ids = self.list_search_excluded_collection_ids()
        remaining = [existing_id for existing_id in ids if existing_id != collection_id]
        if len(remaining) == len(ids):
            return False
        self.set_search_excluded_collection_ids(remaining)
        return True

    def thumbnail_paths_in_use(self) -> set[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT thumbnail_path
                FROM images
                WHERE thumbnail_path IS NOT NULL
                  AND TRIM(thumbnail_path) != ''
                """
            ).fetchall()
        return {str(row["thumbnail_path"]) for row in rows}

    @classmethod
    def _ai_vision_rule_image_ids(
        cls,
        conn: sqlite3.Connection,
        collection_id: int,
        *,
        include_descendants: bool,
    ) -> set[int]:
        collection_ids = (
            cls._collection_subtree_ids(conn, collection_id)
            if include_descendants
            else [collection_id]
        )
        if not collection_ids:
            return set()
        collection_placeholders = ",".join("?" for _ in collection_ids)
        image_placeholders = ",".join("?" for _ in SUPPORTED_IMAGE_EXTENSIONS)
        rows = conn.execute(
            f"""
            SELECT DISTINCT i.id
            FROM images i
            JOIN image_collections ic ON ic.image_id = i.id
            WHERE ic.collection_id IN ({collection_placeholders})
              AND i.is_missing = 0
              AND i.file_ext IN ({image_placeholders})
            """,
            (*collection_ids, *sorted(SUPPORTED_IMAGE_EXTENSIONS)),
        ).fetchall()
        return {int(row["id"]) for row in rows}

    @classmethod
    def _ai_vision_effective_image_ids(cls, conn: sqlite3.Connection) -> set[int]:
        rule_rows = conn.execute(
            """
            SELECT collection_id, mode, include_descendants
            FROM ai_vision_collection_rules
            WHERE mode = 'include'
            ORDER BY mode
            """
        ).fetchall()
        include_ids: set[int] = set()
        for row in rule_rows:
            ids = cls._ai_vision_rule_image_ids(
                conn,
                int(row["collection_id"]),
                include_descendants=bool(row["include_descendants"]),
            )
            include_ids.update(ids)
        return include_ids

    @staticmethod
    def _ai_vision_stats_for_image_ids(
        conn: sqlite3.Connection,
        image_ids: set[int],
        *,
        provider_name: str,
        model_name: str,
        prompt_version: str,
    ) -> dict[str, int]:
        stats = {
            "total": len(image_ids),
            "ready": 0,
            "failed": 0,
            "processing": 0,
            "stale": 0,
            "pending": 0,
            "skipped": 0,
        }
        if not image_ids:
            return stats
        rows: list[sqlite3.Row] = []
        for chunk in MetadataStore._chunks(sorted(image_ids), MetadataStore._sqlite_chunk_size):
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(
                conn.execute(
                    f"""
                    SELECT image_id, provider_name, model_name, prompt_version, status
                    FROM ai_vision_tags
                    WHERE image_id IN ({placeholders})
                    """,
                    tuple(chunk),
                ).fetchall()
            )
        seen: set[int] = set()
        for row in rows:
            image_id = int(row["image_id"])
            seen.add(image_id)
            if (
                str(row["provider_name"]) != provider_name
                or str(row["model_name"]) != model_name
                or str(row["prompt_version"]) != prompt_version
            ):
                stats["stale"] += 1
                continue
            status = str(row["status"])
            if status in stats:
                stats[status] += 1
            else:
                stats["pending"] += 1
        stats["pending"] += max(0, len(image_ids) - len(seen))
        return stats

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
        keys = set(row.keys())
        group_id = row["group_id"] if "group_id" in keys else None
        group_name = row["group_name"] if "group_name" in keys else ""
        return TagItem(
            id=int(row["id"]),
            tag_name=str(row["tag_name"]),
            tag_type=str(row["tag_type"]),
            created_at=str(row["created_at"]),
            group_id=int(group_id) if group_id is not None else None,
            group_name=str(group_name or ""),
        )

    @staticmethod
    def _tag_group_from_row(row: sqlite3.Row) -> TagGroupItem:
        return TagGroupItem(
            id=int(row["id"]),
            name=str(row["name"]),
            sort_order=int(row["sort_order"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
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

    @staticmethod
    def _mark_image_media_stale(
        conn: sqlite3.Connection,
        image_id: int,
        updated_at: str,
    ) -> None:
        conn.execute(
            """
            UPDATE images
            SET thumbnail_status = 'pending',
                thumbnail_path = NULL,
                embedding_status = 'pending'
            WHERE id = ?
            """,
            (image_id,),
        )
        conn.execute(
            """
            UPDATE embeddings
            SET status = 'stale', vector_blob = NULL, updated_at = ?
            WHERE image_id = ?
            """,
            (updated_at, image_id),
        )
        conn.execute(
            """
            UPDATE color_features
            SET status = 'stale', hist_blob = NULL, updated_at = ?
            WHERE image_id = ?
            """,
            (updated_at, image_id),
        )
        conn.execute(
            """
            UPDATE ai_vision_tags
            SET status = 'stale',
                scene_location = NULL,
                environment_type = NULL,
                time_of_day = NULL,
                weather = NULL,
                shot_scale = NULL,
                view_angle = NULL,
                lighting_json = '[]',
                confidence_json = '{}',
                notes = NULL,
                error_message = NULL,
                updated_at = ?
            WHERE image_id = ?
            """,
            (updated_at, image_id),
        )

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

    @classmethod
    def _compact_folder_prefixes(cls, prefixes: Iterable[str]) -> list[str]:
        normalized_prefixes: list[str] = []
        seen: set[str] = set()
        for prefix in prefixes:
            normalized = cls._normalize_folder_prefix(prefix)
            if normalized in seen:
                continue
            seen.add(normalized)
            normalized_prefixes.append(normalized)
        compacted: list[str] = []
        for prefix in sorted(normalized_prefixes, key=lambda value: (value.count(os.sep), value.casefold())):
            if any(cls._path_is_under_prefix(prefix, parent, include_self=True) for parent in compacted):
                continue
            compacted.append(prefix)
        return compacted

    @classmethod
    def _append_excluded_folder_path_clauses(
        cls,
        clauses: list[str],
        params: list[object],
        path_column: str,
        excluded_folder_path_prefixes: Sequence[str] | None,
    ) -> None:
        for prefix in cls._compact_folder_prefixes(excluded_folder_path_prefixes or []):
            clauses.append(f"{path_column} NOT LIKE ? ESCAPE '\\'")
            params.append(cls._folder_path_like(prefix))

    def _compact_collection_ids(self, collection_ids: Iterable[int]) -> list[int]:
        parent_by_id = {
            collection.id: collection.parent_id
            for collection in self.list_collections()
        }
        valid_ids: list[int] = []
        seen: set[int] = set()
        for value in collection_ids:
            try:
                collection_id = int(value)
            except (TypeError, ValueError):
                continue
            if collection_id <= 0 or collection_id not in parent_by_id or collection_id in seen:
                continue
            seen.add(collection_id)
            valid_ids.append(collection_id)
        compacted: list[int] = []
        for collection_id in sorted(
            valid_ids,
            key=lambda value: (self._collection_depth(value, parent_by_id), value),
        ):
            if self._collection_is_under_any(
                collection_id,
                compacted,
                parent_by_id=parent_by_id,
                include_self=True,
            ):
                continue
            compacted.append(collection_id)
        return compacted

    @staticmethod
    def _collection_depth(collection_id: int, parent_by_id: Mapping[int, int | None]) -> int:
        depth = 0
        seen: set[int] = set()
        parent_id = parent_by_id.get(collection_id)
        while parent_id is not None and parent_id not in seen:
            seen.add(parent_id)
            depth += 1
            parent_id = parent_by_id.get(parent_id)
        return depth

    def _collection_is_under_any(
        self,
        collection_id: int,
        ancestor_ids: Iterable[int],
        *,
        parent_by_id: Mapping[int, int | None] | None = None,
        include_self: bool,
    ) -> bool:
        parent_by_id = parent_by_id or {
            collection.id: collection.parent_id
            for collection in self.list_collections()
        }
        ancestor_set = {int(ancestor_id) for ancestor_id in ancestor_ids}
        current_id: int | None = int(collection_id)
        seen: set[int] = set()
        while current_id is not None and current_id not in seen:
            seen.add(current_id)
            if (include_self or current_id != collection_id) and current_id in ancestor_set:
                return True
            current_id = parent_by_id.get(current_id)
        return False

    def _append_excluded_collection_clauses(
        self,
        clauses: list[str],
        params: list[object],
        image_id_column: str,
        excluded_collection_ids: Sequence[int] | None,
    ) -> None:
        collection_ids: set[int] = set()
        for collection_id in self._compact_collection_ids(excluded_collection_ids or []):
            collection_ids.update(self.collection_descendant_ids(collection_id))
        if not collection_ids:
            return
        placeholders = ",".join("?" for _ in collection_ids)
        clauses.append(
            f"""
            NOT EXISTS (
                SELECT 1
                FROM image_collections excluded_ic
                WHERE excluded_ic.image_id = {image_id_column}
                  AND excluded_ic.collection_id IN ({placeholders})
            )
            """
        )
        params.extend(sorted(collection_ids))

    @staticmethod
    def _path_is_under_prefix(path: str, folder_path_prefix: str, *, include_self: bool) -> bool:
        normalized_path = os.path.abspath(os.path.expanduser(path)).rstrip(os.sep) or os.sep
        normalized_prefix = os.path.abspath(os.path.expanduser(folder_path_prefix)).rstrip(os.sep) or os.sep
        if include_self and normalized_path == normalized_prefix:
            return True
        if normalized_prefix == os.sep:
            return normalized_path.startswith(os.sep)
        return normalized_path.startswith(f"{normalized_prefix}{os.sep}")

    @classmethod
    def _replace_path_prefix(
        cls,
        path: str,
        old_prefix: str,
        new_prefix: str,
    ) -> str | None:
        normalized_path = os.path.abspath(os.path.expanduser(path)).rstrip(os.sep) or os.sep
        normalized_old = cls._normalize_folder_prefix(old_prefix)
        normalized_new = cls._normalize_folder_prefix(new_prefix)
        if normalized_path == normalized_old:
            return normalized_new
        if not cls._path_is_under_prefix(normalized_path, normalized_old, include_self=False):
            return None
        relative_path = os.path.relpath(normalized_path, normalized_old)
        return os.path.normpath(os.path.join(normalized_new, relative_path))

    @staticmethod
    def _image_extension_clause(column: str) -> str:
        placeholders = ",".join("?" for _ in SUPPORTED_IMAGE_EXTENSIONS)
        return f"{column} IN ({placeholders})"

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None:
            return None
        return int(value)

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
    def _chunks(values: Sequence[int], size: int) -> Iterator[list[int]]:
        iterator = iter(values)
        while True:
            chunk = list(islice(iterator, max(1, int(size))))
            if not chunk:
                return
            yield chunk

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
