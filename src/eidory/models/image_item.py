from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FolderItem:
    id: int
    folder_path: str
    import_mode: str
    added_at: str
    last_scanned_at: str | None
    is_active: bool


@dataclass(frozen=True)
class CollectionItem:
    id: int
    parent_id: int | None
    name: str
    sort_order: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ImageItem:
    id: int
    folder_id: int
    file_path: str
    file_name: str
    file_ext: str
    file_size: int
    width: int | None
    height: int | None
    created_at: str | None
    modified_at: str | None
    modified_time_ns: int
    imported_at: str
    last_seen_at: str
    thumbnail_path: str | None
    thumbnail_status: str
    embedding_status: str
    is_missing: bool
    is_favorite: bool
    note: str | None
    duration_ms: int | None = None
    score: float | None = None


@dataclass(frozen=True)
class TagItem:
    id: int
    tag_name: str
    tag_type: str
    created_at: str


@dataclass(frozen=True)
class SavedViewItem:
    id: int
    name: str
    payload_json: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class InspirationProjectItem:
    id: int
    title: str
    brief: str
    answers: str
    questions: list[str]
    provider_name: str
    model_name: str
    term_count: int
    selected_count: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class TemporaryProjectItem:
    id: int
    name: str
    image_count: int
    created_at: str
    updated_at: str
    summary: str = ""
    color_hex: str = ""
