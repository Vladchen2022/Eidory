from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from eidory.core.color_features import encode_image_color
from eidory.core.image_loader import open_local_image
from eidory.core.media_types import (
    SUPPORTED_MEDIA_EXTENSIONS,
    is_supported_image,
    is_supported_media,
    is_supported_video,
)
from eidory.core.metadata_store import MetadataStore
from eidory.core.thumbnailer import Thumbnailer
from eidory.core.video_metadata import read_video_metadata


@dataclass(frozen=True)
class ScanResult:
    folder_id: int
    scanned_files: int
    new_files: int
    changed_files: int
    unchanged_files: int
    missing_marked: int
    thumbnail_failures: int
    image_ids: tuple[int, ...]
    removed_thumbnail_paths: tuple[str, ...] = ()


ScanProgressCallback = Callable[[int, str, str], None]


@dataclass(frozen=True)
class _ScannedMedia:
    image_id: int
    file_path: str
    state: str
    duration_ms: int | None


@dataclass(frozen=True)
class _PendingMedia:
    folder_id: int
    file_path: str
    file_size: int
    width: int | None
    height: int | None
    created_time_ns: int | None
    modified_time_ns: int
    duration_ms: int | None


SCAN_UPSERT_BATCH_SIZE = 200


class ImageScanner:
    def __init__(self, store: MetadataStore, thumbnailer: Thumbnailer):
        self.store = store
        self.thumbnailer = thumbnailer

    def scan_folder(
        self,
        folder_path: str,
        *,
        on_progress: ScanProgressCallback | None = None,
        skip_paths: set[str] | None = None,
    ) -> ScanResult:
        return self._scan_folder(
            folder_path,
            on_progress=on_progress,
            mark_missing=True,
            skip_paths=skip_paths,
        )

    def scan_folder_new_only(
        self,
        folder_path: str,
        *,
        on_progress: ScanProgressCallback | None = None,
        skip_paths: set[str] | None = None,
    ) -> ScanResult:
        return self._scan_folder(
            folder_path,
            on_progress=on_progress,
            mark_missing=False,
            skip_paths=skip_paths,
        )

    def _scan_folder(
        self,
        folder_path: str,
        *,
        on_progress: ScanProgressCallback | None,
        mark_missing: bool,
        skip_paths: set[str] | None,
    ) -> ScanResult:
        root = os.path.abspath(os.path.expanduser(folder_path))
        if not os.path.isdir(root):
            folder = self.store.get_folder_by_path(root)
            if folder is None:
                raise FileNotFoundError(f"folder does not exist: {root}")
            thumbnail_paths, removed = self.store.remove_folder_from_library(folder.id)
            return ScanResult(
                folder_id=folder.id,
                scanned_files=0,
                new_files=0,
                changed_files=0,
                unchanged_files=0,
                missing_marked=removed,
                thumbnail_failures=0,
                image_ids=(),
                removed_thumbnail_paths=tuple(thumbnail_paths),
            )

        folder_id = self.store.add_folder(root)
        seen_paths: list[str] = []
        scanned = 0
        new_files = 0
        changed_files = 0
        unchanged_files = 0
        image_ids: list[int] = []
        scanned_media: list[_ScannedMedia] = []
        pending_media: list[_PendingMedia] = []
        normalized_skip_paths = {
            os.path.abspath(os.path.expanduser(path))
            for path in (skip_paths or set())
        }

        def flush_pending_media() -> None:
            nonlocal new_files, changed_files, unchanged_files
            if not pending_media:
                return
            batch = list(pending_media)
            pending_media.clear()
            batch_media, batch_new, batch_changed, batch_unchanged, batch_image_ids = (
                self._upsert_pending_media(batch, on_progress=on_progress)
            )
            scanned_media.extend(batch_media)
            image_ids.extend(batch_image_ids)
            new_files += batch_new
            changed_files += batch_changed
            unchanged_files += batch_unchanged

        for file_path in self._iter_image_files(root):
            if file_path in normalized_skip_paths:
                seen_paths.append(file_path)
                continue
            scanned += 1
            seen_paths.append(file_path)
            stat = os.stat(file_path, follow_symlinks=False)
            width, height, duration_ms = self._read_media_metadata(file_path)
            pending_media.append(
                _PendingMedia(
                    folder_id=folder_id,
                    file_path=file_path,
                    file_size=stat.st_size,
                    width=width,
                    height=height,
                    created_time_ns=getattr(stat, "st_birthtime_ns", None),
                    modified_time_ns=stat.st_mtime_ns,
                    duration_ms=duration_ms,
                )
            )
            if len(pending_media) >= SCAN_UPSERT_BATCH_SIZE:
                flush_pending_media()

        flush_pending_media()

        thumbnail_failures = self._refresh_media_derivatives(scanned_media)

        removed_thumbnail_paths: list[str] = []
        missing_marked = 0
        if mark_missing:
            removed_thumbnail_paths, missing_marked = self.store.remove_unseen_images_for_folder(
                folder_id,
                seen_paths,
            )
        self.store.finish_folder_scan(folder_id)
        return ScanResult(
            folder_id=folder_id,
            scanned_files=scanned,
            new_files=new_files,
            changed_files=changed_files,
            unchanged_files=unchanged_files,
            missing_marked=missing_marked,
            thumbnail_failures=thumbnail_failures,
            image_ids=tuple(image_ids),
            removed_thumbnail_paths=tuple(removed_thumbnail_paths),
        )

    def _upsert_pending_media(
        self,
        pending_media: list[_PendingMedia],
        *,
        on_progress: ScanProgressCallback | None,
    ) -> tuple[list[_ScannedMedia], int, int, int, list[int]]:
        records = [
            {
                "folder_id": item.folder_id,
                "file_path": item.file_path,
                "file_size": item.file_size,
                "width": item.width,
                "height": item.height,
                "created_time_ns": item.created_time_ns,
                "modified_time_ns": item.modified_time_ns,
                "duration_ms": item.duration_ms,
            }
            for item in pending_media
        ]
        results = self.store.upsert_images(records)
        scanned_media: list[_ScannedMedia] = []
        image_ids: list[int] = []
        video_ids: list[int] = []
        new_files = 0
        changed_files = 0
        unchanged_files = 0
        for item, (image_id, state) in zip(pending_media, results):
            image_ids.append(image_id)
            scanned_media.append(
                _ScannedMedia(
                    image_id=image_id,
                    file_path=item.file_path,
                    state=state,
                    duration_ms=item.duration_ms,
                )
            )

            if state == "new":
                new_files += 1
            elif state == "changed":
                changed_files += 1
            else:
                unchanged_files += 1

            if is_supported_video(item.file_path):
                video_ids.append(image_id)

            if on_progress is not None:
                on_progress(image_id, state, item.file_path)
        self.store.mark_embeddings_not_required(video_ids)
        return scanned_media, new_files, changed_files, unchanged_files, image_ids

    def import_files(
        self,
        file_paths: list[str],
        *,
        on_progress: ScanProgressCallback | None = None,
    ) -> ScanResult:
        normalized_paths = []
        for file_path in file_paths:
            normalized = os.path.abspath(os.path.expanduser(file_path))
            if (
                os.path.isfile(normalized)
                and not os.path.islink(normalized)
                and is_supported_media(normalized)
            ):
                normalized_paths.append(normalized)
        normalized_paths = sorted(set(normalized_paths))
        if not normalized_paths:
            raise FileNotFoundError("no supported media files to import")

        first_parent = os.path.dirname(normalized_paths[0])
        folder_id = self.store.add_folder(first_parent)
        scanned = 0
        new_files = 0
        changed_files = 0
        unchanged_files = 0
        image_ids: list[int] = []
        folder_ids_by_path = {first_parent: folder_id}
        pending_media: list[_PendingMedia] = []

        for file_path in normalized_paths:
            scanned += 1
            parent = os.path.dirname(file_path)
            current_folder_id = folder_ids_by_path.get(parent)
            if current_folder_id is None:
                current_folder_id = self.store.add_folder(parent)
                folder_ids_by_path[parent] = current_folder_id
            stat = os.stat(file_path, follow_symlinks=False)
            width, height, duration_ms = self._read_media_metadata(file_path)
            pending_media.append(
                _PendingMedia(
                    folder_id=current_folder_id,
                    file_path=file_path,
                    file_size=stat.st_size,
                    width=width,
                    height=height,
                    created_time_ns=getattr(stat, "st_birthtime_ns", None),
                    modified_time_ns=stat.st_mtime_ns,
                    duration_ms=duration_ms,
                )
            )

        scanned_media, new_files, changed_files, unchanged_files, image_ids = (
            self._upsert_pending_media(pending_media, on_progress=on_progress)
        )
        thumbnail_failures = self._refresh_media_derivatives(scanned_media)

        return ScanResult(
            folder_id=folder_id,
            scanned_files=scanned,
            new_files=new_files,
            changed_files=changed_files,
            unchanged_files=unchanged_files,
            missing_marked=0,
            thumbnail_failures=thumbnail_failures,
            image_ids=tuple(image_ids),
        )

    @staticmethod
    def _iter_image_files(root: str) -> list[str]:
        paths: list[str] = []
        for current_root, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not dirname.startswith(".")
                and not os.path.islink(os.path.join(current_root, dirname))
            ]
            for filename in filenames:
                if filename.startswith("."):
                    continue
                file_path = os.path.join(current_root, filename)
                if os.path.islink(file_path):
                    continue
                if Path(filename).suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS:
                    paths.append(os.path.abspath(file_path))
        paths.sort()
        return paths

    @staticmethod
    def _read_dimensions(file_path: str) -> tuple[int | None, int | None]:
        try:
            with open_local_image(file_path) as image:
                return int(image.width), int(image.height)
        except Exception:
            return None, None

    def _read_media_metadata(self, file_path: str) -> tuple[int | None, int | None, int | None]:
        if is_supported_image(file_path):
            width, height = self._read_dimensions(file_path)
            return width, height, None
        if is_supported_video(file_path):
            metadata = read_video_metadata(file_path)
            return metadata.width, metadata.height, metadata.duration_ms
        return None, None, None

    def _generate_thumbnail(
        self,
        image_id: int,
        file_path: str,
        *,
        duration_ms: int | None = None,
    ) -> Path:
        if is_supported_video(file_path):
            try:
                return self.thumbnailer.generate_video(
                    image_id,
                    file_path,
                    duration_ms=duration_ms,
                )
            except TypeError:
                return self.thumbnailer.generate_video(image_id, file_path)
        return self.thumbnailer.generate(image_id, file_path)

    def _refresh_media_derivatives(self, scanned_media: list[_ScannedMedia]) -> int:
        if not scanned_media:
            return 0

        changed_thumbnail_ids = {
            item.image_id for item in scanned_media if item.state in {"new", "changed"}
        }
        thumbnail_ids = changed_thumbnail_ids | self.store.thumbnail_ids_needing_generation(
            [item.image_id for item in scanned_media]
        )
        thumbnail_failures = 0
        thumbnail_updates: list[tuple[int, str | None, str]] = []
        for item in scanned_media:
            if item.image_id not in thumbnail_ids:
                continue
            try:
                thumbnail_path = self._generate_thumbnail(
                    item.image_id,
                    item.file_path,
                    duration_ms=item.duration_ms,
                )
                thumbnail_updates.append((item.image_id, str(thumbnail_path), "ready"))
            except Exception:
                thumbnail_failures += 1
                thumbnail_updates.append((item.image_id, None, "failed"))
        self.store.update_thumbnails(thumbnail_updates)

        image_items = [
            item for item in scanned_media if is_supported_image(item.file_path)
        ]
        color_ids = self.store.color_feature_ids_needing_generation(
            [item.image_id for item in image_items]
        )
        color_successes = []
        color_failures = []
        for item in image_items:
            if item.image_id in color_ids:
                try:
                    color_successes.append((item.image_id, encode_image_color(item.file_path)))
                except Exception as exc:
                    color_failures.append((item.image_id, str(exc)))
        self.store.upsert_color_feature_successes(color_successes)
        self.store.mark_color_features_failed(color_failures)
        return thumbnail_failures

    def _update_color_feature(
        self,
        image_id: int,
        file_path: str,
        *,
        already_needed: bool = False,
    ) -> None:
        if not already_needed and not self.store.color_feature_needs_generation(image_id):
            return
        try:
            self.store.upsert_color_feature_success(
                image_id=image_id,
                vector=encode_image_color(file_path),
            )
        except Exception as exc:
            self.store.mark_color_feature_failed(image_id, str(exc))
