from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from eidory.core.metadata_store import MetadataStore
from eidory.models import ImageItem


UNASSIGNED_EXPORT_FOLDER = "未归类"


@dataclass(frozen=True)
class ExportResult:
    target_dir: Path
    copied: int = 0
    skipped_missing: int = 0
    failed: int = 0
    directories: int = 0


def export_images_to_directory(
    images: Iterable[ImageItem],
    target_dir: Path | str,
) -> ExportResult:
    target = Path(target_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped_missing = 0
    failed = 0
    for image in images:
        outcome = _copy_image_file(image, target)
        if outcome == "copied":
            copied += 1
        elif outcome == "missing":
            skipped_missing += 1
        else:
            failed += 1
    return ExportResult(
        target_dir=target,
        copied=copied,
        skipped_missing=skipped_missing,
        failed=failed,
        directories=1,
    )


def export_library_to_directory(
    store: MetadataStore,
    target_dir: Path | str,
) -> ExportResult:
    target = Path(target_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped_missing = 0
    failed = 0
    directories = 0

    for collection, segments in store.collection_export_paths():
        export_dir = target.joinpath(*safe_export_segments(segments))
        export_dir.mkdir(parents=True, exist_ok=True)
        directories += 1
        for image in store.list_images_for_collection_direct(collection.id):
            outcome = _copy_image_file(image, export_dir)
            if outcome == "copied":
                copied += 1
            elif outcome == "missing":
                skipped_missing += 1
            else:
                failed += 1

    unassigned = store.list_images_without_collections()
    if unassigned:
        export_dir = target / UNASSIGNED_EXPORT_FOLDER
        export_dir.mkdir(parents=True, exist_ok=True)
        directories += 1
        for image in unassigned:
            outcome = _copy_image_file(image, export_dir)
            if outcome == "copied":
                copied += 1
            elif outcome == "missing":
                skipped_missing += 1
            else:
                failed += 1

    return ExportResult(
        target_dir=target,
        copied=copied,
        skipped_missing=skipped_missing,
        failed=failed,
        directories=directories,
    )


def _copy_image_file(image: ImageItem, target_dir: Path) -> str:
    source = Path(image.file_path)
    if image.is_missing or not source.is_file():
        return "missing"
    try:
        destination = _unique_destination_path(
            target_dir / _safe_filename(image.file_name, image.id, image.file_ext)
        )
        shutil.copy2(source, destination)
        return "copied"
    except OSError:
        return "failed"


def _unique_destination_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 2
    while True:
        candidate = parent / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def safe_export_segments(names: Iterable[str]) -> tuple[str, ...]:
    segments = tuple(_safe_path_segment(name) for name in names)
    return segments or (UNASSIGNED_EXPORT_FOLDER,)


def _safe_path_segment(name: str) -> str:
    clean = re.sub(r"[\x00/]+", "_", name.strip())
    clean = clean.strip(" .")
    return clean[:120] or "未命名"


def _safe_filename(file_name: str, image_id: int, file_ext: str) -> str:
    clean = re.sub(r"[\x00/]+", "_", file_name.strip())
    clean = clean.strip(" .")
    if clean:
        return clean[:180]
    suffix = file_ext if file_ext.startswith(".") else f".{file_ext}" if file_ext else ""
    return f"image-{image_id}{suffix}"
