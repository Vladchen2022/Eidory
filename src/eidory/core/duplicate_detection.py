from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from PIL import Image, ImageOps

from eidory.core.image_loader import open_local_image
from eidory.core.media_types import is_supported_image
from eidory.models import ImageItem


@dataclass(frozen=True)
class DuplicateMember:
    image: ImageItem
    folder_label: str
    file_sha256: str
    dhash: int | None


@dataclass(frozen=True)
class DuplicateGroup:
    kind: str
    reason: str
    members: tuple[DuplicateMember, ...]


@dataclass(frozen=True)
class NearDuplicateCandidate:
    image: ImageItem
    distance: int
    similarity: float
    hash_source: str


@dataclass(frozen=True)
class ImageDHashRecord:
    image: ImageItem
    dhash: int
    hash_source: str


@dataclass(frozen=True)
class ImageHashCacheRecord:
    image_id: int
    file_path: str
    file_size: int
    modified_time_ns: int
    file_sha256: str
    dhash: int
    hash_source: str
    hash_source_size: int
    hash_source_modified_time_ns: int


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, value: int) -> int:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


class _BKNode:
    def __init__(self, value: int, image_id: int):
        self.value = value
        self.image_ids = [image_id]
        self.children: dict[int, _BKNode] = {}


class _BKTree:
    def __init__(self) -> None:
        self.root: _BKNode | None = None

    def insert(self, value: int, image_id: int) -> None:
        if self.root is None:
            self.root = _BKNode(value, image_id)
            return
        node = self.root
        while True:
            distance = hamming_distance(value, node.value)
            if distance == 0:
                node.image_ids.append(image_id)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value, image_id)
                return
            node = child

    def query(self, value: int, max_distance: int) -> list[int]:
        if self.root is None:
            return []
        matches: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming_distance(value, node.value)
            if distance <= max_distance:
                matches.extend(node.image_ids)
            low = max(0, distance - max_distance)
            high = distance + max_distance
            for child_distance, child in node.children.items():
                if low <= child_distance <= high:
                    stack.append(child)
        return matches


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def find_duplicate_groups(
    images: list[ImageItem],
    *,
    folder_label_for_image: dict[int, str] | None = None,
    hash_records: Mapping[int, ImageHashCacheRecord] | None = None,
    on_hash_record: Callable[[ImageHashCacheRecord], None] | None = None,
    near_distance: int = 8,
) -> list[DuplicateGroup]:
    members: dict[int, DuplicateMember] = {}
    exact_by_hash: dict[str, list[int]] = {}
    tree = _BKTree()
    dsu = _DisjointSet()

    for image in images:
        if image.is_missing or not is_supported_image(image.file_path):
            continue
        path = Path(image.file_path)
        if not path.is_file():
            continue
        hash_record = _valid_hash_record_for_image(
            image,
            hash_records or {},
            prefer_thumbnail=False,
        )
        if hash_record is None:
            hash_record = build_image_hash_cache_record(image, prefer_thumbnail=False)
            if hash_record is None:
                continue
            if on_hash_record is not None:
                on_hash_record(hash_record)
        sha = hash_record.file_sha256
        dhash = hash_record.dhash
        if dhash is None:
            continue
        folder_label = (
            folder_label_for_image.get(image.id, "")
            if folder_label_for_image is not None
            else ""
        )
        members[image.id] = DuplicateMember(
            image=image,
            folder_label=folder_label,
            file_sha256=sha,
            dhash=dhash,
        )
        exact_by_hash.setdefault(sha, []).append(image.id)
        dsu.find(image.id)
        if dhash is not None:
            for match_id in tree.query(dhash, near_distance):
                if members[match_id].file_sha256 != sha:
                    dsu.union(image.id, match_id)
            tree.insert(dhash, image.id)

    groups: list[DuplicateGroup] = []
    exact_group_ids: set[frozenset[int]] = set()
    for sha, image_ids in exact_by_hash.items():
        if len(image_ids) < 2:
            continue
        exact_group_ids.add(frozenset(image_ids))
        group_members = _sort_members(members[image_id] for image_id in image_ids)
        groups.append(
            DuplicateGroup(
                kind="exact",
                reason=f"文件内容完全相同：SHA-256 {sha[:12]}",
                members=tuple(group_members),
            )
        )

    near_by_root: dict[int, list[int]] = {}
    for image_id in members:
        root = dsu.find(image_id)
        near_by_root.setdefault(root, []).append(image_id)
    for image_ids in near_by_root.values():
        if len(image_ids) < 2:
            continue
        image_id_set = frozenset(image_ids)
        if image_id_set in exact_group_ids:
            continue
        group_members = _sort_members(members[image_id] for image_id in image_ids)
        groups.append(
            DuplicateGroup(
                kind="near",
                reason=f"感知哈希距离 <= {near_distance}，可能是同图不同尺寸或近似变体",
                members=tuple(group_members),
            )
        )

    groups.sort(key=lambda group: (0 if group.kind == "exact" else 1, -len(group.members), group.reason))
    return groups


def find_near_duplicate_candidates(
    source_path: str | Path,
    images: list[ImageItem] | None = None,
    *,
    hash_records: list[ImageDHashRecord] | None = None,
    near_distance: int = 8,
    limit: int = 5,
    include_same_path: bool = False,
) -> list[NearDuplicateCandidate]:
    source = Path(source_path)
    if not source.is_file() or not is_supported_image(str(source)):
        return []
    try:
        source_hash = image_dhash(source)
    except Exception:
        return []

    source_resolved = _safe_resolve(source)
    candidates: list[NearDuplicateCandidate] = []
    records = hash_records if hash_records is not None else build_image_dhash_records(images or [])
    for record in records:
        image = record.image
        image_path = Path(image.file_path)
        if (
            not include_same_path
            and source_resolved is not None
            and _safe_resolve(image_path) == source_resolved
        ):
            continue
        distance = hamming_distance(source_hash, record.dhash)
        if distance > near_distance:
            continue
        similarity = 1.0 - (distance / 64.0)
        candidates.append(
            NearDuplicateCandidate(
                image=image,
                distance=distance,
                similarity=max(0.0, min(1.0, similarity)),
                hash_source=record.hash_source,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.distance,
            -(candidate.image.width or 0) * (candidate.image.height or 0),
            -candidate.image.file_size,
            candidate.image.file_name.casefold(),
        )
    )
    return candidates[:limit]


def build_image_dhash_records(
    images: list[ImageItem],
    *,
    hash_records: Mapping[int, ImageHashCacheRecord] | None = None,
    on_hash_record: Callable[[ImageHashCacheRecord], None] | None = None,
) -> list[ImageDHashRecord]:
    records: list[ImageDHashRecord] = []
    for image in images:
        if image.is_missing or not is_supported_image(image.file_path):
            continue
        hash_record = _valid_hash_record_for_image(image, hash_records or {})
        if hash_record is None:
            hash_record = build_image_hash_cache_record(image)
            if hash_record is None:
                continue
            if on_hash_record is not None:
                on_hash_record(hash_record)
        if hash_record.dhash is None:
            continue
        records.append(
            ImageDHashRecord(
                image=image,
                dhash=hash_record.dhash,
                hash_source=hash_record.hash_source,
            )
        )
    return records


def build_image_hash_cache_record(
    image: ImageItem,
    *,
    prefer_thumbnail: bool = True,
) -> ImageHashCacheRecord | None:
    if image.is_missing or not is_supported_image(image.file_path):
        return None
    path = Path(image.file_path)
    source_stat = _safe_stat(path)
    if source_stat is None:
        return None
    hash_path = _hash_source_for_image(image, prefer_thumbnail=prefer_thumbnail)
    if hash_path is None:
        return None
    hash_source_stat = _safe_stat(hash_path)
    if hash_source_stat is None:
        return None
    try:
        sha = _file_sha256(path)
        dhash = image_dhash(hash_path)
    except Exception:
        return None
    return ImageHashCacheRecord(
        image_id=int(image.id),
        file_path=str(path),
        file_size=int(source_stat.st_size),
        modified_time_ns=int(source_stat.st_mtime_ns),
        file_sha256=sha,
        dhash=int(dhash),
        hash_source=str(hash_path),
        hash_source_size=int(hash_source_stat.st_size),
        hash_source_modified_time_ns=int(hash_source_stat.st_mtime_ns),
    )


def image_dhash(path: Path, *, hash_size: int = 8) -> int:
    with open_local_image(path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        image = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = image.tobytes()
    bits = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            bits <<= 1
            if pixels[offset + column] > pixels[offset + column + 1]:
                bits |= 1
    return bits


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sort_members(members: Iterable[DuplicateMember]) -> list[DuplicateMember]:
    return sorted(
        list(members),
        key=lambda member: (
            -(member.image.width or 0) * (member.image.height or 0),
            -member.image.file_size,
            member.image.file_name.casefold(),
        ),
    )


def _hash_source_for_image(image: ImageItem, *, prefer_thumbnail: bool = True) -> Path | None:
    if prefer_thumbnail and image.thumbnail_status == "ready" and image.thumbnail_path:
        thumbnail_path = Path(image.thumbnail_path)
        if thumbnail_path.is_file():
            return thumbnail_path
    path = Path(image.file_path)
    return path if path.is_file() else None


def _valid_hash_record_for_image(
    image: ImageItem,
    hash_records: Mapping[int, ImageHashCacheRecord],
    *,
    prefer_thumbnail: bool = True,
) -> ImageHashCacheRecord | None:
    record = hash_records.get(int(image.id))
    if record is None:
        return None
    if record.file_path != image.file_path:
        return None
    path = Path(image.file_path)
    source_stat = _safe_stat(path)
    if source_stat is None:
        return None
    if int(source_stat.st_size) != int(record.file_size):
        return None
    if int(source_stat.st_mtime_ns) != int(record.modified_time_ns):
        return None
    hash_path = _hash_source_for_image(image, prefer_thumbnail=prefer_thumbnail)
    if hash_path is None or str(hash_path) != record.hash_source:
        return None
    hash_source_stat = _safe_stat(hash_path)
    if hash_source_stat is None:
        return None
    if int(hash_source_stat.st_size) != int(record.hash_source_size):
        return None
    if int(hash_source_stat.st_mtime_ns) != int(record.hash_source_modified_time_ns):
        return None
    if not record.file_sha256 or record.dhash is None:
        return None
    return record


def _safe_stat(path: Path):
    try:
        return path.stat()
    except OSError:
        return None


def _safe_resolve(path: Path) -> Path | None:
    try:
        return path.resolve()
    except OSError:
        return None
