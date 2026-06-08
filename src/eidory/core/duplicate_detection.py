from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
        try:
            sha = _file_sha256(path)
            dhash = image_dhash(path)
        except Exception:
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
