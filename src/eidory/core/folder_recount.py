from __future__ import annotations

import os
import stat
from collections import Counter
from dataclasses import dataclass

from eidory.core.metadata_store import MetadataStore


@dataclass(frozen=True)
class FolderRecount:
    before: int
    available: int
    missing: int
    recovered: int
    newly_missing: int
    unchecked: int
    locations: tuple[tuple[str, int], ...]
    errors: tuple[str, ...]


def recount_collection(store: MetadataStore, collection_id: int) -> FolderRecount:
    """Reconcile linked file availability; never import, delete or decode files."""
    records = store.collection_file_records(collection_id)
    before = sum(not row['is_missing'] for row in records)
    updates = []
    unchecked = 0
    errors: dict[str, str] = {}
    roots: dict[str, bool] = {}
    for row in records:
        path = str(row['file_path'])
        root = str(row['folder_path'])
        if root not in roots:
            try:
                roots[root] = stat.S_ISDIR(os.stat(root).st_mode)
                if not roots[root]:
                    errors[root] = f'目录不可用，保留原记录：{root}'
            except OSError as exc:
                roots[root] = False
                errors[root] = f'目录不可用，保留原记录：{root}（{exc.strerror}）'
        if not roots[root]:
            unchecked += 1
            continue
        try:
            missing = not stat.S_ISREG(os.stat(path).st_mode)
        except FileNotFoundError:
            missing = True
        except OSError as exc:
            unchecked += 1
            errors[path] = f'无法核对，保留原记录：{path}（{exc.strerror}）'
            continue
        if missing != bool(row['is_missing']):
            updates.append((int(missing), int(row['id']), path,
                            int(row['is_missing']), int(row['modified_time_ns'])))
    store.apply_file_availability(updates)
    final = store.collection_file_records(collection_id)
    previous = {int(row['id']): bool(row['is_missing']) for row in records}
    return FolderRecount(
        before=before,
        available=sum(not row['is_missing'] for row in final),
        missing=sum(bool(row['is_missing']) for row in final),
        recovered=sum(previous.get(int(row['id']), False) and not row['is_missing'] for row in final),
        newly_missing=sum(not previous.get(int(row['id']), True) and bool(row['is_missing']) for row in final),
        unchecked=unchecked,
        locations=tuple(sorted(Counter(os.path.dirname(str(row['file_path']))
                                       for row in final if not row['is_missing']).items())),
        errors=tuple(errors.values()),
    )
