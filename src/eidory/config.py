from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    data_dir: Path
    thumbnail_dir: Path
    database_path: Path
    log_dir: Path

    @classmethod
    def default(cls) -> "AppPaths":
        from eidory.core.workspace_selection import active_directory
        data_dir = active_directory()
        return cls(
            data_dir=data_dir,
            thumbnail_dir=data_dir / "thumbnails",
            database_path=data_dir / "eidory.sqlite3",
            log_dir=data_dir / "logs",
        )

    def ensure(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
