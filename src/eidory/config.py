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
        override = os.environ.get("EIDORY_DATA_DIR", "").strip()
        data_dir = (
            Path(override).expanduser()
            if override
            else Path.home() / "Library" / "Application Support" / "Eidory"
        )
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
