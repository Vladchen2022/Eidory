from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def find_media_tool(executable_name: str) -> str | None:
    found = shutil.which(executable_name)
    if found:
        return found

    candidates = [
        Path("/opt/homebrew/bin") / executable_name,
        Path("/usr/local/bin") / executable_name,
        Path("/usr/bin") / executable_name,
        Path.home() / ".local" / "bin" / executable_name,
        Path.home() / "bin" / executable_name,
    ]
    candidates.extend(_bundle_tool_candidates(executable_name))

    for candidate in candidates:
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        except OSError:
            continue
    return None


def _bundle_tool_candidates(executable_name: str) -> list[Path]:
    executable = Path(sys.executable).resolve()
    candidates: list[Path] = []
    for parent in [executable, *executable.parents]:
        if parent.name == "MacOS" and parent.parent.name == "Contents":
            contents_dir = parent.parent
            candidates.extend(
                [
                    contents_dir / "Resources" / "bin" / executable_name,
                    contents_dir / "MacOS" / executable_name,
                ]
            )
            break
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(str(sys._MEIPASS))  # type: ignore[attr-defined]
        candidates.extend(
            [
                base / "bin" / executable_name,
                base / executable_name,
            ]
        )
    return candidates
