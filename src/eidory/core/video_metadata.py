from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoMetadata:
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None


def read_video_metadata(video_path: str) -> VideoMetadata:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return VideoMetadata()
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration:format=duration",
        "-of",
        "json",
        video_path,
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return VideoMetadata()
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return VideoMetadata()
    return parse_video_metadata(payload)


def parse_video_metadata(payload: dict[str, object]) -> VideoMetadata:
    streams = payload.get("streams")
    stream = streams[0] if isinstance(streams, list) and streams else {}
    if not isinstance(stream, dict):
        stream = {}
    format_data = payload.get("format")
    if not isinstance(format_data, dict):
        format_data = {}

    width = _positive_int(stream.get("width"))
    height = _positive_int(stream.get("height"))
    duration_seconds = _positive_float(stream.get("duration"))
    if duration_seconds is None:
        duration_seconds = _positive_float(format_data.get("duration"))
    duration_ms = int(duration_seconds * 1000) if duration_seconds is not None else None
    return VideoMetadata(width=width, height=height, duration_ms=duration_ms)


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
