from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageOps

from eidory.core.image_loader import open_local_image
from eidory.core.media_tools import find_media_tool


class Thumbnailer:
    def __init__(self, thumbnail_dir: Path, max_edge: int = 512):
        self.thumbnail_dir = thumbnail_dir
        self.max_edge = max_edge
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def thumbnail_path_for(self, image_id: int) -> Path:
        return self.thumbnail_dir / f"thumb_{image_id:09d}.webp"

    def generate(self, image_id: int, image_path: str) -> Path:
        output_path = self.thumbnail_path_for(image_id)
        with open_local_image(image_path) as image:
            if image.format == "JPEG":
                image.draft("RGB", (self.max_edge, self.max_edge))
            image = ImageOps.exif_transpose(image)
            image.thumbnail((self.max_edge, self.max_edge), Image.Resampling.LANCZOS)
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGB")
            image.save(output_path, "WEBP", quality=82, method=4)
        return output_path

    def generate_video(
        self,
        image_id: int,
        video_path: str,
        *,
        duration_ms: int | None = None,
    ) -> Path:
        ffmpeg = find_media_tool("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found; cannot generate video thumbnail")

        output_path = self.thumbnail_path_for(image_id)
        timestamp = self._video_thumbnail_timestamp(video_path, duration_ms=duration_ms)
        command = [
            ffmpeg,
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-vf",
            f"scale={self.max_edge}:{self.max_edge}:force_original_aspect_ratio=decrease",
            "-quality",
            "82",
            "-y",
            str(output_path),
        ]
        subprocess.run(command, check=True, timeout=30)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg did not create a video thumbnail")
        return output_path

    @staticmethod
    def _video_thumbnail_timestamp(video_path: str, *, duration_ms: int | None = None) -> float:
        duration = (duration_ms / 1000) if duration_ms and duration_ms > 0 else None
        if duration is None:
            duration = Thumbnailer._video_duration(video_path)
        if duration is None or duration <= 0:
            return 5.0
        return max(0.1, min(duration * 0.65, max(0.1, duration - 0.1)))

    @staticmethod
    def _video_duration(video_path: str) -> float | None:
        ffprobe = find_media_tool("ffprobe")
        if ffprobe is None:
            return None
        command = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
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
            return None
        try:
            return float(result.stdout.strip())
        except ValueError:
            return None
