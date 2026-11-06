from __future__ import annotations

import subprocess
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageCms

from eidory.core.image_loader import open_local_image, srgb_pixels
from eidory.core.media_tools import find_media_tool


def upgrade_display_thumbnail(thumbnail_path: str, source_path: str) -> None:
    """Lazily upgrade old untagged cache files, never modify source images."""
    thumbnail = Path(thumbnail_path)
    if not thumbnail.name.startswith('thumb_') or thumbnail.suffix != '.webp' or thumbnail.is_symlink():
        return
    try:
        with Image.open(thumbnail) as cached:
            if cached.info.get('icc_profile'):
                return
            edge = max(cached.size)
        with open_local_image(source_path) as source:
            if not source.info.get('icc_profile'):
                return
            if source.format == 'JPEG':
                source.draft('RGB', (edge, edge))
            corrected = srgb_pixels(source, strict=False)
            corrected.thumbnail((edge, edge), Image.Resampling.LANCZOS)
            with Thumbnailer._atomic_output(thumbnail) as temporary:
                corrected.save(temporary, 'WEBP', quality=82, method=4,
                               icc_profile=corrected.info['icc_profile'])
    except (OSError, ValueError, ImageCms.PyCMSError):
        # Existing caches remain usable if an external drive is disconnected.
        return


class Thumbnailer:
    def __init__(self, thumbnail_dir: Path, max_edge: int = 512):
        self.thumbnail_dir = thumbnail_dir
        self.max_edge = max_edge
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def thumbnail_path_for(self, image_id: int) -> Path:
        return self.thumbnail_dir / f"thumb_{image_id:09d}.webp"

    @staticmethod
    @contextmanager
    def _atomic_output(output_path: Path):
        fd, name = tempfile.mkstemp(prefix='.thumb-', suffix='.webp', dir=output_path.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            yield temporary
            if not temporary.stat().st_size:
                raise RuntimeError('Thumbnail encoder produced an empty file')
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)

    def generate(self, image_id: int, image_path: str) -> Path:
        output_path = self.thumbnail_path_for(image_id)
        with open_local_image(image_path) as image:
            if image.format == "JPEG":
                image.draft("RGB", (self.max_edge, self.max_edge))
            image = srgb_pixels(image, strict=False)
            image.thumbnail((self.max_edge, self.max_edge), Image.Resampling.LANCZOS)
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGB")
            with self._atomic_output(output_path) as temporary:
                image.save(temporary, "WEBP", quality=82, method=4, icc_profile=image.info.get('icc_profile'))
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
        with self._atomic_output(output_path) as temporary:
            command[-1] = str(temporary)
            subprocess.run(command, check=True, timeout=30)
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
