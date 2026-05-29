from __future__ import annotations

from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
SUPPORTED_MEDIA_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS


def extension_for(path: str) -> str:
    return Path(path).suffix.lower()


def is_supported_image(path: str) -> bool:
    return extension_for(path) in SUPPORTED_IMAGE_EXTENSIONS


def is_supported_video(path: str) -> bool:
    return extension_for(path) in SUPPORTED_VIDEO_EXTENSIONS


def is_supported_media(path: str) -> bool:
    return extension_for(path) in SUPPORTED_MEDIA_EXTENSIONS
