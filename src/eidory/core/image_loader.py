from __future__ import annotations

import warnings
from pathlib import Path

from PIL import Image, ImageFile

MAX_DECODED_IMAGE_PIXELS = 100_000_000
MAX_SOURCE_IMAGE_BYTES = 512 * 1024 * 1024
MAX_AI_VISION_UPLOAD_BYTES = 1_600_000

# Eidory indexes local art/reference files, so the limit is intentionally higher
# than Pillow's default warning threshold. It is still finite: bad or hostile
# images must not be allowed to consume unbounded memory.
Image.MAX_IMAGE_PIXELS = MAX_DECODED_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = True


def open_local_image(image_path: str | Path) -> Image.Image:
    path = Path(image_path)
    file_size = path.stat().st_size
    if file_size > MAX_SOURCE_IMAGE_BYTES:
        raise ValueError(f"image file is too large: {file_size:,} bytes")
    Image.MAX_IMAGE_PIXELS = MAX_DECODED_IMAGE_PIXELS
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            image = Image.open(path)
    except Image.DecompressionBombError as exc:
        raise ValueError(f"image dimensions are too large: {path.name}") from exc
    pixels = int(image.width) * int(image.height)
    if pixels > MAX_DECODED_IMAGE_PIXELS:
        image.close()
        raise ValueError(f"image dimensions are too large: {pixels:,} pixels")
    return image
