from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFile

# Eidory indexes trusted local files. Some art/reference archives contain huge
# dezoomified JPEGs that exceed Pillow's conservative web-upload safety limit.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def open_local_image(image_path: str | Path) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    return Image.open(image_path)
