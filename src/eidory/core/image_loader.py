from __future__ import annotations

import warnings
import io
import logging
from pathlib import Path

from PIL import Image, ImageFile, ImageOps, ImageCms

MAX_DECODED_IMAGE_PIXELS = 100_000_000
MAX_SOURCE_IMAGE_BYTES = 512 * 1024 * 1024
MAX_AI_VISION_UPLOAD_BYTES = 1_600_000

# Eidory indexes local art/reference files, so the limit is intentionally higher
# than Pillow's default warning threshold. It is still finite: bad or hostile
# images must not be allowed to consume unbounded memory.
Image.MAX_IMAGE_PIXELS = MAX_DECODED_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = True


def oriented_pixels(image: Image.Image) -> Image.Image:
    """Orient decoded pixels without serializing unrelated source metadata."""
    pixels = image.copy()
    pixels.info = {k: v for k, v in pixels.info.items() if k == 'transparency'}
    orientation = Image.Exif()
    orientation[274] = image.getexif().get(274, 1)
    pixels.info['exif'] = orientation.tobytes()
    ImageOps.exif_transpose(pixels, in_place=True)
    pixels.info.pop('exif', None)
    return pixels


def srgb_pixels(image: Image.Image, *, strict=True) -> Image.Image:
    """Oriented sRGB pixels shared by previews, thumbnails and AI input."""
    pixels = oriented_pixels(image)
    alpha = pixels.convert('RGBA').getchannel('A') if 'A' in pixels.getbands() or 'transparency' in pixels.info else None
    srgb = ImageCms.createProfile('sRGB')
    if image.info.get('icc_profile'):
        try:
            profile = ImageCms.ImageCmsProfile(io.BytesIO(image.info['icc_profile']))
            space = profile.profile.xcolor_space.strip()
            mode = {'RGB': 'RGB', 'GRAY': 'L', 'CMYK': 'CMYK'}.get(space)
            if mode is None:
                raise ValueError(f'暂不支持图片的 {space} 色彩配置。')
            pixels = ImageCms.profileToProfile(pixels.convert(mode), profile, srgb, outputMode='RGB')
        except (OSError, ValueError, ImageCms.PyCMSError) as exc:
            if strict:
                raise ValueError('无法转换图片的嵌入色彩配置，请检查源图 ICC 配置后重试。') from exc
            logging.getLogger('eidory.runtime').warning('Invalid ICC profile; displaying unconverted pixels: %s', exc)
            pixels = pixels.convert('RGB')
    else:
        pixels = pixels.convert('RGB')
    pixels.info.clear()
    pixels.info['icc_profile'] = ImageCms.ImageCmsProfile(srgb).tobytes()
    if alpha is not None:
        pixels.putalpha(alpha)
    return pixels


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
