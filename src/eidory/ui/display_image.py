"""Convert display pixels to sRGB, including Qt-unsupported GRAY ICC profiles."""
from PIL import Image, ImageCms
from PySide6.QtGui import QColorSpace, QImage
from eidory.core.image_loader import open_local_image, srgb_pixels


def read_display_image(reader, path):
    image = reader.read()
    srgb = QColorSpace(QColorSpace.NamedColorSpace.SRgb)
    if not image.isNull() and image.colorSpace().isValid():
        return image.convertedToColorSpace(srgb)
    # Some embedded GRAY/CMYK profiles cannot be represented by QColorSpace.
    # Use the same native-mode ICC conversion as the AI and thumbnail paths.
    try:
        with open_local_image(path) as source:
            if not source.info.get('icc_profile') and not image.isNull():
                image.setColorSpace(srgb)
                return image
            size = reader.scaledSize()
            if size.isValid() and source.format == 'JPEG':
                source.draft('RGB', (size.width(), size.height()))
            pixels = srgb_pixels(source)
            if size.isValid():
                pixels.thumbnail((size.width(), size.height()), Image.Resampling.LANCZOS)
            pixels = pixels.convert('RGBA')
            result = QImage(pixels.tobytes(), pixels.width, pixels.height, pixels.width*4, QImage.Format.Format_RGBA8888).copy()
            result.setColorSpace(srgb)
            return result
    except (OSError, ValueError, ImageCms.PyCMSError):
        return image
