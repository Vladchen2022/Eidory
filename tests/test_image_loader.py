from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from eidory.core import image_loader


class ImageLoaderTest(unittest.TestCase):
    def test_open_local_image_rejects_oversized_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "huge.jpg"
            path.write_bytes(b"not read")

            with patch.object(
                Path,
                "stat",
                return_value=SimpleNamespace(st_size=image_loader.MAX_SOURCE_IMAGE_BYTES + 1),
            ):
                with self.assertRaises(ValueError):
                    image_loader.open_local_image(path)

    def test_open_local_image_rejects_oversized_pixel_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.jpg"
            Image.new("RGB", (8, 8), color="red").save(path)
            fake_image = SimpleNamespace(
                width=image_loader.MAX_DECODED_IMAGE_PIXELS + 1,
                height=1,
                close=lambda: None,
            )

            with patch.object(image_loader.Image, "open", return_value=fake_image):
                with self.assertRaises(ValueError):
                    image_loader.open_local_image(path)

    def test_open_local_image_suppresses_pillow_decompression_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.jpg"
            Image.new("RGB", (8, 8), color="red").save(path)
            fake_image = SimpleNamespace(
                width=image_loader.MAX_DECODED_IMAGE_PIXELS + 1,
                height=1,
                close=lambda: None,
            )

            def open_with_warning(_path: Path) -> SimpleNamespace:
                warnings.warn("too many pixels", image_loader.Image.DecompressionBombWarning)
                return fake_image

            with patch.object(image_loader.Image, "open", side_effect=open_with_warning):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with self.assertRaises(ValueError):
                        image_loader.open_local_image(path)

            leaked = [
                warning
                for warning in caught
                if issubclass(warning.category, image_loader.Image.DecompressionBombWarning)
            ]
            self.assertEqual([], leaked)


if __name__ == "__main__":
    unittest.main()
