from __future__ import annotations

import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
