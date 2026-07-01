from __future__ import annotations

import unittest

import numpy as np
from PIL import Image, ImageDraw

from eidory.core.linetop_processor import LineTopSettings, render_linetop_image


class LineTopProcessorTest(unittest.TestCase):
    def test_line_mode_returns_visible_line_art_without_mutating_source(self) -> None:
        image = Image.new("RGB", (160, 120), color="white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 18, 140, 95), outline="black", width=4)
        before = image.tobytes()

        rendered = render_linetop_image(image, LineTopSettings(mode="line"))
        pixels = np.array(rendered.convert("RGBA"))

        self.assertEqual(image.tobytes(), before)
        self.assertEqual(rendered.size, image.size)
        self.assertLess(pixels[:, :, :3].min(), 245)
        self.assertTrue(np.all(pixels[:, :, 3] == 255))

    def test_color_limit_mode_reduces_palette_and_preserves_alpha(self) -> None:
        image = Image.new("RGBA", (120, 80), color=(255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 59, 79), fill=(240, 20, 20, 255))
        draw.rectangle((60, 0, 119, 79), fill=(20, 80, 240, 128))

        rendered = render_linetop_image(
            image,
            LineTopSettings(
                mode="color_limit",
                color_limit_steps=2,
                smart_enhance=False,
                color_limit_shape_simplification=1,
            ),
        )
        pixels = np.array(rendered.convert("RGBA"))
        colors = np.unique(pixels[:, :, :3].reshape(-1, 3), axis=0)

        self.assertLessEqual(len(colors), 2)
        self.assertEqual(int(pixels[:, :60, 3].max()), 255)
        self.assertEqual(int(pixels[:, 60:, 3].max()), 128)


if __name__ == "__main__":
    unittest.main()
