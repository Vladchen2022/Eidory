from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from eidory.core.art_analysis import analyze_image_art


class ArtAnalysisTest(unittest.TestCase):
    def test_two_color_palette_reports_stable_pixel_proportions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "two-colors.png"
            image = Image.new("RGB", (100, 40), color=(220, 30, 25))
            image.paste((20, 60, 220), (75, 0, 100, 40))
            image.save(image_path)

            first = analyze_image_art(str(image_path), max_edge=200)
            second = analyze_image_art(str(image_path), max_edge=200)

            self.assertEqual(first, second)
            self.assertEqual(len(first.palette), 2)
            self.assertAlmostEqual(first.palette[0].proportion, 0.75, places=3)
            self.assertAlmostEqual(first.palette[1].proportion, 0.25, places=3)
            self.assertAlmostEqual(sum(entry.proportion for entry in first.palette), 1.0, places=6)

    def test_transparent_pixels_do_not_pollute_palette(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "transparent.png"
            image = Image.new("RGBA", (40, 20), color=(0, 255, 0, 0))
            image.paste((255, 0, 0, 255), (0, 0, 20, 20))
            image.save(image_path)

            result = analyze_image_art(str(image_path), max_edge=100)

            self.assertEqual(len(result.palette), 1)
            red, green, blue = result.palette[0].rgb
            self.assertGreater(red, 245)
            self.assertLess(green, 10)
            self.assertLess(blue, 10)
            self.assertAlmostEqual(result.palette[0].proportion, 1.0)
            self.assertEqual(result.sampled_pixels, 400)

    def test_grayscale_image_has_complete_lightness_and_neutral_distributions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "grayscale.png"
            image = Image.new("L", (60, 30), color=128)
            image.save(image_path)

            result = analyze_image_art(str(image_path))

            self.assertAlmostEqual(
                sum(entry.proportion for entry in result.lightness_distribution),
                1.0,
                places=6,
            )
            neutral = next(
                entry for entry in result.temperature_distribution if entry.label == "中性色"
            )
            self.assertAlmostEqual(neutral.proportion, 1.0, places=6)
            self.assertEqual(result.contrast_label, "低对比")


if __name__ == "__main__":
    unittest.main()
