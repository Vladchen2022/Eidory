from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from eidory.core.art_analysis import analyze_image_art, render_art_analysis_view


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
            for entry in first.palette:
                self.assertAlmostEqual(sum(entry.spatial_distribution), 1.0, places=6)

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
            self.assertEqual(result.detail_label, "低细节")
            self.assertAlmostEqual(result.detail_density, 0.0, places=6)
            self.assertAlmostEqual(result.visual_center_x, 0.5, places=6)
            self.assertAlmostEqual(result.visual_center_y, 0.5, places=6)
            self.assertAlmostEqual(result.horizontal_similarity, 1.0, places=6)

    def test_palette_spatial_distribution_tracks_color_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "split.png"
            image = Image.new("RGB", (90, 90), color=(230, 25, 20))
            image.paste((20, 55, 225), (45, 0, 90, 90))
            image.save(image_path)

            result = analyze_image_art(str(image_path), max_edge=180)
            red = max(result.palette, key=lambda entry: entry.rgb[0] - entry.rgb[2])
            left_column = sum(red.spatial_distribution[index] for index in (0, 3, 6))
            right_column = sum(red.spatial_distribution[index] for index in (2, 5, 8))

            self.assertGreater(left_column, right_column)
            self.assertAlmostEqual(sum(red.spatial_distribution), 1.0, places=6)

    def test_lightness_and_tone_distributions_include_spatial_locations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "bands.png"
            image = Image.new("RGB", (90, 90), color=(12, 12, 12))
            image.paste((120, 120, 120), (30, 0, 60, 90))
            image.paste((255, 40, 20), (60, 0, 90, 45))
            image.paste((20, 70, 255), (60, 45, 90, 90))
            image.save(image_path)

            result = analyze_image_art(str(image_path), max_edge=180)
            for distribution in (
                result.lightness_distribution,
                result.saturation_distribution,
                result.temperature_distribution,
            ):
                for entry in distribution:
                    expected = 1.0 if entry.proportion > 0.0 else 0.0
                    self.assertAlmostEqual(
                        sum(entry.spatial_distribution),
                        expected,
                        places=6,
                    )

            darkest = result.lightness_distribution[0]
            left_column = sum(darkest.spatial_distribution[index] for index in (0, 3, 6))
            right_column = sum(darkest.spatial_distribution[index] for index in (2, 5, 8))
            self.assertGreater(left_column, right_column)

    def test_rendered_analysis_views_preserve_size_and_use_palette(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "render.png"
            image = Image.new("RGB", (120, 80), color=(220, 30, 25))
            image.paste((20, 60, 220), (60, 0, 120, 80))
            image.save(image_path)
            result = analyze_image_art(str(image_path), max_edge=160)

            palette_view = render_art_analysis_view(
                str(image_path),
                result,
                mode="palette",
                max_width=240,
                max_height=160,
            )
            selected_index = max(
                range(len(result.palette)),
                key=lambda index: result.palette[index].rgb[0] - result.palette[index].rgb[2],
            )
            highlight_view = render_art_analysis_view(
                str(image_path),
                result,
                mode="highlight",
                selected_palette_index=selected_index,
                max_width=240,
                max_height=160,
            )
            detail_view = render_art_analysis_view(
                str(image_path),
                result,
                mode="detail",
                max_width=240,
                max_height=160,
            )

            self.assertEqual(palette_view.size, image.size)
            self.assertEqual(highlight_view.size, image.size)
            self.assertEqual(detail_view.size, image.size)
            colors = palette_view.convert("RGB").getcolors(
                maxcolors=palette_view.width * palette_view.height
            )
            self.assertIsNotNone(colors)
            self.assertLessEqual(len(colors or []), len(result.palette))
            self.assertEqual(highlight_view.convert("RGB").getpixel((10, 10)), (220, 30, 25))
            self.assertNotEqual(
                highlight_view.convert("RGB").getpixel((10, 10)),
                highlight_view.convert("RGB").getpixel((110, 10)),
            )

    def test_lightness_saturation_and_temperature_views_highlight_selected_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "classification.png"
            image = Image.new("RGB", (120, 80), color=(10, 10, 10))
            image.paste((245, 245, 245), (60, 0, 120, 40))
            image.paste((235, 30, 20), (0, 40, 60, 80))
            image.paste((20, 60, 230), (60, 40, 120, 80))
            image.save(image_path)
            result = analyze_image_art(str(image_path), max_edge=160)

            lightness_view = render_art_analysis_view(
                str(image_path),
                result,
                mode="lightness_highlight",
                selected_palette_index=5,
                max_width=240,
                max_height=160,
            ).convert("RGB")
            saturation_view = render_art_analysis_view(
                str(image_path),
                result,
                mode="saturation_highlight",
                selected_palette_index=2,
                max_width=240,
                max_height=160,
            ).convert("RGB")
            temperature_view = render_art_analysis_view(
                str(image_path),
                result,
                mode="temperature_highlight",
                selected_palette_index=0,
                max_width=240,
                max_height=160,
            ).convert("RGB")

            self.assertEqual(lightness_view.getpixel((90, 20)), (245, 245, 245))
            self.assertNotEqual(lightness_view.getpixel((20, 20)), (10, 10, 10))
            self.assertEqual(saturation_view.getpixel((20, 60)), (235, 30, 20))
            self.assertNotEqual(saturation_view.getpixel((90, 20)), (245, 245, 245))
            self.assertEqual(temperature_view.getpixel((20, 60)), (235, 30, 20))
            self.assertNotEqual(temperature_view.getpixel((90, 60)), (20, 60, 230))


if __name__ == "__main__":
    unittest.main()
