from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from eidory.core.ai_vision import AIVisionProviderError, image_to_data_url, normalize_ai_vision_analysis
from eidory.core.image_loader import MAX_AI_VISION_UPLOAD_BYTES


class AIVisionTest(unittest.TestCase):
    def test_normalize_ai_vision_aliases(self) -> None:
        analysis = normalize_ai_vision_analysis(
            {
                "scene_location": {"value": "outdoors", "confidence": 0.9, "note": ""},
                "environment_type": {"value": "man-made", "confidence": 0.8, "note": ""},
                "time_of_day": {"value": "dusk", "confidence": 0.7, "note": ""},
                "weather": {"value": "clear", "confidence": 0.6, "note": ""},
                "shot_scale": {"value": "full_shot", "confidence": 0.5, "note": ""},
                "view_angle": {"value": "normal", "confidence": 0.4, "note": ""},
                "lighting": [
                    {"value": "artificial_light", "confidence": 0.3, "note": ""},
                    {"value": "soft_light", "confidence": 0.2, "note": ""},
                ],
                "notes": "",
            }
        )

        self.assertEqual(analysis.scene_location, "outdoor")
        self.assertEqual(analysis.environment_type, "built")
        self.assertEqual(analysis.time_of_day, "dawn_dusk")
        self.assertEqual(analysis.weather, "sunny")
        self.assertEqual(analysis.shot_scale, "long")
        self.assertEqual(analysis.view_angle, "eye_level")
        self.assertEqual(analysis.lighting, ["unclear", "diffuse"])

    def test_image_to_data_url_uses_bounded_thumbnail_not_source_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "large.png"
            Image.new("RGB", (2200, 1600), color="red").save(image_path)

            with patch.object(Path, "read_bytes", side_effect=AssertionError("raw file read")):
                data_url = image_to_data_url(image_path)

            self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
            payload = data_url.split(",", 1)[1]
            self.assertLess(len(payload), MAX_AI_VISION_UPLOAD_BYTES * 2)

    def test_image_to_data_url_rejects_bad_images_without_raw_file_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "bad.jpg"
            image_path.write_bytes(b"not a real image")

            with patch.object(Path, "read_bytes", side_effect=AssertionError("raw file read")):
                with self.assertRaises(AIVisionProviderError):
                    image_to_data_url(image_path)


if __name__ == "__main__":
    unittest.main()
