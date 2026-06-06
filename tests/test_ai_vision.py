from __future__ import annotations

import unittest

from eidory.core.ai_vision import normalize_ai_vision_analysis


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


if __name__ == "__main__":
    unittest.main()
