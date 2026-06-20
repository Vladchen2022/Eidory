from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from eidory.core.ai_vision import (
    AI_VISION_MAX_TOKENS,
    AIVisionProvider,
    AIVisionProviderError,
    image_to_data_url,
    normalize_ai_vision_analysis,
)
from eidory.core.image_loader import MAX_AI_VISION_UPLOAD_BYTES


class AIVisionTest(unittest.TestCase):
    def test_analyze_image_repairs_malformed_json_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.jpg"
            Image.new("RGB", (64, 64), color="blue").save(image_path)
            captured_payloads: list[dict[str, object]] = []

            class FakeResponse:
                status_code = 200

                def __init__(self, content: str):
                    self.content = content

                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, object]:
                    return {"choices": [{"message": {"content": self.content}}]}

            fixed_json = """
            {
              "scene_location": {"value": "indoor", "confidence": 0.8, "note": ""},
              "environment_type": {"value": "built", "confidence": 0.7, "note": ""},
              "time_of_day": {"value": "artificial_light", "confidence": 0.6, "note": ""},
              "weather": {"value": "not_visible", "confidence": 0.9, "note": ""},
              "shot_scale": {"value": "medium", "confidence": 0.6, "note": ""},
              "view_angle": {"value": "eye_level", "confidence": 0.6, "note": ""},
              "lighting": [{"value": "diffuse", "confidence": 0.5, "note": ""}],
              "notes": "修复后的结构化结果"
            }
            """
            responses = [
                FakeResponse('{"scene_location":{"value":"indoor","confidence":0.8,"note":""}'),
                FakeResponse('thinking {"value": "not_the_payload"}\n' + fixed_json),
            ]

            def fake_post(*args: object, **kwargs: object) -> FakeResponse:
                captured_payloads.append(dict(kwargs["json"]))  # type: ignore[index]
                return responses.pop(0)

            provider = AIVisionProvider(
                base_url="http://localhost:1234/v1",
                model_name="vision-model",
            )
            with patch("eidory.core.ai_vision.requests.post", side_effect=fake_post):
                analysis = provider.analyze_image(image_path)

            self.assertEqual(analysis.scene_location, "indoor")
            self.assertEqual(analysis.environment_type, "built")
            self.assertEqual(analysis.lighting, ["diffuse"])
            self.assertEqual(len(captured_payloads), 2)
            self.assertIsInstance(captured_payloads[1]["messages"], list)

    def test_chat_completion_allows_reasoning_model_json_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.jpg"
            Image.new("RGB", (64, 64), color="blue").save(image_path)
            captured_payloads: list[dict[str, object]] = []

            class FakeResponse:
                status_code = 200

                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, object]:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "{}",
                                    "reasoning_content": "reasoning tokens",
                                }
                            }
                        ]
                    }

            def fake_post(*args: object, **kwargs: object) -> FakeResponse:
                captured_payloads.append(dict(kwargs["json"]))  # type: ignore[index]
                return FakeResponse()

            provider = AIVisionProvider(
                base_url="http://localhost:1234/v1",
                model_name="vision-model",
            )
            with patch("eidory.core.ai_vision.requests.post", side_effect=fake_post):
                provider._chat_completion(model_name="vision-model", image_path=image_path)

            self.assertEqual(captured_payloads[0]["max_tokens"], AI_VISION_MAX_TOKENS)
            self.assertGreaterEqual(AI_VISION_MAX_TOKENS, 2048)

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
