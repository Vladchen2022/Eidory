from __future__ import annotations

import unittest

from eidory.core.video_metadata import parse_video_metadata


class VideoMetadataTest(unittest.TestCase):
    def test_parse_video_metadata_prefers_stream_duration(self) -> None:
        metadata = parse_video_metadata(
            {
                "streams": [
                    {
                        "width": 1920,
                        "height": 1080,
                        "duration": "12.345",
                    }
                ],
                "format": {"duration": "99.0"},
            }
        )

        self.assertEqual(metadata.width, 1920)
        self.assertEqual(metadata.height, 1080)
        self.assertEqual(metadata.duration_ms, 12_345)

    def test_parse_video_metadata_falls_back_to_format_duration(self) -> None:
        metadata = parse_video_metadata(
            {
                "streams": [{"width": "1280", "height": "720"}],
                "format": {"duration": "5.5"},
            }
        )

        self.assertEqual(metadata.width, 1280)
        self.assertEqual(metadata.height, 720)
        self.assertEqual(metadata.duration_ms, 5_500)


if __name__ == "__main__":
    unittest.main()
