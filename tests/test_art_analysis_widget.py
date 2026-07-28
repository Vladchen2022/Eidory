from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication

from eidory.core.art_analysis import analyze_image_art
from eidory.ui.art_analysis_widget import ArtAnalysisWidget


class ArtAnalysisWidgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_analysis_is_split_into_four_accessible_tabs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.png"
            image = Image.new("RGB", (90, 60), color=(220, 30, 25))
            image.paste((20, 60, 220), (45, 0, 90, 60))
            image.save(image_path)
            result = analyze_image_art(str(image_path))

            widget = ArtAnalysisWidget()
            widget.set_result(result)

            self.assertEqual(widget.tabs.count(), 4)
            self.assertEqual(
                [widget.tabs.tabText(index) for index in range(widget.tabs.count())],
                ["色彩", "明度", "色调", "构图"],
            )
            self.assertFalse(widget.content.isHidden())
            self.assertEqual(len(widget.palette_buttons), len(result.palette))
            self.assertTrue(widget.detail_view_button.isEnabled())
            widget.close()

    def test_palette_and_view_buttons_emit_analysis_view_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.png"
            image = Image.new("RGB", (90, 60), color=(220, 30, 25))
            image.paste((20, 60, 220), (45, 0, 90, 60))
            image.save(image_path)
            result = analyze_image_art(str(image_path))

            widget = ArtAnalysisWidget()
            widget.set_result(result)
            requests: list[tuple[str, int]] = []
            widget.viewRequested.connect(
                lambda mode, index: requests.append((mode, index))
            )

            widget.palette_buttons[-1].click()
            widget.palette_view_button.click()
            widget.lightness_row_widgets[0].selected.emit(0)
            widget.saturation_row_widgets[1].selected.emit(1)
            widget.temperature_row_widgets[2].selected.emit(2)
            widget.detail_view_button.click()
            widget.original_view_button.click()

            self.assertEqual(requests[0], ("highlight", len(result.palette) - 1))
            self.assertIn(("palette", -1), requests)
            self.assertIn(("lightness_highlight", 0), requests)
            self.assertIn(("saturation_highlight", 1), requests)
            self.assertIn(("temperature_highlight", 2), requests)
            self.assertIn(("detail", -1), requests)
            self.assertIn(("original", -1), requests)
            widget.close()


if __name__ == "__main__":
    unittest.main()
