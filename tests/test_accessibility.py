from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

from eidory.ui.accessibility import disable_qt_accessibility, hide_macos_accessibility_tree


class AccessibilityTest(unittest.TestCase):
    def test_disable_qt_accessibility_overrides_external_environment(self) -> None:
        with patch.dict(os.environ, {"QT_ACCESSIBILITY": "1"}):
            disable_qt_accessibility()

            self.assertEqual(os.environ["QT_ACCESSIBILITY"], "0")

    def test_hide_macos_accessibility_tree_ignores_non_macos(self) -> None:
        class Widget:
            def winId(self) -> int:
                raise AssertionError("winId should not be requested outside macOS")

        with patch.object(sys, "platform", "linux"):
            self.assertFalse(hide_macos_accessibility_tree(Widget()))

    def test_hide_macos_accessibility_tree_handles_missing_objc(self) -> None:
        class Widget:
            def winId(self) -> int:
                raise AssertionError("winId should not be requested without objc")

        with (
            patch.object(sys, "platform", "darwin"),
            patch("ctypes.util.find_library", return_value=None),
        ):
            self.assertFalse(hide_macos_accessibility_tree(Widget()))
