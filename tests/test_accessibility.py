from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

from eidory.ui.accessibility import disable_qt_accessibility, hide_macos_accessibility_tree


class AccessibilityTest(unittest.TestCase):
    def test_accessibility_preserves_normal_controls(self) -> None:
        with patch.dict(os.environ, {"QT_ACCESSIBILITY": "1", "QT_QPA_PLATFORM": "cocoa"}), patch("eidory.ui.accessibility._install_item_view_factory"), patch("PySide6.QtGui.QAccessible.setActive") as active:
            disable_qt_accessibility()

            self.assertEqual(os.environ["QT_ACCESSIBILITY"], "1")
            self.assertIn("qt.accessibility.*=false", os.environ["QT_LOGGING_RULES"])
            active.assert_called_once_with(True)

    def test_headless_process_does_not_create_accessibility_cache(self) -> None:
        with patch.dict(os.environ, {"QT_QPA_PLATFORM": "offscreen"}), patch("PySide6.QtGui.QAccessible.setActive") as active:
            disable_qt_accessibility()
            self.assertEqual(os.environ["QT_ACCESSIBILITY"], "0")
            active.assert_called_once_with(False)

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
