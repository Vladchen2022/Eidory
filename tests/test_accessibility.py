from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from eidory.ui.accessibility import disable_qt_accessibility


class AccessibilityTest(unittest.TestCase):
    def test_disable_qt_accessibility_overrides_external_environment(self) -> None:
        with patch.dict(os.environ, {"QT_ACCESSIBILITY": "1"}):
            disable_qt_accessibility()

            self.assertEqual(os.environ["QT_ACCESSIBILITY"], "0")

