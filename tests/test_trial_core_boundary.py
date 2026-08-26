import re
import unittest
from dataclasses import fields
from pathlib import Path

import sys


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_ROOT / "src"))


class TrialCoreBoundaryTests(unittest.TestCase):
    def test_distribution_contains_only_the_public_package(self):
        config = (PUBLIC_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('include = ["aegis*"]', config)
        self.assertIn('"" = "src"', config)
        self.assertNotIn("AEGIS/", config)

    def test_public_sdk_does_not_package_the_proprietary_core(self):
        self.assertFalse((PUBLIC_ROOT / "src" / "aegis_trial_core").exists())

    def test_public_package_has_no_training_actuation_modules(self):
        public_files = [
            path.relative_to(PUBLIC_ROOT / "src" / "aegis").as_posix().lower()
            for path in (PUBLIC_ROOT / "src" / "aegis").rglob("*")
            if path.is_file()
        ]
        forbidden_capabilities = ("optimizer", "checkpoint", "intervention", "rollback", "quarantine")
        self.assertFalse([name for name in public_files if any(term in name for term in forbidden_capabilities)])

    def test_monitor_has_no_control_switch(self):
        # Inspect source rather than importing it: the separately supplied Core
        # is not installed in this test environment.
        source = (PUBLIC_ROOT / "src" / "aegis" / "monitor.py").read_text(encoding="utf-8")
        signature = re.search(r"def __init__\((.*?)\):", source, re.DOTALL)
        self.assertIsNotNone(signature)
        self.assertNotIn("intervention", signature.group(1))

    def test_public_event_exposes_only_documented_generic_fields(self):
        from aegis import AegisEvent

        self.assertEqual(
            [field.name for field in fields(AegisEvent)],
            ["step", "anomaly_state", "detection_confidence", "recommended_next_step"],
        )


if __name__ == "__main__":
    unittest.main()
