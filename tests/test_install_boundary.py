import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PUBLIC_ROOT = Path(__file__).resolve().parents[1]


class InstallBoundaryTests(unittest.TestCase):
    def test_import_succeeds_without_trial_core(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PUBLIC_ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-c", "import aegis; print(aegis.AegisMonitor.__name__)"],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("AegisMonitor", result.stdout)

    def test_live_monitor_has_clean_missing_core_error(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PUBLIC_ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-c", "from aegis import AegisMonitor; AegisMonitor(object())"],
            capture_output=True, text=True, env=env,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("compatible aegis-trial-core wheel", result.stderr)

    def test_preview_command_writes_explicitly_simulated_report(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PUBLIC_ROOT / "src")
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "result.md"
            result = subprocess.run(
                [sys.executable, "-m", "aegis.demo", "--preview", "--output", str(report)],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("SIMULATED PREVIEW", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
