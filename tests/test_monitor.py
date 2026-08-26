import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_ROOT / "src"))


class FakeDetector:
    def __init__(self, model=None, *, expected_microbatches=1):
        self.model = model
        self.expected_microbatches = expected_microbatches
        self.begin_steps = []
        self.attached = []
        self.evaluated = []

    def attach(self, model):
        self.attached.append(model)

    def begin_step(self, step):
        self.begin_steps.append(step)

    def evaluate(self, metrics):
        self.evaluated.append(metrics)
        return {
            "anomaly_state": "clear",
            "detection_confidence": 0.1,
            "recommended_next_step": "Continue normal monitoring.",
        }


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.previous_core = sys.modules.get("aegis_trial_core")
        fake_core = types.ModuleType("aegis_trial_core")
        fake_core.Detector = FakeDetector
        sys.modules["aegis_trial_core"] = fake_core

        sys.modules.pop("aegis", None)
        sys.modules.pop("aegis.monitor", None)
        sys.modules.pop("aegis.events", None)
        from aegis import AegisMonitor
        self.monitor_class = AegisMonitor

    def tearDown(self):
        sys.modules.pop("aegis", None)
        sys.modules.pop("aegis.monitor", None)
        sys.modules.pop("aegis.events", None)
        if self.previous_core is None:
            sys.modules.pop("aegis_trial_core", None)
        else:
            sys.modules["aegis_trial_core"] = self.previous_core

    def test_canonical_lifecycle_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "telemetry.jsonl"
            monitor = self.monitor_class(model=object(), log_path=log_path)
            monitor.begin_step(7)
            event = monitor.end_step(loss=1.25)

            detector = monitor.detector
            self.assertEqual(detector.begin_steps, [7])
            self.assertEqual(detector.evaluated, [{"loss": 1.25}])
            self.assertFalse(event.detected)
            lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIsInstance(json.loads(lines[0]), dict)

    def test_modular_lifecycle_still_begins_before_evaluation(self):
        monitor = self.monitor_class(model=object())
        monitor.begin_step(3)
        monitor.end_step(loss=2.0)
        self.assertEqual(monitor.detector.begin_steps, [3])

    def test_expected_microbatches_reaches_private_core(self):
        monitor = self.monitor_class(model=object(), expected_microbatches=4)
        self.assertEqual(monitor.expected_microbatches, 4)
        self.assertEqual(monitor.detector.expected_microbatches, 4)


if __name__ == "__main__":
    unittest.main()
