import sys
import unittest
from pathlib import Path


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_ROOT / "src"))

from aegis.integrations.ddp import gather_events
from aegis.integrations.transformers import AegisCallback


class Event:
    def to_dict(self):
        return {"step": 4, "anomaly_state": "clear"}


class FakeDist:
    def __init__(self, rank):
        self.rank = rank
        self.received = None

    def get_world_size(self):
        return 2

    def get_rank(self):
        return self.rank

    def gather_object(self, payload, gathered, dst):
        self.received = (payload, gathered, dst)
        if gathered is not None:
            gathered[0] = payload
            gathered[1] = {"step": 4, "anomaly_state": "detected"}


class IntegrationTests(unittest.TestCase):
    def test_rank_zero_receives_one_event_per_worker(self):
        dist = FakeDist(rank=0)
        events = gather_events(Event(), dist)
        self.assertEqual([event["anomaly_state"] for event in events], ["clear", "detected"])

    def test_nonzero_rank_does_not_allocate_global_event_list(self):
        dist = FakeDist(rank=1)
        self.assertIsNone(gather_events(Event(), dist))
        self.assertIsNone(dist.received[1])

    def test_trainer_callback_uses_one_complete_step_lifecycle(self):
        calls = []

        class Monitor:
            def begin_step(self, step):
                calls.append(("begin", step))

            def end_step(self, **kwargs):
                calls.append(("end", kwargs))

        class State:
            global_step = 12

        callback = AegisCallback(Monitor())
        control = object()
        self.assertIs(callback.on_step_begin(None, State(), control), control)
        self.assertIs(callback.on_step_end(None, State(), control), control)
        self.assertEqual(calls, [("begin", 12), ("end", {"metrics": {"trainer_step": 12}})])
