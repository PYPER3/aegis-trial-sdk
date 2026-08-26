"""Optional Hugging Face Trainer callback."""

from typing import Any


class AegisCallback:
    """Trainer callback with one local event per completed Trainer step."""

    def __init__(self, monitor: Any):
        self.monitor = monitor
    def on_step_begin(self, args, state, control, **kwargs):
        self.monitor.begin_step(state.global_step)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self.monitor.end_step(metrics={"trainer_step": state.global_step})
        return control
