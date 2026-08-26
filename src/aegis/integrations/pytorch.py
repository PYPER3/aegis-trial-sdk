"""Small helpers for the canonical PyTorch lifecycle."""

from typing import Any

from ..monitor import AegisMonitor


class AegisStepObserver:
    """Call around an ordinary PyTorch training step without changing it."""

    def __init__(self, monitor: AegisMonitor):
        self.monitor = monitor

    def begin(self, step: int) -> None:
        self.monitor.begin_step(step)

    def end(self, loss: Any, *, metrics: dict[str, Any] | None = None):
        return self.monitor.end_step(loss=loss, metrics=metrics)
