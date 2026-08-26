"""Public interface for the detection-only AEGIS Trial SDK."""

from .events import AegisEvent
from .monitor import AegisMonitor
from .report import write_evaluation_report

__all__ = ["AegisEvent", "AegisMonitor", "write_evaluation_report"]
