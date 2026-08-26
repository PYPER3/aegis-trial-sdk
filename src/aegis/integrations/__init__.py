"""Optional framework adapters for the AEGIS Trial SDK."""

from .ddp import gather_events
from .pytorch import AegisStepObserver
from .transformers import AegisCallback

__all__ = ["AegisCallback", "AegisStepObserver", "gather_events"]
