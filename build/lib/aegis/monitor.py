"""Canonical detection-only training-loop lifecycle."""

import json
import sys
from pathlib import Path
from typing import Any

from .events import AegisEvent


MISSING_CORE_MESSAGE = (
    "The live AEGIS Trial Monitor requires the separately supplied compatible "
    "aegis-trial-core wheel. Install it in this environment, then retry. "
    "The public SDK itself makes no network connection."
)


class AegisMonitor:
    """Local, detection-only monitor for a PyTorch training loop.

    The canonical lifecycle is ``begin_step(step)`` before the monitored forward
    pass followed by ``end_step(loss=loss)`` after that step completes.
    """

    def __init__(self, model: Any, *, log_path: str | Path | None = None):
        try:
            from aegis_trial_core import Detector
        except ImportError as exc:
            raise ImportError(MISSING_CORE_MESSAGE) from exc

        self.log_path = Path(log_path) if log_path else None
        self.current_step: int | None = None
        self.step_history: list[dict[str, Any]] = []
        self.detector = Detector(model=model)
        if hasattr(self.detector, "attach"):
            self.detector.attach(model)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")

    def begin_step(self, step: int) -> None:
        """Start local observation for ``step`` before the model forward pass."""
        if self.current_step is not None:
            raise RuntimeError("begin_step() called before the previous step was finalized.")
        self.current_step = step
        self.detector.begin_step(step)

    def end_step(self, *, loss: Any = None, metrics: dict[str, Any] | None = None) -> AegisEvent:
        """Finalize local observation and return the step's immutable event."""
        if self.current_step is None:
            raise RuntimeError("end_step() called without begin_step().")
        payload = dict(metrics or {})
        if loss is not None:
            payload["loss"] = float(loss.item()) if hasattr(loss, "item") else float(loss)
        decision = self.detector.evaluate(payload)
        event = AegisEvent(
            step=self.current_step,
            anomaly_state=decision.get("anomaly_state", "detected" if decision.get("signal_detected") else "clear"),
            detection_confidence=decision.get("detection_confidence"),
            recommended_next_step=decision.get("recommended_next_step", "Continue normal monitoring."),
        )
        telemetry = {"metrics": payload, **event.to_dict()}
        self.step_history.append(telemetry)
        if len(self.step_history) > 1000:
            self.step_history.pop(0)
        if self.log_path:
            try:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(telemetry) + "\n")
            except OSError as exc:
                print(f"[AEGIS WARNING] Could not write local event log: {exc}", file=sys.stderr)
        self.current_step = None
        return event
