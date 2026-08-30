"""Public event types for the local AEGIS Trial SDK."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AegisEvent:
    """An immutable local detection result; next-step text cannot intervene."""

    step: int
    anomaly_state: str
    detection_confidence: float | None
    recommended_next_step: str

    @property
    def detected(self) -> bool:
        """Whether the local detector marked this step as anomalous."""
        return self.anomaly_state == "detected"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
