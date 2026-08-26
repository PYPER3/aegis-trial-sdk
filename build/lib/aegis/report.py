"""Human-readable local evaluation report rendering."""

from pathlib import Path
from typing import Any


def write_evaluation_report(destination: str | Path, summary: dict[str, Any]) -> Path:
    """Write a screenshot-friendly Markdown result without exposing implementation details."""
    path = Path(destination)
    lines = [
        "# AEGIS Evaluation Result",
        "",
        f"- Model: {summary.get('model_id', 'not recorded')}",
        f"- Training steps: {summary.get('total_steps', 'not recorded')}",
        f"- Anomaly exercise: {summary.get('anomaly_exercise', 'not applicable')}",
        f"- Detection recorded: {summary.get('detection_step', 'not recorded')}",
        f"- Event count: {summary.get('event_count', 'not recorded')}",
        "- Detection only — no training changes",
        f"- Execution: {'SIMULATED PREVIEW' if summary.get('simulated') else 'LIVE LOCAL DETECTION'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
