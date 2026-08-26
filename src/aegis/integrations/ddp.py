"""Local event collection helpers for torch distributed jobs."""

from typing import Any


def gather_events(local_event: Any, dist_module: Any) -> list[dict[str, Any]] | None:
    """Gather one local event per rank to rank zero without external transport."""
    payload = local_event.to_dict() if hasattr(local_event, "to_dict") else dict(local_event)
    world_size = dist_module.get_world_size()
    rank = dist_module.get_rank()
    gathered = [None] * world_size if rank == 0 else None
    dist_module.gather_object(payload, gathered, dst=0)
    return gathered
