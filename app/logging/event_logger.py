"""Append-only JSONL event log: one line per pipeline event."""

import json
from collections.abc import Iterator
from pathlib import Path

from app.schemas.common import utc_now

EVENT_TYPES = frozenset({
    "query_received",
    "query_analyzed",
    "route_selected",
    "generation_started",
    "generation_completed",
    "generation_failed",
    "provider_error",
    "judge_completed",
    "judge_failed",
    "escalation_triggered",
    "pipeline_completed",
    "pipeline_failed",
})


class EventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, query_id: str, event_type: str, **payload: object) -> dict:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {event_type}")
        event = {"timestamp": utc_now(), "query_id": query_id, "event_type": event_type, "payload": payload}
        # ponytail: one open/append per event; fine for single-process V1, use a queue/writer task if write volume grows.
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
        return event

    def read(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
