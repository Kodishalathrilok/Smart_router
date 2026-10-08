"""Structured pipeline events, one row per event (SQLite; see app/logging/store.py)."""

import logging
import sqlite3

from app.logging.store import Store
from app.schemas.common import utc_now

log = logging.getLogger("smartrouter.events")

EVENT_TYPES = frozenset({
    "query_received",
    "query_analyzed",
    "route_selected",
    "generation_started",
    "generation_completed",
    "generation_failed",
    "provider_error",
    "checks_completed",
    "checks_error",
    "judge_completed",
    "judge_failed",
    "second_opinion_completed",
    "second_opinion_failed",
    "escalation_triggered",
    "pipeline_completed",
    "pipeline_failed",
})


class EventLogger:
    def __init__(self, store: Store) -> None:
        self._store = store
        self.write_failures = 0

    def log(self, query_id: str, event_type: str, **payload: object) -> dict:
        """Best effort: a storage failure is reported on the server log and counted (see /health),
        but never turns an answer the user already paid for into an HTTP 500."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {event_type}")
        event = {"timestamp": utc_now(), "query_id": query_id, "event_type": event_type, "payload": payload}
        try:
            self._store.add_event(event["timestamp"], query_id, event_type, payload)
        except sqlite3.Error as exc:
            self.write_failures += 1
            log.error("event write failed (%s): %s %s", type(exc).__name__, query_id, event_type)
        return event

    def read(self, *, query_id: str | None = None, event_type: str | None = None) -> list[dict]:
        return self._store.events(query_id=query_id, event_type=event_type)

    def provider_error_counts(self) -> dict[str, int]:
        return self._store.provider_error_counts()
