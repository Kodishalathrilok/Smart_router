"""Consolidated per-query records, one row per completed pipeline run (SQLite; see app/logging/store.py)."""

import logging
import sqlite3

from pydantic import ValidationError

from app.logging.store import Store
from app.schemas.common import QueryRecord

log = logging.getLogger("smartrouter.queries")


class QueryLogger:
    def __init__(self, store: Store) -> None:
        self._store = store
        self.write_failures = 0

    def write(self, record: QueryRecord) -> None:
        """Best effort, like EventLogger.log: failure is logged and counted, not raised."""
        try:
            self._store.put_record(record.query_id, record.timestamp, record.final_status, record.model_dump_json())
        except sqlite3.Error as exc:
            self.write_failures += 1
            log.error("query record write failed (%s): %s", type(exc).__name__, record.query_id)

    def recent(self, limit: int) -> list[QueryRecord]:
        """Newest `limit` records, oldest first. Rows that no longer match the schema are skipped, not fatal."""
        return [r for r in (_parse(raw) for raw in self._store.recent_records(limit)) if r is not None]

    def get(self, query_id: str) -> QueryRecord | None:
        raw = self._store.get_record(query_id)
        return _parse(raw) if raw is not None else None

    def count(self) -> int:
        return self._store.count_records()


def _parse(raw: str) -> QueryRecord | None:
    try:
        return QueryRecord.model_validate_json(raw)
    except ValidationError:
        log.warning("skipping a stored query record that does not match the current schema")
        return None
