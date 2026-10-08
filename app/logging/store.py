"""SQLite persistence for pipeline events and consolidated query records (stdlib only).

Why SQLite instead of the earlier JSONL files:
  - a write is a transaction, so a crash or full disk cannot leave a half-written line that makes every
    later read fail (one truncated JSONL line used to turn /api/stats, /routes and /failures into 500s);
  - lookups by query_id and "most recent N" reads use indexes instead of scanning the whole file;
  - aggregates (provider error counts) run in SQL.
It is still a single local file with no server, which matches the single-process deployment.

One connection is shared behind a lock (check_same_thread=False) because writes come from the event
loop while tests and the benchmark may read from other threads. Writes are small; WAL mode keeps
readers from blocking the writer.
"""

import json
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    query_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_query ON events(query_id);
CREATE INDEX IF NOT EXISTS events_type ON events(event_type);
CREATE TABLE IF NOT EXISTS queries (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    final_status TEXT NOT NULL,
    record TEXT NOT NULL
);
"""
DB_NAME = "smartrouter.db"


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        try:
            # isolation_level=None: every statement commits on its own (one INSERT = one atomic transaction).
            self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=10)
            self._conn.execute("PRAGMA journal_mode=WAL")
            # NORMAL in WAL mode: a power cut can lose the last few commits but cannot corrupt the database.
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)
        except sqlite3.DatabaseError as exc:
            from app.core.config import ConfigError  # local import: config does not depend on logging

            raise ConfigError(f"Cannot open the log database {path} ({exc}). Move it aside to start with an empty one.") from None

    def add_event(self, timestamp: str, query_id: str, event_type: str, payload: dict) -> None:
        data = json.dumps(payload, default=str)
        with self._lock:
            self._conn.execute(
                "INSERT INTO events(timestamp, query_id, event_type, payload) VALUES (?, ?, ?, ?)", (timestamp, query_id, event_type, data)
            )

    def events(self, *, query_id: str | None = None, event_type: str | None = None) -> list[dict]:
        sql, args, where = "SELECT timestamp, query_id, event_type, payload FROM events", [], []
        if query_id is not None:
            where.append("query_id = ?")
            args.append(query_id)
        if event_type is not None:
            where.append("event_type = ?")
            args.append(event_type)
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY id", args).fetchall()
        return [{"timestamp": t, "query_id": q, "event_type": e, "payload": json.loads(p)} for t, q, e, p in rows]

    def provider_error_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT json_extract(payload, '$.provider') || ':' || json_extract(payload, '$.error_type'), COUNT(*) "
                "FROM events WHERE event_type = 'provider_error' GROUP BY 1 ORDER BY 2 DESC"
            ).fetchall()
        return {key: count for key, count in rows if key}

    def put_record(self, query_id: str, timestamp: str, final_status: str, record_json: str) -> None:
        with self._lock:
            # A re-used query_id (e.g. a benchmark item id rerun into the same directory) replaces the old record.
            self._conn.execute(
                "INSERT INTO queries(query_id, timestamp, final_status, record) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(query_id) DO UPDATE SET timestamp=excluded.timestamp, final_status=excluded.final_status, "
                "record=excluded.record",
                (query_id, timestamp, final_status, record_json),
            )

    def get_record(self, query_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT record FROM queries WHERE query_id = ?", (query_id,)).fetchone()
        return row[0] if row else None

    def recent_records(self, limit: int) -> list[str]:
        """The newest `limit` records, oldest first."""
        with self._lock:
            rows = self._conn.execute("SELECT record FROM queries ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
        return [r[0] for r in reversed(rows)]

    def count_records(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
