"""Consolidated per-query records (JSONL), one line per completed pipeline run."""

import json
from pathlib import Path

from app.schemas.common import QueryRecord


class QueryLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: QueryRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def all(self) -> list[QueryRecord]:
        # ponytail: full-file scan per call; fine for thousands of records, add an index/SQLite when it isn't.
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [QueryRecord.model_validate(json.loads(line)) for line in f if line.strip()]

    def get(self, query_id: str) -> QueryRecord | None:
        return next((r for r in self.all() if r.query_id == query_id), None)
