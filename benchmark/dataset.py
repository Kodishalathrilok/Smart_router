"""Benchmark dataset schema and loader. The dataset is a fixed, versioned JSONL file."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.router.schemas import Category

DATASET_PATH = Path(__file__).parent / "datasets" / "benchmark.jsonl"


class BenchmarkItem(BaseModel):
    id: str
    category: Category
    difficulty: Literal["easy", "medium", "hard"]
    question: str
    reference_answer: str
    answer_type: Literal["free", "numeric", "code"] = "free"
    expected: float | None = None  # numeric items
    tolerance: float | None = None  # absolute tolerance for numeric items (default: relative 1e-4)
    tests: str | None = None  # code items: Python asserts run against the extracted solution


def load_dataset(path: Path = DATASET_PATH) -> list[BenchmarkItem]:
    with path.open(encoding="utf-8") as f:
        items = [BenchmarkItem.model_validate(json.loads(line)) for line in f if line.strip()]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate item ids in {path}")
    return items
