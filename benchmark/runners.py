"""The four benchmark systems. Each is a pipeline mode, so no pipeline logic is duplicated here.

Deviation from a runners/ package with one file per baseline: the four files
would differ by a single string, so they are rows in one table instead.
"""

import asyncio
import sys
from dataclasses import dataclass

from pydantic import BaseModel

from app.core.bootstrap import Runtime
from app.judge.judge import JudgeParseError
from app.providers.base import ProviderError
from app.schemas.common import Mode, QueryRecord
from benchmark.dataset import BenchmarkItem
from benchmark.evaluators import deterministic_check, is_checkable


@dataclass(frozen=True)
class System:
    name: str
    label: str
    mode: Mode
    description: str


# `always_weak` keeps its historical name so older raw runs in data/benchmarks/ still re-analyze.
SYSTEMS: tuple[System, ...] = (
    System("always_weak", "Always Cheap", "always_cheap", "Every query goes to the cheap tier; no judge needed in production."),
    System("always_mid", "Always Mid", "always_mid", "Every query goes to the mid tier."),
    System("always_strong", "Always Strong", "always_strong", "Every query goes to the strong tier (quality ceiling / cost reference)."),
    System("static_router", "Static Router", "static", "Heuristic difficulty routing; no judge-driven escalation."),
    System("smart_router", "SmartRouter", "smart", "Heuristic routing + answer checks + judge-driven escalation."),
)
SYSTEM_BY_NAME = {s.name: s for s in SYSTEMS}


class ItemResult(BaseModel):
    system: str
    item_id: str
    category: str
    difficulty: str
    record: QueryRecord
    reference_quality: float | None = None  # judge WITH the reference answer; used to compare systems
    reference_verdict: str | None = None
    reference_error: str | None = None
    deterministic_correct: bool | None = None  # numeric / code-test check; None when not applicable or not parseable
    checkable: bool = False  # the item has a deterministic check that ran in this benchmark (numeric, or code with --exec-code)
    attempt_correct: list[bool | None] = []  # the same check on every attempt's answer (None: no answer / not checkable)


async def run_item(runtime: Runtime, system: System, item: BenchmarkItem, *, reference_eval: bool, exec_code: bool) -> ItemResult:
    # The in-loop judge never sees the reference answer (production realism).
    record = await runtime.pipeline.run(item.question, mode=system.mode, query_id=f"{system.name}-{item.id}")
    result = ItemResult(system=system.name, item_id=item.id, category=item.category, difficulty=item.difficulty, record=record)
    if not runtime.settings.mock_mode:  # mock answers are synthetic text; deterministic checks would be meaningless
        result.checkable = is_checkable(item, exec_code=exec_code)
        result.deterministic_correct = deterministic_check(item, record.answer, exec_code=exec_code)
        result.attempt_correct = [
            deterministic_check(item, a.answer, exec_code=exec_code) if a.answer is not None else None for a in record.attempts
        ]
    if reference_eval:
        if record.answer is None:
            result.reference_quality, result.reference_verdict = 0.0, "no_answer"
        else:
            try:
                j = await runtime.judge.evaluate(item.question, record.answer, item.category, reference=item.reference_answer)
                result.reference_quality, result.reference_verdict = j.quality_score, j.verdict
            except (ProviderError, JudgeParseError) as exc:
                result.reference_error = runtime.settings.redact(f"{type(exc).__name__}: {exc}")
    return result


async def run_system(
    runtime: Runtime, system: System, items: list[BenchmarkItem], *, concurrency: int, delay_s: float, reference_eval: bool, exec_code: bool
) -> list[ItemResult]:
    """Run one system over all items with bounded concurrency (1 = sequential, the free-tier default)."""
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def one(item: BenchmarkItem) -> ItemResult:
        nonlocal done
        async with semaphore:
            result = await run_item(runtime, system, item, reference_eval=reference_eval, exec_code=exec_code)
            if delay_s:
                await asyncio.sleep(delay_s)
        done += 1
        q = result.record.final_quality_score
        print(f"[{system.name}] {done}/{len(items)} {item.id}: {result.record.final_status}"
              f" tier={result.record.final_tier} q={'n/a' if q is None else f'{q:.2f}'}", file=sys.stderr)
        return result

    return list(await asyncio.gather(*(one(item) for item in items)))
