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
from benchmark.evaluators import deterministic_check


@dataclass(frozen=True)
class System:
    name: str
    label: str
    mode: Mode
    description: str


SYSTEMS: tuple[System, ...] = (
    System("always_weak", "Always Weak", "always_cheap", "Baseline A: every query goes to the cheap tier; no escalation."),
    System("always_strong", "Always Strong", "always_strong", "Baseline B: every query goes to the strong tier (quality/cost reference)."),
    System("static_router", "Static Router", "static", "Baseline C: heuristic difficulty routing; no judge-driven escalation."),
    System("smart_router", "SmartRouter", "smart", "Baseline D: heuristic routing + quality judge + adaptive escalation."),
)


class ItemResult(BaseModel):
    system: str
    item_id: str
    category: str
    difficulty: str
    record: QueryRecord
    reference_quality: float | None = None  # judge WITH the reference answer; used to compare systems
    reference_verdict: str | None = None
    reference_error: str | None = None
    deterministic_correct: bool | None = None  # numeric / code-test check; None when not applicable


async def run_item(runtime: Runtime, system: System, item: BenchmarkItem, *, reference_eval: bool, exec_code: bool) -> ItemResult:
    # The in-loop judge never sees the reference answer (production realism).
    record = await runtime.pipeline.run(item.question, mode=system.mode, query_id=f"{system.name}-{item.id}")
    result = ItemResult(system=system.name, item_id=item.id, category=item.category, difficulty=item.difficulty, record=record)
    if not runtime.settings.mock_mode:  # mock answers are synthetic text; deterministic checks would be meaningless
        result.deterministic_correct = deterministic_check(item, record.answer, exec_code=exec_code)
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
