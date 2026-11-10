"""The four benchmark systems. Each is a pipeline mode, so no pipeline logic is duplicated here.

Deviation from a runners/ package with one file per baseline: the four files
would differ by a single string, so they are rows in one table instead.
"""

import asyncio
import sys
from collections.abc import Callable, Sequence
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


class QuotaStop(Exception):
    """A provider kept rate-limiting after every retry (usually a daily quota): the run stops and can be resumed."""

    def __init__(self, system: str, done: int, total: int, detail: str) -> None:
        super().__init__(f"{system}: rate limited after {done}/{total} items ({detail})")
        self.system, self.done, self.total, self.detail = system, done, total, detail
        self.run_id = ""  # filled in by benchmark.run


def rate_limited(record: QueryRecord) -> str | None:
    """The rate-limit error that survived every retry in this query (generation, judge or second opinion), if any."""
    for a in record.attempts:
        if a.error_type == "RateLimitError":
            return a.error or "RateLimitError"
        if a.second_opinion and (a.second_opinion.error or "").startswith("RateLimitError"):
            return a.second_opinion.error
    return None


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
    runtime: Runtime, system: System, items: list[BenchmarkItem], *, concurrency: int, delay_s: float, reference_eval: bool, exec_code: bool,
    done_before: Sequence[ItemResult] = (), on_result: Callable[[ItemResult], None] | None = None,
) -> list[ItemResult]:
    """Run one system over all items with bounded concurrency (1 = sequential, the free-tier default).

    `done_before` are results saved by an interrupted run: those items are not run again. `on_result` is called
    with each new result as it finishes. An item that ends rate limited after all retries is not kept (it says
    nothing about the model); no further items are started and QuotaStop is raised.
    """
    semaphore = asyncio.Semaphore(concurrency)
    saved = {r.item_id: r for r in done_before}
    done = len(saved)
    stopped: str | None = None

    async def one(item: BenchmarkItem) -> ItemResult | None:
        nonlocal done, stopped
        if item.id in saved:
            return saved[item.id]
        async with semaphore:
            if stopped:
                return None
            result = await run_item(runtime, system, item, reference_eval=reference_eval, exec_code=exec_code)
            limit = rate_limited(result.record)
            if limit:
                stopped = stopped or limit
                return None
            if on_result:
                on_result(result)
            if delay_s:
                await asyncio.sleep(delay_s)
        done += 1
        q = result.record.final_quality_score
        print(f"[{system.name}] {done}/{len(items)} {item.id}: {result.record.final_status}"
              f" tier={result.record.final_tier} q={'n/a' if q is None else f'{q:.2f}'}", file=sys.stderr)
        return result

    results = await asyncio.gather(*(one(item) for item in items))
    if stopped:
        raise QuotaStop(system.name, done, len(items), stopped)
    return list(results)
