"""Evaluation-only analysis of verification and escalation, using exact checks as ground truth.

Every attempt's answer (not only the final one) is exact-checked after the run, which makes it
possible to tell necessary escalations (the escalated-from answer was wrong) from unnecessary ones
(it was already right), and to score each judge decision against the exact check.
"""

from collections.abc import Sequence
from typing import Protocol

from app.core.analyzer import HeuristicAnalyzer
from app.core.escalation import judge_passes
from app.schemas.common import QueryRecord

REASON_CHARS = 90


class Scored(Protocol):
    item_id: str
    record: QueryRecord
    attempt_correct: list[bool | None]
    deterministic_correct: bool | None


def escalation_breakdown(results: Sequence[Scored]) -> dict[str, int]:
    """Classify each escalation by whether the answer it replaced was exact-correct."""
    out = {"total": 0, "necessary": 0, "unnecessary": 0, "unknown": 0}
    for r in results:
        for i in range(len(r.record.attempts) - 1):  # every non-final attempt was escalated from
            correct = r.attempt_correct[i] if i < len(r.attempt_correct) else None
            out["total"] += 1
            out["unknown" if correct is None else "unnecessary" if correct else "necessary"] += 1
    return out


def judge_confusion(results: Sequence[Scored], *, decision: bool) -> dict[str, int]:
    """Judge-vs-exact-check agreement over all judged attempts with a known exact result.

    decision=False scores the primary judge alone; decision=True scores the final accept/escalate
    decision (after answer checks and any second opinion).
    """
    out = {"checked": 0, "agree": 0, "false_accept": 0, "false_reject": 0}
    for r in results:
        for attempt, correct in zip(r.record.attempts, r.attempt_correct, strict=False):  # attempt_correct is empty in mock runs
            if correct is None or attempt.quality_score is None:
                continue
            accepted = attempt.accepted if decision else judge_passes(attempt.quality_score, attempt.verdict, r.record.quality_threshold)
            out["checked"] += 1
            if accepted == correct:
                out["agree"] += 1
            else:
                out["false_accept" if accepted else "false_reject"] += 1
    return out


def analyzer_validation(items: Sequence, cheap_max: float, mid_max: float) -> dict:
    """How well the heuristic analyzer matches the dataset's own labels (no model calls).

    category_accuracy: predicted category == the item's category label.
    label_to_tier: how items of each difficulty label are routed. A useful difficulty estimate should send
    'hard' items up more often than 'easy' ones.
    """
    analyzer = HeuristicAnalyzer()
    correct, routed = 0, {}
    for item in items:
        analysis = analyzer.analyze(item.question)
        correct += analysis.category == item.category
        d = analysis.difficulty
        tier = "cheap" if d < cheap_max else "mid" if d < mid_max else "strong"
        row = routed.setdefault(item.difficulty, {"cheap": 0, "mid": 0, "strong": 0})
        row[tier] += 1
    return {"n": len(items), "category_accuracy": round(correct / len(items), 4) if items else None, "label_to_tier": routed}


def second_opinion_stats(results: Sequence[Scored]) -> dict:
    """Every second-opinion invocation: how often it answered and how long it took (success or failure)."""
    ops = [a.second_opinion for r in results for a in r.record.attempts if a.second_opinion]
    latencies = sorted(o.latency_ms for o in ops)
    succeeded = sum(o.error is None for o in ops)
    return {
        "requested": len(ops),
        "succeeded": succeeded,
        "failed": len(ops) - succeeded,
        "success_rate": round(succeeded / len(ops), 4) if ops else None,
        "latency_ms_median": latencies[len(latencies) // 2] if latencies else None,
        "latency_ms_max": latencies[-1] if latencies else None,
        "by_trigger": {t: sum(o.trigger == t for o in ops) for t in ("primary_rejection", "low_confidence_acceptance")},
    }


def item_rows(results: Sequence[Scored]) -> list[dict]:
    """One row per item: what the first judge saw, what was decided, and how it ended."""
    rows = []
    for r in results:
        rec = r.record
        first = rec.attempts[0]
        rows.append({
            "item_id": r.item_id,
            "initial_model": rec.initial_model,
            "judge_result": None if first.quality_score is None else f"{first.verdict} {first.quality_score:.2f}",
            "judge_confidence": first.judge_confidence,
            "second_opinion": None if not first.second_opinion else (
                ("failed" if first.second_opinion.error else f"{first.second_opinion.verdict} {first.second_opinion.quality_score:.2f}")
                + f" ({first.second_opinion.latency_ms / 1000:.1f}s)"
            ),
            "initial_exact": r.attempt_correct[0] if r.attempt_correct else None,
            "escalated": rec.escalated,
            "escalation_reason": (first.decision_reason or first.error_type or "")[:REASON_CHARS] if rec.escalated else "",
            "final_model": rec.final_model,
            "final_correct": r.deterministic_correct,
            "cost": rec.normalized_cost,
            "judge_calls": rec.judge_calls,
        })
    return rows
