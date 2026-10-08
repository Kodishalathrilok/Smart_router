"""Aggregate metrics over query records. Shared by the benchmark and the dashboard API."""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from statistics import fmean, median

from app.router.schemas import TIERS
from app.schemas.common import Attempt, QueryRecord

REASON_GROUPS = {
    "incorrect": "Incorrect",
    "incomplete": "Incomplete",
    "irrelevant": "Low relevance",
    "reasoning_error": "Reasoning weakness",
    "unsafe": "Unsafe",
    "other": "Other",
}
FAILURE_GROUP_ORDER = (
    "Incorrect", "Incomplete", "Low relevance", "Reasoning weakness", "Provider error", "Rate limit", "Judge error", "Unsafe", "Other",
    "Below threshold (no reason given)",
)


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(fmean(values), 4) if values else None


def med(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(median(values), 4) if values else None


def percentile(values: Iterable[float], pct: float) -> float | None:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, -(-len(ordered) * pct // 100))  # ceil
    return ordered[int(rank) - 1]


def rate(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None


def cost_reduction_pct(cost: float | None, baseline: float | None) -> float | None:
    """Percent reduction of `cost` relative to `baseline` (negative = more expensive)."""
    if cost is None or not baseline:
        return None
    return round((1 - cost / baseline) * 100, 1)


def summarize(records: Sequence[QueryRecord], *, category_of: Callable[[QueryRecord], str] | None = None) -> dict:
    """Quality, cost, utilization, escalation, latency and failure metrics for a set of records."""
    n = len(records)
    if not n:
        return {"n": 0}
    category_of = category_of or (lambda r: r.category)
    scores = [r.final_quality_score for r in records if r.final_quality_score is not None]
    calls = Counter(a.tier for r in records for a in r.attempts)
    total_calls = sum(calls.values())
    finals = Counter(r.final_tier for r in records if r.final_tier)

    by_category: dict[str, list[QueryRecord]] = defaultdict(list)
    for r in records:
        by_category[category_of(r)].append(r)

    return {
        "n": n,
        "quality": {
            "mean": mean(scores),
            "median": med(scores),
            "scored": len(scores),
        },
        "cost": {
            "mean": mean(r.normalized_cost for r in records),
            "total": round(sum(r.normalized_cost for r in records), 2),
        },
        "utilization": {
            "calls": {t: calls.get(t, 0) for t in TIERS},
            "calls_pct": {t: rate(calls.get(t, 0), total_calls) for t in TIERS},
            "final_pct": {t: rate(finals.get(t, 0), n) for t in TIERS},
        },
        "escalation": {
            "rate": rate(sum(r.escalated for r in records), n),
            "mean_count": mean(r.escalation_count for r in records),
        },
        "latency_ms": {
            "mean": mean(r.total_latency_ms for r in records),
            "median": med(r.total_latency_ms for r in records),
            "p95": percentile((r.total_latency_ms for r in records), 95),
            "generation_median": med(r.generation_latency_ms for r in records),
        },
        "judge_calls": sum(r.judge_calls for r in records),
        "failure_groups": failure_groups(records),
        "routing_flow": routing_flow(records),
        "failure_rate": rate(sum(r.final_status == "failed" for r in records), n),
        "below_threshold_rate": rate(sum(r.final_status != "accepted" for r in records), n),
        "per_category": {
            cat: {
                "n": len(rs),
                "quality_mean": mean(r.final_quality_score for r in rs if r.final_quality_score is not None),
                "cost_mean": mean(r.normalized_cost for r in rs),
                "escalation_rate": rate(sum(r.escalated for r in rs), len(rs)),
            }
            for cat, rs in sorted(by_category.items())
        },
    }


def attempt_failure_groups(attempt: Attempt) -> list[str]:
    """Dashboard failure buckets for one non-accepted attempt."""
    if attempt.accepted:
        return []
    if attempt.error_type == "RateLimitError":
        return ["Rate limit"]
    if attempt.error_type == "JudgeParseError":
        return ["Judge error"]
    if attempt.error_type:
        return ["Provider error"]
    groups = [REASON_GROUPS.get(code, "Other") for code in attempt.failure_reasons]
    return list(dict.fromkeys(groups)) or ["Below threshold (no reason given)"]


def failure_groups(records: Iterable[QueryRecord]) -> dict[str, int]:
    counts = Counter(g for r in records for a in r.attempts for g in attempt_failure_groups(a))
    return {g: counts[g] for g in FAILURE_GROUP_ORDER if counts.get(g)}


def routing_flow(records: Iterable[QueryRecord]) -> list[dict]:
    """Edge counts: tier → next tier on escalation, last tier → final outcome."""
    edges: Counter[tuple[str, str]] = Counter()
    for r in records:
        for current, following in zip(r.attempts, r.attempts[1:], strict=False):
            edges[(current.tier, following.tier)] += 1
        last = r.attempts[-1]
        edges[(last.tier, "accepted" if last.accepted else r.final_status)] += 1
    return [{"from": a, "to": b, "count": c} for (a, b), c in sorted(edges.items())]
