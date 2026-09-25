from app.core.metrics import cost_reduction_pct, failure_groups, percentile, routing_flow, summarize
from app.router.schemas import QueryAnalysis
from app.schemas.common import Attempt, QueryRecord


def attempt(tier, q=None, accepted=False, reasons=(), error_type=None, cost=None):
    costs = {"cheap": 1.0, "mid": 3.0, "strong": 10.0}
    return Attempt(number=1, model=f"m-{tier}", provider="p", tier=tier, routing_reason="r", quality_score=q, accepted=accepted,
                   failure_reasons=list(reasons), error_type=error_type, normalized_cost=costs[tier] if cost is None else cost,
                   answer=None if error_type and error_type != "JudgeParseError" else "a")


def record(attempts, status, category="math", latency=100.0):
    final = next((a for a in attempts if a.accepted), None) or (attempts[-1] if status != "failed" else None)
    return QueryRecord(
        query_id="q", timestamp="t", query="x", mode="smart", mock_mode=True, category=category, difficulty=0.5,
        analysis=QueryAnalysis(category=category, difficulty=0.5), quality_threshold=0.78, attempts=attempts,
        initial_model=attempts[0].model, initial_tier=attempts[0].tier,
        final_model=final.model if final else None, final_tier=final.tier if final else None, answer=final and final.answer,
        final_quality_score=final.quality_score if final else None, final_status=status,
        escalated=len(attempts) > 1, escalation_count=len(attempts) - 1,
        generation_latency_ms=latency, judge_latency_ms=0, total_latency_ms=latency,
        normalized_cost=sum(a.normalized_cost for a in attempts), judge_calls=len(attempts), trace=[],
    )


RECORDS = [
    record([attempt("cheap", 0.9, True)], "accepted", "general_qa", 100),
    record([attempt("cheap", 0.5, reasons=["incomplete"]), attempt("mid", 0.8, True)], "accepted", "math", 300),
    record([attempt("strong", 0.6, reasons=["incorrect", "reasoning_error"])], "best_effort", "math", 500),
    record([attempt("cheap", error_type="RateLimitError", cost=0), attempt("mid", error_type="AuthenticationError", cost=0),
            attempt("strong", error_type="ProviderServerError", cost=0)], "failed", "math", 50),
]


def test_percentile_nearest_rank():
    assert percentile(range(1, 101), 95) == 95
    assert percentile([5], 95) == 5
    assert percentile([], 95) is None


def test_cost_reduction():
    assert cost_reduction_pct(2.5, 10) == 75.0
    assert cost_reduction_pct(12, 10) == -20.0
    assert cost_reduction_pct(1, 0) is None


def test_summarize():
    s = summarize(RECORDS)
    assert s["n"] == 4
    assert s["quality"]["mean"] == round((0.9 + 0.8 + 0.6) / 3, 4) and s["quality"]["scored"] == 3
    assert s["cost"]["total"] == 1 + 4 + 10 + 0
    assert s["utilization"]["calls"] == {"cheap": 3, "mid": 2, "strong": 2}
    assert s["escalation"]["rate"] == 0.5 and s["escalation"]["mean_count"] == 0.75
    assert s["failure_rate"] == 0.25 and s["below_threshold_rate"] == 0.5
    assert s["latency_ms"]["median"] == 200.0
    assert s["per_category"]["math"]["n"] == 3


def test_failure_groups_and_flow():
    assert failure_groups(RECORDS) == {
        "Incorrect": 1, "Incomplete": 1, "Reasoning weakness": 1, "Provider error": 2, "Rate limit": 1,
    }
    flow = {(e["from"], e["to"]): e["count"] for e in routing_flow(RECORDS)}
    assert flow == {("cheap", "accepted"): 1, ("cheap", "mid"): 2, ("mid", "accepted"): 1, ("strong", "best_effort"): 1,
                    ("mid", "strong"): 1, ("strong", "failed"): 1}


def test_empty():
    assert summarize([]) == {"n": 0}
