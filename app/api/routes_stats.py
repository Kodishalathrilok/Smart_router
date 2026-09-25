"""Analytics endpoints over logged queries, plus the latest benchmark summary."""

import json
from collections import Counter

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_runtime
from app.core.bootstrap import Runtime
from app.core.metrics import attempt_failure_groups, cost_reduction_pct, failure_groups, summarize

router = APIRouter(prefix="/api")
MAX_ROUTES = 500


@router.get("/stats")
async def stats(runtime: Runtime = Depends(get_runtime)) -> dict:
    """Live-traffic metrics. Cost reduction is against a counterfactual 'every query on the strong tier'
    using the configured normalized weights; it is not a measured baseline (run the benchmark for that)."""
    records = runtime.queries.all()
    summary = summarize(records)
    strong_cost = runtime.registry.for_tier("strong").normalized_cost
    summary["cost_reduction_vs_all_strong_pct"] = cost_reduction_pct(summary.get("cost", {}).get("mean"), strong_cost)
    summary["quality_threshold"] = runtime.settings.quality_threshold
    summary["mock_mode"] = runtime.settings.mock_mode
    summary["provider_errors"] = dict(Counter(
        f"{e['payload']['provider']}:{e['payload']['error_type']}" for e in runtime.events.read() if e["event_type"] == "provider_error"
    ))
    return summary


@router.get("/routes")
async def routes(limit: int = Query(100, ge=1, le=MAX_ROUTES), runtime: Runtime = Depends(get_runtime)) -> list[dict]:
    """Most recent routing decisions, newest first."""
    return [
        {
            "query_id": r.query_id,
            "timestamp": r.timestamp,
            "query": r.query[:200],
            "category": r.category,
            "difficulty": r.difficulty,
            "initial_model": r.initial_model,
            "initial_tier": r.initial_tier,
            "final_model": r.final_model,
            "final_tier": r.final_tier,
            "quality_score": r.final_quality_score,
            "final_status": r.final_status,
            "escalated": r.escalated,
            "attempts": len(r.attempts),
            "latency_ms": r.total_latency_ms,
            "normalized_cost": r.normalized_cost,
        }
        for r in reversed(runtime.queries.all()[-limit:])
    ]


@router.get("/failures")
async def failures(limit: int = Query(50, ge=1, le=MAX_ROUTES), runtime: Runtime = Depends(get_runtime)) -> dict:
    records = runtime.queries.all()
    examples = [
        {
            "query_id": r.query_id,
            "query": r.query[:200],
            "model": a.model,
            "tier": a.tier,
            "quality_score": a.quality_score,
            "groups": attempt_failure_groups(a),
            "error": a.error,
        }
        for r in reversed(records)
        for a in r.attempts
        if not a.accepted
    ][:limit]
    return {"groups": failure_groups(records), "recent": examples}


@router.get("/benchmarks/latest")
async def latest_benchmark(runtime: Runtime = Depends(get_runtime)) -> dict:
    path = runtime.settings.results_dir / "latest.json"
    if not path.exists():
        return {"status": "not_evaluated", "message": "Not evaluated yet. Run `python -m benchmark.run`."}
    return json.loads(path.read_text(encoding="utf-8"))
