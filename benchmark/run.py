"""Run the benchmark: python -m benchmark.run [--dataset NAME] [--systems ...] [--per-category N] [--concurrency N] [--resume RUN_ID]

Writes raw per-item results to BENCHMARK_DIR/<run_id>/ and the summary to
RESULTS_DIR/latest.json, then renders RESULTS_DIR/latest.md.

Every item is saved as it finishes (<system>.partial.jsonl, renamed to <system>.jsonl when the system is complete).
If a provider keeps rate-limiting after all retries (usually a daily quota), the run stops with exit code 3 and
`--resume RUN_ID` with the same arguments continues it without repeating saved work.
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timezone

from app.core.bootstrap import build_runtime
from app.core.config import ConfigError, Settings
from app.core.metrics import cost_reduction_pct, mean, med, rate, record_cost_usd, summarize
from benchmark.analysis import analyzer_validation, capability, escalation_breakdown, item_rows, judge_confusion, second_opinion_stats
from benchmark.dataset import DATASET_NAMES, DATASETS_DIR, BenchmarkItem, load_dataset
from benchmark.report import write_report
from benchmark.runners import SYSTEMS, ItemResult, QuotaStop, System, run_system
from benchmark.stats import claim_verdict, paired, strict_accuracy

# (key, system, baseline): paired item-level comparisons reported in every run
PAIRS = (
    ("smart_vs_cheap", "smart_router", "always_weak"),
    ("smart_vs_mid", "smart_router", "always_mid"),
    ("smart_vs_strong", "smart_router", "always_strong"),
    ("static_vs_cheap", "static_router", "always_weak"),
    ("strong_vs_cheap", "always_strong", "always_weak"),
)


QUOTA_EXIT_CODE = 3
JUDGE_ROLES = ("judge", "judge_fallback", "second_opinion")
GENERATION_ROLES = ("cheap", "mid", "strong")


def serving_cost(system: System, results: list[ItemResult], role_costs: dict[str, float], generation_total: float) -> dict:
    """What production would pay. Baselines never need a judge when serving; SmartRouter pays for every judge
    response it received (parse retries, fallback judge and second opinions included).

    Exact when records carry billed_calls (runs made after the audit). Older records only have judge_calls,
    which counted one call per judged attempt even when a parse retry or the fallback judge made more:
    that undercounts SmartRouter's cost, so it is labeled "estimated_lower_bound".
    """
    n = len(results)
    exact = all(r.record.billed_calls for r in results)
    gen_calls = sum(sum(r.record.billed_calls.get(role, 0) for role in GENERATION_ROLES) for r in results) if exact else sum(
        sum(a.answer is not None for a in r.record.attempts) for r in results)
    if system.mode != "smart":
        judge_calls, judge_cost = 0, 0.0
    elif exact:
        judge_calls = sum(r.record.billed_calls.get(role, 0) for r in results for role in JUDGE_ROLES)
        judge_cost = sum(r.record.billed_calls.get(role, 0) * role_costs.get(role, role_costs["judge"]) for r in results for role in JUDGE_ROLES)
    else:
        judge_calls = sum(r.record.judge_calls for r in results)
        judge_cost = judge_calls * role_costs["judge"]
    return {
        "method": "exact" if exact else "estimated_lower_bound",
        "serving_judge_calls": judge_calls,
        "serving_calls_mean": round((gen_calls + judge_calls) / n, 3) if n else None,
        "cost_incl_judge_mean": round((generation_total + judge_cost) / n, 4) if n else None,
    }


def price_table(profiles: dict[str, dict]) -> dict[str, dict]:
    """role -> list price, from config/models.json-shaped profiles."""
    return {role: {"model": p.get("price_model"), "input": p.get("price_per_mtok_input"), "output": p.get("price_per_mtok_output")}
            for role, p in profiles.items()}


def dollar_cost(results: list[ItemResult], prices: dict[str, dict], roles: tuple[str, ...] | None) -> dict:
    """USD per 1,000 queries from stored tokens. None ("unknown") unless every query could be priced in full."""
    costs = [record_cost_usd(r.record, prices, roles) for r in results]
    known = [c for c in costs if c is not None]
    complete = bool(costs) and len(known) == len(costs)
    return {"per_1k_queries": round(sum(known) / len(known) * 1000, 4) if complete else None, "priced_queries": len(known), "queries": len(costs)}


def summarize_system(system: System, results: list[ItemResult], profiles: dict[str, dict]) -> dict:
    """profiles: role -> config/models.json entry (normalized_cost and list prices)."""
    role_costs = {role: p["normalized_cost"] for role, p in profiles.items()}
    category = {r.record.query_id: r.category for r in results}
    base = summarize([r.record for r in results], category_of=lambda rec: category[rec.query_id])
    ref = [r.reference_quality for r in results if r.reference_quality is not None]
    det = [r.deterministic_correct for r in results if r.deterministic_correct is not None]  # lenient: unparseable answers dropped
    for cat, stats in base["per_category"].items():
        stats["reference_quality_mean"] = mean(r.reference_quality for r in results if r.category == cat and r.reference_quality is not None)
    serving = serving_cost(system, results, role_costs, base["cost"]["total"])
    generation_failures = sum(1 for r in results for a in r.record.attempts if a.answer is None)
    judge_failures = sum(
        1 for r in results for a in r.record.attempts if a.answer is not None and a.quality_score is None
    ) + sum(r.reference_error is not None for r in results)
    judge_usage = Counter(a.judge_model for r in results for a in r.record.attempts if a.judge_model)
    tier_accuracy: dict[str, dict[str, int]] = {}
    for r in results:
        if r.deterministic_correct is None:
            continue
        tier = r.record.final_tier or r.record.attempts[-1].tier  # tier that produced (or failed to produce) the answer
        stats = tier_accuracy.setdefault(tier, {"correct": 0, "checked": 0})
        stats["correct"] += r.deterministic_correct
        stats["checked"] += 1
    return {
        **base,
        "label": system.label,
        "mode": system.mode,
        "description": system.description,
        "reference_quality": {"mean": mean(ref), "median": med(ref), "scored": len(ref), "errors": sum(r.reference_error is not None for r in results)},
        "strict": strict_accuracy(results),
        "capability": capability(results),
        "deterministic": {"checked": len(det), "accuracy": rate(sum(det), len(det))},
        "serving_calls_mean": serving["serving_calls_mean"],
        "cost_method": serving["method"],
        "generation_failures": generation_failures,
        "provider_calls_mean": mean(r.record.provider_calls for r in results if r.record.provider_calls),
        "serving_judge_calls": serving["serving_judge_calls"],
        "judge_failures": judge_failures,
        "judge_usage": dict(judge_usage),
        "tier_accuracy": tier_accuracy,
        "strong_calls": base["utilization"]["calls"]["strong"],
        "escalations": escalation_breakdown(results),
        "judge_accuracy": {"primary_judge": judge_confusion(results, decision=False), "decision": judge_confusion(results, decision=True)},
        "items": item_rows(results),
        "second_opinions": second_opinion_stats(results),
        "cost_incl_judge": {"mean": serving["cost_incl_judge_mean"]},
        # Same serving convention as cost_incl_judge: only SmartRouter pays for judges in production.
        "judge_usd": dollar_cost(results, price_table(profiles), JUDGE_ROLES),  # every judge-side token this system's run used
        "cost_usd": dollar_cost(results, price_table(profiles), None if system.mode == "smart" else GENERATION_ROLES),
    }


def paired_comparisons(results: dict[str, list[ItemResult]]) -> dict:
    return {key: paired(results[a], results[b]) for key, a, b in PAIRS if a in results and b in results}


def compare(systems: dict[str, dict]) -> dict:
    strong, cheap = systems.get("always_strong"), systems.get("always_weak")
    out: dict = {}
    if cheap:
        for name, s in systems.items():
            out.setdefault(name, {})["cost_incl_judge_vs_cheap_x"] = (
                round(s["cost_incl_judge"]["mean"] / cheap["cost_incl_judge"]["mean"], 2) if cheap["cost_incl_judge"]["mean"] else None
            )
    if not strong:
        return out
    for name, s in systems.items():
        ref, strong_ref = s["reference_quality"]["mean"], strong["reference_quality"]["mean"]
        out.setdefault(name, {}).update({
            "cost_reduction_pct_vs_strong": cost_reduction_pct(s["cost"]["mean"], strong["cost"]["mean"]),
            "cost_incl_judge_reduction_pct_vs_strong": cost_reduction_pct(s["cost_incl_judge"]["mean"], strong["cost_incl_judge"]["mean"]),
            "strong_call_reduction_pct": cost_reduction_pct(s["utilization"]["calls"]["strong"], strong["utilization"]["calls"]["strong"]),
            "reference_quality_delta_vs_strong": round(ref - strong_ref, 4) if ref is not None and strong_ref is not None else None,
            "median_latency_delta_ms_vs_strong": round(s["latency_ms"]["median"] - strong["latency_ms"]["median"], 1),
        })
    return out


def assemble(systems: dict[str, dict], results: dict[str, list[ItemResult]]) -> dict:
    """Comparisons, paired tests and the claim verdict. Shared by run() and benchmark.reanalyze."""
    pairs = paired_comparisons(results)
    verdict, explanation = claim_verdict(systems, pairs)
    return {"comparisons": compare(systems), "paired": pairs, "claim": {"verdict": verdict, "explanation": explanation}}


def select_items(items: list[BenchmarkItem], per_category: int | None) -> list[BenchmarkItem]:
    """Pick `per_category` items per category, evenly spaced so the subset spans easy → hard
    (items are stored easiest-first; taking the first N would select only easy items)."""
    if per_category is None:
        return items
    chosen = []
    for category in dict.fromkeys(i.category for i in items):
        pool = [i for i in items if i.category == category]
        k = min(per_category, len(pool))
        step = (len(pool) - 1) / (k - 1) if k > 1 else 0
        chosen += [pool[round(n * step)] for n in range(k)]
    return chosen


async def run(args: argparse.Namespace, settings: Settings) -> dict:
    dataset_path = DATASETS_DIR / f"{args.dataset}.jsonl"
    items = select_items(load_dataset(dataset_path), args.per_category)
    systems = [s for s in SYSTEMS if s.name in args.systems]
    run_id = args.resume or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = settings.benchmark_dir / run_id
    if args.resume and not raw_dir.is_dir():
        raise ConfigError(f"--resume {run_id}: no such run under {settings.benchmark_dir}")
    runtime = await build_runtime(settings, log_dir=raw_dir)
    summaries: dict[str, dict] = {}
    all_results: dict[str, list[ItemResult]] = {}
    try:
        for system in systems:
            final, partial = raw_dir / f"{system.name}.jsonl", raw_dir / f"{system.name}.partial.jsonl"
            wanted = {i.id for i in items}
            if final.exists():  # finished before the run was interrupted
                results = [r for r in _load(final) if r.item_id in wanted]
                if len(results) != len(items):
                    raise ConfigError(f"{final} does not cover the selected items; resume with the arguments the run was started with")
            else:
                with partial.open("a", encoding="utf-8") as f:
                    def save(result: ItemResult, f=f) -> None:
                        f.write(result.model_dump_json() + "\n")
                        f.flush()

                    try:
                        results = await run_system(
                            runtime, system, items, concurrency=args.concurrency, delay_s=args.delay, reference_eval=args.reference_eval,
                            exec_code=args.exec_code, done_before=[r for r in _load(partial) if r.item_id in wanted], on_result=save,
                        )
                    except QuotaStop as stop:
                        stop.run_id = run_id
                        raise
                with final.open("w", encoding="utf-8") as f:
                    f.writelines(r.model_dump_json() + "\n" for r in results)
                partial.unlink()
            all_results[system.name] = results
            summaries[system.name] = summarize_system(system, results, {s.role: s.model_dump() for s in runtime.registry.all()})
    finally:
        await runtime.aclose()

    summary = {
        "status": "complete",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mock_mode": settings.mock_mode,
        "dataset": {
            "path": dataset_path.name,
            "n": len(items),
            "per_category": dict(Counter(i.category for i in items)),
            "per_difficulty": dict(Counter(i.difficulty for i in items)),
        },
        "config": {
            "quality_threshold": settings.quality_threshold,
            "max_escalations": settings.max_escalations,
            "routing_thresholds": {"cheap_max": settings.cheap_max_difficulty, "mid_max": settings.mid_max_difficulty},
            "models": {s.role: {"id": s.id, "provider": s.provider, "normalized_cost": s.normalized_cost,
                                "usd_per_mtok": [s.price_per_mtok_input, s.price_per_mtok_output] if s.price_model == s.id else None}
                       for s in runtime.registry.all()},
            "reference_eval": args.reference_eval,
            "exec_code": args.exec_code,
            "concurrency": args.concurrency,
            "max_provider_calls_per_query": settings.max_provider_calls_per_query,
        },
        "analyzer": analyzer_validation(items, settings.cheap_max_difficulty, settings.mid_max_difficulty),
        "systems": summaries,
        **assemble(summaries, all_results),
        "raw_dir": str(raw_dir),
    }
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, settings.results_dir / "latest.md")
    return summary


def _load(path) -> list[ItemResult]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [ItemResult.model_validate_json(line) for line in f if line.strip()]


def parse_args(argv: list[str] | None, settings: Settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m benchmark.run", description=__doc__.splitlines()[0])
    parser.add_argument("--systems", nargs="+", default=[s.name for s in SYSTEMS], choices=[s.name for s in SYSTEMS])
    parser.add_argument("--dataset", default="benchmark", choices=DATASET_NAMES, help="Dataset file in benchmark/datasets/ (default: benchmark).")
    parser.add_argument("--per-category", type=int, default=None, help="Use N items per category, spread from easy to hard.")
    parser.add_argument("--concurrency", type=int, default=settings.max_concurrent_requests, help="Parallel queries (1 = sequential).")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to pause after each query (free-tier rate limits).")
    parser.add_argument("--no-reference-eval", dest="reference_eval", action="store_false", help="Skip the reference-answer judge pass.")
    parser.add_argument("--exec-code", action="store_true", help="Run model-written code against item tests (executes untrusted code locally).")
    parser.add_argument("--resume", metavar="RUN_ID", default=None,
                        help="Continue an interrupted run (same other arguments): finished systems and saved items are not run again.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    settings = Settings()
    args = parse_args(argv, settings)
    try:
        summary = asyncio.run(run(args, settings))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except QuotaStop as stop:
        given = [a for a in (sys.argv[1:] if argv is None else argv)]
        if "--resume" in given:
            del given[given.index("--resume"):given.index("--resume") + 2]
        print(f"STOPPED: a provider kept rate-limiting after all retries (usually a daily quota): {stop.detail[:200]}\n"
              f"Progress: run {stop.run_id} stopped in {stop.system} at {stop.done}/{stop.total} items; earlier systems are complete "
              "and every finished item is saved.\n"
              f"Resume when the quota resets: python -m benchmark.run {' '.join(given)} --resume {stop.run_id}", file=sys.stderr)
        return QUOTA_EXIT_CODE
    print(f"Wrote {settings.results_dir / 'latest.json'} and latest.md ({summary['dataset']['n']} items, mock_mode={summary['mock_mode']})")
    gen_failures = {name: s["generation_failures"] for name, s in summary["systems"].items() if s.get("generation_failures")}
    if gen_failures:
        print(f"WARNING: generation calls failed {gen_failures}. Failed queries count as wrong, so these systems look worse than they are; "
              "check provider_error events (rate limits, REQUEST_TIMEOUT_S) and consider re-running.", file=sys.stderr)
    print(f"Claim verdict: {summary['claim']['verdict']}", file=sys.stderr)
    failures = {name: s["judge_failures"] for name, s in summary["systems"].items() if s["judge_failures"]}
    if failures:
        print(f"WARNING: incomplete run, judge calls failed {failures}. Quality numbers are not comparable; "
              "see events.jsonl in the raw run directory (often a provider quota limit).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
