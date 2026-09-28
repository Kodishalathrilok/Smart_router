"""Run the benchmark: python -m benchmark.run [--dataset NAME] [--systems ...] [--per-category N] [--concurrency N]

Writes raw per-item results to BENCHMARK_DIR/<run_id>/ and the summary to
RESULTS_DIR/latest.json, then renders RESULTS_DIR/latest.md.
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timezone

from app.core.bootstrap import build_runtime
from app.core.config import ConfigError, Settings
from app.core.metrics import cost_reduction_pct, mean, med, rate, summarize
from benchmark.dataset import DATASET_NAMES, DATASETS_DIR, BenchmarkItem, load_dataset
from benchmark.report import write_report
from benchmark.runners import SYSTEMS, ItemResult, System, run_system


def summarize_system(system: System, results: list[ItemResult], judge_cost: float) -> dict:
    category = {r.record.query_id: r.category for r in results}
    base = summarize([r.record for r in results], category_of=lambda rec: category[rec.query_id])
    ref = [r.reference_quality for r in results if r.reference_quality is not None]
    det = [r.deterministic_correct for r in results if r.deterministic_correct is not None]
    for cat, stats in base["per_category"].items():
        stats["reference_quality_mean"] = mean(r.reference_quality for r in results if r.category == cat and r.reference_quality is not None)
    # Only SmartRouter needs the judge at serving time; for baselines the judge is evaluation-only.
    serving_judge_calls = base["judge_calls"] if system.mode == "smart" else 0
    judge_failures = sum(
        1 for r in results for a in r.record.attempts if a.answer is not None and a.quality_score is None
    ) + sum(r.reference_error is not None for r in results)
    return {
        **base,
        "label": system.label,
        "mode": system.mode,
        "description": system.description,
        "reference_quality": {"mean": mean(ref), "median": med(ref), "scored": len(ref), "errors": sum(r.reference_error is not None for r in results)},
        "deterministic": {"checked": len(det), "accuracy": rate(sum(det), len(det))},
        "serving_judge_calls": serving_judge_calls,
        "judge_failures": judge_failures,
        "cost_incl_judge": {"mean": round((base["cost"]["total"] + serving_judge_calls * judge_cost) / base["n"], 4)},
    }


def compare(systems: dict[str, dict]) -> dict:
    strong = systems.get("always_strong")
    if not strong:
        return {}
    out = {}
    for name, s in systems.items():
        ref, strong_ref = s["reference_quality"]["mean"], strong["reference_quality"]["mean"]
        out[name] = {
            "cost_reduction_pct_vs_strong": cost_reduction_pct(s["cost"]["mean"], strong["cost"]["mean"]),
            "cost_incl_judge_reduction_pct_vs_strong": cost_reduction_pct(s["cost_incl_judge"]["mean"], strong["cost_incl_judge"]["mean"]),
            "strong_call_reduction_pct": cost_reduction_pct(s["utilization"]["calls"]["strong"], strong["utilization"]["calls"]["strong"]),
            "reference_quality_delta_vs_strong": round(ref - strong_ref, 4) if ref is not None and strong_ref is not None else None,
            "median_latency_delta_ms_vs_strong": round(s["latency_ms"]["median"] - strong["latency_ms"]["median"], 1),
        }
    return out


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
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = settings.benchmark_dir / run_id
    runtime = await build_runtime(settings, log_dir=raw_dir)
    summaries = {}
    try:
        for system in systems:
            results = await run_system(
                runtime, system, items, concurrency=args.concurrency, delay_s=args.delay, reference_eval=args.reference_eval, exec_code=args.exec_code
            )
            with (raw_dir / f"{system.name}.jsonl").open("w", encoding="utf-8") as f:
                f.writelines(r.model_dump_json() + "\n" for r in results)
            summaries[system.name] = summarize_system(system, results, runtime.registry.judge.normalized_cost)
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
            "models": {s.role: {"id": s.id, "provider": s.provider, "normalized_cost": s.normalized_cost} for s in runtime.registry.all()},
            "reference_eval": args.reference_eval,
            "exec_code": args.exec_code,
            "concurrency": args.concurrency,
        },
        "systems": summaries,
        "comparisons": compare(summaries),
        "raw_dir": str(raw_dir),
    }
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, settings.results_dir / "latest.md")
    return summary


def parse_args(argv: list[str] | None, settings: Settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m benchmark.run", description=__doc__.splitlines()[0])
    parser.add_argument("--systems", nargs="+", default=[s.name for s in SYSTEMS], choices=[s.name for s in SYSTEMS])
    parser.add_argument("--dataset", default="benchmark", choices=DATASET_NAMES, help="Dataset file in benchmark/datasets/ (default: benchmark).")
    parser.add_argument("--per-category", type=int, default=None, help="Use N items per category, spread from easy to hard.")
    parser.add_argument("--concurrency", type=int, default=settings.max_concurrent_requests, help="Parallel queries (1 = sequential).")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to pause after each query (free-tier rate limits).")
    parser.add_argument("--no-reference-eval", dest="reference_eval", action="store_false", help="Skip the reference-answer judge pass.")
    parser.add_argument("--exec-code", action="store_true", help="Run model-written code against item tests (executes untrusted code locally).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    settings = Settings()
    args = parse_args(argv, settings)
    try:
        summary = asyncio.run(run(args, settings))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {settings.results_dir / 'latest.json'} and latest.md ({summary['dataset']['n']} items, mock_mode={summary['mock_mode']})")
    failures = {name: s["judge_failures"] for name, s in summary["systems"].items() if s["judge_failures"]}
    if failures:
        print(f"WARNING: incomplete run, judge calls failed {failures}. Quality numbers are not comparable; "
              "see events.jsonl in the raw run directory (often a provider quota limit).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
