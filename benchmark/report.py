"""Render RESULTS_DIR/latest.json as Markdown: python -m benchmark.report

Every number in the report comes from the measured summary. Sentences in the
executive summary are generated from those numbers, never pre-written.
"""

import json
import sys
from pathlib import Path

from app.core.config import Settings
from app.core.metrics import FAILURE_GROUP_ORDER
from app.router.schemas import TIERS

CATEGORIES = ("general_qa", "reasoning", "coding", "math", "summarization")


def fmt(value, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def change(reduction_pct) -> str:
    """Render a reduction percentage as 'X% lower' / 'X% higher'."""
    if reduction_pct is None:
        return "n/a"
    return f"{reduction_pct:.1f}% lower" if reduction_pct >= 0 else f"{-reduction_pct:.1f}% higher"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def executive_summary(s: dict) -> str:
    systems, comp = s["systems"], s.get("comparisons", {})
    smart, strong = systems.get("smart_router"), systems.get("always_strong")
    if not smart or not strong:
        return "SmartRouter and Always Strong were not both run, so no head-to-head comparison is available."
    c = comp.get("smart_router", {})
    q_smart, q_strong = smart["reference_quality"]["mean"], strong["reference_quality"]["mean"]
    delta = c.get("reference_quality_delta_vs_strong")
    direction = "n/a" if delta is None else ("higher than" if delta > 0 else "lower than" if delta < 0 else "equal to")
    return "\n".join([
        f"- Mean reference-graded quality: SmartRouter {fmt(q_smart)} vs Always Strong {fmt(q_strong)} "
        f"(SmartRouter is {direction} the strong baseline by {fmt(abs(delta) if delta is not None else None)}).",
        f"- Mean normalized generation cost per query: SmartRouter {fmt(smart['cost']['mean'], 2)} vs Always Strong {fmt(strong['cost']['mean'], 2)} "
        f"({change(c.get('cost_reduction_pct_vs_strong'))}; {change(c.get('cost_incl_judge_reduction_pct_vs_strong'))} once SmartRouter's serving-time judge calls are included).",
        f"- Strong-model calls: SmartRouter {smart['utilization']['calls']['strong']} vs Always Strong {strong['utilization']['calls']['strong']} "
        f"({change(c.get('strong_call_reduction_pct'))}).",
        f"- Median end-to-end latency: SmartRouter {fmt(smart['latency_ms']['median'], 0)} ms vs Always Strong {fmt(strong['latency_ms']['median'], 0)} ms.",
        f"- SmartRouter escalation rate: {pct(smart['escalation']['rate'])}; below-threshold (best-effort or failed) rate: {pct(smart['below_threshold_rate'])}.",
        "",
        "Research hypothesis under test: *adaptive routing can reduce strong-model utilization while retaining comparable answer quality.* "
        "Whether the differences above are meaningful depends on dataset size and judge reliability (see Limitations); no significance test is applied.",
    ])


def render(s: dict) -> str:
    systems: dict = s["systems"]
    names = list(systems)
    ds, cfg = s["dataset"], s["config"]
    out = [f"# SmartRouter Benchmark Report\n\nRun `{s['run_id']}` · generated {s['created_at']}\n"]
    if s["mock_mode"]:
        out.append("> **MOCK RUN — SYNTHETIC DATA.** Answers and judge scores come from the deterministic mock provider. "
                   "These numbers exercise the pipeline and say nothing about real model quality, cost or latency.\n")

    failures = {v["label"]: v.get("judge_failures", 0) for v in systems.values() if v.get("judge_failures")}
    if failures:
        detail = ", ".join(f"{label}: {n}" for label, n in failures.items())
        out.append(f"> **INCOMPLETE RUN.** Judge calls failed ({detail}), usually from provider quota or rate limits. "
                   "Affected answers are unscored, so quality comparisons below are not valid. Re-run once the judge is available.\n")

    out.append("## Executive Summary\n\n" + executive_summary(s) + "\n")

    out.append("## Dataset\n")
    out.append(f"{ds['n']} fixed items from `benchmark/datasets/{ds['path']}`.\n")
    out.append(table(["Category", "Items"], [[c, str(n)] for c, n in sorted(ds["per_category"].items())]) + "\n")
    out.append(table(["Difficulty label", "Items"], [[d, str(n)] for d, n in sorted(ds["per_difficulty"].items())]) + "\n")

    models = cfg["models"]
    out.append("## Methodology\n")
    out.append("\n".join([
        "1. Every item is sent through the production pipeline (`SmartRouterPipeline.run`) once per system; systems differ only in pipeline mode.",
        "2. The in-loop judge (used for SmartRouter's escalation decisions) sees only the query and candidate answer, never the reference answer.",
        "3. Final answers of all systems are then graded by the same judge model **with** the reference answer (reference-graded quality)"
        + ("" if cfg["reference_eval"] else " — *skipped in this run (--no-reference-eval)*") + ".",
        "4. Numeric items are also checked deterministically against the expected value; code items are executed against unit tests"
        + (" (enabled)." if cfg["exec_code"] else " only with `--exec-code` (disabled in this run).") + " Deterministic checks are skipped in mock mode.",
        f"5. Quality threshold {cfg['quality_threshold']}, max escalations {cfg['max_escalations']}, routing thresholds "
        f"cheap < {cfg['routing_thresholds']['cheap_max']} <= mid < {cfg['routing_thresholds']['mid_max']} <= strong, concurrency {cfg['concurrency']}.",
    ]) + "\n")
    out.append(table(["Role", "Model", "Provider", "Normalized cost"],
                     [[r, f"`{m['id']}`", m["provider"], fmt(m["normalized_cost"], 1)] for r, m in models.items()]) + "\n")

    out.append("## Baselines\n")
    out.append(table(["System", "Mode", "Description"], [[v["label"], f"`{v['mode']}`", v["description"]] for v in systems.values()]) + "\n")

    out.append("## Metrics\n")
    out.append("\n".join([
        "- **Ref. quality**: judge quality score with reference answer (0-1). Primary quality metric. Failed queries count as 0.",
        "- **In-loop quality**: reference-free judge score the pipeline itself saw (what drives escalation).",
        "- **Det. acc.**: accuracy on deterministically checkable items (numeric / code tests).",
        "- **Gen cost**: mean normalized cost per query (sum of per-call tier weights; experiment weights, not money).",
        "- **Cost incl. judge**: gen cost plus serving-time judge calls x judge weight (only SmartRouter needs the judge when serving).",
        "- **Latency**: end-to-end pipeline latency including the in-loop judge call for every system; `gen p50` is generation-only.",
        "- **Below thr.**: share of queries whose final answer did not meet the quality threshold (best effort or failed).",
        "- `quality_score` is an LLM evaluation signal, **not** a probability of correctness.",
    ]) + "\n")

    out.append("## Results\n")
    rows = []
    for n in names:
        v, c = systems[n], s["comparisons"].get(n, {})
        rows.append([
            v["label"], fmt(v["reference_quality"]["mean"]), fmt(v["quality"]["mean"]), pct(v["deterministic"]["accuracy"]),
            fmt(v["cost"]["mean"], 2), fmt(v["cost_incl_judge"]["mean"], 2), change(c.get("cost_reduction_pct_vs_strong")),
            fmt(v["latency_ms"]["median"], 0), fmt(v["latency_ms"]["p95"], 0), fmt(v["latency_ms"]["generation_median"], 0),
            str(v["utilization"]["calls"]["strong"]), pct(v["escalation"]["rate"]), pct(v["below_threshold_rate"]), pct(v["failure_rate"]),
        ])
    out.append(table(["System", "Ref. quality", "In-loop quality", "Det. acc.", "Gen cost", "Cost incl. judge", "Gen cost vs strong",
                      "p50 ms", "p95 ms", "gen p50 ms", "Strong calls", "Escalation", "Below thr.", "Failed"], rows) + "\n")

    out.append("### Model utilization (share of generation calls)\n")
    out.append(table(["System"] + [t.capitalize() for t in TIERS],
                     [[systems[n]["label"]] + [pct(systems[n]["utilization"]["calls_pct"][t]) for t in TIERS] for n in names]) + "\n")

    out.append("## Per-category Results\n")
    out.append("Reference-graded quality (mean) / normalized cost (mean):\n")
    out.append(table(["System"] + list(CATEGORIES), [
        [systems[n]["label"]] + [
            f"{fmt(pc.get('reference_quality_mean'))} / {fmt(pc.get('cost_mean'), 2)}" if (pc := systems[n]["per_category"].get(cat)) else "n/a"
            for cat in CATEGORIES
        ]
        for n in names
    ]) + "\n")

    out.append("## Failure Analysis\n")
    out.append("Counts of non-accepted attempts by cause (one attempt can have several judge-reported reasons):\n")
    groups = [g for g in FAILURE_GROUP_ORDER if any(systems[n].get("failure_groups", {}).get(g) for n in names)]
    if groups:
        out.append(table(["System"] + groups, [[systems[n]["label"]] + [str(systems[n].get("failure_groups", {}).get(g, 0)) for g in groups] for n in names]) + "\n")
    else:
        out.append("No failed attempts recorded.\n")
    if "smart_router" in systems:
        flow = systems["smart_router"].get("routing_flow", [])
        out.append("SmartRouter routing flow (edge counts):\n")
        out.append(table(["From", "To", "Count"], [[e["from"], e["to"], str(e["count"])] for e in flow]) + "\n")

    out.append("## Tradeoffs\n")
    out.append(tradeoffs(systems) + "\n")

    out.append("## Limitations\n")
    out.append("\n".join([
        "- The judge is an LLM: it has evaluator, verbosity and self-preference biases, can share blind spots with the models it grades, and scores inconsistently across calls. quality_score is not ground truth.",
        "- Reference answers are short and hand-written; open-ended items (reasoning, summarization) admit many valid answers.",
        f"- {ds['n']} items is small; differences of a few hundredths in mean quality are within plausible noise. No confidence intervals are reported.",
        "- Normalized cost uses configured tier weights, not provider billing. It measures relative model usage, not money.",
        "- Latency depends on provider load, rate limits and retries at run time and is not reproducible across runs.",
        "- Provider model versions change over time; results apply only to the model IDs listed above on the run date.",
        "- The query analyzer is a keyword heuristic; its difficulty estimates are not validated against item difficulty labels.",
    ]) + "\n")
    return "\n".join(out)


def tradeoffs(systems: dict) -> str:
    smart, static = systems.get("smart_router"), systems.get("static_router")
    if not smart or not static:
        return "Run both Static Router and SmartRouter to measure the cost of adaptive escalation."
    return "\n".join([
        f"- Escalation versus static routing: SmartRouter mean gen cost {fmt(smart['cost']['mean'], 2)} vs Static Router {fmt(static['cost']['mean'], 2)}; "
        f"median latency {fmt(smart['latency_ms']['median'], 0)} ms vs {fmt(static['latency_ms']['median'], 0)} ms; "
        f"reference-graded quality {fmt(smart['reference_quality']['mean'])} vs {fmt(static['reference_quality']['mean'])}.",
        f"- SmartRouter needed {smart['serving_judge_calls']} serving-time judge calls for {smart['n']} queries; baselines need none in production.",
        f"- Mean escalations per query: {fmt(smart['escalation']['mean_count'], 2)}.",
    ])


def write_report(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(summary), encoding="utf-8")


def main() -> int:
    settings = Settings()
    source = settings.results_dir / "latest.json"
    if not source.exists():
        print(f"No benchmark results at {source}. Run `python -m benchmark.run` first.", file=sys.stderr)
        return 1
    write_report(json.loads(source.read_text(encoding="utf-8")), settings.results_dir / "latest.md")
    print(f"Wrote {settings.results_dir / 'latest.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
