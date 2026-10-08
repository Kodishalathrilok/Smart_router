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


VERDICT_LABEL = {
    "supported": "SUPPORTED",
    "not_supported": "NOT SUPPORTED",
    "not_supported_cheap_dominates": "NOT SUPPORTED (Always Cheap dominates)",
    "inconclusive": "INCONCLUSIVE",
    "not_tested": "NOT TESTED",
}


def executive_summary(s: dict) -> str:
    """Generated from the measured numbers. The headline compares against Always Cheap as well as Always
    Strong: "X% cheaper than strong" alone hides the case where the cheap model is just as accurate."""
    systems = s["systems"]
    claim = s.get("claim") or {"verdict": "not_tested", "explanation": "No claim evaluation in this summary."}
    lines = [
        "Claim under test: *SmartRouter reduces cost without reducing answer quality.*",
        "",
        f"**Verdict: {VERDICT_LABEL.get(claim['verdict'], claim['verdict'])}.** {claim['explanation']}",
        "",
    ]
    for name in systems:
        v = systems[name]
        st = v.get("strict") or {}
        ci = st.get("ci95")
        lines.append(
            f"- {v['label']}: strict accuracy {pct(st.get('accuracy'))}"
            + (f" (95% CI {pct(ci[0])}-{pct(ci[1])}, n={st['checked']})" if ci else "")
            + f"; cost incl. judge {fmt(v['cost_incl_judge']['mean'], 2)}; median latency {fmt(v['latency_ms']['median'], 0)} ms."
        )
    lines += [
        "",
        "Quality here means exact-check accuracy (numeric answers and unit tests). LLM-judge scores are secondary: the same "
        "judge family drives SmartRouter's escalations, so judging the systems with it would be circular.",
    ]
    return "\n".join(lines)


def paired_section(s: dict) -> str:
    pairs = s.get("paired") or {}
    if not pairs:
        return ""
    names = {"smart_vs_cheap": "SmartRouter vs Always Cheap", "smart_vs_mid": "SmartRouter vs Always Mid",
             "smart_vs_strong": "SmartRouter vs Always Strong", "static_vs_cheap": "Static Router vs Always Cheap",
             "strong_vs_cheap": "Always Strong vs Always Cheap"}
    rows = [[names.get(k, k), str(p["items"]), str(p["wins"]), str(p["losses"]), str(p["both_correct"]), str(p["both_wrong"]),
             fmt(p["sign_test_p"], 3)] for k, p in pairs.items()]
    return ("### Paired comparison (same items, strict exact checks)\n\n"
            "*Wins*: the first system was right and the second wrong. The sign test asks whether wins and losses differ by more "
            "than chance; p >= 0.05 means no detectable difference on this many items.\n\n"
            + table(["Comparison", "Items", "Wins", "Losses", "Both right", "Both wrong", "Sign-test p"], rows) + "\n")


def analyzer_section(s: dict) -> str:
    a = s.get("analyzer")
    if not a:
        return ""
    rows = [[label, *(str(t.get(tier, 0)) for tier in TIERS)] for label, t in sorted(a["label_to_tier"].items())]
    return ("### Router validation (no model calls)\n\n"
            f"Heuristic category matches the dataset label on {pct(a['category_accuracy'])} of {a['n']} items. "
            "Initial tier by the dataset's difficulty label (a useful difficulty estimate sends *hard* items up more often):\n\n"
            + table(["Label", *[t.capitalize() for t in TIERS]], rows) + "\n")


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

    gen_fail = {v["label"]: v["generation_failures"] for v in systems.values() if v.get("generation_failures")}
    if gen_fail:
        detail = ", ".join(f"{label}: {n}" for label, n in gen_fail.items())
        out.append(f"> **PROVIDER FAILURES.** Generation calls failed after retries ({detail}). A failed query has no answer and counts as wrong, "
                   "so a rate-limited or timed-out tier looks less accurate than it is. Check `provider_error` events before trusting the comparison.\n")
    if any(v.get("cost_method") == "estimated_lower_bound" for v in systems.values()):
        out.append("> **Cost is estimated** for these older records: they did not store per-role billed calls, so SmartRouter's judge cost "
                   "is a lower bound (parse retries and fallback-judge calls are not counted).\n")
    if s.get("rescored_by"):
        out.append(f"> Re-scored from the raw per-item results by `{s['rescored_by']}` with the current methodology (no new model calls).\n")
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
        "- **Strict acc.** (primary): correct / all deterministically checkable items (numeric answers, unit tests). An answer with no readable final number counts as wrong (*Unparseable* column). 95% Wilson interval.",
        "- **Ref. judge q.** / **In-loop judge q.**: LLM-judge scores with / without the reference answer (secondary; same judge family as the escalation loop).",
        "- **Calls/query**: generation calls plus serving-time judge calls (what production would pay for), excluding transport retries.",
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
        st = v.get("strict") or {}
        ci = st.get("ci95")
        rows.append([
            v["label"], pct(st.get("accuracy")), f"{pct(ci[0])}-{pct(ci[1])}" if ci else "n/a", str(st.get("unparseable", "n/a")),
            fmt(v["cost"]["mean"], 2), fmt(v["cost_incl_judge"]["mean"], 2),
            f"{c['cost_incl_judge_vs_cheap_x']:.2f}x" if c.get("cost_incl_judge_vs_cheap_x") is not None else "n/a",
            change(c.get("cost_incl_judge_reduction_pct_vs_strong")), fmt(v.get("serving_calls_mean"), 2),
            fmt(v["latency_ms"]["median"], 0), fmt(v["latency_ms"]["p95"], 0),
            pct(v["escalation"]["rate"]), pct(v["failure_rate"]), fmt(v["reference_quality"]["mean"]), fmt(v["quality"]["mean"]),
        ])
    out.append(table(["System", "Strict acc.", "95% CI", "Unparseable", "Gen cost", "Cost incl. judge", "vs cheap", "vs strong (incl. judge)",
                      "Calls/query", "p50 ms", "p95 ms", "Escalation", "Failed", "Ref. judge q.", "In-loop judge q."], rows) + "\n")
    out.append(paired_section(s))
    out.append(analyzer_section(s))

    out.append("### Model utilization (share of generation calls)\n")
    out.append(table(["System"] + [t.capitalize() for t in TIERS],
                     [[systems[n]["label"]] + [pct(systems[n]["utilization"]["calls_pct"][t]) for t in TIERS] for n in names]) + "\n")

    out.append(tier_accuracy_section(systems))
    out.append(judge_usage_section(systems))
    out.append(verification_section(systems))

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
        f"- {ds['n']} items is small: the confidence intervals above are wide, and a sign test on a handful of discordant items cannot detect small differences.",
        "- Normalized cost uses configured tier weights, not provider billing. It measures relative model usage, not money.",
        "- Latency depends on provider load, rate limits and retries at run time and is not reproducible across runs.",
        "- Provider model versions change over time; results apply only to the model IDs listed above on the run date.",
        "- The query analyzer is a keyword heuristic; see *Router validation* for how its estimates line up with the dataset labels.",
        "- Re-running the same items gives repeated measurements of the same questions, not new independent samples.",
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


def tier_accuracy_section(systems: dict) -> str:
    """Exact-check accuracy by the tier that answered, pooled over all systems.

    If a higher tier is not more accurate than a lower one, the tier lineup itself is wrong,
    and routing harder queries "up" makes answers worse.
    """
    pooled: dict[str, list[int]] = {}
    for v in systems.values():
        for tier, st in v.get("tier_accuracy", {}).items():
            acc = pooled.setdefault(tier, [0, 0])
            acc[0] += st["correct"]
            acc[1] += st["checked"]
    if not pooled:
        return "### Accuracy by answering tier\n\nNo exact checks in this run.\n"
    rows = [[t.capitalize(), str(pooled[t][0]), str(pooled[t][1]), pct(pooled[t][0] / pooled[t][1])] for t in TIERS if t in pooled]
    accs = [pooled[t][0] / pooled[t][1] for t in TIERS if t in pooled]
    note = "" if accs == sorted(accs) else ("\n> **Tier order check failed:** a higher tier was less accurate than a lower one on exact checks. "
                                          "Routing harder queries upward may be making answers worse; review the model lineup.\n")
    return ("### Accuracy by answering tier (exact checks, all systems pooled)\n\n"
            + table(["Tier", "Correct", "Checked", "Accuracy"], rows) + "\n" + note)


def judge_usage_section(systems: dict) -> str:
    rows = []
    for v in systems.values():
        usage = v.get("judge_usage", {})
        used = ", ".join(f"`{m}` {n}" for m, n in sorted(usage.items(), key=lambda kv: -kv[1])) or "none"
        rows.append([v["label"], used, str(v.get("judge_failures", 0))])
    return ("### Judge usage\n\nWhich judge graded each attempt. More than one judge means scores mix evaluators.\n\n"
            + table(["System", "Graded by", "Judge failures"], rows) + "\n")


def verification_section(systems: dict) -> str:
    """Totals, escalation necessity and judge accuracy, all measured against exact checks."""
    if not any("escalations" in v for v in systems.values()):
        return ""
    totals = [
        [v["label"], fmt(v["cost"]["total"], 1), str(v.get("strong_calls", "n/a")), str(v["judge_calls"]), str(v.get("judge_failures", 0)),
         "{total} ({necessary} / {unnecessary} / {unknown})".format(**v["escalations"])]
        for v in systems.values()
    ]
    accuracy = []
    for v in systems.values():
        for label, c in (("primary judge", v["judge_accuracy"]["primary_judge"]), ("final decision", v["judge_accuracy"]["decision"])):
            if label == "final decision" and v["mode"] != "smart":
                continue
            accuracy.append([v["label"], label, str(c["checked"]), f"{c['agree']}/{c['checked']}" if c["checked"] else "n/a",
                             str(c["false_accept"]), str(c["false_reject"])])
    out = [
        "### Totals\n",
        table(["System", "Total gen cost", "Strong calls", "Judge calls", "Judge failures", "Escalations (necessary / unnecessary / unknown)"], totals),
        "\nAn escalation is *unnecessary* when the answer it replaced already passed the exact check.\n",
        "### Judge accuracy vs exact checks (every judged attempt)\n",
        table(["System", "Scored", "Checked", "Agree", "False accepts", "False rejects"], accuracy),
        "\n*Primary judge* is the first judge's verdict alone; *final decision* adds answer checks and second opinions.\n",
    ]
    smart = systems.get("smart_router")
    so = (smart or {}).get("second_opinions")
    if so and so["requested"]:
        out.append(
            f"Second opinions (SmartRouter): {so['requested']} requested, {so['succeeded']} succeeded, {so['failed']} failed "
            f"(success rate {pct(so['success_rate'])}); latency median {fmt(so['latency_ms_median'], 0)} ms, "
            f"max {fmt(so['latency_ms_max'], 0)} ms, including failures.\n"
        )
    if smart and smart.get("items"):
        rows = [
            [i["item_id"], f"`{i['initial_model']}`", i["judge_result"] or "n/a", fmt(i["judge_confidence"], 2), i["second_opinion"] or "",
             _yes_no(i["initial_exact"]), "yes" if i["escalated"] else "no", i["escalation_reason"] or "", f"`{i['final_model']}`" if i["final_model"] else "none",
             _yes_no(i["final_correct"]), fmt(i["cost"], 0), str(i["judge_calls"])]
            for i in smart["items"]
        ]
        out += ["### Per-item trace (SmartRouter)\n",
                table(["Item", "Initial model", "Judge result", "Judge conf.", "Second opinion", "Exact (initial)", "Escalated",
                       "Escalation reason", "Final model", "Final correct", "Cost", "Judge calls"], rows), ""]
    return "\n".join(out) + "\n"


def _yes_no(value: bool | None) -> str:
    return "n/a" if value is None else "yes" if value else "no"


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
