import json

from benchmark.report import render
from benchmark.run import parse_args, run
from tests.conftest import make_settings


async def test_mock_benchmark_end_to_end(tmp_path):
    settings = make_settings(tmp_path)
    summary = await run(parse_args(["--per-category", "2"], settings), settings)

    assert summary["mock_mode"] and summary["dataset"]["n"] == 10
    assert list(summary["systems"]) == ["always_weak", "always_mid", "always_strong", "static_router", "smart_router"]
    strong = summary["systems"]["always_strong"]
    assert strong["utilization"]["calls"] == {"cheap": 0, "mid": 0, "strong": 10}
    assert summary["systems"]["always_weak"]["utilization"]["calls"]["strong"] == 0
    assert summary["comparisons"]["always_strong"]["cost_reduction_pct_vs_strong"] == 0.0
    assert summary["systems"]["smart_router"]["serving_judge_calls"] > 0
    assert summary["systems"]["static_router"]["serving_judge_calls"] == 0
    assert summary["systems"]["always_mid"]["utilization"]["calls"] == {"cheap": 0, "mid": 10, "strong": 0}
    assert summary["claim"]["verdict"] == "not_tested"  # mock mode has no exact checks: never a claim from synthetic data
    assert summary["analyzer"]["n"] == 10 and "Router validation" in (settings.results_dir / "latest.md").read_text(encoding="utf-8")

    latest = json.loads((settings.results_dir / "latest.json").read_text(encoding="utf-8"))
    report = (settings.results_dir / "latest.md").read_text(encoding="utf-8")
    assert latest["run_id"] == summary["run_id"]
    assert "MOCK RUN" in report
    for section in ("Executive Summary", "Dataset", "Methodology", "Baselines", "Metrics", "Results",
                    "Per-category Results", "Failure Analysis", "Tradeoffs", "Limitations"):
        assert f"## {section}" in report
    assert report == render(latest)
    assert "### Judge usage" in report and "`mock-judge`" in report
    assert "Accuracy by answering tier" in report  # mock skips exact checks -> explicit "No exact checks" note
    assert summary["systems"]["smart_router"]["judge_usage"]["mock-judge"] > 0
    raw = settings.benchmark_dir / summary["run_id"]
    assert (raw / "smart_router.jsonl").exists() and (raw / "smartrouter.db").exists()
