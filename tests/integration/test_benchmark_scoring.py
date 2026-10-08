"""End-to-end check that the real-mode benchmark scores honestly (no network: a scripted provider stands in).

Scenario: the cheap model answers every hard_v2 item correctly, mid/strong answer wrongly, and the judge rejects
everything from cheap. SmartRouter then escalates and loses. The benchmark must report exactly that.
"""

import functools
import json

import benchmark.run as bench
from app.core.bootstrap import build_runtime
from app.providers.base import LLMProvider, LLMResponse
from benchmark.dataset import DATASETS_DIR, load_dataset
from tests.conftest import judge_json, make_settings

ITEMS = {i.question: i for i in load_dataset(DATASETS_DIR / "hard_v2.jsonl")}
WRONG = "I think the answer is: Final answer: 123456789"


class Scripted(LLMProvider):
    name = "mock"

    async def generate(self, messages, model, *, temperature=0.2, max_tokens=1024, json_mode=False):
        if json_mode:  # judge: reject answers containing the reference (i.e. the correct cheap answers), accept the rest
            prompt = messages[-1]["content"]
            correct = any(i.reference_answer in prompt for i in ITEMS.values())
            return LLMResponse(text=judge_json(0.2, verdict="incorrect", confidence=0.95) if correct else judge_json(0.95, confidence=0.95),
                               model=model, provider="mock")
        item = ITEMS[messages[-1]["content"]]
        return LLMResponse(text=item.reference_answer if model == "m-cheap" else WRONG, model=model, provider="mock")

    async def list_models(self):
        return ["m-cheap", "m-mid", "m-strong", "m-judge"]


async def test_real_mode_benchmark_reports_a_losing_router_as_losing(tmp_path, monkeypatch):
    settings = make_settings(
        tmp_path, mock_mode=False, validate_models=False, second_opinion=False,
        cheap_provider="mock", cheap_model="m-cheap", mid_provider="mock", mid_model="m-mid",
        strong_provider="mock", strong_model="m-strong", judge_provider="mock", judge_model="m-judge",
    )
    monkeypatch.setattr(bench, "build_runtime", functools.partial(build_runtime, providers={"mock": Scripted()}))
    summary = await bench.run(bench.parse_args(["--dataset", "hard_v2", "--exec-code", "--no-reference-eval"], settings), settings)

    s = summary["systems"]
    n = summary["dataset"]["n"]
    assert set(s) == {"always_weak", "always_mid", "always_strong", "static_router", "smart_router"}
    assert s["always_weak"]["strict"]["correct"] == n  # cheap is always right, code items included
    assert s["always_mid"]["strict"]["correct"] == 0 and s["always_strong"]["strict"]["correct"] == 0
    smart = s["smart_router"]
    # The judge rejects every correct cheap answer, so every cheap-routed item escalates to a wrong answer.
    assert smart["strict"]["accuracy"] < s["always_weak"]["strict"]["accuracy"]
    assert smart["cost_method"] == "exact" and smart["escalation"]["rate"] > 0
    esc = smart["escalations"]
    # Escalations away from a (correct) cheap answer are unnecessary; mid -> strong on wrong code answers are necessary.
    assert esc["unnecessary"] >= 1 and esc["necessary"] >= 1 and esc["unknown"] == 0
    assert esc["necessary"] + esc["unnecessary"] == esc["total"]
    # Costs: baselines pay no judge; SmartRouter pays generation + every judge response.
    assert s["always_weak"]["cost_incl_judge"]["mean"] == 1.0 and s["always_strong"]["cost_incl_judge"]["mean"] == 10.0
    assert smart["serving_judge_calls"] == sum(r["judge_calls"] for r in smart["items"])  # no parse retries in this script
    assert summary["claim"]["verdict"].startswith("not_supported")
    assert summary["paired"]["smart_vs_cheap"]["losses"] > 0 and summary["paired"]["smart_vs_cheap"]["wins"] == 0
    report = (settings.results_dir / "latest.md").read_text(encoding="utf-8")
    assert "NOT SUPPORTED" in report and "Always Mid" in report
    raw = json.loads((settings.results_dir / "latest.json").read_text(encoding="utf-8"))
    assert raw["config"]["exec_code"] is True
