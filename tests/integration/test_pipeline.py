"""Pipeline orchestration with scripted providers: escalation, stopping rules, best effort, logging."""

import pytest

from app.core.bootstrap import build_runtime
from app.providers.base import AuthenticationError, RateLimitError
from tests.conftest import FakeProvider, judge_json, make_settings

EASY = "Who wrote Hamlet?"  # heuristic difficulty < 0.35 → cheap
HARD = "Prove step by step why the halting problem is undecidable, analyze the implications and justify each step rigorously."

ANSWERS = {"mock-cheap": "ANS-cheap", "mock-mid": "ANS-mid", "mock-strong": "ANS-strong", "mock-judge": ""}


async def runtime_with(tmp_path, answers=None, judgements=None, **settings_overrides):
    provider = FakeProvider({**ANSWERS, **(answers or {})}, judgements or {})
    runtime = await build_runtime(make_settings(tmp_path, **settings_overrides), providers={"mock": provider})
    return runtime, provider


async def test_accepts_first_answer_when_quality_sufficient(tmp_path):
    rt, provider = await runtime_with(tmp_path, judgements={"ANS-cheap": 0.9})
    r = await rt.pipeline.run(EASY)
    assert (r.final_status, r.initial_tier, r.final_tier, r.escalated) == ("accepted", "cheap", "cheap", False)
    assert r.answer == "ANS-cheap" and r.final_quality_score == 0.9
    assert provider.generation_calls() == ["mock-cheap"]
    assert r.normalized_cost == 1.0 and r.judge_calls == 1


async def test_escalates_cheap_to_mid_to_strong(tmp_path):
    rt, provider = await runtime_with(tmp_path, judgements={"ANS-cheap": 0.5, "ANS-mid": 0.6, "ANS-strong": 0.95})
    r = await rt.pipeline.run(EASY)
    assert provider.generation_calls() == ["mock-cheap", "mock-mid", "mock-strong"]
    assert (r.final_status, r.final_tier, r.escalation_count) == ("accepted", "strong", 2)
    assert r.normalized_cost == 14.0
    assert [a.accepted for a in r.attempts] == [False, False, True]
    assert r.attempts[0].dimensions["correctness"] == 0.5


async def test_max_escalations_enforced(tmp_path):
    rt, provider = await runtime_with(tmp_path, judgements={"ANS-cheap": 0.5, "ANS-mid": 0.6}, max_escalations=1)
    r = await rt.pipeline.run(EASY)
    assert provider.generation_calls() == ["mock-cheap", "mock-mid"]
    assert r.final_status == "best_effort" and r.final_tier == "mid"
    assert "Max escalations reached (1)" in [s.label for s in r.trace]


async def test_strong_start_stops_without_escalation(tmp_path):
    rt, provider = await runtime_with(tmp_path, judgements={"ANS-strong": 0.4})
    r = await rt.pipeline.run(HARD)
    assert provider.generation_calls() == ["mock-strong"]
    assert r.final_status == "best_effort" and r.answer == "ANS-strong"


async def test_best_effort_returns_highest_quality_candidate(tmp_path):
    rt, _ = await runtime_with(tmp_path, judgements={"ANS-cheap": 0.7, "ANS-mid": 0.5, "ANS-strong": 0.6})
    r = await rt.pipeline.run(EASY)
    assert r.final_status == "best_effort" and r.answer == "ANS-cheap" and r.final_quality_score == 0.7


async def test_reject_verdict_blocks_acceptance(tmp_path):
    rt, _ = await runtime_with(tmp_path, judgements={"ANS-cheap": judge_json(0.95, ["incorrect"], verdict="incorrect"), "ANS-mid": 0.9})
    r = await rt.pipeline.run(EASY)
    assert r.final_tier == "mid"


async def test_provider_error_escalates_and_is_recorded(tmp_path):
    rt, _ = await runtime_with(tmp_path, answers={"mock-cheap": RateLimitError("429", provider="mock")}, judgements={"ANS-mid": 0.9})
    r = await rt.pipeline.run(EASY)
    assert r.attempts[0].error_type == "RateLimitError" and r.attempts[0].normalized_cost == 0
    assert r.final_tier == "mid" and r.final_status == "accepted"


async def test_all_fail_is_reported_as_failed(tmp_path):
    err = AuthenticationError("bad key", provider="mock")
    rt, _ = await runtime_with(tmp_path, answers={"mock-cheap": err, "mock-mid": err, "mock-strong": err})
    r = await rt.pipeline.run(EASY)
    assert r.final_status == "failed" and r.answer is None and r.final_quality_score is None


async def test_judge_parse_failure_escalates_not_accepted(tmp_path):
    rt, _ = await runtime_with(tmp_path, judgements={"ANS-cheap": "garbage", "ANS-mid": 0.9})
    r = await rt.pipeline.run(EASY)
    assert r.attempts[0].error_type == "JudgeParseError" and r.attempts[0].answer == "ANS-cheap"
    assert r.final_tier == "mid"


@pytest.mark.parametrize("mode,tiers", [("always_cheap", ["cheap"]), ("always_strong", ["strong"]), ("static", ["cheap"])])
async def test_baseline_modes_never_escalate(tmp_path, mode, tiers):
    rt, _ = await runtime_with(tmp_path, judgements={"ANS-cheap": 0.1, "ANS-strong": 0.1})
    r = await rt.pipeline.run(EASY, mode=mode)
    assert [a.tier for a in r.attempts] == tiers and r.final_status == "best_effort"


async def test_no_duplicate_model_calls_when_tiers_share_a_model(tmp_path):
    rt, provider = await runtime_with(tmp_path, mid_model="mock-cheap", judgements={"ANS-cheap": 0.5, "ANS-strong": 0.9})
    await rt.pipeline.run(EASY)
    assert provider.generation_calls() == ["mock-cheap", "mock-strong"]


async def test_events_and_record_persisted(tmp_path):
    rt, _ = await runtime_with(tmp_path, judgements={"ANS-cheap": 0.5, "ANS-mid": 0.9})
    r = await rt.pipeline.run(EASY)
    events = [e for e in rt.events.read() if e["query_id"] == r.query_id]
    types = [e["event_type"] for e in events]
    for required in ("query_received", "query_analyzed", "route_selected", "generation_started", "generation_completed",
                     "judge_completed", "escalation_triggered", "pipeline_completed"):
        assert required in types
    route = next(e for e in events if e["event_type"] == "route_selected")
    assert {"category", "difficulty", "selected_model", "selected_tier", "reason"} <= route["payload"].keys()
    assert rt.queries.get(r.query_id).final_tier == "mid"


async def test_api_keys_never_logged(tmp_path):
    secret = "sk-SUPERSECRET"
    rt, _ = await runtime_with(tmp_path, groq_api_key=secret,
                               answers={"mock-cheap": AuthenticationError(f"rejected {secret}", provider="mock")},
                               judgements={"ANS-mid": 0.9})
    await rt.pipeline.run(EASY)
    for path in (rt.events.path, rt.queries.path):
        assert secret not in path.read_text()


async def test_mock_mode_end_to_end(tmp_path):
    rt = await build_runtime(make_settings(tmp_path))
    r1 = await rt.pipeline.run(EASY)
    r2 = await rt.pipeline.run(EASY)
    assert r1.mock_mode and r1.answer.startswith("[MOCK")
    assert [a.quality_score for a in r1.attempts] == [a.quality_score for a in r2.attempts]  # deterministic
