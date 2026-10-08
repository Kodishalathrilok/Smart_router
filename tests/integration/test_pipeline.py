"""Pipeline orchestration with scripted providers: escalation, stopping rules, best effort, logging."""

import pytest

from app.core.bootstrap import build_runtime
from app.providers.base import AuthenticationError, LLMResponse, ProviderServerError, RateLimitError
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


async def test_judge_parse_failure_returns_unverified_without_escalating(tmp_path):
    rt, provider = await runtime_with(tmp_path, judgements={"ANS-cheap": "garbage", "ANS-mid": 0.9})
    r = await rt.pipeline.run(EASY)
    assert r.attempts[0].error_type == "JudgeParseError" and r.attempts[0].answer == "ANS-cheap"
    assert provider.generation_calls() == ["mock-cheap"]
    assert (r.final_status, r.answer, r.final_quality_score) == ("best_effort", "ANS-cheap", None)


async def test_judge_outage_does_not_escalate(tmp_path):
    rt, provider = await runtime_with(tmp_path, judgements={"ANS-cheap": RateLimitError("quota", provider="mock")})
    r = await rt.pipeline.run(EASY)
    assert provider.generation_calls() == ["mock-cheap"]
    assert r.final_status == "best_effort" and r.answer == "ANS-cheap"
    assert any("Judge unavailable" in s.label for s in r.trace)


@pytest.mark.parametrize("mode,tiers", [("always_cheap", ["cheap"]), ("always_mid", ["mid"]), ("always_strong", ["strong"]), ("static", ["cheap"])])
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
    await rt.aclose()  # checkpoint the WAL so the database file holds everything
    for path in rt.store.path.parent.iterdir():
        assert secret.encode() not in path.read_bytes()


async def test_mock_mode_end_to_end(tmp_path):
    rt = await build_runtime(make_settings(tmp_path))
    r1 = await rt.pipeline.run(EASY)
    r2 = await rt.pipeline.run(EASY)
    assert r1.mock_mode and r1.answer.startswith("[MOCK")
    assert [a.quality_score for a in r1.attempts] == [a.quality_score for a in r2.attempts]  # deterministic


async def test_fallback_judge_recorded_on_attempt_and_trace(tmp_path):
    from app.providers.base import ProviderServerError

    class PrimaryJudgeDown(FakeProvider):
        async def generate(self, messages, model, **kw):
            if model == "mock-judge":
                self.calls.append((model, True))
                raise ProviderServerError("503", provider="mock")
            return await super().generate(messages, model, **kw)

    provider = PrimaryJudgeDown({**ANSWERS, "mock-backup": ""}, {"ANS-cheap": 0.9})
    rt = await build_runtime(make_settings(tmp_path, judge_fallback_model="mock-backup"), providers={"mock": provider})
    r = await rt.pipeline.run(EASY)
    assert r.final_status == "accepted" and r.attempts[0].judge_model == "mock-backup"
    assert any("judged by fallback mock-backup" in s.label for s in r.trace)

    healthy = FakeProvider({**ANSWERS, "mock-backup": ""}, {"ANS-cheap": 0.9})
    rt = await build_runtime(make_settings(tmp_path / "h", judge_fallback_model="mock-backup"), providers={"mock": healthy})
    r = await rt.pipeline.run(EASY)
    assert r.attempts[0].judge_model == "mock-judge" and not any("fallback" in s.label for s in r.trace)



# ---- Layered verification: failure modes from the first hard-set benchmark ----

class TwoJudges(FakeProvider):
    """Judge output depends on BOTH the answer and which judge model is asked.

    verdicts: answer text -> {judge model id: quality float | raw text | Exception}
    """

    def __init__(self, answers, verdicts):
        super().__init__(answers)
        self.verdicts = verdicts

    async def generate(self, messages, model, *, json_mode=False, **kw):
        if not json_mode:
            return await super().generate(messages, model, **kw)
        self.calls.append((model, True))
        prompt = messages[-1]["content"]
        out = next(v[model] for k, v in self.verdicts.items() if k in prompt)
        if isinstance(out, Exception):
            raise out
        text = judge_json(out, verdict="good" if out >= 0.8 else "incorrect", confidence=0.95) if isinstance(out, float) else out
        return LLMResponse(text=text, model=model, provider="mock", input_tokens=10, output_tokens=20)

    def judge_calls(self):
        return [m for m, is_judge in self.calls if is_judge]


async def two_judge_runtime(tmp_path, verdicts, answers=None, **overrides):
    provider = TwoJudges({**ANSWERS, "mock-backup": "", **(answers or {})}, verdicts)
    settings = make_settings(tmp_path, judge_fallback_model="mock-backup", **overrides)
    return await build_runtime(settings, providers={"mock": provider}), provider


async def test_false_rejection_rescued_by_second_opinion_without_escalating(tmp_path):
    """Correct answer + primary judge wrongly rejects: no stronger model is called."""
    rt, provider = await two_judge_runtime(tmp_path, {"ANS-cheap": {"mock-judge": 0.25, "mock-backup": 0.95}})
    r = await rt.pipeline.run(EASY)
    assert provider.generation_calls() == ["mock-cheap"]  # no unnecessary escalation
    assert r.final_status == "accepted" and not r.escalated
    a = r.attempts[0]
    assert a.quality_score == 0.25 and a.verdict == "incorrect"  # primary judgement is still reported as-is
    assert a.second_opinion.judge_model == "mock-backup" and a.second_opinion.verdict == "good"
    assert a.judge_calls == 2 and r.judge_calls == 2 and "second judge accepted" in a.decision_reason
    assert any("Second opinion from mock-backup" in s.label for s in r.trace)


async def test_true_rejection_confirmed_by_second_judge_escalates(tmp_path):
    rt, provider = await two_judge_runtime(tmp_path, {
        "ANS-cheap": {"mock-judge": 0.2, "mock-backup": 0.3},
        "ANS-mid": {"mock-judge": 0.95, "mock-backup": 0.95},
    })
    r = await rt.pipeline.run(EASY)
    assert provider.generation_calls() == ["mock-cheap", "mock-mid"]
    assert r.final_tier == "mid" and "also rejected" in r.attempts[0].decision_reason
    assert any(s.label.startswith("Escalation triggered: second judge also rejected") for s in r.trace)


async def test_confident_acceptance_skips_second_opinion(tmp_path):
    rt, provider = await two_judge_runtime(tmp_path, {"ANS-cheap": {"mock-judge": 0.95, "mock-backup": 0.1}})
    r = await rt.pipeline.run(EASY)
    assert provider.judge_calls() == ["mock-judge"] and r.final_status == "accepted" and r.judge_calls == 1


async def test_false_acceptance_caught_by_failed_answer_check(tmp_path):
    """Incorrect code + judge wrongly accepts: the failed answer check escalates anyway."""
    query = "Write a Python function `spiral_order(matrix)` returning the clockwise spiral."
    broken = "```python\ndef spiral(matrix):\n    return []\n```"
    good = "```python\ndef spiral_order(matrix):\n    return []\n```"
    rt, provider = await two_judge_runtime(
        tmp_path,
        {broken: {"mock-judge": 0.95, "mock-backup": 0.95}, good: {"mock-judge": 0.95, "mock-backup": 0.95}},
        answers={"mock-cheap": broken, "mock-mid": broken, "mock-strong": good},
    )
    r = await rt.pipeline.run(query)
    first = r.attempts[0]
    assert first.tier == "mid"  # coding query of moderate difficulty
    assert first.verdict == "good" and not first.accepted  # judge accepted, decision did not
    assert "requested_names" in first.decision_reason and r.escalated
    assert r.final_status == "accepted" and r.answer == good


async def test_second_opinion_judge_unavailable_falls_back_to_primary_verdict(tmp_path):
    rt, provider = await two_judge_runtime(tmp_path, {
        "ANS-cheap": {"mock-judge": 0.2, "mock-backup": ProviderServerError("503", provider="mock")},
        "ANS-mid": {"mock-judge": 0.95, "mock-backup": 0.95},
    })
    r = await rt.pipeline.run(EASY)
    first = r.attempts[0]
    assert first.second_opinion.error and "no second opinion available" in first.decision_reason
    assert r.final_tier == "mid"  # primary rejection stands -> escalate, as before this change
    events = [e["event_type"] for e in rt.events.read() if e["query_id"] == r.query_id]
    assert "second_opinion_failed" in events


async def test_both_judges_unavailable_returns_unverified_without_escalating(tmp_path):
    down = ProviderServerError("503", provider="mock")
    rt, provider = await two_judge_runtime(tmp_path, {"ANS-cheap": {"mock-judge": down, "mock-backup": down}})
    r = await rt.pipeline.run(EASY)
    assert provider.generation_calls() == ["mock-cheap"]
    assert r.final_status == "best_effort" and r.answer == "ANS-cheap" and r.final_quality_score is None


async def test_no_unnecessary_strong_escalation_from_mid(tmp_path):
    """Hard query starting at mid: a false rejection must not reach the strong model."""
    medium = "Explain why the sky is blue and compare it to sunsets."  # routes to mid
    rt, provider = await two_judge_runtime(tmp_path, {"ANS-mid": {"mock-judge": 0.1, "mock-backup": 1.0}})
    r = await rt.pipeline.run(medium)
    assert r.initial_tier == "mid" and "mock-strong" not in provider.generation_calls()
    assert r.final_status == "accepted"


async def test_baseline_modes_never_request_second_opinions(tmp_path):
    rt, provider = await two_judge_runtime(tmp_path, {"ANS-cheap": {"mock-judge": 0.2, "mock-backup": 1.0}})
    r = await rt.pipeline.run(EASY, mode="static")
    assert provider.judge_calls() == ["mock-judge"] and r.attempts[0].second_opinion is None and r.judge_calls == 1


async def test_second_opinion_can_be_disabled(tmp_path):
    rt, provider = await two_judge_runtime(
        tmp_path, {"ANS-cheap": {"mock-judge": 0.2, "mock-backup": 1.0}, "ANS-mid": {"mock-judge": 0.95, "mock-backup": 0.95}}, second_opinion=False
    )
    r = await rt.pipeline.run(EASY)
    assert provider.judge_calls() == ["mock-judge", "mock-judge"] and r.final_tier == "mid"
