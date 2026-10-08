"""Second-opinion path inside the pipeline: decisions, fallback behavior, recorded trigger/latency/cost."""

import asyncio

import pytest

from app.core.bootstrap import build_runtime
from app.core.config import ConfigError
from app.models.registry import ModelRegistry
from app.providers.base import EmptyResponseError, LLMResponse
from tests.conftest import FakeProvider, judge_json, make_settings

EASY = "Who wrote Hamlet?"  # routes to cheap
ANSWERS = {"mock-cheap": "ANS-cheap", "mock-mid": "ANS-mid", "mock-strong": "ANS-strong", "mock-judge": "", "mock-backup": "", "mock-so": ""}


class JudgeByModel(FakeProvider):
    """verdicts: answer text -> {judge model: quality float | seconds-to-sleep (negative float) | Exception}."""

    def __init__(self, verdicts, answers=None):
        super().__init__({**ANSWERS, **(answers or {})})
        self.verdicts = verdicts

    async def generate(self, messages, model, *, json_mode=False, **kw):
        if not json_mode:
            return await super().generate(messages, model, **kw)
        self.calls.append((model, True))
        prompt = messages[-1]["content"]
        out = next(v[model] for k, v in self.verdicts.items() if k in prompt)
        if isinstance(out, Exception):
            raise out
        if out < 0:  # simulate a very slow judge
            await asyncio.sleep(-out)
            out = 1.0
        return LLMResponse(text=judge_json(out, verdict="good" if out >= 0.8 else "incorrect", confidence=1.0), model=model, provider="mock")

    def judge_calls(self, model):
        return sum(m == model and j for m, j in self.calls)


async def runtime(tmp_path, verdicts, answers=None, **overrides):
    provider = JudgeByModel(verdicts, answers)
    settings = make_settings(tmp_path, judge_fallback_model="mock-backup", second_opinion_model="mock-so", **overrides)
    return await build_runtime(settings, providers={"mock": provider}), provider


async def test_primary_reject_second_accept_is_accepted_and_recorded(tmp_path):
    rt, provider = await runtime(tmp_path, {"ANS-cheap": {"mock-judge": 0.25, "mock-so": 0.95, "mock-backup": 0.1}})
    r = await rt.pipeline.run(EASY)
    a = r.attempts[0]
    assert r.final_status == "accepted" and not r.escalated and provider.generation_calls() == ["mock-cheap"]
    so = a.second_opinion
    assert so.judge_model == "mock-so" and so.trigger == "primary_rejection" and so.calls == 1 and so.error is None
    assert provider.judge_calls("mock-backup") == 0  # dedicated second-opinion model, not the fallback judge
    assert a.judge_latency_ms >= so.latency_ms and r.judge_calls == 2


async def test_primary_reject_second_reject_escalates(tmp_path):
    rt, provider = await runtime(tmp_path, {
        "ANS-cheap": {"mock-judge": 0.25, "mock-so": 0.2},
        "ANS-mid": {"mock-judge": 0.95, "mock-so": 0.95},
    })
    r = await rt.pipeline.run(EASY)
    assert r.final_tier == "mid" and "also rejected" in r.attempts[0].decision_reason


async def test_second_opinion_unavailable_keeps_primary_verdict(tmp_path):
    rt, provider = await runtime(tmp_path, {
        "ANS-cheap": {"mock-judge": 0.25, "mock-so": EmptyResponseError("empty", provider="mock")},
        "ANS-mid": {"mock-judge": 0.95, "mock-so": 0.95},
    })
    r = await rt.pipeline.run(EASY)
    so = r.attempts[0].second_opinion
    assert so.error and "EmptyResponseError" in so.error and so.calls == 1
    assert r.final_tier == "mid" and "no second opinion available" in r.attempts[0].decision_reason


async def test_second_opinion_timeout_is_bounded_and_recorded(tmp_path):
    rt, provider = await runtime(
        tmp_path,
        {"ANS-cheap": {"mock-judge": 0.25, "mock-so": -5.0}, "ANS-mid": {"mock-judge": 0.95, "mock-so": 0.95}},
        second_opinion_timeout_s=0.3,
    )
    r = await rt.pipeline.run(EASY)
    so = r.attempts[0].second_opinion
    assert so.error and "budget" in so.error
    assert 250 <= so.latency_ms < 1500  # bounded by the 0.3 s budget, not the 5 s judge
    assert r.total_latency_ms < 3000 and r.final_tier == "mid"


async def test_deterministic_check_failure_bypasses_second_opinion(tmp_path):
    query = "Write a Python function `spiral_order(matrix)`."
    broken = "```python\ndef spiral(matrix):\n    return []\n```"
    good = "```python\ndef spiral_order(matrix):\n    return []\n```"
    rt, provider = await runtime(
        tmp_path,
        {broken: {"mock-judge": 0.2, "mock-so": 0.95}, good: {"mock-judge": 0.95, "mock-so": 0.95}},
        answers={"mock-cheap": broken, "mock-mid": broken, "mock-strong": good},
    )
    r = await rt.pipeline.run(query)
    assert provider.judge_calls("mock-so") == 0 and r.attempts[0].second_opinion is None
    assert "answer check failed" in r.attempts[0].decision_reason and r.answer == good


def test_second_opinion_role_is_optional_and_validated(tmp_path):
    assert ModelRegistry.from_settings(make_settings(tmp_path)).second_opinion is None
    reg = ModelRegistry.from_settings(make_settings(tmp_path, second_opinion_model="mock-so"))
    assert reg.second_opinion.id == "mock-so" and reg.second_opinion.role == "second_opinion"
    with pytest.raises(ConfigError, match="SECOND_OPINION_MODEL is not set"):
        ModelRegistry.from_settings(make_settings(tmp_path, mock_mode=False, cheap_model="a", mid_model="b", strong_model="c",
                                                  judge_model="j", second_opinion_provider="groq"))


async def test_second_opinion_model_checked_against_provider_list(tmp_path):
    reg = ModelRegistry.from_settings(make_settings(tmp_path, second_opinion_model="mock-so"))
    provider = FakeProvider({s.id: "" for s in reg.all() if s.id != "mock-so"})
    with pytest.raises(ConfigError, match="SECOND_OPINION_MODEL='mock-so' is not available"):
        await reg.validate({"mock": provider})
