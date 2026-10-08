"""Deterministic verification inside the pipeline: strong evidence escalates, passing evidence is supporting only."""

from app.core.bootstrap import build_runtime
from app.providers.base import LLMResponse
from tests.conftest import FakeProvider, judge_json, make_settings

ARITH = "What is 17 × 23?"  # short arithmetic question: routes to a low tier


class Scripted(FakeProvider):
    """answers: model -> text. verdicts: judge model -> quality for every answer (or a dict keyed by answer text)."""

    def __init__(self, answers, verdicts):
        super().__init__({**answers, "mock-judge": "", "mock-so": ""})
        self.verdicts = verdicts

    async def generate(self, messages, model, *, json_mode=False, **kw):
        if not json_mode:
            return await super().generate(messages, model, **kw)
        self.calls.append((model, True))
        prompt = messages[-1]["content"]
        spec = self.verdicts[model]
        score = next(v for k, v in spec.items() if k in prompt) if isinstance(spec, dict) else spec
        return LLMResponse(text=judge_json(score, verdict="good" if score >= 0.8 else "incorrect", confidence=1.0), model=model, provider="mock")

    def judge_calls(self, model):
        return sum(m == model and j for m, j in self.calls)


async def runtime(tmp_path, answers, verdicts, **overrides):
    provider = Scripted(answers, verdicts)
    settings = make_settings(tmp_path, second_opinion_model="mock-so", **overrides)
    return await build_runtime(settings, providers={"mock": provider}), provider


async def test_math_contradiction_escalates_even_when_judges_accept(tmp_path):
    rt, provider = await runtime(
        tmp_path,
        {"mock-cheap": "Final answer: 381", "mock-mid": "Final answer: 391", "mock-strong": "Final answer: 391"},
        {"mock-judge": 1.0, "mock-so": 1.0},
    )
    r = await rt.pipeline.run(ARITH)
    first = r.attempts[0]
    assert first.verdict == "good" and not first.accepted  # judge said good; the arithmetic says otherwise
    assert "math [math_mismatch]" in first.decision_reason and r.escalated
    assert r.answer == "Final answer: 391" and r.final_status == "accepted"
    assert provider.judge_calls("mock-so") == 0  # strong evidence decides directly


async def test_math_pass_is_supporting_only_and_does_not_override_a_rejection(tmp_path):
    rt, provider = await runtime(
        tmp_path,
        {"mock-cheap": "Final answer: 391", "mock-mid": "Final answer: 391", "mock-strong": "Final answer: 391"},
        {"mock-judge": 0.2, "mock-so": 0.95},
    )
    r = await rt.pipeline.run(ARITH)
    first = r.attempts[0]
    assert [c.evidence for c in first.checks] == ["passed"]
    assert provider.judge_calls("mock-so") == 1  # rejection still goes to a second opinion
    assert "second judge accepted" in first.decision_reason and not r.escalated


async def test_module_self_test_failure_escalates_when_execution_enabled(tmp_path):
    query = "Write a Python function `evaluate(expr: str) -> float`."
    inconsistent = ("```python\ndef evaluate(expr):\n    return 7.0\n\nif __name__ == \"__main__\":\n"
                    "    assert evaluate('2.5 * 4 - 1.5 / .5') == 9.5\n```")
    fixed = "```python\ndef evaluate(expr):\n    return 7.0\n```"
    rt, provider = await runtime(
        tmp_path,
        {"mock-cheap": inconsistent, "mock-mid": inconsistent, "mock-strong": fixed},
        {"mock-judge": 1.0, "mock-so": 1.0},
        verify_execute_code=True,
    )
    r = await rt.pipeline.run(query)
    first = r.attempts[0]
    assert "module_execution [assertion_failed]" in first.decision_reason and not first.accepted
    assert r.answer == fixed


async def test_inconclusive_execution_never_escalates(tmp_path):
    query = "Write a Python function `evaluate(expr: str) -> float`."
    snippet = "```python\ndef evaluate(expr):\n    return 7.0\n\nprint(evaluate(user_input))\n```"
    rt, provider = await runtime(tmp_path, {"mock-cheap": snippet, "mock-mid": snippet, "mock-strong": snippet},
                                 {"mock-judge": 1.0, "mock-so": 1.0}, verify_execute_code=True)
    r = await rt.pipeline.run(query)
    assert not r.escalated and r.final_status == "accepted"
    assert any(c.status == "not_checkable" for c in r.attempts[0].checks)
    assert any("inconclusive" in s.label for s in r.trace)
