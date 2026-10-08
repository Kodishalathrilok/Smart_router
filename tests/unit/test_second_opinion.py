"""Bounded second-opinion judging: budget, retries, empty/malformed output, model selection."""

import asyncio
import time

import pytest

from app.core.executor import Executor
from app.judge.judge import JudgeParseError, QualityJudge, SecondOpinionPolicy
from app.judge.prompts import STRICT_JSON_INSTRUCTION
from app.models.schemas import ModelSpec
from app.providers.base import EmptyResponseError, LLMProvider, LLMResponse, ProviderServerError, ProviderTimeoutError, RateLimitError
from tests.conftest import judge_json

GOOD = judge_json(0.95, confidence=0.9)


def spec(model_id: str, role: str = "judge") -> ModelSpec:
    return ModelSpec(id=model_id, provider="mock", role=role, capability_score=0.9, normalized_cost=3, expected_latency="low")


PRIMARY, FALLBACK, DEDICATED = spec("primary"), spec("fallback", "judge_fallback"), spec("dedicated", "second_opinion")


class ScriptedJudge(LLMProvider):
    """Per-model queue of outputs: str (response text), Exception (raised) or float (sleep seconds, then GOOD)."""

    name = "mock"

    def __init__(self, script: dict[str, list]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[str, list[dict]]] = []

    async def generate(self, messages, model, *, temperature=0.2, max_tokens=1024, json_mode=False) -> LLMResponse:
        self.calls.append((model, messages))
        queue = self.script[model]
        out = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(out, Exception):
            raise out
        if isinstance(out, float):
            await asyncio.sleep(out)
            out = GOOD
        if not out:
            raise EmptyResponseError("empty completion", provider="mock")
        return LLMResponse(text=out, model=model, provider="mock")

    async def list_models(self) -> list[str]:
        return list(self.script)

    def calls_to(self, model: str) -> int:
        return sum(m == model for m, _ in self.calls)


def make(script, *, dedicated=True, timeout_s=5.0, max_retries=1):
    provider = ScriptedJudge(script)
    executor = Executor({"mock": provider}, max_retries=3, base_delay_s=0, max_delay_s=0)
    policy = SecondOpinionPolicy(specs=(DEDICATED,) if dedicated else (), timeout_s=timeout_s, max_retries=max_retries)
    return QualityJudge(executor, [PRIMARY, FALLBACK], max_tokens=256, second_opinion=policy), provider


async def ask(judge):
    return await judge.second_opinion("q", "a", "math", first_judge="primary")


async def test_successful_second_opinion_uses_dedicated_model_and_strict_prompt():
    judge, provider = make({"dedicated": [GOOD], "fallback": [GOOD], "primary": [GOOD]})
    result = await ask(judge)
    assert result.judge_model == "dedicated" and result.verdict == "good"
    assert provider.calls_to("dedicated") == 1 and provider.calls_to("fallback") == 0
    assert STRICT_JSON_INSTRUCTION in provider.calls[0][1][0]["content"]


async def test_primary_judge_prompt_is_unchanged():
    judge, provider = make({"primary": [GOOD], "fallback": [GOOD], "dedicated": [GOOD]})
    await judge.evaluate("q", "a", "math")
    assert STRICT_JSON_INSTRUCTION not in provider.calls[0][1][0]["content"]


async def test_timeout_budget_bounds_latency():
    judge, _ = make({"dedicated": [5.0]}, timeout_s=0.2)
    start = time.perf_counter()
    with pytest.raises(ProviderTimeoutError, match="0.2s budget"):
        await ask(judge)
    assert time.perf_counter() - start < 1.0  # the slow call was cancelled, not awaited


async def test_budget_also_bounds_retries():
    # each call is slow enough that a retry cannot fit: total time stays within the budget
    judge, provider = make({"dedicated": [ProviderServerError("503", provider="mock"), 5.0]}, timeout_s=0.3, max_retries=1)
    start = time.perf_counter()
    with pytest.raises(ProviderTimeoutError):
        await ask(judge)
    assert time.perf_counter() - start < 1.0 and provider.calls_to("dedicated") == 2


async def test_empty_response_fails_immediately_without_retry():
    judge, provider = make({"dedicated": [""]})
    with pytest.raises(EmptyResponseError):
        await ask(judge)
    assert provider.calls_to("dedicated") == 1


async def test_malformed_response_fails_immediately_without_parse_retry():
    judge, provider = make({"dedicated": ["I think it is fine."]})
    with pytest.raises(JudgeParseError):
        await ask(judge)
    assert provider.calls_to("dedicated") == 1


async def test_retry_exhaustion_is_bounded_by_policy():
    judge, provider = make({"dedicated": [ProviderServerError("503", provider="mock")]}, max_retries=1)
    with pytest.raises(ProviderServerError):
        await ask(judge)
    assert provider.calls_to("dedicated") == 2  # 1 + max_retries, not the global 3


async def test_transient_error_retried_once_then_succeeds():
    judge, provider = make({"dedicated": [RateLimitError("429", provider="mock", retry_after=0), GOOD]})
    assert (await ask(judge)).judge_model == "dedicated" and provider.calls_to("dedicated") == 2


async def test_unset_dedicated_model_keeps_previous_behavior():
    judge, provider = make({"fallback": [GOOD], "primary": [GOOD]}, dedicated=False)
    assert (await ask(judge)).judge_model == "fallback"
    assert judge.second_opinion_candidates("primary") == [FALLBACK]
    assert judge.second_opinion_candidates("fallback") == [PRIMARY]


def test_second_opinion_never_uses_the_first_judge():
    judge, _ = make({"dedicated": [GOOD]})
    assert judge.second_opinion_candidates("dedicated") == [] and not judge.has_alternative("dedicated")
    assert judge.has_alternative("primary")
