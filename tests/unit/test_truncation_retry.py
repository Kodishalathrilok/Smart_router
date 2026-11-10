"""A reply that is empty because the model hit the output-token limit is retried once with a larger limit."""

import pytest

from app.core.executor import Executor, call_budget
from app.providers.base import CallBudgetExceededError, EmptyResponseError, LLMProvider, LLMResponse, TruncatedEmptyResponseError


class Scripted(LLMProvider):
    name = "mock"

    def __init__(self, outcomes: list) -> None:
        self.outcomes = outcomes
        self.limits: list[int] = []

    async def generate(self, messages, model, *, temperature=0.2, max_tokens=1024, json_mode=False) -> LLMResponse:
        self.limits.append(max_tokens)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return LLMResponse(text=out, model=model, provider=self.name)

    async def list_models(self) -> list[str]:
        return []


def truncated() -> TruncatedEmptyResponseError:
    return TruncatedEmptyResponseError("empty (finish_reason=length)", provider="mock")


async def generate(provider, registry, *, retry_limit=4096, max_tokens=1024, on_error=None):
    executor = Executor({"mock": provider}, max_retries=3, base_delay_s=0, max_delay_s=0, truncation_retry_max_tokens=retry_limit)
    return await executor.generate(registry.for_tier("cheap"), [{"role": "user", "content": "q"}], temperature=0, max_tokens=max_tokens, on_error=on_error)


def test_truncation_is_an_empty_response_and_not_transient():
    assert issubclass(TruncatedEmptyResponseError, EmptyResponseError) and not TruncatedEmptyResponseError.transient


async def test_retried_once_with_the_larger_limit(registry):
    provider = Scripted([truncated(), "42"])
    assert (await generate(provider, registry)).text == "42"
    assert provider.limits == [1024, 4096]


async def test_second_truncation_is_final(registry):
    provider = Scripted([truncated(), truncated(), "never"])
    with pytest.raises(TruncatedEmptyResponseError):
        await generate(provider, registry)
    assert provider.limits == [1024, 4096]


async def test_other_empty_replies_are_not_retried(registry):
    provider = Scripted([EmptyResponseError("blocked", provider="mock"), "never"])
    with pytest.raises(EmptyResponseError):
        await generate(provider, registry)
    assert provider.limits == [1024]


@pytest.mark.parametrize(("retry_limit", "max_tokens"), [(0, 1024), (4096, 4096), (1024, 2048)])
async def test_no_retry_when_disabled_or_not_larger(registry, retry_limit, max_tokens):
    provider = Scripted([truncated(), "never"])
    with pytest.raises(TruncatedEmptyResponseError):
        await generate(provider, registry, retry_limit=retry_limit, max_tokens=max_tokens)
    assert provider.limits == [max_tokens]


async def test_retry_draws_from_the_call_budget(registry):
    provider = Scripted([truncated(), "42"])
    with call_budget(5) as budget:
        await generate(provider, registry)
    assert budget.used == 2 and budget.billed == {"cheap": 1}

    provider = Scripted([truncated(), "never"])
    with call_budget(1), pytest.raises(CallBudgetExceededError):
        await generate(provider, registry)
    assert provider.limits == [1024]  # the retry was refused before any network call


async def test_error_hook_sees_the_truncation(registry):
    seen = []
    await generate(Scripted([truncated(), "42"]), registry, on_error=lambda exc, attempt: seen.append(exc.error_type))
    assert seen == ["TruncatedEmptyResponseError"]


def test_bootstrap_passes_the_setting(settings):
    assert settings.truncation_retry_max_tokens == 4096
