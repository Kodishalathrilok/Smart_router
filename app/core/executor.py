"""Generic model execution: pick the provider for a spec, retry transient errors, measure latency."""

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import TypeVar

from app.models.schemas import ModelSpec
from app.providers.base import CallBudgetExceededError, LLMProvider, LLMResponse, ProviderError, RateLimitError

T = TypeVar("T")
ErrorHook = Callable[[ProviderError, int], None]


@dataclass
class CallBudget:
    """Hard cap on provider calls (generations, judges, second opinions, and every retry) for one query.

    Without it the worst case multiplies: attempts x (generation + judge parse retries x fallback judges
    + second opinion) x transport retries. The budget makes the worst case a configured number.
    """

    limit: int
    used: int = 0  # every attempt, including ones that failed (what the budget caps)
    billed: dict[str, int] = field(default_factory=dict)  # successful responses per model role (what costs money)

    def take(self, provider: str) -> None:
        if self.used >= self.limit:
            raise CallBudgetExceededError(f"per-query provider call budget ({self.limit}) exhausted", provider=provider)
        self.used += 1


_budget: ContextVar[CallBudget | None] = ContextVar("smartrouter_call_budget", default=None)


@contextmanager
def call_budget(limit: int) -> Iterator[CallBudget]:
    """Every Executor call made inside this block (including tasks it spawns) draws from one budget."""
    budget = CallBudget(limit)
    token = _budget.set(budget)
    try:
        yield budget
    finally:
        _budget.reset(token)


async def with_retries(
    call: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay_s: float,
    max_delay_s: float,
    on_error: ErrorHook | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run `call`, retrying transient ProviderErrors with bounded exponential backoff.

    Honors Retry-After on rate limits. Non-transient errors (bad key, unknown model,
    empty response) are raised immediately. Never retries more than `max_retries` times.
    """
    for attempt in range(max_retries + 1):
        try:
            return await call()
        except ProviderError as exc:
            if on_error:
                on_error(exc, attempt)
            if not exc.transient or attempt == max_retries:
                raise
            delay = base_delay_s * 2**attempt
            if isinstance(exc, RateLimitError) and exc.retry_after is not None:
                delay = exc.retry_after
            await sleep(min(delay, max_delay_s))
    raise AssertionError("unreachable")


class Executor:
    """Calls a model through its provider and returns a normalized, timed `LLMResponse`."""

    def __init__(self, providers: Mapping[str, LLMProvider], *, max_retries: int, base_delay_s: float, max_delay_s: float) -> None:
        self._providers = providers
        self._retry = {"max_retries": max_retries, "base_delay_s": base_delay_s, "max_delay_s": max_delay_s}

    async def generate(
        self,
        spec: ModelSpec,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
        on_error: ErrorHook | None = None,
        max_retries: int | None = None,
    ) -> LLMResponse:
        """Latency covers the whole call including retries/backoff, i.e. what the caller actually waited.

        `max_retries` overrides the global retry policy for this call only (used by bounded second opinions).
        """
        provider = self._providers[spec.provider]
        retry = self._retry if max_retries is None else {**self._retry, "max_retries": max_retries}
        budget = _budget.get()

        async def call() -> LLMResponse:
            if budget is not None:
                budget.take(spec.provider)  # non-transient once exhausted: never retried
            return await provider.generate(messages, spec.id, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)

        start = perf_counter()
        response = await with_retries(call, on_error=on_error, **retry)
        if budget is not None:
            budget.billed[spec.role] = budget.billed.get(spec.role, 0) + 1
        return response.model_copy(update={"latency_ms": round((perf_counter() - start) * 1000, 1)})
