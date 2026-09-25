"""Generic model execution: pick the provider for a spec, retry transient errors, measure latency."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from time import perf_counter
from typing import TypeVar

from app.models.schemas import ModelSpec
from app.providers.base import LLMProvider, LLMResponse, ProviderError, RateLimitError

T = TypeVar("T")
ErrorHook = Callable[[ProviderError, int], None]


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
    ) -> LLMResponse:
        """Latency covers the whole call including retries/backoff, i.e. what the caller actually waited."""
        provider = self._providers[spec.provider]
        start = perf_counter()
        response = await with_retries(
            lambda: provider.generate(messages, spec.id, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode),
            on_error=on_error,
            **self._retry,
        )
        return response.model_copy(update={"latency_ms": round((perf_counter() - start) * 1000, 1)})
