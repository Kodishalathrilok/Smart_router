import pytest

from app.core.executor import with_retries
from app.providers.base import AuthenticationError, ProviderServerError, RateLimitError


def flaky(errors, result="ok"):
    calls = {"n": 0}

    async def call():
        calls["n"] += 1
        if errors:
            raise errors.pop(0)
        return result

    return call, calls


async def test_retries_transient_with_backoff_then_succeeds():
    sleeps = []
    call, calls = flaky([ProviderServerError("x", provider="p"), ProviderServerError("x", provider="p")])

    async def sleep(s):
        sleeps.append(s)

    assert await with_retries(call, max_retries=3, base_delay_s=1, max_delay_s=30, sleep=sleep) == "ok"
    assert calls["n"] == 3 and sleeps == [1, 2]


async def test_honors_retry_after_and_caps_delay():
    sleeps = []
    call, _ = flaky([RateLimitError("x", provider="p", retry_after=99)])

    async def sleep(s):
        sleeps.append(s)

    await with_retries(call, max_retries=2, base_delay_s=1, max_delay_s=10, sleep=sleep)
    assert sleeps == [10]


async def test_bounded_retries_then_raise():
    call, calls = flaky([ProviderServerError("x", provider="p")] * 10)

    async def sleep(s):
        pass

    with pytest.raises(ProviderServerError):
        await with_retries(call, max_retries=2, base_delay_s=0, max_delay_s=0, sleep=sleep)
    assert calls["n"] == 3


async def test_non_transient_not_retried():
    call, calls = flaky([AuthenticationError("bad", provider="p")])
    with pytest.raises(AuthenticationError):
        await with_retries(call, max_retries=3, base_delay_s=0, max_delay_s=0)
    assert calls["n"] == 1
