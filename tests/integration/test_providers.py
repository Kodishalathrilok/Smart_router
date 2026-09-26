"""Provider adapters against a mocked HTTP transport: normalization and error mapping."""

import json

import httpx
import pytest

from app.providers.base import (
    AuthenticationError,
    BadRequestError,
    EmptyResponseError,
    ModelUnavailableError,
    NetworkError,
    ProviderServerError,
    ProviderTimeoutError,
    RateLimitError,
)
from app.providers.gemini_provider import GeminiProvider
from app.providers.groq_provider import GroqProvider

MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


def groq(handler):
    return GroqProvider("sk-test", base_url="https://groq.test/v1", timeout=5, transport=httpx.MockTransport(handler))


def gemini(handler):
    return GeminiProvider("gm-test", base_url="https://gemini.test/v1beta", timeout=5, transport=httpx.MockTransport(handler))


async def test_groq_normalizes_response_and_sends_json_mode():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={
            "model": "m1",
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        })

    r = await groq(handler).generate(MESSAGES, "m1", json_mode=True)
    assert (r.text, r.provider, r.input_tokens, r.output_tokens, r.finish_reason) == ("hello", "groq", 7, 2, "stop")
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["auth"] == "Bearer sk-test"


async def test_gemini_normalizes_response_and_keeps_key_out_of_url():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers["x-goog-api-key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}, {"text": "answer"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
        })

    r = await gemini(handler).generate(MESSAGES, "models/g1", json_mode=True)
    assert (r.text, r.input_tokens, r.output_tokens) == ("answer", 5, 3)
    assert "gm-test" not in seen["url"] and seen["key"] == "gm-test"
    assert seen["url"].endswith("/models/g1:generateContent")
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "be brief"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"


@pytest.mark.parametrize("status,body,headers,error", [
    (401, "bad key", {}, AuthenticationError),
    (400, "API key not valid", {}, AuthenticationError),
    (404, "model_not_found", {}, ModelUnavailableError),
    (429, "slow down", {"retry-after": "7"}, RateLimitError),
    (429, "Request too large for model on output tokens per minute", {}, BadRequestError),
    (503, "overloaded", {}, ProviderServerError),
])
async def test_http_errors_are_typed(status, body, headers, error):
    provider = groq(lambda request: httpx.Response(status, text=body, headers=headers))
    with pytest.raises(error) as exc_info:
        await provider.generate(MESSAGES, "m1")
    if error is RateLimitError:
        assert exc_info.value.retry_after == 7.0
        assert exc_info.value.transient


async def test_timeout_and_network_errors():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderTimeoutError):
        await groq(timeout).generate(MESSAGES, "m1")
    with pytest.raises(NetworkError):
        await gemini(refused).generate(MESSAGES, "g1")


async def test_empty_responses_raise():
    empty_groq = groq(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "  "}, "finish_reason": "length"}]}))
    blocked_gemini = gemini(lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(EmptyResponseError):
        await empty_groq.generate(MESSAGES, "m1")
    with pytest.raises(EmptyResponseError, match="SAFETY"):
        await blocked_gemini.generate(MESSAGES, "g1")


async def test_list_models():
    assert await groq(lambda r: httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})).list_models() == ["a", "b"]

    def paged(request):
        if "pageToken" in request.url.params:
            return httpx.Response(200, json={"models": [{"name": "models/g2"}]})
        return httpx.Response(200, json={"models": [{"name": "models/g1"}], "nextPageToken": "t"})

    assert await gemini(paged).list_models() == ["g1", "g2"]
