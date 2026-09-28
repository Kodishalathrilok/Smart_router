"""Adapter for OpenAI-compatible chat completion APIs (Groq, NVIDIA)."""

import httpx

from app.providers.base import EmptyResponseError, HTTPProvider, LLMResponse, ProviderServerError


class OpenAICompatibleProvider(HTTPProvider):
    """`token_param` differs between vendors: Groq uses `max_completion_tokens`, NVIDIA `max_tokens`."""

    def __init__(
        self,
        name: str,
        api_key: str,
        *,
        base_url: str,
        timeout: float,
        token_param: str = "max_tokens",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        self._token_param = token_param
        super().__init__(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout, transport=transport)

    async def generate(self, messages, model, *, temperature=0.2, max_tokens=1024, json_mode=False) -> LLMResponse:
        body = {"model": model, "messages": messages, "temperature": temperature, self._token_param: max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = await self._request("POST", "/chat/completions", json=body)
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError):
            raise ProviderServerError(f"{self.name} response had no choices", provider=self.name) from None
        text = (choice.get("message") or {}).get("content") or ""
        if not text.strip():
            raise EmptyResponseError(f"{self.name} returned an empty completion (finish_reason={choice.get('finish_reason')})", provider=self.name)
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", model),
            provider=self.name,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason"),
        )

    async def list_models(self) -> list[str]:
        data = await self._request("GET", "/models")
        return [m["id"] for m in data.get("data", [])]
