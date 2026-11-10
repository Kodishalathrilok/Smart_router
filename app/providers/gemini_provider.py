"""Gemini adapter (Generative Language REST API, generateContent)."""

import httpx

from app.providers.base import EmptyResponseError, HTTPProvider, LLMResponse, TruncatedEmptyResponseError


class GeminiProvider(HTTPProvider):
    name = "gemini"

    def __init__(self, api_key: str, *, base_url: str, timeout: float, transport: httpx.AsyncBaseTransport | None = None) -> None:
        # Key goes in a header, never the URL, so it cannot leak through logged URLs.
        super().__init__(base_url=base_url, headers={"x-goog-api-key": api_key}, timeout=timeout, transport=transport)

    async def generate(self, messages, model, *, temperature=0.2, max_tokens=1024, json_mode=False) -> LLMResponse:
        system = [m["content"] for m in messages if m["role"] == "system"]
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
            if m["role"] != "system"
        ]
        config = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if json_mode:
            config["responseMimeType"] = "application/json"
        body = {"contents": contents, "generationConfig": config}
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}

        data = await self._request("POST", f"/models/{model.removeprefix('models/')}:generateContent", json=body)
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise EmptyResponseError(f"gemini returned no candidates ({reason})", provider=self.name)
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = data.get("usageMetadata") or {}
        # Thinking tokens are billed at the output price, so they count as output.
        input_tokens, output_tokens = usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        if not text.strip():
            kind = TruncatedEmptyResponseError if candidate.get("finishReason") == "MAX_TOKENS" else EmptyResponseError
            error = kind(f"gemini returned empty text (finishReason={candidate.get('finishReason')})", provider=self.name)
            error.input_tokens, error.output_tokens = input_tokens, output_tokens
            raise error
        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=candidate.get("finishReason"),
        )

    async def list_models(self) -> list[str]:
        ids: list[str] = []
        params: dict[str, str | int] = {"pageSize": 1000}
        while True:
            data = await self._request("GET", "/models", params=params)
            ids += [m["name"].removeprefix("models/") for m in data.get("models", [])]
            if not data.get("nextPageToken"):
                return ids
            params["pageToken"] = data["nextPageToken"]
