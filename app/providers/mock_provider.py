"""Deterministic offline provider for MOCK_MODE and tests.

Answers and judge scores are SYNTHETIC. They exist so the full pipeline,
benchmark and dashboard run without API keys; they say nothing about real
model quality. Synthetic quality = capability - difficulty penalty + hash jitter,
so escalation genuinely happens in mock runs.
"""

import hashlib
import json
import re
from collections.abc import Callable, Mapping

from app.providers.base import LLMProvider, LLMResponse

HINT = re.compile(r"\(mock quality hint: ([0-9.]+)\)")
QUALITY_OFFSET = 0.35
DIFFICULTY_PENALTY = 0.6
ANSWER_JITTER = 0.08
DIMENSION_JITTER = 0.05
GOOD_ABOVE = 0.8
PARTIAL_ABOVE = 0.6
FAILURE_CODES = ("incomplete", "incorrect", "reasoning_error", "irrelevant")


def _jitter(seed: str, amplitude: float) -> float:
    unit = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return (unit * 2 - 1) * amplitude


def _clamp(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 3)


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, capabilities: Mapping[str, float], difficulty: Callable[[str], float]) -> None:
        self._capabilities = dict(capabilities)
        self._difficulty = difficulty

    async def generate(self, messages, model, *, temperature=0.2, max_tokens=1024, json_mode=False) -> LLMResponse:
        prompt = messages[-1]["content"]
        text = self._judge(prompt) if json_mode else self._answer(prompt, model)
        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            input_tokens=sum(_tokens(m["content"]) for m in messages),
            output_tokens=_tokens(text),
            finish_reason="stop",
        )

    async def list_models(self) -> list[str]:
        return list(self._capabilities)

    def _answer(self, query: str, model: str) -> str:
        capability = self._capabilities.get(model, 0.5)
        quality = _clamp(capability + QUALITY_OFFSET - DIFFICULTY_PENALTY * self._difficulty(query) + _jitter(query + model, ANSWER_JITTER))
        return f"[MOCK {model}] Synthetic answer for: {query[:120]}\n\n(mock quality hint: {quality:.3f})"

    def _judge(self, prompt: str) -> str:
        match = HINT.search(prompt)
        quality = float(match.group(1)) if match else 0.5
        dims = {k: _clamp(quality + _jitter(prompt + k, DIMENSION_JITTER)) for k in ("correctness", "completeness", "relevance", "reasoning_quality")}
        if quality >= GOOD_ABOVE:
            verdict, reasons = "good", []
        else:
            verdict = "partially_correct" if quality >= PARTIAL_ABOVE else "incorrect"
            reasons = [FAILURE_CODES[int(hashlib.sha256(prompt.encode()).hexdigest()[:4], 16) % len(FAILURE_CODES)]]
        return json.dumps({"verdict": verdict, **dims, "failure_reasons": reasons, "judge_confidence": 0.9, "rationale": "Synthetic mock evaluation."})
