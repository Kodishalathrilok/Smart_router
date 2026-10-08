"""Shared fixtures. No test needs real API keys or network."""

import json

import pytest

from app.core.config import Settings
from app.models.registry import ModelRegistry
from app.providers.base import LLMProvider, LLMResponse


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        mock_mode=True,
        log_dir=tmp_path / "logs",
        benchmark_dir=tmp_path / "benchmarks",
        results_dir=tmp_path / "results",
        retry_base_delay_s=0,
        max_retries=1,
        allowed_hosts="testserver,127.0.0.1,localhost",  # TestClient sends Host: testserver
    )
    return Settings(_env_file=None, **{**base, **overrides})


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def registry(settings) -> ModelRegistry:
    return ModelRegistry.from_settings(settings)


def judge_json(score: float, reasons=(), verdict=None, confidence=None) -> str:
    verdict = verdict or ("good" if score >= 0.8 else "partially_correct")
    payload = {
        "verdict": verdict,
        "correctness": score,
        "completeness": score,
        "relevance": score,
        "reasoning_quality": score,
        "failure_reasons": list(reasons),
        "rationale": "test",
    }
    if confidence is not None:
        payload["judge_confidence"] = confidence
    return json.dumps(payload)


class FakeProvider(LLMProvider):
    """Scripted provider.

    answers: model id -> answer text, or an Exception to raise.
    judgements: answer text -> judge output (quality float, raw string, or Exception).
    """

    name = "mock"

    def __init__(self, answers: dict, judgements: dict | None = None) -> None:
        self.answers = answers
        self.judgements = judgements or {}
        self.calls: list[tuple[str, bool]] = []

    async def generate(self, messages, model, *, temperature=0.2, max_tokens=1024, json_mode=False) -> LLMResponse:
        self.calls.append((model, json_mode))
        if json_mode:
            prompt = messages[-1]["content"]
            out = next((v for k, v in self.judgements.items() if k in prompt), 0.5)
        else:
            out = self.answers[model]
        if isinstance(out, Exception):
            raise out
        text = judge_json(out) if isinstance(out, float) else out
        return LLMResponse(text=text, model=model, provider=self.name, input_tokens=10, output_tokens=20)

    async def list_models(self) -> list[str]:
        return list(self.answers)

    def generation_calls(self) -> list[str]:
        return [m for m, is_judge in self.calls if not is_judge]
