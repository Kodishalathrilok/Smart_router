"""Quality judge: an independent LLM grades a candidate answer.

IMPORTANT — the judge is not ground truth. LLM evaluators suffer from evaluator
bias, verbosity bias, errors correlated with the models being judged, their own
reasoning mistakes, and inconsistent scoring across calls. Therefore:

    quality_score != probability that the answer is correct

The benchmark pairs judge scores with reference answers and deterministic
checks (math/code) wherever possible.
"""

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from app.core.executor import ErrorHook, Executor
from app.judge.prompts import build_judge_messages
from app.judge.schemas import JudgeDimensions, Judgement
from app.models.schemas import ModelSpec

JUDGE_TEMPERATURE = 0.0
FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


@dataclass(frozen=True)
class QualityWeights:
    correctness: float
    completeness: float
    relevance: float
    reasoning_quality: float


DEFAULT_WEIGHTS = QualityWeights(correctness=0.40, completeness=0.25, relevance=0.20, reasoning_quality=0.15)
# Category overrides: correctness matters most when there is a single right answer.
CATEGORY_WEIGHTS: dict[str, QualityWeights] = {
    "coding": QualityWeights(correctness=0.50, completeness=0.20, relevance=0.15, reasoning_quality=0.15),
    "math": QualityWeights(correctness=0.50, completeness=0.15, relevance=0.10, reasoning_quality=0.25),
}


class JudgeParseError(ValueError):
    """The judge returned something that is not a valid evaluation."""


def quality_score(dims: JudgeDimensions, category: str) -> float:
    """Weighted aggregate of the judge's dimension scores."""
    w = CATEGORY_WEIGHTS.get(category, DEFAULT_WEIGHTS)
    score = (
        w.correctness * dims.correctness
        + w.completeness * dims.completeness
        + w.relevance * dims.relevance
        + w.reasoning_quality * dims.reasoning_quality
    )
    return round(score, 4)


def parse_judgement(text: str) -> JudgeDimensions:
    """Parse judge output: strict JSON first, then a fenced/embedded JSON object; always schema-validated."""
    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        try:
            return JudgeDimensions.model_validate(data)
        except ValidationError as exc:
            raise JudgeParseError(f"judge JSON failed validation: {exc.errors()[0]['msg']}") from None
    raise JudgeParseError("judge response contained no JSON object")


def _json_candidates(text: str) -> list[str]:
    candidates = [text.strip()]
    candidates += [m.strip() for m in FENCE.findall(text)]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    return candidates


class QualityJudge:
    """Grades answers with the configured judge model via the executor (never picks a provider itself)."""

    def __init__(self, executor: Executor, spec: ModelSpec, *, max_tokens: int, parse_retries: int = 1) -> None:
        self._executor = executor
        self._spec = spec
        self._max_tokens = max_tokens
        self._parse_retries = parse_retries

    @property
    def model(self) -> str:
        return self._spec.id

    async def evaluate(
        self, query: str, answer: str, category: str, *, reference: str | None = None, on_error: ErrorHook | None = None
    ) -> Judgement:
        """Raises ProviderError if the judge call fails, JudgeParseError if output stays malformed after retries."""
        messages = build_judge_messages(query, answer, category, reference)
        latency = 0.0
        error: JudgeParseError | None = None
        for _ in range(self._parse_retries + 1):
            response = await self._executor.generate(
                self._spec, messages, temperature=JUDGE_TEMPERATURE, max_tokens=self._max_tokens, json_mode=True, on_error=on_error
            )
            latency += response.latency_ms
            try:
                dims = parse_judgement(response.text)
            except JudgeParseError as exc:
                error = exc
                continue
            return Judgement(
                **dims.model_dump(),
                quality_score=quality_score(dims, category),
                judge_model=self._spec.id,
                latency_ms=round(latency, 1),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        raise error  # type: ignore[misc]
