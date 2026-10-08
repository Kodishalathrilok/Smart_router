"""Judge output schemas.

The judge produces per-dimension scores; SmartRouter computes the aggregate
`quality_score` itself. quality_score is an LLM evaluation signal, NOT a
probability of correctness (see app/judge/judge.py for known biases).
"""

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Verdict = Literal["good", "partially_correct", "incorrect", "incomplete", "irrelevant"]
FailureCode = Literal["incorrect", "incomplete", "irrelevant", "reasoning_error", "unsafe", "other"]
FAILURE_CODES: tuple[str, ...] = FailureCode.__args__
DIMENSIONS = ("correctness", "completeness", "relevance", "reasoning_quality")
_VERDICT_ALIASES = {"partial": "partially_correct", "partially correct": "partially_correct", "correct": "good"}
_REASON_ALIASES = {"reasoning": "reasoning_error", "wrong": "incorrect", "off_topic": "irrelevant", "unsafe_content": "unsafe"}


class JudgeDimensions(BaseModel):
    """What the judge model must return."""

    verdict: Verdict
    correctness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    reasoning_quality: float = Field(ge=0.0, le=1.0)
    failure_reasons: list[FailureCode] = Field(default_factory=list)
    rationale: str = ""
    judge_confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="The judge's certainty in its own verdict. Not a probability that the answer is correct."
    )

    @field_validator(*DIMENSIONS, "judge_confidence", mode="before")
    @classmethod
    def _strict_score(cls, value: object) -> float | None:
        """Scores must be finite numbers in [0, 1]. Nothing is rescaled or clamped.

        A judge that answers 7 might mean 7/10, 0.7 typed badly, or be confused; guessing would silently
        change the escalation decision. An out-of-range score is a parse error (retried, then the fallback
        judge, then "judge unavailable").
        """
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("score must be a JSON number")  # strings, lists, dicts
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        return score

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalize_verdict(cls, value: object) -> str:
        text = str(value).strip().lower().replace("-", "_")
        return _VERDICT_ALIASES.get(text, text.replace(" ", "_"))

    @field_validator("failure_reasons", mode="before")
    @classmethod
    def _normalize_reasons(cls, value: object) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        codes = []
        for item in items:
            code = str(item).strip().lower().replace(" ", "_").replace("-", "_")
            code = _REASON_ALIASES.get(code, code)
            codes.append(code if code in FAILURE_CODES else "other")
        return list(dict.fromkeys(codes))


class Judgement(JudgeDimensions):
    """Judge dimensions plus the aggregate score and call metadata."""

    quality_score: float
    judge_model: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
