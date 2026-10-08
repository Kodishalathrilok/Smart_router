"""Shared request/response and record schemas."""

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from app.router.schemas import QueryAnalysis, Tier
from app.verification.schemas import CheckResult

Mode = Literal["smart", "static", "always_cheap", "always_mid", "always_strong"]
FinalStatus = Literal["accepted", "best_effort", "failed"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_query_id() -> str:
    return f"q_{uuid4().hex[:12]}"


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str | None = Field(None, max_length=128)

    @field_validator("query")
    @classmethod
    def _encodable(cls, value: str) -> str:
        """A lone surrogate (JSON "\\ud83d") parses fine but crashes UTF-8 encoding downstream (provider call, response)."""
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("query contains invalid Unicode (lone surrogate)") from None
        return value.replace("\x00", "")


class SecondOpinion(BaseModel):
    judge_model: str
    trigger: Literal["primary_rejection", "low_confidence_acceptance"] | None = None
    quality_score: float | None = None
    verdict: str | None = None
    judge_confidence: float | None = None
    latency_ms: float = 0.0  # wall clock, success or failure, retries included
    calls: int = 0  # provider calls made (1 + retries)
    error: str | None = None


class Attempt(BaseModel):
    """One generate(+judge) step. Dimension scores are kept, not just the aggregate."""

    number: int
    model: str
    provider: str
    tier: Tier
    routing_reason: str
    answer: str | None = None
    latency_ms: float = 0.0
    judge_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    normalized_cost: float = 0.0
    quality_score: float | None = None  # primary judge; this is the answer's reported quality
    judge_model: str | None = None
    judge_confidence: float | None = None
    dimensions: dict[str, float] | None = None
    verdict: str | None = None
    checks: list[CheckResult] = Field(default_factory=list)  # answer checks (no answer key)
    second_opinion: SecondOpinion | None = None
    judge_calls: int = 0
    accepted: bool = False  # the escalation decision, not a correctness claim
    decision_reason: str | None = None
    failure_reasons: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error: str | None = None

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "fail"]


class TraceStep(BaseModel):
    kind: Literal["info", "route", "score", "escalate", "accept", "error", "final"]
    label: str


class QueryRecord(BaseModel):
    query_id: str
    timestamp: str
    query: str
    mode: Mode
    mock_mode: bool
    category: str
    difficulty: float
    analysis: QueryAnalysis
    quality_threshold: float
    attempts: list[Attempt]
    initial_model: str
    initial_tier: Tier
    final_model: str | None
    final_tier: Tier | None
    answer: str | None
    final_quality_score: float | None
    final_status: FinalStatus
    escalated: bool
    escalation_count: int
    generation_latency_ms: float
    judge_latency_ms: float
    total_latency_ms: float
    normalized_cost: float = Field(description="Sum of generation-model normalized costs. Judge calls counted separately.")
    judge_calls: int
    provider_calls: int = Field(0, description="Every provider call this query made: generations, judges, second opinions, retries.")
    billed_calls: dict[str, int] = Field(
        default_factory=dict, description="Successful responses per model role (cheap/mid/strong/judge/...): the calls a provider bills."
    )
    trace: list[TraceStep]


class QueryResponse(BaseModel):
    query_id: str
    answer: str | None
    initial_model: str
    initial_tier: Tier
    final_model: str | None
    final_tier: Tier | None
    quality_score: float | None
    final_status: FinalStatus
    escalated: bool
    attempts: int
    latency_ms: float
    normalized_cost: float
    provider_calls: int
    quality_threshold: float
    mock_mode: bool
    trace: list[TraceStep]

    @classmethod
    def from_record(cls, r: QueryRecord) -> "QueryResponse":
        return cls(
            query_id=r.query_id,
            answer=r.answer,
            initial_model=r.initial_model,
            initial_tier=r.initial_tier,
            final_model=r.final_model,
            final_tier=r.final_tier,
            quality_score=r.final_quality_score,
            final_status=r.final_status,
            escalated=r.escalated,
            attempts=len(r.attempts),
            latency_ms=r.total_latency_ms,
            normalized_cost=r.normalized_cost,
            provider_calls=r.provider_calls,
            quality_threshold=r.quality_threshold,
            mock_mode=r.mock_mode,
            trace=r.trace,
        )
