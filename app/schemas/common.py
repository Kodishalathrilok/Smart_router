"""Shared request/response and record schemas."""

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.router.schemas import QueryAnalysis, Tier

Mode = Literal["smart", "static", "always_cheap", "always_strong"]
FinalStatus = Literal["accepted", "best_effort", "failed"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_query_id() -> str:
    return f"q_{uuid4().hex[:12]}"


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str | None = Field(None, max_length=128)


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
    quality_score: float | None = None
    judge_model: str | None = None
    dimensions: dict[str, float] | None = None
    verdict: str | None = None
    accepted: bool = False
    failure_reasons: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error: str | None = None


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
            quality_threshold=r.quality_threshold,
            mock_mode=r.mock_mode,
            trace=r.trace,
        )
