"""Schemas for query analysis and routing decisions."""

from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal["cheap", "mid", "strong"]
TIERS: tuple[Tier, ...] = ("cheap", "mid", "strong")
Category = Literal["general_qa", "reasoning", "coding", "math", "summarization"]


class QueryAnalysis(BaseModel):
    """Features estimated from the query text before any model is called."""

    category: Category
    difficulty: float = Field(ge=0.0, le=1.0)
    requires_reasoning: bool = False
    requires_code: bool = False
    requires_math: bool = False
    requires_knowledge: bool = False
    requires_long_context: bool = False
    word_count: int = 0
    signals: list[str] = Field(default_factory=list, description="Which heuristics fired (for explainability).")


class RouteDecision(BaseModel):
    """Which model should answer, and why."""

    tier: Tier
    selected_model: str
    provider: str
    reason: str
