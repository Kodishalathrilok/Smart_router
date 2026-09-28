"""Model registry schemas."""

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["cheap", "mid", "strong", "judge", "judge_fallback"]


class ModelProfile(BaseModel):
    """Static metadata for a role, from config/models.json."""

    capability_score: float = Field(ge=0.0, le=1.0)
    normalized_cost: float = Field(ge=0.0, description="Experiment weight per call. Not money.")
    expected_latency: Literal["low", "medium", "high"]
    strengths: list[str] = Field(default_factory=list)
    price_per_mtok_input: float | None = Field(None, description="Optional real USD pricing. Kept separate from normalized_cost.")
    price_per_mtok_output: float | None = None


class ModelSpec(ModelProfile):
    """A concrete, configured model filling a role."""

    id: str
    provider: str
    role: Role
