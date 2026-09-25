"""Routing policies: map a query analysis to a model tier. Provider-agnostic.

Extension point: learned, bandit or history-weighted policies implement the same
`select` signature; `history` carries past query records for such policies.
"""

from collections.abc import Sequence
from typing import Any, Protocol

from app.models.registry import ModelRegistry
from app.router.schemas import QueryAnalysis, RouteDecision, Tier


class RoutingPolicy(Protocol):
    def select(self, analysis: QueryAnalysis, registry: ModelRegistry, history: Sequence[Any] | None = None) -> RouteDecision: ...


class ThresholdPolicy:
    """difficulty < cheap_max → cheap; < mid_max → mid; otherwise strong."""

    def __init__(self, cheap_max: float, mid_max: float) -> None:
        self.cheap_max = cheap_max
        self.mid_max = mid_max

    def select(self, analysis: QueryAnalysis, registry: ModelRegistry, history: Sequence[Any] | None = None) -> RouteDecision:
        d = analysis.difficulty
        tier: Tier
        if d < self.cheap_max:
            tier, why = "cheap", f"low estimated difficulty ({d:.2f} < {self.cheap_max})"
        elif d < self.mid_max:
            tier, why = "mid", f"moderate estimated difficulty ({self.cheap_max} <= {d:.2f} < {self.mid_max})"
        else:
            tier, why = "strong", f"high estimated difficulty ({d:.2f} >= {self.mid_max})"
        spec = registry.for_tier(tier)
        return RouteDecision(tier=tier, selected_model=spec.id, provider=spec.provider, reason=f"{analysis.category} query, {why}")
