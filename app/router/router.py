"""Router: analyzer + policy + registry. Decides initial model and escalation targets. No provider code."""

from app.core.analyzer import QueryAnalyzer
from app.models.registry import ModelRegistry
from app.router.routing_policy import RoutingPolicy
from app.router.schemas import TIERS, QueryAnalysis, RouteDecision, Tier


class Router:
    def __init__(self, analyzer: QueryAnalyzer, policy: RoutingPolicy, registry: ModelRegistry) -> None:
        self._analyzer = analyzer
        self._policy = policy
        self._registry = registry

    def analyze(self, query: str) -> QueryAnalysis:
        return self._analyzer.analyze(query)

    def route(self, analysis: QueryAnalysis) -> RouteDecision:
        return self._policy.select(analysis, self._registry)

    def fixed(self, tier: Tier, reason: str) -> RouteDecision:
        spec = self._registry.for_tier(tier)
        return RouteDecision(tier=tier, selected_model=spec.id, provider=spec.provider, reason=reason)

    def escalate(self, current: Tier, tried: set[str]) -> RouteDecision | None:
        """Next stronger tier whose model has not been tried yet, or None (strong → stop)."""
        for tier in TIERS[TIERS.index(current) + 1 :]:
            spec = self._registry.for_tier(tier)
            if model_key(spec.provider, spec.id) not in tried:
                return RouteDecision(tier=tier, selected_model=spec.id, provider=spec.provider, reason=f"escalated from {current}: quality below threshold")
        return None


def model_key(provider: str, model_id: str) -> str:
    return f"{provider}:{model_id}"
