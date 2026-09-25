import pytest

from app.core.analyzer import HeuristicAnalyzer
from app.router.router import Router, model_key
from app.router.routing_policy import ThresholdPolicy
from app.router.schemas import QueryAnalysis


def analysis(d: float) -> QueryAnalysis:
    return QueryAnalysis(category="reasoning", difficulty=d)


@pytest.mark.parametrize("difficulty,tier", [(0.0, "cheap"), (0.34, "cheap"), (0.35, "mid"), (0.69, "mid"), (0.70, "strong"), (1.0, "strong")])
def test_threshold_policy(registry, difficulty, tier):
    decision = ThresholdPolicy(0.35, 0.70).select(analysis(difficulty), registry)
    assert decision.tier == tier
    assert decision.selected_model == registry.for_tier(tier).id
    assert "difficulty" in decision.reason


def test_thresholds_configurable(registry):
    assert ThresholdPolicy(0.1, 0.2).select(analysis(0.3), registry).tier == "strong"


def test_escalation_chain_and_stop(registry):
    router = Router(HeuristicAnalyzer(), ThresholdPolicy(0.35, 0.7), registry)
    tried = {model_key("mock", "mock-cheap")}
    assert router.escalate("cheap", tried).tier == "mid"
    assert router.escalate("mid", tried).tier == "strong"
    assert router.escalate("strong", tried) is None


def test_escalation_skips_already_tried_model(registry):
    router = Router(HeuristicAnalyzer(), ThresholdPolicy(0.35, 0.7), registry)
    assert router.escalate("cheap", {model_key("mock", "mock-mid")}).tier == "strong"
