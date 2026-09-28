"""Wire settings → registry → providers → pipeline. Shared by the API and the benchmark."""

from dataclasses import dataclass
from pathlib import Path

from app.core.analyzer import HeuristicAnalyzer
from app.core.config import Settings
from app.core.executor import Executor
from app.core.pipeline import SmartRouterPipeline
from app.judge.judge import QualityJudge
from app.logging.event_logger import EventLogger
from app.logging.query_logger import QueryLogger
from app.models.registry import ModelRegistry
from app.providers.base import LLMProvider
from app.providers.factory import build_providers
from app.router.router import Router
from app.router.routing_policy import ThresholdPolicy


@dataclass
class Runtime:
    settings: Settings
    registry: ModelRegistry
    providers: dict[str, LLMProvider]
    pipeline: SmartRouterPipeline
    judge: QualityJudge
    events: EventLogger
    queries: QueryLogger

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()


async def build_runtime(settings: Settings, *, log_dir: Path | None = None, providers: dict[str, LLMProvider] | None = None) -> Runtime:
    """Build and validate everything. Raises ConfigError on bad configuration."""
    registry = ModelRegistry.from_settings(settings)
    providers = providers if providers is not None else build_providers(settings, registry.all())
    if settings.validate_models:
        await registry.validate(providers)

    log_dir = log_dir or settings.log_dir
    events = EventLogger(log_dir / "events.jsonl")
    queries = QueryLogger(log_dir / "queries.jsonl")
    executor = Executor(
        providers, max_retries=settings.max_retries, base_delay_s=settings.retry_base_delay_s, max_delay_s=settings.retry_max_delay_s
    )
    router = Router(HeuristicAnalyzer(), ThresholdPolicy(settings.cheap_max_difficulty, settings.mid_max_difficulty), registry)
    judges = [spec for spec in (registry.judge, registry.judge_fallback) if spec]
    judge = QualityJudge(executor, judges, max_tokens=settings.judge_max_tokens)
    pipeline = SmartRouterPipeline(
        settings=settings, registry=registry, router=router, executor=executor, judge=judge, events=events, queries=queries
    )
    return Runtime(
        settings=settings, registry=registry, providers=providers, pipeline=pipeline, judge=judge, events=events, queries=queries
    )
