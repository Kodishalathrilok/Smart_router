"""The self-healing pipeline: analyze → route → generate → judge → accept or escalate.

Modes (all share this code, so benchmark baselines never duplicate pipeline logic):
  smart          policy routing + judge-driven escalation (SmartRouter)
  static         policy routing, judged once, no escalation
  always_cheap   cheap tier only, judged once
  always_strong  strong tier only, judged once
"""

from time import perf_counter

from app.core.config import Settings
from app.core.executor import ErrorHook, Executor
from app.judge.judge import JudgeParseError, QualityJudge
from app.judge.schemas import DIMENSIONS
from app.logging.event_logger import EventLogger
from app.logging.query_logger import QueryLogger
from app.models.registry import ModelRegistry
from app.providers.base import ProviderError
from app.router.router import Router, model_key
from app.router.schemas import QueryAnalysis, RouteDecision
from app.schemas.common import Attempt, Mode, QueryRecord, TraceStep, new_query_id, utc_now

ANSWER_SYSTEM = (
    "You are a helpful, accurate assistant. Answer the user's question directly and concisely. "
    "For math or numeric questions, end with a line 'Final answer: <value>'. "
    "For programming tasks, give the solution as a single Python code block unless another language is requested."
)
# Even a high numeric score is not accepted if the judge's categorical verdict is a hard failure.
REJECT_VERDICTS = frozenset({"incorrect", "irrelevant"})


class SmartRouterPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        registry: ModelRegistry,
        router: Router,
        executor: Executor,
        judge: QualityJudge,
        events: EventLogger,
        queries: QueryLogger,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._router = router
        self._executor = executor
        self._judge = judge
        self._events = events
        self._queries = queries

    async def run(self, query: str, *, mode: Mode = "smart", query_id: str | None = None, reference: str | None = None) -> QueryRecord:
        """Process one query end to end. All per-query state is local to this call."""
        qid = query_id or new_query_id()
        start = perf_counter()
        self._events.log(qid, "query_received", mode=mode, query_chars=len(query))
        try:
            record = await self._run(qid, query, mode, reference, start)
        except Exception as exc:
            self._events.log(qid, "pipeline_failed", error_type=type(exc).__name__, error=self._settings.redact(str(exc)))
            raise
        self._events.log(
            qid,
            "pipeline_completed",
            final_status=record.final_status,
            initial_model=record.initial_model,
            final_model=record.final_model,
            final_quality_score=record.final_quality_score,
            attempts=len(record.attempts),
            escalation_count=record.escalation_count,
            total_latency_ms=record.total_latency_ms,
            normalized_cost=record.normalized_cost,
        )
        self._queries.write(record)
        return record

    async def _run(self, qid: str, query: str, mode: Mode, reference: str | None, start: float) -> QueryRecord:
        trace = [TraceStep(kind="info", label="Query received")]
        analysis = self._router.analyze(query)
        self._events.log(qid, "query_analyzed", **analysis.model_dump())
        trace.append(TraceStep(kind="info", label=f"Category: {analysis.category} · difficulty estimated: {analysis.difficulty:.2f}"))

        decision = self._initial_decision(analysis, mode)
        self._events.log(
            qid,
            "route_selected",
            category=analysis.category,
            difficulty=analysis.difficulty,
            selected_tier=decision.tier,
            selected_model=decision.selected_model,
            provider=decision.provider,
            reason=decision.reason,
            mode=mode,
        )
        trace.append(_route_step(decision))

        attempts: list[Attempt] = []
        tried: set[str] = set()
        while True:
            attempt = await self._attempt(qid, query, analysis, decision, len(attempts) + 1, reference)
            attempts.append(attempt)
            tried.add(model_key(attempt.provider, attempt.model))
            trace.append(self._attempt_step(attempt))
            if attempt.accepted:
                trace.append(TraceStep(kind="accept", label="Accepted"))
                break
            if attempt.answer is not None and attempt.quality_score is None:
                # A judge outage (quota, timeout, malformed output) is not evidence of a bad answer;
                # escalating would spend stronger models without any quality signal.
                trace.append(TraceStep(kind="final", label="Judge unavailable: answer returned unverified, no escalation"))
                break
            if mode != "smart":
                trace.append(TraceStep(kind="final", label=f"Below threshold; no escalation in {mode} mode"))
                break
            if len(attempts) - 1 >= self._settings.max_escalations:
                trace.append(TraceStep(kind="final", label=f"Max escalations reached ({self._settings.max_escalations})"))
                break
            next_decision = self._router.escalate(decision.tier, tried)
            if next_decision is None:
                trace.append(TraceStep(kind="final", label=f"No stronger model available above {decision.tier}"))
                break
            self._events.log(
                qid,
                "escalation_triggered",
                from_tier=decision.tier,
                from_model=decision.selected_model,
                to_tier=next_decision.tier,
                to_model=next_decision.selected_model,
                quality_score=attempt.quality_score,
                threshold=self._settings.quality_threshold,
                cause=attempt.error_type or "below_threshold",
            )
            trace.append(TraceStep(kind="escalate", label="Escalation triggered"))
            trace.append(_route_step(next_decision))
            decision = next_decision

        return self._record(qid, query, mode, analysis, attempts, trace, start)

    def _initial_decision(self, analysis: QueryAnalysis, mode: Mode) -> RouteDecision:
        if mode == "always_cheap":
            return self._router.fixed("cheap", "baseline: always cheap")
        if mode == "always_strong":
            return self._router.fixed("strong", "baseline: always strong")
        return self._router.route(analysis)

    async def _attempt(
        self, qid: str, query: str, analysis: QueryAnalysis, decision: RouteDecision, number: int, reference: str | None
    ) -> Attempt:
        spec = self._registry.for_tier(decision.tier)
        base = dict(number=number, model=spec.id, provider=spec.provider, tier=decision.tier, routing_reason=decision.reason)
        self._events.log(qid, "generation_started", attempt=number, model=spec.id, tier=decision.tier)
        messages = [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": query}]
        try:
            response = await self._executor.generate(
                spec,
                messages,
                temperature=self._settings.generation_temperature,
                max_tokens=self._settings.generation_max_tokens,
                on_error=self._error_hook(qid, "generation"),
            )
        except ProviderError as exc:
            error = self._settings.redact(str(exc))
            self._events.log(qid, "generation_failed", attempt=number, model=spec.id, error_type=exc.error_type, error=error)
            return Attempt(**base, error_type=exc.error_type, error=error)

        self._events.log(
            qid,
            "generation_completed",
            attempt=number,
            model=spec.id,
            tier=decision.tier,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            finish_reason=response.finish_reason,
        )
        generated = dict(
            answer=response.text,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            finish_reason=response.finish_reason,
            normalized_cost=spec.normalized_cost,
        )
        try:
            judgement = await self._judge.evaluate(
                query, response.text, analysis.category, reference=reference, on_error=self._error_hook(qid, "judge")
            )
        except (ProviderError, JudgeParseError) as exc:
            error_type = exc.error_type if isinstance(exc, ProviderError) else "JudgeParseError"
            error = self._settings.redact(f"judge: {exc}")
            self._events.log(qid, "judge_failed", attempt=number, model=spec.id, judge_model=self._judge.model, error_type=error_type, error=error)
            return Attempt(**base, **generated, error_type=error_type, error=error)

        accepted = judgement.quality_score >= self._settings.quality_threshold and judgement.verdict not in REJECT_VERDICTS
        dimensions = {d: getattr(judgement, d) for d in DIMENSIONS}
        self._events.log(
            qid,
            "judge_completed",
            attempt=number,
            model=spec.id,
            judge_model=judgement.judge_model,
            quality_score=judgement.quality_score,
            dimensions=dimensions,
            verdict=judgement.verdict,
            failure_reasons=judgement.failure_reasons,
            accepted=accepted,
            threshold=self._settings.quality_threshold,
            judge_latency_ms=judgement.latency_ms,
        )
        return Attempt(
            **base,
            **generated,
            judge_latency_ms=judgement.latency_ms,
            quality_score=judgement.quality_score,
            dimensions=dimensions,
            verdict=judgement.verdict,
            accepted=accepted,
            failure_reasons=list(judgement.failure_reasons),
        )

    def _error_hook(self, qid: str, stage: str) -> ErrorHook:
        def hook(exc: ProviderError, retry: int) -> None:
            self._events.log(
                qid,
                "provider_error",
                stage=stage,
                provider=exc.provider,
                error_type=exc.error_type,
                status_code=exc.status_code,
                retry=retry,
                transient=exc.transient,
                error=self._settings.redact(str(exc)),
            )

        return hook

    def _attempt_step(self, a: Attempt) -> TraceStep:
        if a.answer is None:
            return TraceStep(kind="error", label=f"Generation failed: {a.error_type}")
        if a.quality_score is None:
            return TraceStep(kind="error", label=f"Judge failed: {a.error_type} (answer unverified)")
        return TraceStep(kind="score", label=f"Quality score: {a.quality_score:.2f} (threshold {self._settings.quality_threshold:.2f}) · verdict: {a.verdict}")

    def _record(
        self, qid: str, query: str, mode: Mode, analysis: QueryAnalysis, attempts: list[Attempt], trace: list[TraceStep], start: float
    ) -> QueryRecord:
        final = next((a for a in attempts if a.accepted), None)
        status = "accepted"
        if final is None:
            judged = [a for a in attempts if a.quality_score is not None]
            answered = [a for a in attempts if a.answer is not None]
            final = max(judged, key=lambda a: a.quality_score) if judged else (answered[-1] if answered else None)
            status = "best_effort" if final else "failed"
            if final:
                score = f"{final.quality_score:.2f}" if final.quality_score is not None else "unscored"
                trace.append(TraceStep(kind="final", label=f"Best effort: returning highest-scoring candidate ({final.tier}, {score}); below threshold"))
            else:
                trace.append(TraceStep(kind="error", label="All attempts failed; no answer produced"))

        return QueryRecord(
            query_id=qid,
            timestamp=utc_now(),
            query=query,
            mode=mode,
            mock_mode=self._settings.mock_mode,
            category=analysis.category,
            difficulty=analysis.difficulty,
            analysis=analysis,
            quality_threshold=self._settings.quality_threshold,
            attempts=attempts,
            initial_model=attempts[0].model,
            initial_tier=attempts[0].tier,
            final_model=final.model if final else None,
            final_tier=final.tier if final else None,
            answer=final.answer if final else None,
            final_quality_score=final.quality_score if final else None,
            final_status=status,
            escalated=len(attempts) > 1,
            escalation_count=len(attempts) - 1,
            generation_latency_ms=round(sum(a.latency_ms for a in attempts), 1),
            judge_latency_ms=round(sum(a.judge_latency_ms for a in attempts), 1),
            total_latency_ms=round((perf_counter() - start) * 1000, 1),
            normalized_cost=sum(a.normalized_cost for a in attempts),
            judge_calls=sum(1 for a in attempts if a.answer is not None),
            trace=trace,
        )


def _route_step(decision: RouteDecision) -> TraceStep:
    return TraceStep(kind="route", label=f"{decision.tier.capitalize()} model selected ({decision.selected_model}): {decision.reason}")
