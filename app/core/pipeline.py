"""The request pipeline: analyze → route → generate → verify → judge → accept or escalate.

Modes (all share this code, so benchmark baselines never duplicate pipeline logic):
  smart          policy routing + judge-driven escalation (SmartRouter)
  static         policy routing, judged once, no escalation
  always_cheap   cheap tier only, judged once
  always_mid     mid tier only, judged once
  always_strong  strong tier only, judged once
"""

import asyncio
from time import perf_counter

from app.core.config import Settings
from app.core.escalation import decide, judge_passes
from app.core.executor import CallBudget, ErrorHook, Executor, call_budget
from app.judge.judge import JudgeParseError, QualityJudge
from app.judge.schemas import DIMENSIONS, Judgement
from app.logging.event_logger import EventLogger
from app.logging.query_logger import QueryLogger
from app.models.registry import ModelRegistry
from app.providers.base import ProviderError
from app.router.router import Router, model_key
from app.router.schemas import QueryAnalysis, RouteDecision
from app.schemas.common import Attempt, Mode, QueryRecord, SecondOpinion, TraceStep, new_query_id, utc_now
from app.verification.checks import check_answer

ANSWER_SYSTEM = (
    "You are a helpful, accurate assistant. Answer the user's question directly and concisely. "
    "For math or numeric questions, end with a line 'Final answer: <value>'. "
    "For programming tasks, give the solution as a single Python code block unless another language is requested."
)


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
        with call_budget(self._settings.max_provider_calls_per_query) as budget:
            try:
                record = await self._run(qid, query, mode, reference, start, budget)
            except asyncio.CancelledError:
                # Request deadline or shutdown: the calls already made were paid for, so leave a trace.
                self._events.log(qid, "pipeline_failed", error_type="Cancelled", provider_calls=budget.used)
                raise
            except Exception as exc:
                self._events.log(
                    qid, "pipeline_failed", error_type=type(exc).__name__, error=self._settings.redact(str(exc)), provider_calls=budget.used
                )
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
            provider_calls=record.provider_calls,
        )
        self._queries.write(record)
        return record

    async def _run(self, qid: str, query: str, mode: Mode, reference: str | None, start: float, budget: CallBudget) -> QueryRecord:
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
            attempt = await self._attempt(qid, query, analysis, decision, len(attempts) + 1, reference, mode)
            attempts.append(attempt)
            tried.add(model_key(attempt.provider, attempt.model))
            trace.extend(self._attempt_steps(attempt))
            if attempt.accepted:
                trace.append(TraceStep(kind="accept", label=f"Accepted: {attempt.decision_reason}"))
                break
            if attempt.answer is not None and attempt.quality_score is None and not attempt.failed_checks:
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
                cause=attempt.decision_reason or attempt.error_type or "below_threshold",
            )
            trace.append(TraceStep(kind="escalate", label=f"Escalation triggered: {attempt.decision_reason or attempt.error_type}"))
            if attempt.answer is None:
                next_decision = next_decision.model_copy(update={"reason": f"escalated from {decision.tier}: generation failed ({attempt.error_type})"})
            trace.append(_route_step(next_decision))
            decision = next_decision

        return self._record(qid, query, mode, analysis, attempts, trace, start, budget)

    def _initial_decision(self, analysis: QueryAnalysis, mode: Mode) -> RouteDecision:
        if mode == "always_cheap":
            return self._router.fixed("cheap", "baseline: always cheap")
        if mode == "always_mid":
            return self._router.fixed("mid", "baseline: always mid")
        if mode == "always_strong":
            return self._router.fixed("strong", "baseline: always strong")
        return self._router.route(analysis)

    async def _attempt(
        self, qid: str, query: str, analysis: QueryAnalysis, decision: RouteDecision, number: int, reference: str | None, mode: Mode
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
            checks = await asyncio.to_thread(
                check_answer,
                query,
                response.text,
                execute_examples=self._settings.verify_execute_code,
                timeout_s=self._settings.code_check_timeout_s,
            )
        except Exception as exc:  # a verifier bug must not destroy an answer that was already paid for
            self._events.log(qid, "checks_error", attempt=number, model=spec.id, error_type=type(exc).__name__)
            checks = []
        if checks:
            self._events.log(qid, "checks_completed", attempt=number, model=spec.id, checks=[c.model_dump() for c in checks])
        generated["checks"] = checks

        try:
            judgement = await self._judge.evaluate(
                query, response.text, analysis.category, reference=reference, on_error=self._error_hook(qid, "judge")
            )
        except (ProviderError, JudgeParseError) as exc:
            error_type = exc.error_type if isinstance(exc, ProviderError) else "JudgeParseError"
            error = self._settings.redact(f"judge: {exc}")
            self._events.log(qid, "judge_failed", attempt=number, model=spec.id, judge_model=self._judge.model, error_type=error_type, error=error)
            failed = [c for c in checks if c.status == "fail"]
            reason = ("judge unavailable; answer check failed (" + "; ".join(f"{c.name}: {c.detail}" for c in failed) + ")") if failed else None
            return Attempt(**base, **generated, judge_calls=1, error_type=error_type, error=error, decision_reason=reason)

        threshold, min_confidence = self._settings.quality_threshold, self._settings.judge_confidence_min
        verdict = decide(checks, judgement, threshold=threshold, min_confidence=min_confidence)
        judge_calls, second, second_record = 1, None, None
        if verdict.action == "second_opinion":
            if mode == "smart" and self._settings.second_opinion and self._judge.has_alternative(judgement.judge_model):
                judge_calls += 1
                trigger = "low_confidence_acceptance" if judge_passes(judgement.quality_score, judgement.verdict, threshold) else "primary_rejection"
                second, second_record = await self._second_opinion(
                    qid, number, spec.id, query, response.text, analysis.category, reference, judgement.judge_model, trigger
                )
            verdict = decide(checks, judgement, threshold=threshold, min_confidence=min_confidence, second=second, second_asked=True)
        accepted = verdict.action == "accept"

        dimensions = {d: getattr(judgement, d) for d in DIMENSIONS}
        self._events.log(
            qid,
            "judge_completed",
            attempt=number,
            model=spec.id,
            judge_model=judgement.judge_model,
            quality_score=judgement.quality_score,
            judge_confidence=judgement.judge_confidence,
            dimensions=dimensions,
            verdict=judgement.verdict,
            failure_reasons=judgement.failure_reasons,
            accepted=accepted,
            decision_reason=verdict.reason,
            threshold=threshold,
            judge_latency_ms=judgement.latency_ms,
        )
        return Attempt(
            **base,
            **generated,
            judge_latency_ms=round(judgement.latency_ms + (second_record.latency_ms if second_record else 0.0), 1),
            quality_score=judgement.quality_score,
            judge_model=judgement.judge_model,
            judge_confidence=judgement.judge_confidence,
            dimensions=dimensions,
            verdict=judgement.verdict,
            second_opinion=second_record,
            judge_calls=judge_calls,
            accepted=accepted,
            decision_reason=verdict.reason,
            failure_reasons=list(judgement.failure_reasons),
        )

    async def _second_opinion(
        self,
        qid: str,
        number: int,
        model: str,
        query: str,
        answer: str,
        category: str,
        reference: str | None,
        first_judge: str,
        trigger: str,
    ) -> tuple[Judgement | None, SecondOpinion]:
        """Ask a model other than `first_judge`, within the second-opinion budget. Failure returns (None, record with the error)."""
        failed_calls = 0
        log_error = self._error_hook(qid, "second_opinion")

        def on_error(exc: ProviderError, retry: int) -> None:
            nonlocal failed_calls
            failed_calls += 1
            log_error(exc, retry)

        start = perf_counter()
        try:
            second = await self._judge.second_opinion(query, answer, category, first_judge=first_judge, reference=reference, on_error=on_error)
        except (ProviderError, JudgeParseError) as exc:
            latency = round((perf_counter() - start) * 1000, 1)
            error = self._settings.redact(f"{type(exc).__name__}: {exc}")
            self._events.log(qid, "second_opinion_failed", attempt=number, model=model, trigger=trigger, latency_ms=latency, error=error)
            candidates = self._judge.second_opinion_candidates(first_judge)
            return None, SecondOpinion(
                judge_model=candidates[0].id if candidates else "unknown",
                trigger=trigger,
                latency_ms=latency,
                calls=max(1, failed_calls),
                error=error,
            )
        latency = round((perf_counter() - start) * 1000, 1)
        self._events.log(
            qid,
            "second_opinion_completed",
            attempt=number,
            model=model,
            trigger=trigger,
            judge_model=second.judge_model,
            quality_score=second.quality_score,
            verdict=second.verdict,
            judge_confidence=second.judge_confidence,
            latency_ms=latency,
        )
        return second, SecondOpinion(
            judge_model=second.judge_model,
            trigger=trigger,
            quality_score=second.quality_score,
            verdict=second.verdict,
            judge_confidence=second.judge_confidence,
            latency_ms=latency,
            calls=failed_calls + 1,
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

    def _attempt_steps(self, a: Attempt) -> list[TraceStep]:
        if a.answer is None:
            return [TraceStep(kind="error", label=f"Generation failed: {a.error_type}")]
        steps = [TraceStep(kind="error", label=f"Answer check failed: {c.name} ({c.detail})") for c in a.failed_checks]
        passed = [c.name for c in a.checks if c.status == "pass"]
        if passed:
            steps.append(TraceStep(kind="info", label=f"Answer checks passed: {', '.join(passed)}"))
        unsure = [f"{c.name} ({c.detail})" for c in a.checks if c.status == "not_checkable"]
        if unsure:
            steps.append(TraceStep(kind="info", label=f"Answer checks inconclusive (no effect): {'; '.join(unsure)}"))
        if a.quality_score is None:
            return [*steps, TraceStep(kind="error", label=f"Judge failed: {a.error_type} (answer unverified)")]
        label = f"Quality score: {a.quality_score:.2f} (threshold {self._settings.quality_threshold:.2f}) · verdict: {a.verdict}"
        if a.judge_confidence is not None:
            label += f" · judge confidence {a.judge_confidence:.2f}"
        if a.judge_model and a.judge_model != self._judge.model:
            label += f" · judged by fallback {a.judge_model}"
        steps.append(TraceStep(kind="score", label=label))
        if a.second_opinion:
            s = a.second_opinion
            detail = f"failed ({s.error})" if s.error else f"{s.verdict}, score {s.quality_score:.2f}"
            steps.append(TraceStep(kind="score", label=f"Second opinion from {s.judge_model}: {detail}"))
        return steps

    def _record(
        self,
        qid: str,
        query: str,
        mode: Mode,
        analysis: QueryAnalysis,
        attempts: list[Attempt],
        trace: list[TraceStep],
        start: float,
        budget: CallBudget,
    ) -> QueryRecord:
        final = next((a for a in attempts if a.accepted), None)
        status = "accepted"
        if final is None:
            final = _best_effort(attempts)
            status = "best_effort" if final else "failed"
            if final:
                score = f"{final.quality_score:.2f}" if final.quality_score is not None else "unscored"
                trace.append(TraceStep(kind="final", label=f"Best effort: no answer was accepted; returning the best candidate ({final.tier}, {score})"))
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
            judge_calls=sum(a.judge_calls for a in attempts),
            provider_calls=budget.used,
            billed_calls=dict(budget.billed),
            trace=trace,
        )


def _best_effort(attempts: list[Attempt]) -> Attempt | None:
    """No attempt was accepted: pick what to return.

    An answer that failed a deterministic check (e.g. its own assertion) is demonstrably wrong, so it ranks
    below any answer without such a failure, whatever the judge scored it. Among the rest: the highest judge
    score; unscored answers (judge unavailable) rank below scored ones, the latest (strongest tier) first.
    """
    answered = [a for a in attempts if a.answer is not None]
    if not answered:
        return None

    def rank(item: tuple[int, Attempt]) -> tuple[bool, bool, float, int]:
        index, a = item
        return (not a.failed_checks, a.quality_score is not None, a.quality_score or 0.0, index)

    return max(enumerate(answered), key=rank)[1]


def _route_step(decision: RouteDecision) -> TraceStep:
    return TraceStep(kind="route", label=f"{decision.tier.capitalize()} model selected ({decision.selected_model}): {decision.reason}")
