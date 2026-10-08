import pytest

from app.core.executor import Executor
from app.judge.judge import JudgeParseError, QualityJudge, parse_judgement, quality_score
from app.judge.schemas import JudgeDimensions
from app.models.schemas import ModelSpec
from app.providers.base import LLMResponse, ProviderServerError
from tests.conftest import FakeProvider, judge_json

VALID = judge_json(0.9)


def test_parses_plain_fenced_and_embedded_json():
    assert parse_judgement(VALID).correctness == 0.9
    assert parse_judgement(f"```json\n{VALID}\n```").verdict == "good"
    assert parse_judgement(f"Here is my evaluation: {VALID} Thanks!").relevance == 0.9


def test_garbage_raises():
    with pytest.raises(JudgeParseError):
        parse_judgement("The answer looks fine to me.")
    with pytest.raises(JudgeParseError):
        parse_judgement('{"verdict": "good"}')  # missing dimensions


@pytest.mark.parametrize("bad", ["42", "8", "1.01", "-0.1", "NaN", "Infinity", '"0.9"'])
def test_out_of_range_or_non_numeric_scores_are_rejected_never_rescaled(bad):
    # Regression: a score of 8 used to be silently read as 0.8 (guessing a 0-10 scale).
    with pytest.raises(JudgeParseError):
        parse_judgement(VALID.replace('"correctness": 0.9', f'"correctness": {bad}'))


def test_boundary_scores_are_accepted():
    assert parse_judgement(VALID.replace('"correctness": 0.9', '"correctness": 1')).correctness == 1.0
    assert parse_judgement(VALID.replace('"correctness": 0.9', '"correctness": 0')).correctness == 0.0


def test_normalizes_verdict_and_reasons():
    raw = VALID.replace('"good"', '"Partially Correct"').replace('"failure_reasons": []', '"failure_reasons": ["Reasoning", "made up"]')
    dims = parse_judgement(raw)
    assert dims.verdict == "partially_correct"
    assert dims.failure_reasons == ["reasoning_error", "other"]


def dims(c, comp, r, rq):
    return JudgeDimensions(verdict="good", correctness=c, completeness=comp, relevance=r, reasoning_quality=rq)


def test_quality_aggregation_default_and_category_weights():
    assert quality_score(dims(1, 0, 0, 0), "general_qa") == 0.40
    assert quality_score(dims(0, 1, 0, 0), "general_qa") == 0.25
    assert quality_score(dims(0, 0, 1, 0), "general_qa") == 0.20
    assert quality_score(dims(0, 0, 0, 1), "general_qa") == 0.15
    assert quality_score(dims(1, 0, 0, 0), "coding") == 0.50
    assert quality_score(dims(1, 1, 1, 1), "math") == 1.0


class RawJudge(FakeProvider):
    """Returns queued raw strings for judge calls."""

    def __init__(self, outputs):
        super().__init__({})
        self.outputs = outputs

    async def generate(self, messages, model, **kw):
        self.answers[model] = self.outputs.pop(0)
        return await super().generate(messages, model)


def make_judge(registry, outputs):
    executor = Executor({"mock": RawJudge(outputs)}, max_retries=0, base_delay_s=0, max_delay_s=0)
    return QualityJudge(executor, [registry.judge], max_tokens=256)


async def test_judge_retries_malformed_output_once(registry):
    result = await make_judge(registry, ["not json", VALID]).evaluate("q", "a", "general_qa")
    assert result.quality_score == 0.9 and result.judge_model == "mock-judge"
    assert result.correctness == 0.9  # dimensions stored alongside aggregate


async def test_judge_gives_up_after_retries(registry):
    with pytest.raises(JudgeParseError):
        await make_judge(registry, ["nope", "still nope"]).evaluate("q", "a", "general_qa")


class PerModelJudge(FakeProvider):
    """Judge outputs keyed by judge model id: a raw string, or an Exception to raise."""

    async def generate(self, messages, model, **kw):
        self.calls.append((model, True))
        out = self.answers[model]
        if isinstance(out, Exception):
            raise out
        return LLMResponse(text=out, model=model, provider="mock")


def fallback_judge(outputs):
    primary = ModelSpec(id="primary", provider="mock", role="judge", capability_score=0.9, normalized_cost=3, expected_latency="medium")
    backup = primary.model_copy(update={"id": "backup", "role": "judge_fallback"})
    provider = PerModelJudge(outputs)
    executor = Executor({"mock": provider}, max_retries=0, base_delay_s=0, max_delay_s=0)
    return QualityJudge(executor, [primary, backup], max_tokens=256), provider


async def test_primary_judge_used_when_healthy():
    judge, provider = fallback_judge({"primary": VALID, "backup": VALID})
    result = await judge.evaluate("q", "a", "general_qa")
    assert result.judge_model == "primary" and [m for m, _ in provider.calls] == ["primary"]


async def test_fallback_judge_on_provider_error():
    judge, provider = fallback_judge({"primary": ProviderServerError("503", provider="mock"), "backup": VALID})
    result = await judge.evaluate("q", "a", "general_qa")
    assert result.judge_model == "backup" and result.quality_score == 0.9
    assert [m for m, _ in provider.calls] == ["primary", "backup"]


async def test_fallback_judge_on_malformed_output():
    judge, provider = fallback_judge({"primary": "not json", "backup": VALID})
    result = await judge.evaluate("q", "a", "general_qa")
    assert result.judge_model == "backup"
    assert [m for m, _ in provider.calls] == ["primary", "primary", "backup"]  # one parse retry, then fallback


async def test_raises_when_every_judge_fails():
    judge, _ = fallback_judge({"primary": ProviderServerError("503", provider="mock"), "backup": "garbage"})
    with pytest.raises(JudgeParseError):
        await judge.evaluate("q", "a", "general_qa")
