"""Dollar cost is computed from stored tokens and list prices; an unpriced model makes the cost unknown, never partial."""

import json
from types import SimpleNamespace as NS

import httpx
import pytest

import benchmark.run as run_module
from app.core.executor import Executor, call_budget
from app.core.metrics import record_cost_usd
from app.providers.base import LLMProvider, LLMResponse, TruncatedEmptyResponseError
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.schemas.common import TokenUsage
from benchmark.report import render, usd
from benchmark.run import GENERATION_ROLES, dollar_cost, price_table
from tests.conftest import FakeProvider, make_settings

PRICES = {
    "cheap": {"model": "small", "input": 0.10, "output": 0.50},
    "strong": {"model": "big", "input": None, "output": None},
    "judge": {"model": "grader", "input": 0.30, "output": 2.50},
}


def record(*attempt_usages, recorded=True):
    return NS(usage_recorded=recorded, attempts=[NS(usage=u) for u in attempt_usages])


def use(model, input_tokens, output_tokens):
    return TokenUsage(model=model, input_tokens=input_tokens, output_tokens=output_tokens)


def test_cost_sums_every_role_of_every_attempt():
    r = record({"cheap": use("small", 1_000_000, 2_000_000), "judge": use("grader", 1_000_000, 0)}, {"cheap": use("small", 0, 1_000_000)})
    assert record_cost_usd(r, PRICES) == pytest.approx(0.10 + 1.00 + 0.30 + 0.50)


def test_roles_filter_leaves_out_the_judge_for_baselines():
    r = record({"cheap": use("small", 1_000_000, 0), "judge": use("grader", 1_000_000, 1_000_000)})
    assert record_cost_usd(r, PRICES, roles=GENERATION_ROLES) == pytest.approx(0.10)


def test_null_price_makes_the_record_unknown_not_a_partial_sum():
    r = record({"cheap": use("small", 1_000_000, 0)}, {"strong": use("big", 10, 10)})
    assert record_cost_usd(r, PRICES) is None


def test_price_only_applies_to_the_model_it_was_looked_up_for():
    assert record_cost_usd(record({"cheap": use("some-other-model", 10, 10)}, recorded=True), PRICES) is None
    assert record_cost_usd(record({"mid": use("x", 10, 10)}), PRICES) is None  # role without a price entry


def test_records_made_before_tokens_were_stored_are_unknown():
    assert record_cost_usd(record({}, recorded=False), PRICES) is None


def test_an_attempt_that_used_no_tokens_costs_nothing():
    assert record_cost_usd(record({}), PRICES) == 0.0  # e.g. the provider returned 503 before billing anything


def test_system_dollars_per_1000_queries_or_unknown():
    cheap = record({"cheap": use("small", 1000, 1000)})
    out = dollar_cost([NS(record=cheap), NS(record=cheap)], PRICES, roles=None)
    assert out == {"per_1k_queries": pytest.approx(0.6), "priced_queries": 2, "queries": 2}
    mixed = dollar_cost([NS(record=cheap), NS(record=record({"strong": use("big", 1, 1)}))], PRICES, roles=None)
    assert mixed == {"per_1k_queries": None, "priced_queries": 1, "queries": 2}
    assert usd(out) == "$0.60" and usd(mixed) == "unknown" and usd(None) == "unknown"


def test_price_table_reads_models_json_shape():
    table = price_table({"cheap": {"price_model": "small", "price_per_mtok_input": 0.1, "price_per_mtok_output": 0.5}, "mid": {}})
    assert table == {"cheap": {"model": "small", "input": 0.1, "output": 0.5}, "mid": {"model": None, "input": None, "output": None}}


def test_committed_prices_name_their_model_and_source():
    profiles = json.loads((run_module.DATASETS_DIR.parents[1] / "config" / "models.json").read_text(encoding="utf-8"))
    assert "http" in profiles["_comment"]
    for role, p in profiles.items():
        if isinstance(p, dict) and p["price_per_mtok_input"] is not None:
            assert p["price_model"] and p["price_per_mtok_output"] is not None, role


class Billing(LLMProvider):
    """First call: truncated empty reply that still reports usage. Second call: an answer."""

    name = "mock"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, messages, model, *, temperature=0.2, max_tokens=1024, json_mode=False) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            error = TruncatedEmptyResponseError("empty (finish_reason=length)", provider="mock")
            error.input_tokens, error.output_tokens = 50, 1024
            raise error
        return LLMResponse(text="42", model=model, provider="mock", input_tokens=50, output_tokens=7)

    async def list_models(self) -> list[str]:
        return []


async def test_executor_records_tokens_of_failed_calls_that_returned_usage(registry):
    executor = Executor({"mock": Billing()}, max_retries=0, base_delay_s=0, max_delay_s=0, truncation_retry_max_tokens=4096)
    with call_budget(5) as budget:
        await executor.generate(registry.for_tier("cheap"), [{"role": "user", "content": "q"}], temperature=0, max_tokens=1024)
    assert budget.tokens == {"cheap": {"model": "mock-cheap", "input_tokens": 100, "output_tokens": 1031}}


async def test_adapters_attach_usage_to_empty_reply_errors():
    groq = OpenAICompatibleProvider("groq", "k", base_url="https://groq.test/v1", timeout=5, transport=httpx.MockTransport(lambda r: httpx.Response(
        200, json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}], "usage": {"prompt_tokens": 9, "completion_tokens": 1024}})))
    gemini = GeminiProvider("k", base_url="https://gemini.test/v1beta", timeout=5, transport=httpx.MockTransport(lambda r: httpx.Response(
        200, json={"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}],
                   "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 0, "thoughtsTokenCount": 1000}})))
    for provider, expected in ((groq, (9, 1024)), (gemini, (9, 1000))):
        with pytest.raises(TruncatedEmptyResponseError) as info:
            await provider.generate([{"role": "user", "content": "hi"}], "m")
        assert (info.value.input_tokens, info.value.output_tokens) == expected


async def test_gemini_output_tokens_include_thinking_because_it_is_billed():
    gemini = GeminiProvider("k", base_url="https://gemini.test/v1beta", timeout=5, transport=httpx.MockTransport(lambda r: httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                   "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "thoughtsTokenCount": 30}})))
    assert (await gemini.generate([{"role": "user", "content": "hi"}], "g")).output_tokens == 32


async def test_pipeline_stores_generation_judge_and_second_opinion_tokens_per_attempt(tmp_path):
    from app.core.bootstrap import build_runtime
    from tests.integration.test_pipeline import ANSWERS, EASY, TwoJudges

    provider = TwoJudges({**ANSWERS, "mock-backup": ""}, {"ANS-cheap": {"mock-judge": 0.25, "mock-backup": 0.95}})
    runtime = await build_runtime(make_settings(tmp_path, judge_fallback_model="mock-backup"), providers={"mock": provider})
    r = await runtime.pipeline.run(EASY)
    assert r.usage_recorded and len(r.attempts) == 1
    expected = {"input_tokens": 10, "output_tokens": 20}
    assert {role: u.model_dump() for role, u in r.attempts[0].usage.items()} == {
        "cheap": {"model": "mock-cheap", **expected},
        "judge": {"model": "mock-judge", **expected},
        "judge_fallback": {"model": "mock-backup", **expected},  # the second opinion came from the other judge
    }


async def test_failed_generation_attempt_keeps_its_usage(tmp_path):
    from app.core.bootstrap import build_runtime

    error = TruncatedEmptyResponseError("empty (finish_reason=length)", provider="mock")
    error.input_tokens, error.output_tokens = 12, 4096
    provider = FakeProvider({"mock-cheap": error, "mock-mid": "ANS-mid", "mock-strong": "x", "mock-judge": ""}, {"ANS-mid": 0.95})
    runtime = await build_runtime(make_settings(tmp_path, truncation_retry_max_tokens=0), providers={"mock": provider})
    r = await runtime.pipeline.run("Who wrote Hamlet?")
    assert r.attempts[0].answer is None and r.attempts[0].usage["cheap"].output_tokens == 4096


async def test_report_shows_dollars_beside_normalized_units_and_unknown_for_mock(tmp_path):
    settings = make_settings(tmp_path)
    summary = await run_module.run(run_module.parse_args(["--per-category", "1", "--no-reference-eval"], settings), settings)
    assert summary["systems"]["smart_router"]["cost_usd"]["per_1k_queries"] is None  # mock models have no price
    text = render(json.loads(json.dumps(summary)))
    assert "$ / 1k queries" in text and "unknown" in text and "Cost incl. judge" in text
    summary["systems"]["smart_router"]["cost_usd"] = {"per_1k_queries": 1.2345, "priced_queries": 5, "queries": 5}
    assert "$1.23" in render(summary)
