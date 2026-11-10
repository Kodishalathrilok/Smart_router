"""The benchmark saves each item as it finishes, stops cleanly when a quota runs out, and resumes without redoing work."""

import pytest

import benchmark.run as run_module
from app.core.bootstrap import build_runtime
from app.providers.base import LLMProvider, LLMResponse, RateLimitError
from benchmark.run import QUOTA_EXIT_CODE, main, parse_args, run
from benchmark.runners import ItemResult, QuotaStop, rate_limited
from tests.conftest import judge_json, make_settings

ARGS = ["--per-category", "1", "--no-reference-eval"]  # 5 items


class Metered(LLMProvider):
    """Answers until `allowance` generation calls have been made, then every generation is rate limited."""

    name = "mock"

    def __init__(self, allowance: int) -> None:
        self.allowance = allowance
        self.generations = 0

    async def generate(self, messages, model, *, temperature=0.2, max_tokens=1024, json_mode=False) -> LLMResponse:
        if json_mode:
            return LLMResponse(text=judge_json(0.95), model=model, provider=self.name)
        self.generations += 1
        if self.generations > self.allowance:
            raise RateLimitError("daily token quota exhausted", provider=self.name, retry_after=0)
        return LLMResponse(text="Final answer: 1", model=model, provider=self.name)

    async def list_models(self) -> list[str]:
        return [f"mock-{role}" for role in ("cheap", "mid", "strong", "judge")]


def patch_provider(monkeypatch, provider):
    async def build(settings, *, log_dir=None, providers=None):
        return await build_runtime(settings, log_dir=log_dir, providers={"mock": provider})

    monkeypatch.setattr(run_module, "build_runtime", build)


def lines(path):
    return [ItemResult.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def test_quota_stop_keeps_finished_work_and_resume_completes_it(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, max_retries=0)
    first = Metered(allowance=7)  # always_weak: 5 items, always_mid: 2 items, then the quota runs out
    patch_provider(monkeypatch, first)
    with pytest.raises(QuotaStop) as stop:
        await run(parse_args(ARGS, settings), settings)
    run_dir = settings.benchmark_dir / stop.value.run_id
    assert stop.value.system == "always_mid" and stop.value.done == 2 and stop.value.total == 5
    assert len(lines(run_dir / "always_weak.jsonl")) == 5  # a finished system is saved as soon as it finishes
    partial = lines(run_dir / "always_mid.partial.jsonl")
    assert len(partial) == 2 and not any(rate_limited(r.record) for r in partial)  # the rate-limited item is not kept
    assert not (settings.results_dir / "latest.json").exists()  # no summary from half a run

    second = Metered(allowance=1000)
    patch_provider(monkeypatch, second)
    summary = await run(parse_args([*ARGS, "--resume", stop.value.run_id], settings), settings)
    assert summary["run_id"] == stop.value.run_id
    assert second.generations >= 3 + 5 * 3  # the rest of always_mid and three more systems...
    assert second.generations < 5 * 4 + 5 * 3  # ...but nothing of always_weak, and not the two saved always_mid items
    assert not (run_dir / "always_mid.partial.jsonl").exists()
    for system in summary["systems"]:
        saved = lines(run_dir / f"{system}.jsonl")
        assert len(saved) == 5 and summary["systems"][system]["n"] == 5
    order = [r.item_id for r in lines(run_dir / "always_weak.jsonl")]
    assert [r.item_id for r in lines(run_dir / "always_mid.jsonl")] == order  # resumed files keep dataset order


def test_main_reports_progress_and_the_resume_command(tmp_path, monkeypatch, capsys):
    settings = make_settings(tmp_path, max_retries=0)
    patch_provider(monkeypatch, Metered(allowance=7))
    monkeypatch.setattr(run_module, "Settings", lambda: settings)
    assert main(ARGS) == QUOTA_EXIT_CODE
    err = capsys.readouterr().err
    assert "always_mid" in err and "2/5" in err and "--resume" in err and "--per-category 1" in err


def test_rate_limit_is_detected_on_generation_judge_and_second_opinion():
    from types import SimpleNamespace as NS

    ok = NS(error_type=None, second_opinion=None)
    assert rate_limited(NS(attempts=[ok])) is None
    assert rate_limited(NS(attempts=[ok, NS(error_type="RateLimitError", error="groq HTTP 429", second_opinion=None)]))
    assert rate_limited(NS(attempts=[NS(error_type=None, second_opinion=NS(error="RateLimitError: groq HTTP 429"))]))
    assert rate_limited(NS(attempts=[NS(error_type="ProviderServerError", error="503", second_opinion=NS(error=None))])) is None
