"""Capability (accuracy among readable answers) is reported separately from plumbing failures."""

import json
from types import SimpleNamespace as NS

import benchmark.run as run_module
from benchmark.analysis import capability, plumbing_failure
from benchmark.report import render
from tests.conftest import make_settings


def result(correct, *, answer="Final answer: 1", finish="stop", error_type=None, error=None, checkable=True):
    attempt = NS(answer=answer, finish_reason=finish if answer else None, error_type=error_type, error=error)
    return NS(checkable=checkable, deterministic_correct=correct, record=NS(answer=answer, attempts=[attempt]))


RIGHT = result(True)
WRONG = result(False)
EMPTY = result(False, answer=None, error_type="EmptyResponseError", error="groq returned an empty completion (finish_reason=stop)")
CUT_EMPTY_OLD = result(False, answer=None, error_type="EmptyResponseError", error="groq returned an empty completion (finish_reason=length)")
CUT_EMPTY = result(False, answer=None, error_type="TruncatedEmptyResponseError", error="x")
CUT_ANSWER = result(None, finish="length")
CUT_GEMINI = result(False, finish="MAX_TOKENS")
UNREADABLE = result(None)
PROVIDER = result(False, answer=None, error_type="ProviderServerError", error="nvidia HTTP 503")


def test_each_plumbing_failure_is_classified():
    assert [plumbing_failure(r) for r in (RIGHT, WRONG)] == [None, None]
    assert plumbing_failure(EMPTY) == "empty_reply"
    assert [plumbing_failure(r) for r in (CUT_EMPTY_OLD, CUT_EMPTY, CUT_ANSWER, CUT_GEMINI)] == ["truncation"] * 4
    assert plumbing_failure(UNREADABLE) == "unreadable_final_number"
    assert plumbing_failure(PROVIDER) == "provider_error"


def test_a_correct_answer_is_never_a_plumbing_failure_even_if_cut_off():
    assert plumbing_failure(result(True, finish="length")) is None


def test_the_final_answer_decides_not_an_earlier_failed_attempt():
    failed = NS(answer=None, finish_reason=None, error_type="TruncatedEmptyResponseError", error="x")
    final = NS(answer="Final answer: 2", finish_reason="stop", error_type=None, error=None)
    escalated = NS(checkable=True, deterministic_correct=False, record=NS(answer="Final answer: 2", attempts=[failed, final]))
    assert plumbing_failure(escalated) is None


def test_capability_figures():
    c = capability([RIGHT, RIGHT, RIGHT, WRONG, EMPTY, CUT_EMPTY, CUT_ANSWER, UNREADABLE, PROVIDER, result(None, checkable=False)])
    assert (c["checked"], c["plumbing_failures"], c["readable"], c["readable_correct"]) == (9, 5, 4, 3)
    assert c["plumbing_rate"] == round(5 / 9, 4)
    assert c["readable_accuracy"] == 0.75  # strict accuracy on the same results is 3/9
    assert c["by_kind"] == {"empty_reply": 1, "truncation": 2, "unreadable_final_number": 1, "provider_error": 1}


def test_capability_with_nothing_readable_or_checkable():
    assert capability([EMPTY])["readable_accuracy"] is None
    assert capability([])["plumbing_rate"] is None


async def test_report_shows_both_accuracies_and_the_plumbing_rate(tmp_path):
    settings = make_settings(tmp_path)
    args = run_module.parse_args(["--per-category", "1", "--no-reference-eval"], settings)
    summary = await run_module.run(args, settings)
    system = summary["systems"]["smart_router"]
    assert system["capability"]["checked"] == 0  # mock runs have no exact checks
    system["strict"] = {"checked": 30, "correct": 27, "accuracy": 0.9, "ci95": [0.74, 0.97], "unparseable": 1}
    system["capability"] = {"checked": 30, "plumbing_failures": 2, "plumbing_rate": 0.0667, "by_kind": {"truncation": 2},
                            "readable": 28, "readable_correct": 27, "readable_accuracy": 0.9643}
    text = render(json.loads(json.dumps(summary)))
    assert "Acc. among readable" in text and "Plumbing failures" in text
    assert "96.4%" in text and "6.7% (2/30)" in text
