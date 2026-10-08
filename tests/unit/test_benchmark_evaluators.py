from benchmark.dataset import BenchmarkItem, load_dataset
from benchmark.evaluators import check_numeric, deterministic_check, extract_number, run_code_tests


def test_dataset_is_valid_and_balanced():
    items = load_dataset()
    assert len(items) == 100
    for category in ("general_qa", "reasoning", "coding", "math", "summarization"):
        assert sum(i.category == category for i in items) == 20
    assert all(i.expected is not None for i in items if i.answer_type == "numeric")
    assert all(i.tests for i in items if i.answer_type == "code")


def test_extract_number():
    assert extract_number("So the total is 12, but wait... Final answer: 1,157.63") == 1157.63
    assert extract_number("The ball costs $0.05.") == 0.05
    assert extract_number("It is 391.") == 391
    assert extract_number("no numbers") is None
    assert extract_number("**Final answer: 34 650**") == 34650
    assert extract_number("Final answer: 34 650") == 34650
    assert extract_number("It is 1,157.63 dollars") == 1157.63


def test_check_numeric_tolerance():
    assert check_numeric("Final answer: 32.4", 32.4)
    assert not check_numeric("Final answer: 32.5", 32.4)
    assert check_numeric("about 29.56", 29.5562, tolerance=0.01)


def test_code_tests_pass_and_fail():
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert run_code_tests(good, "assert add(2, 3) == 5")
    assert not run_code_tests(bad, "assert add(2, 3) == 5")
    assert not run_code_tests("no code here", "assert True")


def test_reference_solutions_for_code_items_are_consistent():
    """Every code item's tests must at least be valid Python."""
    for item in load_dataset():
        if item.tests:
            compile(item.tests, item.id, "exec")


def test_deterministic_check_dispatch():
    numeric = BenchmarkItem(id="m", category="math", difficulty="easy", question="q", reference_answer="r", answer_type="numeric", expected=5)
    code = BenchmarkItem(id="c", category="coding", difficulty="easy", question="q", reference_answer="r", answer_type="code", tests="assert True")
    free = BenchmarkItem(id="f", category="general_qa", difficulty="easy", question="q", reference_answer="r")
    assert deterministic_check(numeric, "x = 5", exec_code=False) is True
    assert deterministic_check(numeric, None, exec_code=False) is False
    assert deterministic_check(code, "```python\npass\n```", exec_code=False) is None
    assert deterministic_check(code, "```python\npass\n```", exec_code=True) is True
    assert deterministic_check(free, "anything", exec_code=True) is None


def test_subset_spans_difficulties():
    from benchmark.run import select_items

    subset = select_items(load_dataset(), 3)
    assert len(subset) == 15
    for category in ("general_qa", "math", "coding"):
        labels = [i.difficulty for i in subset if i.category == category]
        assert labels[0] == "easy" and labels[-1] == "hard"


def test_hard_dataset_answers_verify_against_their_own_references():
    """Every hard item is deterministically checkable, and its reference passes its own check."""
    from benchmark.dataset import DATASETS_DIR

    items = load_dataset(DATASETS_DIR / "hard.jsonl")
    assert len(items) == 30 and all(i.difficulty == "hard" for i in items)
    for item in items:
        assert item.answer_type in ("numeric", "code"), item.id
        assert deterministic_check(item, item.reference_answer, exec_code=True) is True, item.id


def test_tier_accuracy_section_flags_inverted_tiers():
    from benchmark.report import tier_accuracy_section

    ok = {"a": {"tier_accuracy": {"cheap": {"correct": 7, "checked": 10}, "strong": {"correct": 9, "checked": 10}}}}
    bad = {"a": {"tier_accuracy": {"cheap": {"correct": 9, "checked": 10}, "mid": {"correct": 5, "checked": 10}}}}
    assert "Tier order check failed" not in tier_accuracy_section(ok)
    assert "Tier order check failed" in tier_accuracy_section(bad)



def test_ambiguous_final_number_is_not_checkable_instead_of_guessed():
    from benchmark.evaluators import check_numeric, read_final_number

    assert read_final_number("There are 204 squares of all sizes on a standard 8 x 8 chessboard.") == ("ambiguous", None)
    assert check_numeric("There are 204 squares of all sizes on a standard 8 x 8 chessboard.", 204) is None
    assert check_numeric(r"so the count is \[ \boxed{15} \]", 15) is True
    assert check_numeric("I cannot determine this.", 15) is False


def test_escalation_breakdown_and_judge_confusion():
    from types import SimpleNamespace

    from app.schemas.common import Attempt
    from benchmark.analysis import escalation_breakdown, judge_confusion

    def attempt(tier, q, verdict, accepted):
        return Attempt(number=1, model=tier, provider="p", tier=tier, routing_reason="r", answer="a", quality_score=q, verdict=verdict, accepted=accepted)

    record = lambda attempts: SimpleNamespace(attempts=attempts, quality_threshold=0.78)  # noqa: E731
    results = [
        # correct cheap answer, falsely rejected by both -> unnecessary escalation
        SimpleNamespace(item_id="a", record=record([attempt("cheap", 0.2, "incorrect", False), attempt("mid", 1.0, "good", True)]),
                        attempt_correct=[True, True], deterministic_correct=True),
        # wrong cheap answer, rightly rejected -> necessary escalation
        SimpleNamespace(item_id="b", record=record([attempt("cheap", 0.2, "incorrect", False), attempt("mid", 1.0, "good", True)]),
                        attempt_correct=[False, True], deterministic_correct=True),
        # wrong answer, judge accepts -> false accept; primary false reject rescued by decision -> decision agrees
        SimpleNamespace(item_id="c", record=record([attempt("cheap", 1.0, "good", True)]), attempt_correct=[False], deterministic_correct=False),
        SimpleNamespace(item_id="d", record=record([attempt("cheap", 0.2, "incorrect", True)]), attempt_correct=[True], deterministic_correct=True),
    ]
    assert escalation_breakdown(results) == {"total": 2, "necessary": 1, "unnecessary": 1, "unknown": 0}
    assert judge_confusion(results, decision=False) == {"checked": 6, "agree": 3, "false_accept": 1, "false_reject": 2}
    assert judge_confusion(results, decision=True) == {"checked": 6, "agree": 4, "false_accept": 1, "false_reject": 1}


def test_integer_answers_must_match_exactly():
    from benchmark.evaluators import check_numeric

    # Regression: the relative 1e-4 default accepted any value within +-7 of 76127.
    assert check_numeric("Final answer: 76120", 76127) is False
    assert check_numeric("Final answer: 76,127", 76127) is True
    assert check_numeric("Final answer: 0.70631", 0.7063) is True  # non-integers keep the relative tolerance


def test_hard_v2_dataset_is_reproducible_and_self_consistent():
    import json

    from benchmark.build_hard_v2 import OUT, build

    assert [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines()] == build()  # file == generator output
    items = load_dataset(OUT)
    assert len(items) == 22 and {i.answer_type for i in items} == {"numeric", "code"}
    for item in items:
        assert deterministic_check(item, item.reference_answer, exec_code=True) is True, item.id
