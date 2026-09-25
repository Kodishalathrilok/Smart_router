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
