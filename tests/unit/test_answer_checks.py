from app.verification.checks import check_answer, requested_names

Q = "Write a Python function `spiral_order(matrix: list[list[int]]) -> list[int]` returning the clockwise spiral."
RIGHT = '''```python
def spiral_order(matrix):
    out, m = [], [list(r) for r in matrix]
    while m and m[0]:
        out += m.pop(0)
        m = [list(r) for r in zip(*m)][::-1]
    return out
```
```python
>>> spiral_order([[1, 2], [3, 4]])
[1, 2, 4, 3]
```'''
WRONG = RIGHT.replace("[::-1]", "")  # still compiles, but no longer reproduces its own example


def statuses(results):
    return {c.name: c.status for c in results}


def test_requested_names_ignore_builtins():
    assert requested_names("Implement `roman_to_int(s)`; do not use `eval()` or `sorted(x)`.") == ["roman_to_int"]
    assert set(requested_names("Implement a Python class `LRUCache` with `get(key)`.")) == {"LRUCache", "get"}


def test_correct_code_passes_all_checks_including_its_own_examples():
    assert statuses(check_answer(Q, RIGHT, execute_examples=True, timeout_s=10)) == {"syntax": "pass", "requested_names": "pass", "self_examples": "pass"}


def test_wrong_code_fails_its_own_examples():
    result = statuses(check_answer(Q, WRONG, execute_examples=True, timeout_s=10))
    assert result == {"syntax": "pass", "requested_names": "pass", "self_examples": "fail"}


def test_examples_not_executed_unless_enabled():
    assert "self_examples" not in statuses(check_answer(Q, WRONG, execute_examples=False, timeout_s=10))


def test_syntax_error_and_missing_function_fail():
    assert statuses(check_answer(Q, "```python\ndef spiral_order(m)\n    return m\n```", execute_examples=False, timeout_s=10)) == {"syntax": "fail"}
    assert statuses(check_answer(Q, "```python\ndef spiral(m):\n    return m\n```", execute_examples=False, timeout_s=10))["requested_names"] == "fail"
    assert statuses(check_answer(Q, "Just walk the edges clockwise.", execute_examples=False, timeout_s=10)) == {"code_block": "fail"}


def test_not_applicable_cases_return_no_checks():
    assert check_answer("What is the capital of France?", "Paris.", execute_examples=True, timeout_s=10) == []
    assert check_answer("Write `sum_list(xs)` in JavaScript.", "```\nfunction sum_list(xs) {}\n```", execute_examples=True, timeout_s=10) == []


def test_docstring_examples_count_as_source_and_as_examples():
    answer = '```python\ndef double(x):\n    """\n    >>> double(2)\n    4\n    """\n    return x * 2\n```'
    assert statuses(check_answer("Write `double(x)`.", answer, execute_examples=True, timeout_s=10)) == {
        "syntax": "pass", "requested_names": "pass", "self_examples": "pass"}


def test_non_terminating_examples_time_out_as_failure():
    answer = "```python\ndef spin():\n    while True:\n        pass\n```\n```python\n>>> spin()\n```"
    result = check_answer("Write `spin()`.", answer, execute_examples=True, timeout_s=2)
    assert statuses(result)["self_examples"] == "fail" and "did not finish" in result[-1].detail



def test_unlabeled_non_python_block_is_not_a_syntax_failure():
    """Seen in the benchmark: a correct answer opened with an unlabeled grammar sketch."""
    answer = "```\nexpr := term ((\u2018+\u2019|\u2018-\u2019) term)*\n```\n```python\ndef evaluate(expr):\n    return 0.0\n```"
    assert statuses(check_answer("Write `evaluate(expr: str) -> float`.", answer, execute_examples=False, timeout_s=10)) == {
        "syntax": "pass", "requested_names": "pass"}


def test_labeled_python_block_that_does_not_parse_fails():
    answer = "```python\ndef evaluate(expr)\n    return 0\n```"
    assert statuses(check_answer("Write `evaluate(expr)`.", answer, execute_examples=False, timeout_s=10)) == {"syntax": "fail"}
