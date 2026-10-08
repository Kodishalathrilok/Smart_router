"""Executing the answer's own Python (opt-in) and classifying the outcome.

Each case says what evidence is expected and whether it should escalate (status "fail") or not.
"""

import os

from app.verification.checks import check_answer, normalize_examples

Q = "Write a Python function `evaluate(expr: str) -> float` that evaluates arithmetic."


def run(answer: str, query: str = Q, execute: bool = True, timeout_s: float = 10):
    return {c.name: (c.status, c.evidence) for c in check_answer(query, answer, execute_examples=execute, timeout_s=timeout_s)}


def py(code: str) -> str:
    return f"```python\n{code}\n```"


GOOD_FN = "def evaluate(expr):\n    return float(eval(expr, {'__builtins__': {}}))\n"


# ---- the discovered failure: a module-level self-test with a wrong expectation ----

def test_module_level_self_test_failure_is_assertion_failed():
    """coding_hard_03: correct function, but its own __main__ test expects 9.5 for an expression worth 7.0."""
    answer = py(GOOD_FN + '\nif __name__ == "__main__":\n    got = evaluate("2.5 * 4 - 1.5 / .5")\n'
                '    assert got == 9.5, f"{got}, expected 9.5"\n')
    result = run(answer)
    assert result["module_execution"] == ("fail", "assertion_failed")


def test_module_that_crashes_when_run_is_module_execution_failed():
    answer = py(GOOD_FN + '\nprint(evaluate(3 + 4))  # passes an int, not a string\n')
    assert run(answer)["module_execution"] == ("fail", "module_execution_failed")


def test_valid_code_without_any_executable_test_is_not_penalized():
    result = run(py(GOOD_FN))
    assert result == {"syntax": ("pass", "passed"), "requested_names": ("pass", "passed")}  # nothing to run, nothing claimed


def test_valid_code_with_a_passing_self_test():
    answer = py(GOOD_FN + '\nif __name__ == "__main__":\n    assert evaluate("2 + 3 * 4") == 14\n')
    assert run(answer)["module_execution"] == ("pass", "passed")


def test_passing_and_failing_doctest_examples():
    ok = py(GOOD_FN) + "\n```python\n>>> evaluate('2 + 3 * 4')\n14.0\n```"
    bad = py(GOOD_FN) + "\n```python\n>>> evaluate('2 + 3 * 4')\n20.0\n```"
    assert run(ok)["self_examples"] == ("pass", "passed")
    assert run(bad)["self_examples"] == ("fail", "self_example_failed")


def test_unlabeled_non_python_block_is_neither_a_syntax_failure_nor_executed():
    answer = "```\nexpr := term (('+'|'-') term)*\n```\n" + py(GOOD_FN)
    assert run(answer)["syntax"] == ("pass", "passed")
    unlabeled_only = "```\nraise SystemError('would crash if executed')\n```"
    assert "module_execution" not in run(unlabeled_only, query="Show an example.")


def test_nothing_executes_unless_enabled():
    answer = py(GOOD_FN + "\nassert False\n")
    assert "module_execution" not in run(answer, execute=False)


# ---- context-dependent snippets must not become false failures ----

def test_snippet_referencing_missing_context_is_not_checkable():
    answer = py(GOOD_FN + "\nresult = evaluate(user_expression)\n")
    assert run(answer)["module_execution"] == ("not_checkable", "not_checkable")


def test_missing_third_party_import_is_not_checkable():
    answer = py("import numpy_that_is_not_installed\n" + GOOD_FN)
    assert run(answer)["module_execution"] == ("not_checkable", "not_checkable")


def test_interactive_input_is_not_checkable():
    answer = py(GOOD_FN + "\nprint(evaluate(input('expr: ')))\n")
    assert run(answer)["module_execution"] == ("not_checkable", "not_checkable")


def test_deliberate_error_demonstration_is_not_checkable():
    answer = py(GOOD_FN + "\nevaluate(None)  # raises TypeError for non-string input\n")
    assert run(answer)["module_execution"] == ("not_checkable", "not_checkable")


def test_long_running_module_is_not_checkable_and_bounded():
    answer = py(GOOD_FN + "\nwhile True:\n    pass\n")
    assert run(answer, timeout_s=2)["module_execution"] == ("not_checkable", "not_checkable")


def test_sys_exit_zero_is_fine_and_nonzero_is_not_checkable():
    assert run(py(GOOD_FN + "\nimport sys\nsys.exit(0)\n"))["module_execution"] == ("pass", "passed")
    assert run(py(GOOD_FN + "\nimport sys\nsys.exit(2)\n"))["module_execution"] == ("not_checkable", "not_checkable")


# ---- isolation ----

def test_executed_code_cannot_see_api_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-must-not-leak")
    answer = py(GOOD_FN + "\nimport os\nassert 'GROQ_API_KEY' not in os.environ\n")
    assert run(answer)["module_execution"] == ("pass", "passed")
    assert os.environ["GROQ_API_KEY"] == "sk-must-not-leak"


def test_network_is_blocked_and_reported_as_not_checkable():
    answer = py(GOOD_FN + "\nimport socket\nsocket.create_connection(('example.com', 80), timeout=1)\n")
    assert run(answer)["module_execution"] == ("not_checkable", "not_checkable")


# ---- requested function signature ----

def test_signature_mismatch_is_reported():
    answer = py("def evaluate(expr, variables, precision):\n    return 0.0\n")
    assert run(answer, execute=False)["requested_names"] == ("fail", "function_signature_mismatch")


def test_compatible_signatures_pass():
    assert run(py("def evaluate(expr, precision=2):\n    return 0.0\n"), execute=False)["requested_names"] == ("pass", "passed")
    assert run(py("def evaluate(*args):\n    return 0.0\n"), execute=False)["requested_names"] == ("pass", "passed")
    method_q = "Implement class `Cache` with `get(self, key: int) -> int`."
    method = py("class Cache:\n    def get(self, key):\n        return -1\n")
    assert run(method, query=method_q, execute=False)["requested_names"] == ("pass", "passed")


def test_missing_function_is_function_missing():
    assert run(py("def calc(expr):\n    return 0.0\n"), execute=False)["requested_names"] == ("fail", "function_missing")


# ---- example normalization (false-positive guards found in real answers) ----

def test_trailing_comment_in_expected_output_is_ignored():
    answer = py("def evaluate(expr):\n    return [1, 2, 3]\n") + "\n```python\n>>> evaluate('x')\n[1, 2, 3]   # sorted order\n```"
    assert run(answer)["self_examples"] == ("pass", "passed")


def test_example_that_announces_an_error_without_traceback_is_skipped():
    text = ">>> evaluate('')   # raises ValueError\n>>> evaluate('1+1')\n2.0\n"
    normalized = normalize_examples(text)
    assert "raises" not in normalized and ">>> evaluate('1+1')" in normalized


def test_equal_literal_output_is_accepted():
    answer = py("def evaluate(expr):\n    return 2.5\n") + "\n```python\n>>> evaluate('5/2')\n2.50\n```"
    assert run(answer)["self_examples"] == ("pass", "passed")
