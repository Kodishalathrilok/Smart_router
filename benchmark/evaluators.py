"""Deterministic, judge-independent checks: numeric answers and executable code tests."""

from app.verification.checks import extract_code, run_isolated
from app.verification.numbers import extract_number, read_final_number  # noqa: F401  (re-used here and by tests)
from benchmark.dataset import BenchmarkItem

DEFAULT_REL_TOL = 1e-4
CODE_TIMEOUT_S = 10


def check_numeric(answer: str, expected: float, tolerance: float | None = None) -> bool | None:
    """True/False, or None when the answer's final number is ambiguous (not checkable)."""
    status, value = read_final_number(answer)
    if status == "ambiguous":
        return None
    if value is None:
        return False
    if tolerance is not None:
        allowed = tolerance
    elif float(expected).is_integer():
        allowed = 0.0  # integer answers are exact: a relative 1e-4 would accept 76120 for 76127
    else:
        allowed = DEFAULT_REL_TOL * max(1.0, abs(expected))
    return abs(value - expected) <= allowed


def run_code_tests(answer: str, tests: str, timeout_s: float = CODE_TIMEOUT_S) -> bool:
    """Execute model-written code + asserts in an isolated subprocess.

    SECURITY: this runs untrusted model output on your machine. It is opt-in (`--exec-code`), bounded by a
    timeout, and uses the verifier's runner (no API keys in the environment, Python-level sockets patched,
    resource limits on Linux). It is NOT a sandbox: see README "Security model".
    """
    code = extract_code(answer)
    if code is None:
        return False
    return run_isolated(f"{code}\n\n{tests}\n", timeout_s=timeout_s).returncode == 0


def is_checkable(item: BenchmarkItem, *, exec_code: bool) -> bool:
    """Whether this item gets a deterministic verdict in a run (the denominator of strict accuracy)."""
    return (item.answer_type == "numeric" and item.expected is not None) or (item.answer_type == "code" and bool(item.tests) and exec_code)


def deterministic_check(item: BenchmarkItem, answer: str | None, *, exec_code: bool) -> bool | None:
    """True/False when the item supports a deterministic check, None when it does not."""
    if item.answer_type == "numeric" and item.expected is not None:
        return check_numeric(answer, item.expected, item.tolerance) if answer else False
    if item.answer_type == "code" and item.tests and exec_code:
        return bool(answer) and run_code_tests(answer, item.tests)
    return None
