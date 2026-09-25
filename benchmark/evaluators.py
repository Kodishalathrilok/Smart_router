"""Deterministic, judge-independent checks: numeric answers and executable code tests."""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmark.dataset import BenchmarkItem

FINAL_ANSWER = re.compile(r"final answer\s*[:=]?\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", re.I)
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")
CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)
DEFAULT_REL_TOL = 1e-4
CODE_TIMEOUT_S = 10


def extract_number(text: str) -> float | None:
    """Prefer an explicit 'Final answer: X'; otherwise take the last number in the text."""
    match = FINAL_ANSWER.search(text)
    raw = match.group(1) if match else (NUMBER.findall(text) or [None])[-1]
    if raw is None:
        return None
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def check_numeric(answer: str, expected: float, tolerance: float | None = None) -> bool:
    value = extract_number(answer)
    if value is None:
        return False
    allowed = tolerance if tolerance is not None else DEFAULT_REL_TOL * max(1.0, abs(expected))
    return abs(value - expected) <= allowed


def extract_code(text: str) -> str | None:
    blocks = CODE_BLOCK.findall(text)
    return max(blocks, key=len) if blocks else None


def run_code_tests(answer: str, tests: str, timeout_s: float = CODE_TIMEOUT_S) -> bool:
    """Execute model-written code + asserts in an isolated subprocess.

    SECURITY: this runs untrusted model output on your machine. It is opt-in
    (`--exec-code`) and bounded by a timeout, but it is not a sandbox.
    """
    code = extract_code(answer)
    if code is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "solution_test.py"
        script.write_text(f"{code}\n\n{tests}\n", encoding="utf-8")
        try:
            result = subprocess.run([sys.executable, "-I", str(script)], cwd=tmp, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def deterministic_check(item: BenchmarkItem, answer: str | None, *, exec_code: bool) -> bool | None:
    """True/False when the item supports a deterministic check, None when it does not."""
    if item.answer_type == "numeric" and item.expected is not None:
        return bool(answer) and check_numeric(answer, item.expected, item.tolerance)
    if item.answer_type == "code" and item.tests and exec_code:
        return bool(answer) and run_code_tests(answer, item.tests)
    return None
