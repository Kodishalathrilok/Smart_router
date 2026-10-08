"""Answer checks that need no answer key.

They use only the query and the candidate answer, so they are safe to use inside
the serving pipeline, unlike the benchmark's exact checks, which compare against
hidden expected answers and must stay evaluation-only.

A failed check is strong evidence that an answer is wrong or internally inconsistent.
A passed check is weak, supporting evidence only (code that runs can still be wrong).
"not_checkable" means the check applied but could not decide (e.g. a snippet that needs
missing context); it never escalates.
"""

import ast
import builtins
import doctest
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.verification.math_checks import check_math
from app.verification.schemas import CheckResult

__all__ = ["CheckResult", "check_answer", "extract_code", "extract_code_blocks", "requested_names", "run_isolated"]

CODE_BLOCK = re.compile(r"```(python|py)?[ \t]*\n(.*?)```", re.S | re.I)  # labeled python, or unlabeled
REQUESTED_FUNCTION = re.compile(r"`([A-Za-z_]\w*)\s*\(")
REQUESTED_CLASS = re.compile(r"\bclass\s+`([A-Za-z_]\w*)`", re.I)
OTHER_LANGUAGE = re.compile(
    r"\b(?:javascript|typescript|java|rust|golang|ruby|php|kotlin|swift|sql|bash|haskell|scala)\b|c\+\+|c#", re.I
)
BUILTIN_NAMES = frozenset(dir(builtins))
DETAIL_CHARS = 200
RESULT_FILE = "_verify_result.jsonl"
MAX_REPORT_BYTES = 1_000_000
MAX_CAPTURE_CHARS = 20_000
CHILD_MEMORY_BYTES = 1024 * 1024 * 1024  # 1 GiB address space
CHILD_FILE_BYTES = 10 * 1024 * 1024
CHILD_PROCESSES = 256  # RLIMIT_NPROC counts all of the user's processes, so this cannot be tiny

# Exceptions that usually mean "snippet needs context / environment", not "the code is wrong".
CONTEXT_ERRORS = frozenset({"NameError", "ImportError", "EOFError", "KeyboardInterrupt", "MemoryError", "OSError"})
# A comment on the failing line such as "# raises ValueError" marks a deliberate error demonstration.
EXPECTED_ERROR_COMMENT = re.compile(r"#.*\b(raise[sd]?|error|exception|invalid|throws?)\b", re.I)

# Prepended to every executed script, so it runs before any model-written code. Kernel resource
# limits on POSIX (set in the child itself: preexec_fn is unsafe from worker threads) and a
# Python-level socket patch. Bounds damage; does not confine access. Not an OS sandbox.
RESOURCE_GUARD = f'''try:
    import resource as _r
    for _lim, _val in ((_r.RLIMIT_AS, {CHILD_MEMORY_BYTES}), (_r.RLIMIT_FSIZE, {CHILD_FILE_BYTES}),
                       (_r.RLIMIT_NPROC, {CHILD_PROCESSES}), (_r.RLIMIT_CORE, 0)):
        try:
            _r.setrlimit(_lim, (_val, _val))
        except (ValueError, OSError):
            pass
    del _r
except ImportError:  # Windows: no resource module
    pass
'''
NETWORK_GUARD = RESOURCE_GUARD + '''import socket as _socket
def _no_network(*_args, **_kwargs):
    raise OSError("network access is disabled during answer verification")
_socket.socket = _no_network
_socket.create_connection = _no_network
_socket.getaddrinfo = _no_network
del _socket
'''

# Runs the answer's labeled Python as __main__ (so `if __name__ == "__main__":` self-tests run),
# then the answer's own `>>>` examples. Outcomes are written to RESULT_FILE, one JSON object per phase.
MODULE_RUNNER = '''
import doctest as _dt, json as _json, sys as _sys, traceback as _tb

def _report(**data):
    with open(%(result)r, "a", encoding="utf-8") as _f:
        _f.write(_json.dumps(data) + "\\n")

_g = {"__name__": "__main__", "__builtins__": __builtins__}
try:
    exec(compile(_SOURCE, "answer.py", "exec"), _g)
except SystemExit as _e:
    _code = _e.code
    _report(phase="module", outcome="ok" if _code in (0, None) else "exit", code=str(_code))
    if _code not in (0, None):
        _sys.exit(0)
except BaseException as _e:
    _frames = [f for f in _tb.extract_tb(_e.__traceback__) if f.filename == "answer.py"]
    _report(phase="module", outcome="error", error=type(_e).__name__, message=str(_e)[:300],
            mro=[c.__name__ for c in type(_e).__mro__],
            lines=[f.lineno for f in _frames])
    _sys.exit(0)
else:
    _report(phase="module", outcome="ok")

class _LiteralChecker(_dt.OutputChecker):
    """Also accept outputs that are equal as Python literals (e.g. 2.50 vs 2.5, near-equal floats)."""

    def check_output(self, want, got, optionflags):
        if super().check_output(want, got, optionflags):
            return True
        import ast as _ast, math as _math
        try:
            w, g = _ast.literal_eval(want.strip()), _ast.literal_eval(got.strip())
        except Exception:
            return False
        def same(a, b):
            if isinstance(a, float) or isinstance(b, float):
                return isinstance(a, (int, float)) and isinstance(b, (int, float)) and _math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
            if isinstance(a, (list, tuple)) and type(a) is type(b):
                return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
            return a == b
        return same(w, g)

if _EXAMPLES.strip():
    _out = []
    _test = _dt.DocTestParser().get_doctest(_EXAMPLES, _g, "answer", None, 0)
    if _test.examples:
        _runner = _dt.DocTestRunner(checker=_LiteralChecker(), optionflags=_dt.NORMALIZE_WHITESPACE | _dt.ELLIPSIS)
        _runner.run(_test, out=_out.append)
        _report(phase="examples", attempted=len(_test.examples), failures=_runner.failures, output="".join(_out)[:2000])
'''


@dataclass
class IsolatedRun:
    returncode: int | None  # None when it timed out
    stdout: str
    stderr: str
    reports: list[dict]


def _read_reports(path: Path) -> list[dict]:
    """The child writes this file, and the child runs model-written code, so its content is untrusted:
    malformed or oversized lines are ignored instead of crashing the request."""
    if not path.exists() or path.stat().st_size > MAX_REPORT_BYTES:
        return []
    reports = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict) and isinstance(data.get("phase"), str):
            reports.append(data)
    return reports


def run_isolated(script: str, *, timeout_s: float) -> IsolatedRun:
    """Run a Python script in a fresh interpreter: isolated mode (-I), throwaway temp directory,
    empty stdin, a minimal environment (no API keys or other secrets), Python-level sockets patched,
    and on POSIX kernel resource limits (memory, file size, process count) plus a process-group kill
    on timeout so spawned children do not outlive it.

    SECURITY: this is NOT a sandbox. The code runs as your user and can read any file you can
    (including .env), open raw sockets via `_socket`, and spawn processes (bounded on POSIX).
    Only a container/VM boundary would confine it; see README "Security model".
    """
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP", "PATH") if k in os.environ}
    env.update(PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0")
    posix = os.name == "posix"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "answer_check.py"
        path.write_text(NETWORK_GUARD + script, encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-I", str(path)], cwd=tmp, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            start_new_session=posix,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            returncode: int | None = proc.returncode
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            proc.communicate()
            returncode, stdout, stderr = None, "", ""
        else:
            if posix:
                _kill_tree(proc)  # background children it left behind
        reports = _read_reports(Path(tmp) / RESULT_FILE)
    return IsolatedRun(returncode, stdout[-MAX_CAPTURE_CHARS:], stderr[-MAX_CAPTURE_CHARS:], reports)


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "posix":
        import signal

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    else:
        proc.kill()


def extract_code_blocks(text: str) -> list[str]:
    return [body for _, body in CODE_BLOCK.findall(text)]


def extract_code(text: str) -> str | None:
    """The longest Python code block, or None."""
    blocks = extract_code_blocks(text)
    return max(blocks, key=len) if blocks else None


def requested_names(query: str) -> list[str]:
    """Functions/classes the query asks for by name, e.g. `roman_to_int(...)` or class `LRUCache`."""
    names = REQUESTED_FUNCTION.findall(query) + REQUESTED_CLASS.findall(query)
    return [n for n in dict.fromkeys(names) if n not in BUILTIN_NAMES]


def check_answer(query: str, answer: str, *, execute_examples: bool, timeout_s: float) -> list[CheckResult]:
    """All answer checks that apply to this query: Python code checks and deterministic math checks.

    `execute_examples` (VERIFY_EXECUTE_CODE) gates every execution of model-written code.
    Returns [] when no check applies.
    """
    results = _code_checks(query, answer, execute=execute_examples, timeout_s=timeout_s)
    math_result = check_math(query, answer)
    return results + ([math_result] if math_result else [])


def _code_checks(query: str, answer: str, *, execute: bool, timeout_s: float) -> list[CheckResult]:
    if OTHER_LANGUAGE.search(query):
        return []
    wanted = requested_names(query)
    labeled = CODE_BLOCK.findall(answer)
    if not labeled:
        if wanted:
            return [CheckResult(name="code_block", status="fail", detail=f"asked for {', '.join(wanted)} but the answer has no Python code block")]
        return []

    sources: list[str] = []  # every block that parses (used for syntax and name checks)
    runnable: list[str] = []  # only blocks explicitly labeled Python are ever executed
    sessions: list[str] = []  # interactive `>>>` transcripts, not modules
    first_error: SyntaxError | None = None
    for label, block in labeled:
        try:
            ast.parse(block)
        except SyntaxError as exc:
            if ">>>" in block:
                sessions.append(block)
            elif label:  # explicitly labeled Python that does not parse
                return [CheckResult(name="syntax", status="fail", detail=f"python block does not parse: {exc.msg} (line {exc.lineno})")]
            else:  # unlabeled fences often hold grammars, output or pseudocode: not evidence on their own
                first_error = first_error or exc
            continue
        sources.append(block)
        if label:
            runnable.append(block)
    if not sources:
        if first_error and not sessions:
            return [CheckResult(name="syntax", status="fail", detail=f"no code block parses: {first_error.msg} (line {first_error.lineno})")]
        return []

    results = [CheckResult(name="syntax", status="pass")]
    if wanted:
        results.append(_requested_names_check(query, wanted, sources))
    if execute and runnable:
        examples = normalize_examples("\n\n".join(sessions + _docstrings(runnable)))
        results += run_code_evidence("\n\n".join(runnable), examples, timeout_s)
    return results


def normalize_examples(text: str) -> str:
    """Rebuild the answer's `>>>` examples in a form that can be compared fairly.

    - An example whose source says it raises (e.g. `f('')  # raises ValueError`) but shows no
      traceback is dropped: its intent is clear but doctest cannot compare it.
    - A trailing `# comment` on an expected-output line is removed when that makes it a valid
      Python literal (`[1, 2, 3]   # wraps around` -> `[1, 2, 3]`).
    """
    try:
        examples = doctest.DocTestParser().get_examples(text)
    except ValueError:  # malformed transcript (e.g. bad indentation): nothing reliable to compare
        return ""
    rebuilt = []
    for ex in examples:
        if ex.exc_msg is None and EXPECTED_ERROR_COMMENT.search(ex.source):
            continue
        source = ex.source.rstrip("\n").split("\n")
        want = "".join(_strip_output_comment(line) + "\n" for line in ex.want.splitlines())
        rebuilt.append(">>> " + source[0] + "\n" + "".join("... " + s + "\n" for s in source[1:]) + want)
    return "\n".join(rebuilt)


def _strip_output_comment(line: str) -> str:
    candidate = re.sub(r"\s+#.*$", "", line)
    if candidate == line:
        return line
    try:
        ast.literal_eval(line.strip())
        return line  # the '#' is part of a valid literal (e.g. inside a string)
    except (ValueError, SyntaxError):
        pass
    try:
        ast.literal_eval(candidate.strip())
        return candidate
    except (ValueError, SyntaxError):
        return line


# ---------------------------------------------------------------- requested functions


def _requested_names_check(query: str, wanted: list[str], sources: list[str]) -> CheckResult:
    defs: dict[str, list[tuple[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, bool]]] = {}
    for source in sources:
        tree = ast.parse(source)
        methods = {
            id(item)
            for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef)
            for item in cls.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in ast.walk(tree):  # any level: nested helpers still count as "defined"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs.setdefault(node.name, []).append((node, id(node) in methods))
    missing = [n for n in wanted if n not in defs]
    if missing:
        return CheckResult(name="requested_names", status="fail", detail=f"not defined: {', '.join(missing)}")
    for name, wanted_count in _requested_arity(query).items():
        functions = [(d, in_class) for d, in_class in defs.get(name, []) if not isinstance(d, ast.ClassDef)]
        if functions and not any(_accepts(d, wanted_count, in_class) for d, in_class in functions):
            return CheckResult(
                name="requested_names",
                status="fail",
                evidence="function_signature_mismatch",
                detail=f"`{name}` cannot be called with the {wanted_count} argument(s) the question specifies",
            )
    return CheckResult(name="requested_names", status="pass")


def _requested_arity(query: str) -> dict[str, int]:
    """Positional argument counts from backticked signatures in the question, e.g. `f(a: int, b)` -> 2."""
    arity: dict[str, int] = {}
    for match in REQUESTED_FUNCTION.finditer(query):
        name, depth, i = match.group(1), 1, match.end()
        while i < len(query) and depth:
            depth += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(query[i], 0)
            i += 1
        if depth or name in BUILTIN_NAMES:
            continue
        try:
            fn = ast.parse(f"def _f({query[match.end():i - 1]}): pass").body[0]
        except SyntaxError:
            continue
        args = fn.args  # type: ignore[attr-defined]
        if args.vararg or args.kwarg or args.kwonlyargs:
            continue
        params = [a.arg for a in args.posonlyargs + args.args]
        arity[name] = len(params[1:] if params[:1] in (["self"], ["cls"]) else params)
    return arity


def _accepts(fn: ast.FunctionDef | ast.AsyncFunctionDef, count: int, in_class: bool) -> bool:
    params = fn.args.posonlyargs + fn.args.args
    names = [p.arg for p in params]
    if in_class and names[:1] in (["self"], ["cls"]):
        params = params[1:]
    required = len(params) - len(fn.args.defaults)
    maximum = math.inf if fn.args.vararg else len(params)
    return required <= count <= maximum


# ---------------------------------------------------------------- execution


def _has_top_level_code(source: str) -> bool:
    """True if running the module does more than define things (imports, defs, classes, constants, docstrings)."""
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Constant):
            continue
        return True
    return False


def run_code_evidence(source: str, examples_text: str, timeout_s: float) -> list[CheckResult]:
    """Execute the answer's labeled Python (as __main__) and its `>>>` examples; classify the outcome.

    SECURITY: executes model-written code. Only called when VERIFY_EXECUTE_CODE is enabled.
    """
    script = f"_SOURCE = {source!r}\n_EXAMPLES = {examples_text!r}\n" + MODULE_RUNNER % {"result": RESULT_FILE}
    run = run_isolated(script, timeout_s=timeout_s)
    module = next((r for r in run.reports if r["phase"] == "module"), None)
    examples = next((r for r in run.reports if r["phase"] == "examples"), None)
    results: list[CheckResult] = []

    if module is None:  # never finished loading: a server loop, input wait, very slow top-level code...
        results.append(CheckResult(name="module_execution", status="not_checkable", detail=f"module did not finish within {timeout_s:g}s"))
        return results
    outcome = _classify_module(module, source)
    if outcome.status != "pass" or _has_top_level_code(source):
        results.append(outcome)
    if outcome.status != "pass":
        return results

    if examples is not None and isinstance(examples.get("failures"), int) and isinstance(examples.get("attempted"), int):
        if examples["failures"]:
            detail = _first_failure(str(examples.get("output", "")))
            results.append(CheckResult(name="self_examples", status="fail", detail=detail))
        else:
            results.append(CheckResult(name="self_examples", status="pass", detail=f"{examples['attempted']} example(s) reproduced"))
    elif run.returncode is None and doctest_examples_present(examples_text):
        results.append(CheckResult(name="self_examples", status="fail", detail=f"examples did not finish within {timeout_s:g}s"))
    return results


def doctest_examples_present(text: str) -> bool:
    return bool(text.strip()) and ">>>" in text


def _classify_module(report: dict, source: str) -> CheckResult:
    try:
        return _classify_module_report(report, source)
    except (KeyError, TypeError, IndexError, AttributeError):  # forged or partial report from the child
        return CheckResult(name="module_execution", status="not_checkable", detail="unreadable execution report")


def _classify_module_report(report: dict, source: str) -> CheckResult:
    if report["outcome"] == "ok":
        return CheckResult(name="module_execution", status="pass", detail="module ran without errors")
    if report["outcome"] == "exit":
        return CheckResult(name="module_execution", status="not_checkable", detail=f"module called sys.exit({report['code']})")
    error, message = report["error"], report["message"]
    where = f" (line {report['lines'][0]})" if report["lines"] else ""
    detail = f"{error}: {message}{where}"[:DETAIL_CHARS]
    if CONTEXT_ERRORS & set(report["mro"]):
        return CheckResult(name="module_execution", status="not_checkable", detail=f"needs missing context or environment: {detail}")
    lines = source.splitlines()
    for lineno in report["lines"]:
        nearby = lines[max(0, lineno - 2) : lineno]
        if any(EXPECTED_ERROR_COMMENT.search(line) or error in line.split("#", 1)[-1] for line in nearby if "#" in line):
            return CheckResult(name="module_execution", status="not_checkable", detail=f"looks like a deliberate error demonstration: {detail}")
    if "AssertionError" in report["mro"]:
        return CheckResult(name="module_execution", status="fail", evidence="assertion_failed", detail=f"the answer's own assertion failed: {detail}")
    return CheckResult(name="module_execution", status="fail", detail=f"the answer's code crashed when run: {detail}")


def _docstrings(sources: list[str]) -> list[str]:
    """Docstrings of modules, classes and functions (where docstring examples live)."""
    found = []
    for source in sources:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if doc:
                    found.append(doc)
    return found


def _first_failure(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip() and not set(line.strip()) <= {"*"}]
    return " | ".join(lines[:6])[:DETAIL_CHARS] or "the answer's own examples did not reproduce"
