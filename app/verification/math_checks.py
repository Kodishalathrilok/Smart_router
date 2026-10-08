"""Deterministic math checks from the question and the answer alone (never an answer key).

Only question shapes whose meaning is unambiguous are recognized; everything else returns None
(not applicable). This is deliberately NOT a natural-language math solver.

  arithmetic   "What is / Compute / Calculate / Evaluate <expr>?"  (the whole question is the expression)
  remainder    "What is the remainder when <expr> is divided by <expr>?"
  equation     "Solve for x: <lhs> = <rhs>"  (the answer's x is substituted back; no solving)

Expressions are evaluated exactly with fractions by a whitelisting AST walker (never eval), with
size limits on powers. `^` means exponentiation. `%`, words, units and implicit multiplication in
arithmetic questions are not accepted, so "15% of 240" or "6/2(1+2)" are never interpreted.
"""

import ast
import re
from fractions import Fraction

from app.verification.numbers import final_number_token
from app.verification.schemas import CheckResult

EXPR_CHARS = r"[\d\s.+\-*/^()×÷−·]"
ARITHMETIC_Q = re.compile(
    rf"^\s*(?:what\s+is|what's|compute|calculate|evaluate)\s+(?:the\s+value\s+of\s+)?"
    rf"(?P<expr>(?:{EXPR_CHARS}|\bmod(?:ulo)?\b)+?)\s*[?.]?\s*$",
    re.I,
)
REMAINDER_Q = re.compile(
    rf"^\s*(?:what\s+is\s+)?the\s+remainder\s+(?:when|if)\s+(?P<a>{EXPR_CHARS}+?)\s+is\s+divided\s+by\s+(?P<b>{EXPR_CHARS}+?)\s*[?.]?\s*$",
    re.I,
)
EQUATION_Q = re.compile(
    r"^\s*solve(?:\s+for\s+(?P<var>[a-z]))?\s*[:,]?\s*(?P<lhs>[^=?]+?)\s*=\s*(?P<rhs>[^=?]+?)\s*[.?]?\s*$", re.I
)
NOT_A_PLAIN_NUMBER = re.compile(r"\d\s*/\s*\d|\d[eE][+-]?\d|×\s*10\s*\^")  # fractions, scientific notation
MAX_EXPONENT = 10_000
MAX_EXPR_NODES = 400  # a real question is never this long; deeper trees would hit Python's recursion limit
MAX_BITS = 200_000


class _Unsupported(Exception):
    """The expression is outside the safe subset (or too large): the check does not apply."""


def check_math(query: str, answer: str) -> CheckResult | None:
    """None when the question is not a recognized shape; otherwise pass / fail / not_checkable."""
    try:
        if m := ARITHMETIC_Q.match(query):
            expr = _normalize(m.group("expr"))
            if not re.search(r"\d", expr) or not re.search(r"[-+*/%]", expr.lstrip("-")):
                return None
            return _compare(_evaluate(expr), answer, what=m.group("expr").strip())
        if m := REMAINDER_Q.match(query):
            expr = f"({_normalize(m.group('a'))}) % ({_normalize(m.group('b'))})"
            return _compare(_evaluate(expr), answer, what=f"{m.group('a').strip()} mod {m.group('b').strip()}")
        if m := EQUATION_Q.match(query):
            return _check_equation(m, answer)
    except _Unsupported:
        return None
    return None


# ---------------------------------------------------------------- comparison


def _claimed(answer: str, var: str | None = None) -> tuple[Fraction | None, int, str]:
    """(value, decimals shown, reason-if-not-checkable) for the answer's final stated value."""
    if var:
        stated = re.findall(rf"\b{var}\s*=\s*(-?\d[\d,]*(?:\.\d+)?)(?![\d.,]*\s*[/*^\d])", answer)
        if stated:
            token = stated[-1].replace(",", "")
            return Fraction(token), _decimals(token), ""
    status, token, segment = final_number_token(answer)
    if status != "found" or token is None:
        return None, 0, "no single final number in the answer" if status == "none" else "the answer's final number is ambiguous"
    if NOT_A_PLAIN_NUMBER.search(segment):
        return None, 0, "the answer states a fraction or scientific notation"
    return Fraction(token), _decimals(token), ""


def _decimals(token: str) -> int:
    return len(token.split(".", 1)[1]) if "." in token else 0


def _tolerance(decimals: int) -> Fraction:
    """Exact for integers; otherwise half a unit of the last shown digit (a correctly rounded value passes)."""
    return Fraction(0) if decimals == 0 else Fraction(1, 2 * 10**decimals)


def _compare(truth: Fraction, answer: str, *, what: str) -> CheckResult:
    claimed, decimals, why = _claimed(answer)
    if claimed is None:
        return CheckResult(name="math", status="not_checkable", detail=f"{what} = {_show(truth)}; {why}")
    ok = abs(claimed - truth) <= _tolerance(decimals)
    detail = f"{what} evaluates to {_show(truth)}; the answer states {_show(claimed)}"
    return CheckResult(name="math", status="pass" if ok else "fail", detail=detail)


def _check_equation(m: re.Match, answer: str) -> CheckResult | None:
    lhs, rhs = _normalize(m.group("lhs"), implicit=True), _normalize(m.group("rhs"), implicit=True)
    letters = set(re.findall(r"[a-z]", (lhs + rhs).lower()))
    var = (m.group("var") or "").lower() or (next(iter(letters)) if len(letters) == 1 else None)
    if not var or letters != {var}:
        return None  # zero or several unknowns: not a one-variable equation
    claimed, decimals, why = _claimed(answer, var)
    if claimed is None:
        return CheckResult(name="math", status="not_checkable", detail=f"no value for {var} found: {why}")

    def residual(value: Fraction) -> Fraction:
        return _evaluate(lhs, {var: value}) - _evaluate(rhs, {var: value})

    at = residual(claimed)
    tolerance = _tolerance(decimals)
    if at == 0:
        ok = True
    elif tolerance == 0:
        ok = False
    else:  # a rounded root is right if the exact root lies within its rounding interval
        below, above = residual(claimed - tolerance), residual(claimed + tolerance)
        ok = below == 0 or above == 0 or (below < 0) != (above < 0)
    detail = f"substituting {var} = {_show(claimed)} gives {_show(at + _evaluate(rhs, {var: claimed}))} = {_show(_evaluate(rhs, {var: claimed}))}"
    return CheckResult(name="math", status="pass" if ok else "fail", detail=detail)


def _show(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{float(value):.10g}"


# ---------------------------------------------------------------- safe evaluation


def _normalize(expr: str, implicit: bool = False) -> str:
    expr = expr.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
    expr = re.sub(r"\bmod(?:ulo)?\b", "%", expr, flags=re.I)
    if implicit:  # equations only: 3x -> 3*x, 2(x+1) -> 2*(x+1), )( -> )*(
        expr = re.sub(r"(?<=[\d)])\s*(?=[a-zA-Z(])", "*", expr)
        expr = re.sub(r"(?<=[a-zA-Z])\s*(?=\()", "*", expr)
    return expr.strip()


def _evaluate(expr: str, env: dict[str, Fraction] | None = None) -> Fraction:
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, RecursionError, MemoryError, ValueError):
        raise _Unsupported("unparseable expression") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_EXPR_NODES:
        raise _Unsupported("expression too large")
    return _eval(tree.body, env or {})


def _eval(node: ast.AST, env: dict[str, Fraction]) -> Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Fraction(str(node.value))
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval(node.operand, env)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Mod) and isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Pow):
            base, exp, mod = _eval(node.left.left, env), _eval(node.left.right, env), _eval(node.right, env)
            if all(v.denominator == 1 for v in (base, exp, mod)) and exp >= 0 and mod != 0:
                return Fraction(pow(int(base), int(exp), int(mod)))
        left, right = _eval(node.left, env), _eval(node.right, env)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            if right == 0:
                raise _Unsupported("division by zero")
            return left / right
        if isinstance(op, ast.Mod):
            if right == 0 or left.denominator != 1 or right.denominator != 1:
                raise _Unsupported("modulo of non-integers")
            return Fraction(int(left) % int(right))
        if isinstance(op, ast.Pow):
            return _power(left, right)
    raise _Unsupported(type(node).__name__)


def _power(base: Fraction, exp: Fraction) -> Fraction:
    if exp.denominator != 1 or abs(exp) > MAX_EXPONENT:
        raise _Unsupported("non-integer or huge exponent")
    if base == 0 and exp < 0:
        raise _Unsupported("zero to a negative power")
    bits = max(abs(base.numerator).bit_length(), base.denominator.bit_length()) * abs(int(exp))
    if bits > MAX_BITS:
        raise _Unsupported("result too large")
    return base ** int(exp)
