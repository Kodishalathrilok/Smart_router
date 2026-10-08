"""Statistics for the cost-vs-quality claim: strict exact accuracy, Wilson intervals, paired sign tests.

Primary quality metric = STRICT exact accuracy: correct / every item with a deterministic check.
An answer whose final number cannot be read (ambiguous or missing) counts as wrong, because the
system prompt asks for "Final answer: <value>". Dropping such answers from the denominator (the
earlier "lenient" accuracy) lets two systems be scored on different item sets.

LLM-judge scores are reported as secondary signals only; they are produced by the same judge family
that drives SmartRouter's escalations, so using them as the headline would be circular.
"""

from collections.abc import Sequence
from math import comb, sqrt
from typing import Protocol

Z95 = 1.96


class Checked(Protocol):
    item_id: str
    checkable: bool
    deterministic_correct: bool | None


def strict_correct(r: Checked) -> bool:
    return r.deterministic_correct is True


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion (well-behaved at 0% / 100% and small n)."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def strict_accuracy(results: Sequence[Checked]) -> dict:
    checked = [r for r in results if r.checkable]
    correct = sum(strict_correct(r) for r in checked)
    unparseable = sum(r.deterministic_correct is None for r in checked)
    return {
        "checked": len(checked),
        "correct": correct,
        "accuracy": round(correct / len(checked), 4) if checked else None,
        "ci95": wilson(correct, len(checked)),
        "unparseable": unparseable,
    }


def sign_test_p(wins: int, losses: int) -> float | None:
    """Two-sided exact binomial (sign) test on discordant pairs; ties are ignored."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return round(min(1.0, 2 * tail), 4)


def paired(system: Sequence[Checked], baseline: Sequence[Checked]) -> dict:
    """Item-by-item comparison on items both runs checked deterministically."""
    base = {r.item_id: r for r in baseline if r.checkable}
    wins = losses = both = neither = 0
    for r in system:
        b = base.get(r.item_id)
        if not r.checkable or b is None:
            continue
        s_ok, b_ok = strict_correct(r), strict_correct(b)
        wins += s_ok and not b_ok
        losses += b_ok and not s_ok
        both += s_ok and b_ok
        neither += not s_ok and not b_ok
    return {"items": wins + losses + both + neither, "wins": wins, "losses": losses, "both_correct": both, "both_wrong": neither,
            "sign_test_p": sign_test_p(wins, losses)}


def claim_verdict(systems: dict, pairs: dict) -> tuple[str, str]:
    """(verdict, explanation) for: 'SmartRouter reduces cost without reducing quality'.

    Supported only if SmartRouter (a) is not significantly less accurate than Always Strong, (b) is cheaper
    than Always Strong including its serving-time judge, AND (c) is more accurate than Always Cheap on paired
    items with p < 0.05. Without (c), always using the cheap model is the better system and SmartRouter's
    extra cost buys nothing measurable.
    """
    smart, cheap, strong = systems.get("smart_router"), systems.get("always_weak"), systems.get("always_strong")
    if not (smart and cheap and strong):
        return "not_tested", "SmartRouter, Always Cheap and Always Strong must all be run to test the claim."
    if smart["strict"]["checked"] == 0:
        return "not_tested", "No deterministically checkable items; LLM-judge scores alone cannot support the claim."
    vs_strong, vs_cheap = pairs.get("smart_vs_strong", {}), pairs.get("smart_vs_cheap", {})
    cost_s, cost_strong, cost_cheap = smart["cost_incl_judge"]["mean"], strong["cost_incl_judge"]["mean"], cheap["cost_incl_judge"]["mean"]
    worse_than_strong = (vs_strong.get("sign_test_p") or 1) < 0.05 and vs_strong["losses"] > vs_strong["wins"]
    beats_cheap = (vs_cheap.get("sign_test_p") or 1) < 0.05 and vs_cheap["wins"] > vs_cheap["losses"]
    acc = {k: v["strict"]["accuracy"] for k, v in (("smart", smart), ("cheap", cheap), ("strong", strong))}
    facts = (f"strict accuracy SmartRouter {_p(acc['smart'])}, Always Cheap {_p(acc['cheap'])}, Always Strong {_p(acc['strong'])}; "
             f"cost incl. judge {cost_s:.2f} vs {cost_cheap:.2f} vs {cost_strong:.2f}; "
             f"paired vs cheap {vs_cheap.get('wins', 0)} wins / {vs_cheap.get('losses', 0)} losses (p={vs_cheap.get('sign_test_p')})")
    if worse_than_strong:
        return "not_supported", f"SmartRouter is significantly less accurate than Always Strong ({facts})."
    if cost_s >= cost_strong:
        return "not_supported", f"SmartRouter is not cheaper than Always Strong once its judge calls are counted ({facts})."
    if not beats_cheap:
        if cost_cheap < cost_s:
            return ("not_supported_cheap_dominates",
                    "SmartRouter is cheaper than Always Strong without a measurable accuracy loss, but Always Cheap matches it "
                    f"for less money, so routing adds cost without a measurable benefit on this dataset ({facts}).")
        return "inconclusive", f"No significant accuracy difference from Always Cheap ({facts})."
    return "supported", f"SmartRouter beats Always Cheap on accuracy and costs less than Always Strong without a significant accuracy loss ({facts})."


def _p(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"
