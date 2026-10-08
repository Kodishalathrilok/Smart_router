"""Escalation decision: combine answer checks and judge verdicts into accept / escalate.

Four things are kept separate:
  answer correctness   - unknown at serving time; estimated from the signals below
  judge confidence     - a judge's self-reported certainty in its own verdict (not calibrated)
  escalation decision  - this module's output, always with a written reason
  final answer quality - the primary judge's quality_score, reported unchanged

Policy (see README "Layered verification"):
  1. A failed answer check (e.g. code that does not reproduce its own examples) escalates,
     whatever the judge said.
  2. A confident judge acceptance is accepted.
  3. A rejection, or a low-confidence acceptance, is sent to a second judge from another model;
     escalation happens only if the second judge also rejects. If no second opinion is available,
     the primary verdict stands.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.judge.schemas import Judgement
from app.verification.schemas import CheckResult

# Even a high numeric score is not accepted if the judge's categorical verdict is a hard failure.
REJECT_VERDICTS = frozenset({"incorrect", "irrelevant"})
Action = Literal["accept", "escalate", "second_opinion"]


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str


def judge_passes(score: float, verdict: str | None, threshold: float) -> bool:
    return score >= threshold and verdict not in REJECT_VERDICTS


def _describe(j: Judgement) -> str:
    confidence = "" if j.judge_confidence is None else f", confidence {j.judge_confidence:.2f}"
    return f"{j.judge_model}: {j.verdict}, score {j.quality_score:.2f}{confidence}"


def decide(
    checks: Sequence[CheckResult],
    primary: Judgement,
    *,
    threshold: float,
    min_confidence: float,
    second: Judgement | None = None,
    second_asked: bool = False,
) -> Decision:
    """Pure decision function. Call once without a second opinion; if it returns
    "second_opinion", call again with `second_asked=True` and the second judgement (or None)."""
    failed = [c for c in checks if c.status == "fail"]
    if failed:
        return Decision("escalate", "answer check failed (" + "; ".join(f"{c.name} [{c.evidence}]: {c.detail}" for c in failed) + ")")

    primary_pass = judge_passes(primary.quality_score, primary.verdict, threshold)
    confident = primary.judge_confidence is None or primary.judge_confidence >= min_confidence
    if primary_pass and confident:
        return Decision("accept", f"judge accepted ({_describe(primary)})")

    doubt = "low-confidence acceptance" if primary_pass else "rejection"
    if not second_asked:
        return Decision("second_opinion", f"primary judge {doubt} ({_describe(primary)})")
    if second is None:
        verdict = "accepted" if primary_pass else "rejected"
        return Decision("accept" if primary_pass else "escalate", f"no second opinion available; primary judge {verdict} ({_describe(primary)})")
    if judge_passes(second.quality_score, second.verdict, threshold):
        return Decision("accept", f"second judge accepted ({_describe(second)}) after primary {doubt}")
    return Decision("escalate", f"second judge also rejected ({_describe(second)}) after primary {doubt}")
