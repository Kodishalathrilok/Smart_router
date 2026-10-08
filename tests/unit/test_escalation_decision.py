"""The escalation decision for each failure mode seen in the first hard-set benchmark.

"Correct"/"incorrect" below is the (unknown at serving time) truth of the answer; the decision
only sees answer checks and judge verdicts.
"""

from app.core.escalation import decide
from app.judge.schemas import Judgement
from app.verification.checks import CheckResult

THRESHOLD, MIN_CONF = 0.78, 0.7
PASSED_CHECKS = [CheckResult(name="syntax", status="pass"), CheckResult(name="requested_names", status="pass")]
FAILED_EXAMPLES = [CheckResult(name="syntax", status="pass"), CheckResult(name="self_examples", status="fail", detail="spiral_order(m) expected [1, 2, 3, 4, ...]")]


def judgement(score: float, verdict: str, confidence: float | None = 0.9, model: str = "primary") -> Judgement:
    return Judgement(
        verdict=verdict, correctness=score, completeness=score, relevance=score, reasoning_quality=score,
        judge_confidence=confidence, quality_score=score, judge_model=model, latency_ms=1.0,
    )


ACCEPT = judgement(1.0, "good")
REJECT = judgement(0.25, "incorrect")


def run(checks, primary, second="not asked"):
    first = decide(checks, primary, threshold=THRESHOLD, min_confidence=MIN_CONF)
    if first.action != "second_opinion" or second == "not asked":
        return first
    return decide(checks, primary, threshold=THRESHOLD, min_confidence=MIN_CONF, second=second, second_asked=True)


def test_correct_answer_and_judge_acceptance_is_accepted_without_second_opinion():
    d = decide([], ACCEPT, threshold=THRESHOLD, min_confidence=MIN_CONF)
    assert d.action == "accept" and "judge accepted" in d.reason


def test_correct_answer_wrongly_rejected_is_rescued_by_second_opinion():
    """The observed false-reject case: Flash-Lite rejected correct math (115, 15, 734)."""
    first = decide([], REJECT, threshold=THRESHOLD, min_confidence=MIN_CONF)
    assert first.action == "second_opinion"
    final = run([], REJECT, second=judgement(1.0, "good", model="backup"))
    assert final.action == "accept" and "second judge accepted" in final.reason


def test_incorrect_answer_rejected_by_both_judges_escalates():
    d = run([], REJECT, second=judgement(0.2, "incorrect", model="backup"))
    assert d.action == "escalate" and "also rejected" in d.reason


def test_incorrect_answer_wrongly_accepted_is_caught_by_failed_answer_check():
    """The observed false-accept case: code whose own examples do not reproduce, accepted by the judge."""
    d = decide(FAILED_EXAMPLES, ACCEPT, threshold=THRESHOLD, min_confidence=MIN_CONF)
    assert d.action == "escalate" and "self_examples" in d.reason


def test_confident_false_accept_without_check_evidence_is_still_accepted():
    """Known limitation, pinned so it is visible: no signal contradicts a confident wrong acceptance."""
    assert decide(PASSED_CHECKS, ACCEPT, threshold=THRESHOLD, min_confidence=MIN_CONF).action == "accept"


def test_check_disagreeing_with_judge():
    # check says wrong, judge says right -> the check wins
    assert decide(FAILED_EXAMPLES, ACCEPT, threshold=THRESHOLD, min_confidence=MIN_CONF).action == "escalate"
    # checks pass (weak evidence), judge rejects -> not accepted outright, not escalated outright: ask again
    assert decide(PASSED_CHECKS, REJECT, threshold=THRESHOLD, min_confidence=MIN_CONF).action == "second_opinion"


def test_low_confidence_acceptance_gets_second_opinion():
    unsure = judgement(0.9, "good", confidence=0.4)
    assert decide([], unsure, threshold=THRESHOLD, min_confidence=MIN_CONF).action == "second_opinion"
    assert run([], unsure, second=judgement(0.3, "incorrect", model="backup")).action == "escalate"
    assert run([], unsure, second=judgement(0.95, "good", model="backup")).action == "accept"


def test_second_opinion_unavailable_keeps_primary_verdict():
    assert run([], REJECT, second=None).action == "escalate"
    assert run([], judgement(0.9, "good", confidence=0.4), second=None).action == "accept"
    assert "no second opinion" in run([], REJECT, second=None).reason


def test_missing_confidence_is_treated_as_confident_for_backward_compatibility():
    assert decide([], judgement(1.0, "good", confidence=None), threshold=THRESHOLD, min_confidence=MIN_CONF).action == "accept"


def test_reject_verdict_overrides_high_score():
    assert decide([], judgement(0.95, "incorrect"), threshold=THRESHOLD, min_confidence=MIN_CONF).action == "second_opinion"
