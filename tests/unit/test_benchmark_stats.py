from types import SimpleNamespace as R

from benchmark.stats import claim_verdict, paired, sign_test_p, strict_accuracy, wilson


def item(i, correct, checkable=True):
    return R(item_id=f"i{i}", checkable=checkable, deterministic_correct=correct)


def test_unparseable_answers_count_as_wrong_in_strict_accuracy():
    # Lenient accuracy used to drop None (ambiguous) answers, scoring systems on different item sets.
    s = strict_accuracy([item(1, True), item(2, None), item(3, False), item(4, None, checkable=False)])
    assert (s["checked"], s["correct"], s["unparseable"], s["accuracy"]) == (3, 1, 1, 0.3333)


def test_wilson_interval_is_sane_at_the_edges():
    lo, hi = wilson(30, 30)
    assert hi == 1.0 and 0.85 < lo < 0.9
    assert wilson(0, 0) is None


def test_sign_test():
    assert sign_test_p(0, 0) is None
    assert sign_test_p(1, 1) == 1.0
    assert sign_test_p(8, 0) == 0.0078


def test_paired_counts_discordant_items():
    a = [item(1, True), item(2, True), item(3, False), item(4, None)]
    b = [item(1, True), item(2, False), item(3, True), item(4, False)]
    p = paired(a, b)
    assert (p["wins"], p["losses"], p["both_correct"], p["both_wrong"]) == (1, 1, 1, 1)


def system(acc, cost):
    return {"strict": {"accuracy": acc, "checked": 30}, "cost_incl_judge": {"mean": cost}}


def test_claim_not_supported_when_cheap_is_as_good_and_cheaper():
    systems = {"smart_router": system(0.97, 6.5), "always_weak": system(0.96, 0.97), "always_strong": system(0.96, 9.7)}
    pairs = {"smart_vs_cheap": {"wins": 1, "losses": 1, "sign_test_p": 1.0}, "smart_vs_strong": {"wins": 1, "losses": 1, "sign_test_p": 1.0}}
    verdict, why = claim_verdict(systems, pairs)
    assert verdict == "not_supported_cheap_dominates" and "Always Cheap matches it" in why


def test_claim_supported_only_with_significant_win_over_cheap():
    systems = {"smart_router": system(0.9, 4.0), "always_weak": system(0.5, 1.0), "always_strong": system(0.92, 10.0)}
    pairs = {"smart_vs_cheap": {"wins": 12, "losses": 0, "sign_test_p": 0.0005}, "smart_vs_strong": {"wins": 1, "losses": 2, "sign_test_p": 1.0}}
    assert claim_verdict(systems, pairs)[0] == "supported"
