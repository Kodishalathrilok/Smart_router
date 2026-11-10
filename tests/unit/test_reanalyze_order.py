from types import SimpleNamespace as NS

from benchmark.reanalyze import pooled
from benchmark.runners import SYSTEMS


def test_pooled_systems_come_out_in_a_fixed_order():
    """Set iteration order varies between processes; the committed report must not."""
    row = NS(item_id="i", record=NS(query_id="q"), model_copy=lambda deep: NS(item_id="i", record=NS(query_id="q")))
    run = {s.name: [row] for s in reversed(SYSTEMS)}
    assert list(pooled([run, run])) == [s.name for s in SYSTEMS]


def test_by_question_uses_one_majority_result_per_question():
    from benchmark.reanalyze import by_question

    def rows(flags):
        return [NS(item_id=f"q{i}", checkable=True, deterministic_correct=ok) for i, ok in enumerate(flags)]

    def run(smart, cheap):
        return {"smart_router": rows(smart), "always_weak": rows(cheap)}

    # q0: smart right in 2 of 3 runs, cheap in 1 of 3 -> one win. q1: both always right. q2: an unreadable answer (None) is a miss.
    runs = [run([True, True, None], [False, True, True]), run([True, True, False], [True, True, False]), run([False, True, True], [False, True, False])]
    result = by_question(runs)["smart_vs_cheap"]
    assert (result["items"], result["wins"], result["losses"], result["both_correct"], result["both_wrong"]) == (3, 1, 0, 1, 1)
