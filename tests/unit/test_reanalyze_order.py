from types import SimpleNamespace as NS

from benchmark.reanalyze import pooled
from benchmark.runners import SYSTEMS


def test_pooled_systems_come_out_in_a_fixed_order():
    """Set iteration order varies between processes; the committed report must not."""
    row = NS(item_id="i", record=NS(query_id="q"), model_copy=lambda deep: NS(item_id="i", record=NS(query_id="q")))
    run = {s.name: [row] for s in reversed(SYSTEMS)}
    assert list(pooled([run, run])) == [s.name for s in SYSTEMS]
