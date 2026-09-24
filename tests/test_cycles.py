"""Cycle detection: CYCLE status, no partial schedule, real edge-valid cycle.

The reported cycle must:

* be a genuine directed cycle (every listed edge exists in the input);
* close from its last node back to its first;
* start at the smallest id in the cycle;
* be returned instead of (never together with) a partial schedule.
"""

from __future__ import annotations

import random

import pytest

from tests.helpers import build_and_solve


def assert_valid_cycle(cycle: list[str], edges: list[tuple[str, str]]) -> None:
    edge_set = set(edges)
    assert len(cycle) >= 1
    # Evidence starts at the smallest id of the cycle.
    assert cycle[0] == min(cycle)
    # Every consecutive pair -- including the closing edge -- must exist.
    for i in range(len(cycle)):
        edge = (cycle[i], cycle[(i + 1) % len(cycle)])
        assert edge in edge_set, f"cycle uses non-existent edge {edge}"
    # No repeated nodes: it is a simple cycle.
    assert len(set(cycle)) == len(cycle)


@pytest.mark.parametrize(
    "job_specs,edges",
    [
        ([("a", "X"), ("b", "Y"), ("c", "Z")],
         [("a", "b"), ("b", "c"), ("c", "a")]),
        ([("a", "X"), ("b", "Y")], [("a", "b"), ("b", "a")]),
        # Self loop is itself a cycle of length one.
        ([("a", "X"), ("b", "Y")], [("a", "a")]),
        # A cycle with acyclic nodes feeding into / out of it: only nodes
        # outside the cyclic core are peeled by Kahn; evidence stays valid.
        ([("a", "X"), ("b", "X"), ("c", "X"), ("d", "X"), ("e", "X")],
         [("a", "b"), ("b", "c"), ("c", "b"), ("a", "d")]),
        # Two disjoint cycles; whichever is reported must be fully valid.
        ([("0", "F"), ("1", "F"), ("2", "F"), ("3", "F")],
         [("0", "1"), ("1", "0"), ("2", "3"), ("3", "2")]),
        # Cycle embedded after a long acyclic prefix.
        ([("a", "F"), ("b", "F"), ("c", "F"), ("d", "F")],
         [("a", "b"), ("b", "c"), ("c", "d"), ("d", "b")]),
    ],
)
def test_named_cycles(job_specs, edges):
    result = build_and_solve(job_specs, edges)
    assert result.status == "CYCLE"
    # Never a partial schedule.
    assert result.order == []
    assert result.changeovers == []
    assert result.changeover_count == 0
    assert result.cycle is not None
    assert_valid_cycle(result.cycle, list(edges))


def test_cycle_starts_at_minimum_id():
    # The minimum id in the cyclic core is 'm', regardless of entry point.
    job_specs = [(name, "F") for name in ("m", "n", "o", "p")]
    edges = [("n", "o"), ("o", "p"), ("p", "n"), ("m", "n")]
    result = build_and_solve(job_specs, edges)
    assert result.status == "CYCLE"
    assert result.cycle[0] == "n"  # cycle core is {n,o,p}; its min is n
    assert set(result.cycle) == {"n", "o", "p"}


def test_random_cyclic_graphs():
    rng = random.Random(20260924)
    for _ in range(200):
        n = rng.randint(2, 9)
        ids = [chr(ord("a") + i) for i in range(n)]
        fams = {j: rng.choice(("P", "Q")) for j in ids}
        edges: list[tuple[str, str]] = []
        for i in range(n):
            for j in range(n):
                if i != j and rng.random() < 0.18:
                    edges.append((ids[i], ids[j]))
        # Guarantee at least one cycle.
        a, b = rng.sample(ids, 2)
        edges.append((a, b))
        edges.append((b, a))
        result = build_and_solve([(j, fams[j]) for j in ids], edges)
        assert result.status == "CYCLE"
        assert result.order == []
        assert_valid_cycle(result.cycle, edges)
