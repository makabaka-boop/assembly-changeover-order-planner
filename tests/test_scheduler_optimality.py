"""Exhaustive cross-checks of the bit-mask DP against permutation enumeration.

The requirement is an exact method (state DP or equivalent).  These tests
exhaust every topological order on small graphs -- including *all* directed
graphs up to four nodes for several family patterns -- and assert that the
solver returns the globally minimum number of changeovers and, among those,
the lexicographically smallest order.
"""

from __future__ import annotations

import itertools
import random

import pytest

from tests.helpers import (
    assert_changeover_positions,
    assert_topological,
    brute_force_optimum,
    build_and_solve,
)


@pytest.mark.parametrize(
    "job_specs,edges,expected_order,expected_co",
    [
        # Forced chain: no choice at all; first job is free of changeover.
        ([("a", "X"), ("b", "Y"), ("c", "X")],
         [("a", "b"), ("b", "c")], ["a", "b", "c"], 2),
        # No precedence: grouping by family is only optimal if lex agrees;
        # 'a'(B) must start because the two 1-changeover answers are
        # a,d,b,c vs b,c,a,d and a... is lexicographically smaller.
        ([("a", "B"), ("b", "A"), ("c", "A"), ("d", "B")], [],
         ["a", "d", "b", "c"], 1),
        # Pure family grouping possible: P block then Q block, lex inside.
        ([("b", "Q"), ("a", "P"), ("d", "Q"), ("c", "P")], [],
         ["a", "c", "b", "d"], 1),
        # Diamond a->{b,c}->d forces a first and d last; b,c grouped same fam.
        ([("a", "R"), ("b", "G"), ("c", "G"), ("d", "R")],
         [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
         ["a", "b", "c", "d"], 2),
        # Same family everywhere: zero changeovers, plain lex topo order.
        ([("z", "S"), ("y", "S"), ("x", "S")], [], ["x", "y", "z"], 0),
        # Smaller lex id in a different family still wins the tie:
        # f,x,z and x,z,f both have 1 changeover; f,x,z is lex smaller.
        ([("f", "R"), ("x", "P"), ("z", "P")], [], ["f", "x", "z"], 1),
        # Precedence forbids naive family grouping (a(B) before b(A)).
        ([("a", "B"), ("b", "A"), ("c", "A")], [("a", "b")],
         ["a", "b", "c"], 1),
        # Lex tie inside a family under a chain: b and c unordered.
        ([("a", "R"), ("b", "G"), ("c", "G"), ("d", "G")],
         [("a", "d")], ["a", "b", "c", "d"], 1),
    ],
)
def test_handcrafted_examples(
    job_specs, edges, expected_order, expected_co
):
    result = build_and_solve(job_specs, edges)
    assert result.status == "OK"
    assert result.order == expected_order
    assert result.changeover_count == expected_co
    families = dict(job_specs)
    assert_changeover_positions(result, families)
    assert_topological(result.order, edges)
    # The first job never causes a changeover.
    assert all(c.position >= 2 for c in result.changeovers)


def test_first_job_never_changeover():
    job_specs = [("j", "Z"), ("k", "Z"), ("l", "Y"), ("m", "Y")]
    result = build_and_solve(job_specs, [])
    assert result.changeovers == [] or result.changeovers[0].position >= 2
    assert set(result.order) == {"j", "k", "l", "m"}


# Exhaust all directed graphs on 4 labelled nodes for several family layouts.
JOB_IDS = ["a", "b", "c", "d"]
FAMILY_PATTERNS = [
    ["A", "A", "A", "A"],
    ["A", "B", "A", "B"],
    ["A", "B", "C", "A"],
    ["A", "B", "C", "D"],
    ["B", "B", "A", "A"],
    ["A", "A", "B", "B"],
]
ALL_DIRECTED_PAIRS = [
    (JOB_IDS[i], JOB_IDS[j])
    for i in range(4)
    for j in range(4)
    if i != j
]


@pytest.mark.parametrize("pattern", FAMILY_PATTERNS)
def test_exhaustive_all_graphs_n4(pattern):
    job_specs = list(zip(JOB_IDS, pattern))
    families = dict(job_specs)
    for bits in range(1 << len(ALL_DIRECTED_PAIRS)):
        edges = [
            pair
            for bit, pair in enumerate(ALL_DIRECTED_PAIRS)
            if (bits >> bit) & 1
        ]
        result = build_and_solve(job_specs, edges)
        if result.status == "CYCLE":
            # No partial schedule may be returned on a cycle.
            assert result.order == []
            assert result.changeovers == []
            assert result.changeover_count == 0
            continue
        # Acyclic: must equal the brute-force optimum on every objective.
        expected_co, expected_order = brute_force_optimum(job_specs, edges)
        assert result.order == expected_order, (edges, result.order)
        assert result.changeover_count == expected_co
        assert len(result.order) == 4
        assert_topological(result.order, edges)
        assert_changeover_positions(result, families)


def test_exhaustive_family_layouts_n3():
    """n=3: all graphs and every family string in {A, B} (2^3 layouts)."""
    ids = ["a", "b", "c"]
    pairs = [(ids[i], ids[j]) for i in range(3) for j in range(3) if i != j]
    for layout_bits in range(8):
        layout = ["A" if (layout_bits >> i) & 1 else "B" for i in range(3)]
        job_specs = list(zip(ids, layout))
        families = dict(job_specs)
        for bits in range(1 << len(pairs)):
            edges = [
                pairs[k] for k in range(len(pairs)) if (bits >> k) & 1
            ]
            result = build_and_solve(job_specs, edges)
            if result.status == "CYCLE":
                continue
            expected_co, expected_order = brute_force_optimum(job_specs, edges)
            assert result.order == expected_order
            assert result.changeover_count == expected_co
            assert_topological(result.order, edges)
            assert_changeover_positions(result, families)


@pytest.mark.parametrize("seed", [1, 2, 3, 42, 2026])
def test_random_graphs_cross_check(seed):
    """Random small DAGs: the DP answer equals the permutation oracle."""
    rng = random.Random(seed)
    for _ in range(60):
        n = rng.randint(2, 8)
        pool = [chr(ord("a") + i) for i in range(14)]
        ids = rng.sample(pool, n)
        families = {j: rng.choice(["P", "Q", "R", "S"]) for j in ids}
        # Build an acyclic graph from edges respecting a hidden topo ordering.
        topo = ids[:]
        rng.shuffle(topo)
        position = {j: i for i, j in enumerate(topo)}
        max_edges = min(12, n * (n - 1) // 2)
        edge_count = rng.randint(0, max_edges)
        edges = set()
        while len(edges) < edge_count:
            i, j = rng.sample(range(n), 2)
            if position[topo[i]] < position[topo[j]]:
                edges.add((topo[i], topo[j]))
            else:
                edges.add((topo[j], topo[i]))
        job_specs = [(j, families[j]) for j in ids]
        result = build_and_solve(job_specs, edges)
        expected_co, expected_order = brute_force_optimum(job_specs, edges)
        assert result.status == "OK"
        assert result.order == expected_order
        assert result.changeover_count == expected_co
        assert_topological(result.order, edges)
        assert_changeover_positions(result, families)


def test_changeover_positions_one_based_and_complete():
    # Families A B B A with no edges.  a,b,c,d (A B B A) costs 2 changeovers;
    # grouping the two A's then the two B's costs only 1, and the lex-smallest
    # such schedule is a,d,b,c (A A B B) with the single changeover at pos 3.
    job_specs = [("a", "A"), ("b", "B"), ("c", "B"), ("d", "A")]
    result = build_and_solve(job_specs, [])
    assert result.order == ["a", "d", "b", "c"]
    assert result.changeover_count == 1
    assert result.changeovers[0].position == 3
    fam = dict(job_specs)
    transitions = [
        (i, result.order[i], result.order[i - 1])
        for i in range(1, len(result.order))
    ]
    reported = {(c.position - 1, c.job_id) for c in result.changeovers}
    for i, cur, prev in transitions:
        is_co = fam[cur] != fam[prev]
        assert ((i, cur) in reported) == is_co
