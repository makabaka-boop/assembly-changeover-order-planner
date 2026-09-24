"""Shared test helpers: instance builder and an exhaustive brute-force oracle."""

from __future__ import annotations

import itertools
from typing import Iterable

from app.scheduler import ScheduleResult, solve


def build_and_solve(
    job_specs: list[tuple[str, str]],
    edges: Iterable[tuple[str, str]] = (),
) -> ScheduleResult:
    """Run the solver from (id, family) pairs and before->after edges.

    Index assignment mirrors the endpoint: ids sorted by UTF-8 byte order.
    """
    ordered = sorted(job_specs, key=lambda jf: jf[0].encode("utf-8"))
    ids = [j for j, _ in ordered]
    families = [f for _, f in ordered]
    index = {j: i for i, j in enumerate(ids)}
    n = len(ids)
    pred = [0] * n
    succ = [0] * n
    for before, after in edges:
        b, a = index[before], index[after]
        succ[b] |= 1 << a
        pred[a] |= 1 << b
    return solve(ids, families, pred, succ)


def brute_force_optimum(
    job_specs: list[tuple[str, str]],
    edges: Iterable[tuple[str, str]] = (),
) -> tuple[int, list[str]]:
    """Enumerate every permutation; return (min changeovers, lex-min order).

    Raises ValueError when the graph is cyclic (no topological order exists).
    """
    edge_set = set(edges)
    families = dict(job_specs)
    ids = sorted(j for j, _ in job_specs)
    best: tuple[int, list[str]] | None = None
    for perm in itertools.permutations(ids):
        position = {job: i for i, job in enumerate(perm)}
        if any(position[b] >= position[a] for b, a in edge_set):
            continue
        changeovers = sum(
            1
            for i in range(1, len(perm))
            if families[perm[i - 1]] != families[perm[i]]
        )
        candidate = (changeovers, list(perm))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("cycle")
    return best


def assert_changeover_positions(
    result: ScheduleResult, families: dict[str, str]
) -> None:
    """The reported changeover events match adjacent family transitions."""
    expected_positions = [
        i + 1
        for i in range(1, len(result.order))
        if families[result.order[i - 1]] != families[result.order[i]]
    ]
    assert [c.position for c in result.changeovers] == expected_positions
    assert result.changeover_count == len(result.changeovers)
    for event in result.changeovers:
        job = result.order[event.position - 1]
        assert event.job_id == job
        assert event.to_family == families[job]
        assert event.from_family == families[result.order[event.position - 2]]


def assert_topological(order: list[str], edges: Iterable[tuple[str, str]]) -> None:
    position = {job: i for i, job in enumerate(order)}
    for before, after in edges:
        assert position[before] < position[after]
