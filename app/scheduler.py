"""Exact low-changeover topological scheduling for at most 18 work orders.

The problem
===========

* Work orders (jobs) have ``before -> after`` precedence edges; every valid
  answer must be a topological ordering.
* Two adjacent jobs incur a changeover when their ``family`` differs.  The
  first job never incurs a changeover.
* Among schedules with the minimum number of changeovers, the lexicographically
  smallest sequence of job ids (UTF-8 byte order) is required.

Because ``n <= 18`` an exact bit-mask dynamic program is used.

DP state
========

For every precedence-closed (ideal) subset ``S`` and every job ``k`` such that
a valid sequence schedules exactly ``S`` and ends with ``k`` (equivalently,
``S \\ {k}`` is an ideal), keep the best such sequence:

* its changeover count (``cost``);
* a back-pointer to the previous job, for reconstruction;
* a dense global *rank* among all states of the layer ``|S|``.

Transition: from state ending in ``k`` on set ``S``, append every available
job ``j`` (outside ``S`` with all predecessors already in ``S``); the added
cost is 1 iff ``family[k] != family[j]``.

Lexicographic tie-break without storing sequences
-------------------------------------------------

Two distinct states of the same layer represent distinct sequences.  For a
candidate ending in ``j`` the full sequence is its parent sequence followed by
``j``, so comparing candidates of equal cost is done by the pair

    (rank of parent sequence within layer m-1, j)

with job indices assigned in UTF-8 byte order of their ids.  After processing a
layer, every state (mask, last job) is sorted by the same key -- base layer
uses the job index alone -- and given a dense global rank: smaller rank ==
lexicographically smaller represented sequence.  All comparisons are therefore
exact, including across states that end in jobs of the same or different
families.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Changeover:
    """A single changeover event in the produced schedule."""

    position: int
    job_id: str
    from_family: str
    to_family: str

    def as_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "job_id": self.job_id,
            "from_family": self.from_family,
            "to_family": self.to_family,
        }


@dataclass(frozen=True)
class ScheduleResult:
    status: str  # "OK" | "CYCLE"
    order: list[str]
    changeovers: list[Changeover]
    changeover_count: int
    cycle: list[str] | None = None

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "status": self.status,
            "order": self.order,
            "changeover_count": self.changeover_count,
            "changeovers": [c.as_dict() for c in self.changeovers],
        }
        if self.cycle is not None:
            data["cycle"] = self.cycle
        return data


def find_cycle(
    ids: Sequence[str],
    pred: Sequence[int],
    succ: Sequence[int],
) -> list[int]:
    """Return node indices of one real directed cycle, rotated to min id first.

    Kahn's algorithm strips every node that is not part of a cyclic
    sub-graph.  Every remaining node has an incoming edge from another
    remaining node, so repeatedly taking a predecessor that remains yields a
    closed walk; the segment between two visits of the repeated node is a
    genuine directed cycle (collected in reverse edge order, then flipped).
    """
    n = len(ids)
    indeg = [0] * n
    for i in range(n):
        m = pred[i]
        while m:
            b = m.bit_length() - 1
            indeg[i] += 1
            m ^= 1 << b

    remaining = (1 << n) - 1
    queue = [i for i in range(n) if indeg[i] == 0]
    head = 0
    while head < len(queue):
        i = queue[head]
        head += 1
        remaining &= ~(1 << i)
        m = succ[i]
        while m:
            b = m.bit_length() - 1
            indeg[b] -= 1
            if indeg[b] == 0:
                queue.append(b)
            m ^= 1 << b

    if remaining == 0:
        return []

    # Walk backwards along predecessors that survive Kahn until a node repeats.
    # The resulting closed walk is in reverse edge order and is flipped below.
    start = (remaining & -remaining).bit_length() - 1
    seen_at: dict[int, int] = {}
    walk: list[int] = []
    node = start
    while node not in seen_at:
        seen_at[node] = len(walk)
        walk.append(node)
        # predecessors surviving Kahn, ascending index order (deterministic)
        cand = pred[node] & remaining
        b = (cand & -cand).bit_length() - 1
        node = b
    cyc = walk[seen_at[node] :]

    # The walk followed predecessor edges backwards: reverse so edges point
    # from each entry to the next (and last -> first).
    cyc.reverse()

    # Rotate so the smallest job id in the cycle is first.
    min_pos = min(range(len(cyc)), key=lambda p: ids[cyc[p]])
    cyc = cyc[min_pos:] + cyc[:min_pos]
    return cyc


# Packed candidate key:  (changeover_cost, parent_rank, appended_job)
# Cost is in [0, n-1] (6 bits suffice), a layer has at most 18*C(18,9) ~=
# 438k states (20 bits; 24 reserved) and a job index 6 bits: 36 bits total.
J_BITS = 6
RANK_BITS = 24
RANK_MASK = (1 << RANK_BITS) - 1
COST_SHIFT = RANK_BITS + J_BITS
RANK_SHIFT = J_BITS
NO_PARENT = 0xFF


def _solve_dp(
    ids: Sequence[str],
    families: Sequence[str],
    pred: Sequence[int],
) -> list[int]:
    """Return the optimal sequence as a list of job indices (acyclic graph)."""
    n = len(ids)
    size = 1 << n
    full = size - 1

    # Union of predecessor masks of each subset.  mask is an ideal iff
    # pred_union[mask] is a subset of mask.
    pred_union = array("I", [0]) * size
    ideal = bytearray(size)
    ideal[0] = 1
    for mask in range(1, size):
        low = mask & -mask
        j = low.bit_length() - 1
        pu = pred_union[mask ^ low] | pred[j]
        pred_union[mask] = pu
        ideal[mask] = 1 if (pu & ~mask) == 0 else 0

    # Masks grouped by popcount.
    by_pop: list[list[int]] = [[] for _ in range(n + 1)]
    for mask in range(1, size):
        by_pop[mask.bit_count()].append(mask)

    width = size * n
    INF = n + 1  # any real cost is <= n-1
    cost = bytearray([INF]) * width
    parent = bytearray([NO_PARENT]) * width
    ranks = array("I", [0]) * width  # dense rank within the state's layer

    # Layer 1: singleton {j} exists iff j has no predecessors.
    # Records are (sort_key, mask, last_job); sort key == job index here.
    records: list[tuple[int, int, int]] = []
    for j in range(n):
        if pred[j] == 0:
            mask = 1 << j
            cost[mask * n + j] = 0
            records.append((j, mask, j))
    records.sort()
    for dense, (_, mask, j) in enumerate(records):
        ranks[mask * n + j] = dense

    cost_local = cost
    parent_local = parent
    ranks_local = ranks
    fam = families

    for m in range(1, n):
        # Target-oriented relaxation: for every reachable state (t, j) of the
        # next layer scan the states of T = t \\ {j} directly in the flat
        # arrays -- no per-transition dictionary allocation.  A state on t
        # ending in j exists iff T is an ideal (j is a maximal element of t).
        records: list[tuple[int, int, int]] = []
        for t in by_pop[m + 1]:
            if not ideal[t]:
                continue
            tbase = t * n
            jm = t
            while jm:
                j = jm.bit_length() - 1
                jm ^= 1 << j
                tt = t ^ (1 << j)
                # j is a possible last job of t iff T = t \\ {j} is an ideal;
                # equivalently there is a reachable state on T (checked below
                # via cost != INF), so no separate ideal probe is needed.
                fj = fam[j]
                base = tt * n
                best_key = 0xFFFFFFFFFFFFFFFF
                best_k = -1
                km = tt
                while km:
                    k = km.bit_length() - 1
                    km ^= 1 << k
                    c = cost_local[base + k]
                    if c == INF:
                        continue
                    extra = 0 if fam[k] == fj else 1
                    key = (
                        (c + extra) << COST_SHIFT
                        | ranks_local[base + k] << RANK_SHIFT
                        | j
                    )
                    if key < best_key:
                        best_key = key
                        best_k = k
                if best_k < 0:
                    continue
                cost_local[tbase + j] = best_key >> COST_SHIFT
                parent_local[tbase + j] = best_k
                # Lex-order key for rank assignment: (parent rank, j).
                lex = (((best_key >> J_BITS) & RANK_MASK) << J_BITS) | j
                records.append((lex, t, j))
        # Dense rank == pure lexicographic order of the represented sequences.
        # All sequences have the form (prefix, j); distinct states hold
        # distinct prefix states, whose lex order is their previous-layer
        # rank.  Sorting on (parent_rank, j) -- deliberately excluding cost --
        # is exact; cost already decided WHICH sequence each state keeps.
        records.sort()
        for dense, (_, t, j) in enumerate(records):
            ranks_local[t * n + j] = dense

    # Best final state: same ordering (cost, sequence rank, last job).
    best = None
    last = -1
    fbase = full * n
    for j in range(n):
        c = cost_local[fbase + j]
        if c == INF:
            continue
        key = (c << COST_SHIFT) | (ranks_local[fbase + j] << RANK_SHIFT) | j
        if best is None or key < best:
            best = key
            last = j

    if last < 0:  # pragma: no cover - acyclic graphs always have an order
        return []

    seq: list[int] = []
    s = full
    j = last
    while s:
        seq.append(j)
        k = parent_local[s * n + j]
        s ^= 1 << j
        j = k
    seq.reverse()
    return seq


def solve(
    ids: Sequence[str],
    families: Sequence[str],
    pred: Sequence[int],
    succ: Sequence[int],
) -> ScheduleResult:
    """Solve a validated instance.

    Parameters
    ----------
    ids:
        Job ids in the index order used by the masks.  The caller sorts them by
        UTF-8 byte order so smaller index == lexicographically smaller id.
    families, pred, succ:
        Family per index; predecessor / successor bit-masks per index.
    """
    cycle_idx = find_cycle(ids, pred, succ)
    if cycle_idx:
        return ScheduleResult(
            status="CYCLE",
            order=[],
            changeovers=[],
            changeover_count=0,
            cycle=[ids[i] for i in cycle_idx],
        )

    seq = _solve_dp(ids, families, pred)

    order = [ids[i] for i in seq]
    changeovers: list[Changeover] = []
    for pos in range(1, len(seq)):
        prev, cur = seq[pos - 1], seq[pos]
        if families[prev] != families[cur]:
            changeovers.append(
                Changeover(
                    position=pos + 1,  # 1-based position in the order
                    job_id=ids[cur],
                    from_family=families[prev],
                    to_family=families[cur],
                )
            )

    return ScheduleResult(
        status="OK",
        order=order,
        changeovers=changeovers,
        changeover_count=len(changeovers),
    )
