"""FastAPI application: exact low-changeover topological scheduling."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .models import ScheduleRequest
from .scheduler import solve

app = FastAPI(
    title="Low-Changeover Assembly Line Scheduler",
    version="1.0.0",
    description=(
        "Finds a complete topological ordering of work orders that minimizes "
        "adjacent family changeovers, then lexicographically (UTF-8 byte "
        "order) minimizes the order of job ids. Detects precedence cycles "
        "with an actual directed cycle as evidence."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/schedule", status_code=200)
def schedule(req: ScheduleRequest) -> JSONResponse:
    jobs = req.jobs
    edges = req.edges

    # --- semantic validation (422) --------------------------------------
    seen_ids: set[str] = set()
    for job in jobs:
        if job.id in seen_ids:
            raise HTTPException(422, f"duplicate job id: {job.id!r}")
        seen_ids.add(job.id)

    seen_edges: set[tuple[str, str]] = set()
    for edge in edges:
        if edge.before == edge.after:
            raise HTTPException(422, f"self loop on job id: {edge.before!r}")
        if edge.before not in seen_ids:
            raise HTTPException(422, f"unknown job id in edge: {edge.before!r}")
        if edge.after not in seen_ids:
            raise HTTPException(422, f"unknown job id in edge: {edge.after!r}")
        key = (edge.before, edge.after)
        if key in seen_edges:
            raise HTTPException(422, f"duplicate edge: {edge.before!r} -> {edge.after!r}")
        seen_edges.add(key)

    # Index jobs by UTF-8 byte order so DP index order == lexicographic order.
    ordered = sorted(jobs, key=lambda job: job.id.encode("utf-8"))
    index = {job.id: i for i, job in enumerate(ordered)}
    n = len(ordered)

    pred = [0] * n
    succ = [0] * n
    for edge in edges:
        b = index[edge.before]
        a = index[edge.after]
        succ[b] |= 1 << a
        pred[a] |= 1 << b

    result = solve(
        ids=[job.id for job in ordered],
        families=[job.family for job in ordered],
        pred=pred,
        succ=succ,
    )
    return JSONResponse(result.as_dict())
