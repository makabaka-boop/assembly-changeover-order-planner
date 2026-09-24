"""End-to-end HTTP tests for the FastAPI service.

Covers the happy path, cycle responses and every 422 condition:
self loops, duplicate edges, unknown references, extra fields, empty/non-ASCII
identifiers, wrong cardinality and wrong types.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def payload(jobs, edges=None):
    return {"jobs": jobs, "edges": edges or []}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_schedule():
    body = payload(
        [
            {"id": "a", "family": "X"},
            {"id": "b", "family": "X"},
            {"id": "c", "family": "Y"},
        ],
        [{"before": "a", "after": "b"}],
    )
    response = client.post("/schedule", json=body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "OK"
    assert data["order"] == ["a", "b", "c"]
    assert data["changeover_count"] == 1
    assert data["changeovers"] == [
        {"position": 3, "job_id": "c",
         "from_family": "X", "to_family": "Y"}
    ]


def test_valid_no_edges_defaults_to_empty_list():
    response = client.post(
        "/schedule",
        json={"jobs": [
            {"id": "a", "family": "X"},
            {"id": "b", "family": "X"},
        ]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert data["order"] == ["a", "b"]
    assert data["changeover_count"] == 0


def test_cycle_response():
    body = payload(
        [{"id": "a", "family": "X"},
         {"id": "b", "family": "X"},
         {"id": "c", "family": "X"}],
        [{"before": "a", "after": "b"},
         {"before": "b", "after": "c"},
         {"before": "c", "after": "a"}],
    )
    response = client.post("/schedule", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "CYCLE"
    assert data["order"] == []
    assert data["changeovers"] == []
    assert data["cycle"] == ["a", "b", "c"]
    # Exactly the documented fields are present.
    assert set(data.keys()) == {
        "status", "order", "changeover_count", "changeovers", "cycle"
    }


def test_deterministic_output():
    body = payload(
        [{"id": f"j{i:02d}", "family": f"F{i % 3}"} for i in range(12)],
        [],
    )
    first = client.post("/schedule", json=body).json()
    for _ in range(3):
        again = client.post("/schedule", json=body).json()
        assert again == first


# ---------------------------------------------------------------------------
# 422: malformed / illegal requests
# ---------------------------------------------------------------------------


def _expect_422(body):
    response = client.post("/schedule", json=body)
    assert response.status_code == 422, body
    return response


def test_self_loop_is_422():
    _expect_422(payload(
        [{"id": "a", "family": "X"}, {"id": "b", "family": "Y"}],
        [{"before": "a", "after": "a"}],
    ))


def test_duplicate_edge_is_422():
    _expect_422(payload(
        [{"id": "a", "family": "X"}, {"id": "b", "family": "Y"}],
        [{"before": "a", "after": "b"},
         {"before": "a", "after": "b"}],
    ))


def test_unknown_before_reference_is_422():
    _expect_422(payload(
        [{"id": "a", "family": "X"}, {"id": "b", "family": "Y"}],
        [{"before": "zzz", "after": "a"}],
    ))


def test_unknown_after_reference_is_422():
    _expect_422(payload(
        [{"id": "a", "family": "X"}, {"id": "b", "family": "Y"}],
        [{"before": "a", "after": "qqq"}],
    ))


def test_duplicate_job_id_is_422():
    _expect_422(payload([
        {"id": "a", "family": "X"},
        {"id": "a", "family": "Y"},
    ]))


def test_extra_top_level_field_is_422():
    body = payload([{"id": "a", "family": "X"},
                    {"id": "b", "family": "Y"}])
    body["unexpected"] = 1
    _expect_422(body)


def test_extra_job_field_is_422():
    _expect_422(payload([
        {"id": "a", "family": "X", "color": "red"},
        {"id": "b", "family": "Y"},
    ]))


def test_extra_edge_field_is_422():
    _expect_422(payload(
        [{"id": "a", "family": "X"}, {"id": "b", "family": "Y"}],
        [{"before": "a", "after": "b", "weight": 9}],
    ))


def test_empty_id_and_family_are_422():
    _expect_422(payload([{"id": "", "family": "X"},
                         {"id": "b", "family": "Y"}]))
    _expect_422(payload([{"id": "a", "family": ""},
                         {"id": "b", "family": "Y"}]))
    _expect_422(payload(
        [{"id": "a", "family": "X"}, {"id": "b", "family": "Y"}],
        [{"before": "", "after": "a"}],
    ))


def test_non_ascii_is_422():
    _expect_422(payload([{"id": "工单", "family": "X"},
                         {"id": "b", "family": "Y"}]))
    _expect_422(payload([{"id": "a", "family": "製品"},
                         {"id": "b", "family": "Y"}]))
    _expect_422(payload(
        [{"id": "a", "family": "X"}, {"id": "b", "family": "Y"}],
        [{"before": "a", "after": "b→"}],
    ))


def test_too_few_jobs_is_422():
    _expect_422(payload([{"id": "a", "family": "X"}]))


def test_too_many_jobs_is_422():
    jobs = [{"id": f"j{i:02d}", "family": "F"} for i in range(19)]
    _expect_422(payload(jobs))


def test_too_many_edges_is_422():
    jobs = [{"id": f"j{i:02d}", "family": "F"} for i in range(18)]
    edges = [
        {"before": f"j{i:02d}", "after": f"j{(i + 1) % 18:02d}"}
        for i in range(101)
    ]
    _expect_422(payload(jobs, edges))


def test_wrong_types_are_422():
    _expect_422({"jobs": "not-a-list"})
    _expect_422(payload([{"id": 1, "family": "X"},
                         {"id": "b", "family": "Y"}]))
    _expect_422(payload([{"id": "a", "family": 7},
                         {"id": "b", "family": "Y"}]))
    _expect_422(payload(
        [{"id": "a", "family": "X"}, {"id": "b", "family": "Y"}],
        [{"before": "a"}],  # missing 'after'
    ))


def test_missing_jobs_key_is_422():
    _expect_422({"edges": []})


# ---------------------------------------------------------------------------
# Boundary sizes accepted
# ---------------------------------------------------------------------------


def test_exactly_18_jobs_accepted():
    jobs = [{"id": f"j{i:02d}", "family": "S"} for i in range(18)]
    response = client.post("/schedule", json=payload(jobs))
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert len(data["order"]) == 18
    assert data["changeover_count"] == 0
    assert data["order"] == [j["id"] for j in jobs]
