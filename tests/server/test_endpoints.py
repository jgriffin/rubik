"""Tests for the FastAPI solver server.

Set ``RUBIK_USE_STUB_NET=1`` at module level (before importing
``rubik.server.app``) so the lifespan installs a zero-returning stub
Module and skips MPS warmup. This keeps the suite fast and CI-runnable
without a checkpoint or MPS device. The stub net's solve doesn't
actually solve — these tests only assert wire shape.
"""

import os

# Module-scoped because pytest's built-in monkeypatch is function-scoped
# and the env var must be set BEFORE FastAPI's lifespan runs (which
# happens at TestClient context entry, before any test function).
os.environ["RUBIK_USE_STUB_NET"] = "1"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from rubik.server.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Using as a context manager triggers FastAPI lifespan startup/shutdown,
    # which populates app.state.app_state.
    with TestClient(app) as c:
        yield c


def test_health_returns_loaded_stub_state(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {
        "model_loaded",
        "model_path",
        "warmup_done",
        "cube_size",
    }
    assert isinstance(body["model_loaded"], bool)
    assert isinstance(body["model_path"], str)
    assert isinstance(body["warmup_done"], bool)
    assert isinstance(body["cube_size"], int)
    # Stub-net mode: model is loaded, warmup is short-circuited.
    assert body["model_loaded"] is True
    assert body["warmup_done"] is True
    assert body["model_path"] == "<stub-net>"
    assert body["cube_size"] == 3


def test_scramble_returns_real_random_state(client):
    r = client.post("/api/scramble", json={"length": 5, "seed": 0})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"moves", "state"}
    assert isinstance(body["moves"], list)
    assert all(isinstance(m, str) for m in body["moves"])
    assert len(body["moves"]) == 5
    assert isinstance(body["state"], str)
    assert len(body["state"]) == 54
    # Length-5 scramble is not the solved state.
    solved = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9
    assert body["state"] != solved
    # Each face letter still appears 9 times (sticker count is conserved).
    for letter in "URFDLB":
        assert body["state"].count(letter) == 9


def test_scramble_rejects_negative_length(client):
    r = client.post("/api/scramble", json={"length": -1})
    assert r.status_code == 422


def test_solve_returns_shape_correct_under_stub(client):
    facelet = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9
    r = client.post("/api/solve", json={"state": facelet})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"solved", "moves", "stats"}
    # Stub net returns zeros for all states, so solve outcome is
    # whatever the beam happens to land on; only the wire shape is
    # asserted here.
    assert isinstance(body["solved"], bool)
    assert isinstance(body["moves"], list)
    assert all(isinstance(m, str) for m in body["moves"])
    stats = body["stats"]
    assert set(stats.keys()) == {
        "time_ms",
        "beam_width",
        "steps_searched",
        "final_value",
    }
    assert isinstance(stats["time_ms"], int)
    assert isinstance(stats["beam_width"], int)
    assert isinstance(stats["steps_searched"], int)
    assert isinstance(stats["final_value"], float)


def test_solve_rejects_out_of_bounds_beam_width(client):
    r = client.post("/api/solve", json={"state": "abc", "beam_width": 99999})
    assert r.status_code == 422


def test_cors_preflight_allows_vite_origin(client):
    r = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
