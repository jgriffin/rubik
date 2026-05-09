"""Tests for the FastAPI solver server skeleton.

Commit 2 ships pure stubs — TestClient runs without MPS or a real
checkpoint. Schema shape, validation rejection, and CORS preflight only.
"""

import pytest
from fastapi.testclient import TestClient

from rubik.server.app import app


@pytest.fixture(scope="module")
def client():
    # Using as a context manager triggers FastAPI lifespan startup/shutdown,
    # which populates app.state.app_state.
    with TestClient(app) as c:
        yield c


def test_health_returns_initial_stub_state(client):
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
    assert body["model_loaded"] is False
    assert body["warmup_done"] is False
    assert body["cube_size"] == 3


def test_scramble_returns_shape_correct_stub(client):
    r = client.post("/api/scramble", json={"length": 5})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"moves", "state"}
    assert isinstance(body["moves"], list)
    assert all(isinstance(m, str) for m in body["moves"])
    assert isinstance(body["state"], str)
    assert len(body["state"]) == 54


def test_scramble_rejects_negative_length(client):
    r = client.post("/api/scramble", json={"length": -1})
    assert r.status_code == 422


def test_solve_returns_shape_correct_stub(client):
    facelet = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9
    r = client.post("/api/solve", json={"state": facelet})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"solved", "moves", "stats"}
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
