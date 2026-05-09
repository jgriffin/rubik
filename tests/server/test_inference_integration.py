"""Real-model integration tests. Opt-in via ``-m slow``.

Loads the LN-trained 3x3 model from the canonical run dir, runs a real
length-5 scramble + solve, and asserts the network actually solves it.
Default ``pytest`` invocation skips these (see ``addopts`` in
``pyproject.toml``); opt in with ``uv run pytest -m slow``.
"""

import pytest
from fastapi.testclient import TestClient

from rubik.server.app import app


@pytest.mark.slow
def test_real_model_solves_length5_scramble():
    # No RUBIK_USE_STUB_NET set -> real load + MPS warmup. Slow: ~3-8s
    # for model load + warmup solve on M4 Max.
    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["model_loaded"]
        assert body["warmup_done"]
        assert body["model_path"].endswith("net_final.pt")

        r = client.post("/api/scramble", json={"length": 5, "seed": 0})
        assert r.status_code == 200
        scrambled_state = r.json()["state"]

        r = client.post(
            "/api/solve",
            json={"state": scrambled_state, "beam_width": 64, "max_steps": 12},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["solved"] is True, (
            f"network failed to solve length-5 scramble: {body}"
        )
        assert len(body["moves"]) > 0
        assert body["stats"]["time_ms"] >= 0
