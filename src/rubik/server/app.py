"""Rubik solver FastAPI app.

Lifespan loads the trained 3x3 ValueNet (path overridable via
``RUBIK_MODEL_PATH``) and runs ONE warmup solve before flipping
``warmup_done=True``. Setting ``RUBIK_USE_STUB_NET=1`` swaps in a
zero-returning stub Module and skips MPS warmup — used by the fast
endpoint tests and by the Playwright webServer config.
"""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rubik.server import inference
from rubik.server.inference import LoadedModel
from rubik.server.schemas import (
    HealthResponse,
    ScrambleRequest,
    ScrambleResponse,
    SolveRequest,
    SolveResponse,
    SolveStats,
)

_DEFAULT_MODEL_PATH = (
    "experiments/davi-3x3/runs/20260508T084940Z_ln_kmax30_100k/net_final.pt"
)


@dataclass
class AppState:
    loaded: LoadedModel | None = None
    warmup_done: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = AppState()
    if os.environ.get("RUBIK_USE_STUB_NET") == "1":
        state.loaded = inference.load_stub_model()
        state.warmup_done = True  # stub has no MPS kernels to compile
    else:
        model_path = Path(os.environ.get("RUBIK_MODEL_PATH", _DEFAULT_MODEL_PATH))
        state.loaded = inference.load_model(model_path)
        # Single warmup solve to pay MPS kernel-compile cost before serving.
        warm_facelet, _ = inference.scramble_facelet(
            state.loaded.spec, length=5, seed=0
        )
        inference.solve_facelet(state.loaded, warm_facelet, beam_width=64, max_steps=10)
        state.warmup_done = True
    app.state.app_state = state
    yield


app = FastAPI(lifespan=lifespan, title="rubik solver")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _state(app: FastAPI) -> AppState:
    return app.state.app_state  # type: ignore[no-any-return]


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s = _state(app)
    loaded = s.loaded
    return HealthResponse(
        model_loaded=loaded is not None,
        model_path=loaded.model_path if loaded else "<not-loaded>",
        warmup_done=s.warmup_done,
        cube_size=3,
    )


@app.post("/api/scramble", response_model=ScrambleResponse)
def scramble(req: ScrambleRequest) -> ScrambleResponse:
    s = _state(app)
    if s.loaded is None:
        raise HTTPException(503, "model not loaded")
    facelet, moves = inference.scramble_facelet(
        s.loaded.spec, length=req.length, seed=req.seed
    )
    return ScrambleResponse(state=facelet, moves=moves)


@app.post("/api/solve", response_model=SolveResponse)
def solve(req: SolveRequest) -> SolveResponse:
    s = _state(app)
    if s.loaded is None:
        raise HTTPException(503, "model not loaded")
    if not s.warmup_done:
        raise HTTPException(503, "warmup not complete")
    out = inference.solve_facelet(
        s.loaded, req.state, beam_width=req.beam_width, max_steps=req.max_steps
    )
    return SolveResponse(
        solved=out["solved"],
        moves=out["moves"],
        stats=SolveStats(**out["stats"]),
    )


def run() -> None:
    """Entry point for the `rubik-serve` console script."""
    import uvicorn

    uvicorn.run(
        "rubik.server.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
