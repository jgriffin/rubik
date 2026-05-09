"""Rubik solver FastAPI app.

Lifespan loads a model and warms one solve before reporting healthy.
Commit 2 ships the skeleton with stub responses; commit 3 wires the
real load + solve. See plans/m9.1-solver-demo.md.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rubik.server.schemas import (
    HealthResponse,
    ScrambleRequest,
    ScrambleResponse,
    SolveRequest,
    SolveResponse,
    SolveStats,
)


@dataclass
class AppState:
    model: Any = None
    model_path: str = ""
    warmup_done: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Stub for commit 2: nothing loaded, nothing warm.
    # Commit 3 will read RUBIK_MODEL_PATH and call inference.load_model.
    state = AppState(
        model=None,
        model_path="<not-loaded>",
        warmup_done=False,
    )
    app.state.app_state = state
    yield


app = FastAPI(lifespan=lifespan, title="rubik solver")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s = app.state.app_state
    return HealthResponse(
        model_loaded=s.model is not None,
        model_path=s.model_path,
        warmup_done=s.warmup_done,
        cube_size=3,
    )


@app.post("/api/scramble", response_model=ScrambleResponse)
def scramble(req: ScrambleRequest) -> ScrambleResponse:
    # Stub: return a fixed-shape placeholder. Commit 3 wires the real
    # random_scrambles + state_to_facelet path.
    stub_state = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9
    return ScrambleResponse(moves=[], state=stub_state)


@app.post("/api/solve", response_model=SolveResponse)
def solve(req: SolveRequest) -> SolveResponse:
    # Stub: return solved=True with no moves. Commit 3 wires beam_solve_batch.
    return SolveResponse(
        solved=True,
        moves=[],
        stats=SolveStats(
            time_ms=0,
            beam_width=req.beam_width,
            steps_searched=0,
            final_value=0.0,
        ),
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
