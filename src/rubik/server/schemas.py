"""Pydantic v2 schemas for the rubik solver server.

Wire contract documented in plans/m9-ui.md.
"""

from pydantic import BaseModel, ConfigDict, Field


class ScrambleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    length: int = Field(default=20, ge=0, le=100)
    seed: int | None = None


class ScrambleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    moves: list[str]
    state: str


class SolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str
    beam_width: int = Field(default=256, ge=1, le=4096)
    max_steps: int = Field(default=30, ge=0, le=100)


class SolveStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_ms: int
    beam_width: int
    steps_searched: int
    final_value: float


class SolveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    solved: bool
    moves: list[str]
    stats: SolveStats


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_loaded: bool
    model_path: str
    warmup_done: bool
    cube_size: int = 3
