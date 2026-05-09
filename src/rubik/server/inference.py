"""Inference glue between FastAPI handlers and the rubik core.

Real implementations land in commit 3 (model load, scramble generation,
solve via beam_solve_batch); commit 2 ships placeholders.
"""

from pathlib import Path


def load_model(run_dir: Path) -> tuple[object, dict]:
    raise NotImplementedError("wired in commit 3 — see plans/m9.1-solver-demo.md")


def scramble_facelet(
    spec: object, length: int, seed: int | None
) -> tuple[str, list[str]]:
    raise NotImplementedError("wired in commit 3 — see plans/m9.1-solver-demo.md")


def solve_facelet(
    net: object,
    spec: object,
    facelet: str,
    beam_width: int,
    max_steps: int,
) -> dict:
    raise NotImplementedError("wired in commit 3 — see plans/m9.1-solver-demo.md")
