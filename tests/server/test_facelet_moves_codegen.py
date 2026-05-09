"""Snapshot guard: web/src/state/faceletMoves.ts must match what the
current codegen would produce.

Re-runs scripts/codegen/dump_facelet_moves.py in --check mode and fails
if the on-disk TS file has drifted from what the cube tables + facelet
conversion would currently emit. Catches drift if anything upstream
(MOVE_PERM_3X3, internal face order, kociemba face order) changes
without re-running the codegen.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_faceletmoves_ts_matches_codegen():
    result = subprocess.run(
        ["uv", "run", "python", "scripts/codegen/dump_facelet_moves.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "web/src/state/faceletMoves.ts is out of date. "
        "Re-run `uv run python scripts/codegen/dump_facelet_moves.py` to update.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
