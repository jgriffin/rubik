"""Codegen: emit web/src/state/faceletMoves.ts from CUBE_3X3 move tables.

For each of the 12 QTM moves, derives a 54-index permutation `p` such that
applying the move to a kociemba facelet string `f_pre` yields
`f_post[i] = f_pre[p[i]]`. The output is written as a TypeScript
`Record<MoveStr, number[]>` module at `web/src/state/faceletMoves.ts`.
The frontend's `applyMove()` consumes this table to step through a
solution without round-tripping to the server per move click.

The permutation is the composition of two index permutations:

- `move_perm_internal(m)` — the internal-state move permutation; for any
  internal state `s`, `s_post[i] = s_pre[move_perm_internal[i]]`. Derived
  here by running `apply_moves` on `arange(n_stickers, dtype=int8)`, which
  yields exactly that permutation in tensor form.
- `internal_to_kociemba` — the face-block permutation `state_to_facelet`
  applies; for any internal state, `f[k_pos] = letter(s[itk[k_pos]])`.
  Derived from `spec.faces` and the kociemba face order `URFDLB`.

Composition (post-move kociemba char at position k_pos):
    f_post[k_pos] = letter(s_post[itk[k_pos]])
                  = letter(s_pre[mpi[itk[k_pos]]])
                  = f_pre[inv_itk[mpi[itk[k_pos]]]]

so `facelet_perm[k_pos] = inv_itk[mpi[itk[k_pos]]]`.

Sanity check (always run): for each move, derive the perm, apply it to
the solved-state's facelet, and compare against the round-trip via
`state_to_facelet(apply_moves(...))`. Any mismatch raises before write,
so a future change to internal move tables can't silently shift the perm
without the codegen catching it.

Usage:
    uv run python scripts/codegen/dump_facelet_moves.py            # write
    uv run python scripts/codegen/dump_facelet_moves.py --check    # verify

`--check` mode is what `tests/server/test_facelet_moves_codegen.py` calls
to detect drift: re-runs the codegen in-memory and byte-compares against
the on-disk file. Exits 0 on match, 1 on mismatch (with a unified diff).
"""

import argparse
import difflib
import sys
from pathlib import Path

import torch

from rubik.cube.env import apply_moves
from rubik.cube.spec import CUBE_3X3, CubeSpec
from rubik.notation.facelet import state_to_facelet
from rubik.notation.moves import MOVE_STRINGS

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "web" / "src" / "state" / "faceletMoves.ts"

_KOCIEMBA_FACE_ORDER: tuple[str, ...] = ("U", "R", "F", "D", "L", "B")


def _internal_to_kociemba(spec: CubeSpec) -> list[int]:
    """Per-position perm: kociemba slot k -> internal sticker position."""
    internal_idx = {face: i for i, face in enumerate(spec.faces)}
    n = spec.stickers_per_face
    out: list[int] = []
    for k_face_idx in range(spec.n_faces):
        letter = _KOCIEMBA_FACE_ORDER[k_face_idx]
        internal_face_idx = internal_idx[letter]
        out.extend(internal_face_idx * n + within for within in range(n))
    return out


def _invert(perm: list[int]) -> list[int]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def derive_facelet_permutation(move_idx: int, spec: CubeSpec = CUBE_3X3) -> list[int]:
    n = spec.n_stickers
    arange_state = torch.arange(n, dtype=torch.int8).unsqueeze(0)  # (1, n)
    move_t = torch.tensor([move_idx], dtype=torch.int64)
    moved = apply_moves(arange_state, move_t, spec).squeeze(0)
    move_perm_internal = [int(x) for x in moved.tolist()]

    itk = _internal_to_kociemba(spec)
    inv_itk = _invert(itk)
    return [inv_itk[move_perm_internal[itk[k]]] for k in range(n)]


def derive_all_perms(spec: CubeSpec = CUBE_3X3) -> dict[str, list[int]]:
    return {
        MOVE_STRINGS[m]: derive_facelet_permutation(m, spec)
        for m in range(spec.n_moves)
    }


def sanity_check_perms(
    perms_by_move: dict[str, list[int]], spec: CubeSpec = CUBE_3X3
) -> None:
    """Verify each derived perm matches state_to_facelet(apply_moves(...))."""
    pre_state = spec.solved_state
    pre_facelet = state_to_facelet(pre_state, spec)
    for move_idx in range(spec.n_moves):
        move_str = MOVE_STRINGS[move_idx]
        post_state = apply_moves(
            pre_state.unsqueeze(0), torch.tensor([move_idx]), spec
        ).squeeze(0)
        expected = state_to_facelet(post_state, spec)
        perm = perms_by_move[move_str]
        actual = "".join(pre_facelet[perm[i]] for i in range(spec.n_stickers))
        if actual != expected:
            raise AssertionError(
                f"facelet permutation mismatch for {move_str}:\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )


def render_ts(perms_by_move: dict[str, list[int]]) -> str:
    move_strs = list(perms_by_move.keys())
    # MoveStr union, 6 per line for readability.
    type_chunks: list[str] = []
    for i in range(0, len(move_strs), 6):
        chunk = move_strs[i : i + 6]
        joined = " | ".join(f'"{m}"' for m in chunk)
        prefix = "  " if i == 0 else "  | "
        type_chunks.append(prefix + joined)
    type_block = "export type MoveStr =\n" + "\n".join(type_chunks) + ";"

    body_lines: list[str] = []
    for move in move_strs:
        nums = ", ".join(str(x) for x in perms_by_move[move])
        body_lines.append(f'  "{move}": [{nums}],')
    body = "\n".join(body_lines)

    header = (
        "// Generated by scripts/codegen/dump_facelet_moves.py"
        " — do not edit by hand.\n"
        "// Re-run if cube/env.py move tables"
        " or notation/facelet.py conventions change.\n"
        "//\n"
        "// For each QTM move: a 54-index permutation `p` such that\n"
        "//   facelet_post[i] = facelet_pre[p[i]]\n"
        "// Used by web/src/state/applyMove.ts.\n"
    )

    return (
        f"{header}\n"
        f"{type_block}\n\n"
        f"export const FACELET_MOVES: Record<MoveStr, ReadonlyArray<number>> = {{\n"
        f"{body}\n"
        f"}} as const;\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify on-disk file matches what would be generated; exit 1 on drift",
    )
    args = parser.parse_args(argv)

    perms = derive_all_perms()
    sanity_check_perms(perms)
    new_content = render_ts(perms)

    if args.check:
        if not OUTPUT_PATH.exists():
            print(f"missing: {OUTPUT_PATH}", file=sys.stderr)
            return 1
        on_disk = OUTPUT_PATH.read_text()
        if on_disk == new_content:
            print(f"ok: {OUTPUT_PATH} matches codegen")
            return 0
        diff = "".join(
            difflib.unified_diff(
                on_disk.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=str(OUTPUT_PATH) + " (on disk)",
                tofile=str(OUTPUT_PATH) + " (regenerated)",
            )
        )
        print(
            f"DRIFT: {OUTPUT_PATH} does not match codegen output.\n{diff}",
            file=sys.stderr,
        )
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(new_content)
    print(f"wrote: {OUTPUT_PATH} ({len(perms)} moves)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
