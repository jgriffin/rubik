import { FACELET_MOVES, type MoveStr } from "./faceletMoves";

export function applyMove(facelet: string, move: MoveStr): string {
  const perm = FACELET_MOVES[move];
  if (!perm) throw new Error(`unknown move: ${move}`);
  if (facelet.length !== 54) {
    throw new Error(`facelet must be 54 chars, got ${facelet.length}`);
  }
  const out = new Array<string>(54);
  for (let i = 0; i < 54; i++) {
    out[i] = facelet[perm[i]];
  }
  return out.join("");
}

export function applyMoves(
  facelet: string,
  moves: ReadonlyArray<MoveStr>,
): string {
  let s = facelet;
  for (const m of moves) s = applyMove(s, m);
  return s;
}
