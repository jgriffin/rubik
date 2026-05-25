import type { MoveStr } from "./faceletMoves";
import { moveIdxToStr } from "../solver/moves";

// QTM move count: 6 faces × 2 directions. Index order matches solver/moves
// (0="U", 1="U'", 2="L", ... 11="D'"), so `idx >> 1` is the face (0..5).
const N_MOVES = 12;

// Generate a random QTM scramble as a move sequence. Pure client-side —
// scrambling is just random moves on a solved cube, so it needs no backend
// (this is what lets the scramble button work on a static, server-less
// deploy). Consecutive moves on the same face are disallowed so a length-N
// scramble is N moves of real work rather than collapsing via R R'-style
// cancellations. The caller applies these to the solved state to get the
// scrambled facelet, matching the {moves, state} the API would return.
export function randomScrambleMoves(length: number): MoveStr[] {
  const moves: MoveStr[] = [];
  let lastFace = -1;
  for (let i = 0; i < length; i++) {
    let idx: number;
    let face: number;
    do {
      idx = Math.floor(Math.random() * N_MOVES);
      face = idx >> 1;
    } while (face === lastFace);
    lastFace = face;
    moves.push(moveIdxToStr(idx) as MoveStr);
  }
  return moves;
}
