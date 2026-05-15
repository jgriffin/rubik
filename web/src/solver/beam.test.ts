import { describe, expect, it } from "vitest";

import {
  applyAllMoves,
  applyMove,
  beamSolve,
  isSolved,
  stateKey,
  type State,
  type ValueFn,
} from "./beam";
import { N_MOVES_3X3, N_STICKERS_3X3 } from "./moveTables";

function solvedState(): State {
  const s = new Uint8Array(N_STICKERS_3X3);
  for (let f = 0; f < 6; f++) {
    for (let i = 0; i < 9; i++) s[f * 9 + i] = f;
  }
  return s;
}

// Inverse move index for each QTM move: U <-> U' etc. With moves indexed
// face*2 + dir, the inverse flips the dir bit (XOR 1).
const inverse = (m: number) => m ^ 1;

describe("applyMove + applyAllMoves", () => {
  it("changes the solved state for every move", () => {
    const solved = solvedState();
    for (let m = 0; m < N_MOVES_3X3; m++) {
      const after = applyMove(solved, m);
      expect(after).not.toEqual(solved);
    }
  });

  it("M then M' returns to start (every move, on solved + several scrambles)", () => {
    let s = solvedState();
    // Use a fixed scramble to exercise non-trivial states too.
    const scramble = [0, 4, 6, 2, 8, 10, 5, 3];
    const starts: State[] = [solvedState()];
    for (const m of scramble) {
      s = applyMove(s, m);
      starts.push(new Uint8Array(s));
    }
    for (const start of starts) {
      for (let m = 0; m < N_MOVES_3X3; m++) {
        const there = applyMove(start, m);
        const back = applyMove(there, inverse(m));
        expect(back).toEqual(start);
      }
    }
  });

  it("applyAllMoves returns 12 distinct states from solved", () => {
    const children = applyAllMoves(solvedState());
    expect(children).toHaveLength(N_MOVES_3X3);
    const keys = new Set(children.map(stateKey));
    expect(keys.size).toBe(N_MOVES_3X3);
  });
});

describe("isSolved", () => {
  it("true for the canonical solved state", () => {
    expect(isSolved(solvedState())).toBe(true);
  });

  it("false after any single move", () => {
    const s = solvedState();
    for (let m = 0; m < N_MOVES_3X3; m++) {
      expect(isSolved(applyMove(s, m))).toBe(false);
    }
  });

  it("true after M then M'", () => {
    const s = applyMove(applyMove(solvedState(), 0), 1);
    expect(isSolved(s)).toBe(true);
  });
});

// Synthetic ValueFn factory: ranks scores toward the solved state (and a
// supplied list of "preferred" intermediate states). Used to make
// beamSolve deterministic without an ONNX model — the parity gate
// covers the real-model behavior in P3.
function valueFnTowards(targets: State[]): ValueFn {
  const targetKeys = new Set(targets.map(stateKey));
  return async (states: State[]) => {
    const out = new Float32Array(states.length);
    for (let i = 0; i < states.length; i++) {
      if (isSolved(states[i])) {
        out[i] = 0;
      } else if (targetKeys.has(stateKey(states[i]))) {
        out[i] = 0.5;
      } else {
        out[i] = 1;
      }
    }
    return out;
  };
}

describe("beamSolve", () => {
  it("returns solved=true with empty moves on a solved input", async () => {
    const r = await beamSolve(solvedState(), valueFnTowards([]), {
      beamWidth: 4,
      maxSteps: 5,
    });
    expect(r.solved).toBe(true);
    expect(r.moves).toEqual([]);
    expect(r.steps).toBe(0);
  });

  it("solves a 1-move scramble in 1 step with width=1", async () => {
    // Scramble: apply U to solved. The inverse U' (move idx 1) takes
    // it back. Any ValueFn that ranks solved lowest works.
    const scrambled = applyMove(solvedState(), 0);
    const r = await beamSolve(scrambled, valueFnTowards([]), {
      beamWidth: 1,
      maxSteps: 2,
    });
    expect(r.solved).toBe(true);
    expect(r.moves).toEqual([1]); // U'
    expect(r.steps).toBe(1);
  });

  it("solves a 2-move scramble at width=1 with a guided ValueFn", async () => {
    // Scramble: U then R. Inverse path: R' then U'. We supply the
    // intermediate "after one inverse" state as the preferred target so
    // a width=1 (greedy) beam picks R' first.
    const after_U = applyMove(solvedState(), 0);
    const after_U_R = applyMove(after_U, 6);
    const after_U_R_then_Rprime = applyMove(after_U_R, 7); // == after_U
    const r = await beamSolve(after_U_R, valueFnTowards([after_U_R_then_Rprime]), {
      beamWidth: 1,
      maxSteps: 3,
    });
    expect(r.solved).toBe(true);
    expect(r.moves).toEqual([7, 1]); // R', U'
    expect(r.steps).toBe(2);
  });

  it("returns solved=false when the budget is too small", async () => {
    // 2-move scramble, budget=1, no guidance — greedy can't find it.
    const scrambled = applyMove(applyMove(solvedState(), 0), 6);
    const r = await beamSolve(scrambled, valueFnTowards([]), {
      beamWidth: 1,
      maxSteps: 1,
    });
    expect(r.solved).toBe(false);
    expect(r.moves).toEqual([]);
    expect(r.steps).toBe(1);
  });

  it("nExpansions accounts for pre-dedup children (beamWidth × n_moves per step)", async () => {
    const scrambled = applyMove(solvedState(), 0);
    const r = await beamSolve(scrambled, valueFnTowards([]), {
      beamWidth: 4,
      maxSteps: 2,
    });
    // Solved at step 1 (children of the initial beam), so step 0 ran
    // expansion = 1 * 12. r.steps == 1.
    expect(r.solved).toBe(true);
    expect(r.steps).toBe(1);
    expect(r.nExpansions).toBe(1 * N_MOVES_3X3);
  });
});

describe("stateKey", () => {
  it("equal states produce equal keys; different states differ", () => {
    const a = solvedState();
    const b = solvedState();
    expect(stateKey(a)).toBe(stateKey(b));
    const c = applyMove(a, 3);
    expect(stateKey(a)).not.toBe(stateKey(c));
  });
});
