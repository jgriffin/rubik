// 3x3 beam search — TypeScript port of src/rubik/search/beam.py's
// `beam_solve_batch`, restricted to the N=1 path used by the FastAPI
// server (and by the future browser solver in Block C).
//
// The runtime is injected via `ValueFn = (states) => Promise<Float32Array>`
// so this file has no dependency on any specific ONNX runtime. Block B
// wires it up against `onnxruntime-node` for the parity gate; Block C
// will plug in `onnxruntime-web` behind the same signature.
//
// Algorithm shape (mirrors the Python `_beam_solve_cross_batched` for N=1):
//   1. If the input state is already solved, return immediately.
//   2. Each step: expand every survivor by all 12 moves, score with the
//      ValueFn, dedup within the layer (raw-bytes equality, first-occurrence
//      wins — matches np.unique(..., return_index=True)), and take the
//      `beamWidth` lowest-V survivors. Pad with the first survivor if
//      fewer unique classes than beamWidth (matches Python's short-beam
//      handling). Solved children are detected before scoring and trigger
//      path reconstruction via backpointers.

import {
  movePerm3x3,
  N_MOVES_3X3,
  N_STICKERS_3X3,
} from "./moveTables";

export type State = Uint8Array; // length 54, values 0..5

export type ValueFn = (states: State[]) => Promise<Float32Array>;

export interface BeamSolveResult {
  solved: boolean;
  moves: number[]; // move indices solving the cube, empty if unsolved
  nExpansions: number; // diagnostic: matches Python's n_expansions accounting
  steps: number; // number of beam steps actually run (≤ maxSteps)
}

export interface BeamSolveOpts {
  beamWidth: number;
  maxSteps: number;
}

// Solved-state: 9 of each face in order ULFRBD (spec.faces).
const SOLVED_STATE: State = (() => {
  const s = new Uint8Array(N_STICKERS_3X3);
  for (let face = 0; face < 6; face++) {
    for (let i = 0; i < 9; i++) {
      s[face * 9 + i] = face;
    }
  }
  return s;
})();

export function applyMove(state: State, m: number): State {
  if (m < 0 || m >= N_MOVES_3X3) {
    throw new RangeError(`move index out of range: ${m}`);
  }
  const out = new Uint8Array(N_STICKERS_3X3);
  const base = m * N_STICKERS_3X3;
  for (let i = 0; i < N_STICKERS_3X3; i++) {
    out[i] = state[movePerm3x3[base + i]];
  }
  return out;
}

export function applyAllMoves(state: State): State[] {
  const out: State[] = new Array(N_MOVES_3X3);
  for (let m = 0; m < N_MOVES_3X3; m++) {
    out[m] = applyMove(state, m);
  }
  return out;
}

export function isSolved(state: State): boolean {
  if (state.length !== N_STICKERS_3X3) return false;
  for (let i = 0; i < N_STICKERS_3X3; i++) {
    if (state[i] !== SOLVED_STATE[i]) return false;
  }
  return true;
}

// stateKey: a deterministic string-encoding of the raw 54 bytes, used as
// a Map key for within-layer dedup. We pack each byte (0..5) as a Latin-1
// code point — `String.fromCharCode` works in Node and every browser
// without `Buffer` or `btoa` polyfills. The result is a 54-char string;
// equality is byte-equality. Faster than base64 too (no encoding pass).
export function stateKey(state: State): string {
  // Use String.fromCharCode.apply over a small fixed-length array — at
  // length 54 this stays well under the per-call argument limit and
  // avoids the V8 string-concat allocations of a per-byte loop.
  return String.fromCharCode.apply(null, state as unknown as number[]);
}

interface Child {
  state: State;
  parentSlot: number;
  moveIdx: number;
}

export async function beamSolve(
  initial: State,
  valueFn: ValueFn,
  opts: BeamSolveOpts,
): Promise<BeamSolveResult> {
  const { beamWidth, maxSteps } = opts;
  if (beamWidth < 1) {
    throw new RangeError(`beamWidth must be >= 1; got ${beamWidth}`);
  }
  if (maxSteps < 0) {
    throw new RangeError(`maxSteps must be >= 0; got ${maxSteps}`);
  }
  if (initial.length !== N_STICKERS_3X3) {
    throw new RangeError(
      `state length ${initial.length} != ${N_STICKERS_3X3}`,
    );
  }

  if (isSolved(initial)) {
    return { solved: true, moves: [], nExpansions: 0, steps: 0 };
  }

  // backpointers[step][survivorSlot] = [parentSlot_in_prev_beam, moveIdx].
  const backpointers: Array<Array<[number, number]>> = [];

  let beam: State[] = [initial];
  let nExpansions = 0;

  for (let step = 0; step < maxSteps; step++) {
    // Expand: every survivor × every move. Order matches the Python
    // (parent, move) flattening: child index j = parentSlot * n_moves +
    // moveIdx, so parentSlot = floor(j / n_moves), moveIdx = j % n_moves.
    const beamLen = beam.length;
    const childCount = beamLen * N_MOVES_3X3;
    const children: Child[] = new Array(childCount);
    for (let p = 0; p < beamLen; p++) {
      const parentState = beam[p];
      for (let m = 0; m < N_MOVES_3X3; m++) {
        children[p * N_MOVES_3X3 + m] = {
          state: applyMove(parentState, m),
          parentSlot: p,
          moveIdx: m,
        };
      }
    }
    // n_expansions matches Python: rows-active * cur_beam * n_moves, NOT
    // post-dedup count. At N=1 the active-rows term is 1.
    nExpansions += childCount;

    // Solved-child short-circuit: matches the Python `is_first_solve_now`
    // semantics — argmax-on-bool returns the first True index, so we walk
    // left-to-right.
    for (let j = 0; j < childCount; j++) {
      if (isSolved(children[j].state)) {
        const path: number[] = [];
        let cur = children[j].parentSlot;
        for (let s = step - 1; s >= 0; s--) {
          const [parent, move] = backpointers[s][cur];
          path.unshift(move);
          cur = parent;
        }
        path.push(children[j].moveIdx);
        return {
          solved: true,
          moves: path,
          nExpansions,
          steps: step + 1,
        };
      }
    }

    // Score all children via one ValueFn call per step.
    const childStates: State[] = new Array(childCount);
    for (let j = 0; j < childCount; j++) childStates[j] = children[j].state;
    const scores = await valueFn(childStates);
    if (scores.length !== childCount) {
      throw new Error(
        `ValueFn returned ${scores.length} scores for ${childCount} states`,
      );
    }

    // Dedup within layer: first-occurrence wins. Non-winners get +Infinity
    // so they sort to the back. Matches np.unique(..., return_index=True).
    const seen = new Map<string, number>();
    const denseV = new Float32Array(childCount);
    for (let j = 0; j < childCount; j++) {
      denseV[j] = scores[j];
    }
    for (let j = 0; j < childCount; j++) {
      const key = stateKey(children[j].state);
      const prior = seen.get(key);
      if (prior === undefined) {
        seen.set(key, j);
      } else {
        denseV[j] = Infinity;
      }
    }

    // Top-k: sort indices by [value asc, index asc]. The index tiebreak
    // matches torch.topk's per-row-stable behavior (smaller index first
    // when values are equal — also lets us recover Python's argmax-style
    // first-occurrence on ties).
    const sortedIdxs: number[] = new Array(childCount);
    for (let j = 0; j < childCount; j++) sortedIdxs[j] = j;
    sortedIdxs.sort((a, b) => {
      const va = denseV[a];
      const vb = denseV[b];
      if (va < vb) return -1;
      if (va > vb) return 1;
      return a - b;
    });

    // Take first beamWidth survivors. If fewer unique winners exist
    // (some +Infinity slots will be picked), the +Infinity slots point to
    // real children that are byte-duplicates of earlier survivors — those
    // dedup naturally next step. If beamWidth exceeds childCount entirely
    // (only at step 0 with cur_beam=1, beamWidth>12), pad with sortedIdxs[0]
    // to mirror Python's `pad = topk_idxs[:, :1].expand(..., width - k_eff)`.
    const kEff = Math.min(beamWidth, childCount);
    const survivorIdxs: number[] = new Array(beamWidth);
    for (let j = 0; j < kEff; j++) survivorIdxs[j] = sortedIdxs[j];
    if (kEff < beamWidth) {
      const pad = sortedIdxs[0];
      for (let j = kEff; j < beamWidth; j++) survivorIdxs[j] = pad;
    }

    // Record backpointers for this step.
    const layer: Array<[number, number]> = new Array(beamWidth);
    for (let j = 0; j < beamWidth; j++) {
      const c = children[survivorIdxs[j]];
      layer[j] = [c.parentSlot, c.moveIdx];
    }
    backpointers.push(layer);

    // Next beam.
    const nextBeam: State[] = new Array(beamWidth);
    for (let j = 0; j < beamWidth; j++) {
      nextBeam[j] = children[survivorIdxs[j]].state;
    }
    beam = nextBeam;
  }

  return { solved: false, moves: [], nExpansions, steps: maxSteps };
}
