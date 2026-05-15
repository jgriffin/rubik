// Kociemba facelet-string <-> internal state for the 3x3 cube.
//
// Mirrors src/rubik/notation/facelet.py. Internal state is a Uint8Array(54)
// with face order ULFRBD (spec.faces), row-major within each face. Each
// int names the face the sticker's color belongs to (0=U, 1=L, 2=F,
// 3=R, 4=B, 5=D).
//
// Kociemba uses URFDLB face order, same row-major convention within each
// face, so the conversion is a pure face-block permutation.

import { N_STICKERS_3X3 } from "./moveTables";

// Internal face order: must match CUBE_3X3.faces in spec.py.
const INTERNAL_FACES = ["U", "L", "F", "R", "B", "D"] as const;
const KOCIEMBA_FACES = ["U", "R", "F", "D", "L", "B"] as const;
const STICKERS_PER_FACE = 9;

const INTERNAL_IDX: Record<string, number> = (() => {
  const out: Record<string, number> = {};
  INTERNAL_FACES.forEach((f, i) => {
    out[f] = i;
  });
  return out;
})();

// `kociembaSlotForInternal[i]` = which kociemba face slot supplies the
// stickers for internal face i. Inverse of the per-face permutation.
const KOCIEMBA_SLOT_FOR_INTERNAL: number[] = (() => {
  const internalForKociemba = KOCIEMBA_FACES.map((f) => INTERNAL_IDX[f]);
  const out = new Array<number>(INTERNAL_FACES.length);
  internalForKociemba.forEach((internalFaceIdx, kociembaSlot) => {
    out[internalFaceIdx] = kociembaSlot;
  });
  return out;
})();

export function faceletToState(facelet: string): Uint8Array {
  if (facelet.length !== N_STICKERS_3X3) {
    throw new RangeError(
      `facelet length ${facelet.length} != ${N_STICKERS_3X3}`,
    );
  }
  const out = new Uint8Array(N_STICKERS_3X3);
  for (let internalFaceIdx = 0; internalFaceIdx < 6; internalFaceIdx++) {
    const ks = KOCIEMBA_SLOT_FOR_INTERNAL[internalFaceIdx];
    const base = ks * STICKERS_PER_FACE;
    for (let i = 0; i < STICKERS_PER_FACE; i++) {
      const ch = facelet[base + i];
      const colorInt = INTERNAL_IDX[ch];
      if (colorInt === undefined) {
        throw new Error(`unknown facelet character ${JSON.stringify(ch)}`);
      }
      out[internalFaceIdx * STICKERS_PER_FACE + i] = colorInt;
    }
  }
  return out;
}

export function stateToFacelet(state: Uint8Array): string {
  if (state.length !== N_STICKERS_3X3) {
    throw new RangeError(`state length ${state.length} != ${N_STICKERS_3X3}`);
  }
  const out: string[] = new Array(N_STICKERS_3X3);
  // For each kociemba slot k, pull the internal face whose stickers fill it.
  const internalForKociemba = KOCIEMBA_FACES.map((f) => INTERNAL_IDX[f]);
  for (let k = 0; k < 6; k++) {
    const internalFaceIdx = internalForKociemba[k];
    const base = internalFaceIdx * STICKERS_PER_FACE;
    for (let i = 0; i < STICKERS_PER_FACE; i++) {
      out[k * STICKERS_PER_FACE + i] = INTERNAL_FACES[state[base + i]];
    }
  }
  return out.join("");
}
