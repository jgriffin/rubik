// Move string ↔ index mapping for QTM (3x3).
//
// Static snapshot of src/rubik/notation/moves.py — the Python side
// remains the source of truth. The TS view here is read-only and
// hand-aligned with FACE_NAMES = ("U", "L", "F", "R", "B", "D") and
// suffixes ("", "'"), index = face_idx * 2 + direction.
//
// Mapping:
//   0  U    1  U'
//   2  L    3  L'
//   4  F    5  F'
//   6  R    7  R'
//   8  B    9  B'
//   10 D    11 D'

export const FACE_NAMES = ["U", "L", "F", "R", "B", "D"] as const;

export const MOVE_STRS: readonly string[] = (() => {
  const out: string[] = [];
  for (const f of FACE_NAMES) {
    out.push(f);
    out.push(`${f}'`);
  }
  return out;
})();

export function moveIdxToStr(idx: number): string {
  if (!Number.isInteger(idx) || idx < 0 || idx >= MOVE_STRS.length) {
    throw new RangeError(`move index out of range: ${idx}`);
  }
  return MOVE_STRS[idx];
}

export function moveStrToIdx(s: string): number {
  const i = MOVE_STRS.indexOf(s);
  if (i === -1) throw new Error(`unknown move string: ${JSON.stringify(s)}`);
  return i;
}
