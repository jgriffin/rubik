// Pure validator for a 54-char facelet string in URFDLB order.
//
// Accepts the URFDLB charset only, expects exactly 9 of each letter,
// and trims leading/trailing whitespace. Returns a discriminated union
// `{ ok: true, state }` on success or `{ ok: false, error }` with a
// descriptive message on failure.

export type ValidateResult =
  | { ok: true; state: string }
  | { ok: false; error: string };

export const FACELET_LEN = 54;
export const FACELET_CHARS = "URFDLB";
export const FACELET_PER_FACE = 9;

export function validateFacelet(input: string): ValidateResult {
  const s = input.trim();
  if (s.length !== FACELET_LEN) {
    return {
      ok: false,
      error: `expected ${FACELET_LEN} chars, got ${s.length}`,
    };
  }
  const counts: Record<string, number> = {};
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (!FACELET_CHARS.includes(c)) {
      return {
        ok: false,
        error: `invalid char '${c}' at position ${i + 1} (allowed: ${FACELET_CHARS})`,
      };
    }
    counts[c] = (counts[c] ?? 0) + 1;
  }
  for (const c of FACELET_CHARS) {
    if (counts[c] !== FACELET_PER_FACE) {
      return {
        ok: false,
        error: `expected ${FACELET_PER_FACE} of '${c}', got ${counts[c] ?? 0}`,
      };
    }
  }
  return { ok: true, state: s };
}
