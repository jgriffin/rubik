// Move-string parser + formatter for the QTM move set.
//
// QTM only — 12 quarter-turn moves matching the `MoveStr` union exported
// by `faceletMoves.ts`. Broader cubing notation (R2, Rw, M, x, etc.) is
// rejected because the network was trained on QTM only.

import type { MoveStr } from "./faceletMoves";
import { FACELET_MOVES } from "./faceletMoves";

const VALID_MOVES: ReadonlySet<string> = new Set(Object.keys(FACELET_MOVES));

export class ParseMovesError extends Error {
  readonly token: string;
  readonly index: number;

  constructor(token: string, index: number) {
    super(`Invalid move '${token}' at position ${index}`);
    this.name = "ParseMovesError";
    this.token = token;
    this.index = index;
  }
}

/**
 * Parse a whitespace-separated move sequence into a typed `MoveStr[]`.
 *
 * - Leading/trailing whitespace is trimmed.
 * - Internal whitespace runs (spaces, tabs, newlines) are collapsed.
 * - Empty / whitespace-only input returns `[]`.
 * - Throws `ParseMovesError` on the first invalid token, with a 1-based
 *   index for human-readable error messages.
 */
export function parseMoves(input: string): MoveStr[] {
  const trimmed = input.trim();
  if (trimmed.length === 0) {
    return [];
  }
  const tokens = trimmed.split(/\s+/);
  const result: MoveStr[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (!VALID_MOVES.has(token)) {
      throw new ParseMovesError(token, i + 1);
    }
    result.push(token as MoveStr);
  }
  return result;
}

/**
 * Inverse of `parseMoves`: render a `MoveStr[]` as a single space-separated
 * string. Empty array returns `""`.
 */
export function formatMoves(moves: MoveStr[]): string {
  return moves.join(" ");
}

export type ParseMovesPartial = {
  moves: MoveStr[];
  error: ParseMovesError | null;
};

/**
 * Non-throwing variant of `parseMoves`: returns the longest valid prefix
 * of tokens plus the first invalid token (if any) as an error object.
 * Used by the editable MovesGrid's paste handler (partial / mid-typo
 * input is normal; "commit valid prefix, surface first error" is the
 * desired UX).
 */
export function parseMovesPartial(input: string): ParseMovesPartial {
  const trimmed = input.trim();
  if (trimmed.length === 0) {
    return { moves: [], error: null };
  }
  const tokens = trimmed.split(/\s+/);
  const result: MoveStr[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (!VALID_MOVES.has(token)) {
      return { moves: result, error: new ParseMovesError(token, i + 1) };
    }
    result.push(token as MoveStr);
  }
  return { moves: result, error: null };
}
