import { describe, it, expect } from "vitest";
import { randomScrambleMoves } from "./scramble";
import { applyMoves } from "./applyMove";
import type { MoveStr } from "./faceletMoves";

const SOLVED =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);

const VALID = new Set<MoveStr>([
  "U",
  "U'",
  "L",
  "L'",
  "F",
  "F'",
  "R",
  "R'",
  "B",
  "B'",
  "D",
  "D'",
]);

// Inverse of a QTM move is the same face with the prime toggled.
const inverse = (m: MoveStr): MoveStr =>
  (m.endsWith("'") ? m.slice(0, 1) : `${m}'`) as MoveStr;

describe("randomScrambleMoves", () => {
  it("returns exactly `length` moves", () => {
    for (const n of [0, 1, 5, 14, 30]) {
      expect(randomScrambleMoves(n)).toHaveLength(n);
    }
  });

  it("emits only valid QTM moves", () => {
    for (const m of randomScrambleMoves(500)) {
      expect(VALID.has(m)).toBe(true);
    }
  });

  it("never repeats a face on consecutive moves", () => {
    const ms = randomScrambleMoves(500);
    for (let i = 1; i < ms.length; i++) {
      expect(ms[i][0]).not.toBe(ms[i - 1][0]);
    }
  });

  it("changes the cube (non-solved for length 14)", () => {
    for (let t = 0; t < 25; t++) {
      expect(applyMoves(SOLVED, randomScrambleMoves(14))).not.toBe(SOLVED);
    }
  });

  it("is reversible: undoing the scramble returns to solved", () => {
    // Cross-validates that the emitted move strings drive applyMoves
    // correctly — a scramble followed by its reversed inverse is identity.
    for (let t = 0; t < 25; t++) {
      const ms = randomScrambleMoves(14);
      const undo = [...ms].reverse().map(inverse);
      expect(applyMoves(applyMoves(SOLVED, ms), undo)).toBe(SOLVED);
    }
  });
});
