import { describe, it, expect } from "vitest";
import { applyMove, applyMoves } from "./applyMove";
import type { MoveStr } from "./faceletMoves";

const SOLVED =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);

const ALL_MOVES: MoveStr[] = [
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
];

describe("applyMove", () => {
  it("preserves length (54 chars)", () => {
    for (const m of ALL_MOVES) {
      expect(applyMove(SOLVED, m).length).toBe(54);
    }
  });

  it("X then X' returns the input (any X)", () => {
    for (const m of ALL_MOVES) {
      const inv = (m.endsWith("'") ? m.slice(0, -1) : m + "'") as MoveStr;
      expect(applyMoves(SOLVED, [m, inv])).toBe(SOLVED);
    }
  });

  it("X^4 returns the input (quarter turns are order-4)", () => {
    for (const m of ALL_MOVES) {
      expect(applyMoves(SOLVED, [m, m, m, m])).toBe(SOLVED);
    }
  });

  it("preserves sticker counts (9 of each letter)", () => {
    // Apply a 14-move scramble, count letters.
    const scramble: MoveStr[] = [
      "R",
      "U",
      "F",
      "L'",
      "D",
      "B",
      "R'",
      "U'",
      "F'",
      "L",
      "D'",
      "B'",
      "R",
      "U",
    ];
    const s = applyMoves(SOLVED, scramble);
    expect(s.length).toBe(54);
    for (const letter of ["U", "R", "F", "D", "L", "B"]) {
      const count = [...s].filter((c) => c === letter).length;
      expect(count).toBe(9);
    }
  });

  it("rejects unknown move strings", () => {
    expect(() => applyMove(SOLVED, "Z" as MoveStr)).toThrow();
  });

  it("rejects bad facelet length", () => {
    expect(() => applyMove("U", "R")).toThrow();
  });
});
