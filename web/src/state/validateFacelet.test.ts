import { describe, it, expect } from "vitest";
import { validateFacelet, FACELET_LEN } from "./validateFacelet";
import { applyMoves } from "./applyMove";

const SOLVED =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);

describe("validateFacelet — valid inputs", () => {
  it("accepts the SOLVED state", () => {
    const r = validateFacelet(SOLVED);
    expect(r).toEqual({ ok: true, state: SOLVED });
  });

  it("accepts a state produced by applying a move sequence to SOLVED", () => {
    const after = applyMoves(SOLVED, ["R", "U", "R'"]);
    expect(after).toHaveLength(FACELET_LEN);
    const r = validateFacelet(after);
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.state).toBe(after);
    }
  });

  it("trims leading/trailing whitespace", () => {
    const r = validateFacelet("  " + SOLVED + "\n");
    expect(r).toEqual({ ok: true, state: SOLVED });
  });
});

describe("validateFacelet — length errors", () => {
  it("rejects too-short input (53 chars)", () => {
    const short = SOLVED.slice(0, 53);
    const r = validateFacelet(short);
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.error).toContain("expected 54");
      expect(r.error).toContain("53");
    }
  });

  it("rejects too-long input (55 chars)", () => {
    const long = SOLVED + "U";
    const r = validateFacelet(long);
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.error).toContain("expected 54");
      expect(r.error).toContain("55");
    }
  });

  it("rejects empty string", () => {
    const r = validateFacelet("");
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.error).toContain("expected 54");
    }
  });
});

describe("validateFacelet — charset errors", () => {
  it("rejects an invalid uppercase char", () => {
    const bad = "X" + SOLVED.slice(1);
    const r = validateFacelet(bad);
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.error).toContain("'X'");
      expect(r.error).toContain("position 1");
    }
  });

  it("rejects a lowercase char (case-sensitive charset)", () => {
    const bad = "u" + SOLVED.slice(1);
    const r = validateFacelet(bad);
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.error).toContain("'u'");
    }
  });
});

describe("validateFacelet — count errors", () => {
  it("rejects a state with 10 U's and 8 R's", () => {
    // Replace position 9 (first R) with a U — now U=10, R=8.
    const bad = SOLVED.slice(0, 9) + "U" + SOLVED.slice(10);
    expect(bad).toHaveLength(54);
    const r = validateFacelet(bad);
    expect(r.ok).toBe(false);
    if (!r.ok) {
      // Count check happens in URFDLB order; U is checked first and
      // mismatches with 10 instead of 9.
      expect(r.error).toMatch(/expected 9 of '[URFDLB]'/);
    }
  });
});
