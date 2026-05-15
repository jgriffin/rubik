import { describe, expect, it } from "vitest";
import { MOVE_STRS, moveIdxToStr, moveStrToIdx } from "./moves";

describe("QTM move string mapping", () => {
  it("has exactly 12 entries in face_idx*2+direction order", () => {
    expect(MOVE_STRS).toEqual([
      "U", "U'",
      "L", "L'",
      "F", "F'",
      "R", "R'",
      "B", "B'",
      "D", "D'",
    ]);
  });

  it("moveIdxToStr round-trips with moveStrToIdx for all 12 indices", () => {
    for (let i = 0; i < 12; i++) {
      expect(moveStrToIdx(moveIdxToStr(i))).toBe(i);
    }
  });

  it("rejects out-of-range indices", () => {
    expect(() => moveIdxToStr(-1)).toThrow(RangeError);
    expect(() => moveIdxToStr(12)).toThrow(RangeError);
    expect(() => moveIdxToStr(1.5)).toThrow(RangeError);
  });

  it("rejects unknown strings", () => {
    expect(() => moveStrToIdx("X")).toThrow();
    expect(() => moveStrToIdx("U2")).toThrow();
  });
});
