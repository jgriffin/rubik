import { describe, expect, it } from "vitest";

import { faceletToState, stateToFacelet } from "./facelet";
import { N_STICKERS_3X3 } from "./moveTables";

describe("facelet <-> state round-trip", () => {
  it("solved state round-trips through facelet form", () => {
    const solved = new Uint8Array(N_STICKERS_3X3);
    for (let f = 0; f < 6; f++) {
      for (let i = 0; i < 9; i++) solved[f * 9 + i] = f;
    }
    const f = stateToFacelet(solved);
    expect(f.length).toBe(N_STICKERS_3X3);
    // Solved kociemba string is 9 of each of URFDLB.
    expect(f).toBe("UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB");
    const back = faceletToState(f);
    expect(back).toEqual(solved);
  });

  it("rejects wrong-length facelets", () => {
    expect(() => faceletToState("U".repeat(53))).toThrow();
  });

  it("rejects unknown characters", () => {
    const bad = "X".repeat(N_STICKERS_3X3);
    expect(() => faceletToState(bad)).toThrow();
  });
});
