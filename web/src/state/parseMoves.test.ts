import { describe, it, expect } from "vitest";
import { parseMoves, formatMoves, ParseMovesError } from "./parseMoves";
import type { MoveStr } from "./faceletMoves";

describe("parseMoves — valid sequences", () => {
  it("parses a single face move", () => {
    expect(parseMoves("R")).toEqual(["R"]);
  });

  it("parses a single inverse move", () => {
    expect(parseMoves("R'")).toEqual(["R'"]);
  });

  it("parses a sexy-move trigger", () => {
    expect(parseMoves("R U R' U'")).toEqual(["R", "U", "R'", "U'"]);
  });

  it("returns [] for empty string", () => {
    expect(parseMoves("")).toEqual([]);
  });

  it("returns [] for whitespace-only string", () => {
    expect(parseMoves("   ")).toEqual([]);
    expect(parseMoves("\t\n  \t")).toEqual([]);
  });

  it("parses a length-14 explicit sequence", () => {
    const input = "R U R' U' F R B' L D F' B U' D' L'";
    const expected: MoveStr[] = [
      "R", "U", "R'", "U'", "F", "R", "B'",
      "L", "D", "F'", "B", "U'", "D'", "L'",
    ];
    const parsed = parseMoves(input);
    expect(parsed).toEqual(expected);
    expect(parsed).toHaveLength(14);
  });

  it("parses a length-30 sequence (5x sexy-extended cycle)", () => {
    const cycle: MoveStr[] = ["R", "U", "R'", "U'", "F", "F'"];
    const expected: MoveStr[] = [];
    for (let i = 0; i < 5; i++) expected.push(...cycle);
    const input = expected.join(" ");
    const parsed = parseMoves(input);
    expect(parsed).toEqual(expected);
    expect(parsed).toHaveLength(30);
  });
});

describe("parseMoves — whitespace tolerance", () => {
  it("strips leading and trailing spaces", () => {
    expect(parseMoves("  R U R'  ")).toEqual(["R", "U", "R'"]);
  });

  it("collapses multi-space runs", () => {
    expect(parseMoves("R  U   R'")).toEqual(["R", "U", "R'"]);
  });

  it("accepts tabs and newlines as separators", () => {
    expect(parseMoves("R\tU\nR'")).toEqual(["R", "U", "R'"]);
    expect(parseMoves("R \t U \n R'")).toEqual(["R", "U", "R'"]);
  });
});

describe("parseMoves / formatMoves — round-trip", () => {
  it("round-trips empty array", () => {
    expect(parseMoves(formatMoves([]))).toEqual([]);
  });

  it("round-trips a single move", () => {
    const xs: MoveStr[] = ["R"];
    expect(parseMoves(formatMoves(xs))).toEqual(xs);
  });

  it("round-trips the sexy-move-plus-F sequence", () => {
    const xs: MoveStr[] = ["R", "U", "R'", "U'", "F"];
    expect(parseMoves(formatMoves(xs))).toEqual(xs);
  });

  it("round-trips all 12 QTM moves", () => {
    const xs: MoveStr[] = [
      "U", "U'", "L", "L'", "F", "F'",
      "R", "R'", "B", "B'", "D", "D'",
    ];
    expect(parseMoves(formatMoves(xs))).toEqual(xs);
  });
});

describe("parseMoves — rejection", () => {
  function expectParseError(
    input: string,
    expectedToken: string,
    expectedIndex: number,
  ): void {
    try {
      parseMoves(input);
      expect.fail(`expected throw for input ${JSON.stringify(input)}`);
    } catch (e) {
      expect(e).toBeInstanceOf(ParseMovesError);
      const err = e as ParseMovesError;
      expect(err.token).toBe(expectedToken);
      expect(err.index).toBe(expectedIndex);
      expect(err.name).toBe("ParseMovesError");
      expect(err.message).toBe(
        `Invalid move '${expectedToken}' at position ${expectedIndex}`,
      );
    }
  }

  it("rejects double-turn notation 'R2'", () => {
    expectParseError("R2", "R2", 1);
  });

  it("rejects wide-turn notation 'Rw'", () => {
    expectParseError("Rw", "Rw", 1);
  });

  it("rejects slice move 'M'", () => {
    expectParseError("M", "M", 1);
  });

  it("rejects rotation 'x'", () => {
    expectParseError("x", "x", 1);
  });

  it("rejects lowercase face 'r' (case-sensitive)", () => {
    expectParseError("r", "r", 1);
  });

  it("rejects concatenated 'RR'", () => {
    expectParseError("RR", "RR", 1);
  });

  it("rejects detached prime ''U' as second token", () => {
    // "R 'U" — first token "R" is valid, second token "'U" is invalid.
    expectParseError("R 'U", "'U", 2);
  });

  it("reports the position of the first invalid token mid-sequence", () => {
    expectParseError("R U R2", "R2", 3);
  });

  it("rejects a non-move word", () => {
    expectParseError("garbage", "garbage", 1);
  });
});

describe("formatMoves", () => {
  it("returns empty string for empty array", () => {
    expect(formatMoves([])).toBe("");
  });

  it("returns the move itself for a single-element array", () => {
    expect(formatMoves(["R"])).toBe("R");
  });

  it("joins a sequence with single spaces", () => {
    expect(formatMoves(["R", "U", "R'"])).toBe("R U R'");
  });

  it("preserves prime notation in joined output", () => {
    expect(formatMoves(["U", "U'", "L", "L'"])).toBe("U U' L L'");
  });
});
