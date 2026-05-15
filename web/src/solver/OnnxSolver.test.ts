// OnnxSolver tests: interface-contract level only. We mock
// onnxruntime-web so we never touch the real ORT runtime or the 61 MB
// model — the goal is to verify (a) the Solver-interface shape,
// (b) the ValueFn correctly packs BigInt64Array tensors with the right
// dtype + dimensions, and (c) dispose() releases the session.
//
// The actual ONNX execution path is covered by the Node parity gate in
// Block B (web/scripts/parityNode.ts) — duplicating that here would
// require the model bytes in CI.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// In-test state we can inspect from the mock.
const lastRunCalls: Array<{
  feedKeys: string[];
  dataType: string;
  dims: readonly number[];
  data: BigInt64Array;
}> = [];
let createCalls = 0;
let releaseCalls = 0;
let mockSessionShouldThrow: Error | null = null;

vi.mock("onnxruntime-web", () => {
  // Plain class with explicit field assignment — erasableSyntaxOnly
  // (set in our tsconfig) bans parameter-property shorthand.
  class MockTensor {
    type: string;
    data: BigInt64Array;
    dims: readonly number[];
    constructor(type: string, data: BigInt64Array, dims: readonly number[]) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
  }
  return {
    Tensor: MockTensor,
    InferenceSession: {
      create: vi.fn(async () => {
        createCalls++;
        if (mockSessionShouldThrow) {
          const e = mockSessionShouldThrow;
          mockSessionShouldThrow = null; // throw once, succeed on retry
          throw e;
        }
        return {
          run: vi.fn(async (feeds: Record<string, MockTensor>) => {
            const t = feeds.states;
            lastRunCalls.push({
              feedKeys: Object.keys(feeds),
              dataType: t.type,
              dims: t.dims,
              data: t.data,
            });
            // Synthesize a value vector with shape [B]. Return all
            // zeros — beamSolve will treat them as ties; the
            // smallest-index tiebreaker drives the search. Tests below
            // only assert ValueFn packing, not search quality.
            const B = t.dims[0];
            return {
              value: { data: new Float32Array(B) },
            };
          }),
          release: vi.fn(async () => {
            releaseCalls++;
          }),
        };
      }),
    },
    env: {
      wasm: {
        wasmPaths: undefined,
        numThreads: 0,
      },
    },
  };
});

// Module under test is imported AFTER the mock is registered above.
import { OnnxSolver } from "./OnnxSolver";
import { N_STICKERS_3X3 } from "./moveTables";

const SOLVED_3X3 =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);

beforeEach(() => {
  lastRunCalls.length = 0;
  createCalls = 0;
  releaseCalls = 0;
  mockSessionShouldThrow = null;
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("OnnxSolver Solver-interface contract", () => {
  it("starts not-ready, reports loading info before init", () => {
    const s = new OnnxSolver();
    const i0 = s.info();
    expect(i0.kind).toBe("onnx");
    expect(i0.ready).toBe(false);
    expect(i0.modelName).toBe("net_final.onnx");
    expect(i0.loadingMessage).toBe("loading model…");
  });

  it("ready() resolves after first call and is idempotent", async () => {
    const s = new OnnxSolver();
    await s.ready();
    await s.ready(); // second await — must not create a second session
    expect(createCalls).toBe(1);
    expect(s.info().ready).toBe(true);
    expect(s.info().loadDurationMs).toBeGreaterThanOrEqual(0);
  });

  it("solve() on an already-solved facelet returns moves=[] without invoking the session", async () => {
    const s = new OnnxSolver();
    const r = await s.solve({ state: SOLVED_3X3 });
    expect(r.solved).toBe(true);
    expect(r.moves).toEqual([]);
    // Session is created (ready), but no run() call should fire since
    // beamSolve short-circuits before scoring.
    expect(lastRunCalls.length).toBe(0);
    expect(r.stats.beam_width).toBe(128);
  });

  it("dispose() calls session.release once", async () => {
    const s = new OnnxSolver();
    await s.ready();
    await s.dispose();
    expect(releaseCalls).toBe(1);
    // Idempotent: second dispose is a no-op.
    await s.dispose();
    expect(releaseCalls).toBe(1);
  });
});

describe("OnnxSolver ValueFn packing", () => {
  it("packs an int64 tensor of shape [B, 54] keyed by 'states'", async () => {
    const s = new OnnxSolver();
    // Scramble: one U applied to solved. beamSolve will issue one
    // ValueFn call (the children of the initial state, B = 12).
    const facelet =
      // Identical to SOLVED_3X3 but with U applied — easier to just use
      // a single non-solved facelet so beamSolve enters the scoring loop.
      // U' applied: face 0 stays U (still U on top), but the U-band
      // rotates so face permutations land in different rows. We don't
      // care about correctness here; we just need beamSolve to score
      // *something*. Use a 1-move scramble: swap U row stickers.
      "U".repeat(9) +
      "B".repeat(3) +
      "R".repeat(6) +
      "F".repeat(9) +
      "D".repeat(9) +
      "L".repeat(9) +
      "R".repeat(3) +
      "B".repeat(6);
    // Note: this isn't a valid Kociemba scramble — but faceletToState
    // only validates length+alphabet, so it'll parse, and beamSolve
    // will issue a scoring call before failing to find a solve.
    await s.solve({ state: facelet, beam_width: 4, max_steps: 1 });
    expect(lastRunCalls.length).toBeGreaterThanOrEqual(1);
    const first = lastRunCalls[0];
    expect(first.feedKeys).toEqual(["states"]);
    expect(first.dataType).toBe("int64");
    // B*54 entries; B comes from beam_width × 12 children at step 0,
    // but the first call is parent=1 × 12 moves = 12 children.
    expect(first.dims).toEqual([12, N_STICKERS_3X3]);
    expect(first.data).toBeInstanceOf(BigInt64Array);
    expect(first.data.length).toBe(12 * N_STICKERS_3X3);
    // Each value should be a BigInt in 0..5n (color index).
    for (let k = 0; k < first.data.length; k++) {
      const v = first.data[k];
      expect(typeof v).toBe("bigint");
      expect(v >= 0n && v <= 5n).toBe(true);
    }
  });
});

describe("OnnxSolver WebGPU → WASM fallback", () => {
  it("retries on WASM when WebGPU session creation throws", async () => {
    // First create() call throws; second succeeds (the mock only throws
    // once because mockSessionShouldThrow is set then cleared).
    mockSessionShouldThrow = new Error("WebGPU adapter not available");
    // Stub navigator.gpu so auto-detect picks webgpu first.
    const origNavigator = globalThis.navigator;
    Object.defineProperty(globalThis, "navigator", {
      value: { ...origNavigator, gpu: {} },
      configurable: true,
    });
    try {
      const s = new OnnxSolver();
      await s.ready();
      expect(createCalls).toBe(2);
      expect(s.info().ready).toBe(true);
      expect(s.info().provider).toBe("onnx-wasm");
    } finally {
      Object.defineProperty(globalThis, "navigator", {
        value: origNavigator,
        configurable: true,
      });
    }
  });
});
