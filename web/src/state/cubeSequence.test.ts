import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { createCubeSequence } from "./cubeSequence";
import type { MoveStr } from "./faceletMoves";

const SOLVED =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);

// Default-sized fixture: 4 moves, msPerMove=600, gapMs=0 → 2400 ms total.
function makeSeq(opts: Partial<Parameters<typeof createCubeSequence>[0]> = {}) {
  return createCubeSequence({
    startFacelet: SOLVED,
    moves: ["R", "U", "R'", "U'"] as MoveStr[],
    msPerMove: 600,
    gapMs: 0,
    ...opts,
  });
}

// rAF / performance.now() fakes. The factory uses both; we need fake
// timers to advance them in lockstep.
//
// Node doesn't expose `requestAnimationFrame` globally. We install a
// minimal rAF shim backed by `setTimeout(0)` so vitest's fake-timer
// `vi.advanceTimersByTimeAsync(N)` advances the rAF queue alongside
// the simulated wall clock. The shim mirrors how a browser's vsync-
// driven rAF would behave under coarse fake-timer cadence (one frame
// fires per tick, not at 16.6ms intervals — but the controller only
// cares about the wall-clock delta read from `performance.now()`, not
// the inter-frame spacing, so this is faithful for our purposes).
//
// `performance` is explicitly faked via `toFake` so `performance.now()`
// advances with the simulated wall clock; without it the controller
// would compute zero elapsed time and never advance the timestamp.
let rafCallbacks: Array<(now: number) => void> = [];
let nextRafId = 1;

beforeEach(() => {
  rafCallbacks = [];
  nextRafId = 1;
  vi.stubGlobal(
    "requestAnimationFrame",
    (cb: (now: number) => void): number => {
      const id = nextRafId++;
      // Schedule via setTimeout(0) so fake timers carry it. We capture
      // the callback by id so cancelAnimationFrame can remove it.
      const entry = { id, cb };
      rafCallbacks.push(entry as unknown as (now: number) => void);
      setTimeout(() => {
        const idx = rafCallbacks.findIndex(
          (e) => (e as unknown as { id: number }).id === id,
        );
        if (idx === -1) return; // canceled
        rafCallbacks.splice(idx, 1);
        cb(performance.now());
      }, 0);
      return id;
    },
  );
  vi.stubGlobal("cancelAnimationFrame", (id: number) => {
    const idx = rafCallbacks.findIndex(
      (e) => (e as unknown as { id: number }).id === id,
    );
    if (idx !== -1) rafCallbacks.splice(idx, 1);
  });
  vi.useFakeTimers({
    toFake: [
      "setTimeout",
      "clearTimeout",
      "setInterval",
      "clearInterval",
      "Date",
      "performance",
    ],
  });
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  rafCallbacks = [];
});

describe("createCubeSequence — initial state", () => {
  it("starts in idle at timestamp 0 with currentMoveIndex -1", () => {
    const s = makeSeq();
    expect(s.status).toBe("idle");
    expect(s.timestamp).toBe(0);
    expect(s.currentMoveIndex).toBe(-1);
    expect(s.currentMoveProgress).toBe(0);
    expect(s.totalDurationMs).toBe(2400);
    expect(s.msPerMove).toBe(600);
    expect(s.gapMs).toBe(0);
    expect(s.startFacelet).toBe(SOLVED);
    expect(s.moves).toEqual(["R", "U", "R'", "U'"]);
  });

  it("uses DUR_FORWARD_MS (600) as default msPerMove", () => {
    const s = createCubeSequence({ startFacelet: SOLVED, moves: ["R"] });
    expect(s.msPerMove).toBe(600);
    expect(s.totalDurationMs).toBe(600);
  });

  it("totalDurationMs = moves.length * (msPerMove + gapMs)", () => {
    const s = makeSeq({ msPerMove: 500, gapMs: 100 });
    expect(s.totalDurationMs).toBe(4 * 600); // 4 moves * (500 + 100)
  });

  it("autoplay=true starts in 'playing' immediately", () => {
    const s = makeSeq({ autoplay: true });
    expect(s.status).toBe("playing");
  });

  it("empty moves array → status stays idle, play() is a no-op", () => {
    const s = createCubeSequence({ startFacelet: SOLVED, moves: [] });
    expect(s.totalDurationMs).toBe(0);
    expect(s.currentMoveIndex).toBe(-1);
    s.play();
    expect(s.status).toBe("idle");
  });
});

describe("createCubeSequence — validation", () => {
  it("rejects non-54-char startFacelet", () => {
    expect(() =>
      createCubeSequence({ startFacelet: "U".repeat(10), moves: [] }),
    ).toThrow(/54 chars/);
  });

  it("rejects invalid facelet characters", () => {
    const bad = "U".repeat(53) + "Z";
    expect(() =>
      createCubeSequence({ startFacelet: bad, moves: [] }),
    ).toThrow(/invalid char/);
  });

  it("rejects invalid move strings", () => {
    expect(() =>
      createCubeSequence({
        startFacelet: SOLVED,
        moves: ["R2"] as unknown as MoveStr[],
      }),
    ).toThrow(/invalid move/);
  });

  it("rejects non-positive msPerMove", () => {
    expect(() =>
      createCubeSequence({ startFacelet: SOLVED, moves: ["R"], msPerMove: 0 }),
    ).toThrow(/msPerMove/);
    expect(() =>
      createCubeSequence({ startFacelet: SOLVED, moves: ["R"], msPerMove: -1 }),
    ).toThrow(/msPerMove/);
  });

  it("rejects negative gapMs", () => {
    expect(() =>
      createCubeSequence({ startFacelet: SOLVED, moves: ["R"], gapMs: -1 }),
    ).toThrow(/gapMs/);
  });
});

describe("createCubeSequence — state machine transitions", () => {
  it("idle → play() → playing", () => {
    const s = makeSeq();
    s.play();
    expect(s.status).toBe("playing");
  });

  it("playing → pause() → paused; timestamp frozen", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(300);
    const tsBeforePause = s.timestamp;
    expect(tsBeforePause).toBeGreaterThan(0);
    s.pause();
    expect(s.status).toBe("paused");
    // After more wall time, the timestamp does NOT advance.
    await vi.advanceTimersByTimeAsync(500);
    expect(s.timestamp).toBe(tsBeforePause);
  });

  it("paused → play() → playing, resumes from current timestamp", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(300);
    s.pause();
    const tsAtPause = s.timestamp;
    s.play();
    expect(s.status).toBe("playing");
    await vi.advanceTimersByTimeAsync(200);
    expect(s.timestamp).toBeGreaterThanOrEqual(tsAtPause + 200 - 1);
    expect(s.timestamp).toBeLessThanOrEqual(tsAtPause + 200 + 50);
  });

  it("playing → run to end → ended; timestamp clamped to totalDurationMs", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500); // > totalDurationMs (2400)
    expect(s.status).toBe("ended");
    expect(s.timestamp).toBe(2400);
    expect(s.currentMoveIndex).toBe(3);
    expect(s.currentMoveProgress).toBe(1);
  });

  it("ended → play() → playing from timestamp 0 (replay semantics)", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    expect(s.status).toBe("ended");
    s.play();
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBe(0);
    expect(s.currentMoveIndex).toBe(0);
  });

  it("play() while playing is a no-op", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(200);
    const tsBefore = s.timestamp;
    s.play(); // no-op
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBe(tsBefore);
  });

  it("pause() while idle is a no-op", () => {
    const s = makeSeq();
    s.pause();
    expect(s.status).toBe("idle");
  });

  it("pause() while ended is a no-op", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    expect(s.status).toBe("ended");
    s.pause();
    expect(s.status).toBe("ended");
  });
});

describe("createCubeSequence — replay()", () => {
  it("from idle: resets to 0 and starts playing", () => {
    const s = makeSeq();
    s.replay();
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBe(0);
  });

  it("from paused mid-sequence: resets to 0 and plays", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(800);
    s.pause();
    expect(s.timestamp).toBeGreaterThan(0);
    s.replay();
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBe(0);
  });

  it("from ended: same as replay", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replay();
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBe(0);
  });

  it("empty moves: no-op (status stays idle)", () => {
    const s = createCubeSequence({ startFacelet: SOLVED, moves: [] });
    s.replay();
    expect(s.status).toBe("idle");
  });
});

describe("createCubeSequence — seek()", () => {
  it("seek(0) on idle keeps idle", () => {
    const s = makeSeq();
    s.seek(0);
    expect(s.status).toBe("idle");
    expect(s.timestamp).toBe(0);
    expect(s.currentMoveIndex).toBe(-1);
  });

  it("seek(midMove) from idle → paused at the new timestamp", () => {
    const s = makeSeq();
    s.seek(300); // mid-move 0
    expect(s.status).toBe("paused");
    expect(s.timestamp).toBe(300);
    expect(s.currentMoveIndex).toBe(0);
    expect(s.currentMoveProgress).toBeCloseTo(0.5);
  });

  it("seek(totalDurationMs) → ended", () => {
    const s = makeSeq();
    s.seek(2400);
    expect(s.status).toBe("ended");
    expect(s.timestamp).toBe(2400);
    expect(s.currentMoveIndex).toBe(3);
    expect(s.currentMoveProgress).toBe(1);
  });

  it("seek clamps below 0 to 0", () => {
    const s = makeSeq();
    s.seek(800);
    s.seek(-500);
    expect(s.timestamp).toBe(0);
  });

  it("seek clamps above totalDurationMs to totalDurationMs", () => {
    const s = makeSeq();
    s.seek(99999);
    expect(s.timestamp).toBe(2400);
    expect(s.status).toBe("ended");
  });

  it("seek while playing keeps playing from new timestamp", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(100);
    s.seek(1500); // mid-move 2
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBe(1500);
    expect(s.currentMoveIndex).toBe(2);
    expect(s.currentMoveProgress).toBeCloseTo(0.5);
    // And rAF continues from there.
    await vi.advanceTimersByTimeAsync(200);
    expect(s.timestamp).toBeGreaterThan(1500);
  });

  it("seek to exact move boundary → that move with progress=0", () => {
    const s = makeSeq();
    // timestamp 600 = exact start of move 1 (move 0 just finished).
    s.seek(600);
    expect(s.currentMoveIndex).toBe(1);
    expect(s.currentMoveProgress).toBe(0);
  });
});

describe("createCubeSequence — seekToMove()", () => {
  it("seekToMove(0) → timestamp 0, currentMoveIndex 0, progress 0", () => {
    const s = makeSeq();
    s.seekToMove(0);
    expect(s.timestamp).toBe(0);
    // Note: seek(0) while !wasPlaying → idle. Per the contract this
    // is intentional — seekToMove(0) on idle stays idle.
    expect(s.status).toBe("idle");
    expect(s.currentMoveIndex).toBe(-1); // idle+timestamp 0 → -1 per spec
  });

  it("seekToMove(i) → timestamp = i * slotMs, currentMoveIndex = i", () => {
    const s = makeSeq({ msPerMove: 500, gapMs: 100 });
    s.seekToMove(2);
    expect(s.timestamp).toBe(2 * 600); // (500 + 100) * 2
    expect(s.currentMoveIndex).toBe(2);
    expect(s.currentMoveProgress).toBe(0);
  });

  it("seekToMove clamps i to [0, moves.length - 1]", () => {
    const s = makeSeq();
    s.seekToMove(99);
    expect(s.currentMoveIndex).toBe(3);
    expect(s.timestamp).toBe(3 * 600); // last move
    s.seekToMove(-5);
    expect(s.timestamp).toBe(0);
  });

  it("empty moves: no-op", () => {
    const s = createCubeSequence({ startFacelet: SOLVED, moves: [] });
    s.seekToMove(0);
    expect(s.status).toBe("idle");
  });
});

describe("createCubeSequence — gapMs", () => {
  it("during gap dwell: currentMoveIndex=prev, progress=1", () => {
    const s = makeSeq({ msPerMove: 600, gapMs: 100 });
    // totalDurationMs = 4 * 700 = 2800.
    // Slot 0 covers [0, 700). Move portion [0, 600). Gap [600, 700).
    s.seek(650); // mid-gap after move 0.
    expect(s.currentMoveIndex).toBe(0);
    expect(s.currentMoveProgress).toBe(1);
  });

  it("at exact next-slot boundary: next move with progress=0", () => {
    const s = makeSeq({ msPerMove: 600, gapMs: 100 });
    s.seek(700); // start of move 1
    expect(s.currentMoveIndex).toBe(1);
    expect(s.currentMoveProgress).toBe(0);
  });

  it("totalDurationMs accounts for gapMs", () => {
    const s = makeSeq({ msPerMove: 600, gapMs: 100 });
    expect(s.totalDurationMs).toBe(2800);
  });
});

describe("createCubeSequence — subscribe", () => {
  it("listener fires on play / pause / seek / replay", () => {
    const s = makeSeq();
    const cb = vi.fn();
    s.subscribe(cb);
    s.play();
    s.pause();
    s.seek(300);
    s.replay();
    expect(cb.mock.calls.length).toBeGreaterThanOrEqual(4);
  });

  it("listener fires on each rAF tick while playing", async () => {
    const s = makeSeq();
    const cb = vi.fn();
    s.subscribe(cb);
    s.play();
    cb.mockClear();
    await vi.advanceTimersByTimeAsync(100);
    // At ~60 FPS, 100ms ≈ 6 frames. Allow slack for fake-timer cadence.
    expect(cb.mock.calls.length).toBeGreaterThan(0);
  });

  it("unsubscribe stops further notifications", () => {
    const s = makeSeq();
    const cb = vi.fn();
    const off = s.subscribe(cb);
    s.play();
    off();
    cb.mockClear();
    s.pause();
    s.seek(500);
    expect(cb).not.toHaveBeenCalled();
  });

  it("a throwing listener does not break the loop or other listeners", () => {
    const s = makeSeq();
    const bad = vi.fn(() => {
      throw new Error("boom");
    });
    const good = vi.fn();
    const consoleErr = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});
    s.subscribe(bad);
    s.subscribe(good);
    s.play();
    expect(good).toHaveBeenCalled();
    expect(consoleErr).toHaveBeenCalled();
    consoleErr.mockRestore();
  });
});

describe("createCubeSequence — rAF cleanup", () => {
  it("no further rAF callbacks after status reaches 'ended'", async () => {
    const s = makeSeq();
    const cb = vi.fn();
    s.subscribe(cb);
    s.play();
    await vi.advanceTimersByTimeAsync(2500); // run to end
    expect(s.status).toBe("ended");
    cb.mockClear();
    // Advance more wall time — no rAF should fire.
    await vi.advanceTimersByTimeAsync(1000);
    expect(cb).not.toHaveBeenCalled();
  });

  it("pause cancels the rAF loop", async () => {
    const s = makeSeq();
    const cb = vi.fn();
    s.subscribe(cb);
    s.play();
    await vi.advanceTimersByTimeAsync(100);
    s.pause();
    cb.mockClear();
    await vi.advanceTimersByTimeAsync(500);
    expect(cb).not.toHaveBeenCalled();
  });
});

describe("createCubeSequence — currentMoveIndex/Progress derivation", () => {
  it("at timestamp 0 idle: index=-1, progress=0", () => {
    const s = makeSeq();
    expect(s.currentMoveIndex).toBe(-1);
    expect(s.currentMoveProgress).toBe(0);
  });

  it("at timestamp 0 playing: index=0, progress=0", () => {
    const s = makeSeq();
    s.play();
    expect(s.currentMoveIndex).toBe(0);
    expect(s.currentMoveProgress).toBe(0);
  });

  it("midpoint of move 0: index=0, progress=0.5", () => {
    const s = makeSeq();
    s.seek(300);
    expect(s.currentMoveIndex).toBe(0);
    expect(s.currentMoveProgress).toBeCloseTo(0.5);
  });

  it("midpoint of move 2: index=2, progress=0.5", () => {
    const s = makeSeq();
    s.seek(1500);
    expect(s.currentMoveIndex).toBe(2);
    expect(s.currentMoveProgress).toBeCloseTo(0.5);
  });

  it("at end: index=last, progress=1", () => {
    const s = makeSeq();
    s.seek(2400);
    expect(s.currentMoveIndex).toBe(3);
    expect(s.currentMoveProgress).toBe(1);
  });
});
