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

describe("createCubeSequence — replayWithReverse() choreography", () => {
  // The choreography (DUR_REVERSE_MS=150 → DUR_PAUSE_MS=250 → forward
  // play) only engages when the sequence is at or past 99% of its
  // last move. From any other state, replayWithReverse collapses to
  // a forward-only replay() so mid-flight clicks feel snappy.

  it("from idle: collapses to replay() (forward only, no reverse leg)", () => {
    const s = makeSeq();
    s.replayWithReverse();
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBe(0);
  });

  it("from paused mid-sequence: collapses to replay() (forward only)", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(800); // mid-move 1
    s.pause();
    expect(s.timestamp).toBeGreaterThan(0);
    s.replayWithReverse();
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBe(0);
  });

  it("from playing mid-sequence: collapses to replay() (forward only)", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(800);
    s.replayWithReverse();
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBe(0);
  });

  it("from ended: enters 'reversing' status with timestamp at totalDurationMs", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    expect(s.status).toBe("ended");
    s.replayWithReverse();
    expect(s.status).toBe("reversing");
    expect(s.timestamp).toBe(2400);
  });

  it("reverse leg: timestamp interpolates linearly from totalDurationMs → 0 over DUR_REVERSE_MS (150 ms)", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500); // run to end
    s.replayWithReverse();
    // Tick ~75 ms in (half of DUR_REVERSE_MS=150) — timestamp should
    // be ~half of totalDurationMs (~1200). Loose bound to absorb the
    // setTimeout(0) rAF shim's coarse cadence.
    await vi.advanceTimersByTimeAsync(75);
    expect(s.status).toBe("reversing");
    expect(s.timestamp).toBeLessThan(2400);
    expect(s.timestamp).toBeGreaterThan(0);
  });

  it("reverse leg → pausing at timestamp 0 after DUR_REVERSE_MS", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverse();
    // Step past DUR_REVERSE_MS (150 ms) — reverse leg should land at
    // timestamp 0 and transition to 'pausing'.
    await vi.advanceTimersByTimeAsync(200);
    expect(s.status).toBe("pausing");
    expect(s.timestamp).toBe(0);
  });

  it("pausing → playing at timestamp 0 after DUR_PAUSE_MS", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverse();
    // Step past DUR_REVERSE_MS + DUR_PAUSE_MS (150 + 250 = 400) — the
    // pause should expire and forward play should begin.
    await vi.advanceTimersByTimeAsync(450);
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBeGreaterThanOrEqual(0);
    expect(s.timestamp).toBeLessThan(2400);
  });

  it("end-to-end: reversing → pausing → playing → ended (total = 150+250+2400 = 2800 ms)", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverse();
    // Step well past 2800 ms; should land on 'ended' again.
    await vi.advanceTimersByTimeAsync(3000);
    expect(s.status).toBe("ended");
    expect(s.timestamp).toBe(2400);
  });

  it("pause() during reverse leg → status 'paused', timestamp frozen mid-rewind", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverse();
    await vi.advanceTimersByTimeAsync(50);
    const tsMidReverse = s.timestamp;
    expect(s.status).toBe("reversing");
    s.pause();
    expect(s.status).toBe("paused");
    // Timestamp does not continue to fall.
    await vi.advanceTimersByTimeAsync(500);
    expect(s.timestamp).toBe(tsMidReverse);
  });

  it("pause() during pause leg → status 'paused', timestamp held at 0", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverse();
    await vi.advanceTimersByTimeAsync(200); // into pause leg
    expect(s.status).toBe("pausing");
    expect(s.timestamp).toBe(0);
    s.pause();
    expect(s.status).toBe("paused");
    expect(s.timestamp).toBe(0);
  });

  it("play() during reverse leg → abandons leg, forward play resumes from current timestamp", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverse();
    await vi.advanceTimersByTimeAsync(75); // mid-reverse
    const tsMidReverse = s.timestamp;
    expect(s.status).toBe("reversing");
    s.play();
    expect(s.status).toBe("playing");
    // Forward play continues forward from where the reverse leg left off.
    await vi.advanceTimersByTimeAsync(100);
    expect(s.timestamp).toBeGreaterThan(tsMidReverse);
  });

  it("seek() during reverse leg → abandons leg, lands at requested timestamp", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverse();
    await vi.advanceTimersByTimeAsync(50);
    expect(s.status).toBe("reversing");
    s.seek(1500);
    expect(s.timestamp).toBe(1500);
    // Seek from non-playing → paused. The reverse status was not
    // "playing", so wasPlaying=false and we land in paused.
    expect(s.status).toBe("paused");
  });

  it("replay() during reverse leg → resets to 0 and starts forward play", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverse();
    await vi.advanceTimersByTimeAsync(50);
    expect(s.status).toBe("reversing");
    s.replay();
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBe(0);
  });

  it("listener fires on each leg transition (reversing → pausing → playing → ended)", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    const seen: string[] = [];
    s.subscribe(() => {
      const top = seen[seen.length - 1];
      if (top !== s.status) seen.push(s.status);
    });
    s.replayWithReverse();
    await vi.advanceTimersByTimeAsync(3000);
    expect(seen).toContain("reversing");
    expect(seen).toContain("pausing");
    expect(seen).toContain("playing");
    expect(seen).toContain("ended");
    expect(seen.indexOf("reversing")).toBeLessThan(seen.indexOf("pausing"));
    expect(seen.indexOf("pausing")).toBeLessThan(seen.indexOf("playing"));
    expect(seen.indexOf("playing")).toBeLessThan(seen.indexOf("ended"));
  });

  it("empty moves: replayWithReverse is a no-op (status stays idle)", () => {
    const s = createCubeSequence({ startFacelet: SOLVED, moves: [] });
    s.replayWithReverse();
    expect(s.status).toBe("idle");
  });

  it("from progress > 0.99 (very near end but not exactly ended): also engages reverse leg", () => {
    const s = makeSeq();
    // Land just past 99% of the last move: timestamp = 3*600 + 0.995*600 = 2397
    s.seek(2397);
    expect(s.status).toBe("paused");
    expect(s.currentMoveIndex).toBe(3);
    expect(s.currentMoveProgress).toBeGreaterThan(0.99);
    s.replayWithReverse();
    expect(s.status).toBe("reversing");
  });
});

describe("createCubeSequence — replayWithReverseHold + releaseHold", () => {
  // Press-and-hold variant of replayWithReverse: reverse leg lands at
  // timestamp=0 and dwells INDEFINITELY in `pausing` until
  // `releaseHold()` is called. A `releaseHold()` mid-reverse leg
  // "quick-click collapses" — the reverse finishes naturally and skips
  // the held pause to go straight into forward play.

  it("from ended: enters reversing → held pausing at progress=0; stays held until releaseHold()", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    expect(s.status).toBe("ended");

    s.replayWithReverseHold();
    expect(s.status).toBe("reversing");
    expect(s.timestamp).toBe(2400);

    // After DUR_REVERSE_MS (150) the reverse leg lands at progress=0
    // and the held pause begins.
    await vi.advanceTimersByTimeAsync(200);
    expect(s.status).toBe("pausing");
    expect(s.timestamp).toBe(0);

    // Held indefinitely — DUR_PAUSE_MS (250) elapses, still pausing.
    await vi.advanceTimersByTimeAsync(500);
    expect(s.status).toBe("pausing");
    expect(s.timestamp).toBe(0);

    // Way past DUR_PAUSE_MS — still pausing.
    await vi.advanceTimersByTimeAsync(5000);
    expect(s.status).toBe("pausing");
    expect(s.timestamp).toBe(0);

    // releaseHold → forward play from progress=0.
    s.releaseHold();
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBe(0);
    expect(s.currentMoveIndex).toBe(0);

    // Forward leg runs to ended.
    await vi.advanceTimersByTimeAsync(2500);
    expect(s.status).toBe("ended");
    expect(s.timestamp).toBe(2400);
  });

  it("quick-click collapse: releaseHold() mid-reverse → reverse completes → straight to forward (no held pause)", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    expect(s.status).toBe("ended");

    s.replayWithReverseHold();
    expect(s.status).toBe("reversing");

    // Release before reverse leg completes (DUR_REVERSE_MS = 150).
    await vi.advanceTimersByTimeAsync(50);
    expect(s.status).toBe("reversing");
    s.releaseHold();
    // Still reversing — the release is queued.
    expect(s.status).toBe("reversing");

    // Let the reverse leg run its remaining ~100ms. On hitting
    // timestamp=0 the queued release short-circuits the held-pause and
    // we transition directly to playing.
    await vi.advanceTimersByTimeAsync(120);
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBeGreaterThanOrEqual(0);
    // Confirm no held pause ever occurred — total time from
    // replayWithReverseHold to end = DUR_REVERSE_MS + forward time
    // (~150 + 2400 = 2550), not the 2800 the fixed-pause variant takes.
    await vi.advanceTimersByTimeAsync(2500);
    expect(s.status).toBe("ended");
  });

  it("releaseHold during reverse at ~50% completion: reverse runs to 0, then forward (no pause)", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverseHold();
    expect(s.status).toBe("reversing");

    await vi.advanceTimersByTimeAsync(75); // ~half of DUR_REVERSE_MS=150
    expect(s.status).toBe("reversing");
    s.releaseHold();
    expect(s.status).toBe("reversing");

    // Walk through the rest of the reverse leg + a tiny bit. The leg
    // should complete and we should be in "playing" (not "pausing").
    await vi.advanceTimersByTimeAsync(120);
    expect(s.status).toBe("playing");
  });

  it("releaseHold during reverse at ~90% completion: still queues release, lands in forward", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverseHold();
    expect(s.status).toBe("reversing");

    await vi.advanceTimersByTimeAsync(135); // ~90% of 150
    // Still reversing (timestamp very near 0 but tick has not yet
    // crossed it). releaseHold queues for the imminent transition.
    if (s.status === "reversing") {
      s.releaseHold();
    } else {
      // Already past the reverse leg → not the case we're testing.
      // Bail out of the assertion path.
      expect(s.status).toBe("pausing");
      s.releaseHold();
    }
    await vi.advanceTimersByTimeAsync(100);
    expect(s.status).toBe("playing");
  });

  it("releaseHold from non-hold states (idle/playing/paused/ended): no-op, no error, no status change", async () => {
    const s = makeSeq();
    // idle
    expect(s.status).toBe("idle");
    s.releaseHold();
    expect(s.status).toBe("idle");
    expect(s.timestamp).toBe(0);

    // playing
    s.play();
    await vi.advanceTimersByTimeAsync(100);
    const tsPlaying = s.timestamp;
    expect(s.status).toBe("playing");
    s.releaseHold();
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBe(tsPlaying);

    // paused
    s.pause();
    expect(s.status).toBe("paused");
    s.releaseHold();
    expect(s.status).toBe("paused");

    // ended
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    expect(s.status).toBe("ended");
    s.releaseHold();
    expect(s.status).toBe("ended");
  });

  it("from idle (timestamp=0): skips reverse leg, enters held pause immediately at timestamp=0", () => {
    const s = makeSeq();
    expect(s.status).toBe("idle");
    s.replayWithReverseHold();
    // No reverse leg — straight into pausing at timestamp=0.
    expect(s.status).toBe("pausing");
    expect(s.timestamp).toBe(0);
  });

  it("from idle: releaseHold immediately → forward play from progress=0", async () => {
    const s = makeSeq();
    s.replayWithReverseHold();
    expect(s.status).toBe("pausing");
    s.releaseHold();
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBe(0);
    await vi.advanceTimersByTimeAsync(2500);
    expect(s.status).toBe("ended");
  });

  it("from playing mid-forward (progress p): reverse from current timestamp, duration scaled by p", async () => {
    const s = makeSeq();
    s.play();
    // Run to ~25% of the sequence (timestamp ≈ 600 of 2400).
    await vi.advanceTimersByTimeAsync(600);
    const tsAtPress = s.timestamp;
    expect(tsAtPress).toBeGreaterThan(400);
    expect(tsAtPress).toBeLessThan(900);

    s.replayWithReverseHold();
    expect(s.status).toBe("reversing");
    // Reverse duration scales by tsAtPress / 2400 → ~25% * 150ms ≈ 38ms.
    // Step well past that and assert we're in held-pause at 0.
    await vi.advanceTimersByTimeAsync(80);
    expect(s.status).toBe("pausing");
    expect(s.timestamp).toBe(0);

    // Still held even after a long wait.
    await vi.advanceTimersByTimeAsync(1000);
    expect(s.status).toBe("pausing");
  });

  it("from paused mid-sequence: reverse from current ts → held-pause", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(800);
    s.pause();
    expect(s.status).toBe("paused");
    const tsAtPause = s.timestamp;
    expect(tsAtPause).toBeGreaterThan(0);

    s.replayWithReverseHold();
    expect(s.status).toBe("reversing");
    // Wait past the scaled reverse duration.
    await vi.advanceTimersByTimeAsync(200);
    expect(s.status).toBe("pausing");
    expect(s.timestamp).toBe(0);
  });

  it("from reversing (already in a hold): no-op", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverseHold();
    expect(s.status).toBe("reversing");
    const tsBefore = s.timestamp;
    s.replayWithReverseHold(); // no-op
    expect(s.status).toBe("reversing");
    expect(s.timestamp).toBe(tsBefore);
  });

  it("from pausing held: no-op (already at destination)", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverseHold();
    await vi.advanceTimersByTimeAsync(200); // into held pause
    expect(s.status).toBe("pausing");
    s.replayWithReverseHold(); // no-op
    expect(s.status).toBe("pausing");
    expect(s.timestamp).toBe(0);
  });

  it("pause() during held pause → status='paused' at timestamp=0; subsequent play() runs forward", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverseHold();
    await vi.advanceTimersByTimeAsync(200);
    expect(s.status).toBe("pausing");
    expect(s.timestamp).toBe(0);

    s.pause();
    expect(s.status).toBe("paused");
    expect(s.timestamp).toBe(0);

    // play() from paused at timestamp=0 → forward.
    s.play();
    expect(s.status).toBe("playing");
    await vi.advanceTimersByTimeAsync(2500);
    expect(s.status).toBe("ended");
  });

  it("seek() during held pause: jumps to seek target, exits hold", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverseHold();
    await vi.advanceTimersByTimeAsync(200);
    expect(s.status).toBe("pausing");

    s.seek(1500);
    expect(s.timestamp).toBe(1500);
    // wasPlaying=false (pausing), and not the (clamped===0 && !wasPlaying)
    // branch, so we land in "paused".
    expect(s.status).toBe("paused");

    // Subsequent releaseHold is a no-op (hold ended on seek).
    s.releaseHold();
    expect(s.status).toBe("paused");
  });

  it("listener fires on each phase transition (reversing → pausing → playing → ended) for held flow", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    const seen: string[] = [];
    s.subscribe(() => {
      const top = seen[seen.length - 1];
      if (top !== s.status) seen.push(s.status);
    });
    s.replayWithReverseHold();
    // Wait past reverse leg → pausing held.
    await vi.advanceTimersByTimeAsync(200);
    expect(seen).toContain("reversing");
    expect(seen).toContain("pausing");
    // Release → playing → ended.
    s.releaseHold();
    await vi.advanceTimersByTimeAsync(2500);
    expect(seen).toContain("playing");
    expect(seen).toContain("ended");
    expect(seen.indexOf("reversing")).toBeLessThan(seen.indexOf("pausing"));
    expect(seen.indexOf("pausing")).toBeLessThan(seen.indexOf("playing"));
    expect(seen.indexOf("playing")).toBeLessThan(seen.indexOf("ended"));
  });

  it("empty moves: replayWithReverseHold + releaseHold are both no-ops", () => {
    const s = createCubeSequence({ startFacelet: SOLVED, moves: [] });
    s.replayWithReverseHold();
    expect(s.status).toBe("idle");
    s.releaseHold();
    expect(s.status).toBe("idle");
  });

  it("replayWithReverse() during held pause: ends the hold and engages the fixed-pause choreography", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverseHold();
    await vi.advanceTimersByTimeAsync(200);
    expect(s.status).toBe("pausing");
    expect(s.timestamp).toBe(0);

    // replayWithReverse() inspects state — at timestamp=0 with status
    // "pausing" the "atEnd" check is false, so it collapses to
    // replay() (seek 0 + forward). The hold ends regardless.
    s.replayWithReverse();
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBe(0);
    // Subsequent releaseHold is a no-op.
    s.releaseHold();
    expect(s.status).toBe("playing");
  });

  it("replay() during held pause: ends the hold and starts forward play from 0", async () => {
    const s = makeSeq();
    s.play();
    await vi.advanceTimersByTimeAsync(2500);
    s.replayWithReverseHold();
    await vi.advanceTimersByTimeAsync(200);
    expect(s.status).toBe("pausing");

    s.replay();
    expect(s.status).toBe("playing");
    expect(s.timestamp).toBe(0);
    s.releaseHold(); // no-op
    expect(s.status).toBe("playing");
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
