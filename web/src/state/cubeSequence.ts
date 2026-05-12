// Framework-agnostic cube-sequence controller.
//
// A `CubeSequence` walks a (startFacelet, moves[]) tuple through
// animated playback under an external clock (rAF). It owns its
// timestamp, derives `currentMoveIndex / currentMoveProgress` from
// timestamp on demand, and notifies subscribers on state change.
//
// This module is pure TypeScript — no React. The React hook in
// `web/src/hooks/useCubeSequence.ts` wraps it with
// `useSyncExternalStore` for component-driven re-renders. Block D's
// cross-surface scrubber sync will lift one sequence to a context
// and subscribe from multiple components; same factory, same
// contract, no fork.
//
// Structural prototype: the inline `createCubeSequence` factory in
// `web/preview/flat-cube-animated.html` (A·P0). This is the typed
// production port of that factory; the preview keeps its inline
// standalone copy as the design playground.
//
// References:
//   - `plans/m9-cube-animation-system.md` — block plan, architecture
//     section (controller contract).
//   - `web/src/components/cubeNetAnimations.ts` (A·P1) — DUR_FORWARD_MS
//     re-used as default `msPerMove`.

import { DUR_FORWARD_MS } from "../components/cubeNetAnimations";
import { FACELET_MOVES, type MoveStr } from "./faceletMoves";

// ---------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------

export type CubeSequenceStatus = "idle" | "playing" | "paused" | "ended";

export interface CubeSequenceSpec {
  /** 54-char URFDLB facelet to start from. */
  startFacelet: string;
  /** Move sequence to walk through. */
  moves: ReadonlyArray<MoveStr>;
  /** Milliseconds per move (forward play). Default: DUR_FORWARD_MS (600). */
  msPerMove?: number;
  /** Milliseconds of dwell after each move (showing post-move state).
   *  Default: 0 (back-to-back moves, no gap). */
  gapMs?: number;
  /** If true, start playing immediately on creation. Default: false. */
  autoplay?: boolean;
}

export interface CubeSequence {
  // ---- read-only state (current values; mutates over time) ----
  readonly status: CubeSequenceStatus;
  /** Current playback position in ms, in [0, totalDurationMs]. */
  readonly timestamp: number;
  /** Total duration = moves.length * (msPerMove + gapMs). */
  readonly totalDurationMs: number;
  /**
   * Index of the move currently being animated. `-1` before any move
   * starts (status=idle AND timestamp=0), `moves.length - 1` once
   * status=ended. Within the gap-dwell after move `i`, this stays at
   * `i` with `currentMoveProgress=1`.
   */
  readonly currentMoveIndex: number;
  /** Progress within the current move, in [0, 1]. Held at 1 during
   *  the gap-dwell after a move; jumps to 0 when the next move starts. */
  readonly currentMoveProgress: number;
  readonly startFacelet: string;
  readonly moves: ReadonlyArray<MoveStr>;
  readonly msPerMove: number;
  readonly gapMs: number;

  // ---- controls ----
  /** idle/paused → playing. ended → replay (seek 0 + play). no-op while playing or empty moves. */
  play(): void;
  /** playing → paused. no-op otherwise. */
  pause(): void;
  /** Clamp `ms` to [0, totalDurationMs] and jump there.
   *  Status transitions per the contract documented inline. */
  seek(ms: number): void;
  /** Seek to the start of move `i` (timestamp = i * slotMs). */
  seekToMove(i: number): void;
  /** seek(0) + play(); transitions status to "playing". */
  replay(): void;

  // ---- subscription ----
  /** Subscribe to state-change notifications. Returns an unsubscribe fn.
   *  Listeners fire on play / pause / seek / replay AND on each rAF
   *  tick while playing. Errors thrown by listeners are caught and
   *  console.error'd so a misbehaving listener can't break the loop. */
  subscribe(listener: () => void): () => void;
}

// ---------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------

const FACELET_CHARS = new Set(["U", "R", "F", "D", "L", "B"]);

function validateSpec(spec: CubeSequenceSpec): void {
  const { startFacelet, moves, msPerMove, gapMs } = spec;
  if (typeof startFacelet !== "string" || startFacelet.length !== 54) {
    throw new Error(
      `createCubeSequence: startFacelet must be 54 chars, got length ${
        typeof startFacelet === "string" ? startFacelet.length : "n/a"
      }`,
    );
  }
  for (let i = 0; i < startFacelet.length; i++) {
    const c = startFacelet.charAt(i);
    if (!FACELET_CHARS.has(c)) {
      throw new Error(
        `createCubeSequence: startFacelet contains invalid char "${c}" at index ${i}`,
      );
    }
  }
  for (const m of moves) {
    if (!(m in FACELET_MOVES)) {
      throw new Error(
        `createCubeSequence: invalid move "${m}" (allowed QTM strings only)`,
      );
    }
  }
  if (msPerMove !== undefined && (!Number.isFinite(msPerMove) || msPerMove <= 0)) {
    throw new Error(
      `createCubeSequence: msPerMove must be a positive finite number, got ${msPerMove}`,
    );
  }
  if (gapMs !== undefined && (!Number.isFinite(gapMs) || gapMs < 0)) {
    throw new Error(
      `createCubeSequence: gapMs must be a non-negative finite number, got ${gapMs}`,
    );
  }
}

// ---------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------

export function createCubeSequence(spec: CubeSequenceSpec): CubeSequence {
  validateSpec(spec);
  const startFacelet = spec.startFacelet;
  const moves: ReadonlyArray<MoveStr> = spec.moves;
  const msPerMove = spec.msPerMove ?? DUR_FORWARD_MS;
  const gapMs = spec.gapMs ?? 0;
  const autoplay = spec.autoplay ?? false;

  const slotMs = msPerMove + gapMs;
  const totalDurationMs = moves.length * slotMs;

  // ----- internal mutable state -----
  let status: CubeSequenceStatus = "idle";
  let timestamp = 0; // ms, in [0, totalDurationMs]
  let rafId = 0;
  let rafStartWall = 0; // performance.now() when the current play leg began
  let rafStartTimestamp = 0; // controller timestamp when the current play leg began
  const listeners = new Set<() => void>();

  function notify(): void {
    for (const fn of listeners) {
      try {
        fn();
      } catch (e) {
        // Don't let one bad listener break the loop.
        console.error(e);
      }
    }
  }

  // Derive (currentMoveIndex, currentMoveProgress) from the current
  // timestamp + status. Pure — only reads internal state.
  //
  // Boundary convention: at timestamp = i*slotMs (i in [0, moves.length-1]),
  // we report (i, 0) — the START of move i, not the end of move i-1.
  // Within the gap (localMs ∈ [msPerMove, slotMs)) we hold (slot, 1.0)
  // to show the post-move state. At timestamp = totalDurationMs, we
  // report (moves.length-1, 1).
  function indexAndProgress(): {
    currentMoveIndex: number;
    currentMoveProgress: number;
  } {
    if (moves.length === 0) {
      return { currentMoveIndex: -1, currentMoveProgress: 0 };
    }
    // Pre-start: only when truly idle at timestamp 0 (matches preview
    // semantics — once play() advances timestamp, we're on move 0).
    if (status === "idle" && timestamp === 0) {
      return { currentMoveIndex: -1, currentMoveProgress: 0 };
    }
    if (timestamp >= totalDurationMs) {
      return {
        currentMoveIndex: moves.length - 1,
        currentMoveProgress: 1,
      };
    }
    const slot = Math.floor(timestamp / slotMs);
    const localMs = timestamp - slot * slotMs;
    const progress = localMs >= msPerMove ? 1 : localMs / msPerMove;
    return {
      currentMoveIndex: Math.min(slot, moves.length - 1),
      currentMoveProgress: progress,
    };
  }

  function tick(now: number): void {
    if (status !== "playing") {
      rafId = 0;
      return;
    }
    const elapsed = now - rafStartWall;
    timestamp = rafStartTimestamp + elapsed;
    if (timestamp >= totalDurationMs) {
      timestamp = totalDurationMs;
      status = "ended";
      rafId = 0;
      notify();
      return;
    }
    notify();
    rafId = requestAnimationFrame(tick);
  }

  function startRaf(): void {
    if (rafId) cancelAnimationFrame(rafId);
    rafStartWall = performance.now();
    rafStartTimestamp = timestamp;
    rafId = requestAnimationFrame(tick);
  }

  function play(): void {
    if (moves.length === 0) return;
    if (status === "playing") return;
    if (status === "ended") {
      // play() at end → replay semantics: rewind to 0, then play.
      timestamp = 0;
    }
    status = "playing";
    startRaf();
    notify();
  }

  function pause(): void {
    if (status !== "playing") return;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
    status = "paused";
    notify();
  }

  function seek(ms: number): void {
    const clamped = Math.max(0, Math.min(totalDurationMs, ms));
    const wasPlaying = status === "playing";
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
    timestamp = clamped;
    // Status transitions:
    //   was playing + clamped < end       → keep playing from new ts
    //   clamped >= totalDurationMs        → ended
    //   clamped === 0 && !wasPlaying      → idle (reset to fresh start)
    //   otherwise (mid-sequence, not playing) → paused
    if (wasPlaying && clamped < totalDurationMs) {
      status = "playing";
      startRaf();
    } else if (clamped >= totalDurationMs) {
      status = "ended";
    } else if (clamped === 0 && !wasPlaying) {
      status = "idle";
    } else {
      status = "paused";
    }
    notify();
  }

  function seekToMove(i: number): void {
    if (moves.length === 0) return;
    const clamped = Math.max(0, Math.min(moves.length - 1, i));
    seek(clamped * slotMs);
  }

  function replay(): void {
    if (moves.length === 0) return;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
    timestamp = 0;
    status = "playing";
    startRaf();
    notify();
  }

  function subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }

  // The CubeSequence handle: getters expose live state; methods drive
  // it. Caller can read state fields at any moment (no snapshot
  // staleness); the React hook bridges this onto React's re-render
  // model via `useSyncExternalStore`.
  const handle: CubeSequence = {
    get status() {
      return status;
    },
    get timestamp() {
      return timestamp;
    },
    get totalDurationMs() {
      return totalDurationMs;
    },
    get currentMoveIndex() {
      return indexAndProgress().currentMoveIndex;
    },
    get currentMoveProgress() {
      return indexAndProgress().currentMoveProgress;
    },
    get startFacelet() {
      return startFacelet;
    },
    get moves() {
      return moves;
    },
    get msPerMove() {
      return msPerMove;
    },
    get gapMs() {
      return gapMs;
    },
    play,
    pause,
    seek,
    seekToMove,
    replay,
    subscribe,
  };

  if (autoplay) {
    play();
  }
  return handle;
}
