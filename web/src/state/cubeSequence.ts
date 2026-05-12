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
//   - `web/src/components/cube2DKinematics.ts` (A·P1) — DUR_FORWARD_MS
//     re-used as default `msPerMove`.

import {
  DUR_FORWARD_MS,
  DUR_PAUSE_MS,
  DUR_REVERSE_MS,
} from "../components/cube2DKinematics";
import { FACELET_MOVES, type MoveStr } from "./faceletMoves";

// ---------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------

// Statuses:
//   idle      — never played; timestamp at 0; currentMoveIndex=-1.
//   playing   — forward play under rAF; timestamp advances toward end.
//   paused    — mid-sequence hold (timestamp frozen, no rAF running).
//   ended     — reached totalDurationMs; timestamp clamped there.
//   reversing — reverse leg in flight: timestamp decreases from the
//               leg's start value to 0. Used by both
//               `replayWithReverse()` (fixed DUR_REVERSE_MS) and
//               `replayWithReverseHold()` (DUR_REVERSE_MS scaled by the
//               normalized starting timestamp so partial reverses keep
//               the same per-unit speed).
//   pausing   — pause leg at timestamp 0 before the forward leg starts.
//               `replayWithReverse()` exits this leg automatically after
//               DUR_PAUSE_MS. `replayWithReverseHold()` enters a HELD
//               variant that dwells indefinitely until `releaseHold()`
//               is called.
//
// `reversing` and `pausing` are produced by the choreographed-replay
// family (`replayWithReverse()` / `replayWithReverseHold()`).
// `play() / pause() / seek() / replay()` never transition into them.
export type CubeSequenceStatus =
  | "idle"
  | "playing"
  | "paused"
  | "ended"
  | "reversing"
  | "pausing";

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
  /**
   * "Choreographed replay": if progress is at end (>= 0.99), run a
   * reverse leg (timestamp → 0 over DUR_REVERSE_MS) followed by a
   * pause leg (held at 0 for DUR_PAUSE_MS), then the standard forward
   * leg. From any other state (idle / playing / paused / mid-flight)
   * behaves like `replay()` — seek(0) + forward play. Mirrors the
   * rev5 preview's per-panel state machine: when the user clicks an
   * already-completed move, the cube "rewinds" first so the eye
   * registers the pre-move state before the forward animation plays
   * again. Statuses visited during the choreographed legs:
   *   reversing (DUR_REVERSE_MS) → pausing (DUR_PAUSE_MS) → playing.
   * Listeners fire on each leg transition + each rAF tick within a leg.
   */
  replayWithReverse(): void;
  /**
   * Press-and-hold variant of `replayWithReverse()`. Engages a reverse
   * leg into an INDEFINITELY-held pause at timestamp=0. Pair with
   * `releaseHold()` to resume forward play.
   *
   * Interruption matrix:
   *   - ended: standard reverse leg (DUR_REVERSE_MS) → held-pause.
   *   - playing (forward in flight at progress p∈(0,totalDurationMs)):
   *     cancel forward, run reverse leg from p → 0 with duration
   *     scaled by (p / totalDurationMs) so partial reverses keep the
   *     same per-unit speed. Then held-pause.
   *   - paused mid-sequence (timestamp > 0): same as playing — reverse
   *     from current timestamp to 0 (scaled duration), then held-pause.
   *   - idle (currentMoveIndex=-1, timestamp=0): skip reverse leg
   *     (nothing to rewind from); enter held-pause immediately at
   *     timestamp=0. Visually identical to staying idle, but state is
   *     consistent so a subsequent `releaseHold()` plays forward.
   *   - reversing or pausing (already in a hold): no-op.
   *
   * Quick-click collapse: if `releaseHold()` is called while still in
   * the reverse leg, the hold-pause is skipped — the reverse runs to
   * completion at progress=0 and immediately transitions to forward
   * play. No held dwell. Net: a quick click feels reverse → forward
   * with no pause, vs. a deliberate hold gives the user full control
   * over the dwell at the pre-state.
   *
   * Statuses visited (held variant): reversing → pausing (held) →
   * playing → ended.
   */
  replayWithReverseHold(): void;
  /**
   * Pair with `replayWithReverseHold()` — ends the hold and resumes
   * forward play.
   *
   * Behaviour by current status:
   *   - pausing (held at timestamp=0): immediately start forward leg
   *     (currentMoveIndex=0 from progress=0).
   *   - reversing (reverse leg still in flight, user released early):
   *     set an internal `releaseQueued` flag. The reverse leg runs to
   *     completion (lands at timestamp=0); instead of entering the
   *     held-pause it transitions directly to forward play. This is
   *     the "quick-click collapse".
   *   - any other status: no-op (gracefully ignore; called outside an
   *     active hold).
   */
  releaseHold(): void;

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
  // Reverse-leg bookkeeping for `replayWithReverse()` and
  // `replayWithReverseHold()`. Captured once at the start of the leg
  // so the per-tick interpolation is stable.
  // `reverseFromTimestamp` is the timestamp the reverse leg started
  // FROM (e.g. totalDurationMs when called at end-of-sequence). The
  // leg always lands at timestamp=0.
  let reverseFromTimestamp = 0;
  // Duration of the current reverse leg in ms. `replayWithReverse()`
  // uses DUR_REVERSE_MS verbatim; `replayWithReverseHold()` scales it
  // by the normalized starting timestamp so partial-reverses retain
  // the same per-unit reverse speed.
  let reverseDurationMs = DUR_REVERSE_MS;
  // Pause-leg deadline (wall time at which the pause expires and the
  // forward leg begins). Held during the `pausing` status; rAF ticks
  // poll this and transition to forward when wall-time crosses it.
  // For HELD pauses (`replayWithReverseHold()`), this is set to
  // Number.POSITIVE_INFINITY so the pause never auto-expires.
  let pauseUntilWall = 0;
  // True when a held pause is active (entered via
  // `replayWithReverseHold()` and not yet released). Distinguishes the
  // held variant from `replayWithReverse()`'s fixed-duration pause so
  // `releaseHold()` knows whether it's a no-op or an exit-now command.
  let inHold = false;
  // True when `releaseHold()` was called during the reverse leg of a
  // held replay. The reverse leg runs to completion; on hitting
  // timestamp=0 it skips the held-pause and transitions directly to
  // forward play. This produces the "quick-click collapse" — clicks
  // that release before the reverse leg completes get reverse →
  // forward with no pause in between.
  let releaseQueued = false;
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
    if (status === "playing") {
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
      return;
    }
    if (status === "reversing") {
      // Reverse leg: timestamp interpolates linearly from
      // reverseFromTimestamp → 0 over reverseDurationMs.
      const elapsed = now - rafStartWall;
      const t = reverseDurationMs > 0
        ? Math.min(1, elapsed / reverseDurationMs)
        : 1;
      timestamp = reverseFromTimestamp * (1 - t);
      if (t >= 1) {
        timestamp = 0;
        if (inHold && releaseQueued) {
          // Quick-click collapse: user released during the reverse
          // leg. Skip the held pause and transition straight to
          // forward play.
          inHold = false;
          releaseQueued = false;
          status = "playing";
          rafStartWall = now;
          rafStartTimestamp = 0;
          notify();
          rafId = requestAnimationFrame(tick);
          return;
        }
        status = "pausing";
        // Held variant dwells indefinitely; fixed variant expires
        // after DUR_PAUSE_MS.
        pauseUntilWall = inHold
          ? Number.POSITIVE_INFINITY
          : now + DUR_PAUSE_MS;
        notify();
        rafId = requestAnimationFrame(tick);
        return;
      }
      notify();
      rafId = requestAnimationFrame(tick);
      return;
    }
    if (status === "pausing") {
      // Pause leg: hold timestamp at 0 until pauseUntilWall, then
      // enter forward play. For held pauses (`replayWithReverseHold`)
      // pauseUntilWall is +Infinity so the wall-time check never
      // fires — `releaseHold()` is the only exit. Each rAF tick we
      // still notify subscribers so any UI tied to status (e.g. a
      // "rewinding…" cue) updates; the rendered cube is unchanged
      // (timestamp=0 throughout).
      if (now >= pauseUntilWall) {
        inHold = false;
        status = "playing";
        rafStartWall = now;
        rafStartTimestamp = 0;
        timestamp = 0;
        notify();
        rafId = requestAnimationFrame(tick);
        return;
      }
      notify();
      rafId = requestAnimationFrame(tick);
      return;
    }
    // Status was changed externally (paused/seek/etc) — stop the loop.
    rafId = 0;
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
    // If called during a reverse-or-pause leg, abandon the leg and
    // start a fresh forward play from whatever timestamp we're at.
    // The leg's rAF will be cancelled by startRaf(). Any held-replay
    // hold-state is also cleared — play() is an explicit override.
    inHold = false;
    releaseQueued = false;
    status = "playing";
    startRaf();
    notify();
  }

  function pause(): void {
    // Pause from any running leg (forward / reverse / pause) snapshots
    // the current timestamp and stops further automatic progression.
    // From other statuses (idle / paused / ended) pause is a no-op.
    if (
      status !== "playing" &&
      status !== "reversing" &&
      status !== "pausing"
    ) {
      return;
    }
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
    // Hold-state ends — pause is an explicit override.
    inHold = false;
    releaseQueued = false;
    status = "paused";
    notify();
  }

  function seek(ms: number): void {
    const clamped = Math.max(0, Math.min(totalDurationMs, ms));
    // A seek during a reverse-or-pause leg abandons the choreography
    // — treated as "not playing" for the purposes of resume-after-seek.
    const wasPlaying = status === "playing";
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
    // Seek is an explicit jump; any active hold ends.
    inHold = false;
    releaseQueued = false;
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
    // replay() = explicit jump to start + forward play; any active
    // hold ends.
    inHold = false;
    releaseQueued = false;
    timestamp = 0;
    status = "playing";
    startRaf();
    notify();
  }

  function replayWithReverse(): void {
    // Mirrors the rev5 preview's per-panel state machine. The choreography
    // engages only when we'd otherwise be "rewinding from end" — i.e.
    // when the current sequence is essentially fully advanced. In every
    // other case (mid-flight, paused, just-started) we behave like
    // `replay()` so the click feels snappy rather than perfunctory.
    //
    // Threshold: progress > 0.99 against the current move OR status is
    // "ended". The preview uses `state.progress > 0.99` on a single
    // panel; for multi-move sequences we generalise to "near the end of
    // the sequence" by checking ended OR currentMoveProgress > 0.99 on
    // the last move. For one-move card sequences this collapses to the
    // exact preview behaviour.
    if (moves.length === 0) return;
    const { currentMoveIndex: idx, currentMoveProgress: prog } =
      indexAndProgress();
    const atEnd =
      status === "ended" || (idx === moves.length - 1 && prog > 0.99);
    if (!atEnd) {
      replay();
      return;
    }
    // Engage the choreographed path: reverse → pause → forward.
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
    // Non-held variant: clear any prior hold-state.
    inHold = false;
    releaseQueued = false;
    reverseFromTimestamp = totalDurationMs;
    reverseDurationMs = DUR_REVERSE_MS;
    timestamp = totalDurationMs;
    status = "reversing";
    rafStartWall = performance.now();
    rafStartTimestamp = totalDurationMs;
    rafId = requestAnimationFrame(tick);
    notify();
  }

  function replayWithReverseHold(): void {
    // Press-and-hold variant. Engages a reverse leg (or skips it for
    // already-at-start states) into an indefinitely-held pause at
    // timestamp=0. See the CubeSequence interface doc for the full
    // interruption matrix.
    if (moves.length === 0) return;
    // Already in a hold? No-op (an in-flight reverse-into-hold should
    // be left alone; a held pause is exactly where we want to be).
    if (inHold) return;

    const startFromTs = timestamp;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
    inHold = true;
    releaseQueued = false;

    // From idle / fresh-start (timestamp=0): skip the reverse leg —
    // nothing to rewind from — and enter the held pause immediately.
    // Visually identical to staying at the pre-state, but state is
    // consistent so `releaseHold()` will play forward.
    if (startFromTs <= 0) {
      timestamp = 0;
      status = "pausing";
      pauseUntilWall = Number.POSITIVE_INFINITY;
      rafStartWall = performance.now();
      rafStartTimestamp = 0;
      rafId = requestAnimationFrame(tick);
      notify();
      return;
    }

    // Otherwise (ended / playing / paused / mid-flight): reverse leg
    // from startFromTs → 0. Duration scaled by the normalized fraction
    // so partial reverses keep the same per-unit speed.
    //   ended:    fraction = 1   → full DUR_REVERSE_MS.
    //   mid-way:  fraction = p/T → scaled.
    reverseFromTimestamp = startFromTs;
    const fraction = totalDurationMs > 0 ? startFromTs / totalDurationMs : 1;
    reverseDurationMs = DUR_REVERSE_MS * Math.max(0, Math.min(1, fraction));
    timestamp = startFromTs;
    status = "reversing";
    rafStartWall = performance.now();
    rafStartTimestamp = startFromTs;
    rafId = requestAnimationFrame(tick);
    notify();
  }

  function releaseHold(): void {
    // Only meaningful if a hold is active. Outside a hold this is a
    // graceful no-op (callers can fire-and-forget on pointerup).
    if (!inHold) return;

    if (status === "pausing") {
      // Held at progress=0 — exit the hold immediately and start
      // forward play.
      if (rafId) cancelAnimationFrame(rafId);
      rafId = 0;
      inHold = false;
      releaseQueued = false;
      timestamp = 0;
      status = "playing";
      startRaf();
      notify();
      return;
    }

    if (status === "reversing") {
      // Reverse leg still in flight — queue a release so the tick
      // handler skips the held-pause when the leg completes. This is
      // the "quick-click collapse".
      releaseQueued = true;
      return;
    }

    // Any other status: shouldn't reach here (inHold is only set
    // alongside reversing/pausing), but clear flags defensively.
    inHold = false;
    releaseQueued = false;
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
    replayWithReverse,
    replayWithReverseHold,
    releaseHold,
    subscribe,
  };

  if (autoplay) {
    play();
  }
  return handle;
}
