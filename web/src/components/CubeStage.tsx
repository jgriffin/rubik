// CubeStage — single-cube view for `cube` mode (cols=1).
//
// Two playback modes, gated on `isPlaying`:
//
//   Manual mode (isPlaying=false). The cube reacts to one-off
//   `activeIdx` changes. A single-move forward-form spec drives the
//   2D kinematics primitive:
//
//     playForward: cube animates state[landIdx-1] → state[landIdx].
//                  Used on activeIdx increases (single-step or jump).
//     primed:      cube sits at state[landIdx] (= end of the same
//                  forward-form spec). Used on backward steps (snap to
//                  landing state without animation) AND as the resting
//                  state after a playForward animation completes. The
//                  seq is at timestamp=totalDurationMs and is therefore
//                  ready for the press-and-hold gesture:
//                    pointerdown → seq.replayWithReverseHold() →
//                                  reverses to state[landIdx-1] and
//                                  holds at the pre-state.
//                    pointerup   → seq.releaseHold() → forward to
//                                  state[landIdx].
//     snap:        activeIdx === 0 (no previous move). Cube renders
//                  state[0] from an empty-moves spec; pointer handlers
//                  are no-ops.
//
//   Play mode (isPlaying=true). One sequence covering all moves from
//   the play-start position through the end of the trajectory, with
//   `gapMs` providing the inter-move settle pause. The sequence is
//   created once when play starts and ticks through naturally; its
//   `currentMoveIndex` is subscribed to here, and each move-boundary
//   crossing fires `onAutoAdvance(playStartIdx + curIdx + 1)` to bump
//   App's `activeIdx`. On sequence end, `onPlayEnd()` flips App's
//   `isPlaying` back to false. Section ii's cell highlights step in
//   lockstep with the animation because activeIdx is the canonical
//   source for cell highlighting AND the sequence's bookmark.
//
// Why one sequence per play run: the old per-step rebuild caused a
// visible flash between each move's animation. Each rebuild produced
// a fresh `CubeSequence` whose idle initial frame rendered
// `<StaticInner>` (a different SVG structure from `<SvgFromRenderPlan>`)
// for ~1 reconciliation tick before `seq.play()` advanced state. The
// single-sequence design keeps the same SVG tree mounted through the
// entire play run, eliminating the inter-step DOM swap.
//
// Why useLayoutEffect for seek-to-end on primed: when the spec rebuilds
// (e.g. backward click changes landIdx), a fresh CubeSequence is born
// at timestamp=0, which would paint state[landIdx-1] for one frame
// before the effect seeks to end. useLayoutEffect runs the seek
// synchronously before the browser paints, so the first paint shows
// state[landIdx]. Same flash-suppression strategy as Block D iter-2's
// single-sequence playback, applied to the primed-state transition.
//
// Animation lives on the 2D path only. The 3D path stays static this
// phase; per-step 3D animation via `<twisty-player>`'s native playback
// is deferred. Press-and-hold is correspondingly 2D-only.
//
// Play start: handleTogglePlay in App rewinds activeIdx to 0 first if
// at end-of-moves, so playStartIdx is always < moves.length when a
// play spec is built.

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import Cube2D from "./Cube2D";
import TwistyPlayerWrapper from "./TwistyPlayerWrapper";
import { useCubeSequence } from "../hooks/useCubeSequence";
import { ANIM_MS_PER_MOVE, PLAY_STEP_DWELL_MS } from "./cubeStageConstants";
import type { MoveStr } from "../state/faceletMoves";
import type { RenderMode } from "./SolutionGrid";

type Props = {
  states: string[];
  moves: MoveStr[];
  activeIdx: number;
  isPlaying: boolean;
  onAutoAdvance: (idx: number) => void;
  onPlayEnd: () => void;
  scrambleAlg: string;
  renderMode: RenderMode;
  sizePx: number;
};

// Manual-mode animation kind. Forward-form spec is shared between
// playForward and primed (same startFacelet + move) — they differ only
// in initial seq position (idle vs ended), driven by useLayoutEffect.
type ManualAnim =
  | { kind: "snap" }
  | { kind: "playForward"; landIdx: number }
  | { kind: "primed"; landIdx: number };

export default function CubeStage({
  states,
  moves,
  activeIdx,
  isPlaying,
  onAutoAdvance,
  onPlayEnd,
  scrambleAlg,
  renderMode,
  sizePx,
}: Props) {
  // -------- Manual + play bookkeeping (store-info-from-prior-renders) --------
  // React's `react-hooks/set-state-in-effect` rule (and the underlying
  // guidance) prohibits state updates inside effect bodies for state
  // transitions driven by prop changes. The pattern here mirrors
  // `prevIsPlaying`/`playStartIdx` from Block D iter-2: track previous
  // values in state, compare-and-set during render. setState during
  // render is supported by React (it's the documented "store info from
  // prior renders" pattern) and produces the same render-phase update
  // ordering as a synchronous effect, without the cascading-render
  // warning.
  const [manualAnim, setManualAnim] = useState<ManualAnim>(
    activeIdx > 0
      ? { kind: "primed", landIdx: activeIdx }
      : { kind: "snap" },
  );
  const [prevActiveIdx, setPrevActiveIdx] = useState(activeIdx);
  const [prevIsPlaying, setPrevIsPlaying] = useState(isPlaying);
  const [playStartIdx, setPlayStartIdx] = useState<number | null>(
    isPlaying ? activeIdx : null,
  );

  // Play-mode transitions take precedence: when isPlaying flips, we
  // explicitly set manualAnim + playStartIdx + sync the prevActiveIdx
  // mirror so the activeIdx-change branch below doesn't re-fire on the
  // next render with a stale prev value.
  if (prevIsPlaying !== isPlaying) {
    setPrevIsPlaying(isPlaying);
    setPrevActiveIdx(activeIdx);
    if (isPlaying) {
      setPlayStartIdx(activeIdx);
      // Park manualAnim in snap so the spec selector below takes the
      // play branch (and the manual seq doesn't simultaneously try to
      // play a single-move spec underneath the play sequence).
      setManualAnim({ kind: "snap" });
    } else {
      setPlayStartIdx(null);
      // Exit play: settle to primed at current activeIdx so
      // press-and-hold is immediately usable on the last move.
      setManualAnim(
        activeIdx > 0
          ? { kind: "primed", landIdx: Math.min(activeIdx, moves.length) }
          : { kind: "snap" },
      );
    }
  } else if (prevActiveIdx !== activeIdx) {
    // activeIdx changed outside of a play transition. Skip if currently
    // playing — the play sequence is the authoritative driver during
    // playback and updates activeIdx via onAutoAdvance.
    setPrevActiveIdx(activeIdx);
    if (!isPlaying) {
      if (activeIdx === 0) {
        setManualAnim({ kind: "snap" });
      } else if (
        activeIdx > prevActiveIdx &&
        activeIdx <= moves.length
      ) {
        // Forward step (single or jump): animate state[landIdx-1] →
        // state[landIdx].
        setManualAnim({ kind: "playForward", landIdx: activeIdx });
      } else {
        // Backward step or any other activeIdx > 0 transition: snap to
        // primed at landing state. No standalone backward animation —
        // the press-and-hold gesture is the reverse-animation primitive.
        const landIdx = Math.min(activeIdx, moves.length);
        setManualAnim({ kind: "primed", landIdx });
      }
    }
  }

  // -------- Spec selection --------
  // Two useMemos with disjoint deps so the play spec stays referentially
  // stable as activeIdx ticks during play (no rebuild of the sequence
  // between moves — that was the source of the inter-step flash).
  const playSpec = useMemo(() => {
    if (
      !isPlaying ||
      playStartIdx === null ||
      playStartIdx >= moves.length ||
      states.length < playStartIdx + 1
    ) {
      return null;
    }
    return {
      startFacelet: states[playStartIdx],
      moves: moves.slice(playStartIdx),
      msPerMove: ANIM_MS_PER_MOVE,
      gapMs: PLAY_STEP_DWELL_MS,
    };
  }, [isPlaying, playStartIdx, moves, states]);

  // Manual-spec rebuild key. playForward and primed at the same landIdx
  // share the same spec content (forward-form { state[landIdx-1] →
  // moves[landIdx-1] }), so they collapse to the same key — the seq
  // doesn't rebuild on a playForward → primed transition. Only landIdx
  // changes (or snap ↔ non-snap) trigger a rebuild.
  const manualSpecKey =
    manualAnim.kind === "snap" ? "snap" : `fwd:${manualAnim.landIdx}`;
  const manualSpec = useMemo(() => {
    if (isPlaying) return null;
    if (manualAnim.kind === "snap") {
      const idx = Math.max(0, Math.min(activeIdx, states.length - 1));
      return {
        startFacelet: states[idx],
        moves: [] as MoveStr[],
        msPerMove: ANIM_MS_PER_MOVE,
      };
    }
    const i = manualAnim.landIdx;
    return {
      startFacelet: states[i - 1],
      moves: [moves[i - 1]],
      msPerMove: ANIM_MS_PER_MOVE,
    };
    // Intentionally omit manualAnim.kind from deps via manualSpecKey:
    // playForward and primed with the same landIdx produce the same
    // spec content, so the seq stays referentially stable across that
    // transition. The play-vs-seek branch is decided by the
    // useLayoutEffect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, manualSpecKey, activeIdx, states, moves]);

  const spec = playSpec ?? manualSpec ?? {
    startFacelet: states[0] ?? "",
    moves: [] as MoveStr[],
    msPerMove: ANIM_MS_PER_MOVE,
  };

  const seq = useCubeSequence(spec);

  // -------- Initial seq state: play forward, or seek-to-end for primed --------
  // For playForward: call seq.play() — the rAF loop kicks off forward
  // playback.
  // For primed: call seq.seek(seq.totalDurationMs) — the seq lands at
  // end immediately, displaying state[landIdx]. useLayoutEffect (not
  // useEffect) runs synchronously before the browser paints, so a
  // freshly-rebuilt seq doesn't flash its timestamp=0 startFacelet
  // (state[landIdx-1]) before the seek.
  // For play mode (multi-move): the same auto-play branch.
  useLayoutEffect(() => {
    if (isPlaying) {
      if (spec.moves.length > 0) seq.play();
      return;
    }
    if (manualAnim.kind === "playForward" && spec.moves.length > 0) {
      seq.play();
    } else if (manualAnim.kind === "primed" && spec.moves.length > 0) {
      seq.seek(seq.totalDurationMs);
    }
    // snap: nothing — seq renders startFacelet statically from
    // timestamp=0 (Cube2D's empty-moves path).
  }, [seq, spec.moves.length, manualAnim.kind, isPlaying]);

  // -------- Play-mode → activeIdx sync --------
  // While playing, each move-boundary crossing in the sequence ticks
  // activeIdx forward via onAutoAdvance.
  const lastReportedMoveIdxRef = useRef(-1);
  useEffect(() => {
    if (!isPlaying || playStartIdx === null) {
      lastReportedMoveIdxRef.current = -1;
      return;
    }
    const cur = seq.currentMoveIndex;
    if (cur !== lastReportedMoveIdxRef.current && cur >= 0) {
      lastReportedMoveIdxRef.current = cur;
      onAutoAdvance(playStartIdx + cur + 1);
    }
    if (seq.status === "ended") {
      onPlayEnd();
    }
  }, [
    seq.currentMoveIndex,
    seq.status,
    isPlaying,
    playStartIdx,
    onAutoAdvance,
    onPlayEnd,
  ]);

  // -------- Press-and-hold gesture (2D only) --------
  // Mirrors the M9.2 SolutionCard NonStartCard pointer-handler pattern.
  // Active whenever we're in manual mode AND there's a previous move
  // to rewind to (kind !== "snap") AND we have a usable spec. Works in
  // both playForward (animation in flight or just ended) and primed
  // (snapped to landing or post-play settled) — the seq's
  // replayWithReverseHold handles the interruption matrix uniformly:
  //   playing mid-flight  → reverse from current ts to 0 (scaled).
  //   ended               → full reverse leg → held pause at 0.
  // releaseHold resumes forward play, landing back at end (state[landIdx]).
  //
  // pointerHandledRef dedupes the synthetic click event that some
  // browsers fire after pointerup, mirroring SolutionCard. Keyboard
  // activation (Space/Enter on focused button) fires click WITHOUT
  // any pointerdown/up, so the ref stays false and the click branch
  // runs replayWithReverse() as the keyboard-friendly cue.
  const canPressAndHold =
    !isPlaying && manualAnim.kind !== "snap" && spec.moves.length > 0;
  const pointerHandledRef = useRef(false);

  const handlePointerDown = (e: React.PointerEvent<HTMLButtonElement>) => {
    if (!canPressAndHold) return;
    pointerHandledRef.current = true;
    const target = e.currentTarget;
    if (typeof target.setPointerCapture === "function") {
      try {
        target.setPointerCapture(e.pointerId);
      } catch {
        // ignore: pointerId may have been auto-released.
      }
    }
    seq.replayWithReverseHold();
  };

  const handlePointerUp = () => {
    if (!pointerHandledRef.current) return;
    seq.releaseHold();
  };

  const handlePointerCancel = () => {
    if (!pointerHandledRef.current) return;
    seq.releaseHold();
  };

  const handlePointerLeave = () => {
    if (!pointerHandledRef.current) return;
    seq.releaseHold();
  };

  const handleClick = () => {
    // Pointer path already drove the choreography — consume the
    // synthetic click and bail.
    if (pointerHandledRef.current) {
      pointerHandledRef.current = false;
      return;
    }
    // Keyboard activation (Space/Enter): finite-pause replay-with-reverse
    // as the snappy keyboard cue. Same primitive the column-mode cards
    // use for their keyboard fallback.
    if (canPressAndHold) {
      seq.replayWithReverse();
    }
  };

  // -------- Rendering --------
  // 2D — sequence-mode. Wrap in a <button> so pointer + keyboard
  // gestures have an accessibility-correct host. Disabled state when
  // press-and-hold isn't available (activeIdx === 0 or mid-animation):
  // pointer/click handlers are gated by `canPressAndHold` internally,
  // and `aria-disabled` reflects intent to assistive tech without
  // suppressing focusability (so keyboard users can still tab through
  // and read "currently no rewind available").
  const cube2DInner = (
    <Cube2D
      sequence={seq}
      sizePx={sizePx}
      testId={renderMode === "dual" ? "flat-cube-pair" : null}
    />
  );

  const cube2D = (
    <button
      type="button"
      className="cube-stage-press"
      data-testid="cube-stage-press"
      aria-label={
        canPressAndHold
          ? "Press and hold to rewind the current move"
          : "Cube"
      }
      aria-disabled={canPressAndHold ? undefined : true}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerCancel}
      onPointerLeave={handlePointerLeave}
      onClick={handleClick}
    >
      {cube2DInner}
    </button>
  );

  // 3D — static (per-step animation deferred).
  const solutionAlg = moves.slice(0, activeIdx).join(" ");
  const cube3D = (
    <TwistyPlayerWrapper
      mode="static"
      scrambleAlg={scrambleAlg}
      solutionAlg={solutionAlg}
      sizePx={sizePx}
      testId={renderMode === "dual" ? "twisty-cube-pair" : "twisty-cube"}
    />
  );

  if (renderMode === "iso") return cube3D;
  if (renderMode === "dual") {
    return (
      <div className="render-pair">
        {cube2D}
        {cube3D}
      </div>
    );
  }
  return cube2D;
}
