// CubeStage — single-cube view for `cube` mode (cols=1).
//
// Renders one big cube (2D, 3D, or split) whose displayed state is
// driven by `activeIdx` from the editor. On a forward `activeIdx`
// jump (user clicks a later cell, or the moves array grows), the 2D
// path animates the move that lands on the new state: snaps the cube
// to state[activeIdx-1], plays move[activeIdx-1] forward, lands at
// state[activeIdx]. Backward / same jumps snap with no animation.
//
// Animation lives on the 2D path only — `Cube2D` already owns the
// per-move SVG kinematics via `CubeSequence` + `useCubeSequence`. The
// 3D path stays static this phase; per-step 3D animation via
// `<twisty-player>`'s native playback lands in C·P4.
//
// Forward-jump-of-any-size policy: clicking cell 5 from cell 2 still
// animates move 5 (skips through state[3], state[4] visually — snaps
// to state[4] and plays move 5). Multi-move chained playback for
// large jumps is a Block C C·P5 (play/pause) concern, not this phase.

import { useEffect, useMemo, useRef, useState } from "react";
import Cube2D from "./Cube2D";
import TwistyPlayerWrapper from "./TwistyPlayerWrapper";
import { useCubeSequence } from "../hooks/useCubeSequence";
import type { MoveStr } from "../state/faceletMoves";
import type { RenderMode } from "./SolutionGrid";

type Props = {
  states: string[];
  moves: MoveStr[];
  activeIdx: number;
  onActiveChange: (idx: number) => void;
  scrambleAlg: string;
  renderMode: RenderMode;
  sizePx: number;
};

// Animation duration per move (ms). SolutionCard uses 600ms; cube mode
// runs a touch quicker so the playback keeps up with rapid cell clicks
// and the auto-play cadence doesn't feel sluggish.
const ANIM_MS_PER_MOVE = 400;

// Dwell between the end of one auto-play step's animation and the start
// of the next. 100ms reads as a brief "settle" pause without dragging
// the playback. Total per-step duration = ANIM_MS_PER_MOVE + this.
const PLAY_STEP_DWELL_MS = 100;

export default function CubeStage({
  states,
  moves,
  activeIdx,
  onActiveChange,
  scrambleAlg,
  renderMode,
  sizePx,
}: Props) {
  // `animIdx` = the activeIdx of the currently-animating move (i.e. the
  // step the animation lands on), or null when not animating. Decoupled
  // from `activeIdx` so animations survive subsequent ref updates.
  const [animIdx, setAnimIdx] = useState<number | null>(null);
  const prevActiveIdxRef = useRef(activeIdx);

  // Auto-play: advance `activeIdx` forward at a steady cadence riding
  // the per-step animation. Local-only state — switching to columns
  // mode unmounts CubeStage and resets play.
  const [isPlaying, setIsPlaying] = useState(false);

  // Pause on any moves-array change. The user typed / pasted / deleted
  // a move — they've taken manual control of the editor, so auto-play
  // should yield. The "store information from previous renders" pattern
  // (storing the prior moves identity via useState rather than useRef)
  // lets the pause land the SAME render the moves changed in (no effect-
  // tick delay), and setState-in-render is the React-blessed idiom for
  // resetting state on a prop change. `moves` is a fresh array reference
  // on each user edit, stable across unrelated re-renders.
  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  const [prevMoves, setPrevMoves] = useState(moves);
  if (prevMoves !== moves) {
    setPrevMoves(moves);
    if (isPlaying) setIsPlaying(false);
  }

  // Auto-advance timer. When `isPlaying`, schedule the next step
  // (activeIdx + 1) after one full step duration (anim + dwell). Each
  // activeIdx change re-fires this effect and schedules the next step.
  // The end-of-moves stop is folded into the timer callback so its
  // setState lands inside an event, not directly in the effect body.
  // Manual cell clicks that land forward continue the playback chain
  // naturally; backward clicks also continue forward from the new
  // position (since the chain's only invariant is "advance by +1 each
  // tick"). Typing pauses via the moves-change branch above.
  useEffect(() => {
    if (!isPlaying || activeIdx >= moves.length) return;
    const timer = setTimeout(() => {
      const nextIdx = activeIdx + 1;
      onActiveChange(nextIdx);
      if (nextIdx >= moves.length) setIsPlaying(false);
    }, ANIM_MS_PER_MOVE + PLAY_STEP_DWELL_MS);
    return () => clearTimeout(timer);
  }, [isPlaying, activeIdx, moves.length, onActiveChange]);

  useEffect(() => {
    const prev = prevActiveIdxRef.current;
    // Forward jump → animate the move that lands at the new activeIdx.
    // Backward / no-change → snap (clear any in-flight animation).
    if (activeIdx > prev && activeIdx > 0 && activeIdx <= moves.length) {
      setAnimIdx(activeIdx);
    } else if (activeIdx !== prev) {
      setAnimIdx(null);
    }
    prevActiveIdxRef.current = activeIdx;
  }, [activeIdx, moves.length]);

  // Spec the sequence consumes. Animating branch: single-move spec
  // (state[idx-1] → state[idx]). Snap branch: empty-moves spec which
  // Cube2D's animated path silently falls back to static rendering of
  // startFacelet for, so we don't have to flip modes at the call site.
  const spec = useMemo(() => {
    if (animIdx == null || animIdx <= 0 || animIdx > moves.length) {
      const idx = Math.max(0, Math.min(activeIdx, states.length - 1));
      return {
        startFacelet: states[idx],
        moves: [] as MoveStr[],
        msPerMove: ANIM_MS_PER_MOVE,
      };
    }
    return {
      startFacelet: states[animIdx - 1],
      moves: [moves[animIdx - 1]],
      msPerMove: ANIM_MS_PER_MOVE,
    };
  }, [animIdx, activeIdx, states, moves]);

  const seq = useCubeSequence(spec);

  // Kick off play whenever a fresh animation spec arrives. `seq`
  // identity changes 1:1 with `spec` identity (`useCubeSequence`
  // memoizes on the spec object), so depending on `seq` here means
  // exactly one fire per spec change. Snap-to-empty transitions also
  // fire here but the moves.length guard skips play() — Cube2D
  // renders startFacelet statically and we're done.
  useEffect(() => {
    if (spec.moves.length > 0) {
      seq.play();
    }
  }, [seq, spec.moves.length]);

  // 2D — always sequence-mode; Cube2D's empty-moves path handles snap.
  const cube2D = (
    <Cube2D
      sequence={seq}
      sizePx={sizePx}
      testId={renderMode === "dual" ? "flat-cube-pair" : null}
    />
  );

  // 3D — static (per-step animation lands in C·P4).
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

  const canPlay = moves.length > 0;

  function handleTogglePlay() {
    if (isPlaying) {
      setIsPlaying(false);
      return;
    }
    // If at end-of-moves, rewind to start before kicking off — playing
    // from the end position would immediately stop. Letting play
    // re-rewatch the whole sequence is the natural interaction.
    if (activeIdx >= moves.length) {
      onActiveChange(0);
    }
    setIsPlaying(true);
  }

  const cubeRender =
    renderMode === "iso" ? (
      cube3D
    ) : renderMode === "dual" ? (
      <div className="render-pair">
        {cube2D}
        {cube3D}
      </div>
    ) : (
      cube2D
    );

  return (
    <>
      <button
        type="button"
        className="text-action cube-play-btn"
        data-testid="cube-play-button"
        onClick={handleTogglePlay}
        disabled={!canPlay}
      >
        {isPlaying ? "pause" : "play"}
      </button>
      {cubeRender}
    </>
  );
}
