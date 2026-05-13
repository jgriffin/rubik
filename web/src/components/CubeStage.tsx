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
  scrambleAlg: string;
  renderMode: RenderMode;
  sizePx: number;
};

// Animation duration per move. SolutionCard uses 600ms; cube mode runs
// a touch quicker because the playback feel needs to keep up with
// rapid cell clicks. Re-tunable at C·P5 if the play cadence wants it
// slower / faster.
const ANIM_MS_PER_MOVE = 400;

export default function CubeStage({
  states,
  moves,
  activeIdx,
  scrambleAlg,
  renderMode,
  sizePx,
}: Props) {
  // `animIdx` = the activeIdx of the currently-animating move (i.e. the
  // step the animation lands on), or null when not animating. Decoupled
  // from `activeIdx` so animations survive subsequent ref updates.
  const [animIdx, setAnimIdx] = useState<number | null>(null);
  const prevActiveIdxRef = useRef(activeIdx);

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
