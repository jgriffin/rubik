// CubeStage — single-cube view for `cube` mode (cols=1).
//
// Renders one big cube (2D, 3D, or split) whose displayed state is
// driven by `activeIdx` from the editor. On an `activeIdx` change of
// ±1 (user click, arrow-key nav, or App's auto-play timer), the 2D
// path animates the single move that bridges the two states:
//   - Forward jump (idx grows): snap to state[newIdx-1], animate
//     move[newIdx-1] forward, land at state[newIdx].
//   - Backward jump (idx shrinks): snap to state[newIdx+1], animate
//     inverse(move[newIdx]) forward, land at state[newIdx].
// Multi-step jumps of either direction follow the same shape — snap
// to the adjacent state, animate the single inverse-or-forward move.
//
// Animation lives on the 2D path only — `Cube2D` already owns the
// per-move SVG kinematics via `CubeSequence` + `useCubeSequence`. The
// 3D path stays static; per-step 3D animation via `<twisty-player>`'s
// native playback is deferred.
//
// Play/pause: App owns `isPlaying` and the auto-advance timer. The
// button itself lives in section iii's header next to the render-mode
// and view-layout switches. CubeStage just reads activeIdx and animates
// — playback ownership stays decoupled from animation primitives.

import { useEffect, useMemo, useRef, useState } from "react";
import Cube2D from "./Cube2D";
import TwistyPlayerWrapper from "./TwistyPlayerWrapper";
import { useCubeSequence } from "../hooks/useCubeSequence";
import { ANIM_MS_PER_MOVE } from "./cubeStageConstants";
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

// Inverse of a QTM move (R ↔ R', U ↔ U', …). Used when activeIdx steps
// backward — we feed the kinematics engine the inverse and play it
// forward, so state[N] → inverse(move[N-1]) → state[N-1] runs through
// the same animation primitive as the forward path. Keeps the inverse
// logic local to where it's consumed; M9.3 is QTM-only so a one-liner
// suffices.
function inverseMove(m: MoveStr): MoveStr {
  return (m.endsWith("'") ? (m.slice(0, -1) as MoveStr) : (`${m}'` as MoveStr));
}

// AnimSpec captures everything needed to animate a single step in
// either direction. `null` means snap (no animation needed). Tracked
// in component state so the spec survives renders triggered by
// unrelated props until the animation completes.
type AnimSpec =
  | { dir: "forward"; landIdx: number }
  | { dir: "backward"; landIdx: number };

export default function CubeStage({
  states,
  moves,
  activeIdx,
  scrambleAlg,
  renderMode,
  sizePx,
}: Props) {
  // The active animation, or null when idle. Decoupled from `activeIdx`
  // so the animation isn't torn down by unrelated state updates.
  const [animSpec, setAnimSpec] = useState<AnimSpec | null>(null);
  const prevActiveIdxRef = useRef(activeIdx);

  useEffect(() => {
    const prev = prevActiveIdxRef.current;
    prevActiveIdxRef.current = activeIdx;
    if (activeIdx === prev) return;

    if (activeIdx > prev && activeIdx > 0 && activeIdx <= moves.length) {
      // Forward (any step size). Animate the move landing at activeIdx.
      setAnimSpec({ dir: "forward", landIdx: activeIdx });
    } else if (activeIdx < prev && activeIdx >= 0 && activeIdx < moves.length) {
      // Backward (any step size). Animate inverse(move[activeIdx])
      // from state[activeIdx+1] → state[activeIdx]. The inverse runs
      // forward through the same animation primitive.
      setAnimSpec({ dir: "backward", landIdx: activeIdx });
    } else {
      setAnimSpec(null);
    }
  }, [activeIdx, moves.length]);

  // Spec the sequence consumes. Animating branch: single-move spec
  // (forward = move[landIdx-1] from state[landIdx-1]; backward =
  // inverse(move[landIdx]) from state[landIdx+1]). Snap branch:
  // empty-moves spec — Cube2D's animated path silently renders
  // startFacelet statically, so we don't flip modes at the call site.
  const spec = useMemo(() => {
    if (animSpec == null) {
      const idx = Math.max(0, Math.min(activeIdx, states.length - 1));
      return {
        startFacelet: states[idx],
        moves: [] as MoveStr[],
        msPerMove: ANIM_MS_PER_MOVE,
      };
    }
    if (animSpec.dir === "forward") {
      const i = animSpec.landIdx;
      return {
        startFacelet: states[i - 1],
        moves: [moves[i - 1]],
        msPerMove: ANIM_MS_PER_MOVE,
      };
    }
    // backward
    const i = animSpec.landIdx;
    return {
      startFacelet: states[i + 1],
      moves: [inverseMove(moves[i])],
      msPerMove: ANIM_MS_PER_MOVE,
    };
  }, [animSpec, activeIdx, states, moves]);

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
