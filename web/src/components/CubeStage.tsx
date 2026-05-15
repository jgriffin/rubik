// CubeStage — single-cube view for `cube` mode (cols=1).
//
// Two playback modes, gated on `isPlaying`:
//
//   Manual mode (isPlaying=false). The cube reacts to one-off
//   `activeIdx` changes (user clicks a cell, arrow-key nav, sequence-
//   just-ended landing position). A single-move spec drives the 2D
//   kinematics primitive: forward step → animate move[landIdx-1] from
//   state[landIdx-1]; backward step → animate inverse(move[landIdx])
//   from state[landIdx+1]; multi-step jumps snap to the adjacent
//   state then animate the single bridging move (forward or inverse).
//   This is the original block-D animation behavior.
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
// Animation lives on the 2D path only. The 3D path stays static this
// phase; per-step 3D animation via `<twisty-player>`'s native playback
// is deferred.
//
// Play start: handleTogglePlay in App rewinds activeIdx to 0 first if
// at end-of-moves, so playStartIdx is always < moves.length when a
// play spec is built.

import { useEffect, useMemo, useRef, useState } from "react";
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

// Inverse of a QTM move (R ↔ R', U ↔ U', …). Used when activeIdx steps
// backward in manual mode — feed the kinematics engine the inverse
// played forward, so state[N] → inverse(move[N-1]) → state[N-1] runs
// through the same animation primitive as the forward path. M9.3 is
// QTM-only so a one-liner suffices.
function inverseMove(m: MoveStr): MoveStr {
  return (m.endsWith("'") ? (m.slice(0, -1) as MoveStr) : (`${m}'` as MoveStr));
}

// Manual-mode animation spec: capture direction + landing index so a
// single useMemo can produce the right startFacelet + (possibly
// inverted) move slice.
type ManualAnim =
  | { kind: "snap" }
  | { kind: "forward"; landIdx: number }
  | { kind: "backward"; landIdx: number };

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
  // -------- Manual mode bookkeeping --------
  // `manualAnim` is the current single-step animation in manual mode.
  // null/`snap` means render static at activeIdx.
  const [manualAnim, setManualAnim] = useState<ManualAnim>({ kind: "snap" });
  const prevActiveIdxRef = useRef(activeIdx);

  // Detect activeIdx changes for manual-mode animation. Skips entirely
  // while playing — the play sequence is the authoritative driver
  // then, and any activeIdx changes during play are reflections of
  // the sequence's own ticks (via `onAutoAdvance`).
  useEffect(() => {
    const prev = prevActiveIdxRef.current;
    prevActiveIdxRef.current = activeIdx;
    if (isPlaying) {
      // While playing, activeIdx is owned by the sequence — don't
      // run the manual-mode diff machine.
      return;
    }
    if (activeIdx === prev) return;

    if (activeIdx > prev && activeIdx > 0 && activeIdx <= moves.length) {
      setManualAnim({ kind: "forward", landIdx: activeIdx });
    } else if (
      activeIdx < prev &&
      activeIdx >= 0 &&
      activeIdx < moves.length
    ) {
      setManualAnim({ kind: "backward", landIdx: activeIdx });
    } else {
      setManualAnim({ kind: "snap" });
    }
  }, [activeIdx, moves.length, isPlaying]);

  // -------- Play mode bookkeeping --------
  // playStartIdx is the activeIdx captured at the moment isPlaying
  // flipped to true. The play sequence is built from this; subsequent
  // activeIdx changes during play come from the sequence and don't
  // re-trigger spec rebuild.
  //
  // We derive it via the "store information from previous renders"
  // pattern: a parallel state mirrors the play state, and when the
  // mirror disagrees we snapshot the current activeIdx. This lands
  // the snapshot the same render `isPlaying` flips, not one effect
  // tick later (which would cause a one-frame mismatch between the
  // play button reading "pause" and the sequence still being built
  // from a stale playStartIdx).
  const [prevIsPlaying, setPrevIsPlaying] = useState(isPlaying);
  const [playStartIdx, setPlayStartIdx] = useState<number | null>(
    isPlaying ? activeIdx : null,
  );
  if (prevIsPlaying !== isPlaying) {
    setPrevIsPlaying(isPlaying);
    if (isPlaying) {
      setPlayStartIdx(activeIdx);
      // Entering play also clears manual animation so the spec
      // selector below picks the play branch on the next render.
      setManualAnim({ kind: "snap" });
    } else {
      setPlayStartIdx(null);
    }
  }

  // -------- Spec selection --------
  // Play spec: one sequence over moves[playStartIdx..end]. Manual spec:
  // single-move or empty. Two useMemos with disjoint deps so the play
  // spec stays referentially stable as activeIdx ticks during play
  // (no rebuild of the sequence between moves — that was the source
  // of the inter-step flash).
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
    if (manualAnim.kind === "forward") {
      const i = manualAnim.landIdx;
      return {
        startFacelet: states[i - 1],
        moves: [moves[i - 1]],
        msPerMove: ANIM_MS_PER_MOVE,
      };
    }
    // backward
    const i = manualAnim.landIdx;
    return {
      startFacelet: states[i + 1],
      moves: [inverseMove(moves[i])],
      msPerMove: ANIM_MS_PER_MOVE,
    };
  }, [isPlaying, manualAnim, activeIdx, states, moves]);

  // Pick whichever spec is active. The non-null branch is referentially
  // stable across renders that don't cross modes, so the seq returned
  // by useCubeSequence is stable too. A "neither" fallback covers the
  // (impossible-in-practice) case of both being null.
  const spec = playSpec ?? manualSpec ?? {
    startFacelet: states[0] ?? "",
    moves: [] as MoveStr[],
    msPerMove: ANIM_MS_PER_MOVE,
  };

  const seq = useCubeSequence(spec);

  // Kick off play after the sequence mounts. For play spec, this fires
  // once per play run. For manual single-step spec, once per direction
  // change. For empty-moves snap spec, the moves.length guard skips
  // play() — Cube2D renders the startFacelet statically.
  useEffect(() => {
    if (spec.moves.length > 0) {
      seq.play();
    }
  }, [seq, spec.moves.length]);

  // -------- Play-mode → activeIdx sync --------
  // While playing, each move-boundary crossing in the sequence ticks
  // activeIdx forward via onAutoAdvance. We track the last reported
  // move index per playStartIdx so subscribe-fires that don't cross
  // a boundary (every rAF tick) are no-ops on this path.
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
      // Land at the very end. The last onAutoAdvance fire above
      // already set activeIdx = playStartIdx + (moves.length - 1) + 1
      // = end, so just signal end-of-play here and let App flip
      // isPlaying back to false.
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

  // -------- Rendering --------
  // 2D — always sequence-mode; Cube2D's empty-moves path handles snap.
  const cube2D = (
    <Cube2D
      sequence={seq}
      sizePx={sizePx}
      testId={renderMode === "dual" ? "flat-cube-pair" : null}
    />
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
