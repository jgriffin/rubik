import { useEffect, useMemo, useState } from "react";
import Wordmark from "./components/Wordmark";
import CubeSizeSwitch from "./components/CubeSizeSwitch";
import SolvedFooter from "./components/SolvedFooter";
import SectionHeader from "./components/SectionHeader";
import StateGrid from "./components/StateGrid";
import LengthPack from "./components/LengthPack";
import MovesGrid from "./components/MovesGrid";
import SolutionGrid, { type Cols, type RenderMode } from "./components/SolutionGrid";
import RenderModeSwitch from "./components/RenderModeSwitch";
import ColumnsSwitch from "./components/ColumnsSwitch";
import PlayPauseButton from "./components/PlayPauseButton";
import { applyMoves } from "./state/applyMove";
import type { MoveStr } from "./state/faceletMoves";
import { apiHealth, apiScramble, apiSolve, type Health, type SolveStats } from "./api/client";
import { ANIM_MS_PER_MOVE, PLAY_STEP_DWELL_MS } from "./components/cubeStageConstants";

const SOLVED_3X3 =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);

function formatMeta(modelPath: string | null, timeMs: number | null): string | null {
  if (!modelPath || timeMs == null) return null;
  const base = modelPath.split("/").pop() || modelPath;
  const noExt = base.replace(/\.(pt|safetensors|bin)$/i, "");
  return `${noExt} · ${timeMs} ms`;
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [scrambleState, setScrambleState] = useState<string>(SOLVED_3X3);
  const [scrambleMoves, setScrambleMoves] = useState<MoveStr[]>([]);
  const [scrambleLength, setScrambleLength] = useState<number>(14);
  const [moves, setMoves] = useState<MoveStr[]>([]);
  const [solved, setSolved] = useState<boolean | null>(null);
  const [solveStats, setSolveStats] = useState<SolveStats | null>(null);
  const [isSolving, setIsSolving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeIdx, setActiveIdx] = useState(0);
  const [cubeSize, setCubeSize] = useState<2 | 3>(3);
  const [renderMode, setRenderMode] = useState<RenderMode>("net");
  const [cols, setCols] = useState<Cols>(3);
  // Section ii owns trajectory transport: the play/pause control lives
  // in the leading [start] cell and drives `activeIdx` forward at a
  // steady cadence. State here in App so the button can sit in MovesGrid
  // while the animation (CubeStage) reads activeIdx independently. The
  // cadence (ANIM_MS_PER_MOVE + PLAY_STEP_DWELL_MS) is paired with the
  // 2D cube animation duration so each step's animation completes before
  // the next is scheduled — see `cubeStageConstants.ts`.
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    apiHealth()
      .then(setHealth)
      .catch((e) => setHealthError(String(e)));
  }, []);

  function resetSolveState() {
    setMoves([]);
    setSolved(null);
    setSolveStats(null);
    setActiveIdx(0);
    setIsPlaying(false);
  }

  async function handleScramble() {
    setError(null);
    resetSolveState();
    try {
      const r = await apiScramble({ length: scrambleLength });
      setScrambleState(r.state);
      setScrambleMoves(r.moves as MoveStr[]);
    } catch (e) {
      setError(String(e));
    }
  }

  async function handleSolve() {
    // Solve from the cube's *current* state (= scrambleState with the
    // existing `moves` applied). Append solver moves to existing —
    // trajectory tells the story start → user-tinkering → solve. Stale
    // closure caveat: use the functional setMoves form so a race
    // against in-flight edits doesn't drop appended moves (the editor
    // is disabled while isSolving=true so this is belt-and-suspenders).
    setError(null);
    setIsPlaying(false);
    setIsSolving(true);
    const stateAtSolve = applyMoves(scrambleState, moves);
    try {
      const r = await apiSolve({ state: stateAtSolve });
      const newMoves = r.moves as MoveStr[];
      setMoves((prev) => [...prev, ...newMoves]);
      setSolved(r.solved);
      setSolveStats(r.stats);
    } catch (e) {
      setError(String(e));
    } finally {
      setIsSolving(false);
    }
  }

  function handleClear() {
    setError(null);
    resetSolveState();
    setScrambleState(SOLVED_3X3);
    setScrambleMoves([]);
  }

  function handleSetState(newState: string) {
    setError(null);
    resetSolveState();
    setScrambleState(newState);
    setScrambleMoves([]);
  }

  function handleMovesEdit(next: MoveStr[]) {
    // User edited section ii directly. Clear the solve verdict
    // (`solved`/`solveStats` describe the last Solve, not a hand-typed
    // sequence), clamp `activeIdx` so a previously-selected late card
    // doesn't outlive a delete, and pause auto-play — the user has
    // taken manual control of the editor, playback should yield.
    setMoves(next);
    setSolved(null);
    setSolveStats(null);
    setActiveIdx((i) => Math.min(i, next.length));
    setIsPlaying(false);
  }

  function handleTogglePlay() {
    if (isPlaying) {
      setIsPlaying(false);
      return;
    }
    // At end-of-moves, rewind to start before kicking off — playing
    // from the end position would immediately stop. Re-watching the
    // whole sequence is the natural "play" interaction here.
    if (activeIdx >= moves.length) {
      setActiveIdx(0);
    }
    setIsPlaying(true);
  }

  // Auto-advance timer. When `isPlaying`, schedule the next step
  // (activeIdx + 1) after one full step duration (animation + dwell).
  // Each activeIdx change re-fires this effect and schedules the next
  // step; reaching the end folds setIsPlaying(false) into the timer
  // callback so its setState lands inside an event, not the effect body.
  useEffect(() => {
    if (!isPlaying || activeIdx >= moves.length) return;
    const timer = setTimeout(() => {
      const nextIdx = activeIdx + 1;
      setActiveIdx(nextIdx);
      if (nextIdx >= moves.length) setIsPlaying(false);
    }, ANIM_MS_PER_MOVE + PLAY_STEP_DWELL_MS);
    return () => clearTimeout(timer);
  }, [isPlaying, activeIdx, moves.length]);

  const ready = health !== null && health.warmup_done;
  const metaText = formatMeta(health?.model_path ?? null, solveStats?.time_ms ?? null);

  // The cube's *current* state — start state with section ii's moves
  // applied. Drives the Solve affordance: visible only when the cube
  // isn't already solved.
  const currentState = useMemo(
    () => applyMoves(scrambleState, moves),
    [scrambleState, moves],
  );
  const isCubeSolved = currentState === SOLVED_3X3;
  // The starting state itself is solved (common case after Clear, when
  // no scramble has been generated). Drives the "solved" label in section
  // ii's leading [start] cell — symmetric with the trailing cell's
  // "solved" label when the end state is solved.
  const isStartSolved = scrambleState === SOLVED_3X3;
  const canSolve = ready && !isSolving && !isCubeSolved;

  return (
    <main className="col">
      <header className="head">
        <Wordmark />
        <div className="right">
          <CubeSizeSwitch value={cubeSize} onChange={setCubeSize} />
        </div>
      </header>

      {/* Section i — starting state */}
      <SectionHeader
        roman="i."
        name="starting state"
        right={
          <>
            <button
              type="button"
              className="link-btn"
              data-testid="clear-button"
              onClick={handleClear}
              disabled={!ready || isSolving}
              title="reset to solved cube"
            >
              clear
            </button>
            <span className="scramble-divider" />
            <LengthPack
              length={scrambleLength}
              onLengthChange={setScrambleLength}
              onScramble={handleScramble}
              disabled={!ready || isSolving}
            />
          </>
        }
      />
      <StateGrid state={scrambleState} onStateChange={handleSetState} />

      {/* Section ii — moves to apply (editable; the source of truth).
          Solve lives inside the grid as a trailing cell when the cube
          isn't already solved (always-available solve-from-here). */}
      <SectionHeader
        roman="ii."
        name="moves to apply"
        right={
          <>
            <button
              type="button"
              className="link-btn"
              data-testid="clear-moves-button"
              onClick={() => handleMovesEdit([])}
              disabled={!ready || isSolving || moves.length === 0}
              title="erase all moves"
            >
              clear
            </button>
            <span className="scramble-divider" />
            <span>
              {moves.length} {moves.length === 1 ? "move" : "moves"}
            </span>
          </>
        }
      />
      <MovesGrid
        moves={moves}
        onMovesChange={handleMovesEdit}
        activeIdx={activeIdx}
        onActiveChange={setActiveIdx}
        canSolve={canSolve}
        isCubeSolved={isCubeSolved}
        isStartSolved={isStartSolved}
        onSolve={handleSolve}
        isSolving={isSolving}
        disabled={!ready || isSolving}
      />

      {/* Section iii — steps (cards visualize each move in section ii).
          Right-side controls: 2D/3D toggle (left) | view-layout selector
          (cube | 2 | 3 | 4 | 5 | 6). `cube` and the column counts are
          mutually exclusive layouts, so they live in one segmented
          control (`ColumnsSwitch`). The word "columns" is intentionally
          absent — `cube` reads as the special mode and 2-6 implicitly
          mean column counts. */}
      <SectionHeader
        roman="iii."
        name="steps"
        right={
          <>
            <PlayPauseButton
              isPlaying={isPlaying}
              onToggle={handleTogglePlay}
              disabled={!ready || isSolving || moves.length === 0}
            />
            <span className="scramble-divider" />
            <RenderModeSwitch value={renderMode} onChange={setRenderMode} />
            <ColumnsSwitch value={cols} onChange={setCols} />
          </>
        }
      />
      <SolutionGrid
        scrambleState={scrambleState}
        scrambleMoves={scrambleMoves}
        moves={moves}
        isSolving={isSolving}
        cols={cols}
        renderMode={renderMode}
        activeIdx={activeIdx}
        onActiveChange={setActiveIdx}
        isCubeSolved={isCubeSolved}
      />

      {error && (
        <pre data-testid="api-error" style={{ color: "crimson", marginTop: "1rem" }}>
          error: {error}
        </pre>
      )}

      <SolvedFooter
        solved={solved}
        moveCount={moves.length}
        metaText={metaText}
      />

      {healthError && <pre data-testid="health-error">error: {healthError}</pre>}
      {!health && !healthError && (
        <p style={{ color: "var(--dim)", fontSize: 11, marginTop: "1rem" }}>
          loading…
        </p>
      )}
    </main>
  );
}
