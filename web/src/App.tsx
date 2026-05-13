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
import SectionFour from "./components/SectionFour";
import { applyMoves } from "./state/applyMove";
import type { MoveStr } from "./state/faceletMoves";
import { apiHealth, apiScramble, apiSolve, type Health, type SolveStats } from "./api/client";

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
    // sequence) and clamp `activeIdx` so a previously-selected late
    // card doesn't outlive a delete.
    setMoves(next);
    setSolved(null);
    setSolveStats(null);
    setActiveIdx((i) => Math.min(i, next.length));
  }

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
  const canSolve = ready && !isSolving && !isCubeSolved;

  const sectionFourScrambleAlg = scrambleMoves.join(" ");
  const sectionFourSolutionAlg = moves.join(" ");

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
        onSolve={handleSolve}
        isSolving={isSolving}
        disabled={!ready || isSolving}
      />

      {/* Section iii — steps (cards visualize each move in section ii) */}
      <SectionHeader
        roman="iii."
        name="steps"
        right={
          <>
            <RenderModeSwitch value={renderMode} onChange={setRenderMode} />
            <span className="seg-label">columns</span>
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

      {/* Section iv — watch the solve (animated player). Renders only
          once a solution exists; SectionFour itself returns null on
          empty solutionAlg. */}
      <SectionFour
        scrambleAlg={sectionFourScrambleAlg}
        solutionAlg={sectionFourSolutionAlg}
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
