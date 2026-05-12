import { useEffect, useState } from "react";
import SolveButton from "./components/SolveButton";
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
    setError(null);
    setIsSolving(true);
    setActiveIdx(0);
    try {
      const r = await apiSolve({ state: scrambleState });
      setMoves(r.moves as MoveStr[]);
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

  const ready = health !== null && health.warmup_done;
  const metaText = formatMeta(health?.model_path ?? null, solveStats?.time_ms ?? null);

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

      {/* Section ii — moves to apply (the current scramble) */}
      <SectionHeader
        roman="ii."
        name="moves to apply"
        right={
          <span>
            {scrambleMoves.length} {scrambleMoves.length === 1 ? "move" : "moves"}
          </span>
        }
      />
      <MovesGrid moves={scrambleMoves} />

      {/* Section iii — solution (cards with column + render toggles) */}
      <SectionHeader
        roman="iii."
        name="solution"
        right={
          <>
            <SolveButton onSolve={handleSolve} disabled={!ready} isSolving={isSolving} />
            <span className="scramble-divider" />
            <span className="seg-label">render</span>
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
