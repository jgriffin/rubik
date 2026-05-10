import { useEffect, useMemo, useState } from "react";
import FlatCubeRenderer from "./components/FlatCubeRenderer";
import SolveButton from "./components/SolveButton";
import MoveList from "./components/MoveList";
import StepControls from "./components/StepControls";
import MovesField from "./components/MovesField";
import MoveStripView from "./components/MoveStripView";
import Wordmark from "./components/Wordmark";
import CubeSizeSwitch from "./components/CubeSizeSwitch";
import SolvedFooter from "./components/SolvedFooter";
import SectionHeader from "./components/SectionHeader";
import StateGrid from "./components/StateGrid";
import LengthPack from "./components/LengthPack";
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

const STRIP_SIZES = { small: 80, medium: 120, large: 160 } as const;
type StripSize = keyof typeof STRIP_SIZES;

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
  const [scrambleLength, setScrambleLength] = useState<number>(14);
  const [solution, setSolution] = useState<string[] | null>(null);
  const [solved, setSolved] = useState<boolean | null>(null);
  const [solveStats, setSolveStats] = useState<SolveStats | null>(null);
  const [isSolving, setIsSolving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stepIdx, setStepIdx] = useState(0);
  const [stripSize, setStripSize] = useState<StripSize>("medium");
  const [cubeSize, setCubeSize] = useState<2 | 3>(3);

  useEffect(() => {
    apiHealth()
      .then(setHealth)
      .catch((e) => setHealthError(String(e)));
  }, []);

  function resetSolveState() {
    setSolution(null);
    setSolved(null);
    setSolveStats(null);
    setStepIdx(0);
  }

  async function handleScramble() {
    setError(null);
    resetSolveState();
    try {
      const r = await apiScramble({ length: scrambleLength });
      setScrambleState(r.state);
    } catch (e) {
      setError(String(e));
    }
  }

  async function handleSolve() {
    setError(null);
    setIsSolving(true);
    setStepIdx(0);
    try {
      const r = await apiSolve({ state: scrambleState });
      setSolution(r.moves);
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
  }

  function handleSetState(newState: string) {
    setError(null);
    resetSolveState();
    setScrambleState(newState);
  }

  function handleSetMoves(moves: MoveStr[]) {
    setError(null);
    setStepIdx(0);
    setSolution(moves);
    setSolved(null);
    setSolveStats(null);
  }

  const displayedState = useMemo(() => {
    if (!solution || solution.length === 0) return scrambleState;
    return applyMoves(scrambleState, solution.slice(0, stepIdx) as MoveStr[]);
  }, [scrambleState, solution, stepIdx]);

  const ready = health !== null && health.warmup_done;
  const metaText = formatMeta(health?.model_path ?? null, solveStats?.time_ms ?? null);

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

      {/* Transitional: solve trigger + moves textarea + strip + animation player.
          Section ii (Phase 3) replaces MovesField; section iii (Phase 4) absorbs Solve
          into its header and replaces MoveStripView/StepControls/MoveList. */}
      <div style={{ display: "flex", gap: "0.5rem", margin: "1.5rem 0 1rem" }}>
        <SolveButton onSolve={handleSolve} disabled={!ready} isSolving={isSolving} />
      </div>
      <div style={{ display: "flex", gap: "1rem", margin: "1rem 0", flexWrap: "wrap" }}>
        <MovesField
          key={solution === null ? "null" : solution.join(" ")}
          solution={solution}
          onSetMoves={handleSetMoves}
        />
      </div>

      <div
        data-testid="strip-size-controls"
        style={{
          display: "flex",
          gap: "0.5rem",
          alignItems: "center",
          margin: "1rem 0 0.5rem",
          fontSize: "0.8rem",
        }}
      >
        <span
          style={{
            opacity: 0.6,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          cube size
        </span>
        {(["small", "medium", "large"] as const).map((sz) => (
          <button
            key={sz}
            data-testid={`strip-size-${sz}`}
            onClick={() => setStripSize(sz)}
            style={{
              padding: "0.25rem 0.6rem",
              borderRadius: 4,
              border:
                stripSize === sz
                  ? "1.5px solid #4299ff"
                  : "1.5px solid #ccc",
              background:
                stripSize === sz ? "rgba(66, 153, 255, 0.1)" : "transparent",
              cursor: "pointer",
              fontFamily: "inherit",
              textTransform: "capitalize",
            }}
          >
            {sz}
          </button>
        ))}
      </div>
      <MoveStripView
        scrambleState={scrambleState}
        solution={solution}
        stepIdx={stepIdx}
        onJumpTo={setStepIdx}
        cubeSize={STRIP_SIZES[stripSize]}
      />
      <section>
        <h2
          style={{
            fontSize: "0.9rem",
            opacity: 0.7,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
            marginTop: "2rem",
          }}
        >
          animation player
        </h2>
        <FlatCubeRenderer facelet={displayedState} sizePx={160} />
        <MoveList moves={solution} solved={solved} />
        {solution !== null && solution.length > 0 && (
          <StepControls
            stepIdx={stepIdx}
            totalSteps={solution.length}
            onStepChange={setStepIdx}
            disabled={isSolving}
          />
        )}
      </section>
      {error && (
        <pre data-testid="api-error" style={{ color: "crimson" }}>
          error: {error}
        </pre>
      )}

      <SolvedFooter
        solved={solved}
        moveCount={solution?.length ?? 0}
        metaText={metaText}
      />

      {healthError && <pre data-testid="health-error">error: {healthError}</pre>}
      {health ? (
        <pre data-testid="health-json" style={{ fontSize: 10, color: "var(--dim)" }}>
          {JSON.stringify(health, null, 2)}
        </pre>
      ) : (
        <p>loading...</p>
      )}
    </main>
  );
}
