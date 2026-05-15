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
import { apiHealth, apiScramble, type Health, type SolveStats } from "./api/client";
import { ApiSolver } from "./solver/ApiSolver";
import { OnnxSolver } from "./solver/OnnxSolver";
import type { Solver, SolverInfo, SolverKind } from "./solver/Solver";
import SolverSwitch from "./components/SolverSwitch";

const SOLVED_3X3 =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);

// Provider strings rendered in the meta line. `fastapi` (server-side
// solve) vs `onnx/webgpu` and `onnx/wasm` (browser-side, with the
// chosen EP exposed so diagnostic runs read at a glance).
function providerLabel(info: SolverInfo): string {
  switch (info.provider) {
    case "fastapi":
      return "fastapi";
    case "onnx-webgpu":
      return "onnx/webgpu";
    case "onnx-wasm":
      return "onnx/wasm";
  }
}

function formatMeta(info: SolverInfo, timeMs: number | null): string | null {
  if (timeMs == null) return null;
  return `${providerLabel(info)} · ${info.modelName} · ${timeMs} ms`;
}

function formatLoadingMeta(info: SolverInfo): string {
  // While the solver is loading we surface elapsed-load time so the user
  // can tell the 61 MB download is making progress.
  const elapsed = info.loadDurationMs ?? 0;
  return `${providerLabel(info)} · loading model… (${elapsed} ms elapsed)`;
}

const SOLVER_LS_KEY = "rubik:solver";

function readSolverKindFromStorage(): SolverKind {
  if (typeof window === "undefined") return "api";
  try {
    const v = window.localStorage.getItem(SOLVER_LS_KEY);
    if (v === "api" || v === "onnx") return v;
  } catch {
    /* SSR / private-mode / cookie-blocked: fall through */
  }
  return "api";
}

// `?width=N` URL param hook — lets the M11 D measurement harness vary
// beam_width per solve without recompiling. Parsed at module load (a
// page navigation per width is the measurement protocol). Clamped to
// [1, 1024] and silently defaults to undefined on bad input so the
// solver falls back to its own default (128). Not exposed in the UI.
function readBeamWidthFromUrl(): number | undefined {
  if (typeof window === "undefined") return undefined;
  try {
    const raw = new URLSearchParams(window.location.search).get("width");
    if (raw == null) return undefined;
    const n = Number.parseInt(raw, 10);
    if (!Number.isFinite(n) || n < 1 || n > 1024) return undefined;
    return n;
  } catch {
    return undefined;
  }
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
  // Trajectory transport. The play/pause button (section iii header)
  // toggles this. The actual playback timing lives inside a multi-move
  // `CubeSequence` owned by `CubeStage` — when `isPlaying` flips true,
  // CubeStage builds one sequence covering all remaining moves and
  // ticks `activeIdx` forward via `handleAutoAdvance` as each move
  // begins; the per-move `gapMs` dwell handles the inter-move pause.
  // No App-level timer involved — keeping the sequence canonical
  // avoids the inter-step DOM swap that the setTimeout-driven design
  // produced as a visible flicker.
  const [isPlaying, setIsPlaying] = useState(false);

  // Active solver kind — persisted across sessions. Default to the
  // FastAPI path so anonymous users hit the well-behaved server first;
  // the ONNX path is opt-in via the header switch.
  const [solverKind, setSolverKind] = useState<SolverKind>(() =>
    readSolverKindFromStorage(),
  );

  // Beam-width override from `?width=N` URL param (M11 D harness).
  // Captured once at construction; unaffected by user interactions.
  const beamWidthOverride = useMemo(() => readBeamWidthFromUrl(), []);

  // Build the active Solver instance. Keyed on solverKind ONLY — the
  // OnnxSolver does a 61 MB download on first ready(), so we cannot let
  // /api/health resolving mid-session re-create it. ApiSolver receives
  // the model_path via a setter from a separate effect, decoupling its
  // info() refresh from its construction.
  const solver: Solver = useMemo(() => {
    if (solverKind === "onnx") return new OnnxSolver();
    return new ApiSolver();
  }, [solverKind]);
  useEffect(() => {
    if (solver instanceof ApiSolver) {
      solver.setModelPath(health?.model_path ?? null);
    }
  }, [solver, health?.model_path]);

  // Solver-info snapshot for the UI. We keep a tick counter and read
  // `solver.info()` fresh on each render — the tick increments via the
  // polling effect below while the solver is still loading and via
  // `solver.ready().then(...)` once it's done. This sidesteps
  // setState-in-effect cascades while still giving the UI a live
  // view of solver state.
  const [tick, setTick] = useState(0);
  const solverInfo: SolverInfo = useMemo(
    () => solver.info(),
    // `tick` is the live signal — it bumps when info() may have
    // changed. `info()` is impure (reads internal solver state) so
    // eslint can't see why `tick` matters; the disable is intentional.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [solver, tick],
  );

  // Surface solver-load errors. The OnnxSolver's session create can
  // reject after the user flips the switch — without explicit surfacing
  // the rejection gets eaten by the fire-and-forget ready() call below
  // and the UI sits forever on "loading model…".
  const [solverError, setSolverError] = useState<string | null>(null);
  // Clear the prior solver's error when the active instance changes —
  // "store info from prior renders" pattern (per project convention)
  // to satisfy react-hooks/set-state-in-effect.
  const [prevSolverForError, setPrevSolverForError] = useState(solver);
  if (prevSolverForError !== solver) {
    setPrevSolverForError(solver);
    setSolverError(null);
  }

  // Drive readiness + polling. The cleanup disposes the previous
  // solver on swap.
  useEffect(() => {
    let cancelled = false;
    const bump = () => {
      if (!cancelled) setTick((t) => t + 1);
    };
    // Kick ready() so OnnxSolver starts downloading right away —
    // users who flip to the onnx switch want the download to start
    // before they click Solve. Catch rejections to surface them, since
    // a hung ready() otherwise leaves the UI on "loading model…".
    void solver.ready().then(bump).catch((e) => {
      if (cancelled) return;
      setSolverError(String(e?.message ?? e));
      bump();
    });
    // While not-yet-ready, poll at 250 ms so the elapsed-load
    // counter ticks. Once ready, stop polling.
    const poll = window.setInterval(() => {
      if (cancelled) return;
      bump();
      if (solver.info().ready) window.clearInterval(poll);
    }, 250);
    return () => {
      cancelled = true;
      window.clearInterval(poll);
      void solver.dispose();
    };
  }, [solver]);

  function handleSolverKindChange(next: SolverKind) {
    if (next === solverKind) return;
    setSolverKind(next);
    try {
      window.localStorage.setItem(SOLVER_LS_KEY, next);
    } catch {
      /* storage unavailable; choice is session-local */
    }
  }

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
      const r = await solver.solve({
        state: stateAtSolve,
        ...(beamWidthOverride != null ? { beam_width: beamWidthOverride } : {}),
      });
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

  // activeIdx change variants.
  // - `handleUserActiveChange`: cell clicks / arrow-key nav / card
  //   clicks. The user has taken manual control of the trajectory, so
  //   auto-play should yield. Wired to MovesGrid + SolutionGrid.
  // - `handleAutoAdvance`: CubeStage's multi-move play sequence ticks
  //   the active state forward as each move's animation begins. We
  //   don't pause on this path — it IS the playback.
  // Splitting the two avoids the "user clicked but playback overwrote"
  // race the setTimeout-based design had. The auto-advance + dwell
  // cadence now lives inside the play sequence itself (gapMs), not in
  // a competing App-level timer.
  function handleUserActiveChange(idx: number) {
    if (isPlaying) setIsPlaying(false);
    setActiveIdx(idx);
  }
  function handleAutoAdvance(idx: number) {
    setActiveIdx(idx);
  }
  function handlePlayEnd() {
    setIsPlaying(false);
  }

  // The page is "ready to solve" when (a) /api/health returned (we
  // need it for scramble + the apiSolver model name) AND (b) the
  // currently-active solver reports ready. For the API path the
  // solver-ready flag is trivially true; for the ONNX path it gates on
  // the 61 MB download finishing.
  const healthReady = health !== null && health.warmup_done;
  const ready = healthReady && solverInfo.ready;
  // Meta line: while the active solver is still loading, surface the
  // load progress. Once ready, show the per-solve diagnostic.
  const metaText = solverInfo.ready
    ? formatMeta(solverInfo, solveStats?.time_ms ?? null)
    : formatLoadingMeta(solverInfo);

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
          <SolverSwitch value={solverKind} onChange={handleSolverKindChange} />
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
        onActiveChange={handleUserActiveChange}
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
        onActiveChange={handleUserActiveChange}
        isPlaying={isPlaying}
        onAutoAdvance={handleAutoAdvance}
        onPlayEnd={handlePlayEnd}
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

      {/* Solver-load status. Surfaces while the active solver is
          downloading / initializing (ONNX path only — ApiSolver is
          ready immediately). Hidden once ready so it doesn't clutter
          the page during normal use. Error variant takes precedence
          over the elapsed-load counter. */}
      {solverError && (
        <pre
          data-testid="solver-error"
          style={{ color: "crimson", fontSize: 11, marginTop: "0.5rem", whiteSpace: "pre-wrap" }}
        >
          solver load failed: {solverError}
        </pre>
      )}
      {!solverError && !solverInfo.ready && (
        <p
          data-testid="solver-loading"
          style={{ color: "var(--dim)", fontSize: 11, marginTop: "0.5rem" }}
        >
          {formatLoadingMeta(solverInfo)}
        </p>
      )}
    </main>
  );
}
