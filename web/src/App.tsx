import { useEffect, useState } from "react";
import FlatCubeRenderer from "./components/FlatCubeRenderer";
import ScrambleButton from "./components/ScrambleButton";
import SolveButton from "./components/SolveButton";
import MoveList from "./components/MoveList";
import { apiHealth, apiScramble, apiSolve, type Health } from "./api/client";

const SOLVED_3X3 =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [cubeState, setCubeState] = useState<string>(SOLVED_3X3);
  const [solution, setSolution] = useState<string[] | null>(null);
  const [solved, setSolved] = useState<boolean | null>(null);
  const [isSolving, setIsSolving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiHealth()
      .then(setHealth)
      .catch((e) => setHealthError(String(e)));
  }, []);

  async function handleScramble(length: number) {
    setError(null);
    setSolution(null);
    setSolved(null);
    try {
      const r = await apiScramble({ length });
      setCubeState(r.state);
    } catch (e) {
      setError(String(e));
    }
  }

  async function handleSolve() {
    setError(null);
    setIsSolving(true);
    try {
      const r = await apiSolve({ state: cubeState });
      setSolution(r.moves);
      setSolved(r.solved);
    } catch (e) {
      setError(String(e));
    } finally {
      setIsSolving(false);
    }
  }

  const ready = health !== null && health.warmup_done;

  return (
    <main
      style={{ fontFamily: "system-ui, sans-serif", padding: "2rem", maxWidth: 720 }}
    >
      <h1>rubik solver</h1>
      <FlatCubeRenderer facelet={cubeState} sizePx={240} />
      <div style={{ display: "flex", gap: "0.5rem", margin: "1rem 0" }}>
        <ScrambleButton onScramble={handleScramble} disabled={!ready || isSolving} />
        <SolveButton onSolve={handleSolve} disabled={!ready} isSolving={isSolving} />
      </div>
      <MoveList moves={solution} solved={solved} />
      {error && (
        <pre data-testid="api-error" style={{ color: "crimson" }}>
          error: {error}
        </pre>
      )}
      <hr style={{ marginTop: "2rem" }} />
      {healthError && <pre data-testid="health-error">error: {healthError}</pre>}
      {health ? (
        <pre data-testid="health-json">{JSON.stringify(health, null, 2)}</pre>
      ) : (
        <p>loading...</p>
      )}
    </main>
  );
}
