import { useEffect, useState } from "react";
import FlatCubeRenderer from "./components/FlatCubeRenderer";

type Health = {
  model_loaded: boolean;
  model_path: string;
  warmup_done: boolean;
  cube_size: number;
};

const SOLVED_3X3 =
  "U".repeat(9) +
  "R".repeat(9) +
  "F".repeat(9) +
  "D".repeat(9) +
  "L".repeat(9) +
  "B".repeat(9);

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", padding: "2rem" }}>
      <h1>rubik solver</h1>
      <FlatCubeRenderer facelet={SOLVED_3X3} sizePx={240} />
      {error && <pre data-testid="health-error">error: {error}</pre>}
      {health ? (
        <pre data-testid="health-json">{JSON.stringify(health, null, 2)}</pre>
      ) : (
        <p>loading...</p>
      )}
    </main>
  );
}
