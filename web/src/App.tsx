import { useEffect, useState } from "react";

type Health = {
  model_loaded: boolean;
  model_path: string;
  warmup_done: boolean;
  cube_size: number;
};

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
      {error && <pre data-testid="health-error">error: {error}</pre>}
      {health ? (
        <pre data-testid="health-json">{JSON.stringify(health, null, 2)}</pre>
      ) : (
        <p>loading...</p>
      )}
    </main>
  );
}
