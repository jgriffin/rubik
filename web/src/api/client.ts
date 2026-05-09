export type Health = {
  model_loaded: boolean;
  model_path: string;
  warmup_done: boolean;
  cube_size: number;
};

export type ScrambleRequest = {
  length?: number;
  seed?: number | null;
};

export type ScrambleResponse = {
  moves: string[];
  state: string;
};

export type SolveRequest = {
  state: string;
  beam_width?: number;
  max_steps?: number;
};

export type SolveStats = {
  time_ms: number;
  beam_width: number;
  steps_searched: number;
  final_value: number;
};

export type SolveResponse = {
  solved: boolean;
  moves: string[];
  stats: SolveStats;
};

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    throw new Error(`${path} ${r.status}: ${text}`);
  }
  return r.json() as Promise<T>;
}

export async function apiHealth(): Promise<Health> {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error(`/api/health ${r.status}`);
  return r.json() as Promise<Health>;
}

export function apiScramble(req: ScrambleRequest): Promise<ScrambleResponse> {
  return postJson("/api/scramble", req);
}

export function apiSolve(req: SolveRequest): Promise<SolveResponse> {
  return postJson("/api/solve", req);
}
