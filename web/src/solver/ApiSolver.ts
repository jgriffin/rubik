// ApiSolver — wraps the existing FastAPI /api/solve path behind the
// Solver interface. The server already owns model load + MPS warmup +
// beam search; this class only needs to be a thin pass-through.
//
// The modelName is derived from the `model_path` returned by /api/health
// (e.g. "/.../net_final.pt" → "net_final"). If health hasn't loaded yet
// we fall back to "fastapi" — the App will recreate this solver once
// health resolves and the proper name will replace it on the next render.

import { apiSolve } from "../api/client";
import type { SolveRequest, SolveResponse } from "../api/client";
import type { Solver, SolverInfo } from "./Solver";

function deriveModelName(modelPath: string | null): string {
  if (!modelPath) return "fastapi";
  const base = modelPath.split("/").pop() || modelPath;
  return base.replace(/\.(pt|safetensors|bin)$/i, "");
}

export class ApiSolver implements Solver {
  readonly kind = "api" as const;
  private _info: SolverInfo;

  constructor(modelPath: string | null) {
    this._info = {
      kind: "api",
      provider: "fastapi",
      modelName: deriveModelName(modelPath),
      ready: true,
      loadDurationMs: 0,
    };
  }

  async ready(): Promise<void> {
    // Nothing to load — the server owns the model. The /api/health gate
    // (warmup_done) is consulted separately by the App's ready flag.
    return;
  }

  info(): SolverInfo {
    return this._info;
  }

  async solve(req: SolveRequest): Promise<SolveResponse> {
    return apiSolve(req);
  }

  async dispose(): Promise<void> {
    // Nothing to release.
  }
}
