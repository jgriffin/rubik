// ApiSolver — wraps the existing FastAPI /api/solve path behind the
// Solver interface. The server already owns model load + MPS warmup +
// beam search; this class only needs to be a thin pass-through.
//
// The modelName is derived from the `model_path` returned by /api/health
// (e.g. "/.../net_final.pt" → "net_final"). setModelPath() lets the App
// push the path in after /api/health resolves, without forcing a fresh
// ApiSolver — which matters because the active solver is keyed on
// solverKind alone (any health-related dep would invalidate a
// mid-download OnnxSolver and trigger a re-download).

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
  private _modelPath: string | null = null;

  setModelPath(modelPath: string | null) {
    this._modelPath = modelPath;
  }

  async ready(): Promise<void> {
    // Nothing to load — the server owns the model. The /api/health gate
    // (warmup_done) is consulted separately by the App's ready flag.
    return;
  }

  info(): SolverInfo {
    return {
      kind: "api",
      provider: "fastapi",
      modelName: deriveModelName(this._modelPath),
      ready: true,
      loadDurationMs: 0,
    };
  }

  async solve(req: SolveRequest): Promise<SolveResponse> {
    return apiSolve(req);
  }

  async dispose(): Promise<void> {
    // Nothing to release.
  }
}
