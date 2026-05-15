// Solver interface — drives the App's solve path uniformly across the
// FastAPI backend (ApiSolver) and the in-browser ONNX runtime
// (OnnxSolver). Block C wires App.tsx to this interface; Block D will
// add a perf-characterization solver behind the same shape.
//
// The contract is intentionally minimal: ready/info/solve/dispose. The
// async ready() lets a heavyweight solver (ONNX session load over the
// network) defer work until the user actually asks; info() lets the UI
// render a loading/diagnostic line while ready() is pending.

import type { SolveRequest, SolveResponse } from "../api/client";

export type SolverKind = "api" | "onnx";
export type ExecutionProvider = "fastapi" | "onnx-webgpu" | "onnx-wasm";

export interface SolverInfo {
  kind: SolverKind;
  provider: ExecutionProvider;
  // Human-readable model identifier — used in the meta line.
  // e.g. "net_final" for the FastAPI path, "net_final.onnx" for ONNX.
  modelName: string;
  ready: boolean;
  // Present while not-yet-ready; the UI can render this as a status hint.
  loadingMessage?: string;
  // Wall time from solver-instantiated → ready resolved. Populated once
  // the solver finishes loading; useful for the diagnostic meta line.
  loadDurationMs?: number;
}

export interface Solver {
  readonly kind: SolverKind;
  /** Resolves when the solver is usable. Idempotent — subsequent awaits
   * return immediately. Safe to call multiple times. */
  ready(): Promise<void>;
  /** Snapshot of current info — UI polls this for display. */
  info(): SolverInfo;
  /** Solve from a kociemba facelet string. Same response shape as the API. */
  solve(req: SolveRequest): Promise<SolveResponse>;
  /** Release resources (close ORT session, etc.). Called when the active
   * solver changes. */
  dispose(): Promise<void> | void;
}
