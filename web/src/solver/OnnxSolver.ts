// OnnxSolver — runs Block B's beamSolve in the browser, scoring states
// via an onnxruntime-web InferenceSession over net_final.onnx.
//
// Execution provider selection:
//   - WebGPU when `navigator.gpu` is present (preferred — single-tensor
//     calls of B≤1536 stay on-device).
//   - WASM otherwise. WASM is single-threaded for now (numThreads=1) —
//     multi-threaded WASM needs SharedArrayBuffer, which COOP/COEP
//     headers enable but Block D will calibrate whether pthread spawn
//     wins anything at our batch sizes.
//   - URL param `?ep=webgpu` / `?ep=wasm` overrides auto-select, so
//     Block D can pin EP for diagnostic comparison.
//   - WebGPU fall-back to WASM at session-create time: navigator.gpu can
//     exist (Chrome 113+, Safari TP) but session creation still throws
//     on missing flags / context-creation errors. The catch path retries
//     with WASM rather than surfacing an opaque init error.
//
// Lazy load: the 61 MB .onnx + .onnx.data download is deferred until
// the first ready() / solve() call, so users who stay on the FastAPI
// path don't pay for it. ready() is idempotent.

import * as ort from "onnxruntime-web";
import { beamSolve } from "./beam";
import type { ValueFn } from "./beam";
import { faceletToState } from "./facelet";
import { moveIdxToStr } from "./moves";
import { N_STICKERS_3X3 } from "./moveTables";
import type { Solver, SolverInfo, ExecutionProvider } from "./Solver";
import type { SolveRequest, SolveResponse } from "../api/client";

// Where vite-plugin-static-copy stages onnxruntime-web's runtime .wasm
// + .mjs workers. ort.env.wasm.wasmPaths is a prefix that ORT joins
// with the bundled filenames.
const ORT_WASM_PREFIX = "/ort/";
// Where syncModel.sh stages net_final.onnx (+ external data blob).
const MODEL_URL = "/models/net_final.onnx";
const MODEL_EXTERNAL_DATA_URL = "/models/net_final.onnx.data";
// What the ONNX graph references for its external data — the filename
// is baked into the .onnx by torch.onnx.export's external-data writer.
// ort-web matches by this `path`, not by URL.
const MODEL_EXTERNAL_DATA_PATH = "net_final.onnx.data";

// Read ?ep=webgpu / ?ep=wasm at construction time. Returns null if
// param absent or value unrecognized — caller falls back to auto-detect.
function readEpOverride(): "webgpu" | "wasm" | null {
  if (typeof window === "undefined") return null;
  try {
    const v = new URLSearchParams(window.location.search).get("ep");
    if (v === "webgpu" || v === "wasm") return v;
  } catch {
    /* malformed URL, fall through to auto-detect */
  }
  return null;
}

function autoDetectEp(): "webgpu" | "wasm" {
  if (
    typeof navigator !== "undefined" &&
    typeof (navigator as Navigator & { gpu?: unknown }).gpu !== "undefined"
  ) {
    return "webgpu";
  }
  return "wasm";
}

let _wasmConfigured = false;
function configureWasmEnvOnce() {
  if (_wasmConfigured) return;
  // Point ORT at our staged runtime; pin numThreads=1 to skip the
  // pthread spawn path (SAB + worker module). COOP/COEP are set in
  // vite.config.ts so flipping this in Block D is a one-line change.
  ort.env.wasm.wasmPaths = ORT_WASM_PREFIX;
  ort.env.wasm.numThreads = 1;
  _wasmConfigured = true;
}

export class OnnxSolver implements Solver {
  readonly kind = "onnx" as const;
  private _session: ort.InferenceSession | null = null;
  private _ep: ExecutionProvider;
  private _info: SolverInfo;
  private _readyPromise: Promise<void> | null = null;
  private _t0: number;

  constructor() {
    configureWasmEnvOnce();
    const override = readEpOverride();
    const initial = override ?? autoDetectEp();
    this._ep = initial === "webgpu" ? "onnx-webgpu" : "onnx-wasm";
    this._t0 = performance.now();
    this._info = {
      kind: "onnx",
      provider: this._ep,
      modelName: "net_final.onnx",
      ready: false,
      loadingMessage: "loading model…",
    };
  }

  ready(): Promise<void> {
    if (!this._readyPromise) {
      this._readyPromise = this._init();
    }
    return this._readyPromise;
  }

  private async _init(): Promise<void> {
    // ort-web doesn't auto-fetch the external-data sidecar — the graph
    // references it by filename ("net_final.onnx.data") and the runtime
    // throws "Module.MountedFiles is not available" without the explicit
    // descriptor. `path` MUST match what the .onnx graph baked in;
    // `data` is the URL we'll actually fetch.
    const sessionOpts: ort.InferenceSession.SessionOptions = {
      executionProviders:
        this._ep === "onnx-webgpu" ? ["webgpu"] : ["wasm"],
      externalData: [
        { data: MODEL_EXTERNAL_DATA_URL, path: MODEL_EXTERNAL_DATA_PATH },
      ],
    };
    try {
      this._session = await ort.InferenceSession.create(
        MODEL_URL,
        sessionOpts,
      );
    } catch (e) {
      if (this._ep === "onnx-webgpu") {
        // WebGPU init can fail mid-create even when navigator.gpu exists
        // (flag-gated browsers, adapter-not-found). Retry on WASM rather
        // than surfacing an opaque error to the user.
        console.warn(
          "OnnxSolver: WebGPU init failed, falling back to WASM:",
          e,
        );
        this._ep = "onnx-wasm";
        sessionOpts.executionProviders = ["wasm"];
        this._session = await ort.InferenceSession.create(
          MODEL_URL,
          sessionOpts,
        );
      } else {
        throw e;
      }
    }
    this._info = {
      ...this._info,
      ready: true,
      provider: this._ep,
      loadingMessage: undefined,
      loadDurationMs: Math.round(performance.now() - this._t0),
    };
  }

  info(): SolverInfo {
    // Keep loadDurationMs ticking while not yet ready so the UI can show
    // elapsed-load time in the meta line.
    if (!this._info.ready) {
      return {
        ...this._info,
        loadDurationMs: Math.round(performance.now() - this._t0),
      };
    }
    return this._info;
  }

  // ValueFn passed to beamSolve. The Python server uses int64 sticker
  // tensors of shape [B, 54] under the same `states` input name. The
  // output is named `value` (single scalar per row). int64 in JS is
  // BigInt64Array — packs one BigInt per sticker, B*54 entries total.
  private _valueFn: ValueFn = async (states) => {
    if (!this._session) throw new Error("OnnxSolver session not initialized");
    const B = states.length;
    const flat = new BigInt64Array(B * N_STICKERS_3X3);
    for (let i = 0; i < B; i++) {
      const s = states[i];
      const base = i * N_STICKERS_3X3;
      for (let j = 0; j < N_STICKERS_3X3; j++) {
        flat[base + j] = BigInt(s[j]);
      }
    }
    const tensor = new ort.Tensor("int64", flat, [B, N_STICKERS_3X3]);
    const out = await this._session.run({ states: tensor });
    // The 3x3 export emits a single output named "value" of shape [B].
    return out.value.data as Float32Array;
  };

  async solve(req: SolveRequest): Promise<SolveResponse> {
    await this.ready();
    const beamWidth = req.beam_width ?? 128;
    const maxSteps = req.max_steps ?? 22;
    const t0 = performance.now();
    const initial = faceletToState(req.state);
    const result = await beamSolve(initial, this._valueFn, {
      beamWidth,
      maxSteps,
    });
    const time_ms = Math.round(performance.now() - t0);
    return {
      solved: result.solved,
      moves: result.moves.map(moveIdxToStr),
      stats: {
        time_ms,
        beam_width: beamWidth,
        steps_searched: result.steps,
        // The Python server reports the V of the surviving best beam
        // slot at solve time. We don't track that here yet — return 0.0
        // as a placeholder; Block D may surface a richer stats object.
        final_value: 0.0,
      },
    };
  }

  async dispose(): Promise<void> {
    if (this._session) {
      // release() is the ORT teardown hook; null-check covers the case
      // where dispose is called before ready resolved.
      await this._session.release();
      this._session = null;
    }
  }
}
