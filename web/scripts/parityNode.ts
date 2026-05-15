// M11 B·P3: Node-side ONNX parity runner.
//
// Reads tests/data/m11_parity_corpus_3x3.json (produced by
// scripts/parity_python_reference_3x3.py), loads the Block A .onnx via
// onnxruntime-node (CPU EP), and runs the TS beam from
// web/src/solver/beam.ts against each row. Compares aggregate metrics
// against the Python reference and exits non-zero if any tolerance is
// violated.
//
// Run from the repo root:
//     pnpm --filter rubik-web exec tsx web/scripts/parityNode.ts
// or from web/:
//     pnpm tsx scripts/parityNode.ts

import { promises as fs } from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-node";

import { beamSolve, type State, type ValueFn } from "../src/solver/beam";
import { faceletToState } from "../src/solver/facelet";
import { N_STICKERS_3X3 } from "../src/solver/moveTables";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..", "..");

const CORPUS_PATH = path.join(
  REPO_ROOT,
  "tests",
  "data",
  "m11_parity_corpus_3x3.json",
);
const ONNX_PATH = path.join(
  REPO_ROOT,
  "experiments",
  "davi-3x3",
  "runs",
  "20260508T084940Z_ln_kmax30_100k",
  "net_final.onnx",
);
const REPORT_PATH = path.join(__dirname, "parity-report.json");

// Aggregate parity gate.
const SOLVE_RATE_TOL_PP = 0.05; // ±5 percentage points
const MEAN_LEN_TOL = 1.5; // moves
const PER_ROW_AGREEMENT_MIN = 0.8; // 80% of rows agree on solved-status

interface CorpusRow {
  facelet: string;
  solve_len: number;
  solved: boolean;
  solve_moves: number[];
}

interface Corpus {
  seed: string;
  scramble_depth: number;
  beam_width: number;
  max_steps: number;
  batch_size: number;
  py_runtime_ms: number;
  py_solve_rate: number;
  py_mean_solve_len: number;
  py_n_solved: number;
  rows: CorpusRow[];
}

interface TsRow {
  facelet: string;
  py_solved: boolean;
  py_solve_len: number;
  ts_solved: boolean;
  ts_solve_len: number; // -1 if unsolved
  ts_moves: number[];
  agree: boolean;
}

function makeValueFn(session: ort.InferenceSession): ValueFn {
  return async (states: State[]) => {
    const batch = states.length;
    const data = new BigInt64Array(batch * N_STICKERS_3X3);
    for (let i = 0; i < batch; i++) {
      const s = states[i];
      for (let j = 0; j < N_STICKERS_3X3; j++) {
        data[i * N_STICKERS_3X3 + j] = BigInt(s[j]);
      }
    }
    const input = new ort.Tensor("int64", data, [batch, N_STICKERS_3X3]);
    const out = await session.run({ states: input });
    const tensor = out.value;
    // onnxruntime-node returns Float32Array for float32 outputs.
    return tensor.data as Float32Array;
  };
}

async function main(): Promise<void> {
  const corpusJson = await fs.readFile(CORPUS_PATH, "utf8");
  const corpus: Corpus = JSON.parse(corpusJson);
  console.log(
    `loaded corpus: ${corpus.rows.length} rows, seed=${corpus.seed},` +
      ` depth=${corpus.scramble_depth}, width=${corpus.beam_width},` +
      ` max_steps=${corpus.max_steps}`,
  );
  console.log(
    `py reference: solve_rate=${corpus.py_n_solved}/${corpus.batch_size}` +
      ` (${(100 * corpus.py_solve_rate).toFixed(1)}%),` +
      ` mean_solve_len=${corpus.py_mean_solve_len.toFixed(1)},` +
      ` wall=${(corpus.py_runtime_ms / 1000).toFixed(1)}s`,
  );

  console.log(`loading onnx: ${ONNX_PATH}`);
  const session = await ort.InferenceSession.create(ONNX_PATH, {
    executionProviders: ["cpu"],
  });
  const valueFn = makeValueFn(session);

  const t0 = performance.now();
  const tsRows: TsRow[] = new Array(corpus.rows.length);
  for (let i = 0; i < corpus.rows.length; i++) {
    const row = corpus.rows[i];
    const state = faceletToState(row.facelet);
    const r = await beamSolve(state, valueFn, {
      beamWidth: corpus.beam_width,
      maxSteps: corpus.max_steps,
    });
    const tsSolveLen = r.solved ? r.moves.length : -1;
    tsRows[i] = {
      facelet: row.facelet,
      py_solved: row.solved,
      py_solve_len: row.solve_len,
      ts_solved: r.solved,
      ts_solve_len: tsSolveLen,
      ts_moves: r.moves,
      agree: r.solved === row.solved,
    };
  }
  const tsWallMs = Math.round(performance.now() - t0);

  const tsNSolved = tsRows.filter((r) => r.ts_solved).length;
  const tsSolveRate = tsNSolved / tsRows.length;
  const tsSolvedLens = tsRows.filter((r) => r.ts_solved).map((r) => r.ts_solve_len);
  const tsMeanSolveLen = tsSolvedLens.length
    ? tsSolvedLens.reduce((a, b) => a + b, 0) / tsSolvedLens.length
    : 0;
  const nAgree = tsRows.filter((r) => r.agree).length;
  const perRowAgreement = nAgree / tsRows.length;

  // Mean solve_len comparison over rows solved by BOTH (avoids unsolved
  // rows polluting the average asymmetrically).
  const bothSolvedLensTs = tsRows
    .filter((r) => r.ts_solved && r.py_solved)
    .map((r) => r.ts_solve_len);
  const bothSolvedLensPy = tsRows
    .filter((r) => r.ts_solved && r.py_solved)
    .map((r) => r.py_solve_len);
  const meanLenBothTs = bothSolvedLensTs.length
    ? bothSolvedLensTs.reduce((a, b) => a + b, 0) / bothSolvedLensTs.length
    : 0;
  const meanLenBothPy = bothSolvedLensPy.length
    ? bothSolvedLensPy.reduce((a, b) => a + b, 0) / bothSolvedLensPy.length
    : 0;

  const diffs = {
    solve_rate_abs_delta: Math.abs(tsSolveRate - corpus.py_solve_rate),
    mean_solve_len_abs_delta_both_solved: Math.abs(meanLenBothTs - meanLenBothPy),
    per_row_agreement: perRowAgreement,
  };

  const gates = {
    solve_rate_ok: diffs.solve_rate_abs_delta <= SOLVE_RATE_TOL_PP,
    mean_solve_len_ok: diffs.mean_solve_len_abs_delta_both_solved <= MEAN_LEN_TOL,
    per_row_agreement_ok: diffs.per_row_agreement >= PER_ROW_AGREEMENT_MIN,
  };
  const allPass = Object.values(gates).every(Boolean);

  const report = {
    corpus: {
      path: CORPUS_PATH,
      seed: corpus.seed,
      scramble_depth: corpus.scramble_depth,
      beam_width: corpus.beam_width,
      max_steps: corpus.max_steps,
      batch_size: corpus.batch_size,
    },
    py: {
      n_solved: corpus.py_n_solved,
      solve_rate: corpus.py_solve_rate,
      mean_solve_len: corpus.py_mean_solve_len,
      runtime_ms: corpus.py_runtime_ms,
    },
    ts: {
      n_solved: tsNSolved,
      solve_rate: tsSolveRate,
      mean_solve_len: tsMeanSolveLen,
      mean_solve_len_both_solved: meanLenBothTs,
      runtime_ms: tsWallMs,
    },
    py_mean_solve_len_both_solved: meanLenBothPy,
    diffs,
    gates,
    pass: allPass,
    tolerances: {
      solve_rate_tol_pp: SOLVE_RATE_TOL_PP,
      mean_solve_len_tol: MEAN_LEN_TOL,
      per_row_agreement_min: PER_ROW_AGREEMENT_MIN,
    },
    rows: tsRows,
  };

  await fs.writeFile(REPORT_PATH, JSON.stringify(report, null, 2) + "\n");

  const verdict = allPass ? "parity OK" : "PARITY FAIL";
  console.log(
    `${verdict}: TS solve_rate=${tsNSolved}/${tsRows.length}` +
      ` (${(100 * tsSolveRate).toFixed(1)}%) vs PY=${corpus.py_n_solved}/${corpus.batch_size}` +
      ` (${(100 * corpus.py_solve_rate).toFixed(1)}%);` +
      ` mean_solve_len_TS=${meanLenBothTs.toFixed(1)},` +
      ` PY=${meanLenBothPy.toFixed(1)} (over both-solved);` +
      ` per-row agreement=${(100 * perRowAgreement).toFixed(1)}%;` +
      ` node wall=${(tsWallMs / 1000).toFixed(1)}s`,
  );
  console.log(`gates: ${JSON.stringify(gates)}`);
  console.log(`wrote: ${REPORT_PATH}`);

  if (!allPass) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
