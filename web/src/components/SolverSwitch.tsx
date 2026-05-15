// SolverSwitch — segmented control for picking which Solver
// implementation handles section ii's "Solve" button.
//
//   api  → FastAPI backend (existing path; server owns the model)
//   onnx → in-browser onnxruntime-web (lazy-loads the 61 MB model on
//          first activation)
//
// Visual template is RenderModeSwitch (same `.col-seg-inline` class) so
// the header reads as a row of segmented controls. The choice is
// persisted to localStorage by App.tsx.

import type { SolverKind } from "../solver/Solver";

type Props = {
  value: SolverKind;
  onChange: (k: SolverKind) => void;
};

export default function SolverSwitch({ value, onChange }: Props) {
  return (
    <span className="col-seg-inline" data-testid="solver-switch">
      <button
        type="button"
        data-testid="solver-api"
        className={value === "api" ? "on" : ""}
        onClick={() => onChange("api")}
        aria-pressed={value === "api"}
      >
        api
      </button>
      <button
        type="button"
        data-testid="solver-onnx"
        className={value === "onnx" ? "on" : ""}
        onClick={() => onChange("onnx")}
        aria-pressed={value === "onnx"}
      >
        onnx
      </button>
    </span>
  );
}
