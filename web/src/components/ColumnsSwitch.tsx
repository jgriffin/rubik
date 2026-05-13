import type { Cols } from "./SolutionGrid";

// One segmented control wrapping both `cube` (value=1, single-cube view)
// and the column counts 2..6 — they're mutually exclusive view layouts,
// so they share one `.col-seg-inline` block visually.

type Props = {
  value: Cols;
  onChange: (c: Cols) => void;
};

const COL_OPTIONS: Cols[] = [2, 3, 4, 5, 6];

export default function ColumnsSwitch({ value, onChange }: Props) {
  return (
    <span className="col-seg-inline" data-testid="columns-switch">
      <button
        type="button"
        data-testid="cube-mode-button"
        className={value === 1 ? "on" : ""}
        onClick={() => onChange(1)}
      >
        cube
      </button>
      {COL_OPTIONS.map((n) => (
        <button
          key={n}
          type="button"
          data-testid={`columns-${n}`}
          className={value === n ? "on" : ""}
          onClick={() => onChange(n)}
        >
          {n}
        </button>
      ))}
    </span>
  );
}
