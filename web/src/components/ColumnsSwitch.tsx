import type { Cols } from "./SolutionGrid";

type Props = {
  value: Cols;
  onChange: (c: Cols) => void;
};

const OPTIONS: Cols[] = [2, 3, 4, 5, 6];

export default function ColumnsSwitch({ value, onChange }: Props) {
  return (
    <span className="col-seg-inline" data-testid="columns-switch">
      {OPTIONS.map((n) => (
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
