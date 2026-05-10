import type { Cols } from "./SolutionGrid";

type Props = {
  value: Cols;
  onChange: (c: Cols) => void;
};

const OPTIONS: Array<{ value: Cols; disabled: boolean; reason?: string }> = [
  { value: 1, disabled: true, reason: "single-column row layout ships in v2" },
  { value: 2, disabled: false },
  { value: 3, disabled: false },
  { value: 4, disabled: false },
  { value: 6, disabled: false },
];

export default function ColumnsSwitch({ value, onChange }: Props) {
  return (
    <span className="col-seg-inline" data-testid="columns-switch">
      {OPTIONS.map((opt) => (
        <button
          key={opt.value}
          type="button"
          data-testid={`columns-${opt.value}`}
          className={value === opt.value ? "on" : ""}
          onClick={() => {
            if (!opt.disabled) onChange(opt.value);
          }}
          disabled={opt.disabled}
          title={opt.reason}
        >
          {opt.value}
        </button>
      ))}
    </span>
  );
}
