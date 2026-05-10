type Props = {
  value: 2 | 3;
  onChange: (n: 2 | 3) => void;
};

export default function CubeSizeSwitch({ value, onChange }: Props) {
  return (
    <span className="col-seg-inline" data-testid="cube-size-switch">
      <button
        type="button"
        data-testid="cube-size-2"
        className={value === 2 ? "on" : ""}
        onClick={() => onChange(2)}
      >
        2×2
      </button>
      <button
        type="button"
        data-testid="cube-size-3"
        className={value === 3 ? "on" : ""}
        onClick={() => onChange(3)}
      >
        3×3
      </button>
    </span>
  );
}
