type Props = {
  length: number;
  onLengthChange: (n: number) => void;
  onScramble: () => void;
  disabled?: boolean;
  min?: number;
  max?: number;
};

export default function LengthPack({
  length,
  onLengthChange,
  onScramble,
  disabled,
  min = 1,
  max = 30,
}: Props) {
  return (
    <span className="length-pack" data-testid="length-pack">
      <button
        type="button"
        className="length-pack-btn"
        data-testid="scramble-button"
        onClick={onScramble}
        disabled={disabled}
      >
        scramble
      </button>
      <input
        type="range"
        min={min}
        max={max}
        step={1}
        value={length}
        onChange={(e) => onLengthChange(Number(e.target.value))}
        className="length-pack-slider"
        data-testid="length-slider"
        disabled={disabled}
        aria-label="scramble length"
      />
      <span className="val serif" data-testid="length-readout">
        {length}
      </span>
    </span>
  );
}
