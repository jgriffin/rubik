// CubeSizeSwitch — top-right main selection between 2x2 and 3x3.
// Styled as elegant inline serif text with a vertical bar separator;
// active size shows in accent (orange), inactive in ink. Echoes the
// wordmark's typographic feel — this is a primary mode selector, not
// chrome.

type Props = {
  value: 2 | 3;
  onChange: (n: 2 | 3) => void;
};

export default function CubeSizeSwitch({ value, onChange }: Props) {
  return (
    <span className="cube-size-switch" data-testid="cube-size-switch">
      <button
        type="button"
        data-testid="cube-size-2"
        className={`cube-size-opt ${value === 2 ? "on" : ""}`}
        onClick={() => onChange(2)}
        aria-pressed={value === 2}
      >
        2×2
      </button>
      <span className="cube-size-sep" aria-hidden="true" />
      <button
        type="button"
        data-testid="cube-size-3"
        className={`cube-size-opt ${value === 3 ? "on" : ""}`}
        onClick={() => onChange(3)}
        aria-pressed={value === 3}
      >
        3×3
      </button>
    </span>
  );
}
