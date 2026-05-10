import { useState } from "react";

type Props = {
  onScramble: (length: number) => void;
  disabled?: boolean;
};

export default function ScrambleControls({ onScramble, disabled }: Props) {
  const [length, setLength] = useState<number>(14);

  const handleSurprise = () => {
    const rand = 1 + Math.floor(Math.random() * 30);
    setLength(rand);
    onScramble(rand);
  };

  return (
    <div
      data-testid="scramble-controls"
      style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}
    >
      <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
        <input
          data-testid="length-slider"
          type="range"
          min={1}
          max={30}
          step={1}
          value={length}
          onChange={(e) => setLength(Number(e.target.value))}
          disabled={disabled}
        />
        <span
          data-testid="length-readout"
          style={{ fontVariantNumeric: "tabular-nums", opacity: 0.7 }}
        >
          length: {length}
        </span>
      </label>
      <button
        data-testid="scramble-button"
        onClick={() => onScramble(length)}
        disabled={disabled}
      >
        Scramble
      </button>
      <button
        data-testid="surprise-button"
        onClick={handleSurprise}
        disabled={disabled}
      >
        Surprise Me
      </button>
    </div>
  );
}
