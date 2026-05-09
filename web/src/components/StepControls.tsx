import { useEffect, useRef, useState } from "react";

type Props = {
  stepIdx: number;
  totalSteps: number;
  onStepChange: (idx: number) => void;
  disabled?: boolean;
};

export default function StepControls({
  stepIdx,
  totalSteps,
  onStepChange,
  disabled,
}: Props) {
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (!playing || stepIdx >= totalSteps) return;
    timerRef.current = window.setInterval(() => {
      const next = Math.min(stepIdx + 1, totalSteps);
      onStepChange(next);
      if (next >= totalSteps) setPlaying(false);
    }, 500); // 2 moves/sec
    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [playing, stepIdx, totalSteps, onStepChange]);

  if (totalSteps === 0) return null;

  return (
    <div
      data-testid="step-controls"
      style={{
        display: "flex",
        gap: "0.5rem",
        alignItems: "center",
        margin: "0.5rem 0",
      }}
    >
      <button
        data-testid="reset-button"
        onClick={() => onStepChange(0)}
        disabled={disabled || stepIdx === 0}
      >
        ⟲ reset
      </button>
      <button
        data-testid="prev-button"
        onClick={() => onStepChange(Math.max(0, stepIdx - 1))}
        disabled={disabled || stepIdx === 0}
      >
        ‹ prev
      </button>
      <button
        data-testid="play-button"
        onClick={() => setPlaying((p) => !p)}
        disabled={disabled || stepIdx >= totalSteps}
      >
        {playing ? "⏸ pause" : "▶ play"}
      </button>
      <button
        data-testid="next-button"
        onClick={() => onStepChange(Math.min(totalSteps, stepIdx + 1))}
        disabled={disabled || stepIdx >= totalSteps}
      >
        next ›
      </button>
      <span
        data-testid="step-counter"
        style={{ fontVariantNumeric: "tabular-nums", opacity: 0.7 }}
      >
        {stepIdx} / {totalSteps}
      </span>
    </div>
  );
}
