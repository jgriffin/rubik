import { useMemo } from "react";
import FlatCubeRenderer from "./FlatCubeRenderer";
import { applyMoves } from "../state/applyMove";
import type { MoveStr } from "../state/faceletMoves";

type Props = {
  scrambleState: string;
  // null when not solved yet — strip still renders one Start cell.
  solution: string[] | null;
  stepIdx: number;
  onJumpTo: (idx: number) => void;
};

const CUBE_SIZE_PX = 80;

export default function MoveStripView({
  scrambleState,
  solution,
  stepIdx,
  onJumpTo,
}: Props) {
  // Memoize per-cell facelet states so the whole grid only recomputes
  // when scrambleState or solution change — not on every stepIdx tick.
  const states = useMemo(() => {
    const out: string[] = [scrambleState];
    if (solution) {
      let s = scrambleState;
      for (const m of solution) {
        s = applyMoves(s, [m as MoveStr]);
        out.push(s);
      }
    }
    return out;
  }, [scrambleState, solution]);

  return (
    <div
      data-testid="strip-view"
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(6, max-content)",
        gap: "0.75rem",
        margin: "1rem 0",
      }}
    >
      {states.map((facelet, i) => {
        const moveLabel = i === 0 ? "Start" : (solution?.[i - 1] ?? "?");
        const isActive = i === stepIdx;
        return (
          <button
            key={i}
            data-testid={`strip-cell-${i}`}
            data-active={isActive ? "true" : "false"}
            onClick={() => onJumpTo(i)}
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "0.3rem",
              padding: "0.4rem",
              border: isActive ? "2px solid #4299ff" : "2px solid transparent",
              borderRadius: "6px",
              background: "transparent",
              cursor: "pointer",
              fontFamily: "inherit",
              color: "inherit",
            }}
          >
            <span
              style={{
                fontSize: "0.85rem",
                fontWeight: isActive ? 600 : 500,
                fontVariantNumeric: "tabular-nums",
                minHeight: "1.2em",
              }}
            >
              {moveLabel}
            </span>
            <FlatCubeRenderer
              facelet={facelet}
              sizePx={CUBE_SIZE_PX}
              testId={null}
            />
            <span
              style={{
                fontSize: "0.7rem",
                opacity: 0.55,
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {i}
            </span>
          </button>
        );
      })}
    </div>
  );
}
