import { useMemo } from "react";
import FlatCubeRenderer, { type Overlay } from "./FlatCubeRenderer";
import { applyMoves } from "../state/applyMove";
import type { MoveStr } from "../state/faceletMoves";

type Props = {
  scrambleState: string;
  // null when not solved yet — strip still renders one Start cell.
  solution: string[] | null;
  stepIdx: number;
  onJumpTo: (idx: number) => void;
  // Pixel height of one cube. Set by App's size selector
  // (Small=80, Medium=120, Large=160).
  cubeSize: number;
};

const COLS = 5;
const ACCENT = "#4299ff";

// V2 layout overlays drawn INSIDE the cube SVG:
//   - step#: bottom-left corner, ½-cubie padding from bottom + left.
//   - move-label: bottom-LEFT of text aligned with
//       (x = 10·stickerPx, y = 2·stickerPx) — bottom of row 1, left edge
//       of 2nd-from-rightmost cubie column (i.e. the empty zone above R).
//   - cell 0 ("Start") gets no move-label, only the step# "0".
function v2Overlays(
  stepNum: number,
  moveLabel: string | null,
  sizePx: number
): Overlay[] {
  const stickerPx = sizePx / 9;
  const stepFont = stickerPx * 0.85;
  const moveFont = stickerPx * 1.2;
  const overlays: Overlay[] = [
    {
      x: 0.5 * stickerPx,
      y: 9 * stickerPx - 0.5 * stickerPx - stepFont / 2,
      text: String(stepNum),
      fontSize: stepFont,
      weight: 500,
      opacity: 0.55,
      anchor: "start",
      // baseline defaults to "central"
    },
  ];
  if (moveLabel !== null) {
    overlays.push({
      x: 10 * stickerPx,
      y: 2 * stickerPx,
      text: moveLabel,
      fontSize: moveFont,
      weight: 600,
      opacity: 1,
      anchor: "start",
      baseline: "text-after-edge",
    });
  }
  return overlays;
}

export default function MoveStripView({
  scrambleState,
  solution,
  stepIdx,
  onJumpTo,
  cubeSize,
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
        gridTemplateColumns: `repeat(${COLS}, max-content)`,
        gap: "0.75rem",
        margin: "1rem 0",
      }}
    >
      {states.map((facelet, i) => {
        const moveLabel = i === 0 ? null : (solution?.[i - 1] ?? null);
        const isActive = i === stepIdx;
        return (
          <button
            key={i}
            data-testid={`strip-cell-${i}`}
            data-active={isActive ? "true" : "false"}
            onClick={() => onJumpTo(i)}
            style={{
              padding: "0.4rem",
              border: `2px solid ${isActive ? ACCENT : "transparent"}`,
              borderRadius: "6px",
              background: "transparent",
              cursor: "pointer",
              fontFamily: "inherit",
              color: "inherit",
              lineHeight: 0,
            }}
          >
            <FlatCubeRenderer
              facelet={facelet}
              sizePx={cubeSize}
              testId={null}
              overlays={v2Overlays(i, moveLabel, cubeSize)}
            />
          </button>
        );
      })}
    </div>
  );
}
