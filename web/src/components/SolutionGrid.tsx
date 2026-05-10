import { useMemo, type CSSProperties } from "react";
import SolutionCard from "./SolutionCard";
import { applyMoves } from "../state/applyMove";
import type { MoveStr } from "../state/faceletMoves";

export type Cols = 1 | 2 | 3 | 4 | 6;
export type RenderMode = "net" | "iso" | "dual";

type Props = {
  scrambleState: string;
  solution: string[] | null;
  isSolving: boolean;
  cols: Cols;
  renderMode: RenderMode;
  activeIdx: number;
  onActiveChange: (idx: number) => void;
};

// README's per-column cube preview pixel sizes (220 / 200 / 130 / 90).
const CUBE_SIZE_BY_COLS: Record<Cols, number> = {
  1: 220,
  2: 200,
  3: 130,
  4: 130,
  6: 90,
};

export default function SolutionGrid({
  scrambleState,
  solution,
  isSolving,
  cols,
  activeIdx,
  onActiveChange,
}: Props) {
  // Memoize per-step facelet snapshots so changing activeIdx doesn't
  // re-walk the whole solution.
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

  const sizePx = CUBE_SIZE_BY_COLS[cols];
  const gridStyle = { "--cols": cols } as CSSProperties;

  return (
    <div
      className={`sol-grid cols-${cols}`}
      data-testid="solution-grid"
      style={gridStyle}
    >
      <SolutionCard
        stepNum={0}
        moveLabel={null}
        facelet={states[0]}
        sizePx={sizePx}
        isStart
        isActive={activeIdx === 0}
        onClick={() => onActiveChange(0)}
      />
      {solution?.map((m, i) => (
        <SolutionCard
          key={i + 1}
          stepNum={i + 1}
          moveLabel={m}
          facelet={states[i + 1]}
          sizePx={sizePx}
          isStart={false}
          isActive={activeIdx === i + 1}
          onClick={() => onActiveChange(i + 1)}
        />
      ))}
      {isSolving && (
        <div className="sol-loading" data-testid="solution-loading">
          solving…
        </div>
      )}
    </div>
  );
}
