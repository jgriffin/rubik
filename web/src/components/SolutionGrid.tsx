import { useMemo, type CSSProperties } from "react";
import SolutionCard from "./SolutionCard";
import { applyMoves } from "../state/applyMove";
import type { MoveStr } from "../state/faceletMoves";

export type Cols = 1 | 2 | 3 | 4 | 6;
export type RenderMode = "net" | "iso" | "dual";

type Props = {
  scrambleState: string;
  scrambleMoves: MoveStr[];
  moves: MoveStr[];
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

// Halved sizes for dual (split) mode — two renderers fit per card.
const DUAL_SIZE_BY_COLS: Record<Cols, number> = {
  1: 200,
  2: 160,
  3: 100,
  4: 90,
  6: 60,
};

export default function SolutionGrid({
  scrambleState,
  scrambleMoves,
  moves,
  isSolving,
  cols,
  renderMode,
  activeIdx,
  onActiveChange,
}: Props) {
  // Joined scramble alg, used by twisty-player as the static-mode setup-alg.
  // Each card receives the full `moves` array and slices for its step,
  // so the prefix passed to twisty-player matches the per-card facelet
  // already computed below.
  const scrambleAlg = scrambleMoves.join(" ");

  // Memoize per-step facelet snapshots so changing activeIdx doesn't
  // re-walk the whole moves array.
  const states = useMemo(() => {
    const out: string[] = [scrambleState];
    let s = scrambleState;
    for (const m of moves) {
      s = applyMoves(s, [m]);
      out.push(s);
    }
    return out;
  }, [scrambleState, moves]);

  const sizePx =
    renderMode === "dual" ? DUAL_SIZE_BY_COLS[cols] : CUBE_SIZE_BY_COLS[cols];
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
        scrambleAlg={scrambleAlg}
        moves={moves}
        sizePx={sizePx}
        renderMode={renderMode}
        isStart
        isActive={activeIdx === 0}
        onClick={() => onActiveChange(0)}
      />
      {moves.map((m, i) => (
        <SolutionCard
          key={i + 1}
          stepNum={i + 1}
          moveLabel={m}
          facelet={states[i + 1]}
          preFacelet={states[i]}
          move={m}
          scrambleAlg={scrambleAlg}
          moves={moves}
          sizePx={sizePx}
          renderMode={renderMode}
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
