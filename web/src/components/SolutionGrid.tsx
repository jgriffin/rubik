import { useMemo, type CSSProperties } from "react";
import CubeStage from "./CubeStage";
import SolutionCard from "./SolutionCard";
import { applyMoves } from "../state/applyMove";
import type { MoveStr } from "../state/faceletMoves";

export type Cols = 1 | 2 | 3 | 4 | 5 | 6;
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
  isCubeSolved?: boolean;
};

// Per-column cube preview pixel sizes (column-grid mode only — cube mode
// uses its own size set inline below). 200 / 130 / 130 / 110 / 90.
const CUBE_SIZE_BY_COLS: Record<Exclude<Cols, 1>, number> = {
  2: 200,
  3: 130,
  4: 130,
  5: 110,
  6: 90,
};

// Halved sizes for dual (split) mode — two renderers fit per card.
const DUAL_SIZE_BY_COLS: Record<Exclude<Cols, 1>, number> = {
  2: 160,
  3: 100,
  4: 90,
  5: 75,
  6: 60,
};

// Cube-mode pixel sizes — one big card replacing the grid. Larger than
// any column-grid size since the card stands alone.
const CUBE_MODE_SIZE = 380;
const CUBE_MODE_DUAL_SIZE = 300;

export default function SolutionGrid({
  scrambleState,
  scrambleMoves,
  moves,
  isSolving,
  cols,
  renderMode,
  activeIdx,
  onActiveChange,
  isCubeSolved,
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

  // Cube mode (cols=1): single big card driven by activeIdx. Per-step
  // 2D animation on forward activeIdx jumps lives in `CubeStage`.
  // C·P4 will add the matching 3D animation.
  if (cols === 1) {
    const sizePx = renderMode === "dual" ? CUBE_MODE_DUAL_SIZE : CUBE_MODE_SIZE;
    return (
      <div className="sol-cube-stage" data-testid="solution-grid">
        <CubeStage
          states={states}
          moves={moves}
          activeIdx={activeIdx}
          onActiveChange={onActiveChange}
          scrambleAlg={scrambleAlg}
          renderMode={renderMode}
          sizePx={sizePx}
        />
        {isCubeSolved && (
          <span
            className="sol-cell-solved-label"
            data-testid="steps-solved-label"
          >
            solved
          </span>
        )}
        {isSolving && (
          <div className="sol-loading" data-testid="solution-loading">
            solving…
          </div>
        )}
      </div>
    );
  }

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
      {isCubeSolved && (
        <span
          className="sol-cell-solved-label"
          data-testid="steps-solved-label"
        >
          solved
        </span>
      )}
      {isSolving && (
        <div className="sol-loading" data-testid="solution-loading">
          solving…
        </div>
      )}
    </div>
  );
}
