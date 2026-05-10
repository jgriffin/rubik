import FlatCubeRenderer from "./FlatCubeRenderer";
import IsoCubeRenderer from "./IsoCubeRenderer";
import type { RenderMode } from "./SolutionGrid";

type Props = {
  stepNum: number;
  moveLabel: string | null;
  facelet: string;
  sizePx: number;
  renderMode: RenderMode;
  isStart: boolean;
  isActive: boolean;
  onClick: () => void;
};

export default function SolutionCard({
  stepNum,
  moveLabel,
  facelet,
  sizePx,
  renderMode,
  isStart,
  isActive,
  onClick,
}: Props) {
  const main = moveLabel?.[0];
  const mod = moveLabel?.slice(1);
  const modGlyph = mod === "'" ? "′" : mod;

  const cls = ["sol-cell", isStart && "zero", isActive && "active"]
    .filter(Boolean)
    .join(" ");

  return (
    <button
      type="button"
      className={cls}
      data-testid={`sol-card-${stepNum}`}
      data-active={isActive ? "true" : "false"}
      onClick={onClick}
    >
      <div className="top">
        <span className="step-num">{String(stepNum).padStart(2, "0")}</span>
        {isStart ? (
          <span className="move-glyph zero serif">start</span>
        ) : (
          <span className="move-glyph serif">
            {main}
            {mod && <em>{modGlyph}</em>}
          </span>
        )}
      </div>
      <div className="render">
        <div className="net">
          {renderMode === "iso" ? (
            <IsoCubeRenderer facelet={facelet} sizePx={sizePx} testId={null} />
          ) : (
            // net mode + dual fallthrough (dual handled in Phase 3)
            <FlatCubeRenderer facelet={facelet} sizePx={sizePx} testId={null} />
          )}
        </div>
      </div>
    </button>
  );
}
