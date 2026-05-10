import type { MoveStr } from "../state/faceletMoves";

type Props = {
  moves: MoveStr[];
  rowSize?: number;
};

const ROW_TAGS = ["i.", "ii.", "iii.", "iv."];

function MoveCell({ move }: { move: string }) {
  const main = move[0];
  const mod = move.slice(1);
  const modGlyph = mod === "'" ? "′" : mod;
  return (
    <div className="move-cell serif" data-testid="move-cell">
      {main}
      {mod && <em>{modGlyph}</em>}
    </div>
  );
}

function EmptyCell() {
  return (
    <div className="move-cell empty" data-testid="move-cell-empty">
      ·
    </div>
  );
}

export default function MovesGrid({ moves, rowSize = 10 }: Props) {
  // Always show at least one row so section ii holds its rhythm even
  // when the user pasted a state without scrambling.
  const totalRows = Math.max(1, Math.ceil(moves.length / rowSize));
  const rows = [];
  for (let r = 0; r < totalRows; r++) {
    const cells = [];
    for (let i = 0; i < rowSize; i++) {
      const idx = r * rowSize + i;
      const m = moves[idx];
      cells.push(m ? <MoveCell key={i} move={m} /> : <EmptyCell key={i} />);
    }
    rows.push(
      <div key={r} className="moves-row">
        {cells}
        {totalRows > 1 && (
          <span className="row-tag serif">{ROW_TAGS[r] ?? ""}</span>
        )}
      </div>,
    );
  }
  return (
    <div className="moves-grid" data-testid="moves-grid">
      {rows}
    </div>
  );
}
