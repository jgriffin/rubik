type Props = {
  solved: boolean | null;
  moveCount: number;
  metaText: string | null;
};

export default function SolvedFooter({ solved, moveCount, metaText }: Props) {
  if (solved !== true) return null;
  return (
    <div className="summary" data-testid="solved-footer">
      <span className="big serif">
        <em>Solved.</em>
      </span>
      <span className="summary-prose">
        {moveCount} {moveCount === 1 ? "move" : "moves"} applied to the starting state above.
      </span>
      {metaText && <span className="meta">{metaText}</span>}
    </div>
  );
}
