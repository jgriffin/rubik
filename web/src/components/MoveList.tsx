type Props = {
  moves: string[] | null;
  solved: boolean | null;
};

export default function MoveList({ moves, solved }: Props) {
  if (moves === null) return null;
  if (moves.length === 0) {
    return (
      <div data-testid="move-list" data-solved={String(solved)}>
        <p data-testid="solve-status">
          {solved ? "Already solved." : "No solution found within budget."}
        </p>
      </div>
    );
  }
  return (
    <div data-testid="move-list" data-solved={String(solved)}>
      <p data-testid="solve-status">
        {solved
          ? `Solved in ${moves.length} moves:`
          : `Path of ${moves.length} moves (not solved):`}
      </p>
      <ol>
        {moves.map((m, i) => (
          <li key={i} data-testid="move-item">
            {m}
          </li>
        ))}
      </ol>
    </div>
  );
}
