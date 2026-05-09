type Props = {
  onSolve: () => void;
  disabled?: boolean;
  isSolving: boolean;
};

export default function SolveButton({ onSolve, disabled, isSolving }: Props) {
  return (
    <button
      data-testid="solve-button"
      onClick={onSolve}
      disabled={disabled || isSolving}
    >
      {isSolving ? "Solving..." : "Solve"}
    </button>
  );
}
