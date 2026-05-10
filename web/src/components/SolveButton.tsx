type Props = {
  onSolve: () => void;
  disabled?: boolean;
  isSolving: boolean;
};

export default function SolveButton({ onSolve, disabled, isSolving }: Props) {
  return (
    <button
      type="button"
      className="btn-pill solid"
      data-testid="solve-button"
      onClick={onSolve}
      disabled={disabled || isSolving}
    >
      {isSolving ? "solving…" : "solve"}
    </button>
  );
}
