type Props = {
  onScramble: (length: number) => void;
  disabled?: boolean;
};

export default function ScrambleButton({ onScramble, disabled }: Props) {
  return (
    <button
      data-testid="scramble-button"
      onClick={() => onScramble(14)}
      disabled={disabled}
    >
      Scramble (length 14)
    </button>
  );
}
