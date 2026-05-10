import { useState } from "react";
import {
  parseMoves,
  formatMoves,
  ParseMovesError,
} from "../state/parseMoves";
import type { MoveStr } from "../state/faceletMoves";

type Props = {
  solution: string[] | null;
  onSetMoves: (moves: MoveStr[]) => void;
};

export default function MovesField({ solution, onSetMoves }: Props) {
  // Initialized once from prop; parent uses `key={...}` to remount when the
  // underlying solution changes externally (Solve click, Scramble reset).
  const [text, setText] = useState<string>(
    formatMoves((solution ?? []) as MoveStr[]),
  );
  const [error, setError] = useState<string | null>(null);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Clipboard may be unavailable; intentionally swallow.
    }
  }

  function handleSet() {
    try {
      const moves = parseMoves(text);
      setError(null);
      onSetMoves(moves);
    } catch (e) {
      if (e instanceof ParseMovesError) {
        setError(`Invalid move '${e.token}' at position ${e.index}`);
        return;
      }
      throw e;
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
      <textarea
        data-testid="moves-field"
        rows={3}
        cols={48}
        value={text}
        onChange={(e) => setText(e.target.value)}
        spellCheck={false}
        style={{
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          whiteSpace: "pre-wrap",
        }}
      />
      <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
        <button data-testid="moves-copy" onClick={handleCopy}>
          Copy
        </button>
        <button data-testid="moves-set" onClick={handleSet}>
          Set moves
        </button>
        {error !== null && (
          <span data-testid="moves-error" style={{ color: "crimson" }}>
            {error}
          </span>
        )}
      </div>
    </div>
  );
}
