import { useState } from "react";
import { validateFacelet } from "../state/validateFacelet";

type Props = {
  scrambleState: string;
  onSetState: (state: string) => void;
};

export default function StateField({ scrambleState, onSetState }: Props) {
  // Initialized once from prop; parent uses `key={scrambleState}` to remount
  // when the underlying state changes externally (Scramble click, etc.).
  const [text, setText] = useState<string>(scrambleState);
  const [error, setError] = useState<string | null>(null);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Clipboard may be unavailable (insecure context, missing permission);
      // intentionally swallow — the textarea still allows manual selection.
    }
  }

  function handleSet() {
    const result = validateFacelet(text);
    if (result.ok) {
      setError(null);
      onSetState(result.state);
    } else {
      setError(result.error);
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
      <textarea
        data-testid="state-field"
        rows={2}
        cols={54}
        value={text}
        onChange={(e) => setText(e.target.value)}
        spellCheck={false}
        style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}
      />
      <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
        <button data-testid="state-copy" onClick={handleCopy}>
          Copy
        </button>
        <button data-testid="state-set" onClick={handleSet}>
          Set state
        </button>
        {error !== null && (
          <span data-testid="state-error" style={{ color: "crimson" }}>
            {error}
          </span>
        )}
      </div>
    </div>
  );
}
