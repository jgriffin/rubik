import { useRef, useState, type ClipboardEvent, type KeyboardEvent } from "react";
import { validateFacelet } from "../state/validateFacelet";

// Faces in DISPLAY order (U L F R B D, like the reference) but each
// carries the URFDLB storage offset so input ↔ string mapping stays
// canonical regardless of visual position.
const FACES = [
  { letter: "U" as const, name: "top",    offset: 0  },
  { letter: "L" as const, name: "left",   offset: 36 },
  { letter: "F" as const, name: "front",  offset: 18 },
  { letter: "R" as const, name: "right",  offset: 9  },
  { letter: "B" as const, name: "back",   offset: 45 },
  { letter: "D" as const, name: "bottom", offset: 27 },
];

type Props = {
  state: string;
  onStateChange: (state: string) => void;
};

function sliceFaces(state: string): string[] {
  return FACES.map((f) => state.slice(f.offset, f.offset + 9));
}

function assembleURFDLB(drafts: string[]): string {
  const out = new Array<string>(54);
  FACES.forEach((f, i) => {
    const chunk = drafts[i].padEnd(9, " ");
    for (let j = 0; j < 9; j++) out[f.offset + j] = chunk[j];
  });
  return out.join("");
}

export default function StateGrid({ state, onStateChange }: Props) {
  const [drafts, setDrafts] = useState<string[]>(() => sliceFaces(state));
  const [lastExternalState, setLastExternalState] = useState(state);
  const inputRefs = useRef<Array<HTMLInputElement | null>>([]);

  // Render-time sync (React docs pattern). Re-derive drafts when an external
  // change (Scramble / Clear) lands. Mid-typing self-commits assemble back to
  // the same state, so this branch is a no-op in that case and focus survives.
  if (state !== lastExternalState) {
    setLastExternalState(state);
    setDrafts(sliceFaces(state));
  }

  function commit(newDrafts: string[]) {
    setDrafts(newDrafts);
    const assembled = assembleURFDLB(newDrafts);
    if (validateFacelet(assembled).ok) {
      onStateChange(assembled);
    }
  }

  function handleInput(displayIdx: number, raw: string) {
    const cleaned = raw.replace(/[^A-Za-z]/g, "").toUpperCase().slice(0, 9);
    const next = [...drafts];
    next[displayIdx] = cleaned;
    commit(next);
    if (cleaned.length === 9 && displayIdx < FACES.length - 1) {
      inputRefs.current[displayIdx + 1]?.focus();
    }
  }

  function handlePaste(displayIdx: number, e: ClipboardEvent<HTMLInputElement>) {
    const text = e.clipboardData.getData("text").replace(/[^A-Za-z]/g, "").toUpperCase();
    if (text.length === 54) {
      // Whole-state URFDLB paste — distribute by storage offset, not display order.
      // External solvers emit URFDLB; honoring that ordering means a paste at
      // any input correctly fills every face.
      e.preventDefault();
      commit(FACES.map((f) => text.slice(f.offset, f.offset + 9)));
      inputRefs.current[displayIdx]?.blur();
      return;
    }
    if (text.length >= 18) {
      // Cascading paste in display order — preserves the reference UX for
      // partial multi-face fills (e.g. paste 27 chars starting at F).
      e.preventDefault();
      const next = [...drafts];
      let pos = 0;
      for (let i = displayIdx; i < FACES.length && pos < text.length; i++) {
        next[i] = text.slice(pos, pos + 9);
        pos += 9;
      }
      commit(next);
      return;
    }
    // Shorter pastes fall through to default browser handling, then handleInput.
  }

  function handleKeyDown(displayIdx: number, e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Backspace" && drafts[displayIdx] === "" && displayIdx > 0) {
      e.preventDefault();
      const prev = inputRefs.current[displayIdx - 1];
      if (prev) {
        prev.focus();
        prev.setSelectionRange(prev.value.length, prev.value.length);
      }
    }
  }

  return (
    <div className="state-line" data-testid="state-grid">
      {FACES.map((f, i) => (
        <div key={f.letter} className="state-chunk">
          <input
            ref={(el) => {
              inputRefs.current[i] = el;
            }}
            data-testid={`state-input-${f.letter}`}
            type="text"
            maxLength={9}
            value={drafts[i]}
            onChange={(e) => handleInput(i, e.target.value)}
            onPaste={(e) => handlePaste(i, e)}
            onKeyDown={(e) => handleKeyDown(i, e)}
            spellCheck={false}
            autoComplete="off"
          />
          <div className="face-tag">
            <em className="serif">{f.letter}</em>
            {f.name}
          </div>
        </div>
      ))}
    </div>
  );
}
