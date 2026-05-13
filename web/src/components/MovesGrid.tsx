// MovesGrid — editable per-cell input grid for the moves-to-apply
// sequence. Each cell holds one move (1- or 2-char QTM token); cells
// auto-advance on token commit; paste spreads a parsed sequence across
// cells; backspace at an empty cell rewinds focus to the previous cell.
//
// Visible cells = moves.length + 1 trailing empty cell ("next move
// goes here"). Cells are controlled inputs bound to the App-level
// `moves` slice — typing IS editing the source of truth.
//
// Snap-back on invalid input: a small bump counter forces a re-render
// when a keystroke is rejected (e.g. typing "X" into an empty cell),
// so the controlled `value` attribute can reset the DOM back to its
// committed state.

import { useEffect, useRef, useState } from "react";
import type { MoveStr } from "../state/faceletMoves";
import { parseMovesPartial } from "../state/parseMoves";

type Props = {
  moves: MoveStr[];
  onMovesChange: (moves: MoveStr[]) => void;
  activeIdx?: number;
  onActiveChange?: (idx: number) => void;
  rowSize?: number;
  disabled?: boolean;
};

const ROW_TAGS = ["i.", "ii.", "iii.", "iv."];
const FACE_LETTERS = new Set(["R", "L", "U", "D", "F", "B"]);
const VALID_TOKENS = new Set([
  "R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'",
]);

export default function MovesGrid({
  moves,
  onMovesChange,
  activeIdx,
  onActiveChange,
  rowSize = 10,
  disabled,
}: Props) {
  const inputRefs = useRef<Array<HTMLInputElement | null>>([]);
  const pendingFocusRef = useRef<number | null>(null);
  const [, setSnapBumper] = useState(0);

  // Visible cells = moves + 1 trailing empty.
  const cells: string[] = [...moves, ""];

  // Focus the pending cell after a commit that requested it.
  useEffect(() => {
    if (pendingFocusRef.current !== null) {
      const idx = pendingFocusRef.current;
      pendingFocusRef.current = null;
      const el = inputRefs.current[idx];
      if (el) {
        el.focus();
        const len = el.value.length;
        el.setSelectionRange(len, len);
      }
    }
  });

  function commit(newMoves: MoveStr[], focusIdx: number | null) {
    pendingFocusRef.current = focusIdx;
    onMovesChange(newMoves);
  }

  function snapBack() {
    setSnapBumper((v) => v + 1);
  }

  function handleChange(i: number, raw: string) {
    const val = raw.toUpperCase();

    if (val.length === 0) {
      // Cleared. Splice this slot out of moves (trailing cell is a no-op).
      if (i < moves.length) {
        const next = [...moves.slice(0, i), ...moves.slice(i + 1)];
        commit(next, i);
      }
      return;
    }

    if (val.length === 1) {
      if (FACE_LETTERS.has(val)) {
        const next = [...moves];
        if (i < moves.length) next[i] = val as MoveStr;
        else next.push(val as MoveStr);
        commit(next, i); // stay — user may type "'"
      } else {
        snapBack(); // invalid 1-char (e.g. "X")
      }
      return;
    }

    if (val.length === 2) {
      if (VALID_TOKENS.has(val)) {
        const next = [...moves];
        if (i < moves.length) next[i] = val as MoveStr;
        else next.push(val as MoveStr);
        commit(next, i + 1); // advance
        return;
      }
      // Two face letters in one cell → split into i and i+1.
      if (FACE_LETTERS.has(val[0]) && FACE_LETTERS.has(val[1])) {
        const next = [...moves];
        if (i < moves.length) {
          next[i] = val[0] as MoveStr;
          next.splice(i + 1, 0, val[1] as MoveStr);
        } else {
          next.push(val[0] as MoveStr, val[1] as MoveStr);
        }
        commit(next, i + 1);
        return;
      }
      snapBack(); // invalid 2-char (e.g. "X'", "RX")
    }
  }

  function handleKeyDown(i: number, e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Backspace" && e.currentTarget.value === "" && i > 0) {
      e.preventDefault();
      const prev = inputRefs.current[i - 1];
      if (prev) {
        prev.focus();
        const len = prev.value.length;
        prev.setSelectionRange(len, len);
      }
      return;
    }
    if (e.key === " ") {
      // Space advances focus when the current cell holds a valid
      // 1-char token (the "I'm done with this move, next one" cue).
      const v = e.currentTarget.value.toUpperCase();
      if (FACE_LETTERS.has(v) && i < cells.length - 1) {
        e.preventDefault();
        const nextCell = inputRefs.current[i + 1];
        if (nextCell) nextCell.focus();
      } else if (v === "") {
        // Space in an empty cell is a no-op (don't insert literal space).
        e.preventDefault();
      }
      return;
    }
    if (e.key === "ArrowLeft" && e.currentTarget.selectionStart === 0 && i > 0) {
      e.preventDefault();
      const prev = inputRefs.current[i - 1];
      if (prev) {
        prev.focus();
        const len = prev.value.length;
        prev.setSelectionRange(len, len);
      }
      return;
    }
    if (
      e.key === "ArrowRight" &&
      e.currentTarget.selectionStart === e.currentTarget.value.length &&
      i < cells.length - 1
    ) {
      e.preventDefault();
      const next = inputRefs.current[i + 1];
      if (next) {
        next.focus();
        next.setSelectionRange(0, 0);
      }
    }
  }

  function handlePaste(i: number, e: React.ClipboardEvent<HTMLInputElement>) {
    const text = e.clipboardData.getData("text");
    if (!text || text.trim().length === 0) return;
    e.preventDefault();
    const { moves: pasted } = parseMovesPartial(text);
    if (pasted.length === 0) {
      snapBack();
      return;
    }
    // Pasting at the trailing empty cell appends; pasting at any
    // existing cell overwrites that cell and extends from there,
    // preserving any tail beyond the pasted range.
    const next: MoveStr[] =
      i >= moves.length
        ? [...moves, ...pasted]
        : [...moves.slice(0, i), ...pasted, ...moves.slice(i + pasted.length)];
    commit(next, i + pasted.length);
  }

  function handleFocus(i: number) {
    onActiveChange?.(i);
  }

  const totalRows = Math.max(1, Math.ceil(cells.length / rowSize));
  const rows = [];
  for (let r = 0; r < totalRows; r++) {
    const cellEls = [];
    for (let c = 0; c < rowSize; c++) {
      const idx = r * rowSize + c;
      if (idx >= cells.length) {
        // Visual placeholders so the row stays full-width.
        cellEls.push(
          <div
            key={c}
            className="move-cell empty unused"
            data-testid="move-cell-unused"
            aria-hidden="true"
          />,
        );
        continue;
      }
      const val = cells[idx];
      const isTrailing = idx === moves.length;
      const isActive = activeIdx === idx;
      const className = [
        "move-cell",
        "move-cell-input",
        val ? "" : "empty",
        isActive ? "active" : "",
      ]
        .filter(Boolean)
        .join(" ");
      cellEls.push(
        <input
          key={c}
          ref={(el) => {
            inputRefs.current[idx] = el;
          }}
          className={className}
          type="text"
          data-testid={isTrailing ? "move-cell-empty" : "move-cell"}
          data-cell-idx={idx}
          value={val}
          maxLength={2}
          onChange={(e) => handleChange(idx, e.target.value)}
          onKeyDown={(e) => handleKeyDown(idx, e)}
          onPaste={(e) => handlePaste(idx, e)}
          onFocus={() => handleFocus(idx)}
          spellCheck={false}
          autoComplete="off"
          disabled={disabled}
          aria-label={`move ${idx + 1}`}
        />,
      );
    }
    rows.push(
      <div key={r} className="moves-row">
        {cellEls}
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
