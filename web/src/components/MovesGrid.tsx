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

import React, { useEffect, useRef, useState } from "react";
import type { MoveStr } from "../state/faceletMoves";
import { parseMovesPartial } from "../state/parseMoves";

type Props = {
  moves: MoveStr[];
  onMovesChange: (moves: MoveStr[]) => void;
  activeIdx?: number;
  onActiveChange?: (idx: number) => void;
  canSolve?: boolean;
  isCubeSolved?: boolean;
  onSolve?: () => void;
  isSolving?: boolean;
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
  canSolve,
  isCubeSolved,
  onSolve,
  isSolving,
  rowSize = 10,
  disabled,
}: Props) {
  const inputRefs = useRef<Array<HTMLInputElement | null>>([]);
  const pendingFocusRef = useRef<number | null>(null);
  // Distinguishes programmatic focus (auto-advance after a commit;
  // caret-at-end) from user-initiated focus (Tab / click; select-all so
  // Backspace deletes the whole token).
  const programmaticFocusRef = useRef(false);
  const [, setSnapBumper] = useState(0);

  // Visible input cells = moves + 1 trailing empty (the "next move
  // goes here" affordance). The trailing cell ALSO carries the
  // end-of-journey label at its bottom — "solve" (clickable, accent)
  // when the cube can be solved, "solved" (ink) when it already is.
  // Both are positioned at the bottom-center of the same dashed cell
  // so there's no separate floating block beside the trailing input.
  const cells: string[] = [...moves, ""];
  const totalCells = cells.length;

  // Focus the pending cell after a commit that requested it. Sets
  // `programmaticFocusRef` so the onFocus handler skips select-all and
  // positions the caret at end (auto-advance pattern).
  useEffect(() => {
    if (pendingFocusRef.current !== null) {
      const idx = pendingFocusRef.current;
      pendingFocusRef.current = null;
      const el = inputRefs.current[idx];
      if (el) {
        programmaticFocusRef.current = true;
        el.focus();
        const len = el.value.length;
        el.setSelectionRange(len, len);
        queueMicrotask(() => {
          programmaticFocusRef.current = false;
        });
      }
    }
  });

  function commit(newMoves: MoveStr[], focusIdx: number | null) {
    pendingFocusRef.current = focusIdx;
    onMovesChange(newMoves);
  }

  // Programmatic focus helper. `mode` controls where the caret/selection
  // lands so the same path serves auto-advance (caret-end), arrow-key
  // text-mode crossing (caret-start/end), and arrow-key cell-mode
  // crossing (select-all).
  function focusCellWithMode(
    idx: number,
    mode: "select" | "caret-start" | "caret-end",
  ) {
    const el = inputRefs.current[idx];
    if (!el) return;
    programmaticFocusRef.current = true;
    el.focus();
    if (mode === "select") el.select();
    else if (mode === "caret-start") el.setSelectionRange(0, 0);
    else el.setSelectionRange(el.value.length, el.value.length);
    queueMicrotask(() => {
      programmaticFocusRef.current = false;
    });
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
    const el = e.currentTarget;
    // "Cell-selected mode" = full content selected (or empty cell);
    // this is the state immediately after focus or Escape. "Text mode"
    // = collapsed selection / partial selection, after the user clicked
    // an already-focused cell to position the caret.
    const isCellMode =
      el.selectionStart === 0 && el.selectionEnd === el.value.length;

    if (e.key === "Escape") {
      // Back to cell-selected mode. preventDefault blocks browser
      // defaults (some browsers blur on Escape in inputs); rAF
      // defer keeps the select() from being clobbered by anything
      // racing it (e.g. a tail blur).
      e.preventDefault();
      requestAnimationFrame(() => {
        if (document.activeElement === el) {
          el.select();
        } else {
          // If we did get blurred, refocus and select.
          el.focus();
          el.select();
        }
      });
      return;
    }

    if (e.key === "Backspace" && el.value === "" && i > 0) {
      e.preventDefault();
      focusCellWithMode(i - 1, "select");
      return;
    }

    if (e.key === " ") {
      const v = el.value.toUpperCase();
      if (FACE_LETTERS.has(v) && i < cells.length - 1) {
        e.preventDefault();
        focusCellWithMode(i + 1, "select");
      } else if (v === "") {
        e.preventDefault();
      }
      return;
    }

    if (e.key === "ArrowLeft") {
      if (isCellMode && i > 0) {
        // Cell-mode navigation: jump to prev cell, stay in cell mode.
        e.preventDefault();
        focusCellWithMode(i - 1, "select");
        return;
      }
      if (!isCellMode && el.selectionStart === 0 && i > 0) {
        // Text-mode edge crossing: enter prev cell at the right edge
        // (caret-end), stay in text mode.
        e.preventDefault();
        focusCellWithMode(i - 1, "caret-end");
      }
      return;
    }

    if (e.key === "ArrowRight") {
      if (isCellMode && i < cells.length - 1) {
        e.preventDefault();
        focusCellWithMode(i + 1, "select");
        return;
      }
      if (
        !isCellMode &&
        el.selectionStart === el.value.length &&
        i < cells.length - 1
      ) {
        e.preventDefault();
        focusCellWithMode(i + 1, "caret-start");
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

  function handleFocus(i: number, target: HTMLInputElement) {
    onActiveChange?.(i);
    // User-initiated focus (Tab / click): select-all so Backspace
    // deletes the whole token. Programmatic auto-advance focus skips
    // this — see `focusCellWithMode`.
    //
    // Selection is deferred via rAF so it runs *after* the browser's
    // mousedown→mouseup→click chain finishes positioning the caret.
    // Without the defer, the post-click caret-positioning wipes our
    // selection on the first click. Clicking an already-focused cell
    // doesn't trigger focus, so no rAF runs — the user's caret stays
    // where they clicked (text mode entered).
    if (!programmaticFocusRef.current) {
      requestAnimationFrame(() => {
        // Only re-select if focus is still on this cell. If the user
        // tabbed away or pressed Escape-then-blur in the meantime, skip.
        if (document.activeElement === target) {
          target.select();
        }
      });
    }
  }

  const totalRows = Math.max(1, Math.ceil(totalCells / rowSize));
  const rows = [];
  for (let r = 0; r < totalRows; r++) {
    const cellEls = [];
    for (let c = 0; c < rowSize; c++) {
      const idx = r * rowSize + c;

      if (idx >= totalCells) {
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
      const inputClassName = [
        // Non-trailing cells carry the border on the input itself; the
        // trailing cell is wrapped by a div that owns the dashed
        // border (so the bottom solve/solved label sits inside).
        isTrailing ? "" : "move-cell",
        "move-cell-input",
        isTrailing ? "move-cell-input-trailing" : "",
        val ? "" : "empty",
        isActive && !isTrailing ? "active" : "",
      ]
        .filter(Boolean)
        .join(" ");
      const inputEl = (
        <input
          ref={(el) => {
            inputRefs.current[idx] = el;
          }}
          className={inputClassName}
          type="text"
          data-testid={isTrailing ? "move-cell-empty" : "move-cell"}
          data-cell-idx={idx}
          value={val}
          maxLength={2}
          onChange={(e) => handleChange(idx, e.target.value)}
          onKeyDown={(e) => handleKeyDown(idx, e)}
          onPaste={(e) => handlePaste(idx, e)}
          onFocus={(e) => handleFocus(idx, e.currentTarget)}
          spellCheck={false}
          autoComplete="off"
          disabled={disabled}
          aria-label={`move ${idx + 1}`}
        />
      );

      if (isTrailing) {
        cellEls.push(
          <div
            key={c}
            className={`move-cell move-cell-trailing-wrap empty ${isActive ? "active" : ""}`}
            data-testid="move-cell-trailing-wrap"
          >
            {inputEl}
            {canSolve ? (
              <button
                type="button"
                className="trailing-end-label end-solve"
                data-testid="solve-button"
                onClick={onSolve}
                disabled={isSolving}
                title="solve from current state"
              >
                {isSolving ? "…" : "solve"}
              </button>
            ) : isCubeSolved ? (
              <span
                className="trailing-end-label end-solved"
                data-testid="solved-label"
              >
                solved
              </span>
            ) : null}
          </div>,
        );
      } else {
        // Non-trailing cells: bare input (the .move-cell border is on
        // the input itself).
        cellEls.push(
          <React.Fragment key={c}>{inputEl}</React.Fragment>,
        );
      }
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
