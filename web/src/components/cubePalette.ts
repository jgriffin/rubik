// Active cube palette + face geometry consumed by Cube2D.
// COLOR_FOR_LETTER sources from the active named preset in palettes.ts;
// the future palette-picker will flip this re-export by user selection.

import { wcaPalette } from "./palettes";

// Active palette — matches cubing.js stock so 2D and 3D cubes are
// color-identical. Future picker swaps this to chromePalette /
// any other preset on demand.
export const COLOR_FOR_LETTER = wcaPalette.colorForLetter;

// Indices into a 54-char facelet string (URFDLB order). Palette-agnostic.
export const FACE_OFFSETS: Record<string, number> = {
  U: 0,
  R: 9,
  F: 18,
  D: 27,
  L: 36,
  B: 45,
};

// Ink stroke for sticker outlines. Palette-agnostic — the dark ink
// reads well over wca, chrome, and any plausible future preset.
export const INK_STROKE = "#131210";
