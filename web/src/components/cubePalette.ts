// Single source of truth for the chrome palette + face offsets consumed
// by FlatCubeRenderer (the sole consumer; 3D contexts use cubing.js's
// stock palette, which has no public override API). Keep the renderer
// wired through here so the palette is a one-change tweak.

// Chrome palette — designer-tuned, slightly muted vs speedcube classic.
export const COLOR_FOR_LETTER: Record<string, string> = {
  U: "#f5f3ee",
  R: "#d24244",
  F: "#5cb27a",
  D: "#f0c64a",
  L: "#e08a4a",
  B: "#3a6ee0",
};

// Indices into a 54-char facelet string (URFDLB order).
export const FACE_OFFSETS: Record<string, number> = {
  U: 0,
  R: 9,
  F: 18,
  D: 27,
  L: 36,
  B: 45,
};

// Ink stroke for sticker outlines.
export const INK_STROKE = "#131210";
