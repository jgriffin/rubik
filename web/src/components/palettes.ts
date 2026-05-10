// Named cube palettes. Sticker colors keyed by face letter (URFDLB).
// Consumed by FlatCubeRenderer through cubePalette.ts; future palette-
// picker work (ROADMAP backlog) will swap the active palette by name.

export type Palette = {
  name: string;
  colorForLetter: Record<string, string>;
};

// WCA stock — matches cubing.js's <twisty-player> default palette so
// 2D and 3D cubes render identical colors. The active palette today;
// fixed by cubing.js's lack of a public sticker-color override API
// (upstream issue #424). Single source of truth for the values.
export const wcaPalette: Palette = {
  name: "wca",
  colorForLetter: {
    U: "#ffffff",
    L: "#ff8000",
    F: "#44ee00",
    R: "#ff0000",
    B: "#2266ff",
    D: "#f4f400",
  },
};

// Chrome — designer-tuned warm cream / muted accents from the
// design-port-sequence block. Was the active palette through the iso
// renderer era; retired here when we matched 2D to cubing.js's stock.
// Preserved as a named preset so the future picker can offer it back.
export const chromePalette: Palette = {
  name: "chrome",
  colorForLetter: {
    U: "#f5f3ee",
    L: "#e08a4a",
    F: "#5cb27a",
    R: "#d24244",
    B: "#3a6ee0",
    D: "#f0c64a",
  },
};
