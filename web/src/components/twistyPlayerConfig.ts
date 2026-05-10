// Pure config-builder for cubing.js's TwistyPlayer.
//
// Lives next to TwistyPlayerWrapper.tsx so the two evolve together, but in a
// separate file because react-refresh's `only-export-components` rule flags
// non-component exports from .tsx component files. Exported here so the
// vitest contract surface is "the config we hand to `new TwistyPlayer(...)`"
// without having to mock cubing.js or instantiate the wrapper under jsdom.
//
// See .planning/research/m9-twisty-player.md §2-3 for the static-mode recipe.

import type { TwistyPlayerConfig } from "cubing/twisty";

export type TwistyPlayerMode = "static" | "animated";

export function buildTwistyPlayerConfig({
  scrambleAlg,
  solutionAlg,
  mode,
}: {
  scrambleAlg: string;
  solutionAlg: string;
  mode: TwistyPlayerMode;
}): TwistyPlayerConfig {
  return {
    puzzle: "3x3x3",
    experimentalSetupAlg: scrambleAlg,
    experimentalSetupAnchor: "end",
    alg: solutionAlg,
    controlPanel: mode === "static" ? "none" : "bottom-row",
    background: "none",
    backView: "none",
    tempoScale: mode === "static" ? 0 : 1,
    hintFacelets: "none",
  };
}
