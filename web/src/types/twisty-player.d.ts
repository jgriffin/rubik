// JSX runtime augmentation for cubing.js's `<twisty-player>` custom element.
//
// Belt-and-braces: TwistyPlayerWrapper.tsx uses the *imperative* path
// (`new TwistyPlayer(config)` + `appendChild`), so this augmentation is not
// required for the wrapper itself. It exists so any ad-hoc `<twisty-player>`
// JSX usage elsewhere in the app (e.g. one-off animated player) is typed.
//
// React 19 reads JSX intrinsics from `react/jsx-runtime` (the new transform),
// not the legacy global `JSX` namespace. See:
// https://til.jakelazaroff.com/typescript/add-custom-element-to-jsx-intrinsic-elements/

import type { DetailedHTMLProps, HTMLAttributes } from "react";
import type { TwistyPlayer } from "cubing/twisty";

interface TwistyPlayerAttrs {
  alg?: string;
  "experimental-setup-alg"?: string;
  "experimental-setup-anchor"?: "start" | "end";
  puzzle?: "3x3x3" | "2x2x2" | string;
  visualization?: "auto" | "2D" | "3D" | "Cube3D" | "PG3D";
  "control-panel"?: "auto" | "bottom-row" | "none";
  "back-view"?: "auto" | "side-by-side" | "top-right" | "none";
  background?: "auto" | "checkered" | "checkered-transparent" | "none";
  "hint-facelets"?: "auto" | "none" | "floating";
  "tempo-scale"?: number | string;
  "camera-latitude"?: number | string;
  "camera-longitude"?: number | string;
  "camera-distance"?: number | string;
}

declare module "react/jsx-runtime" {
  namespace JSX {
    interface IntrinsicElements {
      "twisty-player": DetailedHTMLProps<
        HTMLAttributes<TwistyPlayer> & TwistyPlayerAttrs,
        TwistyPlayer
      >;
    }
  }
}
