// Imperative React wrapper around cubing.js's `<twisty-player>` custom element.
//
// We use the imperative path (`new TwistyPlayer(config)` inside `useEffect`)
// rather than JSX so we can pass non-attribute config and clean up cleanly on
// unmount via `player.remove()`. Two modes:
//
//   - "static"   : tempo-scale=0, control-panel=none, pointer-events: none.
//                  Renders the post-(setup+alg) state as a non-interactive
//                  3D thumbnail. Used by section iii cards (M9.2 P1+).
//   - "animated" : default control-panel=bottom-row, normal tempo. Used by
//                  section iv "watch the solve" (M9.2 P3).
//
// The static-mode recipe (`experimentalSetupAlg=<scramble>` +
// `experimentalSetupAnchor="end"` + `alg=<solution[:N]>`) displays exactly
// `applyMove(scramble, solution[:N])` — the same ground truth Cube2D
// (static mode) emits. See .planning/research/m9-twisty-player.md §2-3.
//
// Pitfall (research §2): tempo-scale=0 may render the pre-alg frame for one
// tick before settling. If the static HTML preview confirms the flicker,
// add an opacity-on-mount dance here in P1+.

import { useEffect, useRef } from "react";
import { TwistyPlayer } from "cubing/twisty";
import {
  buildTwistyPlayerConfig,
  type TwistyPlayerMode,
} from "./twistyPlayerConfig";

type Props = {
  scrambleAlg: string;
  solutionAlg: string;
  sizePx: number;
  mode: TwistyPlayerMode;
  testId?: string | null;
};

export default function TwistyPlayerWrapper({
  scrambleAlg,
  solutionAlg,
  sizePx,
  mode,
  testId,
}: Props) {
  const hostRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const player = new TwistyPlayer(
      buildTwistyPlayerConfig({ scrambleAlg, solutionAlg, mode }),
    );
    player.style.width = `${sizePx}px`;
    player.style.height = `${sizePx}px`;
    host.appendChild(player);

    return () => {
      player.remove();
    };
  }, [scrambleAlg, solutionAlg, sizePx, mode]);

  // Resolve testid: default "twisty-cube"; explicit null suppresses.
  const testIdProp =
    testId === null ? {} : { "data-testid": testId ?? "twisty-cube" };

  // Static mode disables drag-rotate via CSS (research §2: no
  // `interactive="false"` attribute exists; pointer-events is the supported
  // workaround).
  const style: React.CSSProperties =
    mode === "static" ? { pointerEvents: "none" } : {};

  return <div ref={hostRef} style={style} {...testIdProp} />;
}
