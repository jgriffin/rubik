// SolutionCard — one card in the per-step solution grid.
//
// Discriminated union on `isStart`:
//   - Start card (isStart=true)  : no animation; 2D slot uses
//     <Cube2D facelet={...} /> (static path).
//   - Non-start card             : extracts to <NonStartCard>, which
//     builds a one-move CubeSequence via useCubeSequence and feeds it
//     into <Cube2D sequence={...} /> in the 2D slots. The Twisty
//     (3D) path stays static — animated 3D arrives with Block B's
//     TwistyAnimatedCube wrapper.
//
// Why the sub-component split: hooks can't be called conditionally at
// the top level. NonStartCard is the hook-owning shell so that the
// start card never instantiates a sequence (which would also be
// invalid — no `move` to feed it).
//
// Animation triggers (non-start only):
//   - IntersectionObserver: first time the card scrolls into view,
//     call seq.play() (forward only) and disconnect. One-shot — we
//     don't re-animate on every scroll. From the sequence's idle state
//     (timestamp=0), play() is equivalent to replay() for the first
//     pass; we use play() so the IO trigger is unambiguous about
//     "begin forward playback" semantics.
//   - onClick: alongside the existing onActiveChange, call
//     seq.replayWithReverse() — when the sequence has already settled
//     (status=ended), this runs the rev5 choreography: short reverse
//     leg → pause beat → forward play. From any non-settled state it
//     collapses to the forward-only replay so mid-flight clicks still
//     feel snappy.
//
// References:
//   - hooks/useCubeSequence.ts — spec memoization is the caller's
//     responsibility; we use useMemo over [preFacelet, move].
//   - components/Cube2D.tsx — animated mode falls back to StaticInner
//     when currentMoveIndex === -1, so the SSR / pre-trigger render
//     emits the same data-pos / data-color attributes the e2e suite
//     keys off (verified in SolutionCard.test.tsx).

import { useEffect, useMemo, useRef } from "react";
import Cube2D from "./Cube2D";
import TwistyPlayerWrapper from "./TwistyPlayerWrapper";
import type { RenderMode } from "./SolutionGrid";
import { useCubeSequence } from "../hooks/useCubeSequence";
import type { CubeSequence } from "../state/cubeSequence";
import type { MoveStr } from "../state/faceletMoves";

type CommonProps = {
  stepNum: number;
  moveLabel: string | null;
  facelet: string;
  scrambleAlg: string;
  solution: string[] | null;
  sizePx: number;
  renderMode: RenderMode;
  isActive: boolean;
  onClick: () => void;
};

type StartProps = CommonProps & {
  isStart: true;
  preFacelet?: undefined;
  move?: undefined;
};

type NonStartProps = CommonProps & {
  isStart: false;
  preFacelet: string;
  move: MoveStr;
};

export type SolutionCardProps = StartProps | NonStartProps;

export default function SolutionCard(props: SolutionCardProps) {
  if (props.isStart) {
    return <StartCard {...props} />;
  }
  return <NonStartCard {...props} />;
}

// ---------------------------------------------------------------------
// Shared chrome
// ---------------------------------------------------------------------

function CardShell({
  stepNum,
  moveLabel,
  isStart,
  isActive,
  onClick,
  rootRef,
  children,
}: {
  stepNum: number;
  moveLabel: string | null;
  isStart: boolean;
  isActive: boolean;
  onClick: () => void;
  rootRef?: React.Ref<HTMLButtonElement>;
  children: React.ReactNode;
}) {
  const main = moveLabel?.[0];
  const mod = moveLabel?.slice(1);
  const modGlyph = mod === "'" ? "′" : mod;

  const cls = ["sol-cell", isStart && "zero", isActive && "active"]
    .filter(Boolean)
    .join(" ");

  return (
    <button
      ref={rootRef}
      type="button"
      className={cls}
      data-testid={`sol-card-${stepNum}`}
      data-active={isActive ? "true" : "false"}
      onClick={onClick}
    >
      <div className="top">
        <span className="step-num">{String(stepNum).padStart(2, "0")}</span>
        {isStart ? (
          <span className="move-glyph zero serif">start</span>
        ) : (
          <span className="move-glyph serif">
            {main}
            {mod && <em>{modGlyph}</em>}
          </span>
        )}
      </div>
      <div className="render">
        <div className="net">{children}</div>
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------
// Start card — static path only (no animation).
// ---------------------------------------------------------------------

function StartCard({
  stepNum,
  moveLabel,
  facelet,
  scrambleAlg,
  solution,
  sizePx,
  renderMode,
  isActive,
  onClick,
}: StartProps) {
  const slot =
    renderMode === "iso" ? (
      <TwistyPlayerWrapper
        mode="static"
        scrambleAlg={scrambleAlg}
        solutionAlg={(solution ?? []).slice(0, stepNum).join(" ")}
        sizePx={sizePx}
        testId="twisty-cube"
      />
    ) : renderMode === "dual" ? (
      <div className="render-pair">
        <Cube2D facelet={facelet} sizePx={sizePx} testId="flat-cube-pair" />
        <TwistyPlayerWrapper
          mode="static"
          scrambleAlg={scrambleAlg}
          solutionAlg={(solution ?? []).slice(0, stepNum).join(" ")}
          sizePx={sizePx}
          testId="twisty-cube-pair"
        />
      </div>
    ) : (
      <Cube2D facelet={facelet} sizePx={sizePx} testId={null} />
    );

  return (
    <CardShell
      stepNum={stepNum}
      moveLabel={moveLabel}
      isStart={true}
      isActive={isActive}
      onClick={onClick}
    >
      {slot}
    </CardShell>
  );
}

// ---------------------------------------------------------------------
// Non-start card — owns the CubeSequence + IntersectionObserver.
// ---------------------------------------------------------------------

function NonStartCard({
  stepNum,
  moveLabel,
  preFacelet,
  move,
  scrambleAlg,
  solution,
  sizePx,
  renderMode,
  isActive,
  onClick,
}: NonStartProps) {
  // Memoize spec by primitive identity so the sequence isn't rebuilt
  // on every render. preFacelet is a 54-char string with stable
  // identity within the parent's `states[]` memo; move is a string
  // literal. msPerMove uses the controller's default (600ms) implicitly.
  const spec = useMemo(
    () => ({ startFacelet: preFacelet, moves: [move], msPerMove: 600 }),
    [preFacelet, move],
  );
  const seq = useCubeSequence(spec);

  // IntersectionObserver — one-shot replay on first scroll-into-view.
  // Effect re-runs only when the sequence identity changes (i.e. a
  // new spec replaces the old one); the observer is torn down and
  // re-attached cleanly via cleanup. Some test environments and
  // older browsers lack IO — feature-detect and skip silently.
  const rootRef = useRef<HTMLButtonElement | null>(null);
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            seq.play();
            io.disconnect();
            break;
          }
        }
      },
      // No options: default root (viewport), 0px margin, 0 threshold —
      // fires the moment any pixel of the card enters view, which is
      // what "scroll into view" means here.
    );
    io.observe(el);
    return () => {
      io.disconnect();
    };
  }, [seq]);

  // Compose onClick: original onActiveChange + the choreographed
  // replay. replayWithReverse() switches between two paths based on
  // sequence state:
  //   - settled at end (status=ended OR last move progress > 0.99):
  //     reverse leg (DUR_REVERSE_MS) → pause leg (DUR_PAUSE_MS) →
  //     forward leg (msPerMove). This is the rev5 choreography —
  //     gives the eye a beat to register the pre-move state before
  //     the forward animation replays.
  //   - any other state (idle / playing / paused / mid-flight):
  //     collapses to replay() (seek 0 + forward play) so the click
  //     feels responsive rather than mandatorily playing the
  //     rewind-pause cue every time.
  const handleClick = () => {
    onClick();
    seq.replayWithReverse();
  };

  return (
    <CardShell
      stepNum={stepNum}
      moveLabel={moveLabel}
      isStart={false}
      isActive={isActive}
      onClick={handleClick}
      rootRef={rootRef}
    >
      <NonStartSlot
        renderMode={renderMode}
        sequence={seq}
        scrambleAlg={scrambleAlg}
        solution={solution}
        stepNum={stepNum}
        sizePx={sizePx}
      />
    </CardShell>
  );
}

function NonStartSlot({
  renderMode,
  sequence,
  scrambleAlg,
  solution,
  stepNum,
  sizePx,
}: {
  renderMode: RenderMode;
  sequence: CubeSequence;
  scrambleAlg: string;
  solution: string[] | null;
  stepNum: number;
  sizePx: number;
}) {
  if (renderMode === "iso") {
    return (
      <TwistyPlayerWrapper
        mode="static"
        scrambleAlg={scrambleAlg}
        solutionAlg={(solution ?? []).slice(0, stepNum).join(" ")}
        sizePx={sizePx}
        testId="twisty-cube"
      />
    );
  }
  if (renderMode === "dual") {
    return (
      <div className="render-pair">
        <Cube2D sequence={sequence} sizePx={sizePx} testId="flat-cube-pair" />
        <TwistyPlayerWrapper
          mode="static"
          scrambleAlg={scrambleAlg}
          solutionAlg={(solution ?? []).slice(0, stepNum).join(" ")}
          sizePx={sizePx}
          testId="twisty-cube-pair"
        />
      </div>
    );
  }
  return <Cube2D sequence={sequence} sizePx={sizePx} testId={null} />;
}
