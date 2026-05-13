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
//   - Pointer events (mouse / touch / pen):
//       onPointerDown → onActiveChange() + seq.replayWithReverseHold()
//                       (reverse leg + indefinite held dwell at the
//                       pre-state).
//       onPointerUp / onPointerCancel / onPointerLeave →
//                       seq.releaseHold() (forward play; collapses to
//                       reverse-then-forward if release happens before
//                       the reverse leg completes — see
//                       state/cubeSequence.ts).
//     Pointer-driven interaction lets the user "scrub" the pre-state
//     dwell by holding longer; quick clicks naturally collapse to a
//     snappy reverse → forward.
//   - onClick: kept as a keyboard fallback (Space/Enter on a focused
//     button fires a synthetic click but NOT pointer events). Routed
//     to seq.replayWithReverse() — the finite-pause choreography —
//     giving keyboard users the same quick visual cue without needing
//     a held interaction. A useRef flag set during pointerdown /
//     cleared during pointerup dedupes against the pointer path so
//     the click event fired by browsers on pointer release (when it
//     does) doesn't double-trigger.
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
  moves: string[];
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
  onPointerDown,
  onPointerUp,
  onPointerCancel,
  onPointerLeave,
  rootRef,
  children,
}: {
  stepNum: number;
  moveLabel: string | null;
  isStart: boolean;
  isActive: boolean;
  onClick: (e: React.MouseEvent<HTMLButtonElement>) => void;
  onPointerDown?: (e: React.PointerEvent<HTMLButtonElement>) => void;
  onPointerUp?: (e: React.PointerEvent<HTMLButtonElement>) => void;
  onPointerCancel?: (e: React.PointerEvent<HTMLButtonElement>) => void;
  onPointerLeave?: (e: React.PointerEvent<HTMLButtonElement>) => void;
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
      onPointerDown={onPointerDown}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerCancel}
      onPointerLeave={onPointerLeave}
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
  moves,
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
        solutionAlg={moves.slice(0, stepNum).join(" ")}
        sizePx={sizePx}
        testId="twisty-cube"
      />
    ) : renderMode === "dual" ? (
      <div className="render-pair">
        <Cube2D facelet={facelet} sizePx={sizePx} testId="flat-cube-pair" />
        <TwistyPlayerWrapper
          mode="static"
          scrambleAlg={scrambleAlg}
          solutionAlg={moves.slice(0, stepNum).join(" ")}
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
  moves,
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

  // Pointer-driven press-and-hold interaction:
  //   pointerdown → onActiveChange + replayWithReverseHold() (reverse
  //                 leg + indefinite held dwell at pre-state).
  //   pointerup / pointercancel / pointerleave → releaseHold() (forward
  //                 play; collapses to reverse-then-forward if released
  //                 mid-reverse).
  //
  // pointerHandledRef gates the onClick fallback. A click event fired
  // synchronously after a pointerup (the normal mouse path) finds the
  // ref set and skips the keyboard branch — the pointer handlers
  // already drove the choreography. Keyboard-only activation
  // (Space/Enter on focused button) fires a synthetic click WITHOUT
  // any pointerdown/up, so the ref stays false and the click runs the
  // finite-pause replayWithReverse() as a "snappy keyboard cue".
  const pointerHandledRef = useRef(false);

  const handlePointerDown = (e: React.PointerEvent<HTMLButtonElement>) => {
    pointerHandledRef.current = true;
    // Pointer capture: routes subsequent pointer events to this
    // element even if the cursor drags out, so we still get
    // pointerup → releaseHold instead of being stuck in the held
    // state. Some browsers/test envs don't implement it — guard.
    const target = e.currentTarget;
    if (typeof target.setPointerCapture === "function") {
      try {
        target.setPointerCapture(e.pointerId);
      } catch {
        // ignore: pointerId may have been auto-released by the browser.
      }
    }
    onClick();
    seq.replayWithReverseHold();
  };

  const handlePointerUp = () => {
    if (!pointerHandledRef.current) return;
    seq.releaseHold();
    // Note: pointerHandledRef is cleared on the trailing click event
    // (some browsers fire click after pointerup, others don't). If no
    // click follows (e.g. drag-off + release outside the element), the
    // ref's leftover true state is harmless — it only matters for the
    // next click event, and any subsequent pointerdown resets it.
  };

  const handlePointerCancel = () => {
    // System cancelled the pointer (e.g. scroll gesture stole it,
    // touch was interrupted). Treat as a release so we don't get
    // stuck holding the pre-state.
    if (!pointerHandledRef.current) return;
    seq.releaseHold();
  };

  const handlePointerLeave = () => {
    // User dragged off the card while still pressing. Treat as
    // release; pointer capture should have prevented this for most
    // browsers, but the leave fallback is the safety net.
    if (!pointerHandledRef.current) return;
    seq.releaseHold();
  };

  const handleClick = () => {
    // If a pointer interaction just drove the choreography, the
    // pointer-handled flag is set — consume it and skip. Otherwise
    // this is a keyboard activation (Space/Enter); fall back to the
    // finite-pause replayWithReverse() for a snappy keyboard cue
    // and still notify the parent of the active change.
    if (pointerHandledRef.current) {
      pointerHandledRef.current = false;
      return;
    }
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
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerCancel}
      onPointerLeave={handlePointerLeave}
      rootRef={rootRef}
    >
      <NonStartSlot
        renderMode={renderMode}
        sequence={seq}
        scrambleAlg={scrambleAlg}
        moves={moves}
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
  moves,
  stepNum,
  sizePx,
}: {
  renderMode: RenderMode;
  sequence: CubeSequence;
  scrambleAlg: string;
  moves: string[];
  stepNum: number;
  sizePx: number;
}) {
  if (renderMode === "iso") {
    return (
      <TwistyPlayerWrapper
        mode="static"
        scrambleAlg={scrambleAlg}
        solutionAlg={moves.slice(0, stepNum).join(" ")}
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
          solutionAlg={moves.slice(0, stepNum).join(" ")}
          sizePx={sizePx}
          testId="twisty-cube-pair"
        />
      </div>
    );
  }
  return <Cube2D sequence={sequence} sizePx={sizePx} testId={null} />;
}
