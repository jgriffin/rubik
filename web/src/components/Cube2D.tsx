// 2D cube renderer. One component, two modes:
//
//   <Cube2D facelet="UUU..." sizePx={240} />            // static
//   <Cube2D sequence={seq} sizePx={240} />              // animated
//
// Discriminated-union props enforce mutual exclusivity at the type
// level — passing both `facelet` and `sequence` is a type error.
//
// Static mode is the original `FlatCubeRenderer` body, byte-equivalent
// to the rev5.4-and-earlier renderer. The SVG it emits (12×9 sticker
// grid, viewBox `0 0 widthPx heightPx`, gap + rounded corners, ink
// stroke, `data-pos` + `data-color` attributes per sticker) is the
// contract the e2e specs and `SolutionCard` consumers rely on.
//
// Animated mode subscribes to the passed-in `CubeSequence` via
// `useSyncExternalStore` and renders the rev5 animation overlay by
// consuming `cube2DKinematics.getRenderInstructions(...)`. The
// animated SVG uses the **widget envelope** from cube2DKinematics
// (22×19 sticker viewBox), so ribbon extensions + face-rotation
// swing corners fit cleanly at any rendered scale. Geometry is
// independent of `sizePx` — `sizePx` only sets the rendered pixel
// box; viewBox math + clip-path coordinates are in sticker units.
//
// When `currentMoveIndex === -1` (idle, pre-first-move), animated
// mode falls through to the static renderer rendering the start
// facelet. Block C will trigger `replay()` on intersection-observer,
// so this fall-through is mostly transitional state. Layout note:
// the static SVG is sized 12×9 sticker units while the animated SVG
// is 22×19 (widget envelope); switching modes causes a brief layout
// shift — acceptable since pre-animation is a short-lived state.
//
// References:
//   - `cube2DKinematics.ts` (A·P1) — math + types this component
//     consumes via `getRenderInstructions`.
//   - `web/preview/flat-cube-animated.html` (Block 1B' rev5.4 + A·P0)
//     — design playground; the inline `buildNetCubeSvg` function is
//     the structural prototype for animated mode's JSX layer order.
//   - `state/cubeSequence.ts` (A·P2) — `CubeSequence` type.

import { useId, useSyncExternalStore } from "react";
import {
  COLOR_FOR_LETTER,
  FACE_OFFSETS,
  INK_STROKE,
} from "./cubePalette";
import {
  STICKER_PX,
  WIDGET_ENVELOPE,
  easeOut,
  getRenderInstructions,
  type AnimatedSticker,
  type RenderPlan,
  type RotationGroup,
  type SlideGroup,
} from "./cube2DKinematics";
import { applyMoves } from "../state/applyMove";
import type { CubeSequence } from "../state/cubeSequence";

// ---------------------------------------------------------------------
// Props — discriminated union enforcing mutual exclusivity
// ---------------------------------------------------------------------

type CommonProps = {
  sizePx?: number;
  testId?: string | null;
};

type StaticProps = CommonProps & {
  facelet: string;
  sequence?: undefined;
};

type AnimatedProps = CommonProps & {
  sequence: CubeSequence;
  facelet?: undefined;
};

export type Cube2DProps = StaticProps | AnimatedProps;

// ---------------------------------------------------------------------
// Cube2D — dispatch by mode
// ---------------------------------------------------------------------

export default function Cube2D(props: Cube2DProps) {
  if (props.sequence !== undefined) {
    return (
      <AnimatedInner
        sequence={props.sequence}
        sizePx={props.sizePx}
        testId={props.testId}
      />
    );
  }
  return (
    <StaticInner
      facelet={props.facelet}
      sizePx={props.sizePx}
      testId={props.testId}
    />
  );
}

// ---------------------------------------------------------------------
// StaticInner — byte-equivalent port of the original FlatCubeRenderer.
//
// SVG output, viewBox, sticker geometry, gap, rounded corners, ink
// stroke, and per-sticker `data-pos` / `data-color` attributes must
// match the pre-rename `FlatCubeRenderer` exactly. The e2e suite
// (`web/e2e/renderer.spec.ts`, `solution-grid.spec.ts`) asserts this
// contract directly via `[data-pos]` + `[data-color]` selectors.
// ---------------------------------------------------------------------

// Position of each face in the static cross grid (4 cols × 3 rows of
// faces). This is the COMPACT layout — distinct from the WIDGET
// envelope's padded coordinate system in `cube2DKinematics`. Stays
// here (and matches the original FlatCubeRenderer) because static mode
// is a 1:1 port; the widget envelope is animated-mode-only.
const STATIC_FACE_GRID: Record<string, { col: number; row: number }> = {
  U: { col: 1, row: 0 },
  L: { col: 0, row: 1 },
  F: { col: 1, row: 1 },
  R: { col: 2, row: 1 },
  B: { col: 3, row: 1 },
  D: { col: 1, row: 2 },
};

function StaticInner({
  facelet,
  sizePx = 240,
  testId,
}: {
  facelet: string;
  sizePx?: number;
  testId?: string | null;
}) {
  if (facelet.length !== 54) {
    throw new Error(
      `Cube2D: facelet must be 54 chars, got ${facelet.length}`,
    );
  }
  const stickerPx = sizePx / 9;
  const widthPx = stickerPx * 12;
  const heightPx = stickerPx * 9;
  const gap = stickerPx * 0.06;
  const stroke = INK_STROKE;
  const strokeWidth = Math.max(0.5, stickerPx * 0.025);

  const stickers: Array<{
    x: number;
    y: number;
    color: string;
    pos: number;
    letter: string;
  }> = [];

  for (const face of Object.keys(FACE_OFFSETS) as Array<
    keyof typeof FACE_OFFSETS
  >) {
    const { col, row } = STATIC_FACE_GRID[face];
    const offset = FACE_OFFSETS[face];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const localIdx = r * 3 + c;
        const globalIdx = offset + localIdx;
        const letter = facelet[globalIdx];
        stickers.push({
          x: (col * 3 + c) * stickerPx,
          y: (row * 3 + r) * stickerPx,
          color: COLOR_FOR_LETTER[letter] ?? "#444",
          pos: globalIdx,
          letter,
        });
      }
    }
  }

  const testIdProp =
    testId === null ? {} : { "data-testid": testId ?? "flat-cube" };

  return (
    <svg
      width={widthPx}
      height={heightPx}
      viewBox={`0 0 ${widthPx} ${heightPx}`}
      style={{ display: "block" }}
      {...testIdProp}
    >
      {stickers.map((s) => (
        <rect
          key={s.pos}
          x={s.x + gap / 2}
          y={s.y + gap / 2}
          width={stickerPx - gap}
          height={stickerPx - gap}
          rx={1}
          fill={s.color}
          stroke={stroke}
          strokeWidth={strokeWidth}
          data-pos={s.pos}
          data-color={s.letter}
        />
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------------
// AnimatedInner — subscribes to a CubeSequence and renders the rev5
// overlay via `getRenderInstructions`.
//
// Snapshot strategy: subscribe to `sequence.timestamp` directly. The
// timestamp is a primitive that changes monotonically while playing
// and is === -stable otherwise, so `useSyncExternalStore`'s default
// identity check works without a separate version counter.
//
// `useCubeSequence` uses a separate version counter because IT owns
// the sequence and needs to bump on any state change including pure
// status flips at timestamp=0. Here we're a downstream consumer of an
// externally-owned sequence; `timestamp` is enough because every
// rAF tick + every control method (play/pause/seek/replay) advances
// or jumps the timestamp — and even when timestamp DOESN'T change
// (a pause()/play() pair at timestamp 0), the controller still
// notifies, and getSnapshot returns the same value (0), which is the
// correct React behaviour: no rerender needed since the rendered
// SVG is unchanged.
// ---------------------------------------------------------------------

function AnimatedInner({
  sequence,
  sizePx = 240,
  testId,
}: {
  sequence: CubeSequence;
  sizePx?: number;
  testId?: string | null;
}) {
  useSyncExternalStore(
    sequence.subscribe,
    () => sequence.timestamp,
    () => 0,
  );

  const currentMoveIndex = sequence.currentMoveIndex;
  const progress = sequence.currentMoveProgress;
  const moves = sequence.moves;
  const startFacelet = sequence.startFacelet;

  // Pre-animation state — render the start facelet statically. The
  // controller reports currentMoveIndex = -1 only when idle at
  // timestamp 0; once play() advances, currentMoveIndex jumps to 0.
  if (currentMoveIndex < 0 || moves.length === 0) {
    return (
      <StaticInner
        facelet={startFacelet}
        sizePx={sizePx}
        testId={testId}
      />
    );
  }

  const move = moves[currentMoveIndex];
  // Pre-move facelet (state going INTO the current move):
  // startFacelet with moves[0..currentMoveIndex-1] applied.
  // Per-card consumers (block C) pass moves: [card.move] so the slice
  // is at most 1 move long — cheap. For longer sequences (block D's
  // full-solve viewer) this still runs once per rAF tick, which is
  // fine: applyMove is a 54-char array copy, vanishingly small at
  // 60 FPS. Re-evaluate only if profiling shows hot-path cost.
  const preFacelet = applyMoves(
    startFacelet,
    moves.slice(0, currentMoveIndex),
  );

  const plan = getRenderInstructions(preFacelet, move, progress);
  return (
    <SvgFromRenderPlan
      plan={plan}
      sizePx={sizePx}
      testId={testId}
    />
  );
}

// ---------------------------------------------------------------------
// SvgFromRenderPlan — JSX consumer of a RenderPlan.
//
// Mirrors the preview's `buildNetCubeSvg` z-order:
//   1. <defs> with cross-silhouette clipPath
//   2. Static stickers (the 33 untouched by the move)
//   3. Slide groups (primary + optional secondary), clipped to cross
//   4. Inverse-opacity overlay (hidden in production — overlayOpacity=0)
//   5. Face-rotation group, UNCLIPPED (rotation corners overflow cross)
//   6. Ring-rotation group for F/B, UNCLIPPED
//
// Transforms are computed here by interpolating slide-vector +
// rotation-angle with `easeOut(progress)`. The RenderPlan carries
// TARGET values (at progress=1); the renderer applies the easing.
// This keeps the easing curve in one place (the renderer) and lets
// the math module stay progress-independent.
//
// Per-sticker rendering: viewBox-coord stickers (`x*STICKER_PX,
// y*STICKER_PX`, size `STICKER_PX × STICKER_PX`), stroke `INK_STROKE`
// width 1 (in viewBox coords — invariant across `sizePx`), no gap,
// no rounded corners (matches the preview, not the static renderer's
// rounded-with-gap look — animated mode is the rev5 look).
//
// `clipPath` id must be unique per rendered instance to avoid SVG id
// collisions when multiple Cube2D instances mount simultaneously. We
// scope via `useId()`.
// ---------------------------------------------------------------------

function SvgFromRenderPlan({
  plan,
  sizePx,
  testId,
}: {
  plan: RenderPlan;
  sizePx: number;
  testId?: string | null;
}) {
  const idScope = useId();
  const clipId = `cube2d-clip-${idScope}`;

  // Animated viewBox is the widget envelope (22×19 stickers, padded).
  const vb = WIDGET_ENVELOPE;
  // `sizePx` is interpreted as the rendered HEIGHT of the visible
  // cross (9 sticker rows tall in the cross silhouette), matching the
  // static renderer's `sizePx = 9 stickers tall` convention. The
  // animated SVG envelope is taller (19 rows of which 9 are cross +
  // 10 are padding), so the rendered height = sizePx * (19/9).
  const stickerPx = sizePx / 9;
  const heightPx = stickerPx * (vb.height / STICKER_PX);
  const widthPx = stickerPx * (vb.width / STICKER_PX);

  const testIdProp =
    testId === null ? {} : { "data-testid": testId ?? "flat-cube" };

  // Cross-silhouette clip-path geometry, matching the preview.
  // Vertical strip (U/F/D column): cols 3..5 inclusive of rows 0..8,
  // padded by 2 px on each side (in sticker-coord px) to preserve the
  // outer half of sticker strokes at the clip boundary (rev5.4 fix).
  const cVert = {
    x: 3 * STICKER_PX - 2,
    y: -2,
    w: 3 * STICKER_PX + 4,
    h: 9 * STICKER_PX + 4,
  };
  // Horizontal strip (L/F/R/B row): cols 0..11 inclusive of rows 3..5,
  // same +2 padding.
  const cHorz = {
    x: -2,
    y: 3 * STICKER_PX - 2,
    w: 12 * STICKER_PX + 4,
    h: 3 * STICKER_PX + 4,
  };

  // Slot the groups by kind so we can render them in z-order. We
  // walk plan.groups three times rather than collecting up-front
  // because the JSX reads more clearly with explicit `.filter()`
  // per layer + the discriminant tells TS which kind it has.
  return (
    <svg
      width={widthPx}
      height={heightPx}
      viewBox={vb.viewBoxStr}
      style={{ display: "block", overflow: "visible" }}
      overflow="visible"
      {...testIdProp}
    >
      <defs>
        <clipPath id={clipId} clipPathUnits="userSpaceOnUse">
          <rect x={cVert.x} y={cVert.y} width={cVert.w} height={cVert.h} />
          <rect x={cHorz.x} y={cHorz.y} width={cHorz.w} height={cHorz.h} />
        </clipPath>
      </defs>

      {/* Layer 1: static stickers (the 33 untouched by the move). */}
      <g data-layer="static">
        {plan.staticStickers.map((s, i) => (
          <StickerRect key={`s-${i}`} sticker={s} />
        ))}
      </g>

      {/* Layer 2: slide groups, clipped to cross silhouette. */}
      <g data-layer="ribbon-container" clipPath={`url(#${clipId})`}>
        {plan.groups
          .filter((g): g is SlideGroup => g.kind === "slide")
          .map((g, i) => (
            <SlideGroupG
              key={`slide-${g.role}-${i}`}
              group={g}
              progress={plan.progress}
            />
          ))}
      </g>

      {/* Layer 3: inverse-opacity dimming overlay (hidden in
          production: overlayOpacity=0 → no rect rendered). */}
      {plan.overlayOpacity > 0 && (
        <rect
          x={vb.x}
          y={vb.y}
          width={vb.width}
          height={vb.height}
          fill="currentColor"
          opacity={plan.overlayOpacity}
          pointerEvents="none"
          data-layer="extension-overlay"
        />
      )}

      {/* Layer 4: face-rotation group (UNCLIPPED — rotation corners
          overflow the cross silhouette and must remain visible). */}
      {plan.groups
        .filter(
          (g): g is RotationGroup =>
            g.kind === "rotation" && g.rotationKind === "face",
        )
        .map((g, i) => (
          <RotationGroupG
            key={`face-${i}`}
            group={g}
            progress={plan.progress}
          />
        ))}

      {/* Layer 5: ring-rotation group for F/B (UNCLIPPED). */}
      {plan.groups
        .filter(
          (g): g is RotationGroup =>
            g.kind === "rotation" && g.rotationKind === "ring",
        )
        .map((g, i) => (
          <RotationGroupG
            key={`ring-${i}`}
            group={g}
            progress={plan.progress}
          />
        ))}
    </svg>
  );
}

// One sticker, rendered at its viewBox-coord position. Color comes
// from the palette via `COLOR_FOR_LETTER`. Inline so the SVG tree
// stays compact and the per-sticker map keys read clearly.
function StickerRect({ sticker }: { sticker: AnimatedSticker }) {
  const color = COLOR_FOR_LETTER[sticker.color] ?? "#444";
  return (
    <rect
      x={sticker.x * STICKER_PX}
      y={sticker.y * STICKER_PX}
      width={STICKER_PX}
      height={STICKER_PX}
      fill={color}
      stroke={INK_STROKE}
      strokeWidth={1}
      shapeRendering="crispEdges"
      data-color={sticker.color}
    />
  );
}

function SlideGroupG({
  group,
  progress,
}: {
  group: SlideGroup;
  progress: number;
}) {
  // Slide translate at this progress. Easing applied here so the math
  // module stays progress-independent (target values only).
  const e = easeOut(progress);
  const tx = group.slideVector.dx * STICKER_PX * e;
  const ty = group.slideVector.dy * STICKER_PX * e;
  return (
    <g
      data-layer={`edge-${group.role}`}
      transform={`translate(${tx} ${ty})`}
    >
      {group.stickers.map((s, i) => (
        <StickerRect key={i} sticker={s} />
      ))}
    </g>
  );
}

function RotationGroupG({
  group,
  progress,
}: {
  group: RotationGroup;
  progress: number;
}) {
  const e = easeOut(progress);
  const angle = group.angleDeg * e;
  return (
    <g
      data-layer={group.rotationKind === "face" ? "face" : "edge-ring"}
      transform={`rotate(${angle} ${group.pivotPx.x} ${group.pivotPx.y})`}
    >
      {group.stickers.map((s, i) => (
        <StickerRect key={i} sticker={s} />
      ))}
    </g>
  );
}
