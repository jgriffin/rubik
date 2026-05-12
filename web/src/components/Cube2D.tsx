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
// consuming `cube2DKinematics.getRenderInstructions(...)`. The math
// module's WIDGET_ENVELOPE (22×19 sticker units, with 5 stickers of
// padding around the cross) defines the kinematics coordinate system
// — slide vectors, rotation pivots, and ribbon extension positions
// are all expressed in those coordinates.
//
// The SVG element's **rendered intrinsic size** matches the static
// mode exactly: width = stickerPx*12, height = stickerPx*9 — the
// cross's natural 12×9 sticker footprint. To achieve this while
// preserving the kinematics coordinate system, the viewBox is
// cropped to the cross region of the WIDGET_ENVELOPE
// (`viewBox="0 0 240 180"` in STICKER_PX units). Slide groups
// (clipped to the cross silhouette) render exactly inside this
// region; face-rotation and F/B ring-rotation groups render at their
// kinematics coordinates and rely on `overflow="visible"` so the
// portions that swing outside the cross silhouette during animation
// still draw (outside the SVG's CSS box).
//
// This contract — animated SVG element same size as static SVG
// element — exists because the production layout (.sol-cell .render
// .net svg { max-width: 100% }) scales any SVG larger than the card
// width down to fit. A 22×19-envelope SVG would intrinsically be
// ~2.4× the static SVG's width, causing CSS to scale the entire
// rendering down and produce a visibly smaller cross. Keeping the
// SVG element 12×9 means the cross renders at exactly the same
// pixel size in both modes — and switching modes (start vs
// non-start card) no longer causes a layout shift.
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
// y*STICKER_PX`, nominal size `STICKER_PX × STICKER_PX`), stroke
// `INK_STROKE`. We render with **the static renderer's gap + rounded-
// corner look** (rev5.4 look), expressed in viewBox units:
//   - gap   = STICKER_PX * 0.06        (matches static's stickerPx*0.06)
//   - rx    = STICKER_PX * 0.06        (matches static's rx=1 at the
//                                       same proportion)
//   - strokeWidth = STICKER_PX * 0.025 (matches static's stickerPx*0.025)
// Production C·P1 fix 2: previously the animated mode used the rev5
// preview's no-gap + crisp-edges look (gap=0, rx=0, strokeWidth=1).
// Side-by-side in the production solution grid, the start card (static
// mode, gap+rx) and the non-start cards (animated mode, no-gap-crisp)
// looked visibly different — the non-start cards looked heavier/blockier.
// Aligning the styles here makes ALL cards in the grid render with the
// same visual cell shape. The kinematics math is untouched; only the
// per-sticker rect geometry changed (the gap/rx is purely cosmetic and
// fits inside the same viewBox-coord position).
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

  // SVG element intrinsic dimensions match StaticInner exactly:
  // stickerPx * 12 wide, stickerPx * 9 tall (the cross's natural
  // footprint). `sizePx` is interpreted as the rendered HEIGHT of
  // the cross, same convention as static mode. This keeps the cross
  // visually identical between modes and prevents production CSS
  // (`.sol-cell .render .net svg { max-width: 100% }`) from scaling
  // the animated SVG down relative to the static SVG.
  const stickerPx = sizePx / 9;
  const widthPx = stickerPx * 12;
  const heightPx = stickerPx * 9;

  // ViewBox is cropped to the cross region of the WIDGET_ENVELOPE
  // coordinate system. The cross occupies sticker cols 0..11
  // (x: 0..240) and rows 0..8 (y: 0..180) in viewBox coords; the
  // WIDGET_ENVELOPE's negative-origin padding (-5,-5 sticker units)
  // is dropped from the rendered viewport. Slide groups and ribbon
  // extensions inside the clip path continue to render in the same
  // coords as before. Rotation groups (face + F/B ring) render at
  // their kinematics coords and rely on `overflow="visible"` to draw
  // outside the SVG's CSS bounding box when their swing corners
  // extend past the cross silhouette.
  const crossVbX = 0;
  const crossVbY = 0;
  const crossVbW = 12 * STICKER_PX;
  const crossVbH = 9 * STICKER_PX;
  const viewBoxStr = `${crossVbX} ${crossVbY} ${crossVbW} ${crossVbH}`;

  const testIdProp =
    testId === null ? {} : { "data-testid": testId ?? "flat-cube" };

  // Cross-silhouette clip-path geometry. Padded by ANIM_STROKE_W on
  // each side (viewBox units) to preserve the outer sticker strokes
  // at the clip boundary without exposing a sliver of the slide-group
  // depart copies that overshoot the cross. The rev5.4 preview used a
  // hand-tuned `+2` viewBox-unit padding for a wider visual stroke;
  // here we match the actual stroke width since animated mode now uses
  // the static renderer's stroke geometry (C·P1 fix 2).
  const clipPad = ANIM_STROKE_W;
  const cVert = {
    x: 3 * STICKER_PX - clipPad,
    y: -clipPad,
    w: 3 * STICKER_PX + 2 * clipPad,
    h: 9 * STICKER_PX + 2 * clipPad,
  };
  // Horizontal strip (L/F/R/B row): cols 0..11 inclusive of rows 3..5,
  // same padding.
  const cHorz = {
    x: -clipPad,
    y: 3 * STICKER_PX - clipPad,
    w: 12 * STICKER_PX + 2 * clipPad,
    h: 3 * STICKER_PX + 2 * clipPad,
  };

  // Slot the groups by kind so we can render them in z-order. We
  // walk plan.groups three times rather than collecting up-front
  // because the JSX reads more clearly with explicit `.filter()`
  // per layer + the discriminant tells TS which kind it has.
  return (
    <svg
      width={widthPx}
      height={heightPx}
      viewBox={viewBoxStr}
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
          production: overlayOpacity=0 → no rect rendered). Covers
          the visible viewBox (the cross region); since overlay use
          is dev-tooling-only and the cross region is the visible
          area, this is the correct extent. */}
      {plan.overlayOpacity > 0 && (
        <rect
          x={crossVbX}
          y={crossVbY}
          width={crossVbW}
          height={crossVbH}
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
//
// Geometry mirrors the static renderer (gap+rx; rev5.4 look), expressed
// in viewBox units. See `SvgFromRenderPlan` header for the C·P1 fix 2
// rationale.
const ANIM_GAP = STICKER_PX * 0.06;
const ANIM_RX = STICKER_PX * 0.06;
const ANIM_STROKE_W = STICKER_PX * 0.025;

function StickerRect({ sticker }: { sticker: AnimatedSticker }) {
  const color = COLOR_FOR_LETTER[sticker.color] ?? "#444";
  return (
    <rect
      x={sticker.x * STICKER_PX + ANIM_GAP / 2}
      y={sticker.y * STICKER_PX + ANIM_GAP / 2}
      width={STICKER_PX - ANIM_GAP}
      height={STICKER_PX - ANIM_GAP}
      rx={ANIM_RX}
      fill={color}
      stroke={INK_STROKE}
      strokeWidth={ANIM_STROKE_W}
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
