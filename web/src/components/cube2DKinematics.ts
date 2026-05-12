// rev5 2D cube animation math — pure functions + types.
//
// Source of truth for the 2D move animation model originally locked in
// `web/preview/flat-cube-animated.html` (Block 1B', rev5.4). The
// preview keeps its inline standalone copy as the design playground;
// this module is the production import path for A·P3's `Cube2D.tsx`
// (animated mode) and any downstream consumer.
//
// Visual model summary:
//   - 9-sticker face turn = rotation primitive around face geometric
//     center (col*3 + 1.5, row*3 + 1.5 in sticker units).
//   - 12-sticker edge cycle for U / U' / D / D' = horizontal ribbon
//     slide with off-net virtual extensions ("h-ribbon"). One primary
//     slide group; no secondary.
//   - 12-sticker edge cycle for R / R' / L / L' = vertical ribbon
//     slide ("v-ribbon"). Nine stickers form the primary group at the
//     U/F/D column; three stickers at B's near-F column form a
//     secondary group sliding OPPOSITE primary (B's content arrives
//     over the top/bottom of the cube in 3D, projecting onto the net
//     as a reverse-direction slide).
//   - 12-sticker edge cycle for F / F' / B / B' = ring rotation
//     primitive around F-center ("ring"). No slide; both F and B
//     rings share the F-center pivot since both form closed rings
//     around it.
//
// Slide directions derived data-first from FACELET_MOVES (mode of
// motion vectors across edge stickers, preferring magnitude-3) — no
// hand-authored direction tables. F/B ring angles computed from
// arbitrary edge-sticker source/dest around F-center.
//
// Timing constants (DUR_FORWARD_MS, DUR_REVERSE_MS, DUR_PAUSE_MS)
// carry forward from rev5 verbatim. The cubic ease-out curve carries
// forward as `easeOut(t)`.
//
// References:
//   - `web/preview/flat-cube-animated.html` (rev5.4 + A·P0 sequence
//     widget) — the visual playground this module mirrors.
//   - `plans/m9-cube-animation-system.md` — block plan with
//     architectural rationale.
//   - LOG block A — block opening + per-phase notes.

import { applyMove } from "../state/applyMove";
import { FACELET_MOVES, type MoveStr } from "../state/faceletMoves";

// ---------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------

// Animation timing knobs, in milliseconds. Locked in rev5.3.
//   DUR_FORWARD_MS — forward play (settled → end).
//   DUR_REVERSE_MS — auto-reset reverse leg before replay; short so
//                    the pause leg below stands out from it.
//   DUR_PAUSE_MS   — hold-at-zero between reverse and forward on
//                    auto-reset, gives the eye a beat to register
//                    the pre-move state.
export const DUR_FORWARD_MS = 600;
export const DUR_REVERSE_MS = 150;
export const DUR_PAUSE_MS = 250;

// Logical sticker size in SVG viewBox units. The renderer (A·P3)
// chooses the CSS-px scale; this value is the viewBox-coord size of
// one sticker. All position math below is in these "sticker units"
// unless otherwise marked (`*Px` suffix = STICKER_PX-scaled).
export const STICKER_PX = 20;

// Sticker indices for each face within the 54-char URFDLB facelet
// string. Palette-agnostic; matches `web/src/components/cubePalette.ts`.
export const FACE_OFFSETS = {
  U: 0,
  R: 9,
  F: 18,
  D: 27,
  L: 36,
  B: 45,
} as const;
export type FaceLetter = keyof typeof FACE_OFFSETS;

// Position of each face in the cross-net grid (4 cols × 3 rows of
// faces; each face is 3×3 stickers). Matches Cube2D's static-mode
// FACE_GRID.
export const FACE_GRID: Record<FaceLetter, { col: number; row: number }> = {
  U: { col: 1, row: 0 },
  L: { col: 0, row: 1 },
  F: { col: 1, row: 1 },
  R: { col: 2, row: 1 },
  B: { col: 3, row: 1 },
  D: { col: 1, row: 2 },
};

// SVG viewBox envelopes. Two sizes:
//   PANEL_ENVELOPE  — 17×14 sticker units. Per-move panels (rev5.4).
//                     Compact; depart copies at progress=1 sit exactly
//                     on the right/bottom edge — fine at panel scale.
//   WIDGET_ENVELOPE — 22×19 sticker units. Big-cross sequence widget
//                     at 2× scale. 2 extra stickers of padding on each
//                     side clear worst-case ribbon overshoot + face
//                     rotation swing corners without edge artifacts.
//
// `viewBoxStr` is a pre-formatted string for the SVG `viewBox`
// attribute (avoids template-literal duplication at call sites).
export interface ViewBoxEnvelope {
  x: number;
  y: number;
  width: number;
  height: number;
  viewBoxStr: string;
}
function envelope(
  xCols: number,
  yRows: number,
  wCols: number,
  hRows: number,
): ViewBoxEnvelope {
  const x = xCols * STICKER_PX;
  const y = yRows * STICKER_PX;
  const width = wCols * STICKER_PX;
  const height = hRows * STICKER_PX;
  return { x, y, width, height, viewBoxStr: `${x} ${y} ${width} ${height}` };
}
// Panel envelope: cols -3..13 (17 wide), rows -3..10 (14 tall). 17×14
// is the rev5.4 viewBox preserved byte-for-byte from the preview.
export const PANEL_ENVELOPE: ViewBoxEnvelope = envelope(-3, -3, 17, 14);
// Widget envelope: cols -5..16 (22 wide), rows -5..13 (19 tall). 22×19
// from A·P0 fix — adds 2-sticker headroom over the panel envelope.
export const WIDGET_ENVELOPE: ViewBoxEnvelope = envelope(-5, -5, 22, 19);

// ---------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------

// (col, row) of a face's center sticker in INTEGER sticker grid
// coords. Used for rotation-angle math (atan2 reference point); the
// integer offset cancels uniformly so this gives the correct angle
// without floating-point noise.
function faceCenterXYInt(face: FaceLetter): { x: number; y: number } {
  const { col, row } = FACE_GRID[face];
  return { x: col * 3 + 1, y: row * 3 + 1 };
}
// (px-x, px-y) of a face's true geometric center, scaled by
// STICKER_PX. Used as the SVG rotate() pivot.
export function faceCenterPx(face: FaceLetter): { x: number; y: number } {
  const { col, row } = FACE_GRID[face];
  return {
    x: (col * 3 + 1.5) * STICKER_PX,
    y: (row * 3 + 1.5) * STICKER_PX,
  };
}

// Decompose a facelet sticker index into its (face, local-row,
// local-col) coordinates.
function indexToFaceLocal(
  i: number,
): { face: FaceLetter; r: number; c: number } {
  for (const face of Object.keys(FACE_OFFSETS) as FaceLetter[]) {
    const off = FACE_OFFSETS[face];
    if (i >= off && i < off + 9) {
      const local = i - off;
      return { face, r: Math.floor(local / 3), c: local % 3 };
    }
  }
  throw new Error(`bad facelet index ${i}`);
}
// Return the face a given facelet index belongs to.
function indexFace(i: number): FaceLetter {
  for (const face of Object.keys(FACE_OFFSETS) as FaceLetter[]) {
    const off = FACE_OFFSETS[face];
    if (i >= off && i < off + 9) return face;
  }
  throw new Error(`bad facelet index ${i}`);
}
// Sticker index → (x, y) in INTEGER sticker grid coords (top-left of
// the sticker rect). Multiply by STICKER_PX for SVG viewBox coords.
export function indexToXY(i: number): { x: number; y: number } {
  const { face, r, c } = indexToFaceLocal(i);
  const { col, row } = FACE_GRID[face];
  return { x: col * 3 + c, y: row * 3 + r };
}

// Compute the angle (in degrees, normalized to (-180, 180]) by which
// `srcXY` must be rotated around `centerXY` to land at `dstXY`. Used
// for face-rotation and F/B ring-rotation angle derivation.
function rotationAngleDeg(
  srcXY: { x: number; y: number },
  dstXY: { x: number; y: number },
  centerXY: { x: number; y: number },
): number {
  const a1 = Math.atan2(srcXY.y - centerXY.y, srcXY.x - centerXY.x);
  const a2 = Math.atan2(dstXY.y - centerXY.y, dstXY.x - centerXY.x);
  let d = (a2 - a1) * (180 / Math.PI);
  while (d > 180) d -= 360;
  while (d <= -180) d += 360;
  return d;
}

// ---------------------------------------------------------------------
// Move classification + participation
// ---------------------------------------------------------------------

export type MoveKind = "h-ribbon" | "v-ribbon" | "ring";

function turnedFace(move: MoveStr): FaceLetter {
  return move.charAt(0) as FaceLetter;
}
function moveKind(move: MoveStr): MoveKind {
  const f = turnedFace(move);
  if (f === "U" || f === "D") return "h-ribbon";
  if (f === "R" || f === "L") return "v-ribbon";
  return "ring";
}

// Partition the 54 facelet indices into three disjoint groups for the
// given move:
//   faceIndices   — 9 stickers of the turned face (rotate as a rigid
//                   block around the face center)
//   edgeIndices   — 12 stickers cycling between adjacent faces
//                   (slide for U/D/R/L, ring-rotate for F/B)
//   staticIndices — the remaining 33 (perm[i] === i)
export interface MoveParticipation {
  faceIndices: number[];
  edgeIndices: number[];
  staticIndices: number[];
}
export function participatingForMove(move: MoveStr): MoveParticipation {
  const perm = FACELET_MOVES[move];
  const face = turnedFace(move);
  const off = FACE_OFFSETS[face];
  const faceIndices: number[] = [];
  for (let i = off; i < off + 9; i++) faceIndices.push(i);
  const edgeIndices: number[] = [];
  const staticIndices: number[] = [];
  for (let i = 0; i < 54; i++) {
    if (i >= off && i < off + 9) continue;
    if (perm[i] === i) staticIndices.push(i);
    else edgeIndices.push(i);
  }
  return { faceIndices, edgeIndices, staticIndices };
}

// ---------------------------------------------------------------------
// Slide vector (data-first)
// ---------------------------------------------------------------------

// 2D slide vector in sticker units. (dx, dy) for one face-step of
// translation along the ribbon axis.
export interface SlideVector {
  dx: number;
  dy: number;
}

// Derive the dominant slide vector for a move by taking the MODE of
// (post_xy - pre_xy) across the 12 edge stickers, preferring
// magnitude-3 vectors (single-face slide step). This is the
// "data-first" rule from rev5: hand-authored direction tables silently
// disagree with the physical wrap topology, so the data dictates the
// direction. Returns (0, 0) for ring moves (F/B) where the
// displacements aren't collinear and there's no single mode.
export function slideVector(move: MoveStr): SlideVector {
  const perm = FACELET_MOVES[move];
  const { edgeIndices } = participatingForMove(move);
  const counts = new Map<string, number>();
  for (const iPost of edgeIndices) {
    const post = indexToXY(iPost);
    const pre = indexToXY(perm[iPost]);
    const dx = post.x - pre.x;
    const dy = post.y - pre.y;
    const key = `${dx},${dy}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  let best: SlideVector | null = null;
  let bestScore = -1;
  for (const [key, c] of counts.entries()) {
    const parts = key.split(",").map(Number);
    const dx = parts[0];
    const dy = parts[1];
    const mag2 = dx * dx + dy * dy;
    // Prefer length-3 vectors; otherwise pick most common.
    const score = c + (mag2 === 9 ? 100 : 0);
    if (score > bestScore) {
      bestScore = score;
      best = { dx, dy };
    }
  }
  return best ?? { dx: 0, dy: 0 };
}

// ---------------------------------------------------------------------
// Rotation angles
// ---------------------------------------------------------------------

// Angle by which the turned face rotates around its center to take
// progress=0 → progress=1. Derived from the perm's effect on the
// face's top-left corner sticker.
export function faceRotationAngle(move: MoveStr): number {
  const perm = FACELET_MOVES[move];
  const face = turnedFace(move);
  const off = FACE_OFFSETS[face];
  const center = faceCenterXYInt(face);
  const iDst = off;
  const iSrc = perm[iDst];
  const dst = indexToXY(iDst);
  const src = indexToXY(iSrc);
  return rotationAngleDeg(src, dst, center);
}
// Angle by which the F/B ring rotates around F-center. Both F and B
// edge cycles form closed rings around F-center (different radii but
// same pivot). Returns 0 for non-ring moves.
export function ringRotationAngle(move: MoveStr): number {
  if (moveKind(move) !== "ring") return 0;
  const perm = FACELET_MOVES[move];
  const { edgeIndices } = participatingForMove(move);
  const center = faceCenterXYInt("F");
  const iDst = edgeIndices[0];
  const iSrc = perm[iDst];
  const dst = indexToXY(iDst);
  const src = indexToXY(iSrc);
  return rotationAngleDeg(src, dst, center);
}

// ---------------------------------------------------------------------
// MoveKinematics — derived data-first plan for one move (move-only;
// independent of preFacelet). Stable for a given move.
// ---------------------------------------------------------------------

export interface MoveKinematics {
  move: MoveStr;
  kind: MoveKind;
  turnedFace: FaceLetter;
  // Primary slide vector (h-ribbon: only group; v-ribbon: U/F/D
  // column; ring: zero).
  slideVector: SlideVector;
  // Target angle of the turned-face rotation at progress=1, in degrees.
  faceRotationAngle: number;
  // Target angle of the F/B ring rotation at progress=1, in degrees.
  // Zero for non-ring moves.
  ringRotationAngle: number;
  // Pivot point of the turned-face rotation, in SVG viewBox px.
  faceCenterPx: { x: number; y: number };
  // Pivot point of the F/B ring rotation, in SVG viewBox px. Always
  // F-center regardless of move (F and B share the pivot).
  ringCenterPx: { x: number; y: number };
}

export function computeMoveKinematics(move: MoveStr): MoveKinematics {
  const kind = moveKind(move);
  const sv = slideVector(move);
  const turned = turnedFace(move);
  return {
    move,
    kind,
    turnedFace: turned,
    slideVector: sv,
    faceRotationAngle: faceRotationAngle(move),
    ringRotationAngle: kind === "ring" ? ringRotationAngle(move) : 0,
    faceCenterPx: faceCenterPx(turned),
    ringCenterPx: faceCenterPx("F"),
  };
}

// ---------------------------------------------------------------------
// Easing
// ---------------------------------------------------------------------

// Quadratic ease-out: `1 - (1 - t)^2`. Maps [0, 1] → [0, 1] with fast
// start and gentle landing — matches the rev5 forward-leg feel.
export function easeOut(t: number): number {
  const u = 1 - t;
  return 1 - u * u;
}

// ---------------------------------------------------------------------
// Per-sticker visual model — depart / arrive copies
// ---------------------------------------------------------------------

// A single visible sticker rect within an animated group.
//
// `x`, `y` are the sticker's POSITION AT PROGRESS=0, in integer
// sticker-grid coords (multiply by STICKER_PX for SVG viewBox coords).
// `color` is the facelet letter ('U' | 'R' | 'F' | 'D' | 'L' | 'B') —
// renderers map to palette colors.
//
// Why a letter (not a hex code): the production renderer applies a
// theme/palette swap; carrying the letter keeps `cube2DKinematics.ts`
// palette-agnostic and matches `Cube2D`'s static-mode data flow.
export interface AnimatedSticker {
  x: number;
  y: number;
  color: FaceLetter;
}

// Rotation group — either the 9 turned-face stickers or the 12 F/B
// ring-cycle stickers. Renders ABOVE the inverse-opacity overlay and
// OUTSIDE the cross-silhouette clip — rotation swing corners are
// never clipped or dimmed.
export interface RotationGroup {
  kind: "rotation";
  // What kind of rotation: `face` = 9 turned-face stickers around the
  // face center; `ring` = 12 F/B edge-cycle stickers around F-center.
  rotationKind: "face" | "ring";
  // Stickers in this group, positioned at their PROGRESS=0 sticker
  // coords. The renderer applies a single `rotate(angleDeg pivot)`
  // transform to the whole group.
  stickers: AnimatedSticker[];
  pivotPx: { x: number; y: number };
  // Target angle at progress=1, degrees. The renderer interpolates
  // 0 → angleDeg via the easing curve.
  angleDeg: number;
  // false: rotation groups never get clip-path applied (they swing
  // through cross corners and must remain fully visible).
  clipped: false;
}

// Slide group — primary or secondary ribbon. Each group contains
// (depart + arrive) sticker pairs at their canonical positions, then
// translates as a rigid block.
//
// For h-ribbon (U/D): one primary group with 24 stickers
// (12 depart + 12 arrive). No secondary group.
//
// For v-ribbon (R/L): a primary group with 18 stickers (9 depart +
// 9 arrive) at the U/F/D column, plus a secondary group with 6
// stickers (3 depart + 3 arrive) at B's near-F column. Secondary's
// `slideVector` is the NEGATION of primary's (`-sv`) — the wrap
// content over the top/bottom of the cube projects onto B from the
// opposite direction in the 2D net.
export interface SlideGroup {
  kind: "slide";
  role: "primary" | "secondary";
  // Stickers in this group, positioned at their PROGRESS=0 sticker
  // coords. Renderer applies a single `translate(tx ty)` transform
  // computed as `slideVector * STICKER_PX * easeOut(progress)`.
  stickers: AnimatedSticker[];
  // Per-step slide vector (sticker units). Renderer multiplies by
  // STICKER_PX and the eased progress to get the px translate.
  slideVector: SlideVector;
  // true by default (cross-silhouette clip-path applied). The
  // renderer flips this when "Show extensions" debug mode is on.
  clipped: true;
}

export type AnimatedGroup = RotationGroup | SlideGroup;

// ---------------------------------------------------------------------
// getRenderInstructions — full render plan for one move at one
// progress value. Pure: depends only on (preFacelet, move, progress).
// ---------------------------------------------------------------------

// One full render plan for the cross at a single moment in time.
//
// The renderer (A·P3) consumes this by:
//   - Drawing `staticStickers` at fixed positions (the 33 untouched
//     stickers + 21 face-rotation-group placeholders — the rotation
//     stickers' colors are also drawn into the static layer at
//     progress=0 positions for completeness; the rotation transform
//     visually overrides them).
//   - Drawing each `groups[i]` according to its kind:
//       rotation → group transform = rotate(angleDeg * easeOut(progress) pivot)
//       slide    → group transform = translate(sv * STICKER_PX * easeOut(progress))
//   - Applying the cross-silhouette clip-path to slide groups when
//     `slideGroup.clipped === true` (production view default).
//   - Applying the inverse-opacity dimming overlay when extensions
//     are visible (debug mode); `overlayOpacity` controls intensity.
//
// The plan is "raw kinematic", i.e. the angle and slide-vector are
// the TARGET values at progress=1; the renderer interpolates by
// applying `easeOut(progress)`. This matches the preview's render
// pipeline and keeps the easing curve in one place (the renderer's
// transform) rather than baked into the plan.
export interface RenderPlan {
  // Pre-move facelet — colors for static stickers and the turned-face
  // rotation group come from here.
  preFacelet: string;
  // Move being animated.
  move: MoveStr;
  // Current progress in [0, 1]. The plan's group transforms are
  // unaffected by progress (the renderer applies it); progress is
  // recorded here for renderer reference and identity-test contracts.
  progress: number;
  // The 33 stickers not touched by the move, at their canonical
  // positions, colored from `preFacelet`.
  staticStickers: AnimatedSticker[];
  // Animated groups (rotation + slide). Order is canonical:
  //   1. Slide groups (primary, then secondary if present)
  //   2. Face rotation group
  //   3. Ring rotation group (only for F/B)
  // Renderer z-order: slide groups render UNDER the overlay; rotation
  // groups render ABOVE. The order here makes that explicit.
  groups: AnimatedGroup[];
  // Inverse-opacity dimming overlay opacity. Zero by default
  // (production view, extensions clipped). The widget / debug toggle
  // sets to a small value (~0.7) when extensions are visible.
  // The renderer is responsible for the actual mask geometry; this
  // value just gates display.
  overlayOpacity: number;
}

export interface GetRenderInstructionsOpts {
  /** Override overlay opacity (debug mode). Default 0. */
  overlayOpacity?: number;
}

export function getRenderInstructions(
  preFacelet: string,
  move: MoveStr,
  progress: number,
  opts: GetRenderInstructionsOpts = {},
): RenderPlan {
  if (preFacelet.length !== 54) {
    throw new Error(
      `preFacelet must be 54 chars, got ${preFacelet.length}`,
    );
  }
  if (!Number.isFinite(progress)) {
    throw new Error(`progress must be finite, got ${progress}`);
  }
  // Clamp out-of-range progress to [0, 1] — caller might pass slightly
  // out-of-band values from rAF jitter and we shouldn't blow up.
  const p = Math.max(0, Math.min(1, progress));

  const perm = FACELET_MOVES[move];
  if (!perm) throw new Error(`unknown move: ${move}`);

  const kin = computeMoveKinematics(move);
  const { faceIndices, edgeIndices, staticIndices } =
    participatingForMove(move);

  // Static layer: the 33 untouched stickers.
  const staticStickers: AnimatedSticker[] = staticIndices.map((i) => {
    const pos = indexToXY(i);
    return {
      x: pos.x,
      y: pos.y,
      color: preFacelet.charAt(i) as FaceLetter,
    };
  });

  // Face rotation group — the 9 turned-face stickers, drawn at their
  // canonical positions, rotated as a block.
  const faceStickers: AnimatedSticker[] = faceIndices.map((i) => {
    const pos = indexToXY(i);
    return {
      x: pos.x,
      y: pos.y,
      color: preFacelet.charAt(i) as FaceLetter,
    };
  });
  const faceGroup: RotationGroup = {
    kind: "rotation",
    rotationKind: "face",
    stickers: faceStickers,
    pivotPx: kin.faceCenterPx,
    angleDeg: kin.faceRotationAngle,
    clipped: false,
  };

  const groups: AnimatedGroup[] = [];
  // SlideGroup.clipped is a `true` literal in the type — the
  // "show extensions" debug toggle is the renderer's responsibility
  // (it flips the clip-path attribute at render time based on user
  // state). The plan reflects production semantics: slides clipped,
  // rotations unclipped. Debug mode is controlled via the renderer's
  // own prop + the `overlayOpacity` field below.

  if (kin.kind === "ring") {
    // F/B ring rotation. No slide groups; one ring group.
    const ringStickers: AnimatedSticker[] = edgeIndices.map((i) => {
      const pos = indexToXY(i);
      return {
        x: pos.x,
        y: pos.y,
        color: preFacelet.charAt(i) as FaceLetter,
      };
    });
    const ringGroup: RotationGroup = {
      kind: "rotation",
      rotationKind: "ring",
      stickers: ringStickers,
      pivotPx: kin.ringCenterPx,
      angleDeg: kin.ringRotationAngle,
      clipped: false,
    };
    groups.push(faceGroup, ringGroup);
  } else {
    // h-ribbon (U/D) or v-ribbon (R/L). Build slide groups.
    const sv = kin.slideVector;
    const primaryItems: AnimatedSticker[] = [];
    const secondaryItems: AnimatedSticker[] = [];
    const secondarySv: SlideVector = { dx: -sv.dx, dy: -sv.dy };

    for (const i of edgeIndices) {
      const face = indexFace(i);
      const post = indexToXY(i);
      // depart copy: at i's canonical position, colored from preFacelet[i].
      const departX = post.x;
      const departY = post.y;
      const departColor = preFacelet.charAt(i) as FaceLetter;

      if (kin.kind === "v-ribbon" && face === "B") {
        // Secondary ribbon. Arrival placed at (post - secondarySv) so
        // it lands at post after translating by secondarySv.
        const arriveX = post.x - secondarySv.dx;
        const arriveY = post.y - secondarySv.dy;
        const arriveColor = preFacelet.charAt(perm[i]) as FaceLetter;
        secondaryItems.push(
          { x: departX, y: departY, color: departColor },
          { x: arriveX, y: arriveY, color: arriveColor },
        );
      } else {
        // Primary ribbon. Natural slide.
        const arriveX = post.x - sv.dx;
        const arriveY = post.y - sv.dy;
        const arriveColor = preFacelet.charAt(perm[i]) as FaceLetter;
        primaryItems.push(
          { x: departX, y: departY, color: departColor },
          { x: arriveX, y: arriveY, color: arriveColor },
        );
      }
    }

    const primaryGroup: SlideGroup = {
      kind: "slide",
      role: "primary",
      stickers: primaryItems,
      slideVector: sv,
      clipped: true,
    };
    groups.push(primaryGroup);
    if (secondaryItems.length > 0) {
      const secondaryGroup: SlideGroup = {
        kind: "slide",
        role: "secondary",
        stickers: secondaryItems,
        slideVector: secondarySv,
        clipped: true,
      };
      groups.push(secondaryGroup);
    }
    groups.push(faceGroup);
  }

  return {
    preFacelet,
    move,
    progress: p,
    staticStickers,
    groups,
    overlayOpacity: opts.overlayOpacity ?? 0,
  };
}

// ---------------------------------------------------------------------
// Identity utility — convenience wrapper for tests.
//
// `appliedFaceletAt(preFacelet, move, 1)` should equal
// `applyMove(preFacelet, move)`. Provided as an exported helper so
// downstream consumers (and the test contract) don't have to import
// `applyMove` separately to assert the rev5 visual-model identity.
// ---------------------------------------------------------------------

/**
 * The facelet state after a move has fully completed. Equivalent to
 * `applyMove(preFacelet, move)`. Provided as a re-export so the
 * "post-progress=1 ↔ applyMove" identity can be tested without
 * importing `applyMove` separately at the consumer.
 */
export function appliedFaceletAt(preFacelet: string, move: MoveStr): string {
  return applyMove(preFacelet, move);
}
