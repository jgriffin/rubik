// 3D iso renderer (U + F + R visible, hidden D/L/B faces dropped).
// 30° iso projection of a cube spanning (0,0,0)..(3,3,3) with x=right, y=up,
// z=toward viewer. Chrome palette + ink stroke matches FlatCubeRenderer;
// right + front faces are tinted (shadeRight=0.88, shadeFront=0.96), top
// stays full chroma. Math ported verbatim from the design handoff bundle's
// svgIso (cube.js:81-154).

import {
  COLOR_FOR_LETTER,
  FACE_OFFSETS,
  INK_STROKE,
} from "../styles/cubePalette";

type Props = {
  facelet: string;
  sizePx?: number;
  testId?: string | null;
};

const COS = Math.cos(Math.PI / 6);
const SIN = Math.sin(Math.PI / 6);

const SHADE_FRONT = 0.96;
const SHADE_RIGHT = 0.88;
// Top stays full chroma — no shading multiplier needed.

const GAP_MUL = 0.05;
const STROKE_MUL = 0.025;
// viewBox spans ~5 units across; pad is unit-space.

type Vec3 = readonly [number, number, number];

function project(p: Vec3): readonly [number, number] {
  return [(p[0] - p[2]) * COS, -p[1] + (p[0] + p[2]) * SIN];
}

// Multiply each RGB channel of a #rrggbb hex string by `mul`, round, re-encode.
// Matches cube.js:100-107 exactly.
function tint(hex: string, mul: number): string {
  const m = hex.match(/^#?([\da-f]{6})$/i);
  if (!m) return hex;
  const v = parseInt(m[1], 16);
  const r = Math.round(((v >> 16) & 255) * mul);
  const g = Math.round(((v >> 8) & 255) * mul);
  const b = Math.round((v & 255) * mul);
  return "#" + ((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1);
}

type FacePolygon = {
  faceLetter: string;
  letter: string;
  fill: string;
  points: string;
  key: string;
};

function buildFacePolygons(
  facelet: string,
  faceLetter: "U" | "F" | "R",
  cornerFn: (r: number, c: number) => Vec3[],
  shade: number,
): FacePolygon[] {
  const out: FacePolygon[] = [];
  const off = FACE_OFFSETS[faceLetter];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const letter = facelet[off + r * 3 + c];
      const baseFill = COLOR_FOR_LETTER[letter] ?? "#444";
      const fill = shade !== 1 ? tint(baseFill, shade) : baseFill;
      const corners = cornerFn(r, c).map(project);
      const points = corners
        .map((p) => p[0].toFixed(2) + "," + p[1].toFixed(2))
        .join(" ");
      out.push({
        faceLetter,
        letter,
        fill,
        points,
        key: `${faceLetter}-${r}-${c}`,
      });
    }
  }
  return out;
}

export default function IsoCubeRenderer({
  facelet,
  sizePx = 240,
  testId,
}: Props) {
  if (facelet.length !== 54) {
    throw new Error(
      `IsoCubeRenderer: facelet must be 54 chars, got ${facelet.length}`,
    );
  }

  // px per cubie edge in 3D space (s) — used only to convert gap-px to unit-space.
  const s = sizePx / 3;
  // stroke-width in unit space; strokeMul is unit-space (viewBox spans ~5 units).
  const sw = Math.max(0.012, STROKE_MUL);
  const gap = s * GAP_MUL; // gap between stickers, in 3D-space units (px)
  const g = gap / 2; // half-gap (px)
  const gu = g / s; // half-gap converted to unit space (cube spans 0..3)

  // F face: z=0, x: c→c+1, y: (3-r-1)→(3-r). Inset by gu in x and y.
  const fPolygons = buildFacePolygons(
    facelet,
    "F",
    (r, c) => {
      const x0 = c + gu;
      const x1 = c + 1 - gu;
      const y0 = 3 - r - 1 + gu;
      const y1 = 3 - r - gu;
      return [
        [x0, y1, 0],
        [x1, y1, 0],
        [x1, y0, 0],
        [x0, y0, 0],
      ];
    },
    SHADE_FRONT,
  );

  // R face: x=3, depth z: 0→-3. Cell at col c spans z from -c to -(c+1). y as F.
  const rPolygons = buildFacePolygons(
    facelet,
    "R",
    (r, c) => {
      const z0 = -c - gu; // nearer (less negative)
      const z1 = -(c + 1) + gu; // farther (more negative)
      const y0 = 3 - r - 1 + gu;
      const y1 = 3 - r - gu;
      return [
        [3, y1, z0],
        [3, y1, z1],
        [3, y0, z1],
        [3, y0, z0],
      ];
    },
    SHADE_RIGHT,
  );

  // U face: y=3, x: c→c+1, z: row r=0 is back (z=-3), r=2 is front (z=0).
  // Cell at row r spans z from -(3-r) to -(2-r).
  const uPolygons = buildFacePolygons(
    facelet,
    "U",
    (r, c) => {
      const x0 = c + gu;
      const x1 = c + 1 - gu;
      const z0 = -(3 - r) + gu; // farther (more negative)
      const z1 = -(2 - r) - gu; // nearer
      return [
        [x0, 3, z0],
        [x1, 3, z0],
        [x1, 3, z1],
        [x0, 3, z1],
      ];
    },
    1,
  );

  // viewBox: project the 8 cube corners and pad by sw * 4.
  const corners3: Vec3[] = [
    [0, 0, 0],
    [3, 0, 0],
    [3, 3, 0],
    [0, 3, 0],
    [0, 0, -3],
    [3, 0, -3],
    [3, 3, -3],
    [0, 3, -3],
  ];
  const projected = corners3.map(project);
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of projected) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const pad = sw * 4;
  const vbX = minX - pad;
  const vbY = minY - pad;
  const vbW = maxX - minX + pad * 2;
  const vbH = maxY - minY + pad * 2;

  // Render order matches cube.js:153 — `${U}${R}${F}`. F draws last (on top),
  // which is correct because F is the front-facing face in iso projection.
  const allPolygons = [...uPolygons, ...rPolygons, ...fPolygons];

  const testIdProp =
    testId === null ? {} : { "data-testid": testId ?? "iso-cube" };

  const swStr = sw.toFixed(3);

  return (
    <svg
      width={sizePx}
      height={sizePx}
      viewBox={`${vbX.toFixed(2)} ${vbY.toFixed(2)} ${vbW.toFixed(2)} ${vbH.toFixed(2)}`}
      preserveAspectRatio="xMidYMid meet"
      style={{ display: "block" }}
      {...testIdProp}
    >
      {allPolygons.map((p) => (
        <polygon
          key={p.key}
          points={p.points}
          fill={p.fill}
          stroke={INK_STROKE}
          strokeWidth={swStr}
          strokeLinejoin="round"
          data-color={p.letter}
          data-face={p.faceLetter}
        />
      ))}
    </svg>
  );
}
