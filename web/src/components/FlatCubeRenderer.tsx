// 4×3 net renderer (the cube unfolded in a cross / T shape).
// Chrome palette + ink stroke + subtle 1px corner radius — matches the
// Sequence design reference. SVG is rendered at the requested pixel size
// with display: block; parents control flow/centering.

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

// Position of each face in the cross grid (4 cols × 3 rows).
const FACE_GRID: Record<string, { col: number; row: number }> = {
  U: { col: 1, row: 0 },
  L: { col: 0, row: 1 },
  F: { col: 1, row: 1 },
  R: { col: 2, row: 1 },
  B: { col: 3, row: 1 },
  D: { col: 1, row: 2 },
};

export default function FlatCubeRenderer({
  facelet,
  sizePx = 240,
  testId,
}: Props) {
  if (facelet.length !== 54) {
    throw new Error(
      `FlatCubeRenderer: facelet must be 54 chars, got ${facelet.length}`,
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

  for (const face of Object.keys(FACE_OFFSETS) as Array<keyof typeof FACE_OFFSETS>) {
    const { col, row } = FACE_GRID[face];
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
