// Subset of SVG `dominant-baseline` values we actually use. Keeps the
// React JSX prop type happy (it expects this enum, not `string`).
export type OverlayBaseline =
  | "auto"
  | "central"
  | "middle"
  | "hanging"
  | "alphabetic"
  | "ideographic"
  | "mathematical"
  | "text-after-edge"
  | "text-before-edge";

export type Overlay = {
  x: number; // SVG-units (already in pixels per current code)
  y: number;
  text: string;
  fontSize: number;
  weight?: number; // default 500
  opacity?: number; // default 1
  anchor?: "start" | "middle" | "end"; // default "middle"
  baseline?: OverlayBaseline; // default "central"
};

type Props = {
  facelet: string;
  sizePx?: number;
  // Default `"flat-cube"`. Pass `null` to omit the testid entirely
  // (used by MoveStripView so per-cell renderers don't collide with
  // the contract-bearing big secondary cube).
  testId?: string | null;
  // Optional <text> overlays drawn on top of the sticker grid. Used by
  // MoveStripView to render move-label + step# inside the cube SVG.
  overlays?: ReadonlyArray<Overlay>;
};

const COLOR_FOR_LETTER: Record<string, string> = {
  U: "#ffffff",
  R: "#cc1f25",
  F: "#0eaf3f",
  D: "#ffd200",
  L: "#ff7423",
  B: "#1453d6",
};

// Indices into a kociemba 54-char facelet string for each face.
// Order in the string: URFDLB.
const FACE_OFFSETS: Record<string, number> = {
  U: 0,
  R: 9,
  F: 18,
  D: 27,
  L: 36,
  B: 45,
};

// Position of each face in the cross grid (columns 0..3, rows 0..2).
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
  overlays,
}: Props) {
  if (facelet.length !== 54) {
    throw new Error(
      `FlatCubeRenderer: facelet must be 54 chars, got ${facelet.length}`
    );
  }
  const stickerPx = sizePx / 9; // 9 sticker rows tall
  const widthPx = stickerPx * 12;
  const heightPx = stickerPx * 9;
  const gap = stickerPx * 0.06; // small black gap

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

  // testId === null → omit the data-testid entirely.
  // testId === undefined (default) → "flat-cube" for back-compat.
  // Any other string → use it.
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
          fill={s.color}
          stroke="#111"
          strokeWidth={Math.max(1, stickerPx * 0.04)}
          data-pos={s.pos}
          data-color={s.letter}
        />
      ))}
      {overlays?.map((o, i) => (
        <text
          key={i}
          x={o.x}
          y={o.y}
          fontSize={o.fontSize}
          fontWeight={o.weight ?? 500}
          opacity={o.opacity ?? 1}
          textAnchor={o.anchor ?? "middle"}
          dominantBaseline={o.baseline ?? "central"}
          fill="currentColor"
          fontFamily="system-ui, sans-serif"
          style={{ fontVariantNumeric: "tabular-nums" }}
        >
          {o.text}
        </text>
      ))}
    </svg>
  );
}
