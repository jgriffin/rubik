import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import IsoCubeRenderer from "./IsoCubeRenderer";

const SOLVED =
  "UUUUUUUUU" +
  "RRRRRRRRR" +
  "FFFFFFFFF" +
  "DDDDDDDDD" +
  "LLLLLLLLL" +
  "BBBBBBBBB";

// Match every <polygon ... /> tag and pull out attributes via simple regex.
// The renderer emits self-closing polygon tags via React's static markup,
// so the parser is tiny.
type Poly = {
  fill: string | null;
  stroke: string | null;
  dataColor: string | null;
  dataFace: string | null;
  strokeWidth: string | null;
};

function attr(tag: string, name: string): string | null {
  const re = new RegExp(`\\b${name}="([^"]*)"`);
  const m = tag.match(re);
  return m ? m[1] : null;
}

function extractPolygons(svg: string): Poly[] {
  const out: Poly[] = [];
  const re = /<polygon\b[^>]*\/?>/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(svg)) !== null) {
    const tag = m[0];
    out.push({
      fill: attr(tag, "fill"),
      stroke: attr(tag, "stroke"),
      dataColor: attr(tag, "data-color"),
      dataFace: attr(tag, "data-face"),
      strokeWidth: attr(tag, "stroke-width"),
    });
  }
  return out;
}

function render(props: {
  facelet: string;
  sizePx?: number;
  testId?: string | null;
}): string {
  return renderToStaticMarkup(createElement(IsoCubeRenderer, props));
}

describe("IsoCubeRenderer — polygon counts + face attributes", () => {
  it("renders exactly 27 polygons (3 visible faces × 9 stickers)", () => {
    const svg = render({ facelet: SOLVED });
    const polys = extractPolygons(svg);
    expect(polys).toHaveLength(27);
  });

  it("each visible face contributes 9 polygons (data-face = U / F / R)", () => {
    const svg = render({ facelet: SOLVED });
    const polys = extractPolygons(svg);
    const byFace: Record<string, number> = {};
    for (const p of polys) {
      const face = p.dataFace ?? "";
      byFace[face] = (byFace[face] ?? 0) + 1;
    }
    expect(byFace).toEqual({ U: 9, F: 9, R: 9 });
  });

  it("does NOT emit polygons for hidden faces (D, L, B)", () => {
    const svg = render({ facelet: SOLVED });
    const polys = extractPolygons(svg);
    for (const p of polys) {
      expect(p.dataFace).toMatch(/^[UFR]$/);
    }
  });
});

describe("IsoCubeRenderer — solved-cube center stickers carry expected colors", () => {
  it("U center sticker (offset 0+4) has data-color='U'", () => {
    // U face polygons appear first (render order), and within the face,
    // r=1, c=1 is the 5th polygon (index 4) — same r*3+c ordering as the loop.
    const svg = render({ facelet: SOLVED });
    const polys = extractPolygons(svg);
    const uPolys = polys.filter((p) => p.dataFace === "U");
    expect(uPolys[4].dataColor).toBe("U");
  });

  it("R center sticker (offset 9+4) has data-color='R'", () => {
    const svg = render({ facelet: SOLVED });
    const polys = extractPolygons(svg);
    const rPolys = polys.filter((p) => p.dataFace === "R");
    expect(rPolys[4].dataColor).toBe("R");
  });

  it("F center sticker (offset 18+4) has data-color='F'", () => {
    const svg = render({ facelet: SOLVED });
    const polys = extractPolygons(svg);
    const fPolys = polys.filter((p) => p.dataFace === "F");
    expect(fPolys[4].dataColor).toBe("F");
  });
});

describe("IsoCubeRenderer — facelet validation", () => {
  it("throws for a 53-char facelet (mentions '54')", () => {
    expect(() => render({ facelet: SOLVED.slice(0, 53) })).toThrowError(
      /54/,
    );
  });

  it("throws for an empty facelet", () => {
    expect(() => render({ facelet: "" })).toThrowError(/54/);
  });

  it("throws for a 55-char facelet", () => {
    expect(() => render({ facelet: SOLVED + "U" })).toThrowError(/54/);
  });
});

describe("IsoCubeRenderer — testId behavior", () => {
  it("defaults data-testid to 'iso-cube'", () => {
    const svg = render({ facelet: SOLVED });
    expect(svg).toMatch(/data-testid="iso-cube"/);
  });

  it("accepts a custom testId override", () => {
    const svg = render({ facelet: SOLVED, testId: "iso-cube-pair" });
    expect(svg).toMatch(/data-testid="iso-cube-pair"/);
    expect(svg).not.toMatch(/data-testid="iso-cube"(?!-)/);
  });

  it("suppresses data-testid when testId is null", () => {
    const svg = render({ facelet: SOLVED, testId: null });
    expect(svg).not.toMatch(/data-testid=/);
  });
});

describe("IsoCubeRenderer — tinted face fills", () => {
  it("F face fill is chrome F #5cb27a tinted by 0.96 → #58ab75", () => {
    // 92*0.96=88.32→88 (0x58); 178*0.96=170.88→171 (0xab); 122*0.96=117.12→117 (0x75).
    const svg = render({ facelet: SOLVED });
    const polys = extractPolygons(svg);
    const fPolys = polys.filter((p) => p.dataFace === "F");
    for (const p of fPolys) {
      expect(p.fill).toBe("#58ab75");
    }
  });

  it("R face fill is chrome R #d24244 tinted by 0.88 → #b93a3c", () => {
    // 210*0.88=184.8→185 (0xb9); 66*0.88=58.08→58 (0x3a); 68*0.88=59.84→60 (0x3c).
    const svg = render({ facelet: SOLVED });
    const polys = extractPolygons(svg);
    const rPolys = polys.filter((p) => p.dataFace === "R");
    for (const p of rPolys) {
      expect(p.fill).toBe("#b93a3c");
    }
  });

  it("U face fill is chrome U #f5f3ee at full chroma (no tint)", () => {
    const svg = render({ facelet: SOLVED });
    const polys = extractPolygons(svg);
    const uPolys = polys.filter((p) => p.dataFace === "U");
    for (const p of uPolys) {
      expect(p.fill).toBe("#f5f3ee");
    }
  });
});

describe("IsoCubeRenderer — stroke + structural attrs", () => {
  it("uses ink stroke #131210 on every polygon", () => {
    const svg = render({ facelet: SOLVED });
    const polys = extractPolygons(svg);
    for (const p of polys) {
      expect(p.stroke).toBe("#131210");
    }
  });

  it("emits a square SVG footprint with width=height=sizePx", () => {
    const svg = render({ facelet: SOLVED, sizePx: 180 });
    expect(svg).toMatch(/width="180"/);
    expect(svg).toMatch(/height="180"/);
  });
});
