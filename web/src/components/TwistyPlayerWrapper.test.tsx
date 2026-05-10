import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import TwistyPlayerWrapper from "./TwistyPlayerWrapper";
import { buildTwistyPlayerConfig } from "./twistyPlayerConfig";

// The wrapper's contract surface has two parts:
//
//   1. The imperative config passed to `new TwistyPlayer(...)` — covered by
//      testing `buildTwistyPlayerConfig` directly. We don't try to render
//      cubing.js's custom element under vitest's `node` environment; the
//      `useEffect` body never runs during SSR, which is fine.
//
//   2. The wrapper's outer DOM (testid + static-mode CSS). Covered via
//      `renderToStaticMarkup`, mirroring IsoCubeRenderer.test.ts.

function render(props: {
  scrambleAlg?: string;
  solutionAlg?: string;
  sizePx?: number;
  mode: "static" | "animated";
  testId?: string | null;
}): string {
  return renderToStaticMarkup(
    createElement(TwistyPlayerWrapper, {
      scrambleAlg: "",
      solutionAlg: "",
      sizePx: 130,
      ...props,
    }),
  );
}

describe("buildTwistyPlayerConfig — imperative config snapshot", () => {
  it("static mode sets tempoScale=0, controlPanel='none'", () => {
    const cfg = buildTwistyPlayerConfig({
      scrambleAlg: "R U R'",
      solutionAlg: "R U' R'",
      mode: "static",
    });
    expect(cfg.tempoScale).toBe(0);
    expect(cfg.controlPanel).toBe("none");
  });

  it("animated mode keeps tempoScale=1, controlPanel='bottom-row'", () => {
    const cfg = buildTwistyPlayerConfig({
      scrambleAlg: "R U R'",
      solutionAlg: "R U' R'",
      mode: "animated",
    });
    expect(cfg.tempoScale).toBe(1);
    expect(cfg.controlPanel).toBe("bottom-row");
  });

  it("anchor is always 'end' so display = applyMove(scramble, alg)", () => {
    // The static-mode recipe from .planning/research/m9-twisty-player.md §3.
    // anchor=end is the load-bearing piece; both modes need it for the
    // static-frame and the animated-from-end-of-setup contracts.
    const staticCfg = buildTwistyPlayerConfig({
      scrambleAlg: "R U R'",
      solutionAlg: "",
      mode: "static",
    });
    const animatedCfg = buildTwistyPlayerConfig({
      scrambleAlg: "R U R'",
      solutionAlg: "R U' R'",
      mode: "animated",
    });
    expect(staticCfg.experimentalSetupAnchor).toBe("end");
    expect(animatedCfg.experimentalSetupAnchor).toBe("end");
  });

  it("threads scrambleAlg + solutionAlg into setupAlg + alg", () => {
    const cfg = buildTwistyPlayerConfig({
      scrambleAlg: "F U2 L2 B'",
      solutionAlg: "B L2 U2 F'",
      mode: "static",
    });
    expect(cfg.experimentalSetupAlg).toBe("F U2 L2 B'");
    expect(cfg.alg).toBe("B L2 U2 F'");
  });

  it("pins puzzle=3x3x3, background/backView=none, hintFacelets=none", () => {
    // These don't change per mode — they're "match the existing flat look,
    // single front view, save a renderer slot" defaults.
    const cfg = buildTwistyPlayerConfig({
      scrambleAlg: "",
      solutionAlg: "",
      mode: "static",
    });
    expect(cfg.puzzle).toBe("3x3x3");
    expect(cfg.background).toBe("none");
    expect(cfg.backView).toBe("none");
    expect(cfg.hintFacelets).toBe("none");
  });
});

describe("TwistyPlayerWrapper — outer DOM (SSR markup)", () => {
  it("static mode applies pointer-events: none (drag-rotate disabled)", () => {
    const html = render({ mode: "static" });
    expect(html).toMatch(/pointer-events:\s*none/);
  });

  it("animated mode does NOT set pointer-events: none", () => {
    const html = render({ mode: "animated" });
    expect(html).not.toMatch(/pointer-events:\s*none/);
  });
});

describe("TwistyPlayerWrapper — testId behavior", () => {
  it("defaults data-testid to 'twisty-cube'", () => {
    const html = render({ mode: "static" });
    expect(html).toMatch(/data-testid="twisty-cube"/);
  });

  it("accepts a custom testId override", () => {
    const html = render({ mode: "static", testId: "twisty-cube-pair" });
    expect(html).toMatch(/data-testid="twisty-cube-pair"/);
    expect(html).not.toMatch(/data-testid="twisty-cube"(?!-)/);
  });

  it("suppresses data-testid when testId is null", () => {
    const html = render({ mode: "static", testId: null });
    expect(html).not.toMatch(/data-testid=/);
  });
});

describe("TwistyPlayerWrapper — smoke", () => {
  it("renders without throwing for static + animated modes", () => {
    expect(() => render({ mode: "static" })).not.toThrow();
    expect(() => render({ mode: "animated" })).not.toThrow();
  });
});
