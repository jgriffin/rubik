// SolverSwitch tests — presentation contract only (SSR-static).
// Project has no @testing-library/react / jsdom; we render with
// `renderToStaticMarkup` and assert the rendered HTML reflects the
// `value` prop. Click-handler invocation is exercised at the e2e
// layer (Playwright) where a real DOM is available.

import { describe, it, expect, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import SolverSwitch from "./SolverSwitch";

describe("SolverSwitch presentation", () => {
  it("renders both api and onnx options", () => {
    const html = renderToStaticMarkup(
      createElement(SolverSwitch, { value: "api", onChange: vi.fn() }),
    );
    expect(html).toContain('data-testid="solver-api"');
    expect(html).toContain('data-testid="solver-onnx"');
    expect(html).toContain(">api</button>");
    expect(html).toContain(">onnx</button>");
  });

  it("applies the 'on' class to the active option (api)", () => {
    const html = renderToStaticMarkup(
      createElement(SolverSwitch, { value: "api", onChange: vi.fn() }),
    );
    // api button: aria-pressed=true and class includes "on".
    expect(html).toMatch(
      /<button[^>]*data-testid="solver-api"[^>]*class="on"[^>]*aria-pressed="true"/,
    );
    // onnx button: aria-pressed=false and class is empty.
    expect(html).toMatch(
      /<button[^>]*data-testid="solver-onnx"[^>]*class=""[^>]*aria-pressed="false"/,
    );
  });

  it("applies the 'on' class to the active option (onnx)", () => {
    const html = renderToStaticMarkup(
      createElement(SolverSwitch, { value: "onnx", onChange: vi.fn() }),
    );
    expect(html).toMatch(
      /<button[^>]*data-testid="solver-onnx"[^>]*class="on"[^>]*aria-pressed="true"/,
    );
    expect(html).toMatch(
      /<button[^>]*data-testid="solver-api"[^>]*class=""[^>]*aria-pressed="false"/,
    );
  });

  it("uses the col-seg-inline container class (matches RenderModeSwitch)", () => {
    const html = renderToStaticMarkup(
      createElement(SolverSwitch, { value: "api", onChange: vi.fn() }),
    );
    expect(html).toContain('class="col-seg-inline"');
    expect(html).toContain('data-testid="solver-switch"');
  });
});
