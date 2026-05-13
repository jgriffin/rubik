// RenderModeSwitch — two independent toggle buttons (2D / 3D). Pressing
// one alone selects that renderer; pressing both gives the side-by-side
// split view. At least one must be pressed at all times.
//
// The App-level state still uses the RenderMode enum ("net" / "iso" /
// "dual") so consumers (SolutionCard, etc.) don't need to change.
// This component just provides a two-toggle UI on top of that enum.

import type { RenderMode } from "./SolutionGrid";

type Props = {
  value: RenderMode;
  onChange: (m: RenderMode) => void;
};

function compute(is2D: boolean, is3D: boolean): RenderMode {
  if (is2D && is3D) return "dual";
  if (is3D) return "iso";
  return "net";
}

export default function RenderModeSwitch({ value, onChange }: Props) {
  const is2D = value === "net" || value === "dual";
  const is3D = value === "iso" || value === "dual";

  function toggle2D() {
    const next2D = !is2D;
    if (!next2D && !is3D) return; // would leave both off
    onChange(compute(next2D, is3D));
  }
  function toggle3D() {
    const next3D = !is3D;
    if (!next3D && !is2D) return;
    onChange(compute(is2D, next3D));
  }

  return (
    <span className="col-seg-inline" data-testid="render-mode-switch">
      <button
        type="button"
        data-testid="render-mode-2d"
        className={is2D ? "on" : ""}
        onClick={toggle2D}
        aria-pressed={is2D}
      >
        2D
      </button>
      <button
        type="button"
        data-testid="render-mode-3d"
        className={is3D ? "on" : ""}
        onClick={toggle3D}
        aria-pressed={is3D}
      >
        3D
      </button>
    </span>
  );
}
