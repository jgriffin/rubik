import type { RenderMode } from "./SolutionGrid";

type Props = {
  value: RenderMode;
  onChange: (m: RenderMode) => void;
};

const MODES: Array<{ value: RenderMode; label: string; disabled: boolean; reason?: string }> = [
  { value: "net", label: "2D", disabled: false },
  { value: "iso", label: "3D", disabled: true, reason: "isometric renderer ships in v2" },
  { value: "dual", label: "split", disabled: true, reason: "split view ships in v2" },
];

export default function RenderModeSwitch({ value, onChange }: Props) {
  return (
    <span className="col-seg-inline" data-testid="render-mode-switch">
      {MODES.map((m) => (
        <button
          key={m.value}
          type="button"
          data-testid={`render-mode-${m.value}`}
          className={value === m.value ? "on" : ""}
          onClick={() => {
            if (!m.disabled) onChange(m.value);
          }}
          disabled={m.disabled}
          title={m.reason}
        >
          {m.label}
        </button>
      ))}
    </span>
  );
}
