// Section iii's auto-play trigger. Lives alongside the render-mode +
// view-layout switches in the section iii header. Drives App's
// `isPlaying` state; the auto-advance timer (also in App) ticks
// `activeIdx` forward at the cadence in `cubeStageConstants.ts`.
// Visual animation richness is mode-dependent (cube + 2D animates
// per step; 3D + column modes snap), but the playback chain itself
// is mode-agnostic.

type Props = {
  isPlaying: boolean;
  onToggle: () => void;
  disabled?: boolean;
};

export default function PlayPauseButton({
  isPlaying,
  onToggle,
  disabled,
}: Props) {
  return (
    <button
      type="button"
      className="play-pause-btn text-action"
      data-testid="play-pause-button"
      onClick={onToggle}
      disabled={disabled}
      aria-label={isPlaying ? "pause" : "play"}
    >
      {isPlaying ? "pause" : "play"}
    </button>
  );
}
