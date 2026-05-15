// Animation + playback cadence constants shared between CubeStage
// (which owns the per-step 2D animation) and App (which now owns the
// auto-play timer driving activeIdx forward). Co-locating them keeps
// the playback cadence physically paired with the animation duration
// it rides on top of.

// Animation duration per move on the 2D cube path. Read by CubeStage's
// useCubeSequence spec; also read by App's auto-play timer so each
// step's animation has time to complete before the next is scheduled.
// Tuned for "you can follow the moves" — at 400ms the eye lags the
// animation; 700ms feels deliberate without dragging.
export const ANIM_MS_PER_MOVE = 700;

// Dwell between consecutive auto-play steps — a "settle" pause after
// a step's animation finishes before the next step is scheduled. The
// pattern the user asked for is animate → pause → animate → pause, so
// the dwell needs to feel like a deliberate beat, not an artifact.
// Total per-step duration = ANIM_MS_PER_MOVE + PLAY_STEP_DWELL_MS
// (700 + 350 = 1050ms per move during auto-play).
export const PLAY_STEP_DWELL_MS = 350;
