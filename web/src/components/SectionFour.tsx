// Section iv — "watch the solve". Renders an animated <twisty-player>
// below the solution card grid once a solution exists. Conditional render:
// returns null while solutionAlg is empty (no solve yet, or just cleared)
// so the section is fully absent — not a dim placeholder — until there's
// something to play.
//
// Caller (App.tsx) joins scrambleMoves and solution into space-separated
// algs; this component is state-agnostic and treats empty solutionAlg as
// "nothing to render". Per-card animations + cross-surface scrubber sync
// are 1B/1C territory; here we ship twisty-player's stock bottom-row
// control panel and call it done for 1A.
import SectionHeader from "./SectionHeader";
import TwistyPlayerWrapper from "./TwistyPlayerWrapper";

type Props = {
  scrambleAlg: string;
  solutionAlg: string;
};

export default function SectionFour({ scrambleAlg, solutionAlg }: Props) {
  if (solutionAlg.trim() === "") return null;

  return (
    <>
      <SectionHeader roman="iv." name="watch the solve" />
      <div className="section-four-stage" data-testid="section-four">
        <TwistyPlayerWrapper
          mode="animated"
          scrambleAlg={scrambleAlg}
          solutionAlg={solutionAlg}
          sizePx={360}
          testId="twisty-section-four"
        />
      </div>
    </>
  );
}
