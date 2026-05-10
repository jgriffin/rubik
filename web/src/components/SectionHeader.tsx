import type { ReactNode } from "react";

type Props = {
  roman: string;
  name: string;
  right?: ReactNode;
};

export default function SectionHeader({ roman, name, right }: Props) {
  return (
    <div className="lbl">
      <span className="roman serif">{roman}</span>
      <span className="name">{name}</span>
      {right && <span className="right">{right}</span>}
    </div>
  );
}
