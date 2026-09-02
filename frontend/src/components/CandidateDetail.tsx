import type { AnalyzeResponse, ViewMode } from "../types";
import { computeVerdict } from "../lib/verdict";
import VerdictCard from "./VerdictCard";
import SkillsTable from "./SkillsTable";
import ResumeDrilldown from "./ResumeDrilldown";

interface Props {
  result: AnalyzeResponse;
  view: ViewMode;
  title?: string;
}

export default function CandidateDetail({ result, view, title }: Props) {
  const verdict = computeVerdict(result.skills);
  const showDrilldown = view === "recruiter" && verdict.recommended;

  return (
    <div className="flex flex-col gap-6">
      {title && <h3 className="text-base font-semibold text-slate-100">{title}</h3>}

      <VerdictCard verdict={verdict} view={view} />

      {result.github_note && (
        <div className="rounded-lg border border-white/[0.06] bg-white/[0.03] px-3.5 py-2.5 text-[13px] text-slate-400">
          {result.github_note}
        </div>
      )}

      <SkillsTable skills={result.skills} />

      {showDrilldown && (
        <ResumeDrilldown claims={result.claims} evidenceBullets={result.evidence_bullets} />
      )}
    </div>
  );
}
