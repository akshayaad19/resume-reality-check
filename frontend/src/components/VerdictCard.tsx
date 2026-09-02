import type { ViewMode } from "../types";
import type { Verdict } from "../lib/verdict";
import { PROVEN_SCORE_THRESHOLD } from "../lib/verdict";

interface Props {
  verdict: Verdict;
  view: ViewMode;
}

export default function VerdictCard({ verdict, view }: Props) {
  const { recommended, proven, total, usedRequiredOnly } = verdict;
  const isRecruiter = view === "recruiter";
  const skillWord = usedRequiredOnly ? "required skills" : "skills";

  const badgeLabel = isRecruiter
    ? recommended
      ? "Recommended"
      : "Not recommended"
    : recommended
    ? "Strong match"
    : "Needs work";

  const headline = isRecruiter
    ? recommended
      ? "This candidate has provable evidence for most required skills."
      : "This candidate is missing evidence for key required skills."
    : recommended
    ? "Your resume backs up most of what this role needs — with real proof, not just mentions."
    : "Your resume doesn't yet prove enough of what this role needs.";

  return (
    <div
      className={`animate-rise flex flex-col gap-4 rounded-2xl border p-6 shadow-card sm:flex-row sm:items-center ${
        recommended
          ? "border-emerald-500/20 bg-emerald-500/[0.07]"
          : "border-rose-500/20 bg-rose-500/[0.07]"
      }`}
    >
      <div
        className={`inline-flex w-fit items-center gap-1.5 whitespace-nowrap rounded-full px-4 py-2 text-sm font-bold ${
          recommended ? "bg-emerald-500/15 text-emerald-400" : "bg-rose-500/15 text-rose-400"
        }`}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          {recommended ? (
            <path
              d="M5 13l4 4L19 7"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ) : (
            <path
              d="M6 6l12 12M18 6L6 18"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
        </svg>
        {badgeLabel}
      </div>

      <div>
        <div className="text-[15px] font-semibold text-slate-100">{headline}</div>
        <div className="mt-0.5 text-sm text-slate-400">
          {proven} of {total} {skillWord} have solid or strong evidence (score ≥{" "}
          {PROVEN_SCORE_THRESHOLD}/3).
        </div>
      </div>
    </div>
  );
}
