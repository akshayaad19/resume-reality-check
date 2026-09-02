import type { Candidate } from "../types";
import { computeVerdict } from "../lib/verdict";

interface Props {
  candidates: Candidate[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

function displayName(fileName: string) {
  return fileName.replace(/\.[^./]+$/, "");
}

function StatusPill({ candidate }: { candidate: Candidate }) {
  if (candidate.status === "queued") {
    return <span className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Queued</span>;
  }
  if (candidate.status === "analyzing") {
    return (
      <span className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-accent-300">
        <svg className="h-3 w-3 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
        </svg>
        Analyzing
      </span>
    );
  }
  if (candidate.status === "error") {
    return <span className="text-[11px] font-semibold uppercase tracking-wide text-rose-400">Failed</span>;
  }
  if (!candidate.result) return null;
  const verdict = computeVerdict(candidate.result.skills);
  return (
    <span
      className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wide ${
        verdict.recommended ? "bg-emerald-500/15 text-emerald-400" : "bg-rose-500/15 text-rose-400"
      }`}
    >
      {verdict.recommended ? "Recommended" : "Not recommended"}
    </span>
  );
}

export default function CandidateRoster({ candidates, selectedId, onSelect }: Props) {
  const ranked = [...candidates].sort((a, b) => {
    const va = a.result ? computeVerdict(a.result.skills) : null;
    const vb = b.result ? computeVerdict(b.result.skills) : null;
    if (va && vb) {
      if (va.recommended !== vb.recommended) return va.recommended ? -1 : 1;
      return vb.ratio - va.ratio;
    }
    if (va) return -1;
    if (vb) return 1;
    return 0;
  });

  const recommendedCount = candidates.filter(
    (c) => c.result && computeVerdict(c.result.skills).recommended
  ).length;

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-[#111726] shadow-card">
      <div className="flex items-baseline justify-between border-b border-white/[0.06] px-5 py-4">
        <h3 className="text-sm font-semibold text-slate-100">Applicants</h3>
        <span className="text-xs text-slate-500">
          {recommendedCount} of {candidates.length} recommended
        </span>
      </div>

      <ul className="max-h-[70vh] divide-y divide-white/[0.05] overflow-y-auto">
        {ranked.map((candidate) => {
          const selected = candidate.id === selectedId;
          const verdict = candidate.result ? computeVerdict(candidate.result.skills) : null;
          return (
            <li key={candidate.id}>
              <button
                type="button"
                onClick={() => candidate.status === "done" && onSelect(candidate.id)}
                disabled={candidate.status !== "done"}
                className={`flex w-full flex-col gap-1 px-5 py-3.5 text-left transition-colors disabled:cursor-default ${
                  selected
                    ? "border-l-2 border-accent-400 bg-accent-500/[0.08]"
                    : "border-l-2 border-transparent hover:bg-white/[0.03]"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-sm font-medium text-slate-100">
                    {displayName(candidate.fileName)}
                  </span>
                  <StatusPill candidate={candidate} />
                </div>
                {verdict && (
                  <span className="text-xs text-slate-500">
                    {verdict.proven} of {verdict.total} required skills proven
                  </span>
                )}
                {candidate.status === "error" && (
                  <span className="text-xs text-rose-400/80">{candidate.error}</span>
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
