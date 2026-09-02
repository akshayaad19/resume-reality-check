import type { SkillRow } from "../types";

interface Props {
  skills: SkillRow[];
}

const SCORE_COLORS = ["bg-slate-600", "bg-amber-400", "bg-sky-400", "bg-emerald-400"];

function ScoreBar({ score }: { score: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex gap-[3px]">
        {[1, 2, 3].map((seg) => (
          <span
            key={seg}
            className={`h-2 w-4 rounded-sm ${seg <= score ? SCORE_COLORS[score] : "bg-white/[0.08]"}`}
          />
        ))}
      </div>
      <span className="text-[13px] font-semibold text-slate-300">{score}/3</span>
    </div>
  );
}

export default function SkillsTable({ skills }: Props) {
  const ordered = [...skills].sort((a, b) => {
    if (a.category !== b.category) return a.category === "required" ? -1 : 1;
    return b.evidence_score - a.evidence_score;
  });

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-[#111726] p-6 shadow-card">
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-base font-semibold text-slate-100">Skill match breakdown</h3>
        <div className="flex gap-3 text-xs text-slate-400">
          {["No evidence", "Weak", "Solid", "Strong"].map((label, i) => (
            <span key={label} className="flex items-center gap-1.5">
              <span className={`h-2 w-2 rounded-full ${SCORE_COLORS[i]}`} />
              {i} — {label}
            </span>
          ))}
        </div>
      </div>

      <div className="-mx-6 overflow-x-auto px-6">
        <table className="w-full min-w-[640px] border-collapse text-sm">
          <thead>
            <tr className="border-b-2 border-white/[0.06] text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
              <th className="py-2.5 pr-3">Skill</th>
              <th className="py-2.5 pr-3">Type</th>
              <th className="py-2.5 pr-3">Claimed</th>
              <th className="py-2.5 pr-3">Evidence</th>
              <th className="py-2.5">Why</th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((row) => (
              <tr key={row.skill} className="border-b border-white/[0.05] align-top last:border-0">
                <td className="py-3 pr-3 font-semibold text-slate-100">{row.skill}</td>
                <td className="py-3 pr-3">
                  <span
                    className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wide ${
                      row.category === "required"
                        ? "bg-amber-400/10 text-amber-300"
                        : "bg-sky-400/10 text-sky-300"
                    }`}
                  >
                    {row.category}
                  </span>
                </td>
                <td className="py-3 pr-3">
                  <span
                    className={`text-[13px] font-semibold ${
                      row.claimed_on_resume ? "text-emerald-400" : "text-slate-500"
                    }`}
                  >
                    {row.claimed_on_resume ? "Claimed" : "Not claimed"}
                  </span>
                </td>
                <td className="py-3 pr-3">
                  <ScoreBar score={row.evidence_score} />
                </td>
                <td className="max-w-xs py-3 text-slate-400">
                  {row.justification}
                  {row.cited_evidence && (
                    <span className="mt-1 block border-l-2 border-white/[0.1] pl-2 italic text-slate-300">
                      "{row.cited_evidence}"
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
