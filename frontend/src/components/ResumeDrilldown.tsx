import { useState } from "react";

interface Props {
  claims: string[];
  evidenceBullets: string[];
}

export default function ResumeDrilldown({ claims, evidenceBullets }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-[#111726] p-6 shadow-card">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 text-[13px] font-bold text-accent-400 hover:text-accent-300"
      >
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          className={`transition-transform ${open ? "rotate-90" : ""}`}
          aria-hidden="true"
        >
          <path
            d="M9 18l6-6-6-6"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        View candidate's resume detail
      </button>

      {open && (
        <div className="mt-5 grid animate-fade-in gap-6 sm:grid-cols-2">
          <div>
            <h4 className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-500">
              Claims (skills listed on resume)
            </h4>
            {claims.length === 0 ? (
              <p className="text-sm text-slate-500">No explicit skill claims found.</p>
            ) : (
              <ul className="space-y-1.5 text-sm text-slate-300">
                {claims.map((claim, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-slate-600">•</span>
                    {claim}
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div>
            <h4 className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-500">
              Evidence (what the resume actually describes)
            </h4>
            {evidenceBullets.length === 0 ? (
              <p className="text-sm text-slate-500">No evidence bullets extracted.</p>
            ) : (
              <ul className="space-y-1.5 text-sm text-slate-300">
                {evidenceBullets.map((bullet, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-slate-600">•</span>
                    {bullet}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
