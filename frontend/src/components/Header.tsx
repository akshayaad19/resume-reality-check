import type { ViewMode } from "../types";

interface Props {
  view: ViewMode;
  onChange: (view: ViewMode) => void;
}

export default function Header({ view, onChange }: Props) {
  return (
    <header className="sticky top-0 z-10 border-b border-white/[0.06] bg-[#0a0e1a]/80 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-500 text-sm font-bold text-white shadow-[0_0_16px_rgba(99,102,241,0.45)]">
            R
          </div>
          <span className="text-[15px] font-semibold tracking-tight text-slate-100">
            Resume Reality Check
          </span>
        </div>

        <div
          role="tablist"
          aria-label="View mode"
          className="flex gap-1 rounded-full border border-white/[0.06] bg-white/[0.04] p-1"
        >
          {(["recruiter", "candidate"] as ViewMode[]).map((mode) => (
            <button
              key={mode}
              role="tab"
              aria-selected={view === mode}
              onClick={() => onChange(mode)}
              className={`rounded-full px-4 py-1.5 text-[13px] font-semibold transition-colors ${
                view === mode
                  ? "bg-accent-500 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {mode === "recruiter" ? "Recruiter view" : "Candidate view"}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
}
