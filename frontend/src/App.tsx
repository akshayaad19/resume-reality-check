import { useState } from "react";
import Header from "./components/Header";
import AnalyzeForm from "./components/AnalyzeForm";
import CandidateRoster from "./components/CandidateRoster";
import CandidateDetail from "./components/CandidateDetail";
import { analyzeResume, AnalyzeError } from "./lib/api";
import type { AnalyzeResponse, Candidate, ViewMode } from "./types";

function makeId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function App() {
  const [view, setView] = useState<ViewMode>("recruiter");

  // Recruiter view: a roster of candidates analyzed against one JD.
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [batchRunning, setBatchRunning] = useState(false);

  // Candidate view: a single self-check, no roster.
  const [selfResult, setSelfResult] = useState<AnalyzeResponse | null>(null);
  const [selfLoading, setSelfLoading] = useState(false);
  const [selfError, setSelfError] = useState<string | null>(null);

  function handleViewChange(next: ViewMode) {
    setView(next);
    setCandidates([]);
    setSelectedId(null);
    setSelfResult(null);
    setSelfError(null);
  }

  async function handleRecruiterSubmit(params: {
    resumes: File[];
    jobDescription: string;
    githubUsername?: string;
  }) {
    const queued: Candidate[] = params.resumes.map((f) => ({
      id: makeId(),
      fileName: f.name,
      status: "queued",
    }));
    setCandidates(queued);
    setSelectedId(null);
    setBatchRunning(true);

    for (let i = 0; i < params.resumes.length; i++) {
      const { id } = queued[i];
      setCandidates((prev) => prev.map((c) => (c.id === id ? { ...c, status: "analyzing" } : c)));
      try {
        const data = await analyzeResume({
          resume: params.resumes[i],
          jobDescription: params.jobDescription,
          githubUsername: params.githubUsername,
        });
        setCandidates((prev) =>
          prev.map((c) => (c.id === id ? { ...c, status: "done", result: data } : c))
        );
        setSelectedId((prev) => prev ?? id);
      } catch (err) {
        setCandidates((prev) =>
          prev.map((c) =>
            c.id === id
              ? {
                  ...c,
                  status: "error",
                  error: err instanceof AnalyzeError ? err.message : "Something went wrong.",
                }
              : c
          )
        );
      }
    }

    setBatchRunning(false);
  }

  async function handleCandidateSubmit(params: { resumes: File[]; jobDescription: string }) {
    setSelfLoading(true);
    setSelfError(null);
    setSelfResult(null);
    try {
      const data = await analyzeResume({ resume: params.resumes[0], jobDescription: params.jobDescription });
      setSelfResult(data);
    } catch (err) {
      setSelfError(err instanceof AnalyzeError ? err.message : "Something went wrong.");
    } finally {
      setSelfLoading(false);
    }
  }

  const selectedCandidate = candidates.find((c) => c.id === selectedId) ?? null;

  if (view === "candidate") {
    return (
      <div className="min-h-screen">
        <Header view={view} onChange={handleViewChange} />
        <main className="mx-auto flex max-w-2xl flex-col gap-6 px-6 py-10">
          <AnalyzeForm view={view} loading={selfLoading} onSubmit={handleCandidateSubmit} />

          {selfLoading && (
            <div className="animate-fade-in rounded-xl border border-accent-500/20 bg-accent-500/10 px-4 py-3 text-sm font-medium text-accent-300">
              Analyzing resume against job description — this can take up to ~2 minutes…
            </div>
          )}

          {selfError && (
            <div className="animate-fade-in rounded-xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm font-medium text-rose-300">
              {selfError}
            </div>
          )}

          {selfResult && <CandidateDetail result={selfResult} view={view} />}
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <Header view={view} onChange={handleViewChange} />
      <main className="mx-auto flex max-w-[1400px] flex-col gap-6 px-6 py-8">
        <div className="grid gap-6 lg:grid-cols-[400px_1fr] lg:items-start">
          <div className="flex flex-col gap-6">
            <AnalyzeForm view={view} loading={batchRunning} onSubmit={handleRecruiterSubmit} />
            {candidates.length > 0 && (
              <CandidateRoster
                candidates={candidates}
                selectedId={selectedId}
                onSelect={setSelectedId}
              />
            )}
          </div>

          <div className="lg:sticky lg:top-24">
            {selectedCandidate?.result ? (
              <CandidateDetail
                result={selectedCandidate.result}
                view={view}
                title={selectedCandidate.fileName.replace(/\.[^./]+$/, "")}
              />
            ) : (
              <div className="flex min-h-[300px] flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-white/[0.08] px-6 py-16 text-center">
                <span className="text-sm font-medium text-slate-300">
                  {candidates.length === 0
                    ? "No candidates yet"
                    : batchRunning
                    ? "Analyzing…"
                    : "Select a candidate"}
                </span>
                <span className="max-w-xs text-sm text-slate-500">
                  {candidates.length === 0
                    ? "Upload one or more resumes and a job description to build the applicant list."
                    : "Click a completed applicant on the left to see their full skill breakdown."}
                </span>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
