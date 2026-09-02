import { useState, type DragEvent, type FormEvent } from "react";
import type { ViewMode } from "../types";

interface Props {
  view: ViewMode;
  loading: boolean;
  onSubmit: (params: { resumes: File[]; jobDescription: string; githubUsername?: string }) => void;
}

const ACCEPTED_EXT = [".pdf", ".txt"];

function isAccepted(file: File) {
  return ACCEPTED_EXT.some((ext) => file.name.toLowerCase().endsWith(ext));
}

function fileKey(file: File) {
  return `${file.name}-${file.size}-${file.lastModified}`;
}

export default function AnalyzeForm({ view, loading, onSubmit }: Props) {
  const [resumes, setResumes] = useState<File[]>([]);
  const [jobDescription, setJobDescription] = useState("");
  const [githubUsername, setGithubUsername] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  const isRecruiter = view === "recruiter";
  const multiple = isRecruiter;

  function addFiles(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) return;
    const incoming = Array.from(fileList);
    const rejected = incoming.filter((f) => !isAccepted(f));
    if (rejected.length > 0) {
      setFormError("Please only upload .pdf or .txt files.");
      return;
    }
    setFormError(null);

    if (!multiple) {
      setResumes([incoming[0]]);
      return;
    }

    setResumes((prev) => {
      const seen = new Set(prev.map(fileKey));
      const merged = [...prev];
      for (const f of incoming) {
        const key = fileKey(f);
        if (!seen.has(key)) {
          seen.add(key);
          merged.push(f);
        }
      }
      return merged;
    });
  }

  function removeFile(key: string) {
    setResumes((prev) => prev.filter((f) => fileKey(f) !== key));
  }

  function handleDrop(e: DragEvent<HTMLLabelElement>) {
    e.preventDefault();
    setDragActive(false);
    addFiles(e.dataTransfer.files);
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (resumes.length === 0) {
      setFormError(isRecruiter ? "Please attach at least one resume." : "Please attach a resume.");
      return;
    }
    if (!jobDescription.trim()) {
      setFormError("Please paste a job description.");
      return;
    }
    setFormError(null);
    onSubmit({
      resumes,
      jobDescription: jobDescription.trim(),
      githubUsername:
        isRecruiter && resumes.length === 1 ? githubUsername.trim() || undefined : undefined,
    });
  }

  return (
    <form onSubmit={handleSubmit} className="rounded-2xl border border-white/[0.06] bg-[#111726] p-7 shadow-card">
      <h2 className="text-lg font-semibold text-slate-100">
        {isRecruiter ? "Evaluate candidates" : "Check your resume"}
      </h2>
      <p className="mt-1.5 text-sm leading-relaxed text-slate-400">
        {isRecruiter ? (
          <>
            Upload one or more resumes against a single job description. We separate what each resume{" "}
            <em className="not-italic font-medium text-slate-200">claims</em> from what it can actually{" "}
            <em className="not-italic font-medium text-slate-200">prove</em>.
          </>
        ) : (
          "See how your resume holds up against a job description before you apply — and which skills need stronger proof, not just a mention."
        )}
      </p>

      <div className="mt-6 space-y-5">
        <div>
          <label
            htmlFor="resume-input"
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            className={`flex cursor-pointer flex-col items-center justify-center gap-1 rounded-xl border-2 border-dashed px-4 py-8 text-center transition-colors ${
              dragActive
                ? "border-accent-400 bg-accent-500/10"
                : resumes.length > 0
                ? "border-white/[0.12] bg-white/[0.03]"
                : "border-white/[0.08] hover:border-white/[0.16] hover:bg-white/[0.03]"
            }`}
          >
            <svg
              width="28"
              height="28"
              viewBox="0 0 24 24"
              fill="none"
              className="text-slate-500"
              aria-hidden="true"
            >
              <path
                d="M12 16V4m0 0L7 9m5-5l5 5M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            {resumes.length > 0 && !multiple ? (
              <span className="text-sm font-medium text-slate-200">{resumes[0].name}</span>
            ) : (
              <>
                <span className="text-sm font-medium text-slate-300">
                  {multiple
                    ? "Drop resumes here, or click to browse"
                    : "Drop a resume here, or click to browse"}
                </span>
                <span className="text-xs text-slate-500">
                  PDF or plain text{multiple ? " · multiple files ok" : ""}
                </span>
              </>
            )}
            <input
              id="resume-input"
              type="file"
              accept=".pdf,.txt"
              multiple={multiple}
              className="hidden"
              onChange={(e) => addFiles(e.target.files)}
            />
          </label>

          {multiple && resumes.length > 0 && (
            <ul className="mt-3 flex flex-col gap-1.5 animate-fade-in">
              {resumes.map((f) => (
                <li
                  key={fileKey(f)}
                  className="flex items-center justify-between gap-2 rounded-lg border border-white/[0.06] bg-white/[0.03] px-3 py-1.5 text-[13px] text-slate-300"
                >
                  <span className="truncate">{f.name}</span>
                  <button
                    type="button"
                    onClick={() => removeFile(fileKey(f))}
                    className="shrink-0 text-slate-500 hover:text-rose-400"
                    aria-label={`Remove ${f.name}`}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                      <path
                        d="M6 6l12 12M18 6L6 18"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div>
          <label htmlFor="jd-input" className="mb-1.5 block text-sm font-semibold text-slate-300">
            Job description
          </label>
          <textarea
            id="jd-input"
            rows={7}
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Paste the job description here…"
            className="w-full resize-y rounded-lg border border-white/[0.08] bg-white/[0.03] px-3 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:border-accent-400 focus:bg-white/[0.05] focus:outline-none focus:ring-2 focus:ring-accent-500/20"
          />
        </div>

        {isRecruiter && resumes.length <= 1 && (
          <div className="animate-fade-in">
            <label htmlFor="gh-input" className="mb-1.5 block text-sm font-semibold text-slate-300">
              GitHub username <span className="font-normal text-slate-500">(optional, single candidate only)</span>
            </label>
            <input
              id="gh-input"
              type="text"
              value={githubUsername}
              onChange={(e) => setGithubUsername(e.target.value)}
              placeholder="octocat"
              className="w-full rounded-lg border border-white/[0.08] bg-white/[0.03] px-3 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:border-accent-400 focus:bg-white/[0.05] focus:outline-none focus:ring-2 focus:ring-accent-500/20"
            />
          </div>
        )}
      </div>

      {formError && (
        <p className="mt-4 rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-sm font-medium text-rose-300">
          {formError}
        </p>
      )}

      <button
        type="submit"
        disabled={loading}
        className="mt-6 flex w-full items-center justify-center gap-2 rounded-lg bg-accent-500 px-5 py-2.5 text-sm font-semibold text-white shadow-[0_0_20px_rgba(99,102,241,0.25)] transition-colors hover:bg-accent-600 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto"
      >
        {loading && (
          <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
            />
          </svg>
        )}
        {loading
          ? "Analyzing…"
          : isRecruiter && resumes.length > 1
          ? `Analyze ${resumes.length} candidates`
          : "Analyze"}
      </button>
    </form>
  );
}
