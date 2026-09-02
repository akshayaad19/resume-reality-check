import type { AnalyzeResponse } from "../types";

export class AnalyzeError extends Error {}

export async function analyzeResume(params: {
  resume: File;
  jobDescription: string;
  githubUsername?: string;
}): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("resume", params.resume);
  formData.append("job_description", params.jobDescription);
  if (params.githubUsername) {
    formData.append("github_username", params.githubUsername);
  }

  const res = await fetch("/analyze", { method: "POST", body: formData });

  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new AnalyzeError(detail?.detail || `Request failed (${res.status})`);
  }

  return res.json();
}
