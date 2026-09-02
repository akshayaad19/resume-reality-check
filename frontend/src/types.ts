export type SkillCategory = "required" | "preferred";

export interface SkillRow {
  skill: string;
  category: SkillCategory;
  claimed_on_resume: boolean;
  evidence_score: number; // 0-3
  justification: string;
  cited_evidence: string | null;
}

export interface AnalyzeResponse {
  claims: string[];
  evidence_bullets: string[];
  skills: SkillRow[];
  github_note: string | null;
}

export type ViewMode = "recruiter" | "candidate";

export type CandidateStatus = "queued" | "analyzing" | "done" | "error";

export interface Candidate {
  id: string;
  fileName: string;
  status: CandidateStatus;
  result?: AnalyzeResponse;
  error?: string;
}
