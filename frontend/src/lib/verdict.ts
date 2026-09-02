import type { SkillRow } from "../types";

// A skill counts as "proven" (not just claimed) once the LLM judge scores its
// cited evidence at 2/3 or higher. Recommending on required-skill coverage
// only (falling back to all skills if a JD somehow has none marked
// "required") mirrors how a recruiter actually screens: preferred skills are
// a bonus, not a gate.
export const PROVEN_SCORE_THRESHOLD = 2;
export const RECOMMEND_RATIO_THRESHOLD = 0.6;

export interface Verdict {
  recommended: boolean;
  proven: number;
  total: number;
  ratio: number;
  usedRequiredOnly: boolean;
}

export function computeVerdict(skills: SkillRow[]): Verdict {
  const required = skills.filter((s) => s.category === "required");
  const pool = required.length > 0 ? required : skills;
  const proven = pool.filter((s) => s.evidence_score >= PROVEN_SCORE_THRESHOLD).length;
  const ratio = pool.length > 0 ? proven / pool.length : 0;

  return {
    recommended: pool.length > 0 && ratio >= RECOMMEND_RATIO_THRESHOLD,
    proven,
    total: pool.length,
    ratio,
    usedRequiredOnly: required.length > 0,
  };
}
