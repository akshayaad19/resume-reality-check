"""FastAPI app: resume + job description in, per-skill evidence-depth table out."""
from __future__ import annotations

from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app import extraction, github_evidence, judge, parsing
from app.retrieval import EvidenceIndex, merge_ranked_chunks, skill_is_claimed

app = FastAPI(title="Resume Reality Check")

TOP_K_CHUNKS = 5


class SkillRow(BaseModel):
    skill: str
    category: str  # "required" or "preferred"
    claimed_on_resume: bool
    evidence_score: int
    justification: str
    cited_evidence: Optional[str]


class AnalyzeResponse(BaseModel):
    claims: List[str]
    evidence_bullets: List[str]
    skills: List[SkillRow]
    github_note: Optional[str] = None


def run_analysis(
    resume_text: str,
    job_description: str,
    job_skills: Optional[extraction.JobSkills] = None,
    github_username: Optional[str] = None,
) -> AnalyzeResponse:
    """Core /analyze pipeline: extraction, hybrid retrieval, and judging.

    Takes already-extracted resume/job-description text, so it can be reused
    by both the HTTP endpoint (after file upload + parsing) and the eval
    script (reading text files directly). `job_skills` can be passed in to
    reuse a previously extracted (e.g. cached) result instead of calling
    extract_job_skills() again. `github_username`, if given, pulls in extra
    evidence chunks from that user's public GitHub repos.

    Resume and GitHub evidence are each indexed separately (their own BM25
    index and embeddings, built only from their own chunks) and searched
    independently per skill; only the resulting ranked chunks are merged,
    after scoring. This avoids corpus-pollution, where merging chunks from
    both sources into one BM25 index before indexing lets one source's term
    statistics silently shift relevance scores for the other's evidence -
    see debugging-log.md."""
    resume_extraction = extraction.extract_resume_claims_and_evidence(resume_text)
    if job_skills is None:
        job_skills = extraction.extract_job_skills(job_description)

    github_note: Optional[str] = None
    github_chunks: List[str] = []
    if github_username:
        try:
            github_chunks = github_evidence.fetch_github_evidence(github_username)
        except github_evidence.GitHubFetchError as e:
            github_note = f"GitHub evidence unavailable: {e}"
        else:
            if not github_chunks:
                github_note = f"No usable evidence found in {github_username}'s public repos."

    resume_index = EvidenceIndex(resume_extraction.evidence)
    github_index = EvidenceIndex(github_chunks)
    combined_evidence = list(resume_extraction.evidence) + github_chunks

    skill_entries = [
        (skill, "required") for skill in job_skills.required_skills
    ] + [(skill, "preferred") for skill in job_skills.preferred_skills]

    skill_evidence = [
        (
            skill,
            [
                chunk
                for chunk, _score in merge_ranked_chunks(
                    resume_index.search(skill, top_k=TOP_K_CHUNKS),
                    github_index.search(skill, top_k=TOP_K_CHUNKS),
                )
            ],
        )
        for skill, _category in skill_entries
    ]
    judged_skills = judge.judge_all_skills(skill_evidence)

    skill_rows = [
        SkillRow(
            skill=judged.skill,
            category=category,
            claimed_on_resume=skill_is_claimed(judged.skill, resume_extraction.claims),
            evidence_score=judged.score,
            justification=judged.justification,
            cited_evidence=judged.cited_evidence,
        )
        for (_skill, category), judged in zip(skill_entries, judged_skills)
    ]

    return AnalyzeResponse(
        claims=resume_extraction.claims,
        evidence_bullets=combined_evidence,
        skills=skill_rows,
        github_note=github_note,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    github_username: Optional[str] = Form(None),
) -> AnalyzeResponse:
    resume_bytes = await resume.read()
    if not resume_bytes:
        raise HTTPException(status_code=400, detail="Uploaded resume file is empty")

    resume_text = parsing.extract_text(resume_bytes, resume.filename or "")
    if not resume_text:
        raise HTTPException(
            status_code=400, detail="Could not extract any text from the resume"
        )

    return run_analysis(resume_text, job_description, github_username=github_username)
