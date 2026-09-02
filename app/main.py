"""FastAPI app: resume + job description in, per-skill evidence-depth table out."""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import extraction, github_evidence, judge, parsing
from app.retrieval import EvidenceIndex, merge_ranked_chunks, search_with_fallback, skill_is_claimed

app = FastAPI(title="Resume Reality Check")

logger = logging.getLogger("uvicorn.error")

TOP_K_CHUNKS = 5


@app.exception_handler(judge.DailyQuotaExhaustedError)
async def handle_daily_quota_exhausted(
    request: Request, exc: judge.DailyQuotaExhaustedError
) -> JSONResponse:
    """Turns the backend's fail-fast daily-quota error into a clean response.

    Without this, it surfaces to callers as an unhandled 500 with a raw
    traceback message - this gives the frontend (and `/docs` users) a
    specific, actionable detail and status code (503: temporarily
    unavailable, not the caller's fault) instead.
    """
    logger.warning(f"{request.method} {request.url.path} - Gemini daily quota exhausted")
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "The Gemini API's free-tier daily quota has been used up for this "
                "API key's project. This resets roughly at midnight Pacific Time - "
                "try again later, or use an API key from a different Google Cloud "
                "project/account."
            )
        },
    )


@app.middleware("http")
async def log_request_duration(request: Request, call_next):
    """Logs elapsed time for each request, on completion.

    Only fires if the request finishes (success or handled error) - a
    process that dies mid-request (OOM kill, crash) never reaches the log
    line below, so its absence in the logs is itself the crash-vs-timeout
    signal: Render's platform-level request log shows the request started,
    but no "completed" line ever follows.
    """
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(f"{request.method} {request.url.path} completed in {elapsed_ms:.0f}ms")
    return response


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
    independently per skill via search_with_fallback() (self-healing: if a
    skill's literal wording scores below MIN_RELEVANCE_SCORE against a
    source, Gemini-reformulated alternates are tried before giving up on
    that source); only the resulting ranked chunks are merged, after
    scoring. This avoids corpus-pollution, where merging chunks from both
    sources into one BM25 index before indexing lets one source's term
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
                    search_with_fallback(skill, resume_index, top_k=TOP_K_CHUNKS),
                    search_with_fallback(skill, github_index, top_k=TOP_K_CHUNKS),
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


# Mounted last so it never shadows the API routes above; serves the frontend
# (recruiter/candidate views) at "/" (index.html, styles.css, app.js).
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
