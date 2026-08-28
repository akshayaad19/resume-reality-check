"""FastAPI app: resume + job description in, per-skill evidence-depth table out."""
from __future__ import annotations

from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app import extraction, judge, parsing
from app.retrieval import EvidenceIndex

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


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
) -> AnalyzeResponse:
    resume_bytes = await resume.read()
    if not resume_bytes:
        raise HTTPException(status_code=400, detail="Uploaded resume file is empty")

    resume_text = parsing.extract_text(resume_bytes, resume.filename or "")
    if not resume_text:
        raise HTTPException(
            status_code=400, detail="Could not extract any text from the resume"
        )

    resume_extraction = extraction.extract_resume_claims_and_evidence(resume_text)
    job_skills = extraction.extract_job_skills(job_description)

    index = EvidenceIndex(resume_extraction.evidence)
    claims_lower = {claim.lower() for claim in resume_extraction.claims}

    skill_entries = [
        (skill, "required") for skill in job_skills.required_skills
    ] + [(skill, "preferred") for skill in job_skills.preferred_skills]

    skill_evidence = [
        (skill, [chunk for chunk, _score in index.search(skill, top_k=TOP_K_CHUNKS)])
        for skill, _category in skill_entries
    ]
    judged_skills = judge.judge_all_skills(skill_evidence)

    skill_rows = [
        SkillRow(
            skill=judged.skill,
            category=category,
            claimed_on_resume=judged.skill.lower() in claims_lower,
            evidence_score=judged.score,
            justification=judged.justification,
            cited_evidence=judged.cited_evidence,
        )
        for (_skill, category), judged in zip(skill_entries, judged_skills)
    ]

    return AnalyzeResponse(
        claims=resume_extraction.claims,
        evidence_bullets=resume_extraction.evidence,
        skills=skill_rows,
    )
