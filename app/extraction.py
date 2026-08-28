"""Gemini-backed extraction of resume claims/evidence and job description skills."""
from __future__ import annotations

import os
import time
from typing import List

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

MODEL = "gemini-3.6-flash"
RATE_LIMIT_RETRY_DELAY_SECONDS = 15
MAX_RETRIES = 5

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set")
        _client = genai.Client(api_key=api_key)
    return _client


def _generate_with_retry(client: genai.Client, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.ClientError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                time.sleep(RATE_LIMIT_RETRY_DELAY_SECONDS)
                continue
            raise


class ResumeExtraction(BaseModel):
    claims: List[str]
    evidence: List[str]


class JobSkills(BaseModel):
    required_skills: List[str]
    preferred_skills: List[str]


RESUME_EXTRACTION_PROMPT = """\
You are analyzing a resume. Split its content into two categories:

1. "claims": bare skill mentions with no supporting detail - typically names of \
technologies, tools, or skills listed in a "Skills", "Technologies", or similar \
section, with no accompanying description of how they were used.

2. "evidence": full experience or project bullet points, copied VERBATIM from the \
resume, that describe what the person actually did (responsibilities, projects, \
achievements). Do not summarize or paraphrase - copy the bullet text exactly as \
it appears.

Resume text:
---
{resume_text}
---
"""

JOB_SKILLS_PROMPT = """\
You are analyzing a job description. Extract the skills it asks for, split into:

1. "required_skills": skills explicitly stated as required, must-have, or mandatory.
2. "preferred_skills": skills explicitly stated as preferred, nice-to-have, or a plus.

Job description:
---
{job_description}
---
"""


def extract_resume_claims_and_evidence(resume_text: str) -> ResumeExtraction:
    client = _get_client()
    response = _generate_with_retry(
        client,
        model=MODEL,
        contents=RESUME_EXTRACTION_PROMPT.format(resume_text=resume_text),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ResumeExtraction,
        ),
    )
    return ResumeExtraction.model_validate_json(response.text)


def extract_job_skills(job_description: str) -> JobSkills:
    client = _get_client()
    response = _generate_with_retry(
        client,
        model=MODEL,
        contents=JOB_SKILLS_PROMPT.format(job_description=job_description),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JobSkills,
        ),
    )
    return JobSkills.model_validate_json(response.text)
