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

2. "evidence": full experience or project bullet points, that describe what the \
person actually did (responsibilities, projects, achievements). Copy each bullet's \
text VERBATIM from the resume - do not summarize or paraphrase it.

For bullets under a PROJECT (a project is introduced by a header line naming the \
project and listing its tech stack, commonly in a format like \
"<Project Name> — <description> | <tech stack> | <team size>"), prepend the \
project's tech-stack tags to the bullet text before adding it to "evidence", as \
"[Technologies used: <tech stack>] <verbatim bullet text>". Take the tech stack \
straight from that project's header line (the pipe-delimited list of \
technologies/tools, not the team-size segment), and apply it to every bullet that \
falls under that project header, until the next project or section header. Do not \
add this tag to bullets outside a project section (e.g. general work-experience \
bullets that aren't under a project header) - copy those verbatim with no prefix.

Also add "claims" derived from the EDUCATION section: for each degree, include \
the field of study itself (e.g. "Information Technology") plus the closely \
related domain terms it commonly implies (e.g. a degree in "Information \
Technology" also implies "Computer Science" and "Software Development"). Add \
each of these as its own separate claim string.

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
