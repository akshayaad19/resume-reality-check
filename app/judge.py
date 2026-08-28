"""LLM-as-judge: score evidence depth for claimed skills using a fixed rubric.

All skills for a resume are scored in a single Gemini call (rather than one call
per skill) to stay within tight free-tier daily request quotas.
"""
from __future__ import annotations

import os
import time
from typing import List, Tuple

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


RUBRIC = """\
Score the depth of evidence for the claimed skill using this rubric:

0 - No evidence: none of the provided chunks demonstrate use of the skill.
1 - Superficial: the skill is mentioned in passing, with no concrete detail about \
how it was applied or what was achieved.
2 - Applied: at least one chunk shows the skill being applied to a real task, with \
some concrete detail (what was built, done, or solved).
3 - Deep expertise: at least one chunk shows sustained, non-trivial use of the \
skill with specific, measurable outcomes or impact (e.g. scale, performance, \
ownership, results).
"""

SKILL_BLOCK_TEMPLATE = """\
Skill: {skill}
Evidence chunks (numbered, 0-indexed, specific to this skill):
{numbered_chunks}
"""

BATCH_JUDGE_PROMPT = """\
{rubric}

You will be given several claimed skills, each with its own list of evidence \
chunks retrieved from a resume. For EACH skill, score its evidence depth from 0-3 \
using the rubric, and cite the index of the single chunk (from THAT skill's own \
numbered list) that most justifies the score. If no chunk supports the skill at \
all, use cited_chunk_index: -1. Return exactly one judgment per skill listed \
below, with the "skill" field set to the exact skill name given.

{skill_blocks}
"""


class SkillJudgment(BaseModel):
    skill: str
    score: int
    cited_chunk_index: int
    justification: str


class BatchJudgment(BaseModel):
    judgments: List[SkillJudgment]


class JudgedSkill(BaseModel):
    skill: str
    score: int
    justification: str
    cited_evidence: str | None


def judge_all_skills(
    skill_evidence: List[Tuple[str, List[str]]]
) -> List[JudgedSkill]:
    """Score evidence depth for multiple skills in a single Gemini call.

    `skill_evidence` is a list of (skill, evidence_chunks) pairs. Results are
    returned in the same order. Skills with no evidence chunks are scored 0
    locally, without spending an API call on them.
    """
    results: dict[int, JudgedSkill] = {}
    to_judge: List[Tuple[int, str, List[str]]] = []
    for i, (skill, chunks) in enumerate(skill_evidence):
        if chunks:
            to_judge.append((i, skill, chunks))
        else:
            results[i] = JudgedSkill(
                skill=skill,
                score=0,
                justification="No evidence chunks were retrieved for this skill.",
                cited_evidence=None,
            )

    if to_judge:
        skill_blocks = "\n".join(
            SKILL_BLOCK_TEMPLATE.format(
                skill=skill,
                numbered_chunks="\n".join(
                    f"[{j}] {chunk}" for j, chunk in enumerate(chunks)
                ),
            )
            for _, skill, chunks in to_judge
        )
        client = _get_client()
        response = _generate_with_retry(
            client,
            model=MODEL,
            contents=BATCH_JUDGE_PROMPT.format(
                rubric=RUBRIC, skill_blocks=skill_blocks
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BatchJudgment,
            ),
        )
        batch = BatchJudgment.model_validate_json(response.text)
        judgment_by_skill = {j.skill.strip().lower(): j for j in batch.judgments}

        for i, skill, chunks in to_judge:
            judgment = judgment_by_skill.get(skill.strip().lower())
            if judgment is None:
                results[i] = JudgedSkill(
                    skill=skill,
                    score=0,
                    justification="Judge did not return a score for this skill.",
                    cited_evidence=None,
                )
                continue

            cited_evidence = None
            if 0 <= judgment.cited_chunk_index < len(chunks):
                cited_evidence = chunks[judgment.cited_chunk_index]

            results[i] = JudgedSkill(
                skill=skill,
                score=judgment.score,
                justification=judgment.justification,
                cited_evidence=cited_evidence,
            )

    return [results[i] for i in range(len(skill_evidence))]
