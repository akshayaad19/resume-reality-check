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


class DailyQuotaExhaustedError(RuntimeError):
    """Raised when a 429 is Gemini's daily request quota (not a per-minute
    rate limit). Backoff-and-retry can't help here - the cap resets on a
    calendar-day boundary, not within seconds - so retrying just burns
    additional quota-counted requests for no benefit."""


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set")
        _client = genai.Client(api_key=api_key)
    return _client


def _quota_violations(error: genai_errors.ClientError) -> List[dict]:
    """Extract the QuotaFailure violations (each with a quotaId) from a 429's
    error body, e.g. {"error": {"details": [{"@type": "...QuotaFailure",
    "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", ...}]}]}}."""
    details = error.details if isinstance(error.details, dict) else {}
    body = details.get("error", details)
    violations = []
    for detail in body.get("details", []) or []:
        if str(detail.get("@type", "")).endswith("QuotaFailure"):
            violations.extend(detail.get("violations", []) or [])
    return violations


def _is_daily_quota_error(error: genai_errors.ClientError) -> bool:
    """True if this 429 is a per-day quota cap rather than a per-minute rate
    limit, distinguished via the quotaId Gemini reports (e.g. ends in
    "PerDay..." vs "PerMinute...")."""
    return any(
        "day" in str(v.get("quotaId", "")).lower()
        for v in _quota_violations(error)
    )


def _generate_with_retry(client: genai.Client, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.ClientError as e:
            if e.code == 429 and _is_daily_quota_error(e):
                raise DailyQuotaExhaustedError(
                    f"Gemini daily request quota exhausted for model {MODEL!r} - "
                    "retrying won't help until the quota resets, so failing "
                    f"immediately instead of retrying {MAX_RETRIES} times. "
                    f"Original error: {e.message}"
                ) from e
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

Structural signal rule (apply on top of the levels above, before settling on a \
final score):
A "structural signal" is a verified, checkable artifact reported directly from a \
repository's own file listing (e.g. "Dockerfile present", "k8s/helm config \
present", "CI workflow present") - it is a confirmed fact about what exists in \
the repo, not a self-reported claim about what someone did. Only treat a \
structural signal as evidence for the specific skill it corresponds to \
(Dockerfile -> Docker/containerization, k8s/helm config -> Kubernetes, CI \
workflow -> CI/CD) - never as generic confirmation for an unrelated skill just \
because it happens to appear in the same repo's chunk.
- If a skill has BOTH descriptive prose evidence (a resume bullet or README \
description that describes using the skill) AND a matching structural signal \
confirming it, score 3, regardless of whether the prose alone would only reach \
"Applied" level. A description plus a verified artifact together are the \
strongest possible evidence.
- If a skill has ONLY a matching structural signal for it, with no descriptive \
prose mentioning that skill anywhere in the provided chunks, score at least 2 \
("Applied") - never 0 or 1. A structural signal alone is a verified fact, not an \
unverified claim, so it counts as real evidence on its own even without a \
sentence describing how it was used.

Verified dependency lists: a chunk may include a line like "Verified \
dependencies (from requirements.txt): fastapi, sentence-transformers, \
rank-bm25, pydantic" or "Verified dependencies (from package.json): react, \
express". These are parsed directly from a real dependency-management file, \
typically machine-generated from what's actually installed (e.g. via `pip \
freeze`), so an individual package name listed there is hard to fake and \
counts as strong, specific evidence - but ONLY for a skill that specific \
package actually names or directly implements (e.g. "sentence-transformers" \
supports "embeddings" or "semantic search"; "fastapi" supports "REST APIs" or \
"backend development"; "rank-bm25" supports "BM25" or "keyword search"; \
"react" supports "frontend development"). Apply the same BOTH/ONLY rule above \
using a matching package name in place of a structural signal - a package name \
plus descriptive prose for the same skill scores 3, a matching package name \
alone (no prose for that skill) scores at least 2. The mere presence of a \
requirements.txt or package.json line - i.e. having *some* dependency list, \
with no package name in it relevant to the skill being scored - is NOT \
evidence for that skill, and must not be used to justify a score above what \
the skill's own remaining evidence would earn on its own.

In the justification, state explicitly which case applies, e.g. "Structural \
signal found (Dockerfile) in repo X's file listing, no descriptive text - scored \
2 based on verified artifact alone", "Structural signal (CI workflow) AND \
descriptive README text both confirm this skill - scored 3", or "Verified \
dependency 'sentence-transformers' in repo X's requirements.txt confirms this \
skill - scored 2 based on the verified package alone."
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
                temperature=0,
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
