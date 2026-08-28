"""Evaluate the /analyze pipeline's current output against hand-labeled judgments.

Reads eval/labeled_pairs.json (skill-level human labels: my_score,
correct_claimed_on_resume), re-runs the full /analyze pipeline on
test_resume.txt + test_jd.txt, and reports:
  1. Agreement between my_score and the system's current evidence_score.
  2. Which skills disagree, and by how much.
  3. Accuracy of claimed_on_resume against the labeled expectation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from dotenv import load_dotenv

load_dotenv(dotenv_path=REPO_ROOT / ".env")

from app.main import run_analysis
from app.retrieval import _get_embedder

LABELED_PAIRS_PATH = REPO_ROOT / "eval" / "labeled_pairs.json"
RESUME_PATH = REPO_ROOT / "test_resume.txt"
JD_PATH = REPO_ROOT / "test_jd.txt"

# extract_job_skills() is a live LLM call and re-splits/rewords skills
# differently across runs (e.g. "Data structures and algorithms" -> separate
# "Data structures" + "Algorithms"), so labeled skill names are matched to the
# current run's skill names by nearest embedding neighbor rather than exact
# string equality. Below this similarity, a labeled skill is treated as
# having no confident match in the current run.
SKILL_MATCH_MIN_SIMILARITY = 0.60


def _nearest_skill_match(labeled_skill: str, current_skills: list[str]) -> tuple[str, float]:
    embedder = _get_embedder()
    labeled_embedding = embedder.encode([labeled_skill], normalize_embeddings=True)[0]
    current_embeddings = embedder.encode(current_skills, normalize_embeddings=True)
    sims = current_embeddings @ labeled_embedding
    best_idx = int(np.argmax(sims))
    return current_skills[best_idx], float(sims[best_idx])


def main() -> None:
    labeled_pairs = json.loads(LABELED_PAIRS_PATH.read_text())
    labeled_by_skill = {entry["skill"]: entry for entry in labeled_pairs}

    resume_text = RESUME_PATH.read_text()
    job_description = JD_PATH.read_text()
    result = run_analysis(resume_text, job_description)
    system_by_skill = {row.skill: row for row in result.skills}
    current_skill_names = list(system_by_skill.keys())

    matched = []
    missing = []
    for labeled_skill, labeled in labeled_by_skill.items():
        if labeled_skill in system_by_skill:
            matched.append((labeled_skill, labeled_skill, labeled, system_by_skill[labeled_skill], 1.0))
            continue
        nearest_skill, similarity = _nearest_skill_match(labeled_skill, current_skill_names)
        if similarity >= SKILL_MATCH_MIN_SIMILARITY:
            matched.append((labeled_skill, nearest_skill, labeled, system_by_skill[nearest_skill], similarity))
        else:
            missing.append((labeled_skill, nearest_skill, similarity))

    print("=" * 88)
    print("EVAL: labeled_pairs.json vs. current /analyze output")
    print(f"Resume: {RESUME_PATH.name} | Job description: {JD_PATH.name}")
    print("=" * 88)

    fuzzy_matches = [m for m in matched if m[4] < 1.0]
    if fuzzy_matches:
        print(f"\nNOTE: job-skill extraction reworded {len(fuzzy_matches)} labeled skill(s) "
              f"this run. Matched by nearest embedding similarity instead of exact name:")
        for labeled_skill, matched_skill, _labeled, _system, similarity in fuzzy_matches:
            print(f"  - {labeled_skill!r} -> {matched_skill!r} (similarity={similarity:.3f})")

    if missing:
        print(f"\nWARNING: {len(missing)} labeled skill(s) have no confident match "
              f"(similarity < {SKILL_MATCH_MIN_SIMILARITY}) in the current /analyze output:")
        for labeled_skill, nearest_skill, similarity in missing:
            print(f"  - {labeled_skill!r} (closest was {nearest_skill!r} at {similarity:.3f})")

    # --- (1) Score agreement ---
    print("\n" + "-" * 88)
    print("(1) SCORE AGREEMENT: my_score vs. current evidence_score")
    print("-" * 88)
    print(f"{'Skill':<48} {'my_score':<9} {'current':<9} {'match':<6}")
    score_matches = 0
    for labeled_skill, _matched_skill, labeled, system, _similarity in matched:
        my_score = labeled["my_score"]
        current_score = system.evidence_score
        match = my_score == current_score
        score_matches += match
        print(f"{labeled_skill:<48} {my_score:<9} {current_score:<9} {'yes' if match else 'no'}")

    score_agreement_pct = 100 * score_matches / len(matched) if matched else 0.0
    print(f"\nScore agreement: {score_matches}/{len(matched)} ({score_agreement_pct:.1f}%)")

    # --- (2) Disagreements ---
    print("\n" + "-" * 88)
    print("(2) DISAGREEMENTS: skills where my_score != current evidence_score")
    print("-" * 88)
    disagreements = [
        (labeled_skill, labeled["my_score"], system.evidence_score,
         system.evidence_score - labeled["my_score"])
        for labeled_skill, _matched_skill, labeled, system, _similarity in matched
        if labeled["my_score"] != system.evidence_score
    ]
    if not disagreements:
        print("None - current evidence_score matches my_score for every labeled skill.")
    else:
        disagreements.sort(key=lambda d: abs(d[3]), reverse=True)
        for labeled_skill, my_score, current_score, delta in disagreements:
            sign = "+" if delta > 0 else ""
            print(f"  {labeled_skill:<48} my_score={my_score}  current={current_score}  "
                  f"delta={sign}{delta}")

    # --- (3) claimed_on_resume accuracy ---
    print("\n" + "-" * 88)
    print("(3) claimed_on_resume ACCURACY vs. correct_claimed_on_resume")
    print("-" * 88)
    print(f"{'Skill':<48} {'expected':<10} {'current':<9} {'match':<6}")
    claim_matches = 0
    for labeled_skill, _matched_skill, labeled, system, _similarity in matched:
        expected = labeled["correct_claimed_on_resume"]
        current = system.claimed_on_resume
        match = expected == current
        claim_matches += match
        print(f"{labeled_skill:<48} {str(expected):<10} {str(current):<9} "
              f"{'yes' if match else 'no'}")

    claim_accuracy_pct = 100 * claim_matches / len(matched) if matched else 0.0
    print(f"\nclaimed_on_resume accuracy: {claim_matches}/{len(matched)} "
          f"({claim_accuracy_pct:.1f}%)")

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"Score agreement:            {score_matches}/{len(matched)} "
          f"({score_agreement_pct:.1f}%)")
    print(f"claimed_on_resume accuracy: {claim_matches}/{len(matched)} "
          f"({claim_accuracy_pct:.1f}%)")


if __name__ == "__main__":
    main()
