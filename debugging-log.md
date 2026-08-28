# Debugging Log

Track issues found while running/testing resume-reality-check here.

## Format

- **Date:**
- **Issue:**
- **Symptom / error:**
- **Root cause:**
- **Fix:**

## 2026-08-28

- **Date:** 2026-08-28
- **Issue:** `/analyze` returns 500 when calling Gemini.
- **Symptom / error:** `google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED` on
  `gemini-3.6-flash`, quotaId `GenerateRequestsPerDayPerProjectPerModel-FreeTier`,
  quotaValue `20`.
- **Root cause:** The free-tier Gemini API key is capped at 20 `generateContent`
  requests per day for this model - a daily cap, not a per-minute one. One `/analyze`
  call makes 2 extraction calls + 1 judge call per required/preferred skill, so a
  job description with 10+ skills alone exceeds the daily cap. Retry-with-backoff
  (added in extraction.py/judge.py for 429s) only helps with the earlier, separate
  5-requests/minute free-tier limit - it cannot work around a 20/day hard cap.
- **Fix:** Reworked `judge.judge_skill_evidence` (per-skill call) into
  `judge.judge_all_skills` (app/judge.py), which scores every required/preferred
  skill in a single Gemini call via a `BatchJudgment` schema, matching results back
  to skills by name. `main.py` updated to call it once per `/analyze` request
  instead of once per skill. This drops the request count per `/analyze` call from
  `2 + N` (N = number of skills) down to a flat 3 (2 extraction calls + 1 batched
  judge call). Verified the code path is correct, but could NOT verify against the
  live API yet - the day's 20-request quota was already spent by earlier test
  attempts before the fix landed, so a same-day retry still returns 429
  RESOURCE_EXHAUSTED. Needs a fresh `/analyze` run once the daily quota resets (or
  with a billing-enabled key) to confirm end-to-end.
