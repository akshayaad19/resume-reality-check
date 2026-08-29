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

- **Date:** 2026-08-28
- **Issue:** Re-running `eval/run_eval.py` with no code changes produces different
  score/claims results between runs, making it hard to tell whether a diff between
  two eval runs reflects an actual code change or just noise.
- **Symptom / error:** Same resume/JD input, same code, but `evidence_score` and
  `claimed_on_resume` for some skills differ run-to-run, and the extracted JD skill
  names themselves sometimes change (e.g. "Data structures and algorithms" splits
  into separate "Data structures" and "Algorithms" skills on some runs).
- **Root cause:** This is a real, confirmed limitation of hosted LLM APIs, not a
  bug in our code - Gemini (and other hosted providers) do not guarantee
  bit-identical outputs even at `temperature=0`, because of serving-side batching
  and mixture-of-experts routing effects on the provider's side. Both
  `extract_job_skills()` (JD skill extraction) and the resume-side
  extraction/judge calls are affected, but the JD skill list is the one that
  matters most for eval stability, since it's the reference set every run is
  scored against - if it shifts between runs, before/after comparisons aren't
  trustworthy even when the code under test hasn't changed.
- **Fix:** `eval/run_eval.py` now caches the JD skill extraction: on first run it
  calls `extract_job_skills()` on `test_jd.txt` once and saves the result to
  `eval/job_skills.json`; on every subsequent run it loads the skill list from
  that cache file instead of calling the API again, and passes it into
  `run_analysis()` via a new optional `job_skills` parameter
  (`app/main.py`) that skips re-extraction when a value is supplied. Verified by
  running the eval script twice in a row with no other changes: `eval/job_skills.json`
  was byte-identical both times (same md5 hash), confirming the JD skill list is
  now fixed across runs. The resume-side extraction and judge scoring still varied
  slightly between the two runs (one skill's `evidence_score` differed by 1 point),
  which is expected residual non-determinism from those still-live LLM calls -
  caching only stabilizes the JD skill reference set, not the full pipeline.

- **Date:** 2026-08-29
- **Issue:** Adding GitHub evidence (`github_username`) silently changed evidence
  scores for skills that had nothing to do with GitHub at all - five
  previously-correct skills (Software Development, Algorithms, Collaboration,
  Cloud Platforms, API-based Integrations) dropped to `evidence_score: 0` the
  moment a `github_username` was supplied, even though their supporting resume
  bullets were untouched and still present in the combined evidence list.
- **Symptom / error:** `run_analysis()` built one `EvidenceIndex` over
  `resume_extraction.evidence + github_chunks` combined into a single list
  before indexing (`app/main.py`, old code: `index = EvidenceIndex(combined_evidence)`).
  BM25's IDF term-weights are corpus-relative - a term's weight depends on how
  rare it is across *all* documents in the index, not just the ones relevant to
  a given query. Mixing GitHub README/structural-signal text into the same
  corpus as the resume bullets shifted those IDF weights for every term, which
  in turn changed BM25's ranking of purely-resume chunks for purely-resume
  queries, even when zero GitHub chunks were topically relevant and none were
  ever retrieved for that skill. This is corpus pollution: unrelated documents
  in a shared BM25 index can silently move scores for existing, unrelated
  evidence, purely through shared term-rarity statistics.
- **Root cause:** Single shared `EvidenceIndex` over evidence pooled from
  multiple sources (resume + GitHub) before BM25/embedding indexing.
- **Fix:** `app/main.py` now builds two independent `EvidenceIndex` objects -
  `resume_index` over `resume_extraction.evidence` only, and `github_index`
  over `github_chunks` only - each with its own BM25 index and its own
  embeddings, computed solely from its own chunks. Each skill is searched
  against both indexes independently (existing hybrid BM25+semantic+RRF search
  is unchanged, it just runs twice, once per source), and only the resulting
  *already-scored* top-k chunks from each source are merged into one list
  (`retrieval.merge_ranked_chunks`, sorted by score) before that list goes to
  the judge. This guarantees, by construction, that a source's corpus
  statistics can never influence another source's ranking - `resume_index`
  has no reference to `github_chunks` at all, so resume-only retrieval for a
  given skill is provably identical whether or not GitHub evidence is present.
- **Verification:** Retrieval-layer check (no LLM calls, so unaffected by
  Gemini API non-determinism/quota): for all 18 job skills, resume-only search
  against `resume_index` returned exactly the same 5 chunks regardless of
  whether `github_index` was built alongside it - confirming isolation is
  structural, not just observed. End-to-end (`run_analysis` with vs. without
  `github_username`, via Gemini judge): none of the five previously-affected
  skills dropped to 0 after the fix, and "Documentation" correctly gained
  GitHub-sourced evidence (`0 -> 2`, citing a GitHub repo README). Could not
  get a second clean live-judge run to confirm exact score parity beyond
  "no longer 0" - the free-tier 20-requests/day Gemini quota (see the first
  entry above) was exhausted mid-comparison, apparently faster than expected
  because `judge._generate_with_retry`'s retry-on-429 loop burns additional
  quota-counted requests per logical call. Separately noticed while testing:
  with only 4 GitHub repos fetched, `MIN_RELEVANCE_SCORE`'s "gate on the top
  fused score only" design (`app/retrieval.py`) means nearly every skill's
  GitHub search returns all 4 chunks regardless of actual topical relevance,
  since RRF scores over a 4-document corpus cluster tightly near the maximum -
  a distinct, smaller-scale limitation from corpus pollution, not yet
  addressed, worth revisiting if the GitHub evidence source grows.

- **Date:** 2026-08-29
- **Issue:** `judge._generate_with_retry`'s backoff-and-retry logic was making
  the free-tier 20-requests/day Gemini cap (see the first entry above) worse,
  not better: once the *daily* quota was exhausted, every subsequent
  `judge_all_skills` call still retried up to `MAX_RETRIES` (5) times with a
  15s sleep between attempts, and each retried request appears to count
  against the same daily quota - so one logical call during an outage could
  burn up to 5x its fair share of the day's already-exhausted budget, and
  cost ~75s doing it, before finally raising.
- **Root cause:** `_generate_with_retry` treated every 429 the same way.
  Gemini's 429 response body distinguishes *why* via a `quotaId` in the
  `QuotaFailure` error detail (e.g.
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier` for the daily cap vs. a
  per-minute variant for short-lived rate limiting) - backoff only makes sense
  for the latter, since a daily cap can't reset within a retry window.
- **Fix:** Added `_is_daily_quota_error()` (`app/judge.py`), which inspects
  the 429's `error.details` for a `QuotaFailure` violation whose `quotaId`
  contains "day", and a new `DailyQuotaExhaustedError` exception.
  `_generate_with_retry` now checks this before its retry loop: a daily-quota
  429 raises `DailyQuotaExhaustedError` immediately with a clear message,
  while a non-daily 429 (e.g. per-minute) still goes through the existing
  backoff-and-retry path unchanged.
- **Verification:** Reproduced against a real, currently-exhausted daily
  quota (this project's key was mid-cap while investigating a prior bug) -
  confirmed `judge_all_skills` now raises `DailyQuotaExhaustedError` in
  ~0.4s with no retry sleeps, versus the old behavior of 5 retries over ~75s
  before failing with the same underlying error.

- **Date:** 2026-08-29
- **Issue:** Final end-to-end confirmation that today's two fixes (dual-index
  retrieval, daily-quota fail-fast) hold together in one clean, real
  `/analyze` run - not just in isolated retrieval-layer or single-skill
  checks.
- **Verification:** Ran `run_analysis(test_resume.txt, test_jd.txt,
  github_username="akshayaad19")` once, end to end, with no errors. GitHub
  fetch succeeded (`github_note: None`, 4 repos -> 4 evidence chunks, combined
  with 13 resume evidence bullets = 17 total). Full skill table (18 skills):
  | Skill | Category | Claimed | Score | Cited source |
  |---|---|---|---|---|
  | Computer Science | required | yes | 2 | GitHub: resume-reality-check |
  | Software Development | required | yes | 3 | resume bullet |
  | Data Structures | required | yes | 2 | GitHub: leetcode-solutions |
  | Algorithms | required | yes | 2 | GitHub: resume-reality-check |
  | Scalable System Design | required | no | 3 | resume bullet |
  | AI Orchestration | required | yes | 3 | resume bullet |
  | RAG | required | yes | 2 | resume bullet |
  | Business Process Automation | required | yes | 2 | resume bullet |
  | Python | required | yes | 2 | GitHub: resume-reality-check |
  | ML Operations | required | yes | 3 | resume bullet |
  | Software Testing | required | yes | 2 | resume bullet |
  | Software Deployment | required | yes | 0 | none |
  | Communication | required | yes | 2 | resume bullet |
  | Documentation | required | no | 2 | GitHub: resume-reality-check |
  | Collaboration | required | yes | 2 | resume bullet |
  | Practical AI Application | preferred | yes | 3 | resume bullet |
  | Cloud Platforms | preferred | yes | 1 | resume bullet |
  | API-based Integrations | preferred | yes | 2 | resume bullet |

  Confirms, in one live run: (1) all five skills previously broken by
  corpus-pollution (Software Development, Algorithms, Collaboration, Cloud
  Platforms, API-based Integrations) score non-zero; (2) GitHub evidence
  contributes the winning cited citation for 5 of 18 skills (Computer
  Science, Data Structures, Algorithms, Python, Documentation) - a genuine,
  judge-verified contribution, not retrieval noise, since each citation's
  justification names concrete repo content (hybrid-retrieval implementation,
  LeetCode problem set, the project README's own architecture); (3) Software
  Deployment correctly scores 0 with justification "None of the provided
  chunks mention software deployment practices, CI/CD pipelines, or
  deployment infrastructure" - matching the earlier retrieval-layer finding
  that neither source contains genuine deployment evidence; (4) the call
  completed via the real `judge_all_skills` batched path without hitting the
  daily quota, confirming the retry-logic fix didn't need to trigger for a
  successful call to still work correctly end to end.
