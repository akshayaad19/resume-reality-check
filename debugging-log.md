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

- **Date:** 2026-08-29
- **Issue:** The judge rubric had no explicit handling for GitHub structural
  signals (`Dockerfile present`, `k8s/helm config present`, `CI workflow
  present`, etc. - `app/github_evidence.py`'s `_detect_structural_signals`).
  These are verified, checkable facts pulled directly from a repo's file
  listing, not self-reported prose, but the original rubric only scored
  based on descriptive detail ("what was built, done, or solved") - so a
  skill backed only by a structural signal, with no sentence describing it,
  had no clear path to a score above 0 or 1, undervaluing evidence that is
  actually more trustworthy than prose (it can't be exaggerated the way a
  bullet point can).
- **Fix:** Added an explicit "Structural signal rule" section to `RUBRIC`
  (`app/judge.py`), on top of the existing 0-3 levels: (1) prose evidence
  AND a structural signal together for the same skill -> score 3 regardless
  of what the prose alone would earn; (2) a structural signal alone, with no
  descriptive prose anywhere in the provided chunks -> score at least 2,
  never 0 or 1. The rubric also now instructs the judge to name which case
  applied directly in its justification text.
- **Verification:** Ran two live test cases through `judge_all_skills`
  (batched, 1 Gemini call): (1) a single chunk with only
  `"Structural signals found: Dockerfile present, CI workflow present"` and
  no prose mentioning Docker anywhere -> scored 2, justification: "Structural
  signal found (Dockerfile present) in repo infra-tools's file listing, no
  descriptive text - scored 2 based on verified artifact alone." (2) the same
  structural-signal chunk plus a prose bullet describing containerizing a
  service with Docker -> scored 3, justification: "Structural signal
  (Dockerfile present) AND descriptive text both confirm this skill - scored
  3." Both match the new rule exactly, including citing the correct case in
  the justification text.

- **Date:** 2026-08-29
- **Issue:** A Docker deployment smoke test (real containerized `/analyze`
  call, `test_resume.txt` + `test_jd.txt` + GitHub evidence) surfaced a
  regression from the structural-signal rubric rule added earlier the same
  day: `requirements.txt present` / `package.json present` were boolean
  "file exists" signals with no information about what's actually in the
  file, but the rubric treated any structural signal as confirming evidence
  for whatever skill it happened to sit next to in a chunk. Combined with
  the already-known small-GitHub-corpus retrieval issue (nearly every skill's
  search returns all fetched repo chunks regardless of relevance), this meant
  a generic `requirements.txt present` line was inflating unrelated skills
  to score 3 - e.g. "Documentation skills" and "API-based integrations" both
  cited "Structural signal (requirements.txt) AND descriptive README text
  both confirm this skill - scored 3", even though a requirements.txt has
  nothing to do with documentation or specific API integrations.
- **Root cause:** `_detect_structural_signals` (`app/github_evidence.py`)
  only checked whether `requirements.txt`/`package.json` existed at a repo's
  top level, discarding the one piece of information that would make it
  actually useful - which packages are listed inside it.
- **Fix:** `app/github_evidence.py` now fetches and parses the real contents
  of `requirements.txt` (`_parse_requirements_txt`, stripping comments,
  version pins, extras, and environment markers) and `package.json`'s
  `dependencies`/`devDependencies` (`_parse_package_json`), and includes the
  actual package names in the evidence chunk as a clearly-labeled line, e.g.
  `"Verified dependencies (from requirements.txt): fastapi, sentence-
  transformers, rank-bm25, pydantic"` - distinguishable from README prose and
  from the (now dependency-content-free) `Structural signals found: ...`
  line. `RUBRIC` (`app/judge.py`) was updated to treat an individual listed
  package name as strong evidence only for the specific skill it actually
  implements (e.g. `fastapi` -> REST APIs/backend, `sentence-transformers` ->
  embeddings/semantic search), and explicitly states that merely having *some*
  dependency list, with no package name relevant to the skill being scored,
  is not evidence for that skill.
- **Verification:** Rebuilt the Docker image and re-ran the identical
  containerized `/analyze` smoke test. Skills previously inflated by the
  generic signal dropped to their genuinely-earned score with no dependency-
  file mention at all (Documentation: 3 -> 2, justification now cites only
  the README's own documentation content; RAG: 3 -> 2, justification now
  cites only the resume's CLIP/Qdrant bullet). Skills with a real matching
  package kept a high score but for a legitimate, specific reason instead of
  a generic one - e.g. API-based integrations stayed at 3, now justified as
  "Verified dependency 'fastapi' in requirements.txt AND descriptive README
  text detailing building an API service both confirm this skill - scored 3
  based on prose and verified package combined."

- **Date:** 2026-08-29
- **Issue:** In production on Render's free tier, a `POST /analyze` request
  (resume + JD + `github_username`) returned a 502. Render's logs showed the
  server process restarting mid-request ("Detected a new open port",
  "Started server process [1]" reappearing), consistent with the platform
  OOM-killing and restarting the container - Render's free tier caps memory
  at 512MB.
- **Root cause (confirmed, not assumed):** Reproduced the exact failure by
  running the (pre-fix) Docker image with `--memory=512m --memory-swap=512m`
  (Render's exact cap). Got a real, confirmed OOM kill
  (`OOMKilled=true`, exit 137) under modest extra memory pressure; a single
  `/analyze` call alone already used 454.8MB/512MB (89%), and the container
  briefly touched 507MB (99%) at idle startup - essentially zero headroom.
  An import-by-import measurement (`/proc/self/status` VmRSS in an isolated
  process) isolated the cause: `import torch` alone costs +176MB RSS, and
  `import sentence_transformers` (before loading any model) costs another
  +161MB - ~340MB of baseline memory before a single request or GitHub API
  call happens. Capping torch's thread pool (`OMP_NUM_THREADS=1` etc.) was
  tested and made no measurable difference. GitHub evidence fetching was
  ruled out as a contributor - each chunk is capped at ~2KB, negligible next
  to the ~450MB baseline.
- **Fix:** Replaced the sentence-transformers/PyTorch embedding backend in
  `app/retrieval.py` with ONNX Runtime + the standalone `tokenizers` library,
  using the pre-converted `onnx/model.onnx` export already published in the
  `sentence-transformers/all-MiniLM-L6-v2` HF repo (fp32, not quantized - no
  precision tradeoff). `_OnnxEmbedder` replicates sentence-transformers' own
  mean-pooling (attention-mask-weighted average of token embeddings) +
  L2-normalization by hand, and keeps the exact same `_get_embedder()` /
  `.encode(texts, normalize_embeddings=...)` interface the rest of the
  codebase (including `eval/run_eval.py`) already depends on, so no other
  file needed to change. `requirements.txt` drops `sentence-transformers`
  (and, transitively, `torch`) for `onnxruntime` + `tokenizers`. The
  Dockerfile now downloads `model.onnx` + `tokenizer.json` directly at build
  time (plain `urllib.request`, no heavy client library needed) instead of
  installing CPU torch and invoking `sentence_transformers.SentenceTransformer`.
- **Verification (before implementing further):** (1) Raw embedding
  equivalence - cosine similarity of 1.00000 between the old and new
  backends across 9 held-out test sentences including "Kubernetes" and real
  evidence-chunk text. (2) Full hybrid-search (BM25+semantic+RRF) ranking
  equivalence - ran the actual `app.retrieval.EvidenceIndex` (now
  ONNX-backed) against a fresh sentence-transformers baseline built
  independently, over the real resume+GitHub evidence set, for 8 real
  project queries ("Kubernetes", "Software Deployment", "RAG",
  "Documentation", "Python", "Communication", "Algorithms", "API-based
  integrations") - identical top-5 chunk order and RRF scores matching to
  4 decimal places on every query. Only after both checks passed cleanly was
  the Docker image rebuilt and requirements/Dockerfile changed.
- **Result:** Image size dropped 2.2GB -> 701MB. Import-by-import RSS:
  `onnxruntime` +17.5MB and `tokenizers` +3.2MB (vs. torch's +176MB and
  sentence-transformers' +161MB); loading the ONNX model + running inference
  add ~135MB more, for a ~212MB steady-state total (vs. ~494MB before) - a
  ~57% reduction. Idle memory under the same `--memory=512m` cap dropped
  from 387-507MB to 84.7MB (17%), since the model now loads lazily on first
  request instead of `import`-time. A real `/analyze` call (resume + JD +
  GitHub evidence) under the same 512MB cap peaked at 256.8MB (50%) - half
  the previous 454.8MB (89%) - and a second consecutive call only added
  ~15MB, confirming no unbounded growth. No OOM kill; container stayed
  healthy through both requests.

- **Date:** 2026-08-29
- **Issue:** A separate production 502/restart was reported after the ONNX
  fix above, and it wasn't clear whether it was a repeat of the earlier OOM
  (memory creeping back up since the fix) or a genuinely different failure
  mode, e.g. the request simply taking too long. Investigated two angles:
  (1) whether the app records exact per-request duration anywhere, and (2)
  whether Render's free-tier request timeout is a fixed platform limit or
  something we could tune.
- **Finding (duration visibility):** `app/main.py` had no request timing
  instrumentation at all - no middleware, no per-request logging - so there
  was no way to get an exact duration for the request that preceded the
  restart, only a rough bound from diffing Render's log timestamps around
  the restart event. Added a `@app.middleware("http")` handler
  (`log_request_duration`) that logs elapsed ms per request on completion.
  First attempt logged via `logging.getLogger("uvicorn.access")`, which
  crashed uvicorn's `AccessFormatter` (`ValueError: not enough values to
  unpack`) because that formatter expects uvicorn's own 5-tuple access-log
  args, not a plain message - caught by uvicorn so it didn't take down the
  request, but the log line was corrupted. Second attempt used a fresh
  `logging.getLogger("app.timing")`, which produced no output at all (Python
  root logger has no handler by default, and uvicorn only wires handlers to
  its own `uvicorn`/`uvicorn.error`/`uvicorn.access` loggers). Fixed by
  logging to `uvicorn.error` (uvicorn's general-purpose log channel, despite
  the name - already has an INFO-level handler with a plain `DefaultFormatter`
  wired up), verified end-to-end in a container: `INFO:     POST /analyze
  completed in 65647ms` printed cleanly. Deliberate tradeoff: this only logs
  on completion, so a request that dies mid-flight (OOM kill, crash) never
  produces the "completed" line - that absence, cross-referenced against
  Render's platform-level "request started" log line, is itself the
  crash-vs-timeout signal going forward.
- **Finding (Render timeout policy):** Per Render's own docs, HTTP requests
  may run up to 100 minutes - a fixed platform ceiling, not a per-service
  configurable setting, and far above any observed `/analyze` duration.
  Render's Free tier docs separately state "Render might restart a Free web
  service at any time," i.e. free-tier restarts aren't necessarily tied to
  any single request's duration at all.
- **Finding (memory re-check):** Rebuilt the current image and re-ran the
  identical `--memory=512m --memory-swap=512m` test from the OOM
  investigation above: idle RSS 73.97MB (14.45%, vs. 84.7MB/17% previously -
  consistent within noise), and a real `/analyze` call (resume + full
  `test_jd.txt`, ~30 skills + GitHub evidence) peaked at 232-257MiB
  (45-50%), matching the prior 256.8MB (50%) measurement almost exactly.
  **Memory has not crept back up** - the ONNX fix is holding, and this rules
  out OOM as the cause of the new restart.
- **Observation worth following up:** the two live `/analyze` runs against
  the full `test_jd.txt` (a JD with ~30 required/preferred skills, notably
  larger than earlier test runs) took 65.6s and 80.5s end-to-end - all
  Gemini-call time (extraction + batched judge), not memory pressure. Not
  yet confirmed as the restart's cause, but worth checking next: whether the
  specific JD/resume pair that triggered the restart was unusually large,
  and whether Render's free-tier CPU throttling (not just its memory cap)
  could be stretching a normally-fast call past whatever the client or a
  reverse proxy in front of it enforces - since 65-80s is well within
  Render's own 100-minute limit but could exceed a shorter client-side or
  browser fetch timeout.

- **Date:** 2026-08-29
- **Issue:** While attempting a fresh live `/analyze` run (to collect a
  current list of zero-evidence skills for self-healing-retrieval test
  planning), the request took 82.45s and then failed with a raw 500 instead
  of the fast, clear `DailyQuotaExhaustedError` the daily-quota fix
  (documented earlier in this log) was supposed to guarantee.
- **Root cause:** The daily-quota fail-fast fix was only ever applied to
  `judge.py`'s `_generate_with_retry` (used by `judge_all_skills`).
  `extraction.py` has its own separate, near-identical `_generate_with_retry`
  (used by `extract_resume_claims_and_evidence` and `extract_job_skills`)
  that never got the same fix - it still retried every 429, daily-quota or
  not, the old way (5 attempts x 15s sleep). The 82.45s duration matches
  this exactly: ~75s of retry sleeps before the final attempt's raw
  `google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED` (quotaId
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, confirming a real,
  currently-exhausted daily cap, not a transient rate limit) propagated
  uncaught to a 500.
- **Fix:** Moved `DailyQuotaExhaustedError` and the daily-vs-per-minute
  quotaId check (renamed from the module-private `_is_daily_quota_error` to
  the now cross-module `is_daily_quota_error`, since `extraction.py` imports
  it from `judge.py` rather than duplicating it - keeping one exception type
  and one detection function shared across both call sites instead of two
  independent copies of the same logic) into a shared import:
  `extraction.py` now does `from app.judge import DailyQuotaExhaustedError,
  is_daily_quota_error` and applies the identical daily-quota check inside
  its own `_generate_with_retry` before falling through to the existing
  retry loop.
- **Verification:** Could not reproduce against a second live daily-quota
  exhaustion today (would require burning another day's cap just to test
  the fix), so verified with a mocked `google.genai.errors.ClientError`
  instead: (1) a daily-quota-shaped 429 (quotaId
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier`) now raises
  `DailyQuotaExhaustedError` after exactly 1 call attempt and 0 retry
  sleeps (down from 5 attempts / ~75s); (2) a per-minute-quota-shaped 429
  (quotaId `GenerateRequestsPerMinutePerProjectPerModel-FreeTier`) still
  retries with the existing 15s backoff and recovers on the next attempt,
  confirming the fix only short-circuits the daily-cap case and doesn't
  change behavior for genuine transient rate limits. Both `extraction.py`
  and `judge.py` now import the same `DailyQuotaExhaustedError` class
  (`extraction.DailyQuotaExhaustedError is judge.DailyQuotaExhaustedError`
  -> `True`), so a caller can catch one exception type regardless of which
  module's Gemini call hit the daily cap. Full live end-to-end confirmation
  (a fresh `/analyze` run producing a current zero-evidence-skill list, the
  original goal) is still blocked on the day's quota resetting or a new API
  key.

- **Date:** 2026-08-29
- **Issue:** Before shipping `search_with_fallback()` (a proposed
  self-healing addition to hybrid retrieval: if a skill's literal wording
  scores below `MIN_RELEVANCE_SCORE`, ask Gemini for alternate phrasings
  via a new `reformulate_query()` and retry), tested it against 3 known
  cases first, using the real 13-bullet evidence set from `test_resume.txt`
  (no GitHub evidence) and real `reformulate_query()` Gemini calls (not
  hand-picked phrasings): `Observability` (expected to recover - a resume
  bullet describes "daily error monitoring and root-cause analysis" without
  using the word), `Reranking` (expected NOT to recover - an earlier probe
  showed a literal-vs-hand-picked-alternate comparison crossing
  `MIN_RELEVANCE_SCORE` by matching a topically-unrelated chunk), and
  `MLOps` (expected NOT to recover - `eval/labeled_pairs.json` has a human
  label calling an earlier system version's MLOps evidence a "false
  positive," since the specific claimed tools - Langfuse, OpenTelemetry,
  Chronosphere, SigNoz - have zero evidence bullets).
- **First implementation and why it wasn't accepted as-is:** Initial
  design only required a reformulated query's score to clear
  `MIN_RELEVANCE_SCORE`, same as a normal search. A prior probe (same
  session, hand-picked alternate phrasings rather than live
  `reformulate_query()` output) showed every candidate query's top fused
  score - relevant or not - clustering within a ~0.0013 band (0.0315-0.0328)
  around `MIN_RELEVANCE_SCORE` (0.0320), on this project's ~13-chunk resume
  evidence corpus: with `RRF_K=60`, `1/(RRF_K + rank + 1)` fusion means the
  top-ranked chunk for almost any query lands near rank 0-1 in both BM25
  and semantic rankings purely from small corpus size, so the top fused
  score barely tracks actual relevance. `Reranking` demonstrated this
  concretely: a hand-picked alternate phrasing ("ranking retrieved results
  by relevance") crossed the threshold by matching a chunk about
  "vision-based document classification" - unrelated to reranking. Fixed,
  before ever merging, by adding `QUERY_REFORMULATION_MARGIN` (0.0050, ~4x
  the measured noise spread): a reformulated query's score must clear
  `MIN_RELEVANCE_SCORE` **and** beat the original query's own score by more
  than this margin, not just cross the fixed line.
- **Verification result: the margin fix does NOT separate the 3 cases as
  intended, and this is a real, unresolved finding, not a margin-tuning
  problem:**

  | Skill | Score A (literal) | Best Score B (reformulated) | B - A | Expected | Actual |
  |---|---|---|---|---|---|
  | Observability | 0.0317 | 0.0323 | **0.0006** | recover | did not recover |
  | Reranking | 0.0318 | 0.0328 | 0.0010 | must not recover | did not recover |
  | MLOps | 0.0315 | 0.0325 | 0.0010 | must not recover | did not recover |

  `QUERY_REFORMULATION_MARGIN=0.0050` correctly rejected all three (none of
  the deltas reach 0.0050), so no false positive slipped through - but it
  also rejected `Observability`, the one case that should have recovered.
  Worse: **no margin value could have separated these correctly.**
  `Observability`'s delta (0.0006) is the *smallest* of the three, while
  `Reranking` and `MLOps` - the two cases that must NOT recover - both show
  a *larger* delta (0.0010) than the genuine wording-mismatch case. Any
  margin low enough to accept `Observability` (< 0.0006) would also accept
  both cases it must reject; any margin that rejects `Reranking`/`MLOps`
  (>= 0.0010) also rejects `Observability`. The ranking of score deltas is
  inverted relative to what the margin mechanism assumes. Likely cause
  (not yet independently verified): `Observability`'s literal-query score
  (0.0317) already sits close to the ceiling other queries top out at on
  this corpus, leaving little headroom to "improve" on retry, while
  `Reranking`/`MLOps`'s lower starting scores (0.0318, 0.0315) leave more
  headroom for an unrelated-but-differently-ranked chunk to produce a
  larger, noise-driven delta on retry - i.e. the margin is measuring how
  much room a query's starting score had to move, not how relevant the
  reformulated match actually is.
- **Status:** NOT shipped. `search_with_fallback()`, `reformulate_query()`,
  and `QUERY_REFORMULATION_MARGIN` are implemented in `app/retrieval.py` /
  `app/extraction.py` but not wired into `main.py`'s live `/analyze`
  pipeline - this pre-ship test caught that the margin-over-original-score
  mechanism, as specified, doesn't work on this corpus size before it could
  reach production and silently reintroduce the exact false-positive
  pattern (`eval/labeled_pairs.json`'s MLOps case) the feature was meant to
  avoid. Needs a different accept/reject signal before shipping - candidates
  not yet tried: a semantic-only cosine-similarity floor computed
  independently of RRF rank position (rank-based fusion is the specific
  mechanism producing the corpus-size noise), or dropping the retrieval-side
  gate entirely and relying on the downstream LLM judge to reject an
  irrelevant cited chunk instead.

- **Date:** 2026-08-29
- **Issue:** Follow-up to the entry above. Tried the first untried
  candidate - a margin over semantic-only (cosine similarity) top-1 score
  instead of the RRF-fused score, since RRF's rank-based fusion was the
  specific thing implicated in the inverted ordering. Measured the
  semantic-only score spread across the same 3 cases first (reusing the
  exact reformulated phrasings already obtained from live
  `reformulate_query()` calls in the prior entry, to keep this a controlled
  comparison of the scoring mechanism against the same inputs, and to avoid
  spending more of the daily quota) before picking any margin, per the
  plan.
- **Result: numeric ordering technically separated the 3 cases this time -
  but a deeper check showed the mechanism still isn't trustworthy:**

  | Skill | sem_A (literal) | Best sem_B (reformulated) | Delta | Ordering vs. RRF attempt |
  |---|---|---|---|---|
  | Observability | 0.1455 | 0.3970 (`"Distributed tracing"`) | **+0.2515 (largest)** | inverted (was smallest) |
  | Reranking | 0.2187 | 0.4048 (`"Two-stage retrieval"`) | +0.1861 (smallest) | inverted (was tied-largest) |
  | MLOps | 0.2387 | 0.4830 (`"ML Pipeline Automation"`) | +0.2443 | inverted (was tied-largest) |

  By delta magnitude alone, `Observability` now has the largest
  improvement, correctly ahead of both cases that shouldn't recover - the
  RRF-based inversion is gone. But inspecting which chunk each reformulated
  query actually top-ranks (top-3 semantic matches per query, printed and
  checked by hand) found two deeper problems this metric alone couldn't
  see: (1) for `Observability`, the real "daily error monitoring and
  root-cause analysis" bullet (chunk index 6) never ranked #1 for the
  literal query OR any of the 3 real reformulated phrasings tried
  (`"Application performance monitoring"`, `"Distributed tracing"`,
  `"Metrics logging and alerting"`) - it topped out at rank 2. The
  highest-scoring chunk for the winning reformulation was instead an
  unrelated Playwright/Redis/Kafka async-workflow bullet - i.e. the
  numerically "best" case's own top-1 recovery would have been *wrong
  evidence*, even though its score delta correctly ranked highest. (2) For
  `Reranking`, the real reformulated phrasing `"Two-stage retrieval"`
  legitimately top-matched a CLIP+Qdrant vector-search chunk that IS
  plausibly relevant to reranking - undermining `Reranking`'s status as an
  unambiguous "shouldn't recover" negative control under this run's actual
  Gemini-generated phrasings (vs. the hand-picked phrasing used in the
  original RRF-based probe, which had clearly recovered irrelevant
  content). Conclusion: a top-1 chunk-similarity score, whichever metric
  it's computed from (RRF-fused or semantic-only), isn't a reliable enough
  proxy for genuine topical relevance to gate on by itself on this
  project's small evidence corpus - the score magnitude and the content
  correctness of what it's pointing at came apart in both attempts.
- **Fix (per plan: stop margin-tuning, try the LLM-judge-based fallback
  instead):** Rewrote `search_with_fallback()` (`app/retrieval.py`) to drop
  the retrieval-side numeric gate on the reformulation tier entirely.
  Behavior now: literal skill search still gates on `MIN_RELEVANCE_SCORE`
  as before (unchanged, cheap, and never shown to be the problem); once
  that fails, EVERY reformulated attempt runs (no early-accept short
  circuit), the single best-scoring one by semantic-only top score is
  selected, and its top-`k` chunks (not just its top-1) are passed straight
  to the downstream LLM judge with no further filtering - the judge decides
  relevance from actual chunk content, not a similarity number. Removed the
  now-unused `QUERY_REFORMULATION_MARGIN` constant and its comment block;
  added a new `EvidenceIndex._search_semantic()` (cosine-similarity-only,
  no BM25/RRF) and `_search_scores()` (returns both the fused and
  semantic-only lists from one embedding call, so the literal query's
  semantic score - not currently used by this version, but kept for the
  original score-A lookup - doesn't need a second `encode()`).
  `github_index`'s fallback tier is unchanged (still gated normally via
  `.search()`) since it's a separately-proven code path, not implicated in
  either failed experiment above.
- **Verification: ran the actual rewritten `search_with_fallback()` end to
  end (fresh live `reformulate_query()` calls, not reused phrasings this
  time) followed by the real `judge.judge_all_skills()` on the recovered
  chunks - the full intended pipeline, not just the retrieval layer:**

  | Skill | Static (current) score | Self-healing score | Judge's justification |
  |---|---|---|---|
  | Observability | 0 (no chunks retrieved) | **2** | "Chunk 3 demonstrates the skill applied to a real task through daily error monitoring, root-cause analysis..." - correctly cited chunk 6, the genuine evidence bullet |
  | Reranking | 0 | 0 | "None of the provided chunks mention or demonstrate the use of reranking algorithms or techniques" |
  | MLOps | 0 | 0 | "None of the provided chunks demonstrate MLOps practices such as model tracking, automated deployment pipelines, or model monitoring infrastructure" - matches the `eval/labeled_pairs.json` human label exactly |

  All three outcomes now match expectation: `Observability` recovered
  genuine evidence and was scored correctly, while `Reranking` and `MLOps`
  were correctly rejected despite each having a superficially
  plausible-looking chunk in their top-5 (the CLIP/Qdrant vector-search
  chunk for `Reranking`, a generic extraction-pipeline chunk for `MLOps`) -
  confirming the judge, given full chunk content and the actual skill name
  together, succeeds at exactly the disambiguation that neither top-1
  numeric metric could do reliably on its own. Notably, `Observability`'s
  correct outcome held even though its own top-ranked chunk was still the
  wrong one (the Playwright/Kafka bullet, per the finding above) - passing
  the full top-5 (not just top-1) gave the judge enough surrounding context
  to find and cite the right chunk anyway.
- **Status:** Implemented and verified in `app/retrieval.py` /
  `app/extraction.py`. Still not wired into `main.py`'s live `/analyze`
  pipeline - that integration, plus a broader eval-set run (beyond these 3
  hand-picked cases) to confirm no regression on skills the static search
  already handles correctly, is the next step before shipping.

- **Date:** 2026-08-29
- **Issue:** Wire `search_with_fallback()` into `main.py`'s live
  `/analyze` pipeline (`run_analysis()`) and run the full `eval/
  labeled_pairs.json` set through `eval/run_eval.py`'s comparison logic to
  confirm no regression before shipping.
- **Bug caught before wiring in:** `search_with_fallback()` had no
  short-circuit for an empty `evidence_index` (e.g. `github_index` when no
  `github_username` is given - the common case). Without a guard, every
  skill's github-side call would score 0 on an empty corpus, fall below
  `MIN_RELEVANCE_SCORE`, and burn a `reformulate_query()` Gemini call
  anyway - at ~15-19 skills per `/analyze` request against the 20-request/
  day free-tier cap, wiring this in unguarded would have let a single
  request exhaust the entire day's quota by itself. Fixed: added an
  early-return in `search_with_fallback()` for `not evidence_index.chunks`,
  skipping straight to the `github_index` fallback tier (if any) with no
  `reformulate_query()` call.
- **Integration:** `main.py`'s `run_analysis()` replaced its two direct
  `resume_index.search(skill, ...)` / `github_index.search(skill, ...)`
  calls with `search_with_fallback(skill, resume_index, ...)` /
  `search_with_fallback(skill, github_index, ...)`, still merged via the
  existing `merge_ranked_chunks()` - preserving the corpus-pollution-safe
  dual-independent-index architecture (see the 2026-08-29 GitHub-evidence
  entry above): each source still gets its own self-healing retry, and
  results are only combined after scoring, never before.
- **Eval run: hit the daily quota wall mid-run** on the first attempt
  (`DailyQuotaExhaustedError` raised immediately and cleanly at the
  baseline judge call, confirming the fail-fast fix from earlier today
  works correctly under real use, not just a mocked test) - today's 20-
  request budget had been fully spent by this session's cumulative testing.
  Resumed after the user supplied a genuinely fresh API key (different
  underlying project this time, confirmed via a working single-call smoke
  test before spending it on the full eval).
- **Method:** To isolate the retrieval-mechanism change from resume/JD
  extraction's already-documented run-to-run non-determinism (see the
  2026-08-28 entries above), `extract_resume_claims_and_evidence()` was
  called ONCE and its result shared between two separate `judge_all_skills()`
  runs - one using the old static `.search()` + merge (baseline), one using
  `search_with_fallback()` + merge (updated) - against the same cached
  `eval/job_skills.json` (18 skills: 15 required + 3 preferred).
- **Raw result: 4 of 12 labeled skills changed score (2 apparent
  regressions, 2 apparent improvements), baseline and updated tied at 9/12
  label agreement.** Investigated each changed skill individually before
  treating this as a real effect - re-ran retrieval alone (no judge call,
  free) for `Computer Science`, `Software Deployment`, `Data Structures`,
  and `RAG`, comparing `.search()` + merge against
  `search_with_fallback()` + merge chunk-for-chunk, using the exact same
  shared resume evidence:

  | Skill | static chunks == fallback chunks? |
  |---|---|
  | Computer Science | **identical** |
  | Software Deployment | **identical** |
  | Data Structures | **identical** |
  | RAG | **identical** |

  All four are byte-identical. **None of the observed score changes are
  caused by `search_with_fallback()` - they are pure judge non-determinism**
  between the baseline and updated judge calls (two separate Gemini API
  calls on identical input evidence), the same effect already documented in
  the 2026-08-28 "re-running eval produces different results" entry above.
  This was confirmed structurally, not just inferred: since the chunks sent
  to the judge were provably identical, any score difference could only
  come from the judge call itself.
- **Why retrieval never actually differed on this eval set:** every one of
  the 12 labeled skills' literal, system-extracted wording (from
  `eval/job_skills.json`, e.g. `"ML Operations"`, not the abbreviated
  `"MLOps"` used in the human label text) already cleared
  `MIN_RELEVANCE_SCORE` on the literal search alone, so
  `search_with_fallback()` took its early-return path and never invoked
  `reformulate_query()` for any of them. Confirmed directly: literal `"ML
  Operations"` scores 0.0320 (>= threshold, passes), while literal
  `"MLOps"` (the exact string tested standalone in the entry above) scores
  0.0315 (< threshold, triggers fallback) - same underlying skill, two
  different literal strings, two different outcomes. **This eval run
  proves no regression by construction** (retrieval is provably unchanged
  for every skill it covers) but does not exercise the new fallback code
  path at all - that exercise remains the standalone `Observability` /
  `Reranking` / `MLOps` test from the entry above, which did trigger it and
  produced 3/3 correct outcomes via the real judge.
- **Conclusion:** No regression, confirmed at the retrieval level (stronger
  than a judge-score comparison, since it rules out judge noise entirely).
  No measurable change either, on this particular labeled set, since none
  of its skills' literal wording falls below threshold - self-healing is a
  no-op whenever the static search would already have succeeded, exactly as
  designed. The demonstrated benefit remains the earlier `Observability`
  recovery; this eval run's contribution is confirming that benefit doesn't
  come at the cost of disturbing skills static search already handles.
- **Status:** Shipped. `search_with_fallback()` is now live in `main.py`'s
  `/analyze` pipeline for both the resume and GitHub evidence indexes.
