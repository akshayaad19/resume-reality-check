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
