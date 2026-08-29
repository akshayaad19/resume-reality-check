# Resume Reality Check

**A resume–job description evidence verifier: it doesn't just check if you claimed a skill — it checks whether you can actually prove it.**

Most resume-matching tools compare keywords. This one separates what a candidate *claims* from what their actual experience *demonstrates*, using hybrid retrieval (keyword + semantic search) and an LLM-as-judge scored against a fixed rubric — then validates its own accuracy against a hand-labeled evaluation set.

On my own resume, this system found that I claimed "Python" in my skills section, but **zero** of my evidence bullets described using it — because the language was only ever named in a project's tech-stack tag line, never in the bullet text itself. That single finding kicked off nine real debugging investigations, documented below.

**Live demo:** [resume-reality-check-production.up.railway.app/docs](https://resume-reality-check-production.up.railway.app/docs) — first response can take up to ~2 minutes (multiple sequential LLM calls; see finding #9 for why this is deployed on Railway rather than Render).

---

## What it does

1. Upload a resume + a job description
2. The system splits the resume into **claims** (bare skill mentions, e.g. a skills list) and **evidence** (actual experience/project bullets) — a claim alone is never treated as proof
3. For each skill the job description requires, it searches the evidence using **hybrid retrieval**: BM25 (keyword) + sentence-embedding similarity (semantic), fused with Reciprocal Rank Fusion
4. An LLM judge scores the retrieved evidence 0–3 against a fixed rubric, citing the exact bullet that justified the score
5. Output: a per-skill table — skill, claimed-on-resume (yes/no), evidence score, and the cited proof

## Why this is different from a keyword matcher

A keyword matcher checks: *does the word "Kubernetes" appear on this resume?*
This checks: *is there a bullet point that actually describes using Kubernetes — and if the resume says "container orchestration" instead, can the system still find that connection?*

The claims-vs-evidence separation is enforced architecturally: the retrieval index is built **only** from evidence bullets. The claims list is never searched — it exists solely as a side-by-side comparison, so the output can say "claimed: yes, evidence: none" as a distinct, meaningful signal.

---

## Architecture

```
Resume (PDF/text) ──┐
                     ├─→ Parsing (pypdf) → clean text
Job Description ─────┘
                            │
                            ▼
              Extraction (Gemini, temperature=0)
         ┌──────────────────┴──────────────────┐
         ▼                                      ▼
  Resume: claims + evidence            JD: required + preferred skills
         │                                      │
         ▼                                      │
  EvidenceIndex (built once per request)        │
    - BM25 keyword index                        │
    - sentence-transformer embeddings           │
    (all-MiniLM-L6-v2, local, no API cost)       │
         │                                      │
         └──────────────◄───────────────────────┘
                for each JD skill:
                hybrid search (BM25 + semantic)
                → RRF fusion → top-k chunks
                → minimum relevance threshold
                (weak matches rejected, not forced)
                            │
                            ▼
              Batched LLM judge (Gemini, temperature=0)
              scores each skill 0-3 vs. a fixed rubric,
              cites the chunk that justified the score
                            │
                            ▼
              claimed_on_resume: fuzzy semantic match
              (embedding similarity, threshold 0.50)
              against the claims list — independent of
              the score, used only for display
                            │
                            ▼
              GitHub evidence (optional, if username given)
              - fetched via GitHub REST API, own separate
                EvidenceIndex (BM25 + embeddings), so its
                corpus statistics never mix with the resume's
              - two evidence types per repo: README prose
                (searched like any evidence bullet) + verified
                structural signals (Dockerfile, k8s/, CI
                workflow, actual dependency names parsed from
                requirements.txt/package.json) — structural
                signals are rubric-weighted higher than prose,
                since they're checkable artifacts, not claims
              - results merged with resume results only AFTER
                each source's own retrieval completes
                            │
                            ▼
                    Per-skill output table
```

**Tech stack:** FastAPI · Gemini (gemini-3.6-flash) · ONNX Runtime (embeddings) · rank-bm25 · Pydantic structured outputs · Docker · deployed on Railway

---

## What's built and validated (today's work)

### Core pipeline
- PDF/text parsing
- Claims vs. evidence extraction with structured output (Pydantic schemas)
- Hybrid retrieval: BM25 + semantic search, fused with RRF (k=60)
- Batched LLM-as-judge scoring (all skills in one call, to respect API rate limits)
- Per-skill output table with cited evidence

### Evaluation methodology
- 12 hand-labeled skill/evidence pairs (`eval/labeled_pairs.json`) — my own judgment, recorded independently, used as ground truth
- `eval/run_eval.py` — automatically compares fresh pipeline output against these labels, reports agreement %
- This is a fixed, read-only reference file; the eval script never modifies it. It's the "answer key" I check new code changes against.

### Five real bugs found and fixed, in order

**1. Evidence bullets didn't inherit their project's tech-stack context**
Resume bullets like *"Developed the retrieval flow using CLIP embeddings and Qdrant vector search"* never literally say "Python" — even though the project header explicitly tags it. Before the fix, "Python" scored 0/3 despite being used in every project on the resume.
**Fix:** Prepend each project's tech-stack tags to its evidence bullets during extraction (`[Technologies used: Python, GPT-4o, CLIP, Qdrant, AWS S3] Developed the retrieval flow...`).
**Result:** Python correctly scores 2/3, citing real project work.

**2. Retrieval always returns its "best available" chunks, even when nothing is genuinely relevant (low precision)**
Skills like "Data structures and algorithms" and "MLOps" scored false positives — the system cited a semantically-adjacent-but-irrelevant bullet (e.g. justifying "MLOps" from a bullet about weekly performance reporting) rather than reporting no real evidence.
**Fix:** Added a minimum relevance threshold to the hybrid search — if even the top-ranked chunk doesn't clear the bar, the skill is scored 0 locally instead of being forced through the judge.
**Result:** Score agreement with my hand-labeled set improved from ~77% to 90% on the labeled cases.
**Concept:** this is a deliberate precision-over-recall tradeoff — in this domain, a false positive (falsely claiming evidence exists) is more costly than a false negative (being conservative), because it actively misleads the person relying on the tool.

**3. Claims-matching used exact string comparison, missing legitimate matches**
`claimed_on_resume` showed "No" for skills the resume clearly claims — e.g., the JD says "RAG" but the resume says "Retrieval Augmented Generation (RAG)"; JD says "Cloud platforms," resume says "AWS/Azure." Exact-string `in` checks can't see these are the same thing.
**Fix:** Replaced exact matching with semantic similarity (the same embedding model already used for retrieval) between each JD skill and each resume claim.

**4. The first fuzzy-matching threshold (0.60) couldn't cleanly separate correct from incorrect matches**
Diagnostic scoring showed false positives and true positives *interleaved* on the score axis — no single threshold value could separate them. Investigating further revealed two of the "false positives" were actually mislabeled in my own initial ground truth: "Practical AI application" ↔ "Artificial Intelligence (AI)" and "Business process automation" ↔ "Intelligent Automation" are reasonably close matches, not false positives. Once corrected, a threshold of **0.50** cleanly separates all 8 labeled cases.
**Lesson:** the eval set caught an error in my own judgment, not just the system's — exactly what a proper evaluation process is supposed to do.

**5. Eval results were unstable run-to-run, traced to LLM temperature**
Re-running the eval script with zero code changes produced different numbers each time — job-skill extraction would sometimes split "Data structures and algorithms" into "Data Structures" alone, or "Communication, documentation, and collaboration" into just "Documentation," changing which fuzzy matches succeeded.
**Fix:** Set `temperature=0` on all extraction and judge LLM calls, so the same input reliably produces the same output.
**Why it matters beyond testing:** this affects the real product too — without it, a recruiter running the same resume twice could get different results between runs.

**6. `temperature=0` reduced but did not eliminate non-determinism — traced to a deeper, unfixable infrastructure cause**
Even with `temperature=0` on all three LLM calls, two consecutive eval runs still disagreed on 4 of 11 skills — one run's extracted skill set included "MLOps" but not "RAG"; the other included "RAG" but not "MLOps." Investigating further: this is a known, documented characteristic of hosted LLM APIs, not a bug in this codebase. Providers batch multiple users' requests together on shared infrastructure for efficiency; combined with Mixture-of-Experts routing (where only a subset of a model's specialized sub-networks handle any given request) and floating-point non-associativity (the order in which a computer sums numbers can produce tiny rounding differences), the *exact* probability distribution the model computes can vary slightly between otherwise-identical requests — occasionally enough to flip which token or expert is "most likely," even though `temperature=0` deterministically always selects the top-ranked option. This happens upstream of the model's own reasoning, at the routing/serving layer.
**Mitigation:** Since this can't be fully eliminated, I isolated it instead — cached the job-description skill extraction (the step most affected, since skill-boundary decisions often sit at genuine ambiguity thresholds) to a file after the first run, so subsequent eval runs compare against a fixed, stable reference rather than a fresh, potentially-different extraction each time. The resume-side extraction still calls the API fresh each run; this is a known, accepted remaining source of minor variation, not yet mitigated.
**Why this finding matters:** it's a real limit of building on hosted LLM infrastructure, not something more careful prompting or code can fully solve — the honest fix is to isolate and stabilize what you can (the eval process), rather than pursue perfect reproducibility that hosted APIs don't currently offer.

**7. Merging GitHub evidence into the resume's search index silently broke unrelated resume-only scores (BM25 corpus pollution)**
After adding GitHub evidence, five skills that previously had correct, well-evidenced resume-only scores (e.g. "Collaboration," "Cloud platforms") dropped to 0 with no cited evidence — even though nothing about the resume itself changed. Root cause: BM25's term weighting is computed relative to the *entire* corpus it's searching. Merging 3-4 GitHub chunks into the same pool as the 13 resume chunks shifted the corpus-wide word-rarity statistics for every query, including ones that had nothing to do with GitHub — pushing some previously-fine resume scores just below the relevance threshold from finding #2.
**Fix:** Build two fully separate `EvidenceIndex` objects — one for resume evidence, one for GitHub evidence — each with its own independent BM25 corpus. Search each independently, then merge only the already-ranked *results* afterward, never the raw chunks. This guarantees, by construction, that one source's statistics can never influence the other's scoring.
**Verified:** for all 18 job skills, resume-only retrieval now returns byte-identical results whether or not GitHub evidence is present alongside it — checked structurally, not just observed once.

**8. A generic "requirements.txt exists" signal was treated as strong evidence for unrelated skills**
The structural-signal rubric rule (any structural signal + descriptive prose → score 3) didn't check whether the signal was topically relevant to the specific skill being scored. Since `requirements.txt` exists in nearly every Python repo, it was inflating scores for "Documentation" and "API-based integrations" — skills that have nothing to do with having a dependency file.
**Fix:** Instead of a boolean "does requirements.txt exist" check, parse its actual contents and surface the specific package names found (e.g. "Verified dependencies: fastapi, sentence-transformers"). The rubric now only credits a *specific matching package name* as strong evidence (e.g. `fastapi` → "REST APIs"), not the file's mere existence. This is also a better signal in principle: dependency files are typically auto-generated (`pip freeze`), making the package list itself harder to fake than prose.
**Result:** previously-inflated scores dropped to correct levels with no file-existence claim; genuine matches (e.g. Python confirmed via `fastapi`/`pydantic` in a real requirements.txt) kept their score, now for the right, specific reason.

**9. Free-tier hosting killed long-running requests mid-flight — root-caused across four ruled-out theories before finding the real one**
Deployed to Render's free tier; `/analyze` calls (which make several sequential Gemini calls, 65-115s total) intermittently returned 502s. Investigated four hypotheses in order, each with real measurements, not guesses:
- *Memory (OOM)* — genuinely was the cause initially (idle memory sat at 507/512MB, 99%, just from importing PyTorch + sentence-transformers). **Fixed** by replacing the sentence-transformers/PyTorch embedding stack with ONNX Runtime running the same model (`all-MiniLM-L6-v2`) — same weights, same architecture, lighter execution engine. Verified retrieval rankings were unaffected (identical top-5 order and RRF scores to 4 decimal places across 8 real queries) before trusting the memory numbers. Idle memory dropped to 17%; peak under full load dropped to ~50%.
- *Render's platform timeout* — ruled out; Render's own docs state free-tier HTTP requests may run up to 100 minutes, far beyond any observed request duration.
- *Swagger UI's client-side timeout* — ruled out by calling `/analyze` with `curl` directly (no browser/Swagger UI involved) and an explicit generous `--max-time`; the request completed cleanly in ~79s with a well-formed response.
- *Memory regression after the ONNX fix* — re-measured under the exact same 512MB constraint with the current deployed code; idle and peak usage were statistically unchanged from the post-fix baseline. Memory was not the cause of this second round of failures.
With memory, platform timeout, and client timeout all directly ruled out by measurement, added request-duration logging middleware and confirmed via Render's own logs that failing requests died mid-flight (no completion line ever logged) shortly after the first LLM call started — not a graceful timeout, an abrupt kill.
**Resolution:** deployed the identical Docker image, unchanged, to Railway. The same request that reliably failed on Render completed successfully (200 OK, 114 seconds) on the first attempt. This cross-platform comparison is what confirms the failure was Render free-tier infrastructure behavior specifically (their own docs note free instances "might restart... at any time" for platform-side reasons) — not a bug in the application.
**Why this finding matters:** four hypotheses, four real tests, three ruled out with hard evidence before accepting the real cause — and the final proof was empirical (same artifact, different platform, different outcome), not theoretical.

---

## Known limitations (stated honestly, not hidden)

- **Single-resume eval set.** 12 hand-labeled pairs, all from one resume against one JD. The methodology is sound; generalization to other resumes hasn't been tested yet.
- **In-memory indexing only.** No persistent vector database (e.g. Qdrant) — every request re-embeds the same evidence from scratch. Fine for a single resume at low volume; would need real indexing to scale.
- **GitHub evidence only sees public repos**, by design — this matches exactly what a human recruiter clicking a candidate's profile link would also see (private repos are invisible to both). A candidate's best work, if kept private, isn't captured.
- **GitHub evidence's small corpus (a handful of repos) makes the relevance threshold less discriminating** than on the larger resume corpus — with only 3-4 documents, similarity scores cluster more tightly, so weak GitHub matches are less reliably filtered out than weak resume matches.
- **A single embedding-similarity threshold has a real, provable limit.** Diagnostic testing showed that topically-adjacent-but-distinct skills can score similarly to genuine paraphrases using cosine similarity alone — no single global threshold fully resolves this. A more robust fix would use an LLM to disambiguate borderline cases rather than relying purely on a numeric cutoff.
- **Perfect reproducibility is not achievable with hosted LLM APIs.** `temperature=0` reduces but does not eliminate output variation, due to serving-side batching and Mixture-of-Experts routing effects outside the application's control. The job-skill extraction step is cached to stabilize evaluation; resume-side extraction still has minor run-to-run variation, unmitigated.
- **Education/degree credentials aren't used as evidence at all.** Claims extraction only reads the resume's skills section; a relevant degree (e.g. a B.Tech in a CS-adjacent field) never becomes a claim or a piece of evidence, which can under-score fundamentals-related skills even when a relevant degree exists.
- **No authentication or rate limiting.** This is a validated pipeline with real deployment, not a hardened production system.

---

## Planned next

- [x] ~~GitHub evidence layer~~ — done (finding #7, #8)
- [x] ~~Deployment~~ — done (finding #9)
- [ ] Self-healing multi-agent retrieval — if initial retrieval confidence is low, an agent reformulates the query and retries, or falls back to GitHub evidence, before giving up and reporting no evidence (a dynamic recovery strategy, replacing the current static reject-on-low-confidence approach)
- [ ] Expand the eval set with 1–2 more resume/JD pairs to test generalization beyond a single resume
- [ ] Minimal frontend (currently interact via FastAPI's auto-generated `/docs`)
- [ ] Persistent vector database (Qdrant) to replace in-memory indexing
- [ ] MCP wrapper, so the scoring engine can be called directly as a tool from Claude Desktop/Code
- [ ] Basic auth and rate limiting
- [ ] Pull requirements.txt/education parsing improvements noted in Known Limitations

---

## Try it

**Live:** [resume-reality-check-production.up.railway.app/docs](https://resume-reality-check-production.up.railway.app/docs) — upload a resume, paste a job description, optionally add a GitHub username. First response takes up to ~2 minutes (several sequential LLM calls, no caching between requests yet).

**Locally:**
```bash
uvicorn app.main:app --reload
```
or via Docker:
```bash
docker build -t resume-reality-check .
docker run -p 8000:8000 --env-file .env resume-reality-check
```