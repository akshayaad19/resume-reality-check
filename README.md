# Resume Reality Check

**A resume–job description evidence verifier: it doesn't just check if you claimed a skill — it checks whether you can actually prove it.**

Most resume-matching tools compare keywords. This one separates what a candidate *claims* from what their actual experience *demonstrates*, using hybrid retrieval (keyword + semantic search) and an LLM-as-judge scored against a fixed rubric — then validates its own accuracy against a hand-labeled evaluation set.

On my own resume, this system found that I claimed "Python" in my skills section, but **zero** of my evidence bullets described using it — because the language was only ever named in a project's tech-stack tag line, never in the bullet text itself. That single finding kicked off five real debugging investigations, documented below.

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
                    Per-skill output table
```

**Tech stack:** FastAPI · Gemini (gemini-3.6-flash) · sentence-transformers · rank-bm25 · Pydantic structured outputs

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

---

## Known limitations (stated honestly, not hidden)

- **Single-resume eval set.** 12 hand-labeled pairs, all from one resume against one JD. The methodology is sound; generalization to other resumes hasn't been tested yet.
- **In-memory indexing only.** No persistent vector database (e.g. Qdrant) — every request re-embeds the same evidence from scratch. Fine for a single resume at low volume; would need real indexing to scale.
- **No GitHub evidence layer yet.** The original design includes verifying claims against a candidate's public GitHub activity (structural signals like Dockerfile/k8s presence, plus README/commit content) as a second, harder-to-fake evidence source. Not yet built.
- **A single embedding-similarity threshold has a real, provable limit.** Diagnostic testing showed that topically-adjacent-but-distinct skills can score similarly to genuine paraphrases using cosine similarity alone — no single global threshold fully resolves this. A more robust fix would use an LLM to disambiguate borderline cases rather than relying purely on a numeric cutoff.
- **No authentication, rate limiting, or deployment infrastructure yet.** This is a validated core pipeline, not a production system.

---

## Planned next

- [ ] GitHub evidence layer — verify claims against public repos (structural checks: Dockerfile, k8s manifests, CI configs; textual checks: README/commit content via the same hybrid retrieval pipeline)
- [ ] Self-healing multi-agent retrieval — if initial retrieval confidence is low, an agent reformulates the query and retries, or falls back to GitHub evidence, before giving up and reporting no evidence (a dynamic recovery strategy, replacing the current static reject-on-low-confidence approach)
- [ ] Expand the eval set with 1–2 more resume/JD pairs to test generalization beyond a single resume
- [ ] Minimal frontend (currently interact via FastAPI's auto-generated `/docs`)
- [ ] Persistent vector database (Qdrant) to replace in-memory indexing
- [ ] MCP wrapper, so the scoring engine can be called directly as a tool from Claude Desktop/Code
- [ ] Basic auth, rate limiting, deployment (Docker, hosted)

---

## Try it

Currently runs locally / via Codespace — interactive API docs at `/docs` after starting the server:

```bash
uvicorn app.main:app --reload
```

*(Live deployed demo link — coming soon.)*