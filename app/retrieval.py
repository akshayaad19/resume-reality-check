"""Chunking, embedding, and hybrid (BM25 + semantic) retrieval over resume evidence."""
from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import requests
from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer

# Embeddings run on ONNX Runtime + a standalone tokenizer rather than
# sentence-transformers/PyTorch: importing torch + sentence-transformers adds
# ~340MB of baseline RSS (measured), which alone exceeds the safety margin on
# a 512MB deployment target. The ONNX export below is numerically equivalent
# to the original PyTorch model (verified: cosine similarity 1.0 on held-out
# text, byte-identical hybrid-search rankings on this project's test
# queries) - see debugging-log.md.
EMBEDDING_MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
_ONNX_MODEL_URL = f"https://huggingface.co/{EMBEDDING_MODEL_REPO}/resolve/main/onnx/model.onnx"
_TOKENIZER_URL = f"https://huggingface.co/{EMBEDDING_MODEL_REPO}/resolve/main/tokenizer.json"
_MODEL_DIR = Path(os.environ.get(
    "EMBEDDING_MODEL_DIR", str(Path(__file__).resolve().parent.parent / ".cache" / "embedding_model")
))
MAX_SEQ_LENGTH = 256
RRF_K = 60
MIN_RELEVANCE_SCORE = 0.0320
CLAIM_MATCH_THRESHOLD = 0.50

_logger = logging.getLogger("uvicorn.error")


def _download(url: str, dest: Path) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)


def _ensure_model_files() -> Tuple[Path, Path]:
    """Downloads the ONNX model + tokenizer once and caches them under
    _MODEL_DIR. In the Docker image these are pre-downloaded at build time
    (see Dockerfile), so this is a no-op at runtime there; for local dev it
    downloads on first use, same convenience sentence-transformers used to
    provide."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODEL_DIR / "model.onnx"
    tokenizer_path = _MODEL_DIR / "tokenizer.json"
    if not model_path.exists():
        _download(_ONNX_MODEL_URL, model_path)
    if not tokenizer_path.exists():
        _download(_TOKENIZER_URL, tokenizer_path)
    return model_path, tokenizer_path


class _OnnxEmbedder:
    """all-MiniLM-L6-v2 via ONNX Runtime, replicating sentence-transformers'
    own mean-pooling + L2-normalization so outputs match the original
    PyTorch-based pipeline (see the module docstring note above)."""

    def __init__(self) -> None:
        model_path, tokenizer_path = _ensure_model_files()
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        self._tokenizer.enable_padding(
            pad_id=self._tokenizer.token_to_id("[PAD]"), pad_token="[PAD]"
        )
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def encode(self, texts: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(list(texts))
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)
        (last_hidden_state,) = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        mask = attention_mask[..., None].astype(np.float32)
        summed = (last_hidden_state * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = summed / counts
        if normalize_embeddings:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings


_embedder: _OnnxEmbedder | None = None


def _get_embedder() -> _OnnxEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = _OnnxEmbedder()
    return _embedder


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def chunk_evidence(evidence_bullets: List[str]) -> List[str]:
    """Each evidence bullet is already a self-contained chunk."""
    return [bullet.strip() for bullet in evidence_bullets if bullet.strip()]


class EvidenceIndex:
    """Hybrid BM25 + semantic index over a resume's evidence chunks."""

    def __init__(self, evidence_bullets: List[str]):
        self.chunks = chunk_evidence(evidence_bullets)
        self._bm25 = (
            BM25Okapi([_tokenize(chunk) for chunk in self.chunks])
            if self.chunks
            else None
        )
        self._embeddings = (
            _get_embedder().encode(self.chunks, normalize_embeddings=True)
            if self.chunks
            else None
        )

    def _search_scores(
        self, query: str, top_k: int = 5
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Runs BM25 + semantic scoring for a query once, returning both the
        RRF-fused (chunk_index, score) list and the raw semantic-only
        (chunk_index, cosine_score) list from the same embedding call.
        Internal building block for search() (uses only the fused list) and
        search_with_fallback() (uses both: the fused list to check the
        original query against MIN_RELEVANCE_SCORE, and the semantic-only
        list - not the fused one - for its reformulation-acceptance
        comparison, so the original query's semantic score doesn't need a
        second, redundant encode() call)."""
        if not self.chunks:
            return [], []

        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_ranked = np.argsort(bm25_scores)[::-1]

        query_embedding = _get_embedder().encode([query], normalize_embeddings=True)[0]
        semantic_scores = self._embeddings @ query_embedding
        semantic_ranked = np.argsort(semantic_scores)[::-1]

        rrf_scores: dict[int, float] = defaultdict(float)
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[int(idx)] += 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(semantic_ranked):
            rrf_scores[int(idx)] += 1.0 / (RRF_K + rank + 1)

        fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        semantic_only = [(int(idx), float(semantic_scores[idx])) for idx in semantic_ranked[:top_k]]
        return fused, semantic_only

    def _search_fused(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Hybrid BM25 + semantic RRF search, returning (chunk_index, score)
        pairs ungated by MIN_RELEVANCE_SCORE. Thin wrapper over
        _search_scores() for callers that only need the fused list."""
        fused, _ = self._search_scores(query, top_k)
        return fused

    def _search_semantic(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Semantic-only (cosine similarity) top-k, no BM25/RRF involved.
        Used by search_with_fallback() to score each reformulated query:
        RRF's rank-based fusion was found (see debugging-log.md) to produce
        a noise-dominated comparison on this project's small evidence
        corpus - the case that should have recovered had a *smaller*
        fused-score improvement than two cases that shouldn't have -
        so the reformulation-acceptance decision uses raw semantic
        similarity instead, which isn't subject to that rank-position
        compression effect."""
        if not self.chunks:
            return []
        query_embedding = _get_embedder().encode([query], normalize_embeddings=True)[0]
        semantic_scores = self._embeddings @ query_embedding
        ranked = np.argsort(semantic_scores)[::-1][:top_k]
        return [(int(idx), float(semantic_scores[idx])) for idx in ranked]

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Hybrid search combining BM25 keyword ranking and semantic similarity
        ranking via Reciprocal Rank Fusion (RRF).

        Returns an empty list if the top fused score falls below
        MIN_RELEVANCE_SCORE, treating the query as having no relevant evidence
        rather than returning weak, low-confidence chunks."""
        fused = self._search_fused(query, top_k)
        if not fused or fused[0][1] < MIN_RELEVANCE_SCORE:
            return []
        return [(self.chunks[idx], score) for idx, score in fused]


def search_with_fallback(
    skill: str,
    evidence_index: "EvidenceIndex",
    github_index: Optional["EvidenceIndex"] = None,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Self-healing hybrid search for a skill: if the skill's literal wording
    scores below MIN_RELEVANCE_SCORE against `evidence_index`, asks Gemini
    for alternate phrasings (reformulate_query) and retries with each one,
    taking whichever scores highest by semantic-only (cosine) similarity. If
    `github_index` is given and the literal skill still hasn't cleared the
    bar, also tries a normal, threshold-gated search there. Returns the
    winning attempt's chunks, or an empty list (identical to plain
    EvidenceIndex.search()) if nothing is available at all.

    Deliberately NOT gated by any numeric threshold on the reformulation
    tier - two were tried and both failed (see debugging-log.md for full
    data): a margin over the RRF-fused score had an inverted ordering (the
    case that should have recovered had the *smallest* score improvement,
    smaller than two cases that shouldn't recover at all), and a margin
    over semantic-only cosine similarity, while numerically ordered
    correctly, still frequently picked the wrong chunk - e.g. for
    "Observability", the genuinely relevant "error monitoring and
    root-cause analysis" bullet never ranked #1 under any of 4 phrasings
    tried, real evidence-adjacent chunks lost to unrelated ones. A top-1
    chunk-similarity score, on this project's small (~13-chunk) evidence
    corpus, isn't a reliable enough proxy for genuine topical relevance to
    gate on by itself, whichever metric it's built from. So instead: once a
    skill fails the literal search, its best-scoring reformulation is always
    passed through to the downstream LLM judge, unfiltered - the judge sees
    the actual skill name AND the actual chunk text together (not just a
    similarity number) and is far better positioned to reject a
    superficially-similar-but-wrong chunk, and its rubric already documents
    how to score exactly that case (RUBRIC in judge.py: "0 - No evidence:
    none of the provided chunks demonstrate use of the skill").
    """
    from app.extraction import reformulate_query

    def top_score(fused: List[Tuple[int, float]]) -> float:
        return fused[0][1] if fused else 0.0

    if not evidence_index.chunks:
        # No chunks to search regardless of query wording (e.g. github_index
        # when no github_username was given) - skip straight to the
        # github_index fallback (if any) rather than wasting a
        # reformulate_query() call retrying an empty corpus. Without this,
        # every skill on every request would burn a reformulation call on
        # main.py's routinely-empty github_index leg alone - at ~15-20
        # skills per request against a 20-request/day free-tier quota, one
        # /analyze call could exhaust the entire day's budget by itself.
        _logger.info(f"[search_with_fallback] skill={skill!r} evidence_index has no chunks - skipping reformulation")
        if github_index is not None:
            github_result = github_index.search(skill, top_k)
            if github_result:
                _logger.info(f"[search_with_fallback] skill={skill!r} ACCEPTED via github_index")
                return github_result
        return []

    fused_a, _ = evidence_index._search_scores(skill, top_k)
    score_a = top_score(fused_a)
    _logger.info(
        f"[search_with_fallback] skill={skill!r} original query top_score={score_a:.4f} "
        f"(MIN_RELEVANCE_SCORE={MIN_RELEVANCE_SCORE})"
    )
    if score_a >= MIN_RELEVANCE_SCORE:
        return [(evidence_index.chunks[idx], score) for idx, score in fused_a]

    best_score_b = -1.0
    best_chunks_b: List[Tuple[str, float]] = []
    best_query_b = None
    alt_queries = reformulate_query(skill)
    for alt in alt_queries:
        alt_semantic = evidence_index._search_semantic(alt, top_k)
        alt_score = top_score(alt_semantic)
        _logger.info(
            f"[search_with_fallback] skill={skill!r} reformulation {alt!r} "
            f"semantic_top_score={alt_score:.4f}"
        )
        if alt_score > best_score_b:
            best_score_b = alt_score
            best_query_b = alt
            best_chunks_b = [(evidence_index.chunks[idx], score) for idx, score in alt_semantic]

    if best_chunks_b:
        _logger.info(
            f"[search_with_fallback] skill={skill!r} passing best reformulation "
            f"{best_query_b!r} (semantic_score={best_score_b:.4f}) through to the judge "
            "unfiltered - no retrieval-side threshold on this tier, see docstring"
        )
        return best_chunks_b

    if github_index is not None:
        github_result = github_index.search(skill, top_k)
        if github_result:
            _logger.info(f"[search_with_fallback] skill={skill!r} ACCEPTED via github_index")
            return github_result

    _logger.info(
        f"[search_with_fallback] skill={skill!r} nothing available - returning empty, "
        "same as static search"
    )
    return []


def merge_ranked_chunks(
    *result_lists: List[Tuple[str, float]]
) -> List[Tuple[str, float]]:
    """Merge already-ranked (chunk, score) results from independent
    EvidenceIndex.search() calls into one list, sorted by score descending.

    Each input list's scores are RRF fusion scores computed entirely within
    that source's own BM25/embedding index, so combining lists here - after
    scoring is complete - never lets one source's corpus statistics (e.g.
    BM25 term-rarity weights) influence another source's ranking. See
    debugging-log.md for the corpus-pollution bug this guards against."""
    merged = [item for result_list in result_lists for item in result_list]
    merged.sort(key=lambda item: item[1], reverse=True)
    return merged


def skill_is_claimed(skill: str, claims: List[str]) -> bool:
    """True if any resume claim is semantically similar enough to the skill,
    via cosine similarity of all-MiniLM-L6-v2 embeddings (catches wording
    differences that exact string matching misses, e.g. JD "RAG" vs resume
    "Retrieval Augmented Generation (RAG)")."""
    if not claims:
        return False
    embedder = _get_embedder()
    skill_embedding = embedder.encode([skill], normalize_embeddings=True)[0]
    claim_embeddings = embedder.encode(claims, normalize_embeddings=True)
    best_similarity = float(np.max(claim_embeddings @ skill_embedding))
    return best_similarity >= CLAIM_MATCH_THRESHOLD
