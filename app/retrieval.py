"""Chunking, embedding, and hybrid (BM25 + semantic) retrieval over resume evidence."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60
MIN_RELEVANCE_SCORE = 0.0320
CLAIM_MATCH_THRESHOLD = 0.50

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
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

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Hybrid search combining BM25 keyword ranking and semantic similarity
        ranking via Reciprocal Rank Fusion (RRF).

        Returns an empty list if the top fused score falls below
        MIN_RELEVANCE_SCORE, treating the query as having no relevant evidence
        rather than returning weak, low-confidence chunks."""
        if not self.chunks:
            return []

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
        if not fused or fused[0][1] < MIN_RELEVANCE_SCORE:
            return []
        return [(self.chunks[idx], score) for idx, score in fused]


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
