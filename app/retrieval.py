"""Chunking, embedding, and hybrid (BM25 + semantic) retrieval over resume evidence."""
from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

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
