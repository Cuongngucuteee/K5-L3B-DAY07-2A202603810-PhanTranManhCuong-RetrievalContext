"""Retrieval methods over the Day 6 corpus. Every retriever returns ranked chunk dicts:
{id, text, source_id, source, page, metadata, score, rank, strategy}. Filters are applied BEFORE
ranking (SQL WHERE for pgvector, candidate mask for BM25). Errors propagate (no silent fallback).

Methods
- dense  : query embedding (AITeamVN/Vietnamese_Embedding, 1024-d) -> exact cosine on pgvector (Day 6 index B).
- bm25   : Okapi BM25 over syllable tokens, optionally + syllable bigrams (Vietnamese words are mostly 2 syllables).
- hybrid : Reciprocal Rank Fusion of dense and BM25 rankings (ranks only, so no score scaling issue).
- rerank : hybrid candidates re-scored by a cross-encoder (BAAI/bge-reranker-v2-m3).
"""
from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections import Counter
from typing import Any

from .common import DATABASE_URL, DAY6_STRATEGY, EMBEDDING_MODEL, RERANK_MODEL, load_chunks

FILTERABLE = {"year": int}


def _check_filters(filters: dict | None) -> dict:
    filters = dict(filters or {})
    for k in filters:
        if k not in FILTERABLE:
            raise ValueError(f"filter '{k}' không được hỗ trợ")
        filters[k] = FILTERABLE[k](filters[k])
    return filters


def _passes(chunk: dict, filters: dict) -> bool:
    return all(chunk["metadata"].get(k) == v for k, v in filters.items())


def _ranked(rows: list[dict], strategy: str, top_k: int) -> list[dict]:
    return [{**r, "rank": i, "strategy": strategy} for i, r in enumerate(rows[:top_k], start=1)]


# ---------------------------------------------------------------- dense (pgvector)
_EMBEDDER = None


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        model.max_seq_length = 1024  # same as Day 6 ingest
        _EMBEDDER = model
    return _EMBEDDER


class DenseRetriever:
    name = "dense"

    def __init__(self) -> None:
        import psycopg
        self.model = get_embedder()
        self.dimensions = int(self.model.get_sentence_embedding_dimension())
        self.conn = psycopg.connect(DATABASE_URL, connect_timeout=5, autocommit=True)
        with self.conn.cursor() as cur:
            cur.execute("SELECT DISTINCT model, vector_dims(embedding) FROM chunks WHERE strategy = %s", (DAY6_STRATEGY,))
            rows = cur.fetchall()
        if rows != [(EMBEDDING_MODEL, self.dimensions)]:
            raise RuntimeError(f"Index {rows} không tương thích query embedder {EMBEDDING_MODEL}/{self.dimensions}")

    def embed(self, text: str) -> list[float]:
        # Day 6 encodes queries and documents the same way: no prefix, L2-normalized.
        return self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        filters = _check_filters(filters)
        vec = "[" + ",".join(f"{x:.7f}" for x in self.embed(query)) + "]"
        where, params = ["strategy = %s"], [DAY6_STRATEGY]
        for k, v in filters.items():
            where.append(f"{k} = %s")
            params.append(v)
        with self.conn.cursor() as cur:
            cur.execute(f"""SELECT chunk_id, document_id, source, page, articles, year, text, metadata,
                                   1 - (embedding <=> %s::vector) AS score
                            FROM chunks WHERE {' AND '.join(where)}
                            ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s""", [vec, *params, vec, top_k])
            rows = cur.fetchall()
        out = []
        for cid, doc, source, page, arts, year, text, meta, score in rows:
            meta = meta if isinstance(meta, dict) else json.loads(meta)
            out.append({"id": cid, "text": text, "source_id": doc, "source": source, "page": page,
                        "metadata": {**meta, "articles": list(arts), "year": year}, "score": round(float(score), 6)})
        return _ranked(out, self.name, top_k)


# ---------------------------------------------------------------- BM25
def syllables(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def tokenize(text: str, bigrams: bool) -> list[str]:
    toks = syllables(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])] if bigrams else toks


class BM25Retriever:
    def __init__(self, chunks: list[dict] | None = None, *, bigrams: bool = True, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks if chunks is not None else load_chunks()
        self.bigrams, self.k1, self.b = bigrams, k1, b
        self.name = "bm25_bigram" if bigrams else "bm25_syllable"
        self.tf = [Counter(tokenize(c["text"], bigrams)) for c in self.chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(term for t in self.tf for term in t)
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        filters = _check_filters(filters)
        q = [t for t in set(tokenize(query, self.bigrams)) if t in self.idf]
        rows = []
        for c, tf, ln in zip(self.chunks, self.tf, self.len):
            if not _passes(c, filters):
                continue
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            if s > 0:
                rows.append({**c, "score": round(s, 6)})
        rows.sort(key=lambda r: (-r["score"], r["id"]))
        return _ranked(rows, self.name, top_k)


# ---------------------------------------------------------------- hybrid RRF
class HybridRetriever:
    """RRF: score(d) = sum_i w_i / (k + rank_i(d)). Ranks from `depth` candidates of each retriever."""

    def __init__(self, dense: DenseRetriever, lexical: BM25Retriever, *, k: int = 60, depth: int = 30,
                 weights: tuple[float, float] = (1.0, 1.0)) -> None:
        self.dense, self.lexical, self.k, self.depth, self.weights = dense, lexical, k, depth, weights
        self.name = "hybrid_rrf"

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        fused: dict[str, dict] = {}
        for w, retr in zip(self.weights, (self.dense, self.lexical)):
            for r in retr.search(query, filters=filters, top_k=self.depth):
                e = fused.setdefault(r["id"], {**{k: v for k, v in r.items() if k not in ("score", "rank", "strategy")},
                                               "score": 0.0, "components": {}})
                e["score"] += w / (self.k + r["rank"])
                e["components"][retr.name] = {"rank": r["rank"], "score": r["score"]}
        rows = sorted(fused.values(), key=lambda r: (-r["score"], r["id"]))
        for r in rows:
            r["score"] = round(r["score"], 6)
        return _ranked(rows, self.name, top_k)


# ---------------------------------------------------------------- cross-encoder rerank
_RERANKER = None


def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        import torch
        from sentence_transformers import CrossEncoder
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _RERANKER = CrossEncoder(RERANK_MODEL, device=device, max_length=512)
    return _RERANKER


class RerankRetriever:
    def __init__(self, base, *, candidates: int = 20) -> None:
        self.base, self.candidates = base, candidates
        self.model = get_reranker()
        self.name = f"{base.name}+rerank"

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        cands = self.base.search(query, filters=filters, top_k=self.candidates)
        if not cands:
            return []
        scores = self.model.predict([(query, c["text"]) for c in cands], batch_size=8, show_progress_bar=False)
        rows = [{**c, "first_stage": {"rank": c["rank"], "score": c["score"]}, "score": round(float(s), 6)}
                for c, s in zip(cands, scores)]
        rows.sort(key=lambda r: (-r["score"], r["id"]))
        return _ranked(rows, self.name, top_k)


def build_retriever(method: str, *, cache: dict | None = None) -> Any:
    """method in {dense, bm25_syllable, bm25_bigram, hybrid, hybrid_rerank, dense_rerank}."""
    cache = cache if cache is not None else {}

    def get(key, factory):
        if key not in cache:
            cache[key] = factory()
        return cache[key]

    dense = lambda: get("dense", DenseRetriever)
    bm25 = lambda: get("bm25_bigram", lambda: BM25Retriever(bigrams=True))
    return {
        "dense": dense,
        "bm25_syllable": lambda: get("bm25_syllable", lambda: BM25Retriever(bigrams=False)),
        "bm25_bigram": bm25,
        "hybrid": lambda: get("hybrid", lambda: HybridRetriever(dense(), bm25())),
        "hybrid_rerank": lambda: get("hybrid_rerank", lambda: RerankRetriever(get("hybrid", lambda: HybridRetriever(dense(), bm25())))),
        "dense_rerank": lambda: get("dense_rerank", lambda: RerankRetriever(dense())),
    }[method]()


def timed_search(retriever, query: str, **kw) -> tuple[list[dict], float]:
    t = time.perf_counter()
    rows = retriever.search(query, **kw)
    return rows, (time.perf_counter() - t) * 1000
