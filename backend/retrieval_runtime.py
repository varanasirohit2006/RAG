"""Retrieval pipeline: Vector, BM25, Hybrid RRF, Multi-Query, and Cohere Reranking (rag-for-beginners style)."""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from collections import defaultdict
from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel

from backend.generation_runtime import content_text, document_key, get_generation_model


def tokenize(text):
    return re.findall(r"\w+", text.lower())


# ── Multi-Query Retrieval (from 10_multi_query_retrieval.py) ─────────────────

class QueryVariations(BaseModel):
    queries: List[str]


def generate_query_variations(query, *, model=None):
    llm = model or get_generation_model()
    prompt = (
        "Generate 3 different variations of this query that would help retrieve relevant documents:\n\n"
        f"Original query: {query}\n\n"
        "Return 3 alternative queries that rephrase or approach the same question from different angles."
    )
    try:
        response = llm.with_structured_output(QueryVariations).invoke(prompt)
        variations = [q.strip()[:4000] for q in (response.queries or []) if str(q).strip()]
        if variations:
            return variations[:3]
    except Exception:
        pass
    response = llm.invoke(f"Give exactly three search-query paraphrases, one per line, without explanations, for: {query}")
    return [line.strip(" -0123456789.")[:4000] for line in content_text(response.content).splitlines() if line.strip()][:3]


# ── Reciprocal Rank Fusion (from 11_reciprocal_rank_fusion.py) ──────────────

def reciprocal_rank_fusion(chunk_lists, k=60, weights=None):
    rrf_scores = defaultdict(float)
    unique_chunks = {}
    weights = weights or [1.0] * len(chunk_lists)
    for chunks, weight in zip(chunk_lists, weights):
        for position, chunk in enumerate(chunks, 1):
            key = document_key(chunk)
            unique_chunks.setdefault(key, chunk)
            rrf_scores[key] += weight / (k + position)
    ranked_keys = sorted(rrf_scores, key=lambda key: (-rrf_scores[key], key))
    return [(unique_chunks[key], rrf_scores[key]) for key in ranked_keys]


# ── Hybrid Retriever (from 12_hybrid_search.ipynb) ───────────────────────────

class HybridRetriever:
    def __init__(self, vector, keyword, k):
        self.vector, self.keyword, self.k = vector, keyword, k

    def invoke(self, query):
        fused = reciprocal_rank_fusion([self.vector.invoke(query), self.keyword.invoke(query)], k=60, weights=[0.5, 0.5])
        return [doc for doc, _ in fused[:self.k * 2]]

    async def ainvoke(self, query):
        return await asyncio.to_thread(self.invoke, query)


def get_retrievers(db, *, k=10):
    from langchain_community.retrievers import BM25Retriever

    if not 1 <= k <= 100:
        raise ValueError("Retrieval k must be between 1 and 100.")
    records = db.get()
    metadatas = records.get("metadatas") or [{} for _ in records["ids"]]
    documents = []
    for chunk_id, text, meta in zip(records["ids"], records["documents"], metadatas):
        meta = dict(meta or {})
        meta.setdefault("chunk_id", chunk_id)
        documents.append(Document(page_content=text, metadata=meta))
    if not documents:
        raise ValueError("The selected vector store is empty.")

    sample_k = min(k, len(documents))
    vector = db.as_retriever(search_kwargs={"k": sample_k})
    keyword = BM25Retriever.from_documents(documents, preprocess_func=tokenize, k=sample_k)
    hybrid = HybridRetriever(vector, keyword, k)
    return vector, keyword, hybrid, documents


# ── Cohere Reranking (from 13_reranker.ipynb) ────────────────────────────────

class CohereReranker:
    def __init__(self, model, top_n, timeout):
        import cohere

        key = os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError("Cohere reranking needs COHERE_API_KEY.")
        self.client = cohere.ClientV2(api_key=key, timeout=timeout)
        self.model, self.top_n, self.timeout = model, top_n, timeout

    def compress_documents(self, documents, query):
        if not documents:
            return []
        response = self.client.rerank(
            model=self.model, query=query,
            documents=[doc.page_content for doc in documents],
            top_n=self.top_n,
            request_options={"max_retries": 0, "timeout_in_seconds": self.timeout},
        )
        results = []
        for item in response.results:
            score = float(item.relevance_score)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Cohere returned an invalid relevance score.")
            orig = documents[item.index]
            results.append(Document(page_content=orig.page_content, metadata={**orig.metadata, "relevance_score": score}))
        return results


@lru_cache(maxsize=8)
def _cohere_reranker(model, top_n, timeout):
    return CohereReranker(model, top_n, timeout)


def get_reranker(*, provider=None, top_n=3):
    provider = (provider or os.getenv("RERANKER_PROVIDER", "cohere" if os.getenv("COHERE_API_KEY") else "none")).strip().lower()
    if provider == "none":
        return None
    if provider != "cohere":
        raise ValueError("RERANKER_PROVIDER must be cohere or none.")
    timeout = int(os.getenv("RERANK_TIMEOUT_SECONDS", "15"))
    return _cohere_reranker(os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"), top_n, timeout)


def retrieve_documents(db, query, *, k=10, top_n=3, reranker_provider=None, use_reranker=True, progress_callback=None):
    if not 1 <= top_n <= k:
        raise ValueError("Final context size top_n must be between 1 and retrieval k.")
    started = time.perf_counter()
    if progress_callback:
        progress_callback(0, 2, "Searching semantic and keyword indexes...")
    _, _, hybrid, _ = get_retrievers(db, k=k)
    candidates = hybrid.invoke(query)
    timings = {"retrieval_seconds": time.perf_counter() - started, "candidate_count": len(candidates), "reranker": "none"}

    rerank_started = time.perf_counter()
    if progress_callback:
        progress_callback(1, 2, "Selecting the strongest document evidence...")
    try:
        reranker = get_reranker(provider=reranker_provider, top_n=top_n) if use_reranker else None
        if reranker:
            selected = list(reranker.compress_documents(candidates, query))
            timings["reranker"] = "cohere"
        else:
            selected = candidates[:top_n]
    except Exception as exc:
        selected = candidates[:top_n]
        timings["reranking_error"] = type(exc).__name__
    timings["reranking_seconds"] = time.perf_counter() - rerank_started
    if progress_callback:
        progress_callback(2, 2, "Evidence retrieved.")
    return selected, timings
