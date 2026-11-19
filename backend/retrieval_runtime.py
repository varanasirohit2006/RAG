"""Reusable local lexical/semantic retrieval and bounded optional reranking."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import threading
import time
from collections import OrderedDict
from functools import lru_cache

from langchain_core.documents import Document

from backend.generation_runtime import document_key, document_payload
from backend.index_runtime import read_index_manifest

_cache_lock = threading.Lock()
_retrievers = OrderedDict()


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def _evidence_key(document):
    payload = document_payload(document)
    images = [str(path).rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for path in payload["image_paths"]]
    images.extend(hashlib.sha256(image.encode()).hexdigest() for image in payload["images_base64"])
    return hashlib.sha256(json.dumps({
        "text": document.page_content, "raw_text": payload["raw_text"],
        "tables": payload["tables_html"], "images": images,
    }, sort_keys=True).encode()).hexdigest()


def _page_numbers(metadata):
    value = metadata.get("pages", metadata.get("page", []))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = []
    if isinstance(value, int):
        value = [value]
    return value if isinstance(value, list) else []


def _canonical_evidence(documents):
    groups = {}
    for document in documents:
        groups.setdefault(_evidence_key(document), []).append(document)
    canonical = {}
    for key, group in groups.items():
        primary = min(group, key=document_key)
        ids = sorted({document_key(document) for document in group})
        metadata = dict(primary.metadata)
        metadata["duplicate_chunk_ids"] = json.dumps(ids)
        pages = sorted({page for document in group for page in _page_numbers(document.metadata)})
        if pages:
            metadata["pages"] = json.dumps(pages)
        canonical[key] = Document(page_content=primary.page_content, metadata=metadata)
    return canonical


class DeduplicatingRetriever:
    def __init__(self, base, canonical, k):
        self.base, self.canonical, self.k = base, canonical, k

    def invoke(self, query):
        result, seen = [], set()
        for document in self.base.invoke(query):
            key = _evidence_key(document)
            if key in seen:
                continue
            seen.add(key)
            result.append(self.canonical.get(key, document))
            if len(result) >= self.k:
                break
        return result

    async def ainvoke(self, query):
        return await asyncio.to_thread(self.invoke, query)


class HybridRetriever:
    def __init__(self, vector, keyword, k):
        self.vector, self.keyword, self.k = vector, keyword, k

    def invoke(self, query):
        scores, candidates = {}, {}
        for result in (self.vector.invoke(query), self.keyword.invoke(query)):
            for rank, document in enumerate(result, 1):
                key = document_key(document)
                candidates.setdefault(key, document)
                scores[key] = scores.get(key, 0) + 0.5 / (60 + rank)
        keys = sorted(scores, key=lambda key: (-scores[key], key))
        return [candidates[key] for key in keys[:self.k * 2]]

    async def ainvoke(self, query):
        return await asyncio.to_thread(self.invoke, query)


def get_retrievers(db, *, k=10):
    if not 1 <= k <= 100:
        raise ValueError("Retrieval k must be between 1 and 100.")
    directory = getattr(db, "_persist_directory", None)
    manifest = read_index_manifest(directory) if directory else None
    revision = manifest.get("corpus_version") if manifest else None
    embedding_signature = tuple((manifest or {}).get(field) for field in (
        "schema_version", "pipeline_version", "embedding_model", "embedding_revision",
    ))
    count = db._collection.count()
    key = (str(directory) if directory else id(db), revision, embedding_signature, count, k)
    with _cache_lock:
        cached = _retrievers.get(key)
        if cached:
            _retrievers.move_to_end(key)
            return cached
        from langchain_community.retrievers import BM25Retriever

        records = db.get()
        metadatas = records.get("metadatas") or [{} for _ in records["ids"]]
        documents = []
        for chunk_id, text, metadata in zip(records["ids"], records["documents"], metadatas):
            metadata = dict(metadata or {})
            metadata.setdefault("chunk_id", chunk_id)
            documents.append(Document(page_content=text, metadata=metadata))
        if not documents:
            raise ValueError("The selected vector store is empty.")
        canonical = _canonical_evidence(documents)
        sample_k = min(k * 4, count)
        keyword_base = BM25Retriever.from_documents(documents, preprocess_func=tokenize, k=sample_k)
        vector_base = db.as_retriever(search_kwargs={"k": sample_k})
        keyword = DeduplicatingRetriever(keyword_base, canonical, k)
        vector = DeduplicatingRetriever(vector_base, canonical, k)
        hybrid = HybridRetriever(vector, keyword, k)
        result = (vector, keyword, hybrid, list(canonical.values()))
        _retrievers[key] = result
        while len(_retrievers) > 8:
            _retrievers.popitem(last=False)
        return result


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
            documents=[document.page_content for document in documents],
            top_n=self.top_n,
            request_options={"max_retries": 0, "timeout_in_seconds": self.timeout},
        )
        results = []
        for item in response.results:
            score = float(item.relevance_score)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Cohere returned an invalid relevance score.")
            original = documents[item.index]
            results.append(Document(
                page_content=original.page_content,
                metadata={**original.metadata, "relevance_score": score},
            ))
        return results


@lru_cache(maxsize=8)
def _cohere_reranker(model, top_n, timeout):
    return CohereReranker(model, top_n, timeout)


def get_reranker(*, provider=None, top_n=3):
    provider = provider or os.getenv(
        "RERANKER_PROVIDER", "cohere" if os.getenv("COHERE_API_KEY") else "none"
    )
    provider = provider.strip().lower()
    if provider == "none":
        return None
    if provider != "cohere":
        raise ValueError("RERANKER_PROVIDER must be cohere or none.")
    timeout = int(os.getenv("RERANK_TIMEOUT_SECONDS", "15"))
    if not 1 <= timeout <= 60:
        raise ValueError("RERANK_TIMEOUT_SECONDS must be between 1 and 60.")
    return _cohere_reranker(os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"), top_n, timeout)


def retrieve_documents(db, query, *, k=10, top_n=3, reranker_provider=None, use_reranker=True,
                       progress_callback=None):
    if not 1 <= top_n <= k:
        raise ValueError("Final context size top_n must be between 1 and retrieval k.")
    started = time.perf_counter()
    if progress_callback:
        progress_callback(0, 2, "Searching semantic and keyword indexes...")
    _, _, hybrid, _ = get_retrievers(db, k=k)
    candidates = hybrid.invoke(query)
    retrieval_seconds = time.perf_counter() - started
    rerank_started = time.perf_counter()
    timings = {
        "retrieval_seconds": retrieval_seconds, "candidate_count": len(candidates),
        "reranker": "none",
    }
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
        # Degradation remains explicit for both application and evaluation reports.
        # Only index/client objects are cached; failed rerank responses are not.
        selected = candidates[:top_n]
        timings["reranking_error"] = type(exc).__name__
    timings["reranking_seconds"] = time.perf_counter() - rerank_started
    if progress_callback:
        progress_callback(2, 2, "Evidence retrieved.")
    return selected, timings
