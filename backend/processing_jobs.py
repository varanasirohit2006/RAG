"""Bounded background work and content-addressed extraction caching."""

import asyncio
import gzip
import hashlib
import json
import os
import tempfile
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from backend.document_extraction import extract_document
from backend.files import PROJECT_ROOT

EXTRACTION_CACHE_VERSION = 2
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-job")


class JobCancelled(RuntimeError):
    pass


class BackgroundJob:
    """Cancellable background work with thread-safe progress snapshots."""

    def __init__(self, target, *, key=None, on_change=None):
        self.key = key
        self.on_change = on_change
        self.cancellation_event = threading.Event()
        self._lock = threading.Lock()
        self.status = "queued"
        self.result = self.error = None
        self.done, self.total, self.message = 0, 1, "Queued"
        self.partial_text = ""
        self.events = deque(maxlen=32)
        self.created = time.perf_counter()
        self.finished = None
        self.future = _executor.submit(self._run, target)

    def _run(self, target):
        with self._lock:
            self.status = "running"
        try:
            self._publish()
            self.check_cancelled()
            result = target(self)
            self.check_cancelled()
            with self._lock:
                self.result = result
                self.status, self.message = "succeeded", "Complete"
                self.done = self.total
        except (Exception, asyncio.CancelledError) as exc:
            with self._lock:
                self.error = str(exc) or type(exc).__name__
                self.status = "cancelled" if (
                    self.cancellation_event.is_set() or isinstance(exc, asyncio.CancelledError)
                ) else "failed"
                self.message = "Cancelled" if self.status == "cancelled" else "Failed"
        finally:
            with self._lock:
                self.finished = time.perf_counter()
            self._publish()

    def _publish(self):
        if self.on_change:
            self.on_change(self.snapshot())

    def check_cancelled(self):
        if self.cancellation_event.is_set():
            raise JobCancelled("Work cancelled.")

    def progress(self, done, total, message):
        self.check_cancelled()
        with self._lock:
            self.done, self.total = max(0, int(done)), max(1, int(total))
            self.message = str(message)
            if not self.events or self.events[-1][1] != self.message:
                self.events.append((round(time.perf_counter() - self.created, 2), self.message))
        self._publish()

    def token(self, text):
        self.check_cancelled()
        with self._lock:
            self.partial_text += str(text)
        self._publish()

    def cancel(self):
        self.cancellation_event.set()
        if self.future.cancel():
            with self._lock:
                self.status, self.message = "cancelled", "Cancelled"
                self.finished = time.perf_counter()
            self._publish()

    def snapshot(self):
        with self._lock:
            return {
                "status": self.status, "done": self.done, "total": self.total,
                "message": self.message, "partial_text": self.partial_text,
                "elapsed_seconds": round((self.finished or time.perf_counter()) - self.created, 2),
                "events": list(self.events), "result": self.result, "error": self.error,
            }


def extraction_cache_key(document_hash, settings):
    value = {"version": EXTRACTION_CACHE_VERSION, "document_hash": document_hash,
             "settings": asdict(settings)}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def extract_with_cache(file_path, document_hash, settings, *, progress_callback=None,
                       cancellation_event=None, cache_dir=PROJECT_ROOT / ".cache" / "extractions"):
    from unstructured.staging.base import elements_from_dicts, elements_to_dicts

    key = extraction_cache_key(document_hash, settings)
    directory = Path(cache_dir)
    path = directory / f"{key}.json.gz"
    if cancellation_event and cancellation_event.is_set():
        raise JobCancelled("Work cancelled.")
    if path.is_file():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                record = json.load(stream)
            if isinstance(record, dict) and record.get("cache_key") == key and record.get("version") == EXTRACTION_CACHE_VERSION:
                elements = elements_from_dicts(record["elements"])
                if elements:
                    if progress_callback:
                        progress_callback(1, 1, "Reused completed PDF extraction")
                    return elements, True
        except (ValueError, KeyError, OSError, TypeError, AttributeError):
            if progress_callback:
                progress_callback(0, 1, "Incomplete extraction cache; processing the PDF")
    elements = extract_document(file_path, settings=settings, progress_callback=progress_callback,
                                cancellation_event=cancellation_event)
    if cancellation_event and cancellation_event.is_set():
        raise JobCancelled("Work cancelled.")
    record = {"version": EXTRACTION_CACHE_VERSION, "cache_key": key,
              "document_hash": document_hash, "elements": elements_to_dicts(elements)}
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    return elements, False


def prepare_document(file_path, document_hash, index_key, settings, job, *, source_documents=None, storage_root=None):
    from backend.rag_pipeline import get_or_create_vector_store

    root = Path(storage_root) if storage_root else PROJECT_ROOT
    started = time.perf_counter()
    elements, hit = extract_with_cache(file_path, document_hash, settings,
        progress_callback=job.progress, cancellation_event=job.cancellation_event,
        cache_dir=root / ".cache" / "extractions")
    extraction_seconds = time.perf_counter() - started
    job.check_cancelled()
    job.progress(0, 1, "Preparing document search index")
    started = time.perf_counter()
    directory = f"db/chroma_{index_key}"
    target = str(root / directory)
    database = get_or_create_vector_store(elements, target, document_hash=document_hash,
                                          progress_callback=job.progress, source_documents=source_documents)
    return {"elements": elements, "stats": dict(Counter(element.category for element in elements)),
            "database": database, "index_path": directory, "extraction_cache_hit": hit,
            "timings": {"extraction_seconds": round(extraction_seconds, 3),
                        "index_seconds": round(time.perf_counter() - started, 3)}}


def answer_question(database, query, job, *, use_reranker=True):
    from backend.generation_runtime import generate_final_answer
    from backend.retrieval_runtime import retrieve_documents

    job.progress(0, 3, "Finding relevant document passages")
    started = time.perf_counter()
    documents, timings = retrieve_documents(database, query,
        k=int(os.getenv("RETRIEVAL_K", "10")), top_n=int(os.getenv("RETRIEVAL_TOP_N", "3")),
                                            use_reranker=use_reranker)
    job.check_cancelled()
    job.progress(1, 3, "Generating a grounded answer")
    generation_started = time.perf_counter()
    answer = generate_final_answer(documents, query, token_callback=job.token,
                                   progress_callback=job.progress)
    timings = dict(timings)
    timings.update(generation_seconds=round(time.perf_counter() - generation_started, 3),
                   total_seconds=round(time.perf_counter() - started, 3))
    return {"query": query, "answer": answer, "documents": documents, "timings": timings}


def evaluate_document(database_path, dataset, document_hash, profile, job, *, force=False, output_dir=PROJECT_ROOT / 'evaluation_runs'):
    from backend.advanced_evaluation import run_evaluation

    return asyncio.run(run_evaluation(db_path=database_path, dataset=dataset,
        expected_document_hash=document_hash, profile=profile, progress_callback=job.progress,
        cancellation_event=job.cancellation_event, force=force, output_dir=output_dir))
