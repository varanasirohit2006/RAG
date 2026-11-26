"""Automatically draft and source-verify RAG evaluation test cases."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.evaluation_data import (
    EvaluationDataError, document_id, document_pages,
    draft_source_texts, evidence_labels, load_evaluation_dataset,
)
from backend.evaluation_judge import JUDGE_API_KEYS, JudgeSettings, RagasJudge, _parse_structured_text
from backend.files import PROJECT_ROOT, write_json
from backend.provider_limits import provider_status_code, provider_timeout

SCHEMA_VERSION = 1
CATEGORIES = ("factual", "paraphrase", "comparison", "multi_chunk", "numeric")
_generation_guard = threading.Lock()
_active_generation_keys = set()


@contextmanager
def _generation_lock(key):
    with _generation_guard:
        if key in _active_generation_keys:
            raise EvaluationDataError("Automatic question generation for this source is already running.")
        _active_generation_keys.add(key)
    try:
        yield
    finally:
        with _generation_guard:
            _active_generation_keys.remove(key)


class SupportingQuote(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    chunk_id: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=12, max_length=1500)


class DraftQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: str = Field(min_length=10, max_length=4000)
    reference_answer: str = Field(min_length=1, max_length=6000)
    category: Literal["factual", "paraphrase", "comparison", "multi_chunk", "numeric"]
    supporting_quotes: list[SupportingQuote] = Field(min_length=1, max_length=4)


class SourceVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    supported: bool
    unambiguous: bool
    all_evidence_relevant: bool
    category_valid: bool
    reason: str = Field(min_length=1, max_length=1200)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _normalized(text):
    return " ".join(text.split())


def _question_key(text):
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def _model_identity(name):
    return name.strip().lower().removesuffix(":free").rsplit("/", 1)[-1]


def generation_plan(question_count=10):
    if isinstance(question_count, bool) or not isinstance(question_count, int) or not 1 <= question_count <= 50:
        raise EvaluationDataError("Automatic evaluation needs 1-50 questions.")
    return {
        "question_count": question_count,
        "maximum_candidates": question_count * 2,
        "maximum_drafting_operations": question_count * 2,
        "maximum_verification_operations": question_count * 2,
        "reference_origin": "synthetic_source_verified",
    }


async def _load_sources(db_path, expected_hash):
    from backend.rag_pipeline import load_vector_store
    from backend.retrieval_runtime import get_retrievers

    target = Path(db_path).resolve()
    try:
        manifest = json.loads((target / "index_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDataError("Automatic question generation needs a prepared index manifest.") from exc
    if manifest.get("document_hash") != expected_hash:
        raise EvaluationDataError("Selected index belongs to another PDF.")
    db = await asyncio.to_thread(load_vector_store, str(target))
    if db is None:
        raise EvaluationDataError("Prepare this document before automatic evaluation.")
    *_, documents = await asyncio.to_thread(get_retrievers, db, k=5)
    return documents, manifest


def _validate_quotes(draft, sources):
    if re.search(r"\b(?:title|name)\s+of\s+(?:the|this)\s+(?:handbook|document|pdf|file)\b", draft.question, re.IGNORECASE):
        raise EvaluationDataError("Generic document-title questions do not test substantive source content.")
    by_id = {source["chunk_id"]: source for source in sources}
    for evidence in draft.supporting_quotes:
        source = by_id.get(evidence.chunk_id)
        if source is None:
            raise EvaluationDataError("Generated evidence names a chunk outside the supplied source window.")
        quote = _normalized(evidence.quote)
        if len(quote) < 12:
            raise EvaluationDataError("Supporting quotes need at least 12 characters.")
        if quote not in _normalized(source["text"]):
            raise EvaluationDataError("A generated supporting quote does not occur in its source chunk.")
    if draft.category == "multi_chunk" and len({q.chunk_id for q in draft.supporting_quotes}) < 2:
        raise EvaluationDataError("A multi-chunk question requires evidence from at least two chunks.")


def _cached_dataset(path, configuration, sources):
    try:
        dataset = json.loads(path.read_text(encoding="utf-8"))
        if dataset.get("provenance", {}).get("configuration") != configuration:
            return None
        if len(dataset.get("questions", [])) != configuration["question_count"]:
            return None
        load_evaluation_dataset(dataset, expected_document_hash=configuration["document_hash"])
        return dataset
    except (OSError, KeyError, TypeError, ValueError):
        return None


async def _run_with_timeout(awaitable, timeout, cancellation_event):
    task = asyncio.create_task(awaitable)
    started = time.monotonic()
    try:
        while not task.done():
            if cancellation_event and cancellation_event.is_set():
                raise asyncio.CancelledError("Automatic question generation cancelled.")
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError(f"Model request exceeded {timeout:.0f}s.")
            await asyncio.wait({task}, timeout=min(0.25, remaining))
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def generate_evaluation_dataset(
    db_path, *, expected_document_hash, question_count=10, progress_callback=None,
    cancellation_event=None, output_dir=PROJECT_ROOT / "evaluation_datasets", force=False,
):
    """Draft and source-verify RAG test cases directly from indexed document passages."""
    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage
    from backend.generation_runtime import _build_model, close_ollama_model, model_name_for, ollama_json_schema

    load_dotenv()
    plan = generation_plan(question_count)
    expected_hash = str(expected_document_hash).lower()
    if not re.fullmatch(r"[a-f0-9]{64}", expected_hash):
        raise EvaluationDataError("Automatic evaluation requires the source PDF SHA-256.")

    done = 0
    def check_cancelled():
        if cancellation_event and cancellation_event.is_set():
            raise asyncio.CancelledError("Automatic question generation cancelled.")

    def emit(message):
        if progress_callback:
            progress_callback(done, question_count, message)

    check_cancelled()
    emit("Selecting source passages across the prepared document")
    documents, index_manifest = await _load_sources(db_path, expected_hash)
    sources, seen = [], set()
    source_texts, footer_labels = draft_source_texts(documents)
    for doc, text in zip(documents, source_texts):
        chunk_id = document_id(doc)
        if not chunk_id or chunk_id in seen or len(text) < 40:
            continue
        seen.add(chunk_id)
        item = {"chunk_id": chunk_id, "text": text[:12000], "pages": sorted(document_pages(doc))}
        if doc.metadata.get("source_pages"):
            item["original_sources"] = json.loads(doc.metadata["source_pages"])
        sources.append(item)
    sources.sort(key=lambda s: (s["pages"] or [10**9], s["chunk_id"]))
    if not sources:
        raise EvaluationDataError("No readable source passages available for automatic questions.")

    provider = os.getenv("SYNTHETIC_GENERATION_PROVIDER", os.getenv("GENERATION_PROVIDER", "google")).strip().lower()
    model = os.getenv("SYNTHETIC_GENERATION_MODEL", "").strip() or model_name_for(provider)
    verifier_settings = JudgeSettings.from_env()
    same_model = _model_identity(model) == _model_identity(verifier_settings.model)
    allow_same = os.getenv("SYNTHETIC_ALLOW_SAME_MODEL_VERIFICATION", "false").lower() == "true"
    if same_model and not allow_same:
        raise EvaluationDataError("Automatic source verification requires a different model unless SYNTHETIC_ALLOW_SAME_MODEL_VERIFICATION=true.")

    timeout = int(os.getenv("SYNTHETIC_GENERATION_TIMEOUT", "60"))
    request_timeout = int(os.getenv("SYNTHETIC_REQUEST_TIMEOUT_SECONDS", str(timeout)))
    tokens = int(os.getenv("SYNTHETIC_GENERATION_MAX_TOKENS", "4096"))
    concurrency = int(os.getenv("SYNTHETIC_GENERATION_CONCURRENCY", "2"))

    configuration = {
        "schema_version": SCHEMA_VERSION, "document_hash": expected_hash,
        "corpus_version": index_manifest.get("corpus_version"), "source_fingerprint": _hash(sources),
        "question_count": question_count, "draft_provider": provider, "draft_model": model,
        "draft_timeout": timeout, "draft_max_tokens": tokens, "verifier": asdict(verifier_settings),
        "independent_reference_verification": not same_model,
        "source_documents": index_manifest.get("source_documents", []),
        "removed_footer_labels": footer_labels,
    }
    run_key = _hash(configuration)
    path = Path(output_dir).resolve() / f"automatic-{run_key}.json"

    with _generation_lock(run_key):
        cached = _cached_dataset(path, configuration, sources) if not force else None
        if cached:
            done = question_count
            emit("Reusing cached verified question set")
            return {**cached, "cache_reused": True, "dataset_path": str(path)}

        emit("Preparing question drafting and source verification")
        draft_model = _build_model(provider, model, request_timeout, tokens)
        verifier = RagasJudge(verifier_settings)
        verifier.prepare()

        accepted, rejected, question_keys = [], [], set()
        work_counts = {"drafting_operations": 0, "verification_operations": 0}
        semaphore = asyncio.Semaphore(concurrency)
        started = time.perf_counter()
        schema = json.dumps(DraftQuestion.model_json_schema(), ensure_ascii=False)
        provider_limit = None

        async def process_candidate(index):
            nonlocal provider_limit
            async with semaphore:
                check_cancelled()
                if provider_limit:
                    return None, {"candidate": index + 1, "stage": "budget", "reason": "Stopped after rate limit."}
                sweep, pos = divmod(index, question_count)
                src_idx = (min(len(sources) - 1, pos * len(sources) // question_count) + sweep) % len(sources)
                category = CATEGORIES[index % len(CATEGORIES)]
                window = [sources[src_idx]]
                if category in {"comparison", "multi_chunk"} and len(sources) > 1:
                    window.append(sources[(src_idx + 1) % len(sources)])
                if category == "numeric" and not any(re.search(r"\d", item["text"]) for item in window):
                    category = "factual"

                prompt = (
                    "Create one standalone RAG evaluation question and reference answer using ONLY the supplied passages. "
                    f"Category: {category}. Copy exact supporting quotes with chunk_id. "
                    "Return one JSON object matching this schema:\n" + schema
                    + "\nSOURCE PASSAGES:\n" + json.dumps(window, ensure_ascii=False)
                )
                try:
                    emit(f"Drafting candidate {index + 1}; {done}/{question_count} verified")
                    work_counts["drafting_operations"] += 1
                    options = {"format": ollama_json_schema(DraftQuestion)} if provider == "ollama" else {}
                    resp = await _run_with_timeout(draft_model.ainvoke([HumanMessage(content=prompt)], **options), request_timeout, cancellation_event)
                    draft = _parse_structured_text(resp.content, DraftQuestion)
                    _validate_quotes(draft, window)

                    verify_prompt = (
                        "Verify this RAG question/answer against the source passages. Return supported, unambiguous, "
                        "all_evidence_relevant, category_valid, and a brief reason.\nCANDIDATE:\n"
                        + json.dumps(draft.model_dump(), ensure_ascii=False)
                        + "\nSOURCE PASSAGES:\n" + json.dumps(window, ensure_ascii=False)
                    )
                    work_counts["verification_operations"] += 1
                    verdict = await _run_with_timeout(verifier.llm.agenerate(verify_prompt, SourceVerdict), verifier_settings.timeout, cancellation_event)
                    verdict = SourceVerdict.model_validate(verdict)
                    if not all((verdict.supported, verdict.unambiguous, verdict.all_evidence_relevant, verdict.category_valid)):
                        return None, {"candidate": index + 1, "stage": "verification", "reason": verdict.reason}

                    ids, groups = evidence_labels(draft, sources)
                    pages = sorted({p for s in sources if s["chunk_id"] in ids for p in s["pages"]})
                    row = {
                        **draft.model_dump(), "id": "auto-" + _hash([draft.question, ids])[:16],
                        "document_hash": expected_hash, "answerable": True, "reference_pages": pages,
                        "relevant_chunk_ids": ids, "relevant_chunk_groups": groups, "verification": verdict.model_dump(),
                    }
                    load_evaluation_dataset([row], expected_document_hash=expected_hash)
                    return row, None
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if provider_status_code(exc) == 429:
                        provider_limit = {"status_code": 429, "model": model}
                    return None, {"candidate": index + 1, "stage": "error", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}

        try:
            next_index = 0
            while len(accepted) < question_count and next_index < plan["maximum_candidates"] and not provider_limit:
                count = min(question_count - len(accepted), plan["maximum_candidates"] - next_index)
                results = await asyncio.gather(*(process_candidate(i) for i in range(next_index, next_index + count)))
                next_index += count
                for row, err in results:
                    if err:
                        rejected.append(err)
                    elif (key := _question_key(row["question"])) not in question_keys:
                        question_keys.add(key)
                        accepted.append(row)
                        done = len(accepted)
                        emit(f"Verified {done}/{question_count} questions")
            if not accepted:
                last_reason = rejected[-1]["reason"] if rejected else "No candidate passed verification."
                raise EvaluationDataError(f"Automatic evaluation produced no source-verified questions: {last_reason}")

            warnings = [
                "References are synthetic and automatically source-verified; they are not human-reviewed ground truth.",
                "Same model drafted and verified these references." if same_model else "Drafting and verification used separate models.",
            ]
            dataset = {
                "schema_version": SCHEMA_VERSION, "document_hash": expected_hash, "questions": accepted,
                "provenance": {
                    "origin": "synthetic_source_verified", "human_reviewed": False,
                    "created_at": datetime.now(timezone.utc).isoformat(), "configuration": configuration,
                    "requested_questions": question_count, "verified_questions": len(accepted),
                    "rejected_candidates": rejected, "work_counts": work_counts,
                    "verifier_usage": dict(getattr(verifier.llm, "usage", {})),
                    "provider_limit": provider_limit, "seconds": round(time.perf_counter() - started, 3),
                    "categories": sorted({r["category"] for r in accepted}), "source_chunks": len(sources),
                },
                "warnings": warnings,
            }
            load_evaluation_dataset(dataset, expected_document_hash=expected_hash)
            write_json(path, dataset)
            emit(f"Automatic dataset ready: {len(accepted)} verified questions")
            return {**dataset, "cache_reused": False, "dataset_path": str(path)}
        finally:
            await verifier.close()
            if provider == "ollama":
                await close_ollama_model(draft_model)
