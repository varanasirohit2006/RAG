"""Document-linked evaluations with bounded parallel work, exact evidence, and reusable results."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import importlib.metadata
import json
import os
import random
import re
import math
import statistics
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from evaluation_data import (
    EvaluationDataError, dataset_template, document_id, document_pages, lexical_reference_metrics,
    load_evaluation_dataset, resolve_relevance, retrieval_metrics, validate_score,
)
from evaluation_judge import JudgeSettings, RagasJudge, JudgeQuotaExceeded, provider_status_code
from provider_limits import (OpenRouterLimitExceeded, current_openrouter_limit, limit_message,
                             record_openrouter_limit)

DEFAULT_STRATEGIES = {
    "preview": ("Production",),
    "full": ("Keyword", "Vector", "Hybrid", "Reranking"),
    "retrieval": ("Keyword", "Vector", "Hybrid"),
}
PROFILE_METRICS = {
    "preview": ("faithfulness", "factual_correctness", "context_precision", "context_recall", "answer_relevancy"),
    "full": ("faithfulness", "factual_correctness", "context_precision", "context_recall", "answer_relevancy"),
    "retrieval": (),
}
_run_locks = {}
_run_lock_guard = threading.Lock()


def _metric(label, description, *, method="local", minimum=0, direction="higher", applicability="answerable cases"):
    return {"label": label, "description": description, "method": method,
            "range": [minimum, 1], "direction": direction, "applicability": applicability}


METRIC_CATALOG = {
    "faithfulness": _metric("Faithfulness", "Fraction of answer claims supported by the exact evidence supplied to the generator.", method="Ragas judge"),
    "answer_relevancy": _metric("Answer relevancy", "Cosine similarity of the question to questions reconstructed from the answer, penalizing noncommittal answers. Negative cosine values are possible.", method="Ragas judge + local embeddings", minimum=-1),
    "context_precision": _metric("Context precision", "Rank-sensitive precision of retrieved text judged against the source reference answer.", method="Ragas judge", applicability="answerable cases with text evidence"),
    "context_recall": _metric("Context recall", "Fraction of reference answer claims supported by the retrieved text.", method="Ragas judge", applicability="answerable cases with text evidence"),
    "factual_correctness": _metric("Factual correctness", "Claim-level F1 of the generated answer against the source reference answer.", method="Ragas judge"),
    "multimodal_faithfulness": _metric("Visual faithfulness", "Answer grounding in the actual text and image evidence; requires a judge with vision support.", method="Ragas multimodal judge", applicability="answerable cases with image evidence"),
    "precision_at_k": _metric("Precision@k", "Relevant unique retrieved chunks divided by the declared retrieval cutoff k. Missing slots count against precision."),
    "recall_at_k": _metric("Recall@k", "Relevant unique chunks retrieved divided by the source-labelled relevant chunk count."),
    "mrr": _metric("MRR", "Reciprocal rank of the first relevant chunk, or zero if none is retrieved."),
    "ndcg_at_k": _metric("nDCG@k", "Discounted graded relevance of the retrieved ranking relative to the ideal source-labelled ranking."),
    "hit_rate_at_k": _metric("Hit rate@k", "One when at least one relevant chunk appears in the first k unique results, otherwise zero."),
    "map_at_k": _metric("MAP@k", "Per-question average precision at relevant ranks divided by min(k, number of relevant chunks); the report averages these values."),
    "normalized_exact_match": _metric("Exact match", "Exact reference match after Unicode, case and whitespace normalization. This lexical diagnostic does not establish correctness."),
    "token_f1": _metric("Token F1", "Lexical token overlap F1 against the reference; preserves numeric differences and does not measure semantic correctness."),
    "citation_page_validity": _metric("Citation page validity", "Fraction of cited page numbers present in the actual retrieved evidence. Structural validity does not establish claim support."),
    "citation_page_coverage": _metric("Citation evidence-page coverage", "Fraction of labelled reference pages mentioned in citations. This is evidence-page coverage, not answer-claim coverage.", applicability="answerable cases with source page labels"),
    "citation_source_validity": _metric("Citation source validity", "Fraction of cited source numbers corresponding to actual generator sources. Structural validity does not establish semantic support."),
    "abstention_accuracy": _metric("Correct abstention", "Judge classification of whether an unanswerable response explicitly abstains without inventing an answer.", method="structured judge", applicability="explicitly labelled unanswerable cases"),
    "false_answer_rate": _metric("False-answer rate", "One minus correct abstention, on explicitly labelled unanswerable cases only.", method="derived from abstention judge", direction="lower", applicability="explicitly labelled unanswerable cases"),
    "failure_rate": _metric("Failed or partial attempt rate", "Failed or partial samples divided by completed, partial and failed attempts; cancelled samples are excluded.", direction="lower", applicability="attempted samples"),
    "error_rate": _metric("Error rate", "Failed samples divided by completed, partial and failed attempts; cancelled samples are excluded.", direction="lower", applicability="attempted samples"),
    "partial_rate": _metric("Partial-result rate", "Partial samples divided by completed, partial and failed attempts; cancelled samples are excluded.", direction="lower", applicability="attempted samples"),
}
for _stage in ("retrieval_seconds", "local_retrieval_seconds", "reranking_seconds", "generation_seconds", "judge_seconds", "sample_seconds"):
    METRIC_CATALOG[_stage] = {"label": _stage.replace("_", " ").capitalize(), "description": "Measured duration, including completed and failed attempts with recorded timing; cache replay remains included. p50/p95 use linear interpolation over observed values.",
                             "method": "wall clock", "range": [0, None], "direction": "lower", "applicability": "samples with recorded duration"}


def _rank_diagnostics(retrieved_ids, grades, *, k):
    relevant = {chunk_id for chunk_id, grade in grades.items() if grade > 0}
    if not relevant:
        return {}
    ranked = list(dict.fromkeys(retrieved_ids))[:k]
    hits = 0
    precision_sum = 0.0
    for rank, chunk_id in enumerate(ranked, 1):
        if chunk_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return {"hit_rate_at_k": float(hits > 0), "map_at_k": precision_sum / min(k, len(relevant))}


def _citation_diagnostics(response, documents, reference_pages=()):
    supplied_pages = set().union(*(document_pages(document) for document in documents))
    cited_pages = []
    for match in re.finditer(r"\b(?:pages?|pp?\.)\s*[:#]?\s*(\d+(?:\s*(?:,|and|[-–])\s*\d+)*)", response, flags=re.I):
        value = match.group(1)
        # Expand short page ranges, rejecting oversized ranges as invalid endpoints.
        for segment in re.split(r"\s*(?:,|and)\s*", value):
            endpoints = re.split(r"\s*[-–]\s*", segment)
            first = int(endpoints[0])
            if len(endpoints) == 2 and first <= int(endpoints[1]) <= first + 500:
                cited_pages.extend(range(first, int(endpoints[1]) + 1))
            else:
                cited_pages.extend(int(item) for item in endpoints)
    cited_pages = set(cited_pages)
    cited_sources = {int(value) for value in re.findall(r"\bsource\s+(\d+)", response, flags=re.I)}
    values = {}
    if supplied_pages:
        values["citation_page_validity"] = len(cited_pages & supplied_pages) / len(cited_pages) if cited_pages else 0.0
    if reference_pages:
        values["citation_page_coverage"] = len(cited_pages & set(reference_pages)) / len(set(reference_pages))
    if documents:
        values["citation_source_validity"] = sum(1 <= rank <= len(documents) for rank in cited_sources) / len(cited_sources) if cited_sources else 0.0
    return values


def _percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _error(exc):
    message = str(exc) or type(exc).__name__
    for name in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"):
        value = os.getenv(name)
        if value:
            message = message.replace(value, "[redacted]")
    return f"{type(exc).__name__}: {message[:1500]}"


def _revision():
    root = Path(__file__).resolve().parent
    files = ("advanced_evaluation.py", "evaluation_data.py", "evaluation_judge.py",
             "rag_pipeline.py", "generation_runtime.py", "retrieval_runtime.py", "index_runtime.py")
    return _hash({name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                  for name in files if (root / name).is_file()})


def _reranker_configuration(strategies):
    """Mirror runtime defaults without storing a credential value in provenance."""
    if not ({"Production", "Reranking"} & set(strategies)):
        return {"provider": "none", "model": None, "timeout_seconds": None, "credentials_available": None}
    provider = os.getenv("RERANKER_PROVIDER", "cohere" if os.getenv("COHERE_API_KEY") else "none")
    return {
        "provider": provider,
        "model": os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0") if provider == "cohere" else None,
        "timeout_seconds": int(os.getenv("RERANK_TIMEOUT_SECONDS", "15")) if provider == "cohere" else None,
        "credentials_available": bool(os.getenv("COHERE_API_KEY")) if provider == "cohere" else None,
    }


def get_db_path():
    directories = sorted(path for path in Path("db").glob("chroma_*") if path.is_dir())
    if len(directories) == 1:
        return str(directories[0])
    raise EvaluationDataError("Select an explicit --db index; evaluation must not silently choose another document.")


def _selection(cases, profile, strategies=None):
    if profile not in DEFAULT_STRATEGIES:
        raise ValueError("Evaluation profile must be preview, full, or retrieval.")
    strategies = tuple(dict.fromkeys(strategies or DEFAULT_STRATEGIES[profile]))
    if not strategies or set(strategies) - {"Keyword", "Vector", "Hybrid", "Production", "Reranking"}:
        raise ValueError("Strategies must be Keyword, Vector, Hybrid, Reranking, or Production.")
    if "Production" in strategies and "Reranking" in strategies:
        raise ValueError("Production and Reranking use the same pipeline; select one label to avoid duplicate samples.")
    if profile == "retrieval" and ({"Production", "Reranking"} & set(strategies)):
        raise ValueError("The local retrieval profile excludes Production's external reranker. Use Keyword, Vector, or Hybrid.")
    selected = list(cases)
    if profile == "preview":
        limit = int(os.getenv("EVALUATION_PREVIEW_QUESTIONS", "5"))
        if not 1 <= limit <= 20:
            raise ValueError("EVALUATION_PREVIEW_QUESTIONS must be between 1 and 20.")
        selected = sorted(cases, key=lambda case: _hash([case.id, case.question]))[:limit]
        unanswerable = next((case for case in cases if not case.answerable), None)
        if unanswerable and selected and all(case.answerable for case in selected):
            selected[-1] = unanswerable
    return selected, strategies


def evaluation_plan(dataset, *, profile="preview", strategies=None):
    cases = load_evaluation_dataset(dataset)
    selected, strategies = _selection(cases, profile, strategies)
    return {
        "profile": profile, "total_questions": len(cases), "sampled_questions": len(selected),
        "strategies": list(strategies), "samples": len(selected) * len(strategies),
        "judge_metrics": list(PROFILE_METRICS[profile]), "generator_required": profile != "retrieval",
        "scope": "deterministic preview subset" if profile == "preview" else "all imported cases",
        "metric_catalog": METRIC_CATALOG,
    }


def _bootstrap_interval(values):
    if len(values) < 2:
        return None, None
    rng = random.Random(0)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(500))
    return means[12], means[487]


def summarize_records(records):
    """Every aggregate has a denominator and failed/skipped counts; invalid scores never rank."""
    import pandas as pd

    metric_names = sorted({name for record in records for name in record.get("metrics", {})})
    summary = []
    for strategy in dict.fromkeys(record["strategy"] for record in records):
        rows = [record for record in records if record["strategy"] == strategy]
        row = {
            "strategy": strategy, "samples": len(rows),
            "completed": sum(record["status"] == "completed" for record in rows),
            "failed": sum(record["status"] in {"failed", "partial"} for record in rows),
            "skipped": sum(record["status"] in {"skipped", "cancelled"} for record in rows),
            "retrieval_seconds": statistics.mean(record.get("retrieval_seconds", 0) for record in rows),
            "local_retrieval_seconds": statistics.mean(record.get("local_retrieval_seconds", 0) for record in rows),
            "reranking_seconds": statistics.mean(record.get("reranking_seconds", 0) for record in rows),
            "generation_seconds": statistics.mean(record.get("generation_seconds", 0) for record in rows),
            "judge_seconds": statistics.mean(record.get("judge_seconds", 0) for record in rows),
        }
        attempted = [record for record in rows if record["status"] in {"completed", "partial", "failed"}]
        row.update({
            "attempted": len(attempted),
            "failure_rate": sum(record["status"] in {"partial", "failed"} for record in attempted) / len(attempted) if attempted else None,
            "error_rate": sum(record["status"] == "failed" for record in attempted) / len(attempted) if attempted else None,
            "partial_rate": sum(record["status"] == "partial" for record in attempted) / len(attempted) if attempted else None,
            "reranked_samples": sum(record.get("retrieval_trace", {}).get("reranker", "none") != "none" for record in rows),
            "degraded_samples": sum(bool(record.get("degraded")) for record in rows),
        })
        for stage in ("retrieval_seconds", "local_retrieval_seconds", "reranking_seconds", "generation_seconds", "judge_seconds", "sample_seconds"):
            measured = [float(record[stage]) for record in rows if isinstance(record.get(stage), (int, float))
                        and not isinstance(record[stage], bool) and math.isfinite(record[stage]) and record[stage] >= 0]
            row[stage] = statistics.mean(measured) if measured else None
            row[stage + "_count"] = len(measured)
            row[stage + "_p50"] = _percentile(measured, .5)
            row[stage + "_p95"] = _percentile(measured, .95)
        for name in metric_names:
            entries = [record.get("metrics", {}).get(name, {"status": "not_applicable"}) for record in rows]
            values = []
            failed = 0
            for entry in entries:
                if entry.get("status") == "completed":
                    try:
                        values.append(validate_score(name, entry.get("value")))
                    except ValueError:
                        failed += 1
                elif entry.get("status") == "failed":
                    failed += 1
            lower, upper = _bootstrap_interval(values)
            row[name] = statistics.mean(values) if values else None
            row[name + "_count"] = len(values)
            row[name + "_failed"] = failed
            row[name + "_skipped"] = len(entries) - len(values) - failed
            row[name + "_lower95"] = lower
            row[name + "_upper95"] = upper
        summary.append(row)
    return pd.DataFrame(summary)


def _result_attrs(summary, manifest, directory, *, cached=False):
    summary.attrs.update({
        **manifest,
        "manifest_path": str((directory / "manifest.json").resolve()),
        "sample_path": str((directory / "samples.jsonl").resolve()),
        "summary_path": str((directory / "summary.csv").resolve()),
        "cache_reused": cached,
    })
    return summary


def _valid_completed_records(records):
    try:
        for record in records:
            if record.get("status") != "completed" or not isinstance(record.get("metrics"), dict):
                return False
            for name, result in record["metrics"].items():
                if result.get("status") == "completed":
                    validate_score(name, result.get("value"))
                elif result.get("status") != "not_applicable":
                    return False
        return True
    except (TypeError, ValueError):
        return False


async def run_evaluation(
    db_path=None, *, dataset=None, profile="preview", strategies=None,
    progress_callback=None, cancellation_event=None, output_dir="evaluation_runs",
    force=False, expected_document_hash=None,
):
    """Score reviewed cases with bounded concurrency; callbacks are (done,total,message)."""
    from dotenv import load_dotenv

    load_dotenv()
    dataset_provenance = dataset.get("provenance") if isinstance(dataset, dict) else None
    dataset_warnings = dataset.get("warnings", []) if isinstance(dataset, dict) else []
    dataset_warnings = [str(message) for message in dataset_warnings] if isinstance(dataset_warnings, list) else []
    cases = load_evaluation_dataset(dataset, expected_document_hash=expected_document_hash)
    selected, strategies = _selection(cases, profile, strategies)
    total = len(selected) * len(strategies)
    done = 0

    def emit(message):
        if progress_callback:
            progress_callback(done, total, message)

    def cancelled():
        return bool(cancellation_event and cancellation_event.is_set())

    if cancelled():
        raise asyncio.CancelledError("Evaluation cancelled before work started.")
    concurrency = int(os.getenv("EVALUATION_CONCURRENCY", "4"))
    if not 1 <= concurrency <= 8:
        raise ValueError("EVALUATION_CONCURRENCY must be between 1 and 8.")
    from rag_pipeline import get_embedding_model, load_vector_store, generate_final_answer
    from retrieval_runtime import get_retrievers, retrieve_documents
    from generation_runtime import prepare_generation_context

    emit("Loading the prepared index; no document ingestion is repeated")
    target = Path(db_path or get_db_path()).resolve()
    db = await asyncio.to_thread(load_vector_store, str(target))
    if db is None:
        raise EvaluationDataError("Prepared vector index was not found; prepare this document first.")
    index_manifest = _read_json(target / "index_manifest.json") or {}
    index_hash = index_manifest.get("document_hash")
    if index_hash and cases[0].document_hash != index_hash:
        raise EvaluationDataError("Dataset document_hash does not match the vector index's source PDF.")
    if not index_hash and not expected_document_hash:
        raise EvaluationDataError("Legacy index has no source document hash. Supply --document-hash or rebuild this document's index.")
    retrieval_k = int(os.getenv("EVALUATION_K", "5")) if profile == "retrieval" else int(os.getenv("RETRIEVAL_K", "10"))
    final_k = retrieval_k if profile == "retrieval" else int(os.getenv("RETRIEVAL_TOP_N", "3"))
    if not 1 <= final_k <= retrieval_k <= 50:
        raise ValueError("Retrieval limits must satisfy 1 <= final context count <= retrieval k <= 50.")
    vector, keyword, hybrid, all_docs = await asyncio.to_thread(get_retrievers, db, k=retrieval_k)
    relevance = {case.id: resolve_relevance(case, all_docs) for case in cases}
    corpus_version = index_manifest.get("corpus_version") or _hash(sorted(document_id(document) for document in all_docs))
    generator_provider = os.getenv("GENERATION_PROVIDER", "google")
    generator_model = os.getenv("GENERATION_MODEL", "gemini-3.5-flash-lite")
    judge_settings = JudgeSettings.from_env() if profile != "retrieval" else None
    metric_timeout = int(os.getenv('EVALUATION_METRIC_TIMEOUT', str(max(300, judge_settings.timeout * 4)))) if judge_settings else None
    if metric_timeout is not None and not 15 <= metric_timeout <= 1200:
        raise ValueError('EVALUATION_METRIC_TIMEOUT must be between 15 and 1200 seconds.')
    versions = {}
    for package in ("ragas", "langchain-google-genai", "langchain-openai", "openai", "sentence-transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    config = {
        "implementation_revision": _revision(), "package_versions": versions,
        "index_path": str(target), "document_hash": cases[0].document_hash,
        "corpus_version": corpus_version, "profile": profile, "strategies": list(strategies),
        'corpus_type': index_manifest.get('corpus_type', 'single_pdf'),
        'source_documents': index_manifest.get('source_documents', []),
        "index_configuration": {name: index_manifest.get(name) for name in (
            "embedding_model", "embedding_revision", "pipeline_version", "chunk_config_version",
        )},
        "dataset": [case.to_dict() for case in cases], "selected_ids": [case.id for case in selected],
        "dataset_provenance": dataset_provenance,
        "generator_provider": generator_provider if profile != "retrieval" else None,
        "generator_model": generator_model if profile != "retrieval" else None,
        "judge": asdict(judge_settings) if judge_settings else None,
        "metric_timeout_seconds": metric_timeout,
        "retrieval_k": retrieval_k, "metric_k": final_k,
        "generation_limits": {name: os.getenv(name) for name in (
            "GENERATION_MAX_CONTEXT_CHARS", "GENERATION_MAX_IMAGES", "GENERATION_MAX_OUTPUT_TOKENS",
            "MODEL_TIMEOUT_SECONDS", "GENERATION_IMAGE_MAX_DIM", "GENERATION_THINKING_BUDGET",
        )},
        "reranker": _reranker_configuration(strategies),
    }
    run_key = _hash(config)
    directory = Path(output_dir).resolve() / run_key
    manifest_path = directory / "manifest.json"
    old_manifest = _read_json(manifest_path)
    if old_manifest and old_manifest.get("status") == "completed" and not force:
        try:
            records = [json.loads(line) for line in (directory / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
        except (OSError, json.JSONDecodeError):
            records = []
        if len(records) == total and _valid_completed_records(records):
            done = total
            emit("Reusing completed evaluation with identical source, questions, models, and configuration")
            return _result_attrs(summarize_records(records), old_manifest, directory, cached=True)
    with _run_lock_guard:
        lock = _run_locks.setdefault(run_key, threading.Lock())
    if not lock.acquire(blocking=False):
        raise RuntimeError("An evaluation with this exact configuration is already running.")
    started = time.perf_counter()
    manifest = {
        "schema_version": 2, "run_id": run_key[:16], "run_key": run_key,
        "status": "running", "started_at": _utc(), "profile": profile,
        "total_questions": len(cases), "sampled_questions": len(selected), "samples": total,
        "strategies": list(strategies), "document_hash": cases[0].document_hash,
        "corpus_version": corpus_version, "configuration": config,
        'corpus_type': config['corpus_type'], 'source_documents': config['source_documents'],
        "scope": "deterministic preview subset" if profile == "preview" else "all imported cases",
        "warnings": ["Preview results do not establish full-corpus quality."] if profile == "preview" else [],
        "metric_catalog": METRIC_CATALOG,
        "dataset_provenance": dataset_provenance,
        "dataset_warnings": dataset_warnings,
        "timing_scope": "Per-stage timings include measured failed attempts and cache replay. Sample durations exclude queue wait; report seconds measure the overall evaluation run.",
    }
    if dataset_provenance:
        manifest["scope"] = "deterministic subset of generated cases" if profile == "preview" else "all generated cases"
    manifest["warnings"].extend(dataset_warnings)
    if "Reranking" in strategies and config["reranker"]["provider"] == "none":
        manifest["warnings"].append("Reranking is disabled; its requested strategy uses Hybrid evidence. No reranker quality comparison was performed.")
    if judge_settings and generator_provider == judge_settings.provider and generator_model == judge_settings.model:
        manifest["warnings"].append("Generator and judge use the same model; an independent judge model is recommended.")
    if any(case.reference_pages and not (case.relevant_chunk_ids or case.relevance_grades) for case in cases):
        manifest["warnings"].append("Some retrieval labels are page-level; all chunks on those pages count as relevant. Review finer chunk labels for precision claims.")
    cache_directory = Path(output_dir).resolve() / "cache"
    records = []
    judge = None
    shared_answers = {}
    shared_metrics = {}
    sample_semaphore = asyncio.Semaphore(concurrency)
    call_semaphore = asyncio.Semaphore(concurrency)
    cache_hits = 0
    work_counts = {"generation_operations": 0, "judge_metric_operations": 0}
    provider_limit = None

    async def answer_for(case, docs, context):
        nonlocal cache_hits
        identity = {
            "revision": config["implementation_revision"], "index_path": str(target),
            "provider": generator_provider, "model": generator_model,
            "question": case.question,
            "prompt": context.get("prompt"), "texts": context.get("text_contexts"),
            "images": [_hash(payload) for payload in context.get("images_base64", [])],
            "limits": config["generation_limits"],
        }
        answer_key = _hash(identity)
        path = cache_directory / "answers" / (answer_key + ".json")
        cached = _read_json(path) if not force else None
        if cached and isinstance(cached.get("answer"), str) and cached["answer"].strip():
            cache_hits += 1
            return cached["answer"], True
        if answer_key not in shared_answers:
            async def generate():
                async with call_semaphore:
                    if cancelled():
                        raise asyncio.CancelledError()
                    if provider_limit and generator_provider == provider_limit.get('provider') and (
                        generator_provider == 'openrouter' or generator_model == provider_limit.get('model')):
                        raise JudgeQuotaExceeded(provider_limit)
                    work_counts["generation_operations"] += 1
                    answer = await asyncio.to_thread(
                        generate_final_answer, docs, case.question,
                        provider=generator_provider, model_name=generator_model,
                    )
                if not isinstance(answer, str) or not answer.strip() or answer == "Error generating answer.":
                    raise RuntimeError("Generator did not return a usable answer; failures cannot be scored as responses.")
                _write_json(path, {"answer": answer, "created_at": _utc(), "provider": generator_provider, "model": generator_model})
                return answer
            shared_answers[answer_key] = asyncio.create_task(generate())
            reused = False
        else:
            reused = True
            cache_hits += 1
        answer = await asyncio.shield(shared_answers[answer_key])
        return answer, reused

    async def metric_for(name, case, response, context, image_paths):
        nonlocal cache_hits
        identity = {
            "revision": config["implementation_revision"], "index_path": str(target),
            "index_configuration": config["index_configuration"],
            "judge": config["judge"], "name": name, "question": case.question,
            "response": response, "reference": case.reference_answer,
            "texts": context["text_contexts"], "images": [_hash(payload) for payload in context.get("images_base64", [])],
        }
        key = _hash(identity)
        path = cache_directory / "metrics" / (key + ".json")
        cached = _read_json(path) if not force else None
        if cached:
            try:
                value = validate_score(name, cached["value"])
                cache_hits += 1
                return {"status": "completed", "value": value, "reason": cached.get("reason"), "cache_hit": True}
            except (KeyError, ValueError):
                pass
        if key not in shared_metrics:
            async def evaluate():
                async with call_semaphore:
                    if cancelled():
                        return {"status": "skipped", "reason": "cancelled"}
                    if provider_limit:
                        return {"status": "skipped", "reason": limit_message(provider_limit), "provider_status_code": 429}
                    work_counts["judge_metric_operations"] += 1
                    budget = asyncio.timeout(metric_timeout)
                    try:
                        async with budget:
                            result = await judge.score(
                                name, question=case.question, response=response, reference=case.reference_answer,
                                contexts=context["text_contexts"], image_contexts=image_paths)
                    except TimeoutError as exc:
                        explanation = f'{name} exceeded its {metric_timeout}-second complete-metric deadline' if budget.expired() else f'A provider request timed out during {name} (request limit: {judge_settings.timeout} seconds)'
                        raise TimeoutError(explanation + '; completed judging steps are retained for retry.') from exc
                value = validate_score(name, result["value"])
                _write_json(path, {"value": value, "reason": result.get("reason"), "created_at": _utc()})
                return {"status": "completed", "value": value, "reason": result.get("reason"), "cache_hit": False}
            shared_metrics[key] = asyncio.create_task(evaluate())
            reused = False
        else:
            cache_hits += 1
            reused = True
        result = await asyncio.shield(shared_metrics[key])
        return {**result, "cache_hit": reused}

    async def process(case, strategy):
        nonlocal done, provider_limit
        record = {
            "run_id": manifest["run_id"],
            "question_id": case.id, "question": case.question, "reference_answer": case.reference_answer,
            "answerable": case.answerable, "document_hash": case.document_hash, "strategy": strategy,
            "status": "running", "metrics": {}, "retrieval_k": final_k,
            "label_granularity": "chunk" if case.relevant_chunk_ids or case.relevance_grades else "page" if case.reference_pages else "unanswerable",
        }
        for name in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "hit_rate_at_k", "map_at_k"):
            record["metrics"][name] = {"status": "skipped" if case.answerable else "not_applicable",
                                       "reason": "Retrieval did not complete." if case.answerable else "No positive relevance labels for an unanswerable case."}
        if profile != "retrieval":
            for name in PROFILE_METRICS[profile]:
                record["metrics"][name] = {
                    "status": "skipped" if case.answerable else "not_applicable",
                    "reason": "Answer generation did not complete." if case.answerable else "Unanswerable case is scored separately for abstention.",
                }
            if not case.answerable:
                record["metrics"]["abstention_accuracy"] = {"status": "skipped", "reason": "Answer generation did not complete."}
        async with sample_semaphore:
            sample_started = time.perf_counter()
            try:
                if cancelled():
                    record["status"] = "cancelled"
                    return record
                emit(f"{strategy} / {case.id}: retrieving evidence")
                stage = time.perf_counter()
                try:
                    if strategy in {"Production", "Reranking"}:
                        docs, trace = await asyncio.to_thread(retrieve_documents, db, case.question, k=retrieval_k, top_n=final_k)
                        record["retrieval_trace"] = trace
                        record["local_retrieval_seconds"] = round(trace.get("retrieval_seconds", 0), 4)
                        record["reranking_seconds"] = round(trace.get("reranking_seconds", 0), 4)
                        record["effective_strategy"] = "Reranking" if trace.get("reranker", "none") != "none" else "Hybrid"
                        if trace.get("reranking_error") or (strategy == "Reranking" and trace.get("reranker", "none") == "none"):
                            record["degraded"] = True
                            reason = f"{strategy}: reranking failed ({trace['reranking_error']}); Hybrid evidence was used." if trace.get("reranking_error") else "Reranking disabled; Hybrid evidence was used."
                            record["warnings"] = [reason]
                            if reason not in manifest["warnings"]:
                                manifest["warnings"].append(reason)
                    else:
                        retriever = {"Keyword": keyword, "Vector": vector, "Hybrid": hybrid}[strategy]
                        docs = (await asyncio.to_thread(retriever.invoke, case.question))[:final_k]
                finally:
                    record["retrieval_seconds"] = round(time.perf_counter() - stage, 4)
                    if strategy not in {"Production", "Reranking"}:
                        record["local_retrieval_seconds"] = record["retrieval_seconds"]
                        record["reranking_seconds"] = 0.0
                if not docs:
                    raise RuntimeError("No source evidence was retrieved.")
                context = prepare_generation_context(docs, case.question, provider=generator_provider) if profile != "retrieval" else {
                    "chunk_ids": [document_id(document) for document in docs],
                    "text_contexts": [document.page_content for document in docs], "images_base64": [],
                }
                record["retrieved_chunk_ids"] = context["chunk_ids"]
                record["contexts"] = context["text_contexts"]
                record["source_image_paths"] = context.get("image_paths", [])
                record["image_contexts"] = []
                for payload in context.get("images_base64", []):
                    payload = payload.split(",", 1)[-1] if payload.startswith("data:") else payload
                    image_path = directory / "images" / (_hash(payload) + ".jpg")
                    if not image_path.is_file():
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        image_path.write_bytes(base64.b64decode(payload, validate=True))
                    record["image_contexts"].append(str(image_path))
                for name, value in retrieval_metrics(record["retrieved_chunk_ids"], relevance[case.id], k=final_k).items():
                    record["metrics"][name] = {"status": "completed", "value": validate_score(name, value)}
                for name, value in _rank_diagnostics(record["retrieved_chunk_ids"], relevance[case.id], k=final_k).items():
                    record["metrics"][name] = {"status": "completed", "value": validate_score(name, value)}
                if profile == "retrieval":
                    record["status"] = "completed"
                    return record
                if cancelled():
                    record["status"] = "cancelled"
                    return record
                emit(f"{strategy} / {case.id}: generating an answer from the production evidence")
                stage = time.perf_counter()
                try:
                    response, reused = await answer_for(case, docs, context)
                finally:
                    record["generation_seconds"] = round(time.perf_counter() - stage, 4)
                record["generation_cache_hit"] = reused
                record["response"] = response
                if case.answerable:
                    for name, value in lexical_reference_metrics(response, case.reference_answer).items():
                        record["metrics"][name] = {"status": "completed", "value": value}
                    reference_pages = set(case.reference_pages)
                    if not reference_pages:
                        reference_pages = set().union(*(document_pages(document) for document in all_docs if relevance[case.id].get(document_id(document), 0) > 0))
                    # Context limits can omit later retrieved passages. Citations must
                    # be checked against the evidence actually supplied to generation.
                    supplied_ids = set(context["chunk_ids"])
                    supplied_documents = [document for document in docs if document_id(document) in supplied_ids]
                    for name, value in _citation_diagnostics(response, supplied_documents, reference_pages).items():
                        record["metrics"][name] = {"status": "completed", "value": validate_score(name, value)}
                names = list(PROFILE_METRICS[profile]) if case.answerable else ["abstention_accuracy"]
                if context.get("images_base64") and case.answerable:
                    names = ["multimodal_faithfulness" if name == "faithfulness" else name for name in names]
                    record["metrics"]["faithfulness"] = {"status": "not_applicable", "reason": "Actual visual evidence is scored by multimodal_faithfulness."}
                    for name in ("context_precision", "context_recall"):
                        if name in names:
                            names.remove(name)
                            record["metrics"][name] = {"status": "not_applicable", "reason": "Text context metrics do not score visual evidence; use retrieval labels and multimodal faithfulness."}
                if not case.answerable:
                    for name in PROFILE_METRICS[profile]:
                        record["metrics"][name] = {"status": "not_applicable", "reason": "Unanswerable case is scored separately for abstention."}
                emit(f"{strategy} / {case.id}: judging {len(names)} applicable quality metrics")
                stage = time.perf_counter()
                async def score_one(name):
                    nonlocal provider_limit
                    try:
                        result = await metric_for(name, case, response, context, record["image_contexts"])
                    except asyncio.CancelledError:
                        result = {"status": "skipped", "reason": "cancelled"}
                    except Exception as exc:
                        if provider_status_code(exc) == 429:
                            details = record_openrouter_limit(exc) if judge_settings.provider == 'openrouter' else {}
                            provider_limit = {**details, "status_code": 429, "provider": judge_settings.provider, "model": judge_settings.model,
                                              "reason": limit_message(details) if details else str(JudgeQuotaExceeded())}
                            result = {"status": "skipped" if isinstance(exc, JudgeQuotaExceeded) else "failed",
                                      "error": provider_limit['reason'], "reason": provider_limit['reason'], "provider_status_code": 429}
                        else:
                            result = {"status": "failed", "error": _error(exc)}
                    record["metrics"][name] = result
                try:
                    await asyncio.gather(*(score_one(name) for name in names))
                finally:
                    record["judge_seconds"] = round(time.perf_counter() - stage, 4)
                if not case.answerable:
                    abstention = record["metrics"].get("abstention_accuracy", {})
                    record["metrics"]["false_answer_rate"] = {
                        **abstention,
                        **({"value": 1 - abstention["value"]} if abstention.get("status") == "completed" else {}),
                    }
                record["status"] = "cancelled" if cancelled() or any(
                    metric.get("status") == "skipped" and metric.get("reason") == "cancelled"
                    for metric in record["metrics"].values()
                ) else "partial" if any(
                    metric.get("status") == "failed" or metric.get("provider_status_code") == 429 for metric in record["metrics"].values()
                ) else "completed"
                return record
            except asyncio.CancelledError:
                record["status"] = "cancelled"
                return record
            except Exception as exc:
                if provider_status_code(exc) == 429:
                    details = record_openrouter_limit(exc) if generator_provider == 'openrouter' else {}
                    provider_limit = {**details, 'provider': generator_provider, 'model': generator_model, 'status_code': 429}
                    provider_limit['reason'] = limit_message(provider_limit)
                    record['status'] = 'skipped'
                    record['error'] = provider_limit['reason']
                    for name, metric in record['metrics'].items():
                        if metric['status'] == 'skipped':
                            record['metrics'][name] = {'status': 'skipped', 'reason': provider_limit['reason'], 'provider_status_code': 429}
                else:
                    record["status"] = "failed"
                    record["error"] = _error(exc)
                return record
            finally:
                record["sample_seconds"] = round(time.perf_counter() - sample_started, 4)
                if record["status"] == "running":
                    record["status"] = "failed"
                records.append(record)
                case_path = directory / "cases" / (_hash([case.id, strategy]) + ".json")
                _write_json(case_path, record)
                done += 1
                emit(f"{done}/{total} samples finished; {strategy} / {case.id}: {record['status']}")

    try:
        _write_json(manifest_path, manifest)
        if profile != "retrieval":
            emit("Preparing the configured evaluation judge; completed answers and scores will be reused")
            judge = RagasJudge(judge_settings, embedding_model=get_embedding_model())
            judge.prepare()
        # Rotate questions while alternating strategies: quota cannot all go to Keyword.
        scheduled = [(selected[(round_index + strategy_index) % len(selected)], strategy)
                     for round_index in range(len(selected)) for strategy_index, strategy in enumerate(strategies)]
        await asyncio.gather(*(process(case, strategy) for case, strategy in scheduled))
        record_order = {(case.id, strategy): index for index, (case, strategy) in enumerate(
            (case, strategy) for case in selected for strategy in strategies
        )}
        records.sort(key=lambda record: record_order[(record["question_id"], record["strategy"])])
        sample_path = directory / "samples.jsonl"
        sample_path.write_text("".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records), encoding="utf-8")
        failed = sum(record["status"] in {"failed", "partial"} for record in records)
        skipped = sum(record["status"] in {"skipped", "cancelled"} for record in records)
        status = "cancelled" if cancelled() else "partial" if (failed or skipped or provider_limit) and any(record["status"] in {"completed", "partial"} for record in records) else "failed" if failed or provider_limit else "completed"
        manifest.update({
            "status": status, "finished_at": _utc(), "seconds": round(time.perf_counter() - started, 3),
            "failed_samples": failed, "skipped_samples": skipped, "cache_hits": cache_hits,
            "completed_samples": sum(record["status"] == "completed" for record in records),
            "work_counts": work_counts,
        })
        if judge is not None and hasattr(getattr(judge, "llm", None), "usage"):
            manifest["judge_usage"] = judge.llm.usage
        if provider_limit:
            manifest["provider_limit"] = provider_limit
            manifest["warnings"].append(provider_limit['reason'])
        summary = summarize_records(records)
        summary.to_csv(directory / "summary.csv", index=False)
        _write_json(manifest_path, manifest)
        emit(f"Evaluation {status}: {len(records)} samples, {failed} failed/partial, {skipped} cancelled/skipped")
        return _result_attrs(summary, manifest, directory)
    except BaseException as exc:
        manifest.update({"status": "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                         "finished_at": _utc(), "error": _error(exc), "seconds": round(time.perf_counter() - started, 3)})
        _write_json(manifest_path, manifest)
        raise
    finally:
        shared_tasks = list(shared_answers.values()) + list(shared_metrics.values())
        for task in shared_tasks:
            if not task.done():
                task.cancel()
        if shared_tasks:
            await asyncio.gather(*shared_tasks, return_exceptions=True)
        try:
            if judge is not None:
                await judge.close()
        finally:
            lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="Prepared document index; inferred only when exactly one exists.")
    dataset_input = parser.add_mutually_exclusive_group(required=True)
    dataset_input.add_argument("--dataset", type=Path, help="Source-linked JSON or CSV questions.")
    dataset_input.add_argument("--automatic", action="store_true", help="Generate source-verified questions automatically, then evaluate all strategies.")
    parser.add_argument("--questions", type=int, default=10, help="Number of automatically generated source-verified questions (default: 10).")
    parser.add_argument("--profile", choices=tuple(DEFAULT_STRATEGIES), default=None)
    parser.add_argument("--strategies", nargs="+", choices=("Keyword", "Vector", "Hybrid", "Reranking", "Production"))
    parser.add_argument("--document-hash", help="Source PDF SHA-256, required for an old index without a manifest hash.")
    parser.add_argument("--output-dir", default="evaluation_runs")
    parser.add_argument("--force", action="store_true", help="Rerun instead of reusing identical completed work.")
    args = parser.parse_args()
    try:
        dataset = args.dataset
        document_hash = args.document_hash
        if args.automatic:
            from evaluation_generation import generate_evaluation_dataset

            db_path = Path(args.db or get_db_path()).resolve()
            document_hash = document_hash or (_read_json(db_path / "index_manifest.json") or {}).get("document_hash")
            if not document_hash:
                raise EvaluationDataError("Automatic evaluation needs the source PDF SHA-256; supply --document-hash or rebuild the index.")
            dataset = asyncio.run(generate_evaluation_dataset(
                str(db_path), expected_document_hash=document_hash, question_count=args.questions,
                force=args.force, progress_callback=lambda done, total, message: print(f"[questions {done}/{total}] {message}", flush=True),
            ))
            for warning in dataset.get("warnings", []):
                print(f"Dataset: {warning}")
        summary = asyncio.run(run_evaluation(
            db_path=args.db, dataset=dataset, profile=args.profile or ("full" if args.automatic else "preview"), strategies=args.strategies,
            expected_document_hash=document_hash, output_dir=args.output_dir, force=args.force,
            progress_callback=lambda done, total, message: print(f"[{done}/{total}] {message}", flush=True),
        ))
        print(summary.to_string(index=False))
        print(f"Profile: {summary.attrs['profile']}; status: {summary.attrs['status']}; report: {summary.attrs['manifest_path']}")
        return int(summary.attrs["status"] != "completed")
    except Exception as exc:
        print(_error(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
