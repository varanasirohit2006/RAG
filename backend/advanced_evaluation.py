"""Automated multi-strategy RAG evaluation runner, IR/Ragas scoring, and summary aggregation."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import random
import re
import statistics
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from backend.evaluation_data import (
    EvaluationDataError, document_id, document_pages, lexical_reference_metrics,
    load_evaluation_dataset, resolve_relevance, retrieval_metrics, validate_score,
)
from backend.evaluation_judge import JUDGE_RUBRIC_VERSION, JudgeQuotaExceeded, JudgeSettings, RagasJudge
from backend.files import PROJECT_ROOT, read_json, write_json
from backend.provider_limits import limit_message, provider_status_code

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
    "recall_at_k": _metric("Recall@k", "Fraction of source-labelled relevant evidence retrieved."),
    "mrr": _metric("MRR", "Reciprocal rank of the first relevant chunk, or zero if none is retrieved."),
    "ndcg_at_k": _metric("nDCG@k", "Discounted graded relevance of the retrieved ranking relative to the ideal source-labelled ranking."),
    "hit_rate_at_k": _metric("Hit rate@k", "One when at least one relevant chunk appears in the first k unique results, otherwise zero."),
    "map_at_k": _metric("MAP@k", "Per-question average precision at relevant ranks divided by min(k, number of relevant chunks)."),
    "normalized_exact_match": _metric("Exact match", "Exact reference match after Unicode, case and whitespace normalization. This lexical diagnostic does not establish correctness."),
    "token_f1": _metric("Token F1", "Lexical token overlap F1 against the reference; preserves numeric differences and does not measure semantic correctness."),
    "citation_page_validity": _metric("Citation page validity", "Fraction of cited page numbers present in the actual retrieved evidence. Structural validity does not establish claim support."),
    "citation_page_coverage": _metric("Citation evidence-page coverage", "Fraction of labelled reference pages mentioned in citations. This is evidence-page coverage, not answer-claim coverage.", applicability="answerable cases with source page labels"),
    "citation_source_validity": _metric("Citation source validity", "Fraction of cited source numbers corresponding to actual generator sources. Structural validity does not establish semantic support."),
    "abstention_accuracy": _metric("Correct abstention", "Judge classification of whether an unanswerable response explicitly abstains without inventing an answer.", method="structured judge", applicability="explicitly labelled unanswerable cases"),
    "false_answer_rate": _metric("False-answer rate", "One minus correct abstention, on explicitly labelled unanswerable cases only.", method="derived from abstention judge", direction="lower", applicability="explicitly labelled unanswerable cases"),
    "failure_rate": _metric("Failed or partial attempt rate", "Failed or partial samples divided by attempted samples.", direction="lower", applicability="attempted samples"),
    "error_rate": _metric("Error rate", "Failed samples divided by attempted samples.", direction="lower", applicability="attempted samples"),
    "partial_rate": _metric("Partial-result rate", "Partial samples divided by attempted samples.", direction="lower", applicability="attempted samples"),
}
for _stage in ("retrieval_seconds", "local_retrieval_seconds", "reranking_seconds", "generation_seconds", "judge_seconds", "sample_seconds"):
    METRIC_CATALOG[_stage] = {"label": _stage.replace("_", " ").capitalize(), "description": "Measured duration in seconds.",
                              "method": "wall clock", "range": [0, None], "direction": "lower", "applicability": "samples with recorded duration"}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * quantile
    lo, hi = math.floor(idx), math.ceil(idx)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def _bootstrap_interval(values):
    if len(values) < 2:
        return None, None
    rng = random.Random(0)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(500))
    return means[12], means[487]


def _rank_diagnostics(retrieved_ids, grades, *, k):
    relevant = {cid for cid, g in grades.items() if g > 0}
    if not relevant:
        return {}
    ranked = list(dict.fromkeys(retrieved_ids))[:k]
    hits, p_sum = 0, 0.0
    for rank, cid in enumerate(ranked, 1):
        if cid in relevant:
            hits += 1
            p_sum += hits / rank
    return {"hit_rate_at_k": float(hits > 0), "map_at_k": p_sum / min(k, len(relevant))}


def _citation_diagnostics(response, documents, reference_pages=(), *, reference_page_groups=()):
    supplied_pages = set().union(*(document_pages(d) for d in documents))
    cited_pages = []
    for match in re.finditer(r"\b(?:pages?|pp?\.)\s*[:#]?\s*(\d+(?:\s*(?:,|and|[-–])\s*\d+)*)", response, flags=re.I):
        for seg in re.split(r"\s*(?:,|and)\s*", match.group(1)):
            endpoints = [p for p in re.split(r"\s*[-–]\s*", seg.strip()) if p.isdigit()]
            if not endpoints:
                continue
            first = int(endpoints[0])
            if len(endpoints) == 2 and first <= int(endpoints[1]) <= first + 500:
                cited_pages.extend(range(first, int(endpoints[1]) + 1))
            else:
                cited_pages.extend(int(p) for p in endpoints)
    cited_pages = set(cited_pages)
    cited_sources = {int(v) for v in re.findall(r"\bsource\s+(\d+)", response, flags=re.I)}
    values = {}
    if supplied_pages:
        values["citation_page_validity"] = len(cited_pages & supplied_pages) / len(cited_pages) if cited_pages else 0.0
    if reference_pages:
        values["citation_page_coverage"] = (
            sum(bool(cited_pages & supplied_pages & g) for g in reference_page_groups) / len(reference_page_groups)
            if reference_page_groups else len(cited_pages & supplied_pages & set(reference_pages)) / len(set(reference_pages))
        )
    if documents:
        values["citation_source_validity"] = sum(1 <= r <= len(documents) for r in cited_sources) / len(cited_sources) if cited_sources else 0.0
    return values


def _selection(cases, profile, strategies=None):
    if profile not in DEFAULT_STRATEGIES:
        raise ValueError("Evaluation profile must be preview, full, or retrieval.")
    strategies = tuple(dict.fromkeys(strategies or DEFAULT_STRATEGIES[profile]))
    if not strategies or set(strategies) - {"Keyword", "Vector", "Hybrid", "Production", "Reranking"}:
        raise ValueError("Invalid strategy selected.")
    if "Production" in strategies and "Reranking" in strategies:
        raise ValueError("Select only one of Production or Reranking to avoid duplicate samples.")
    if profile == "retrieval" and ({"Production", "Reranking"} & set(strategies)):
        raise ValueError("Retrieval profile excludes external reranker; use Keyword, Vector, or Hybrid.")
    selected = list(cases)
    if profile == "preview":
        limit = int(os.getenv("EVALUATION_PREVIEW_QUESTIONS", "5"))
        selected = sorted(cases, key=lambda c: _hash([c.id, c.question]))[:limit]
        unans = next((c for c in cases if not c.answerable), None)
        if unans and selected and all(c.answerable for c in selected):
            selected[-1] = unans
    return selected, strategies


def summarize_records(records):
    import pandas as pd

    metric_names = sorted({n for r in records for n in r.get("metrics", {})})
    strategies = list(dict.fromkeys(r["strategy"] for r in records))
    matched_ids = {}
    for name in metric_names:
        per_strategy = [
            {r["question_id"] for r in records if r["strategy"] == s and r.get("question_id")
             and r.get("metrics", {}).get(name, {}).get("status") == "completed"}
            for s in strategies
        ]
        matched_ids[name] = set.intersection(*per_strategy) if per_strategy else set()

    summary = []
    for strategy in strategies:
        rows = [r for r in records if r["strategy"] == strategy]
        attempted = [r for r in rows if r["status"] in {"completed", "partial", "failed"}]
        row = {
            "strategy": strategy, "samples": len(rows), "attempted": len(attempted),
            "completed": sum(r["status"] == "completed" for r in rows),
            "failed": sum(r["status"] in {"failed", "partial"} for r in rows),
            "skipped": sum(r["status"] in {"skipped", "cancelled"} for r in rows),
            "failure_rate": sum(r["status"] in {"partial", "failed"} for r in attempted) / len(attempted) if attempted else None,
            "error_rate": sum(r["status"] == "failed" for r in attempted) / len(attempted) if attempted else None,
            "partial_rate": sum(r["status"] == "partial" for r in attempted) / len(attempted) if attempted else None,
            "reranked_samples": sum(r.get("retrieval_trace", {}).get("reranker", "none") != "none" for r in rows),
            "degraded_samples": sum(bool(r.get("degraded")) for r in rows),
            "generation_reused_samples": sum(r.get("generation_mode") in {"cache", "shared"} or r.get("generation_cache_hit") is True for r in rows),
            "judge_reused_samples": sum(r.get("judge_reused") is True for r in rows),
        }
        for stage in ("retrieval_seconds", "local_retrieval_seconds", "reranking_seconds", "generation_seconds", "judge_seconds", "sample_seconds"):
            vals = [float(r[stage]) for r in rows if isinstance(r.get(stage), (int, float)) and not isinstance(r[stage], bool)
                    and math.isfinite(r[stage]) and r[stage] >= 0 and r.get("status") != "timeout" and not r.get("timeout")]
            row[stage] = statistics.mean(vals) if vals else None
            row[stage + "_count"] = len(vals)
            row[stage + "_p50"] = _percentile(vals, .5)
            row[stage + "_p95"] = _percentile(vals, .95)
        for stage, mode in (("generation_seconds", "generation_mode"), ("judge_seconds", "judge_reused")):
            fresh = [float(r[stage]) for r in rows if isinstance(r.get(stage), (int, float)) and r[stage] >= 0
                     and ((r.get(mode) == "fresh" and r.get("status") in {"completed", "partial"})
                          if mode == "generation_mode" else (r.get(mode) is False and r.get("status") == "completed"))]
            row[stage + "_fresh_count"] = len(fresh)
            row[stage + "_fresh_p50"] = _percentile(fresh, .5)
            row[stage + "_fresh_p95"] = _percentile(fresh, .95)
        for name in metric_names:
            entries = [r.get("metrics", {}).get(name, {"status": "not_applicable"}) for r in rows]
            vals, failed = [], 0
            for e in entries:
                if e.get("status") == "completed":
                    try:
                        vals.append(validate_score(name, e.get("value")))
                    except ValueError:
                        failed += 1
                elif e.get("status") == "failed":
                    failed += 1
            lo, hi = _bootstrap_interval(vals)
            row.update({
                name: statistics.mean(vals) if vals else None,
                f"{name}_count": len(vals), f"{name}_failed": failed,
                f"{name}_skipped": len(entries) - len(vals) - failed,
                f"{name}_lower95": lo, f"{name}_upper95": hi,
            })
            matched = []
            for r in rows:
                if r.get("question_id") in matched_ids[name]:
                    try:
                        matched.append(validate_score(name, r["metrics"][name]["value"]))
                    except (KeyError, ValueError):
                        pass
            mlo, mhi = _bootstrap_interval(matched)
            row.update({f"{name}_matched": statistics.mean(matched) if matched else None,
                        f"{name}_matched_count": len(matched), f"{name}_matched_lower95": mlo, f"{name}_matched_upper95": mhi})
        summary.append(row)

    frame = pd.DataFrame(summary)
    frame.attrs["matched_question_ids"] = {n: sorted(ids) for n, ids in matched_ids.items()}
    return frame


def _reranker_configuration(strategies):
    if not ({"Production", "Reranking"} & set(strategies)):
        return {"provider": "none", "model": None, "timeout_seconds": None, "credentials_available": None}
    provider = os.getenv("RERANKER_PROVIDER", "cohere" if os.getenv("COHERE_API_KEY") else "none")
    return {
        "provider": provider,
        "model": os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0") if provider == "cohere" else None,
        "timeout_seconds": int(os.getenv("RERANK_TIMEOUT_SECONDS", "15")) if provider == "cohere" else None,
        "credentials_available": bool(os.getenv("COHERE_API_KEY")) if provider == "cohere" else None,
    }


async def run_evaluation(
    db_path=None, *, dataset=None, profile="preview", strategies=None,
    progress_callback=None, cancellation_event=None, output_dir=PROJECT_ROOT / "evaluation_runs",
    force=False, expected_document_hash=None,
):
    from dotenv import load_dotenv
    from backend.generation_runtime import generate_final_answer, model_name_for, openai_base_url, prepare_generation_context
    from backend.rag_pipeline import get_embedding_model, load_vector_store
    from backend.retrieval_runtime import get_retrievers, retrieve_documents

    load_dotenv()
    dataset_provenance = dataset.get("provenance") if isinstance(dataset, dict) else None
    dataset_warnings = [str(w) for w in (dataset.get("warnings", []) if isinstance(dataset, dict) else [])]
    cases = load_evaluation_dataset(dataset, expected_document_hash=expected_document_hash)
    selected, strategies = _selection(cases, profile, strategies)
    total, done = len(selected) * len(strategies), 0
    cancelled = lambda: bool(cancellation_event and cancellation_event.is_set())
    emit = lambda msg: progress_callback(done, total, msg) if progress_callback else None
    if cancelled():
        raise asyncio.CancelledError("Evaluation cancelled.")

    concurrency = int(os.getenv("EVALUATION_CONCURRENCY", "4"))
    emit("Loading prepared vector index")
    target = Path(db_path).resolve() if db_path else sorted(p for p in (PROJECT_ROOT / "db").glob("chroma_*") if p.is_dir())[0]
    db = await asyncio.to_thread(load_vector_store, str(target))
    if db is None:
        raise EvaluationDataError("Prepared vector index not found.")
    index_manifest = read_json(target / "index_manifest.json") or {}
    if index_manifest.get("document_hash") and cases[0].document_hash != index_manifest["document_hash"]:
        raise EvaluationDataError("Dataset document_hash does not match the vector index.")

    retrieval_k = int(os.getenv("EVALUATION_K", "5")) if profile == "retrieval" else int(os.getenv("RETRIEVAL_K", "10"))
    final_k = retrieval_k if profile == "retrieval" else int(os.getenv("RETRIEVAL_TOP_N", "3"))
    vector, keyword, hybrid, all_docs = await asyncio.to_thread(get_retrievers, db, k=retrieval_k)
    relevance = {case.id: resolve_relevance(case, all_docs) for case in cases}
    corpus_version = index_manifest.get("corpus_version") or _hash(sorted(document_id(d) for d in all_docs))
    generator_provider = os.getenv("GENERATION_PROVIDER", "google")
    generator_model = model_name_for(generator_provider)
    judge_settings = JudgeSettings.from_env() if profile != "retrieval" else None
    metric_timeout = int(os.getenv("EVALUATION_METRIC_TIMEOUT", "360")) if judge_settings else None

    config = {
        "index_path": str(target), "document_hash": cases[0].document_hash, "corpus_version": corpus_version,
        "index_configuration": {k: index_manifest.get(k) for k in ("embedding_model", "embedding_revision", "pipeline_version", "chunk_config_version")},
        "profile": profile, "strategies": list(strategies), "corpus_type": index_manifest.get("corpus_type", "single_pdf"),
        "source_documents": index_manifest.get("source_documents", []),
        "dataset": [c.to_dict() for c in cases], "selected_ids": [c.id for c in selected],
        "dataset_provenance": dataset_provenance,
        "generator_provider": generator_provider if profile != "retrieval" else None,
        "generator_model": generator_model if profile != "retrieval" else None,
        "generator_base_url": openai_base_url() if profile != "retrieval" and generator_provider == "openai" else None,
        "judge": asdict(judge_settings) if judge_settings else None,
        "judge_base_url": openai_base_url() if judge_settings and judge_settings.provider == "openai" else None,
        "judge_rubric_version": JUDGE_RUBRIC_VERSION if judge_settings else None,
        "reranker": _reranker_configuration(strategies),
        "retrieval_k": retrieval_k, "metric_k": final_k,
    }
    run_key = _hash(config)
    directory = Path(output_dir).resolve() / run_key
    manifest_path = directory / "manifest.json"
    old_manifest = read_json(manifest_path)
    if old_manifest and old_manifest.get("status") == "completed" and not force:
        samples_file = directory / "samples.jsonl"
        if samples_file.is_file():
            records = [json.loads(line) for line in samples_file.read_text(encoding="utf-8").splitlines() if line]
            if len(records) == total and all(r.get("status") == "completed" for r in records):
                done = total
                emit("Reusing completed evaluation run")
                summary = summarize_records(records)
                summary.attrs.update({**old_manifest, "manifest_path": str(manifest_path), "sample_path": str(samples_file),
                                      "summary_path": str(directory / "summary.csv"), "cache_reused": True})
                return summary

    with _run_lock_guard:
        lock = _run_locks.setdefault(run_key, threading.Lock())
    if not lock.acquire(blocking=False):
        raise RuntimeError("This exact evaluation run is already in progress.")

    started = time.perf_counter()
    scope_label = ("deterministic subset of generated cases" if dataset_provenance else "deterministic preview subset") if profile == "preview" else ("all generated cases" if dataset_provenance else "all imported cases")
    manifest = {
        "schema_version": 2, "run_id": run_key[:16], "run_key": run_key, "status": "running",
        "started_at": _utc(), "profile": profile, "scope": scope_label, "total_questions": len(cases),
        "sampled_questions": len(selected), "samples": total, "strategies": list(strategies),
        "document_hash": cases[0].document_hash, "corpus_version": corpus_version, "configuration": config,
        "warnings": list(dataset_warnings), "metric_catalog": METRIC_CATALOG, "dataset_provenance": dataset_provenance,
    }
    if "Reranking" in strategies and config["reranker"]["provider"] == "none":
        manifest["warnings"].append("Reranking is disabled; Hybrid evidence was used. No reranker quality comparison was performed.")
    if profile != "retrieval" and ((generator_provider == "openai" and generator_model in {"auto", "auto:fast"}) or (judge_settings and judge_settings.provider == "openai" and judge_settings.model in {"auto", "auto:fast"})):
        manifest["warnings"].append("Automatic model routing can change the underlying model between requests.")
    if judge_settings and generator_provider == judge_settings.provider and generator_model == judge_settings.model:
        manifest["warnings"].append("Generator and judge use the same model; an independent judge model is recommended.")
    if profile != "retrieval" and len(selected) < 10:
        manifest["warnings"].append(f"Only {len(selected)} unique questions were evaluated. Treat these scores as exploratory.")

    cache_dir = Path(output_dir).resolve() / "cache"
    records, shared_answers, shared_metrics = [], {}, {}
    sample_sem, call_sem = asyncio.Semaphore(concurrency), asyncio.Semaphore(concurrency)
    judge, cache_hits, provider_limit = None, 0, None
    work_counts = {"generation_operations": 0, "judge_metric_operations": 0}

    async def answer_for(case, docs, context, record):
        nonlocal cache_hits
        ans_key = _hash({"model": generator_model, "base_url": config["generator_base_url"], "q": case.question, "texts": context["text_contexts"], "imgs": [_hash(p) for p in context.get("images_base64", [])]})
        path = cache_dir / "answers" / f"{ans_key}.json"
        cached = read_json(path) if not force else None
        if cached and cached.get("answer", "").strip():
            cache_hits += 1
            record["generation_mode"] = "cache"
            return cached["answer"], True
        if ans_key not in shared_answers:
            record["generation_mode"] = "fresh"
            async def _gen():
                async with call_sem:
                    if cancelled():
                        raise asyncio.CancelledError()
                    work_counts["generation_operations"] += 1
                    ans = await asyncio.to_thread(generate_final_answer, docs, case.question, provider=generator_provider, model_name=generator_model)
                    write_json(path, {"answer": ans, "created_at": _utc()})
                    return ans
            shared_answers[ans_key] = asyncio.create_task(_gen())
            reused = False
        else:
            record["generation_mode"] = "shared"
            reused = True
            cache_hits += 1
        task = shared_answers[ans_key]
        try:
            return await asyncio.shield(task), reused
        except BaseException:
            if shared_answers.get(ans_key) is task:
                shared_answers.pop(ans_key, None)
            raise

    async def metric_for(name, case, response, context, image_paths):
        nonlocal cache_hits
        m_key = _hash({"index": config["index_configuration"], "judge": config["judge"], "base_url": config["judge_base_url"],
                       "name": name, "q": case.question, "r": response, "ref": case.reference_answer, "texts": context["text_contexts"]})
        path = cache_dir / "metrics" / f"{m_key}.json"
        cached = read_json(path) if not force else None
        if cached and "value" in cached:
            cache_hits += 1
            return {"status": "completed", "value": validate_score(name, cached["value"]), "reason": cached.get("reason"), "cache_hit": True, "provider_requests": 0, "provider_cache_hits": 0}
        if m_key not in shared_metrics:
            async def _eval():
                async with call_sem:
                    if cancelled():
                        return {"status": "skipped", "reason": "cancelled"}
                    work_counts["judge_metric_operations"] += 1
                    async with asyncio.timeout(metric_timeout):
                        res = await judge.score(name, question=case.question, response=response, reference=case.reference_answer, contexts=context["text_contexts"], image_contexts=image_paths)
                    val = validate_score(name, res["value"])
                    write_json(path, {"value": val, "reason": res.get("reason"), "created_at": _utc()})
                    return {"status": "completed", "value": val, "reason": res.get("reason"), "cache_hit": False,
                            **{k: res[k] for k in ("provider_requests", "provider_cache_hits") if k in res}}
            shared_metrics[m_key] = asyncio.create_task(_eval())
            reused = False
        else:
            cache_hits += 1
            reused = True
        res = await asyncio.shield(shared_metrics[m_key])
        return {**res, "cache_hit": reused}

    async def process(case, strategy):
        nonlocal done, provider_limit
        record = {"run_id": manifest["run_id"], "question_id": case.id, "question": case.question, "reference_answer": case.reference_answer,
                  "answerable": case.answerable, "document_hash": case.document_hash, "strategy": strategy, "status": "running", "metrics": {}, "retrieval_k": final_k}
        for name in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "hit_rate_at_k", "map_at_k"):
            record["metrics"][name] = {"status": "skipped" if case.answerable else "not_applicable", "reason": "Retrieval did not complete."}
        if profile != "retrieval":
            for name in PROFILE_METRICS[profile]:
                record["metrics"][name] = {"status": "skipped" if case.answerable else "not_applicable", "reason": "Answer generation did not complete."}
            if not case.answerable:
                record["metrics"]["abstention_accuracy"] = {"status": "skipped", "reason": "Answer generation did not complete."}
        async with sample_sem:
            t0 = time.perf_counter()
            try:
                if cancelled():
                    record["status"] = "cancelled"
                    return record
                emit(f"{strategy} / {case.id}: retrieving evidence")
                t_ret = time.perf_counter()
                try:
                    if strategy in {"Production", "Reranking"}:
                        docs, trace = await asyncio.to_thread(retrieve_documents, db, case.question, k=retrieval_k, top_n=final_k)
                        record["retrieval_trace"] = trace
                        record["local_retrieval_seconds"] = round(trace.get("retrieval_seconds", 0), 4)
                        record["reranking_seconds"] = round(trace.get("reranking_seconds", 0), 4)
                        record["effective_strategy"] = "Reranking" if trace.get("reranker", "none") != "none" else "Hybrid"
                        if trace.get("reranking_error") or (strategy == "Reranking" and trace.get("reranker", "none") == "none"):
                            record["degraded"] = True
                    else:
                        retriever = {"Keyword": keyword, "Vector": vector, "Hybrid": hybrid}[strategy]
                        docs = (await asyncio.to_thread(retriever.invoke, case.question))[:final_k]
                finally:
                    record["retrieval_seconds"] = round(time.perf_counter() - t_ret, 4)
                    record.setdefault("local_retrieval_seconds", record["retrieval_seconds"])
                    record.setdefault("reranking_seconds", 0.0)

                context = prepare_generation_context(docs, case.question, provider=generator_provider) if profile != "retrieval" else {
                    "chunk_ids": [document_id(d) for d in docs], "text_contexts": [d.page_content for d in docs], "images_base64": [],
                }
                record["retrieved_chunk_ids"] = context["chunk_ids"]
                record["contexts"] = context["text_contexts"]
                record["image_contexts"] = []
                for payload in context.get("images_base64", []):
                    img_path = directory / "images" / (_hash(payload) + ".jpg")
                    if not img_path.is_file():
                        img_path.parent.mkdir(parents=True, exist_ok=True)
                        img_path.write_bytes(base64.b64decode(payload, validate=True))
                    record["image_contexts"].append(str(img_path))

                for name, val in {**retrieval_metrics(record["retrieved_chunk_ids"], relevance[case.id], k=final_k),
                                  **_rank_diagnostics(record["retrieved_chunk_ids"], relevance[case.id], k=final_k)}.items():
                    record["metrics"][name] = {"status": "completed", "value": validate_score(name, val)}
                if profile == "retrieval":
                    record["status"] = "completed"
                    return record
                if cancelled():
                    record["status"] = "cancelled"
                    return record

                emit(f"{strategy} / {case.id}: generating answer")
                t_gen = time.perf_counter()
                try:
                    response, reused = await answer_for(case, docs, context, record)
                finally:
                    record["generation_seconds"] = round(time.perf_counter() - t_gen, 4)
                record["generation_cache_hit"] = reused
                record["response"] = response

                if case.answerable:
                    for name, val in lexical_reference_metrics(response, case.reference_answer).items():
                        record["metrics"][name] = {"status": "completed", "value": val}
                    ref_pages = set(case.reference_pages) or set().union(*(document_pages(d) for d in all_docs if relevance[case.id].get(document_id(d), 0) > 0))
                    supplied = [d for d in docs if document_id(d) in set(context["chunk_ids"])]
                    pg_groups = [set().union(*(document_pages(d) for d in all_docs if document_id(d) in g)) for g in case.relevant_chunk_groups]
                    for name, val in _citation_diagnostics(response, supplied, ref_pages, reference_page_groups=pg_groups).items():
                        record["metrics"][name] = {"status": "completed", "value": validate_score(name, val)}

                names = list(PROFILE_METRICS[profile]) if case.answerable else ["abstention_accuracy"]
                if context.get("images_base64") and case.answerable:
                    names = ["multimodal_faithfulness" if n == "faithfulness" else n for n in names if n not in {"context_precision", "context_recall"}]

                emit(f"{strategy} / {case.id}: judging {len(names)} metrics")
                t_judge = time.perf_counter()
                async def _score(name):
                    nonlocal provider_limit
                    try:
                        record["metrics"][name] = await metric_for(name, case, response, context, record["image_contexts"])
                    except Exception as exc:
                        code = provider_status_code(exc)
                        if code == 429:
                            provider_limit = {"status_code": 429, "provider": judge_settings.provider, "model": judge_settings.model, "reason": limit_message()}
                        record["metrics"][name] = {"status": "skipped" if isinstance(exc, JudgeQuotaExceeded) else "failed", "error": f"{type(exc).__name__}: {exc}", **({"provider_status_code": 429} if code == 429 else {})}
                await asyncio.gather(*(_score(n) for n in names))
                record["judge_seconds"] = round(time.perf_counter() - t_judge, 4)
                record["judge_reused"] = any(record["metrics"][n].get("cache_hit") for n in names)
                if not case.answerable:
                    ab = record["metrics"].get("abstention_accuracy", {})
                    record["metrics"]["false_answer_rate"] = {**ab, **({"value": 1 - ab["value"]} if ab.get("status") == "completed" else {})}
                record["status"] = "cancelled" if cancelled() or any(m.get("reason") == "cancelled" for m in record["metrics"].values()) else "partial" if any(m.get("status") == "failed" for m in record["metrics"].values()) else "completed"
                return record
            except asyncio.CancelledError:
                record["status"] = "cancelled"
                return record
            except Exception as exc:
                record["status"] = "skipped" if provider_status_code(exc) == 429 else "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                return record
            finally:
                record["sample_seconds"] = round(time.perf_counter() - t0, 4)
                records.append(record)
                write_json(directory / "cases" / f"{_hash([case.id, strategy])}.json", record)
                done += 1
                emit(f"{done}/{total} samples finished; {strategy} / {case.id}: {record['status']}")

    try:
        write_json(manifest_path, manifest)
        if profile != "retrieval":
            judge = RagasJudge(judge_settings, embedding_model=get_embedding_model())
            judge.prepare()
        scheduled = [(selected[(r + s) % len(selected)], strategy) for r in range(len(selected)) for s, strategy in enumerate(strategies)]
        await asyncio.gather(*(process(c, s) for c, s in scheduled))

        order = {(c.id, s): i for i, (c, s) in enumerate((c, s) for c in selected for s in strategies)}
        records.sort(key=lambda r: order[(r["question_id"], r["strategy"])])
        sample_path = directory / "samples.jsonl"
        sample_path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records), encoding="utf-8")

        failed = sum(r["status"] in {"failed", "partial"} for r in records)
        skipped = sum(r["status"] in {"skipped", "cancelled"} for r in records)
        status = "cancelled" if cancelled() else "partial" if (failed or skipped or provider_limit) and any(r["status"] in {"completed", "partial"} for r in records) else "failed" if failed or provider_limit else "completed"
        manifest.update({
            "status": status, "finished_at": _utc(), "seconds": round(time.perf_counter() - started, 3),
            "failed_samples": failed, "skipped_samples": skipped, "cache_hits": cache_hits,
            "completed_samples": sum(r["status"] == "completed" for r in records),
            "partial_samples": sum(r["status"] == "partial" for r in records), "work_counts": work_counts,
        })
        if judge and hasattr(getattr(judge, "llm", None), "usage"):
            manifest["judge_usage"] = judge.llm.usage
        if provider_limit:
            manifest["provider_limit"] = provider_limit
        summary = summarize_records(records)
        manifest["matched_question_ids"] = summary.attrs["matched_question_ids"]
        summary.to_csv(directory / "summary.csv", index=False)
        write_json(manifest_path, manifest)
        summary.attrs.update({**manifest, "manifest_path": str(manifest_path.resolve()), "sample_path": str(sample_path.resolve()),
                              "summary_path": str((directory / "summary.csv").resolve()), "cache_reused": False})
        emit(f"Evaluation {status}: {len(records)} samples")
        return summary
    finally:
        for t in list(shared_answers.values()) + list(shared_metrics.values()):
            if not t.done():
                t.cancel()
        if judge:
            await judge.close()
        lock.release()
