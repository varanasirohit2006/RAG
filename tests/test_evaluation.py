import asyncio
import json
import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from backend import advanced_evaluation as evaluation, generation_runtime
from backend.evaluation_data import (
    EvaluationDataError, lexical_reference_metrics, load_evaluation_dataset,
    resolve_relevance, retrieval_metrics, validate_score,
)

HASH = "a" * 64


def row(**kwargs):
    return {"id": "q1", "question": "What is the invoice total?", "reference_answer": "123.45 USD",
            "document_hash": HASH, "answerable": True, "reference_pages": [1], **kwargs}


@pytest.mark.parametrize("changes", [
    {"reference_answer": ""}, {"document_hash": "wrong"}, {"reference_pages": []},
    {"answerable": "yes"}, {"reference_pages": [0]}, {"relevance_grades": {"c1": float("nan")}},
])
def test_invalid_dataset_is_rejected_before_model_work(changes):
    with pytest.raises(EvaluationDataError):
        load_evaluation_dataset([row(**changes)])


def test_dataset_hash_and_duplicate_questions_are_validated():
    with pytest.raises(EvaluationDataError, match="another PDF"):
        load_evaluation_dataset([row()], expected_document_hash="b" * 64)
    with pytest.raises(EvaluationDataError, match="unique"):
        load_evaluation_dataset([row(), row()])
    with pytest.raises(EvaluationDataError, match="reviewed"):
        load_evaluation_dataset(None)


def test_csv_and_json_keep_explicit_unanswerable_labels():
    csv = f'id,question,reference_answer,document_hash,answerable,reference_pages\nq1,Who is CEO?,Source does not say.,{HASH},false,\n'
    case = load_evaluation_dataset(csv, format="csv")[0]
    assert case.answerable is False
    encoded = json.dumps({"document_hash": HASH, "questions": [row()]})
    assert load_evaluation_dataset(encoded)[0].document_hash == HASH


def test_alias_labels_count_one_evidence_group_and_preserve_pages():
    documents = [Document(page_content="Invoice total 123.45 USD", metadata={
        "chunk_id": "primary", "duplicate_chunk_ids": '["copy1", "copy2"]', "pages": "[1, 2]",
    })]
    case = load_evaluation_dataset([row(relevant_chunk_ids=["copy1", "copy2"], reference_pages=[])])[0]
    grades = resolve_relevance(case, documents)
    assert grades == {"primary": 1.0}
    metrics = retrieval_metrics(["primary"], grades, k=1)
    assert metrics["recall_at_k"] == 1
    assert metrics["precision_at_k"] == 1


def test_explicit_chunk_labels_take_precedence_over_page_references():
    documents = [Document(page_content=chunk_id, metadata={"chunk_id": chunk_id, "pages": "[1]"})
                 for chunk_id in ("c1", "c2", "noise")]
    case = load_evaluation_dataset([row(relevant_chunk_ids=["c1"])])[0]
    grades = resolve_relevance(case, documents)
    assert grades == {"c1": 1.0}
    assert retrieval_metrics(["noise", "c1"], grades, k=2)["precision_at_k"] == 0.5
    graded = load_evaluation_dataset([row(relevance_grades={"c1": 3, "c2": 0})])[0]
    assert resolve_relevance(graded, documents) == {"c1": 3.0, "c2": 0.0}
    page_only = load_evaluation_dataset([row()])[0]
    assert resolve_relevance(page_only, documents) == {"c1": 1.0, "c2": 1.0, "noise": 1.0}


def test_reranker_provenance_uses_effective_runtime_defaults_without_secrets(monkeypatch):
    for name in ("RERANKER_PROVIDER", "COHERE_RERANK_MODEL", "RERANK_TIMEOUT_SECONDS", "COHERE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert evaluation._reranker_configuration(["Production"])["provider"] == "none"
    monkeypatch.setenv("COHERE_API_KEY", "private-test-credential")
    settings = evaluation._reranker_configuration(["Production"])
    assert settings == {"provider": "cohere", "model": "rerank-english-v3.0", "timeout_seconds": 15,
                        "credentials_available": True}
    assert "private-test-credential" not in json.dumps(settings)
    monkeypatch.setenv("COHERE_RERANK_MODEL", "custom-reranker")
    monkeypatch.setenv("RERANK_TIMEOUT_SECONDS", "7")
    assert evaluation._reranker_configuration(["Production"])["model"] == "custom-reranker"
    assert evaluation._reranker_configuration(["Production"])["timeout_seconds"] == 7
    assert evaluation._reranker_configuration(["Hybrid"])["provider"] == "none"
    monkeypatch.setenv("RERANKER_PROVIDER", "none")
    assert evaluation._reranker_configuration(["Production"])["model"] is None


def test_rank_metrics_use_actual_rank_and_unique_ids():
    metrics = retrieval_metrics(["noise", "c1", "c1"], {"c1": 3.0, "c2": 2.0}, k=5)
    assert metrics["precision_at_k"] == pytest.approx(1 / 5)
    assert metrics["recall_at_k"] == pytest.approx(1 / 2)
    assert metrics["mrr"] == pytest.approx(1 / 2)
    expected = (7 / math.log2(3)) / (7 + 3 / math.log2(3))
    assert metrics["ndcg_at_k"] == pytest.approx(expected)
    assert retrieval_metrics(["anything"], {}, k=5) == {}


def test_lexical_diagnostics_preserve_numeric_mismatches():
    perfect = lexical_reference_metrics("123.45 USD", "123.45 USD")
    wrong = lexical_reference_metrics("128.45 USD", "123.45 USD")
    assert perfect == {"normalized_exact_match": 1, "token_f1": 1}
    assert wrong["normalized_exact_match"] == 0
    assert wrong["token_f1"] == 0.5


@pytest.mark.parametrize("value", [5.001232, float("nan"), float("inf"), True])
def test_invalid_scores_never_enter_summary(value):
    with pytest.raises(ValueError):
        validate_score("faithfulness", value)
    records = [{"strategy": "Production", "status": "partial", "metrics": {
        "faithfulness": {"status": "completed", "value": value},
        "factual_correctness": {"status": "completed", "value": 0.8},
    }}]
    summary = evaluation.summarize_records(records).iloc[0]
    assert summary["faithfulness_count"] == 0
    assert summary["faithfulness_failed"] == 1
    assert summary["factual_correctness"] == 0.8
    assert summary["failed"] == 1


def test_preview_declares_subset_and_keeps_refusal_case(monkeypatch):
    monkeypatch.setenv("EVALUATION_PREVIEW_QUESTIONS", "2")
    data = [row(id=f"q{i}") for i in range(5)] + [row(id="refusal", answerable=False, reference_pages=[])]
    selected, strategies = evaluation._selection(load_evaluation_dataset(data), "preview")
    assert len(selected) == 2
    assert any(not case.answerable for case in selected)
    assert strategies == ("Production",)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    import dotenv
    from backend import rag_pipeline
    from backend import retrieval_runtime

    monkeypatch.setattr(dotenv, "load_dotenv", lambda: False)
    monkeypatch.setenv("EVALUATION_CONCURRENCY", "2")
    monkeypatch.setenv("EVALUATION_PREVIEW_QUESTIONS", "5")
    monkeypatch.setenv("GENERATION_PROVIDER", "google")
    monkeypatch.setenv("GENERATION_MODEL", "test-generator")
    monkeypatch.setenv("EVALUATION_MODEL", "test-judge")
    monkeypatch.setenv("EVALUATION_PROVIDER", "google")
    monkeypatch.setenv("RETRIEVAL_K", "5")
    monkeypatch.setenv("RETRIEVAL_TOP_N", "3")
    docs = [Document(page_content="Invoice total is 123.45 USD", metadata={"chunk_id": "c1", "pages": "[1]"})]
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "index_manifest.json").write_text(json.dumps({"document_hash": HASH, "corpus_version": "corpus-v1"}))
    state = {"generation_calls": 0, "judge_calls": 0, "prepared": 0, "active": 0, "maximum_active": 0, "invalid": False}
    lock = threading.Lock()
    class Retriever:
        def invoke(self, question):
            return docs
    retriever = Retriever()
    monkeypatch.setattr(rag_pipeline, "load_vector_store", lambda path: object())
    monkeypatch.setattr(rag_pipeline, "get_embedding_model", lambda: object())
    monkeypatch.setattr(retrieval_runtime, "get_retrievers", lambda db, k=10: (retriever, retriever, retriever, docs))
    monkeypatch.setattr(retrieval_runtime, "retrieve_documents", lambda db, question, **kwargs: (docs, {"reranker": "none"}))
    def prepare(docs, question, provider=None):
        return {"prompt": question + docs[0].page_content, "text_contexts": [docs[0].page_content],
                "chunk_ids": ["c1"], "image_paths": [], "images_base64": []}
    monkeypatch.setattr(generation_runtime, "prepare_generation_context", prepare)
    def generate(docs, question, **kwargs):
        with lock:
            state["generation_calls"] += 1
            state["active"] += 1
            state["maximum_active"] = max(state["maximum_active"], state["active"])
        time.sleep(0.03)
        with lock:
            state["active"] -= 1
        return "123.45 USD"
    monkeypatch.setattr(generation_runtime, "generate_final_answer", generate)
    class Judge:
        def __init__(self, *args, **kwargs):
            pass
        def prepare(self):
            state["prepared"] += 1
        async def score(self, name, **kwargs):
            state["judge_calls"] += 1
            await asyncio.sleep(0.02)
            return {"value": 5.001232 if state["invalid"] and name == "faithfulness" else 0.8}
        async def close(self):
            pass
    monkeypatch.setattr(evaluation, "RagasJudge", Judge)
    return db_dir, tmp_path / "results", state


def test_parallel_work_reuses_identical_answers_and_metrics_and_cached_run(runtime):
    db_dir, output, state = runtime
    data = [row(id="q1"), row(id="q2", question="State the invoice amount.")]
    summary = asyncio.run(evaluation.run_evaluation(
        str(db_dir), dataset=data, profile="preview", strategies=["Vector", "Hybrid"], output_dir=output,
    ))
    assert summary.attrs["status"] == "completed"
    assert summary.attrs["samples"] == 4
    assert state["generation_calls"] == 2
    assert state["judge_calls"] == 10
    assert state["maximum_active"] == 2
    before = dict(state)
    cached = asyncio.run(evaluation.run_evaluation(
        str(db_dir), dataset=data, profile="preview", strategies=["Vector", "Hybrid"], output_dir=output,
    ))
    assert cached.attrs["cache_reused"] is True
    assert state == before
    records = [json.loads(line) for line in Path(summary.attrs["sample_path"]).read_text().splitlines()]
    assert len(records) == 4
    assert all(record["contexts"] == ["Invoice total is 123.45 USD"] for record in records)


def test_judge_failure_is_partial_with_denominators(runtime):
    db_dir, output, state = runtime
    state["invalid"] = True
    summary = asyncio.run(evaluation.run_evaluation(str(db_dir), dataset=[row()], output_dir=output))
    assert summary.attrs["status"] == "partial"
    assert summary.attrs["failed_samples"] == 1
    assert summary.iloc[0]["faithfulness_count"] == 0
    assert summary.iloc[0]["faithfulness_failed"] == 1
    assert summary.iloc[0]["factual_correctness_count"] == 1
    assert summary.iloc[0]["judge_seconds"] > 0


@pytest.mark.parametrize("stage", ["generator", "judge"])
def test_proxy_endpoint_changes_invalidate_run_and_stage_caches(runtime, monkeypatch, stage):
    db_dir, output, state = runtime
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:31415/v1")
    if stage == "generator":
        monkeypatch.setenv("GENERATION_PROVIDER", "openai")
        monkeypatch.setenv("GENERATION_MODEL", "auto:fast")
    else:
        monkeypatch.setenv("EVALUATION_PROVIDER", "openai")
        monkeypatch.setenv("EVALUATION_MODEL", "auto:fast")
    first = asyncio.run(evaluation.run_evaluation(str(db_dir), dataset=[row()], output_dir=output))
    before = dict(state)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:31416/v1")
    second = asyncio.run(evaluation.run_evaluation(str(db_dir), dataset=[row()], output_dir=output))
    assert first.attrs["run_key"] != second.attrs["run_key"]
    assert second.attrs["cache_reused"] is False
    assert second.attrs["configuration"][stage + "_base_url"] == "http://127.0.0.1:31416/v1"
    assert any("Automatic model routing" in warning for warning in second.attrs["warnings"])
    if stage == "generator":
        assert state["generation_calls"] > before["generation_calls"]
    else:
        assert state["judge_calls"] > before["judge_calls"]
        assert state["generation_calls"] == before["generation_calls"]


def test_reranker_changes_invalidate_the_completed_run_cache(runtime, monkeypatch):
    db_dir, output, state = runtime
    monkeypatch.delenv("RERANKER_PROVIDER", raising=False)
    monkeypatch.setenv("COHERE_API_KEY", "private-test-credential")
    monkeypatch.setenv("COHERE_RERANK_MODEL", "first-reranker")
    first = asyncio.run(evaluation.run_evaluation(str(db_dir), dataset=[row()], output_dir=output))
    monkeypatch.setenv("COHERE_RERANK_MODEL", "second-reranker")
    second = asyncio.run(evaluation.run_evaluation(str(db_dir), dataset=[row()], output_dir=output))
    assert first.attrs["run_key"] != second.attrs["run_key"]
    assert second.attrs["cache_reused"] is False
    monkeypatch.setenv("RERANK_TIMEOUT_SECONDS", "8")
    third = asyncio.run(evaluation.run_evaluation(str(db_dir), dataset=[row()], output_dir=output))
    assert second.attrs["run_key"] != third.attrs["run_key"]
    monkeypatch.delenv("COHERE_API_KEY")
    fourth = asyncio.run(evaluation.run_evaluation(str(db_dir), dataset=[row()], output_dir=output))
    assert fourth.attrs["configuration"]["reranker"]["provider"] == "none"
    assert third.attrs["run_key"] != fourth.attrs["run_key"]


def test_embedding_revision_invalidates_completed_runs_and_metric_cache(runtime):
    db_dir, output, state = runtime
    manifest_path = db_dir / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    configuration = {"embedding_model": "test-embedder", "embedding_revision": "revision-a",
                     "pipeline_version": "fast-v3", "chunk_config_version": "native-v3"}
    manifest.update(configuration)
    manifest_path.write_text(json.dumps(manifest))
    first = asyncio.run(evaluation.run_evaluation(str(db_dir), dataset=[row()], output_dir=output))
    assert first.attrs["configuration"]["index_configuration"] == configuration
    assert state["judge_calls"] == 5
    manifest["embedding_revision"] = "revision-b"
    manifest_path.write_text(json.dumps(manifest))
    second = asyncio.run(evaluation.run_evaluation(str(db_dir), dataset=[row()], output_dir=output))
    assert first.attrs["corpus_version"] == second.attrs["corpus_version"]
    assert first.attrs["run_key"] != second.attrs["run_key"]
    assert second.attrs["cache_reused"] is False
    assert state["judge_calls"] == 10
    # An identical delivered context can still reuse its completed model answer.
    assert state["generation_calls"] == 1


def test_failed_generation_keeps_duration_and_skips_quality_scores(runtime, monkeypatch):
    db_dir, output, state = runtime
    def unavailable(*args, **kwargs):
        time.sleep(0.015)
        raise RuntimeError("Provider unavailable (503)")
    monkeypatch.setattr(generation_runtime, "generate_final_answer", unavailable)
    summary = asyncio.run(evaluation.run_evaluation(str(db_dir), dataset=[row()], output_dir=output))
    assert summary.attrs["status"] == "failed"
    assert summary.iloc[0]["generation_seconds"] >= 0.015
    assert summary.iloc[0]["faithfulness_count"] == 0
    assert summary.iloc[0]["faithfulness_skipped"] == 1
    assert summary.attrs["work_counts"] == {"generation_operations": 1, "judge_metric_operations": 0}
    assert state["judge_calls"] == 0


def test_failed_retrieval_keeps_duration_and_starts_no_model_work(runtime, monkeypatch):
    from backend import retrieval_runtime
    db_dir, output, state = runtime
    retriever = retrieval_runtime.get_retrievers(None)[0]
    def unavailable(question):
        time.sleep(0.015)
        raise RuntimeError("Index unavailable")
    monkeypatch.setattr(retriever, "invoke", unavailable)
    summary = asyncio.run(evaluation.run_evaluation(str(db_dir), dataset=[row()], profile="retrieval", output_dir=output))
    assert summary.attrs["status"] == "failed"
    assert all(summary["retrieval_seconds"] >= 0.015)
    assert all(summary["local_retrieval_seconds"] >= 0.015)
    assert all(summary["reranking_seconds"] == 0)
    assert summary.attrs["work_counts"] == {"generation_operations": 0, "judge_metric_operations": 0}


def test_retrieval_profile_makes_no_generation_or_judge_calls(runtime):
    db_dir, output, state = runtime
    summary = asyncio.run(evaluation.run_evaluation(str(db_dir), dataset=[row()], profile="retrieval", output_dir=output))
    assert summary.attrs["status"] == "completed"
    assert state["generation_calls"] == 0
    assert state["judge_calls"] == 0
    assert state["prepared"] == 0
    assert "faithfulness" not in summary.columns
    assert summary["recall_at_k"].tolist() == [1, 1, 1]


def test_cancellation_stops_queued_generation_and_judging(runtime, monkeypatch):
    db_dir, output, state = runtime
    cancellation = threading.Event()
    monkeypatch.setenv("EVALUATION_CONCURRENCY", "1")
    def generate(*args, **kwargs):
        state["generation_calls"] += 1
        cancellation.set()
        return "123.45 USD"
    monkeypatch.setattr(generation_runtime, "generate_final_answer", generate)
    data = [row(id="q1"), row(id="q2", question="State the amount.")]
    summary = asyncio.run(evaluation.run_evaluation(
        str(db_dir), dataset=data, output_dir=output, cancellation_event=cancellation,
    ))
    assert summary.attrs["status"] == "cancelled"
    assert state["generation_calls"] == 1
    assert state["judge_calls"] == 0
    assert summary.attrs["skipped_samples"] == 2


def test_gemma_adapter_repairs_json_without_tools_or_multiple_candidates():
    from pydantic import BaseModel
    from backend.evaluation_judge import JudgeSettings, create_json_judge_llm
    from ragas.llms.base import InstructorBaseRagasLLM

    class Verdict(BaseModel):
        faithful: bool
        reason: str
    class Model:
        def __init__(self):
            self.calls = []
        async def ainvoke(self, messages):
            self.calls.append(messages)
            text = 'not JSON' if len(self.calls) == 1 else '```json\n{"faithful":true,"reason":"Source supports it."}\n```'
            return SimpleNamespace(content=[{"text": text}], response_metadata={"finish_reason": "STOP"})
    model = Model()
    adapter = create_json_judge_llm(JudgeSettings(model="gemma-4-31b-it"), model=model)
    assert isinstance(adapter, InstructorBaseRagasLLM)
    result = asyncio.run(adapter.agenerate("Judge the statement.", Verdict))
    assert result.faithful is True
    assert len(model.calls) == 2
    assert adapter.usage["json_repairs"] == 1
    assert "Required JSON schema" in model.calls[0][0].content[0]["text"]


def test_gemma_adapter_does_not_score_truncated_verdicts():
    from pydantic import BaseModel
    from backend.evaluation_judge import JudgeSettings, create_json_judge_llm

    class Verdict(BaseModel):
        faithful: bool
    class Model:
        async def ainvoke(self, messages):
            return SimpleNamespace(content='{"faithful":true}', response_metadata={"finish_reason": "MAX_TOKENS"})
    adapter = create_json_judge_llm(JudgeSettings(), model=Model())
    with pytest.raises(ValueError, match="truncated"):
        asyncio.run(adapter.agenerate("Judge the statement.", Verdict))
    assert adapter.usage["structured_generation_requests"] == 2
    assert adapter.usage['truncation_retries'] == 1


def test_scheduling_alternates_strategies_instead_of_using_keyword_first(runtime, monkeypatch):
    from backend import retrieval_runtime
    db, output, state = runtime
    monkeypatch.setenv('EVALUATION_CONCURRENCY', '1')
    order = []
    class Retriever:
        def __init__(self, strategy):
            self.strategy = strategy
        def invoke(self, question):
            order.append(self.strategy)
            return [Document(page_content=self.strategy + ' evidence', metadata={'chunk_id': 'c1', 'pages': [1]})]
    docs = Retriever('Keyword').invoke('setup')
    order.clear()
    monkeypatch.setattr(retrieval_runtime, 'get_retrievers', lambda db, k=10: (Retriever('Vector'), Retriever('Keyword'), Retriever('Hybrid'), docs))
    asyncio.run(evaluation.run_evaluation(str(db), dataset=[row(id='q1'), row(id='q2', question='Another question?')],
        profile='full', strategies=['Keyword', 'Vector', 'Hybrid'], output_dir=output))
    assert order == ['Keyword', 'Vector', 'Hybrid', 'Keyword', 'Vector', 'Hybrid']


def test_complete_metric_deadline_is_separate_from_each_request_deadline(runtime, monkeypatch):
    db, output, _ = runtime
    # Production settings remain valid; scale the request limit in the test only.
    settings = evaluation.JudgeSettings(provider='google', model='test-judge')
    object.__setattr__(settings, 'timeout', 0.02)
    monkeypatch.setattr(evaluation.JudgeSettings, 'from_env', lambda: settings)
    monkeypatch.setenv('EVALUATION_METRIC_TIMEOUT', '15')
    class MultiStepJudge:
        def __init__(self, *args, **kwargs):
            pass
        def prepare(self):
            pass
        async def score(self, name, **kwargs):
            for _ in range(4):
                await asyncio.sleep(0.01)  # Each step fits; their sum exceeds 0.02s.
            return {'value': 0.8}
        async def close(self):
            pass
    monkeypatch.setattr(evaluation, 'RagasJudge', MultiStepJudge)
    result = asyncio.run(evaluation.run_evaluation(str(db), dataset=[row(id='q1')],
        profile='full', strategies=['Vector'], output_dir=output))
    assert result.attrs['status'] == 'completed'
    assert result.iloc[0]['factual_correctness_count'] == 1
