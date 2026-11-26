"""Evaluation reporting must expose real denominators and structural metric limits."""

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from backend import advanced_evaluation as evaluation, generation_runtime
from backend.evaluation_data import load_evaluation_dataset, validate_score
from test_evaluation import runtime, row


def test_screenshot_metrics_are_in_both_judged_profiles():
    screenshot = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    for profile in ("preview", "full"):
        assert screenshot <= set(evaluation.PROFILE_METRICS[profile])
    assert evaluation.DEFAULT_STRATEGIES["full"] == ("Keyword", "Vector", "Hybrid", "Reranking")
    assert evaluation.METRIC_CATALOG["answer_relevancy"]["range"] == [-1, 1]
    assert evaluation.METRIC_CATALOG["sample_seconds"]["range"] == [0, None]
    with pytest.raises(ValueError, match="outside"):
        validate_score("answer_relevancy", 5.001232)


def test_average_precision_uses_unique_results_and_missing_relevance():
    values = evaluation._rank_diagnostics(["noise", "c1", "c1", "c2"], {"c1": 1, "c2": 1, "c3": 1}, k=3)
    assert values["hit_rate_at_k"] == 1
    assert values["map_at_k"] == pytest.approx((1 / 2 + 2 / 3) / 3)
    assert evaluation._rank_diagnostics(["noise"], {"c1": 1}, k=3) == {"hit_rate_at_k": 0, "map_at_k": 0}
    assert evaluation._rank_diagnostics(["noise"], {}, k=3) == {}


def test_citation_diagnostics_distinguish_missing_and_invented_evidence():
    docs = [Document(page_content="Invoice total 123.45 USD", metadata={"chunk_id": "c1", "pages": "[2,3]"})]
    valid = evaluation._citation_diagnostics("123.45 USD [Source 1; pages 2-3]", docs, {2, 3})
    assert valid == {"citation_page_validity": 1, "citation_page_coverage": 1, "citation_source_validity": 1}
    invented = evaluation._citation_diagnostics("123.45 USD [Source 8; pages 2, 99]", docs, {2, 3})
    assert invented == {"citation_page_validity": .5, "citation_page_coverage": .5, "citation_source_validity": 0}
    missing = evaluation._citation_diagnostics("123.45 USD", docs, {2, 3})
    assert all(value == 0 for value in missing.values())
    assert "does not establish" in evaluation.METRIC_CATALOG["citation_page_validity"]["description"]
    assert "not answer-claim coverage" in evaluation.METRIC_CATALOG["citation_page_coverage"]["description"]


def test_live_style_abbreviated_citations_use_actual_source_pages():
    docs = [
        Document(page_content="RAG components", metadata={"pages": "[2]"}),
        Document(page_content="Ragas metrics", metadata={"pages": "[42]"}),
        Document(page_content="Pipeline stages", metadata={"pages": "[3]"}),
    ]
    answer = "Retriever and generator [Source 1, p. 2]. Pipeline stages [Source 3, p. 3]."
    assert evaluation._citation_diagnostics(answer, docs, {2}) == {
        "citation_page_validity": 1, "citation_page_coverage": 1, "citation_source_validity": 1,
    }


@pytest.mark.parametrize("label,pages", [("p.", "2, 3"), ("P.", "2 and 3"),
                                       ("pp.", "2–3"), ("PP.", "2-3")])
def test_abbreviated_page_lists_and_ranges_preserve_coverage(label, pages):
    docs = [Document(page_content="Source text", metadata={"pages": "[2,3]"})]
    metrics = evaluation._citation_diagnostics(f"Answer [Source 1; {label} {pages}]", docs, {2, 3})
    assert metrics["citation_page_validity"] == 1
    assert metrics["citation_page_coverage"] == 1


def test_abbreviated_citations_do_not_accept_invented_pages_or_oversized_ranges():
    docs = [Document(page_content="Source text", metadata={"pages": "[2,3]"})]
    invented = evaluation._citation_diagnostics("Answer [Source 1, pp. 2, 99]", docs, {2, 3})
    assert invented["citation_page_validity"] == .5
    assert invented["citation_page_coverage"] == .5
    oversized = evaluation._citation_diagnostics("Answer [Source 1, pp. 2–999]", docs, {2, 3})
    assert oversized["citation_page_validity"] == .5
    assert oversized["citation_page_coverage"] == .5
    omitted = evaluation._citation_diagnostics("Answer [Source 2, p. 99]", docs, {2, 3})
    assert omitted == {"citation_page_validity": 0, "citation_page_coverage": 0, "citation_source_validity": 0}


def test_latency_denominators_exclude_missing_values_and_report_failures():
    records = [
        {"strategy": "Vector", "status": "completed", "retrieval_seconds": 1, "generation_seconds": 2, "metrics": {}},
        {"strategy": "Vector", "status": "partial", "retrieval_seconds": 3, "generation_seconds": 4, "metrics": {}},
        {"strategy": "Vector", "status": "failed", "retrieval_seconds": 5, "metrics": {}},
        {"strategy": "Vector", "status": "cancelled", "metrics": {}},
    ]
    summary = evaluation.summarize_records(records).iloc[0]
    assert summary["attempted"] == 3
    assert summary["failure_rate"] == pytest.approx(2 / 3)
    assert summary["partial_rate"] == pytest.approx(1 / 3)
    assert summary["error_rate"] == pytest.approx(1 / 3)
    assert summary["retrieval_seconds_count"] == 3
    assert summary["retrieval_seconds_p50"] == 3
    assert summary["retrieval_seconds_p95"] == pytest.approx(4.8)
    assert summary["generation_seconds_count"] == 2
    assert summary["generation_seconds"] == 3
    assert summary["judge_seconds_count"] == 0
    assert summary["judge_seconds"] is None


def test_citations_cannot_reference_passages_omitted_by_context_limits(runtime, monkeypatch):
    from backend import retrieval_runtime

    db_dir, output, _ = runtime
    documents = [
        Document(page_content="Invoice total is 123.45 USD", metadata={"chunk_id": "c1", "pages": "[1]"}),
        Document(page_content="Unrelated appendix", metadata={"chunk_id": "c2", "pages": "[99]"}),
    ]

    class Retriever:
        def invoke(self, question):
            return documents

    retriever = Retriever()
    monkeypatch.setattr(retrieval_runtime, "get_retrievers", lambda db, k=10: (retriever, retriever, retriever, documents))
    monkeypatch.setattr(generation_runtime, "prepare_generation_context", lambda docs, question, provider=None: {
        "prompt": question + documents[0].page_content, "text_contexts": [documents[0].page_content],
        "chunk_ids": ["c1"], "image_paths": [], "images_base64": [],
    })
    monkeypatch.setattr(generation_runtime, "generate_final_answer", lambda *args, **kwargs: "123.45 USD [Source 2; page 99]")
    report = asyncio.run(evaluation.run_evaluation(
        str(db_dir), dataset=[row()], profile="full", strategies=["Vector"], output_dir=output,
    ))
    record = json.loads(Path(report.attrs["sample_path"]).read_text().splitlines()[0])
    assert record["retrieved_chunk_ids"] == ["c1"]
    assert record["metrics"]["citation_page_validity"]["value"] == 0
    assert record["metrics"]["citation_source_validity"]["value"] == 0


def test_cancelled_only_report_has_no_fake_zero_failure_rate():
    summary = evaluation.summarize_records([{"strategy": "Hybrid", "status": "cancelled", "metrics": {}}]).iloc[0]
    assert summary["failure_rate"] is None
    assert summary["attempted"] == 0


def test_reranking_alias_reports_actual_hybrid_fallback_and_dataset_provenance(runtime, monkeypatch):
    db_dir, output, state = runtime
    monkeypatch.setenv("RERANKER_PROVIDER", "none")
    dataset = {"document_hash": "a" * 64, "questions": [row()],
               "provenance": {"kind": "synthetic", "verification": "source-grounded"},
               "warnings": ["Automatically generated references are not human calibration labels."]}
    report = asyncio.run(evaluation.run_evaluation(str(db_dir), dataset=dataset, profile="full", strategies=["Reranking"], output_dir=output))
    assert report.attrs["status"] == "completed"
    assert report.attrs["dataset_provenance"] == dataset["provenance"]
    assert report.attrs["scope"] == "all generated cases"
    assert report.attrs["metric_catalog"]["faithfulness"]["method"] == "Ragas judge"
    assert any("No reranker quality comparison" in warning for warning in report.attrs["warnings"])
    assert report.iloc[0]["reranked_samples"] == 0
    assert report.iloc[0]["degraded_samples"] == 1
    assert state["judge_calls"] == 5
    record = json.loads(Path(report.attrs["sample_path"]).read_text().splitlines()[0])
    assert record["strategy"] == "Reranking"
    assert record["effective_strategy"] == "Hybrid"
    assert record["sample_seconds"] > 0
    assert record["metrics"]["citation_page_validity"]["value"] == 0
    dataset["provenance"]["verification"] = "updated-source-verifier"
    updated = asyncio.run(evaluation.run_evaluation(str(db_dir), dataset=dataset, profile="full", strategies=["Reranking"], output_dir=output))
    assert updated.attrs["run_key"] != report.attrs["run_key"]


def test_reranking_alias_cannot_duplicate_production_or_run_local_profile():
    with pytest.raises(ValueError, match="duplicate"):
        evaluation._selection(load_evaluation_dataset([row()]), "preview", strategies=["Production", "Reranking"])
    with pytest.raises(ValueError, match="external reranker"):
        evaluation._selection(load_evaluation_dataset([row()]), "retrieval", strategies=["Reranking"])


def test_false_answer_rate_is_derived_only_for_unanswerable_cases(runtime):
    db_dir, output, state = runtime
    report = asyncio.run(evaluation.run_evaluation(
        str(db_dir), dataset=[row(answerable=False, reference_pages=[])], profile="preview", output_dir=output,
    ))
    assert report.iloc[0]["abstention_accuracy"] == .8
    assert report.iloc[0]["false_answer_rate"] == pytest.approx(.2)
    assert report.iloc[0]["faithfulness_count"] == 0
    assert state["judge_calls"] == 1


def test_judge_token_usage_is_measured_when_provider_supplies_it():
    from pydantic import BaseModel
    from types import SimpleNamespace
    from backend.evaluation_judge import JudgeSettings, create_json_judge_llm

    class Verdict(BaseModel):
        faithful: bool

    class Model:
        async def ainvoke(self, messages):
            return SimpleNamespace(content='{"faithful":true}', response_metadata={"finish_reason": "STOP"},
                                   usage_metadata={"input_tokens": 12, "output_tokens": 6, "total_tokens": 18})

    judge = create_json_judge_llm(JudgeSettings(), model=Model())
    asyncio.run(judge.agenerate("Check evidence", Verdict))
    assert judge.usage["token_usage_requests"] == 1
    assert judge.usage["total_tokens"] == 18
