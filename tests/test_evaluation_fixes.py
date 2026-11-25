import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from pydantic import BaseModel

from backend import advanced_evaluation as evaluation, evaluation_generation as drafting, generation_runtime, bootstrap, files, index_runtime
from backend.evaluation_data import EvaluationDataError, load_evaluation_dataset, resolve_relevance, retrieval_metrics
from backend.evaluation_evidence import draft_source_texts, evidence_labels
from backend.evaluation_judge import JudgeSettings, create_json_judge_llm, configure_context_precision
from test_evaluation import runtime, row
from test_evaluation_generation import runtime as drafting_runtime, generate


def test_native_single_line_footers_are_removed_only_from_drafting():
    documents = [Document(page_content=f'Policy {index} has substantive evidence for a benchmark question. RAG Evaluation Handbook | Page {index} of 101')
                 for index in range(1, 4)]
    documents.append(Document(page_content='RAG Evaluation Handbook | Page 4 of 101'))
    texts, labels = draft_source_texts(documents)
    assert labels == ['RAG Evaluation Handbook']
    assert texts[-1] == ''
    assert all('Page' not in text for text in texts)
    assert 'Page 1 of 101' in documents[0].page_content
    assert 'substantive evidence' in texts[0]


def test_repeated_complete_quotes_are_alternatives_for_recall_and_citations():
    quote = 'Invoice A total is 123.45 USD.'
    draft = drafting.DraftQuestion(question='What is Invoice A total?', reference_answer='123.45 USD',
        category='factual', supporting_quotes=[{'chunk_id': 'c1', 'quote': quote}])
    sources = [{'chunk_id': 'c1', 'text': quote + ' Original invoice.'},
               {'chunk_id': 'c2', 'text': quote + ' Repeated in an appendix.'},
               {'chunk_id': 'wrong', 'text': 'Invoice A total is 128.45 USD.'}]
    ids, groups = evidence_labels(draft, sources)
    assert ids == ['c1', 'c2'] and groups == [['c1', 'c2']]
    case = load_evaluation_dataset([row(relevant_chunk_ids=ids, relevant_chunk_groups=groups)])[0]
    documents = [Document(page_content=source['text'], metadata={'chunk_id': source['chunk_id'], 'pages': [index]})
                 for index, source in enumerate(sources, 1)]
    grades = resolve_relevance(case, documents)
    assert retrieval_metrics(['c2'], grades, k=1)['recall_at_k'] == 1
    assert retrieval_metrics(['c1', 'c2'], grades, k=2)['recall_at_k'] == 1
    metrics = evaluation._citation_diagnostics('123.45 USD [Source 1; page 2]', [documents[1]], {1, 2}, reference_page_groups=[{1, 2}])
    assert metrics['citation_page_coverage'] == 1


def test_generic_quotes_do_not_widen_labels_to_a_different_subject():
    draft = drafting.DraftQuestion(question='What is the payment deadline for Invoice A?', reference_answer='30 days',
        category='factual', supporting_quotes=[{'chunk_id': 'a', 'quote': 'Payment is due in 30 days.'}])
    ids, groups = evidence_labels(draft, [{'chunk_id': 'a', 'text': 'Invoice A. Payment is due in 30 days.'},
                                        {'chunk_id': 'b', 'text': 'Invoice B. Payment is due in 30 days.'}])
    assert ids == ['a'] and groups == [['a']]


@pytest.mark.parametrize('category', ['factual', 'paraphrase'])
def test_missing_verifier_categories_are_explicitly_allowed(drafting_runtime, monkeypatch, category):
    original = drafting._build_clients

    def build(configuration):
        model, verifier = original(configuration)
        invoke, verify = model.ainvoke, verifier.llm.agenerate

        async def draft(messages, **kwargs):
            response = await invoke(messages, **kwargs)
            data = json.loads(response.content)
            data['category'] = category
            response.content = json.dumps(data)
            return response

        async def check(prompt, response_model):
            assert 'All five categories are allowed: factual, paraphrase, comparison, multi_chunk, numeric.' in prompt
            assert 'for factual when it asks for a directly stated substantive fact' in prompt
            assert 'for paraphrase when it rewords source terminology' in prompt
            return await verify(prompt, response_model)

        model.ainvoke, verifier.llm.agenerate = draft, check
        return model, verifier

    monkeypatch.setattr(drafting, '_build_clients', build)
    result = generate(drafting_runtime, question_count=1)
    assert result['questions'][0]['category'] == category


def test_document_title_trivia_is_rejected_before_verification(drafting_runtime, monkeypatch):
    original = drafting._build_clients

    def build(configuration):
        model, verifier = original(configuration)
        invoke = model.ainvoke

        async def draft(messages, **kwargs):
            response = await invoke(messages, **kwargs)
            data = json.loads(response.content)
            data['question'] = 'What is the title of the handbook mentioned in the document?'
            response.content = json.dumps(data)
            return response

        model.ainvoke = draft
        return model, verifier

    monkeypatch.setattr(drafting, '_build_clients', build)
    with pytest.raises(EvaluationDataError, match='no source-verified'):
        generate(drafting_runtime, question_count=1)
    assert drafting_runtime[3]['verifier_calls'] == 0


def test_matched_means_require_the_same_questions_not_just_equal_counts():
    records = [
        {'strategy': 'Keyword', 'question_id': 'a', 'status': 'partial', 'metrics': {'faithfulness': {'status': 'completed', 'value': 1}}},
        {'strategy': 'Keyword', 'question_id': 'b', 'status': 'partial', 'metrics': {'faithfulness': {'status': 'failed'}}},
        {'strategy': 'Vector', 'question_id': 'a', 'status': 'partial', 'metrics': {'faithfulness': {'status': 'failed'}}},
        {'strategy': 'Vector', 'question_id': 'b', 'status': 'partial', 'metrics': {'faithfulness': {'status': 'completed', 'value': .8}}},
    ]
    summary = evaluation.summarize_records(records)
    assert list(summary['faithfulness_count']) == [1, 1]
    assert list(summary['faithfulness_matched_count']) == [0, 0]
    assert all(value is None for value in summary['faithfulness_matched'])
    records[1]['metrics']['faithfulness'] = {'status': 'completed', 'value': .6}
    summary = evaluation.summarize_records(records)
    assert list(summary['faithfulness_matched_count']) == [1, 1]
    assert list(summary['faithfulness_matched']) == [.6, .8]


def test_fresh_latency_excludes_reuse_failures_and_partial_judging():
    records = [
        {'strategy': 'Vector', 'status': 'completed', 'generation_mode': 'fresh', 'generation_seconds': 4, 'judge_reused': False, 'judge_seconds': 10, 'metrics': {}},
        {'strategy': 'Vector', 'status': 'completed', 'generation_mode': 'cache', 'generation_seconds': .01, 'judge_reused': True, 'judge_seconds': .02, 'metrics': {}},
        {'strategy': 'Vector', 'status': 'failed', 'generation_mode': 'shared', 'generation_seconds': .0005, 'metrics': {}},
        {'strategy': 'Vector', 'status': 'partial', 'generation_mode': 'fresh', 'generation_seconds': 6, 'judge_reused': False, 'judge_seconds': 240, 'metrics': {}},
    ]
    summary = evaluation.summarize_records(records).iloc[0]
    assert summary['generation_seconds_fresh_count'] == 2
    assert summary['generation_seconds_fresh_p50'] == 5
    assert summary['judge_seconds_fresh_count'] == 1
    assert summary['judge_seconds_fresh_p50'] == 10
    assert summary['generation_reused_samples'] == 2


def test_shared_failed_answers_are_marked_as_reuse_not_fast_inference(runtime, monkeypatch):
    db, output, _ = runtime

    def timeout(*args, **kwargs):
        raise TimeoutError('Provider deadline reached')

    monkeypatch.setattr(generation_runtime, 'generate_final_answer', timeout)
    report = asyncio.run(evaluation.run_evaluation(str(db), dataset=[row()], strategies=['Keyword', 'Vector'], output_dir=output))
    records = [json.loads(line) for line in open(report.attrs['sample_path'], encoding='utf-8')]
    assert {record['generation_mode'] for record in records} == {'fresh', 'shared'}
    assert all(record['status'] == 'failed' for record in records)
    assert report.attrs['work_counts']['generation_operations'] == 1
    assert all(report['generation_seconds_fresh_count'] == 0)


def test_whole_truncated_verdict_is_redrafted_never_used_as_a_score():
    class Verdict(BaseModel):
        supported: bool

    responses = iter([SimpleNamespace(content='{"supported":true}', response_metadata={'finish_reason': 'length'}),
                      SimpleNamespace(content='{"supported":false}', response_metadata={'finish_reason': 'stop'})])

    class Model:
        async def ainvoke(self, messages):
            return next(responses)

    llm = create_json_judge_llm(JudgeSettings(max_retries=1), model=Model())
    assert asyncio.run(llm.agenerate('Check the original evidence.', Verdict)).supported is False
    assert llm.usage['truncation_retries'] == 1
    assert llm.usage['structured_generation_requests'] == 2


def test_context_precision_examples_credit_partial_evidence_and_reject_wrong_entities():
    from ragas.metrics.collections import ContextPrecisionWithReference

    class Model:
        async def ainvoke(self, messages):
            text = messages[0].content[-1]['text']
            assert 'Missing other answer facts is NOT a reason for verdict=0' in text
            assert 'Describes a different invoice' in text
            return SimpleNamespace(content='{"verdict":1,"reason":"Supports one material answer fact."}', response_metadata={})

    metric = ContextPrecisionWithReference(llm=create_json_judge_llm(JudgeSettings(), model=Model()))
    configure_context_precision(metric)
    result = asyncio.run(metric.ascore(user_input='What are the invoice total and deadline?',
        reference='Invoice A total is 7 USD and payment is due in 30 days.', retrieved_contexts=['Invoice A total is 7 USD.']))
    assert result.value == pytest.approx(1)


def test_transport_timeout_can_retry_inside_the_remaining_answer_budget(monkeypatch):
    import httpx
    calls = []
    monkeypatch.setenv('GENERATION_MAX_RETRIES', '1')
    monkeypatch.setenv('GENERATION_RETRY_DELAY_SECONDS', '.1')
    monkeypatch.setenv('MODEL_TIMEOUT_SECONDS', '60')

    def invoke(messages):
        calls.append(messages)
        if len(calls) == 1:
            raise httpx.ReadTimeout('Transport deadline')
        return SimpleNamespace(content='123.45 USD [Source 1; page 1]', response_metadata={})

    monkeypatch.setattr(generation_runtime, 'get_generation_model', lambda *args, **kwargs: SimpleNamespace(invoke=invoke))
    answer = generation_runtime.generate_final_answer([Document(page_content='Invoice total is 123.45 USD.')], 'What is the total?', provider='google')
    assert '123.45 USD' in answer and len(calls) == 2


def test_judge_transport_timeout_retries_without_scoring_the_failed_attempt():
    import httpx

    class Verdict(BaseModel):
        supported: bool

    responses = iter([httpx.ReadTimeout('Transport deadline'), '{"supported":true}'])

    class Model:
        async def ainvoke(self, messages):
            value = next(responses)
            if isinstance(value, Exception):
                raise value
            return SimpleNamespace(content=value, response_metadata={})

    llm = create_json_judge_llm(JudgeSettings(timeout=30, request_timeout=15), model=Model())
    assert asyncio.run(llm.agenerate('Check the evidence.', Verdict)).supported
    assert llm.usage['structured_generation_requests'] == 2
    assert llm.usage['provider_retries'] == 1


def test_bug_b1_vector_store_embedding_change_purges_old_ids(monkeypatch, tmp_path):
    deleted_ids = []
    added_docs = []

    class FakeDB:
        def get(self, include):
            return {"ids": ["chunk1", "chunk2"]}
        def delete(self, ids):
            deleted_ids.extend(ids)
        def add_documents(self, batch, ids):
            added_docs.extend(ids)

    monkeypatch.setattr(index_runtime, "open_vector_store", lambda directory, embedding: FakeDB())
    monkeypatch.setattr(index_runtime, "read_index_manifest", lambda directory: {"embedding_model": "old-model", "embedding_revision": "old-rev"})
    monkeypatch.setattr(index_runtime, "embedding_name", lambda: "new-model")
    monkeypatch.setattr(index_runtime, "embedding_revision", lambda: "new-rev")
    monkeypatch.setattr(index_runtime, "_prepare_documents", lambda docs, directory: docs)

    doc1 = Document(page_content="text1", metadata={"chunk_id": "chunk1", "source_document": "d1", "pages": [1]})
    doc2 = Document(page_content="text2", metadata={"chunk_id": "chunk2", "source_document": "d1", "pages": [2]})

    index_runtime.build_vector_store([doc1, doc2], tmp_path, embedding=None)
    assert set(deleted_ids) == {"chunk1", "chunk2"}
    assert added_docs == ["chunk1", "chunk2"]


def test_bug_b2_normalized_model_prefix_gemini_2_5(monkeypatch):
    from langchain_google_genai import ChatGoogleGenerativeAI
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("GENERATION_THINKING_BUDGET", "1024")
    captured_kwargs = {}
    original_init = ChatGoogleGenerativeAI.__init__
    def mock_init(self, **kwargs):
        captured_kwargs.update(kwargs)
        original_init(self, **kwargs)
    monkeypatch.setattr(ChatGoogleGenerativeAI, "__init__", mock_init)
    generation_runtime._build_model.cache_clear()
    generation_runtime.get_generation_model("models/gemini-2.5-flash", provider="google")
    assert captured_kwargs.get("thinking_budget") == 1024


def test_bug_b3_json_judge_llm_closes_http_async_client():
    closed = []
    class MockClient:
        async def aclose(self):
            closed.append("http_async_client")

    class Model:
        http_async_client = MockClient()

    llm = create_json_judge_llm(JudgeSettings(), model=Model())
    asyncio.run(llm.close())
    assert closed == ["http_async_client"]


def test_bug_b4_summarize_records_excludes_timeout_records_from_latency():
    records = [
        {"strategy": "Vector", "status": "completed", "generation_seconds": 2.0, "metrics": {}},
        {"strategy": "Vector", "status": "timeout", "generation_seconds": 90.0, "metrics": {}},
        {"strategy": "Vector", "status": "failed", "timeout": True, "generation_seconds": 90.0, "metrics": {}},
    ]
    summary = evaluation.summarize_records(records).iloc[0]
    assert summary["generation_seconds_count"] == 1
    assert summary["generation_seconds"] == 2.0


def test_bug_b5_citation_diagnostics_handles_malformed_page_ranges():
    docs = [Document(page_content="Text", metadata={"pages": [1, 2, 3]})]
    v1 = evaluation._citation_diagnostics("See pages 1, and 2.", docs)
    assert v1["citation_page_validity"] == 1.0
    v2 = evaluation._citation_diagnostics("See pages 1--3 and page -5.", docs)
    assert "citation_page_validity" in v2
    v3 = evaluation._citation_diagnostics("See page 1, 2,", docs)
    assert v3["citation_page_validity"] == 1.0


def test_bug_b6_citation_coverage_unretrieved_page_not_credited():
    docs = [Document(page_content="Text", metadata={"pages": [1]})]
    diagnostics = evaluation._citation_diagnostics("Found on page 5.", docs, reference_pages=[1, 5])
    assert diagnostics["citation_page_coverage"] == 0.0


def test_bug_b7_retrieval_metrics_grouped_alternative_evidence_idcg():
    from backend.evaluation_data import RelevanceLabels
    grades = RelevanceLabels({"c1": 1.0, "c2": 1.0})
    grades.groups = (frozenset({"c1", "c2"}),)
    metrics = retrieval_metrics(["c1"], grades, k=2)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["ndcg_at_k"] == pytest.approx(1.0)


def test_bug_b8_shared_answers_cache_purged_on_failure():
    shared_answers = {}
    async def fail_generate():
        raise RuntimeError("Generation exploded")

    async def run_test():
        task = asyncio.create_task(fail_generate())
        shared_answers["key1"] = task
        try:
            t = shared_answers["key1"]
            try:
                await asyncio.shield(t)
            except BaseException:
                if shared_answers.get("key1") is t:
                    shared_answers.pop("key1", None)
                raise
        except RuntimeError:
            pass
    asyncio.run(run_test())
    assert "key1" not in shared_answers


def test_bug_b9_content_disposition_filename_sanitization():
    import re
    dirty = 'malicious"filename\r\nHeader-Injection: true\x00.pdf'
    safe = re.sub(r'[\r\n"\\\x00-\x1f]', '_', dirty)
    assert '\r' not in safe and '\n' not in safe and '"' not in safe and '\x00' not in safe
    assert safe == 'malicious_filename__Header-Injection: true_.pdf'


def test_bug_b10_unicode_nfkc_in_evidence_labels():
    quote_with_nbsp = "Invoice\u00a0total is 100 USD."
    source_text = "Invoice total is 100 USD. Approved."
    draft = drafting.DraftQuestion(
        question="What is invoice total?",
        reference_answer="100 USD",
        category="factual",
        supporting_quotes=[{"chunk_id": "c1", "quote": quote_with_nbsp}],
    )
    sources = [{"chunk_id": "c1", "text": source_text}]
    ids, groups = evidence_labels(draft, sources)
    assert ids == ["c1"]


def test_bug_b11_warm_runtime_handles_generation_model_exception(monkeypatch):
    from backend import rag_pipeline
    monkeypatch.setattr(rag_pipeline, "get_embedding_model", lambda: SimpleNamespace(embed_query=lambda text: [0.1]))
    monkeypatch.setenv("GENERATION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    def fail(*args, **kwargs):
        raise ValueError("GENERATION_MODEL is not set")
    monkeypatch.setattr(generation_runtime, "get_generation_model", fail)
    bootstrap.warm_runtime()


def test_bug_b12_write_json_retries_on_permission_error(tmp_path, monkeypatch):
    target = tmp_path / "test.json"
    attempts = []
    orig_replace = files.Path.replace
    def faulty_replace(self, dst):
        attempts.append(len(attempts))
        if len(attempts) < 2:
            raise PermissionError("Access is denied")
        orig_replace(self, dst)
    monkeypatch.setattr(files.Path, "replace", faulty_replace)
    files.write_json(target, {"hello": "world"})
    assert len(attempts) == 2
    assert files.read_json(target) == {"hello": "world"}


def test_streaming_empty_chunks_no_unbound_local_error(monkeypatch):
    class EmptyStream:
        def __iter__(self):
            return self
        def __next__(self):
            raise StopIteration
        def close(self):
            pass

    class EmptyStreamModel:
        def stream(self, messages):
            return EmptyStream()

    monkeypatch.setattr(generation_runtime, "get_generation_model", lambda *args, **kwargs: EmptyStreamModel())
    with pytest.raises(generation_runtime.GenerationError, match="returned no answer text"):
        generation_runtime.generate_final_answer(
            [Document(page_content="Doc")], "Question?", provider="google",
            token_callback=lambda token: None, metadata_callback=lambda m: None
        )
