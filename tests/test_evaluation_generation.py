import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import evaluation_generation as generation
from evaluation_data import EvaluationDataError, load_evaluation_dataset

HASH = "a" * 64


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda: False)
    monkeypatch.setenv("GENERATION_PROVIDER", "google")
    monkeypatch.setenv("GENERATION_MODEL", "test-drafter")
    monkeypatch.setenv("EVALUATION_MODEL", "test-verifier")
    monkeypatch.setenv("EVALUATION_PROVIDER", "google")
    for name in ("SYNTHETIC_GENERATION_PROVIDER", "SYNTHETIC_GENERATION_MODEL", "SYNTHETIC_GENERATION_TIMEOUT", "SYNTHETIC_GENERATION_MAX_TOKENS", "SYNTHETIC_GENERATION_CONCURRENCY", 'SYNTHETIC_ALLOW_SAME_MODEL_VERIFICATION'):
        monkeypatch.delenv(name, raising=False)
    docs = [Document(page_content=f"Invoice {index} total is {100 + index}.45 USD. Payment is due within thirty calendar days.",
                     metadata={"chunk_id": f"c{index}", "pages": [index]}) for index in range(1, 7)]
    state = {"draft_calls": 0, "verifier_calls": 0, "closed": 0, "bad_quote": False,
             "unknown_chunk": False, "reject": False, "duplicate": False, "cancel": None, "wait": False, "outage": False, "verifier_outage": False}

    async def sources(db_path, expected_hash):
        return docs, {"document_hash": HASH, "corpus_version": "corpus-v1"}

    class DraftModel:
        async def ainvoke(self, messages):
            state["draft_calls"] += 1
            if state["outage"]:
                raise RuntimeError("Provider unavailable (503)")
            if state["cancel"]:
                state["cancel"].set()
            if state["wait"]:
                await asyncio.sleep(10)
            prompt = messages[0].content
            source_data = prompt.split("SOURCE PASSAGES:\n", 1)[1]
            window, _ = json.JSONDecoder().raw_decode(source_data)
            source = window[0]
            draft = {
                "question": "What is the payment deadline?" if state["duplicate"] else f"What is the total for invoice {source['chunk_id']}?",
                "reference_answer": source["text"].split(". Payment", 1)[0] + ".",
                "category": "factual",
                "supporting_quotes": [{"chunk_id": "invented" if state["unknown_chunk"] else source["chunk_id"],
                                       "quote": "The total is 900.00 EUR." if state["bad_quote"] else source["text"]}],
            }
            return SimpleNamespace(content=json.dumps(draft), response_metadata={"finish_reason": "STOP"})

    class VerifierLLM:
        async def agenerate(self, prompt, response_model):
            state["verifier_calls"] += 1
            if state["verifier_outage"]:
                raise RuntimeError("Verifier unavailable (503)")
            return response_model(supported=not state["reject"], unambiguous=True,
                                  all_evidence_relevant=True, category_valid=True,
                                  reason="Reference checked against the supplied source.")

    class Verifier:
        llm = VerifierLLM()

        async def close(self):
            state["closed"] += 1

    monkeypatch.setattr(generation, "_load_sources", sources)
    monkeypatch.setattr(generation, "_build_clients", lambda config: (DraftModel(), Verifier()))
    return tmp_path / "db", tmp_path / "datasets", docs, state


def generate(runtime, **kwargs):
    db, output, _, _ = runtime
    return asyncio.run(generation.generate_evaluation_dataset(db, expected_document_hash=HASH, output_dir=output, **kwargs))


def assert_no_valid_dataset(runtime):
    assert not [path for path in runtime[1].glob("automatic-*.json") if not path.name.endswith(".failure.json")]


def test_automatic_questions_have_verified_exact_evidence_and_honest_provenance(runtime):
    _, _, docs, state = runtime
    dataset = generate(runtime, question_count=3)
    cases = load_evaluation_dataset(dataset, expected_document_hash=HASH)
    assert len(cases) == 3
    assert all(case.answerable and case.relevant_chunk_ids and case.reference_pages for case in cases)
    assert dataset["provenance"]["origin"] == "synthetic_source_verified"
    assert dataset["provenance"]["human_reviewed"] is False
    assert dataset["provenance"]["work_counts"] == {"drafting_operations": 3, "verification_operations": 3}
    assert dataset["cache_reused"] is False
    assert Path(dataset["dataset_path"]).is_file()
    by_id = {doc.metadata["chunk_id"]: doc.page_content for doc in docs}
    for row in dataset["questions"]:
        assert row["verification"]["supported"] is True
        assert all(quote["quote"] in by_id[quote["chunk_id"]] for quote in row["supporting_quotes"])
    assert state["closed"] == 1
    before = dict(state)
    cached = generate(runtime, question_count=3)
    assert cached["cache_reused"] is True
    assert state == before


def test_openrouter_drafting_inherits_ultra_and_closes_each_jobs_transport(runtime, monkeypatch):
    import httpx
    model = 'nvidia/nemotron-3-ultra-550b-a55b:free'
    monkeypatch.setenv('GENERATION_PROVIDER', 'openrouter')
    monkeypatch.setenv('GENERATION_MODEL', model)
    original = generation._build_clients
    clients = []
    def build(configuration):
        assert configuration['draft_provider'] == 'openrouter' and configuration['draft_model'] == model
        draft, verifier = original(configuration)
        draft.http_async_client = httpx.AsyncClient()
        clients.append(draft.http_async_client)
        return draft, verifier
    monkeypatch.setattr(generation, '_build_clients', build)
    for _ in range(2):
        result = generate(runtime, question_count=1, force=True)
        assert result['provenance']['configuration']['draft_model'] == model
    assert len(clients) == 2 and clients[0] is not clients[1]
    assert all(client.is_closed for client in clients)


def test_explicit_same_model_generation_is_labelled_as_source_checking(runtime, monkeypatch):
    monkeypatch.setenv('GENERATION_MODEL', 'gemini-3.1-flash-lite')
    monkeypatch.setenv('EVALUATION_MODEL', 'gemini-3.1-flash-lite')
    monkeypatch.setenv('SYNTHETIC_ALLOW_SAME_MODEL_VERIFICATION', 'true')
    dataset = generate(runtime, question_count=1)
    assert dataset['provenance']['configuration']['independent_reference_verification'] is False
    assert any('not independent verification' in warning for warning in dataset['warnings'])


def test_combined_questions_balance_original_pdfs_and_report_reference_coverage(runtime, monkeypatch):
    db, output, documents, _ = runtime
    original_sources = [{'document_id': 'pdf-a', 'name': 'A.pdf'}, {'document_id': 'pdf-b', 'name': 'B.pdf'}]
    for index, document in enumerate(documents):
        document.metadata['source_pages'] = json.dumps([original_sources[0 if index < 3 else 1]])
    async def sources(*args):
        return documents, {'document_hash': HASH, 'corpus_version': 'combined', 'source_documents': original_sources}
    monkeypatch.setattr(generation, '_load_sources', sources)
    result = generate(runtime, question_count=2)
    coverage = result['provenance']['source_document_coverage']
    assert len(coverage) == 2 and all(item['linked_questions'] >= 1 for item in coverage)


@pytest.mark.parametrize("failure", ["bad_quote", "unknown_chunk"])
def test_invented_evidence_is_rejected_before_independent_model_call(runtime, failure):
    state = runtime[3]
    state[failure] = True
    with pytest.raises(EvaluationDataError, match="no source-verified"):
        generate(runtime, question_count=2)
    assert state["draft_calls"] == 4
    assert state["verifier_calls"] == 0
    assert state["closed"] == 1
    assert_no_valid_dataset(runtime)


def test_failed_source_verdict_never_becomes_a_reference(runtime):
    state = runtime[3]
    state["reject"] = True
    with pytest.raises(EvaluationDataError, match="no source-verified"):
        generate(runtime, question_count=2)
    assert state["verifier_calls"] == 4
    assert_no_valid_dataset(runtime)


def test_duplicate_questions_are_excluded_and_shortfall_is_explicit(runtime):
    state = runtime[3]
    state["duplicate"] = True
    dataset = generate(runtime, question_count=3)
    assert len(dataset["questions"]) == 1
    assert state["draft_calls"] == 6
    assert dataset["provenance"]["requested_questions"] == 3
    assert dataset["provenance"]["verified_questions"] == 1
    assert any("Only 1 of 3" in warning for warning in dataset["warnings"])
    assert any("Duplicate" in item["reason"] for item in dataset["provenance"]["rejected_candidates"])


def test_source_and_verifier_configuration_invalidate_cached_questions(runtime, monkeypatch):
    dataset = generate(runtime, question_count=2)
    docs, state = runtime[2:]
    docs[0].page_content += " An early settlement discount of 5 USD is available."
    changed = generate(runtime, question_count=2)
    assert changed["dataset_path"] != dataset["dataset_path"]
    assert changed["cache_reused"] is False
    monkeypatch.setenv("EVALUATION_MODEL", "another-verifier")
    third = generate(runtime, question_count=2)
    assert third["dataset_path"] != changed["dataset_path"]
    assert state["verifier_calls"] == 6


def test_tampered_cached_evidence_is_revalidated_and_rebuilt(runtime):
    dataset = generate(runtime, question_count=2)
    path = Path(dataset["dataset_path"])
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["questions"][0]["supporting_quotes"][0]["quote"] = "The invented total is 500 EUR."
    path.write_text(json.dumps(stored), encoding="utf-8")
    rebuilt = generate(runtime, question_count=2)
    assert rebuilt["cache_reused"] is False
    assert runtime[3]["verifier_calls"] == 4


def test_cancellation_interrupts_active_model_and_writes_no_dataset(runtime):
    state = runtime[3]
    cancellation = threading.Event()
    state["cancel"] = cancellation
    state["wait"] = True
    with pytest.raises(asyncio.CancelledError):
        generate(runtime, question_count=3, cancellation_event=cancellation)
    assert state["verifier_calls"] == 0
    assert state["closed"] == 1
    assert not runtime[1].exists()


def test_same_drafter_and_verifier_model_is_rejected_before_requests(runtime, monkeypatch):
    monkeypatch.setenv("EVALUATION_MODEL", "test-drafter")
    with pytest.raises(EvaluationDataError, match="different model"):
        generate(runtime, question_count=1)
    assert runtime[3]["draft_calls"] == 0


def test_same_model_on_another_provider_cannot_verify_its_own_drafts(runtime, monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "gemma-4-31b-it")
    monkeypatch.setenv("EVALUATION_PROVIDER", "openrouter")
    monkeypatch.setenv("EVALUATION_MODEL", "google/gemma-4-31b-it:free")
    with pytest.raises(EvaluationDataError, match="different model"):
        generate(runtime, question_count=1)
    assert runtime[3]["draft_calls"] == runtime[3]["verifier_calls"] == 0


@pytest.mark.parametrize("count", [0, 51, -1, True, 1.5])
def test_invalid_requested_question_counts_are_rejected(count):
    with pytest.raises(EvaluationDataError, match="1-50"):
        generation.generation_plan(count)


def test_multi_chunk_labels_need_two_genuinely_distinct_sources():
    draft = generation.DraftQuestion(
        question="How do invoice totals differ?", reference_answer="They differ by one dollar.", category="multi_chunk",
        supporting_quotes=[generation.SupportingQuote(chunk_id="c1", quote="The invoice total is 101.45 USD.")],
    )
    with pytest.raises(EvaluationDataError, match="at least two"):
        generation._validate_quotes(draft, [{"chunk_id": "c1", "text": "The invoice total is 101.45 USD.", "pages": [1]}])


@pytest.mark.parametrize("quote", [" " * 12, "\n \t" * 6, "X" + " " * 12])
def test_whitespace_cannot_satisfy_exact_supporting_evidence(quote):
    draft = generation.DraftQuestion(
        question="What fact does the report state?", reference_answer="It states X.", category="factual",
        supporting_quotes=[generation.SupportingQuote(chunk_id="c1", quote=quote)],
    )
    with pytest.raises(EvaluationDataError, match="after whitespace normalization"):
        generation._validate_quotes(draft, [{"chunk_id": "c1", "text": "X is the only original fact.", "pages": [1]}])


def test_source_index_mismatch_is_rejected_before_loading_or_model_requests(tmp_path, monkeypatch):
    import rag_pipeline

    manifest = tmp_path / "index_manifest.json"
    manifest.write_text(json.dumps({"document_hash": "b" * 64}), encoding="utf-8")
    def unexpected_load(path):
        pytest.fail("A mismatched index must not be loaded.")
    monkeypatch.setattr(rag_pipeline, "load_vector_store", unexpected_load)
    with pytest.raises(EvaluationDataError, match="another PDF"):
        asyncio.run(generation._load_sources(tmp_path, HASH))


def test_repeated_provider_outage_stops_calls_within_bounded_failure_budget(runtime):
    state = runtime[3]
    state["outage"] = True
    with pytest.raises(EvaluationDataError, match="503"):
        generate(runtime, question_count=10)
    assert 3 <= state["draft_calls"] <= 4
    assert state["verifier_calls"] == 0
    assert_no_valid_dataset(runtime)


def test_verifier_outage_is_not_reset_by_successful_question_drafting(runtime):
    state = runtime[3]
    state["verifier_outage"] = True
    with pytest.raises(EvaluationDataError, match="503"):
        generate(runtime, question_count=10)
    assert 3 <= state["verifier_calls"] <= 4
    assert state["draft_calls"] <= 4
    assert_no_valid_dataset(runtime)


def test_cached_reference_changes_require_new_source_verification(runtime):
    dataset = generate(runtime, question_count=2)
    path = Path(dataset["dataset_path"])
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["questions"][0]["reference_answer"] = "The invented total is 500 EUR."
    path.write_text(json.dumps(stored), encoding="utf-8")
    rebuilt = generate(runtime, question_count=2)
    assert rebuilt["cache_reused"] is False
    assert runtime[3]["verifier_calls"] == 4


def test_missing_model_credentials_fail_before_client_construction(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(EvaluationDataError, match="GOOGLE_API_KEY"):
        generation._build_clients({"draft_provider": "google", "verifier": {"provider": "google"}})


@pytest.mark.parametrize("outcome", ["success", "cancel"])
def test_concurrent_generation_rejected_before_paid_calls_and_lock_released(runtime, monkeypatch, outcome):
    db, output, _, state = runtime
    original_build = generation._build_clients
    constructed = 0

    async def exercise():
        nonlocal constructed
        gate = asyncio.Event()

        def gated_clients(configuration):
            nonlocal constructed
            constructed += 1
            model, verifier = original_build(configuration)
            original_invoke = model.ainvoke

            async def gated_invoke(messages):
                await gate.wait()
                return await original_invoke(messages)

            model.ainvoke = gated_invoke
            return model, verifier

        monkeypatch.setattr(generation, "_build_clients", gated_clients)

        async def run(force=False):
            return await generation.generate_evaluation_dataset(
                db, expected_document_hash=HASH, output_dir=output, question_count=1, force=force,
            )

        first = asyncio.create_task(run())
        try:
            while not constructed:
                await asyncio.sleep(.001)
            with pytest.raises(EvaluationDataError, match="already running"):
                await run()
            assert constructed == 1 and state["draft_calls"] == 0 and state["verifier_calls"] == 0
            if outcome == "cancel":
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                gate.set()
            else:
                gate.set()
                dataset = await first
                assert len(dataset["questions"]) == 1
            before = state["draft_calls"]
            retried = await run(force=True)
            assert len(retried["questions"]) == 1
            assert state["draft_calls"] == before + 1
            assert constructed == 2
        finally:
            if not first.done():
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)

    asyncio.run(exercise())


def test_client_setup_failure_releases_generation_lock(runtime, monkeypatch):
    original_build = generation._build_clients

    def unavailable(configuration):
        raise RuntimeError("Client setup failed before inference")

    monkeypatch.setattr(generation, "_build_clients", unavailable)
    with pytest.raises(RuntimeError, match="Client setup failed"):
        generate(runtime, question_count=1)
    monkeypatch.setattr(generation, "_build_clients", original_build)
    assert len(generate(runtime, question_count=1)["questions"]) == 1


def test_rejected_candidates_release_generation_lock(runtime):
    runtime[3]["bad_quote"] = True
    with pytest.raises(EvaluationDataError, match="no source-verified"):
        generate(runtime, question_count=1)
    runtime[3]["bad_quote"] = False
    assert len(generate(runtime, question_count=1)["questions"]) == 1


@pytest.mark.parametrize("explicit_model", [None, "explicit-local-model"])
def test_ollama_drafting_uses_local_model_unless_explicitly_overridden(runtime, monkeypatch, explicit_model):
    monkeypatch.setenv("SYNTHETIC_GENERATION_PROVIDER", "ollama")
    monkeypatch.setenv("LOCAL_GENERATION_MODEL", "local-test-model")
    if explicit_model:
        monkeypatch.setenv("SYNTHETIC_GENERATION_MODEL", explicit_model)
    dataset = generate(runtime, question_count=1)
    assert dataset["provenance"]["configuration"]["draft_model"] == (explicit_model or "local-test-model")


@pytest.mark.parametrize("first_failure", ["validation", "request"])
def test_generation_failure_reports_request_stage_model_and_all_rejections(runtime, monkeypatch, first_failure):
    monkeypatch.setenv("SYNTHETIC_GENERATION_CONCURRENCY", "1")
    original_build = generation._build_clients
    state = runtime[3]
    state["verifier_outage"] = True

    def clients(configuration):
        model, verifier = original_build(configuration)
        original_invoke = model.ainvoke

        async def invoke(messages):
            state["bad_quote"] = state["draft_calls"] == (0 if first_failure == "validation" else 1)
            return await original_invoke(messages)

        model.ainvoke = invoke
        return model, verifier

    monkeypatch.setattr(generation, "_build_clients", clients)
    with pytest.raises(EvaluationDataError) as error:
        generate(runtime, question_count=1)
    message = str(error.value)
    assert "verification request (test-verifier)" in message and "503" in message
    assert "availability" in message
    failures = list(runtime[1].glob("automatic-*.failure.json"))
    assert len(failures) == 1
    diagnostic = json.loads(failures[0].read_text(encoding="utf-8"))
    assert diagnostic["status"] == "failed" and diagnostic["accepted_questions"] == 0
    assert len(diagnostic["rejected_candidates"]) == 2
    assert {item["stage"] for item in diagnostic["rejected_candidates"]} == {"draft_validation", "verification_request"}
    assert {item["model"] for item in diagnostic["rejected_candidates"]} == {"test-drafter", "test-verifier"}
    assert "questions" not in diagnostic
    assert_no_valid_dataset(runtime)


def test_failure_diagnostics_and_actionable_error_redact_credentials(runtime, monkeypatch):
    secret = "fake-provider-secret-for-test"
    monkeypatch.setenv("GOOGLE_API_KEY", secret)
    original_build = generation._build_clients

    def clients(configuration):
        model, verifier = original_build(configuration)

        async def unavailable(prompt, response_model):
            raise RuntimeError("Verifier unavailable (503), credential=" + secret)

        verifier.llm.agenerate = unavailable
        return model, verifier

    monkeypatch.setattr(generation, "_build_clients", clients)
    with pytest.raises(EvaluationDataError) as error:
        generate(runtime, question_count=1)
    assert secret not in str(error.value) and "[redacted]" in str(error.value)
    diagnostic = next(runtime[1].glob("automatic-*.failure.json")).read_text(encoding="utf-8")
    assert secret not in diagnostic and "[redacted]" in diagnostic


@pytest.mark.parametrize("code", [503, 504, None])
def test_temporary_failure_message_reports_availability_without_blame_on_credentials(runtime, monkeypatch, code):
    original_build = generation._build_clients

    class ProviderError(Exception):
        def __init__(self):
            self.code = code
            super().__init__(f"{code}: model temporarily unavailable")

    def clients(configuration):
        model, verifier = original_build(configuration)

        async def unavailable(prompt, response_model):
            if code is None:
                raise TimeoutError()
            raise ProviderError()

        verifier.llm.agenerate = unavailable
        return model, verifier

    monkeypatch.setattr(generation, "_build_clients", clients)
    with pytest.raises(EvaluationDataError) as error:
        generate(runtime, question_count=1)
    message = str(error.value)
    assert "Retry later" in message and "do not indicate invalid credentials" in message
    assert len(message) < 600 and message.endswith(".failure.json.")
    if code is None:
        assert "configured deadline" in message
    diagnostic = json.loads(next(runtime[1].glob("automatic-*.failure.json")).read_text(encoding="utf-8"))
    assert all(item["provider_status_code"] == code for item in diagnostic["rejected_candidates"])
    assert all(item["model_request_timed_out"] == (code is None) for item in diagnostic["rejected_candidates"])
    assert_no_valid_dataset(runtime)
