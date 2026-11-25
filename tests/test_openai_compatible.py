import asyncio
import json
from dataclasses import asdict

import httpx
import pytest
from langchain_core.documents import Document
from pydantic import BaseModel

from backend import evaluation_generation as drafting, generation_runtime as generation
from backend.evaluation_data import EvaluationDataError
from backend.evaluation_judge import JudgeSettings, RagasJudge

BASE_URL = "http://127.0.0.1:31415/v1"
MODEL = "auto:fast"
KEY = "private-test-proxy-key"


@pytest.fixture
def settings(monkeypatch):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda: False)
    for name, value in {
        "OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": BASE_URL,
        "GENERATION_PROVIDER": "openai", "GENERATION_MODEL": MODEL,
        "SYNTHETIC_GENERATION_PROVIDER": "openai", "SYNTHETIC_GENERATION_MODEL": MODEL,
        "EVALUATION_PROVIDER": "openai", "EVALUATION_MODEL": MODEL,
        "SYNTHETIC_ALLOW_SAME_MODEL_VERIFICATION": "true",
    }.items():
        monkeypatch.setenv(name, value)
    generation._build_model.cache_clear()
    yield
    generation._build_model.cache_clear()


def completion(content, *, finish_reason="stop"):
    return {"id": "fixture", "object": "chat.completion", "model": "fixture-routed-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}


def assert_proxy_request(request):
    assert str(request.url) == BASE_URL + "/chat/completions"
    assert request.headers["authorization"] == "Bearer " + KEY
    body = json.loads(request.content)
    assert body["model"] == MODEL
    assert "tools" not in body and "provider" not in body and "reasoning" not in body
    return body


@pytest.mark.parametrize("streaming", [False, True])
def test_proxy_answers_use_configured_url_key_and_chat_completions(settings, monkeypatch, streaming):
    import langchain_openai

    requests = []
    answer = "The invoice total is 7 USD [Source 1; page 2]."

    def respond(request):
        requests.append(request)
        body = assert_proxy_request(request)
        assert "7 USD" in body["messages"][0]["content"][0]["text"]
        if streaming:
            chunks = [{"id": "fixture", "object": "chat.completion.chunk", "model": MODEL,
                       "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": reason}]}
                      for text, reason in [(answer[:15], None), (answer[15:], None), ("", "stop")]]
            content = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=content)
        return httpx.Response(200, json=completion(answer))

    real = langchain_openai.ChatOpenAI
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: real(**kwargs, http_client=transport))
        tokens = []
        result = generation.generate_final_answer(
            [Document(page_content="Invoice total is 7 USD.", metadata={"pages": [2]})],
            "What is the invoice total?", token_callback=tokens.append if streaming else None)
        assert result == answer and len(requests) == 1
        if streaming:
            assert "".join(tokens) == answer


@pytest.mark.parametrize("entry", ["answer", "draft", "judge"])
def test_missing_proxy_key_is_reported_before_a_request(settings, monkeypatch, entry):
    monkeypatch.delenv("OPENAI_API_KEY")
    if entry == "answer":
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            generation.get_generation_model()
    elif entry == "draft":
        with pytest.raises(EvaluationDataError, match="OPENAI_API_KEY"):
            drafting._build_clients({"draft_provider": "openai", "verifier": asdict(JudgeSettings(provider="openai", model=MODEL))})
    else:
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            RagasJudge(JudgeSettings(provider="openai", model=MODEL)).prepare()


class Verdict(BaseModel):
    supported: bool


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_proxy_judging_validates_json_records_usage_and_closes_transport(settings, monkeypatch, finish_reason):
    requests, models = [], []
    original = generation.create_openai_generation_model

    def build(*args, **kwargs):
        model = original(*args, **kwargs)
        models.append(model)
        return model

    async def send(client, request, **kwargs):
        requests.append(request)
        assert_proxy_request(request)
        content = "not JSON" if len(requests) == 1 else '{"supported":true}'
        return httpx.Response(200, request=request, json=completion(content, finish_reason=finish_reason))

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr(generation, "create_openai_generation_model", build)

    async def run():
        judge = RagasJudge(JudgeSettings(provider="openai", model=MODEL, max_retries=1))
        judge.prepare()
        assert not requests
        assert {"faithfulness", "context_precision", "context_recall", "factual_correctness"} <= judge.metrics.keys()
        try:
            if finish_reason == "length":
                with pytest.raises(ValueError, match="truncated"):
                    await judge.llm.agenerate("Verify source evidence.", Verdict)
                assert len(requests) == 2 and judge.llm.usage['truncation_retries'] == 1
            else:
                assert (await judge.llm.agenerate("Verify source evidence.", Verdict)).supported
                assert judge.llm.usage["total_tokens"] == 60
                assert judge.llm.usage["json_repairs"] == 1
                assert len(requests) == 2
        finally:
            await judge.close()
        assert models[0].http_async_client.is_closed

    asyncio.run(run())


def test_proxy_drafts_and_verifies_exact_sources_with_separate_run_transports(settings, monkeypatch, tmp_path):
    requests, clients = [], []
    document_hash = "a" * 64
    source_text = "Invoice A total is 123.45 USD. Payment is due within thirty calendar days."

    async def sources(*args):
        return [Document(page_content=source_text, metadata={"chunk_id": "c1", "pages": [1]})], {"document_hash": document_hash}

    async def send(client, request, **kwargs):
        requests.append(request)
        body = assert_proxy_request(request)
        content = body["messages"][0]["content"]
        if isinstance(content, str):
            assert "SOURCE PASSAGES:\n" in content
            response = {"question": "What is the total for Invoice A?", "reference_answer": "123.45 USD",
                        "category": "factual", "supporting_quotes": [{"chunk_id": "c1", "quote": source_text}]}
        else:
            response = {"supported": True, "unambiguous": True, "all_evidence_relevant": True,
                        "category_valid": True, "reason": "Supported by the supplied invoice."}
        return httpx.Response(200, request=request, json=completion(json.dumps(response)))

    original = generation.create_openai_generation_model

    def build(*args, **kwargs):
        model = original(*args, **kwargs)
        clients.append(model.http_async_client)
        return model

    monkeypatch.setattr(drafting, "_load_sources", sources)
    monkeypatch.setattr(generation, "create_openai_generation_model", build)
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    for _ in range(2):
        result = asyncio.run(drafting.generate_evaluation_dataset(
            tmp_path / "db", expected_document_hash=document_hash, question_count=1,
            output_dir=tmp_path / "datasets", force=True))
        config = result["provenance"]["configuration"]
        assert config["draft_base_url"] == BASE_URL and config["verifier_base_url"] == BASE_URL
        assert KEY not in json.dumps(config)
        assert config["independent_reference_verification"] is False
        assert result["questions"][0]["supporting_quotes"][0]["quote"] == source_text
        assert any("Automatic model routing" in warning for warning in result["warnings"])
    assert len(clients) == 4 and len({id(client) for client in clients}) == 4
    assert all(client.is_closed for client in clients)
    # Second-run verification reuses the valid completed verdict from this endpoint.
    assert len(requests) == 3


def test_proxy_judging_cache_is_specific_to_the_endpoint(settings, monkeypatch):
    requests = []

    async def send(client, request, **kwargs):
        requests.append(request)
        return httpx.Response(200, request=request, json=completion('{"supported":true}'))

    monkeypatch.setattr(httpx.AsyncClient, "send", send)

    async def run():
        for endpoint in [BASE_URL, BASE_URL, "http://127.0.0.1:31416/v1"]:
            monkeypatch.setenv("OPENAI_BASE_URL", endpoint)
            judge = RagasJudge(JudgeSettings(provider="openai", model=MODEL))
            judge.prepare()
            try:
                assert (await judge.llm.agenerate("Verify identical evidence.", Verdict)).supported
            finally:
                await judge.close()

    asyncio.run(run())
    assert [str(request.url) for request in requests] == [
        BASE_URL + "/chat/completions", "http://127.0.0.1:31416/v1/chat/completions"]


def test_proxy_truncation_retry_increases_the_actual_request_budget_without_using_partial_json(settings, monkeypatch):
    limits = []

    async def send(client, request, **kwargs):
        body = assert_proxy_request(request)
        limits.append(body.get('max_completion_tokens', body.get('max_tokens')))
        return httpx.Response(200, request=request, json=completion(
            '{"supported":true}' if len(limits) == 1 else '{"supported":false}',
            finish_reason='length' if len(limits) == 1 else 'stop'))

    monkeypatch.setattr(httpx.AsyncClient, 'send', send)

    async def run():
        judge = RagasJudge(JudgeSettings(provider='openai', model=MODEL, max_tokens=8192))
        judge.prepare()
        try:
            assert (await judge.llm.agenerate('Check the supplied fact.', Verdict)).supported is False
        finally:
            await judge.close()

    asyncio.run(run())
    assert limits == [512, 1024]


@pytest.mark.parametrize("status_code", [401, 429])
def test_proxy_answer_errors_do_not_retry_or_expose_key(settings, monkeypatch, status_code):
    import langchain_openai

    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status_code, json={"error": {"message": KEY}})

    real = langchain_openai.ChatOpenAI
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: real(**kwargs, http_client=transport))
        with pytest.raises(generation.GenerationError) as error:
            generation.generate_final_answer([Document(page_content="Source evidence")], "Question?")
        assert str(status_code) in str(error.value) and KEY not in str(error.value)
        assert len(requests) == 1
