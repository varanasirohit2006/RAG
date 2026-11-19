import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from backend.evaluation_judge import JudgeSettings, RagasJudge, create_openrouter_json_llm

FREE_MODEL = "inclusionai/ling-3.2-flash:free"


class Verdict(BaseModel):
    supported: bool
    reason: str


class Model:
    async def ainvoke(self, messages):
        return SimpleNamespace(content='{"supported":true,"reason":"Supported by source."}',
                               response_metadata={"finish_reason": "STOP"},
                               usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30})


@pytest.mark.parametrize("model", ["openrouter/free", "openrouter/auto", "openrouter/auto:free",
                                  "inclusionai/ling-3.2-flash", "ling-3.2-flash:free"])
def test_paid_models_and_random_routers_are_rejected(model):
    with pytest.raises(ValueError, match="fixed model ID"):
        JudgeSettings(provider="openrouter", model=model)


def test_openrouter_client_uses_dedicated_secret_and_zero_price_routes(monkeypatch):
    import langchain_openai

    monkeypatch.setenv("OPENROUTER_API_KEY", "private-openrouter-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "different-test-key")
    calls = []
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: calls.append(kwargs) or Model())
    settings = JudgeSettings(provider="openrouter", model=FREE_MODEL, timeout=120, max_retries=2)
    llm = create_openrouter_json_llm(settings)
    assert not llm.usage["structured_generation_requests"]
    options = calls[0]
    assert options["api_key"] == "private-openrouter-test-key"
    assert options["base_url"] == "https://openrouter.ai/api/v1"
    assert options["model"] == FREE_MODEL
    assert options["extra_body"]["provider"]["max_price"] == {"prompt": 0, "completion": 0}
    assert "models" not in options["extra_body"]
    assert options["timeout"] == 120 and options["max_retries"] == 0
    assert options["use_responses_api"] is False
    assert llm.provider == "openrouter"
    result = asyncio.run(llm.agenerate("Verify source evidence.", Verdict))
    assert result.supported and llm.usage["total_tokens"] == 30
    asyncio.run(options["http_async_client"].aclose())


@pytest.mark.parametrize("entry", ["adapter", "judge"])
def test_missing_openrouter_secret_cannot_fall_back_to_openai_key(monkeypatch, entry):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "different-test-key")
    settings = JudgeSettings(provider="openrouter", model=FREE_MODEL)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        if entry == "adapter":
            create_openrouter_json_llm(settings)
        else:
            RagasJudge(settings).prepare()


def test_free_judge_retains_ragas_metrics_and_closes_async_client(monkeypatch):
    import langchain_openai

    events = []

    class AsyncClient:
        async def close(self):
            events.append("closed")

    model = Model()
    model.root_async_client = AsyncClient()
    options = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: options.append(kwargs) or model)
    judge = RagasJudge(JudgeSettings(provider="openrouter", model=FREE_MODEL))
    judge.prepare()
    assert {"faithfulness", "context_precision", "context_recall", "factual_correctness"} <= judge.metrics.keys()
    assert judge.llm.provider == "openrouter" and judge.client is judge.llm
    asyncio.run(judge.close())
    assert events == ["closed"]
    asyncio.run(options[0]["http_async_client"].aclose())


def test_real_verifier_and_scoring_clients_remain_isolated_across_runs(monkeypatch):
    import langchain_openai

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    original = langchain_openai.ChatOpenAI
    models = []

    def capture(**kwargs):
        model = original(**kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", capture)
    settings = JudgeSettings(provider="openrouter", model=FREE_MODEL)

    async def check():
        verifier = create_openrouter_json_llm(settings)
        scorer = create_openrouter_json_llm(settings)
        clients = [model.http_async_client for model in models]
        assert clients[0] is not clients[1]
        assert not any(client.is_closed for client in clients)
        await verifier.close()
        assert clients[0].is_closed and not clients[1].is_closed
        later_run = create_openrouter_json_llm(settings)
        assert models[2].http_async_client is not clients[0]
        assert not models[2].http_async_client.is_closed
        await scorer.close()
        assert clients[1].is_closed and not models[2].http_async_client.is_closed
        await later_run.close()
        assert models[2].http_async_client.is_closed

    asyncio.run(check())


def test_openrouter_secret_is_redacted_from_both_evaluation_errors(monkeypatch):
    from backend import advanced_evaluation
    from backend import evaluation_generation

    secret = "private-openrouter-test-key"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    error = RuntimeError("Provider failed with key=" + secret)
    for redact in (advanced_evaluation._error, evaluation_generation._safe_error):
        message = redact(error)
        assert secret not in message and "[redacted]" in message


def test_automatic_generation_checks_openrouter_secret_before_drafting(monkeypatch):
    from backend import evaluation_generation

    monkeypatch.setenv("GOOGLE_API_KEY", "test-draft-key")
    monkeypatch.setenv("OPENAI_API_KEY", "different-test-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        evaluation_generation._build_clients({"draft_provider": "google", "verifier": {"provider": "openrouter"}})
