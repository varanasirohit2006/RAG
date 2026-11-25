import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from backend.evaluation_judge import JudgeSettings, RagasJudge, create_google_json_llm, create_json_judge_llm, provider_status_code


class Verdict(BaseModel):
    supported: bool


class ProviderError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(f"Provider status {code}")


class Model:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(content=value, response_metadata={"finish_reason": "STOP"})


def test_provider_code_is_preserved_through_wrappers_and_cycles_are_bounded():
    from google.genai.errors import ClientError

    underlying = ClientError(403, {"error": {"code": 403, "message": "Access denied", "status": "PERMISSION_DENIED"}})
    wrapper = RuntimeError("Wrapped provider error")
    wrapper.__cause__ = underlying
    assert provider_status_code(wrapper) == 403
    wrapper.__cause__ = wrapper
    assert provider_status_code(wrapper) is None
    assert provider_status_code(RuntimeError("text mentions 503")) is None


def test_gemma_client_honors_configured_timeout_without_hidden_sdk_retries(monkeypatch):
    import langchain_google_genai

    calls = []
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI",
                        lambda **kwargs: calls.append(kwargs) or Model([]))
    create_google_json_llm(JudgeSettings(timeout=120, max_retries=2))
    assert calls[0]["timeout"] == 120
    assert calls[0]["max_retries"] == 1


def test_openai_compatible_client_honors_configured_timeout(monkeypatch):
    import openai
    import ragas.llms

    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: calls.append(kwargs) or object())
    # Keep the real metrics construction while replacing network clients only.
    llm = create_google_json_llm(JudgeSettings(), model=Model([]))
    monkeypatch.setattr(ragas.llms, "llm_factory", lambda *args, **kwargs: llm)
    RagasJudge(JudgeSettings(provider="openai", model="test-judge", timeout=120)).prepare()
    assert calls[0]["timeout"] == 120
    assert calls[0]["max_retries"] == 0


def test_gemini_31_uses_google_json_client_and_minimal_reasoning(monkeypatch):
    import langchain_google_genai
    options = []
    monkeypatch.setenv('GOOGLE_API_KEY', 'test-google-key')
    monkeypatch.setattr(langchain_google_genai, 'ChatGoogleGenerativeAI', lambda **kwargs: options.append(kwargs) or Model([]))
    judge = RagasJudge(JudgeSettings(provider='google', model='gemini-3.1-flash-lite', reasoning_effort='minimal'))
    judge.prepare()
    assert judge.llm.provider == 'google' and judge.llm.model == 'gemini-3.1-flash-lite'
    assert options[0]['reasoning_effort'] == 'minimal' and options[0]['include_thoughts'] is False
    assert options[0]['max_retries'] == 1 and options[0]['timeout'] == 90


def test_completed_judging_step_survives_a_later_timeout(tmp_path):
    model = Model(['{"supported":true}'])
    settings = JudgeSettings()
    first = create_json_judge_llm(settings, model=model, response_cache=tmp_path)
    assert asyncio.run(first.agenerate('First completed workflow step', Verdict)).supported
    failing = Model([ProviderError(503)])
    second = create_json_judge_llm(settings, model=failing, response_cache=tmp_path)
    assert asyncio.run(second.agenerate('First completed workflow step', Verdict)).supported
    assert failing.calls == 0 and second.usage['response_cache_hits'] == 1


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_transient_provider_error_retries_the_same_verdict(code):
    model = Model([ProviderError(code), '{"supported":true}'])
    llm = create_google_json_llm(JudgeSettings(max_retries=1), model=model)
    assert asyncio.run(llm.agenerate("Verify source evidence.", Verdict)).supported
    assert model.calls == 2
    assert llm.usage["provider_retries"] == 1
    assert llm.usage["json_repairs"] == 0


@pytest.mark.parametrize("code", [400, 401, 403, 404, 429])
def test_nontransient_errors_fail_without_retrying(code):
    model = Model([ProviderError(code), '{"supported":true}'])
    llm = create_google_json_llm(JudgeSettings(max_retries=2), model=model)
    with pytest.raises(ProviderError) as error:
        asyncio.run(llm.agenerate("Verify source evidence.", Verdict))
    assert error.value.code == code
    assert model.calls == 1


@pytest.mark.parametrize("responses,retries", [
    ([ProviderError(503), ProviderError(504), '{"supported":true}'], 1),
    ([ProviderError(503), '{"supported":true}'], 0),
    ([ProviderError(503), 'not JSON', '{"supported":true}'], 1),
])
def test_provider_retries_and_json_repairs_share_one_attempt_budget(responses, retries):
    model = Model(responses)
    llm = create_google_json_llm(JudgeSettings(max_retries=retries), model=model)
    with pytest.raises((ProviderError, ValueError)):
        asyncio.run(llm.agenerate("Verify source evidence.", Verdict))
    assert model.calls == retries + 1


@pytest.mark.parametrize("phase", ["request", "backoff"])
def test_total_deadline_cancels_active_requests_and_retry_backoff(phase):
    events = []

    class SlowModel:
        async def ainvoke(self, messages):
            events.append("request")
            if phase == "backoff":
                raise ProviderError(503)
            try:
                await asyncio.sleep(10)
            finally:
                events.append("cancelled")

    # Scale down the valid production settings for a fast real cancellation test.
    settings = SimpleNamespace(**{**asdict(JudgeSettings()), "timeout": 0.03})
    llm = create_google_json_llm(settings, model=SlowModel())

    async def check():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(llm.agenerate("Verify source evidence.", Verdict), timeout=0.5)
        await asyncio.sleep(0.04)
        assert events == (["request", "cancelled"] if phase == "request" else ["request"])
        assert llm.usage["structured_generation_requests"] == 1

    asyncio.run(check())
