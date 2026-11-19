from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from backend import generation_runtime as generation


class ProviderError(Exception):
    def __init__(self, code):
        self.status_code = code
        super().__init__(f"Provider status {code}")


@pytest.fixture
def retry_runtime(monkeypatch):
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("GENERATION_MAX_RETRIES", "1")
    monkeypatch.setenv("GENERATION_RETRY_DELAY_SECONDS", "1")
    clock = [0.0]
    monkeypatch.setattr(generation.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(generation.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    return clock, [Document(page_content="The invoice total is 123.45 USD.", metadata={"chunk_id": "invoice", "pages": [1]})]


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_temporary_answer_failure_retries_with_the_remaining_deadline(retry_runtime, monkeypatch, code):
    clock, documents = retry_runtime
    calls, options, progress = [], [], []

    def invoke(messages):
        calls.append(messages)
        if len(calls) == 1:
            clock[0] += 3
            raise ProviderError(code)
        return AIMessage(content="123.45 USD [Source 1; page 1].")

    def model(*args, **kwargs):
        options.append(kwargs)
        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(generation, "get_generation_model", model)
    answer = generation.generate_final_answer(documents, "What is the invoice total?", provider="google",
                                               progress_callback=lambda *event: progress.append(event))
    assert "123.45 USD" in answer
    assert len(calls) == 2 and calls[0] == calls[1]
    assert options[1]["timeout_seconds"] == 56
    assert any(f"HTTP {code}" in event[-1] for event in progress)


@pytest.mark.parametrize("code", [400, 401, 403, 404, 429])
def test_authentication_quota_and_request_errors_do_not_retry(retry_runtime, monkeypatch, code):
    calls = []

    def invoke(messages):
        calls.append(messages)
        raise ProviderError(code)

    monkeypatch.setattr(generation, "get_generation_model", lambda *args, **kwargs: SimpleNamespace(invoke=invoke))
    with pytest.raises(generation.GenerationError, match=f"code {code}"):
        generation.generate_final_answer(retry_runtime[1], "What is the invoice total?", provider="google")
    assert len(calls) == 1


@pytest.mark.parametrize("code", [503, None])
def test_wrapped_structured_status_is_used_instead_of_error_prose(retry_runtime, monkeypatch, code):
    calls = []

    def invoke(messages):
        calls.append(messages)
        if len(calls) == 1:
            if code:
                try:
                    raise ProviderError(code)
                except ProviderError as underlying:
                    raise RuntimeError("Transport wrapper") from underlying
            raise RuntimeError("Prose mentions 503")
        return AIMessage(content="123.45 USD")

    monkeypatch.setattr(generation, "get_generation_model", lambda *args, **kwargs: SimpleNamespace(invoke=invoke))
    if code:
        assert generation.generate_final_answer(retry_runtime[1], "What is total?", provider="google") == "123.45 USD"
        assert len(calls) == 2
    else:
        with pytest.raises(generation.GenerationError):
            generation.generate_final_answer(retry_runtime[1], "What is total?", provider="google")
        assert len(calls) == 1


@pytest.mark.parametrize("retry_count", [0, 1, 2])
def test_answer_attempts_are_bounded(retry_runtime, monkeypatch, retry_count):
    monkeypatch.setenv("GENERATION_MAX_RETRIES", str(retry_count))
    calls = []

    def invoke(messages):
        calls.append(messages)
        raise ProviderError(503)

    monkeypatch.setattr(generation, "get_generation_model", lambda *args, **kwargs: SimpleNamespace(invoke=invoke))
    with pytest.raises(generation.GenerationError):
        generation.generate_final_answer(retry_runtime[1], "What is total?", provider="google")
    assert len(calls) == retry_count + 1


def test_retry_does_not_start_when_backoff_exceeds_remaining_time(retry_runtime, monkeypatch):
    clock, documents = retry_runtime
    calls = []

    def invoke(messages):
        calls.append(messages)
        clock[0] = 59.5
        raise ProviderError(503)

    monkeypatch.setattr(generation, "get_generation_model", lambda *args, **kwargs: SimpleNamespace(invoke=invoke))
    with pytest.raises(generation.GenerationError):
        generation.generate_final_answer(documents, "What is total?", provider="google")
    assert len(calls) == 1 and clock[0] == 59.5


@pytest.mark.parametrize("already_emitted", [False, True])
def test_stream_retries_only_before_any_text_is_emitted_and_closes_iterators(retry_runtime, monkeypatch, already_emitted):
    calls, closed, tokens = [], [], []

    def stream(messages):
        calls.append(messages)
        attempt = len(calls)
        try:
            if attempt == 1:
                if already_emitted:
                    yield AIMessage(content="Partial answer")
                raise ProviderError(503)
            yield AIMessage(content="123.45 USD")
        finally:
            closed.append(attempt)

    monkeypatch.setattr(generation, "get_generation_model", lambda *args, **kwargs: SimpleNamespace(stream=stream))
    if already_emitted:
        with pytest.raises(generation.GenerationError) as error:
            generation.generate_final_answer(retry_runtime[1], "What is total?", provider="google", token_callback=tokens.append)
        assert error.value.partial_answer == "Partial answer"
        assert tokens == ["Partial answer"] and len(calls) == 1
    else:
        assert generation.generate_final_answer(retry_runtime[1], "What is total?", provider="google",
                                                token_callback=tokens.append) == "123.45 USD"
        assert tokens == ["123.45 USD"] and len(calls) == 2
    assert closed == list(range(1, len(calls) + 1))


def test_stream_deadline_includes_retry_backoff(retry_runtime, monkeypatch):
    clock, documents = retry_runtime
    calls, closed = [], []

    def stream(messages):
        calls.append(messages)
        attempt = len(calls)
        try:
            if attempt == 1:
                clock[0] = 58
                raise ProviderError(503)
            clock[0] = 60.1
            yield AIMessage(content="Too late")
        finally:
            closed.append(attempt)

    monkeypatch.setattr(generation, "get_generation_model", lambda *args, **kwargs: SimpleNamespace(stream=stream))
    tokens = []
    with pytest.raises(generation.GenerationError, match="deadline"):
        generation.generate_final_answer(documents, "What is total?", provider="google", token_callback=tokens.append)
    assert len(calls) == 2 and closed == [1, 2] and tokens == []


def test_timeout_override_reaches_provider_client_without_extending_budget(monkeypatch):
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "60")
    captured = []
    monkeypatch.setattr(generation, "_build_model", lambda *args: captured.append(args) or object())
    generation.get_generation_model("gemini-3.1-flash-lite", provider="google", timeout_seconds=12.5)
    assert captured[0][2] == 12.5
    with pytest.raises(ValueError, match="Retry timeout"):
        generation.get_generation_model("gemini-3.1-flash-lite", provider="google", timeout_seconds=61)
