import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from backend.evaluation_judge import JudgeSettings, RagasJudge, create_ollama_json_llm
from backend.generation_runtime import ollama_json_schema


class Verdict(BaseModel):
    supported: bool
    reason: str


class Model:
    def __init__(self, content='{"supported":true,"reason":"Supported by source."}', done_reason="stop"):
        self.content, self.done_reason = content, done_reason
        self.requests = []

    async def ainvoke(self, messages, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(content=self.content, response_metadata={"done_reason": self.done_reason},
                               usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30})


def disable_remote_clients(monkeypatch):
    import langchain_google_genai
    import langchain_openai
    import openai

    def unexpected(**kwargs):
        pytest.fail("Local evaluation attempted to construct a remote client")

    for name in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", unexpected)
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", unexpected)
    monkeypatch.setattr(openai, "AsyncOpenAI", unexpected)


def test_local_verifier_and_scorer_use_requested_model_without_remote_credentials(monkeypatch):
    import langchain_ollama

    disable_remote_clients(monkeypatch)
    monkeypatch.setenv("EVALUATION_PROVIDER", "ollama")
    monkeypatch.setenv("EVALUATION_MODEL", "gemma3:270m")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    options, models = [], []

    def build(**kwargs):
        options.append(kwargs)
        model = Model()
        models.append(model)
        return model

    monkeypatch.setattr(langchain_ollama, "ChatOllama", build)
    judge = RagasJudge(JudgeSettings.from_env())
    judge.prepare()
    assert not models[0].requests
    assert {"faithfulness", "context_precision", "context_recall", "factual_correctness"} <= judge.metrics.keys()
    assert judge.llm.provider == "ollama" and judge.llm.model == "gemma3:270m"
    assert options[0]["model"] == "gemma3:270m"
    assert options[0]["base_url"] == "http://localhost:11434"
    assert options[0]["client_kwargs"] == {"timeout": judge.settings.timeout}
    assert options[0]["num_predict"] == judge.settings.max_tokens
    assert asyncio.run(judge.llm.agenerate("Verify source evidence.", Verdict)).supported
    assert models[0].requests[0]["format"] == ollama_json_schema(Verdict)
    assert judge.llm.usage["total_tokens"] == 30
    models[0].content = '{"abstained":true,"reason":"No answer in the source."}'
    score = asyncio.run(judge.score("abstention_accuracy", question="Unknown?", response="The source does not say.",
                                   reference="", contexts=[]))
    assert score["value"] == 1.0
    asyncio.run(judge.close())


@pytest.mark.parametrize("content,done_reason,error", [
    ('{"supported":true,"reason":"Complete-looking but truncated."}', "length", "truncated"),
    ('{"supported":true}', "stop", "JSON validation failed"),
    ('not JSON', "stop", "JSON validation failed"),
])
def test_local_invalid_or_truncated_verdicts_are_never_scored(content, done_reason, error):
    model = Model(content, done_reason)
    llm = create_ollama_json_llm(JudgeSettings(provider="ollama", model="gemma3:270m", max_retries=0), model=model)
    with pytest.raises(ValueError, match=error):
        asyncio.run(llm.agenerate("Verify source evidence.", Verdict))
    assert len(model.requests) == 1


def test_local_verifier_and_scoring_transports_are_isolated_and_closed(monkeypatch):
    import langchain_ollama

    original = langchain_ollama.ChatOllama
    models = []

    def build(**kwargs):
        model = original(**kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(langchain_ollama, "ChatOllama", build)
    settings = JudgeSettings(provider="ollama", model="gemma3:270m")

    async def check():
        verifier = create_ollama_json_llm(settings)
        scorer = create_ollama_json_llm(settings)
        clients = [model._async_client._client for model in models]
        assert clients[0] is not clients[1]
        await verifier.close()
        assert clients[0].is_closed and models[0]._client._client.is_closed
        assert not clients[1].is_closed and not models[1]._client._client.is_closed
        await scorer.close()
        assert clients[1].is_closed and models[1]._client._client.is_closed

    asyncio.run(check())


def test_automatic_drafting_and_source_verification_work_without_remote_clients(monkeypatch, tmp_path):
    import dotenv
    import langchain_ollama
    from langchain_core.documents import Document
    from backend import evaluation_generation as generation

    disable_remote_clients(monkeypatch)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: False)
    for name in ("GENERATION_PROVIDER", "SYNTHETIC_GENERATION_PROVIDER", "EVALUATION_PROVIDER"):
        monkeypatch.setenv(name, "ollama")
    for name in ("LOCAL_GENERATION_MODEL", "GENERATION_MODEL", "SYNTHETIC_GENERATION_MODEL", "EVALUATION_MODEL"):
        monkeypatch.setenv(name, "gemma3:270m")
    monkeypatch.setenv("SYNTHETIC_ALLOW_SAME_MODEL_VERIFICATION", "true")
    text = "The invoice total is 123.45 USD. Payment is due within thirty calendar days."
    document_hash = "a" * 64

    async def sources(*args):
        return [Document(page_content=text, metadata={"chunk_id": "invoice", "pages": [1]})], {"document_hash": document_hash}

    monkeypatch.setattr(generation, "_load_sources", sources)
    options, models = [], []
    draft = ('{"question":"What is the invoice total?","reference_answer":"123.45 USD",'
             '"category":"factual","supporting_quotes":[{"chunk_id":"invoice","quote":"' + text + '"}]}')
    verdict = ('{"supported":true,"unambiguous":true,"all_evidence_relevant":true,'
               '"category_valid":true,"reason":"The amount appears in the exact quote."}')

    def build(**kwargs):
        options.append(kwargs)
        model = Model(draft if not models else verdict)
        models.append(model)
        return model

    monkeypatch.setattr(langchain_ollama, "ChatOllama", build)
    dataset = asyncio.run(generation.generate_evaluation_dataset(tmp_path / "index", expected_document_hash=document_hash,
                         question_count=1, output_dir=tmp_path / "datasets"))
    assert len(dataset["questions"]) == 1
    assert dataset["questions"][0]["verification"]["supported"]
    assert models[0].requests[0]["format"] == ollama_json_schema(generation.DraftQuestion)
    assert models[1].requests[0]["format"] == ollama_json_schema(generation.SourceVerdict)
    assert all(option["model"] == "gemma3:270m" for option in options)
    assert dataset["provenance"]["configuration"]["independent_reference_verification"] is False
    assert any("same model drafts and verifies" in warning for warning in dataset["warnings"])


@pytest.mark.parametrize("model", ["gemma3:270m", "gemma3:1b", "gemma3:270m-it-q4_K_M"])
def test_text_only_gemma_models_omit_images_and_preserve_extracted_text(model):
    from langchain_core.documents import Document
    from backend.generation_runtime import prepare_generation_context

    document = Document(page_content="Invoice total: 123.45 USD.",
                        metadata={"chunk_id": "invoice", "original_content": '{"raw_text":"Invoice total: 123.45 USD.","images_base64":["unused-image"]}'})
    context = prepare_generation_context([document], "What is the total?", provider="ollama", model_name=model)
    assert context["visual_evidence_omitted"]
    assert not context["images_base64"]
    assert "123.45 USD" in context["prompt"]
    assert all(part["type"] == "text" for part in context["messages"][0].content)


def test_local_generation_detects_ollama_output_limit():
    from backend.generation_runtime import _finish_reason

    assert _finish_reason(SimpleNamespace(response_metadata={"done_reason": "length"})) == "LENGTH"


def test_local_decoding_schema_preserves_structure_without_large_string_repetitions():
    from backend.evaluation_generation import DraftQuestion

    original = DraftQuestion.model_json_schema()
    grammar = ollama_json_schema(DraftQuestion)
    assert grammar["required"] == original["required"]
    assert grammar["additionalProperties"] is False
    assert grammar["properties"]["category"] == original["properties"]["category"]
    assert grammar["properties"]["supporting_quotes"] == original["properties"]["supporting_quotes"]
    assert grammar["$defs"]["SupportingQuote"]["required"] == original["$defs"]["SupportingQuote"]["required"]
    assert "maxLength" not in grammar["properties"]["reference_answer"]
    assert "minLength" not in grammar["properties"]["question"]
    assert "maxLength" not in grammar["$defs"]["SupportingQuote"]["properties"]["quote"]
    assert DraftQuestion.model_json_schema() == original
    assert original["properties"]["reference_answer"]["maxLength"] == 6000


@pytest.mark.parametrize("reason", ["", "x" * 1201])
def test_local_verdict_still_enforces_string_bounds_after_decoding(reason):
    import json
    from backend.evaluation_generation import SourceVerdict

    content = json.dumps({"supported": True, "unambiguous": True, "all_evidence_relevant": True,
                          "category_valid": True, "reason": reason})
    model = Model(content)
    llm = create_ollama_json_llm(JudgeSettings(provider="ollama", model="gemma3:270m", max_retries=0), model=model)
    with pytest.raises(ValueError, match="JSON validation failed"):
        asyncio.run(llm.agenerate("Verify source evidence.", SourceVerdict))
    assert "maxLength" not in model.requests[0]["format"]["properties"]["reason"]
