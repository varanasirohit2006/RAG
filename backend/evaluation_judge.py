"""Unified Ragas LLM-as-a-judge wrapper supporting Ollama, Google Gemini, and OpenAI-compatible models."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache

from backend.evaluation_data import validate_score
from backend.files import read_json, write_json
from backend.provider_limits import cache_root, limit_message, provider_status_code, provider_timeout

JUDGE_API_KEYS = {"google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
JUDGE_RUBRIC_VERSION = "partial-evidence-v2"
_request_usage = ContextVar("judge_request_usage", default=None)


class JudgeQuotaExceeded(RuntimeError):
    status_code = 429

    def __init__(self, limit=None):
        self.limit = limit
        super().__init__(limit_message(limit) if limit else "Judge rate or quota limit reached (HTTP 429).")


@dataclass(frozen=True)
class JudgeSettings:
    provider: str = "google"
    model: str = "gemini-3.1-flash-lite"
    timeout: int = 90
    max_retries: int = 1
    max_tokens: int = 4096
    reasoning_effort: str = "minimal"
    request_timeout: int | None = None

    def __post_init__(self):
        if self.provider not in {*JUDGE_API_KEYS, "ollama"}:
            raise ValueError("EVALUATION_PROVIDER must be google, openai, openrouter or ollama.")
        if not self.model:
            raise ValueError("EVALUATION_MODEL is required for judged evaluation.")
        if not 15 <= self.timeout <= 300 or not 0 <= self.max_retries <= 2 or not 1024 <= self.max_tokens <= 8192:
            raise ValueError("Invalid judge timeout, retry, or max_tokens setting.")

    @classmethod
    def from_env(cls):
        return cls(
            provider=os.getenv("EVALUATION_PROVIDER", "google"),
            model=os.getenv("EVALUATION_MODEL", "gemini-3.1-flash-lite"),
            timeout=int(os.getenv("EVALUATION_TIMEOUT", "90")),
            max_retries=int(os.getenv("EVALUATION_MAX_RETRIES", "1")),
            max_tokens=int(os.getenv("EVALUATION_MAX_TOKENS", "4096")),
            reasoning_effort=os.getenv("EVALUATION_REASONING_EFFORT", "minimal"),
            request_timeout=int(os.environ["EVALUATION_REQUEST_TIMEOUT"]) if os.getenv("EVALUATION_REQUEST_TIMEOUT") else None,
        )

    @property
    def transport_timeout(self):
        return self.request_timeout or self.timeout


def _parse_structured_text(content, response_model):
    from backend.generation_runtime import content_text

    text = re.sub(r"<think>.*?</think>", "", content_text(content), flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return response_model.model_validate_json(text)
    except ValueError:
        decoder = json.JSONDecoder()
        candidates = []
        for match in re.finditer(r"[\[{]", text):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
                candidates.append(response_model.model_validate(value))
            except (ValueError, TypeError):
                continue
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError("Judge did not return one complete schema-valid JSON object.")


def _build_judge_chat_model(settings):
    timeout = settings.transport_timeout
    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.model, temperature=0,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            num_predict=settings.max_tokens, num_ctx=max(8192, settings.max_tokens * 2),
            keep_alive="10m", client_kwargs={"timeout": timeout},
        )
    if settings.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        options = dict(model=settings.model, temperature=0, timeout=timeout, max_retries=1, max_output_tokens=settings.max_tokens)
        if settings.model.removeprefix("models/").startswith("gemini-3."):
            options.update(reasoning_effort=settings.reasoning_effort, include_thoughts=False)
        return ChatGoogleGenerativeAI(**options)
    from backend.generation_runtime import create_openai_generation_model, create_openrouter_generation_model
    if settings.provider == "openrouter":
        return create_openrouter_generation_model(settings.model, timeout, settings.max_tokens, asynchronous=True)
    return create_openai_generation_model(settings.model, timeout, settings.max_tokens, asynchronous=True)


def create_json_judge_llm(settings, *, model=None, response_cache=None):
    """Schema-validated JSON judging compatible with Ragas InstructorBaseRagasLLM."""
    from langchain_core.messages import HumanMessage
    from ragas.llms.base import InstructorBaseRagasLLM
    from backend.generation_runtime import close_ollama_model, ollama_json_schema

    owned = model is None
    if owned:
        model = _build_judge_chat_model(settings)
        response_cache = response_cache or (cache_root() / "judge_responses")

    class JSONJudgeLLM(InstructorBaseRagasLLM):
        def __init__(self):
            self.model = settings.model
            self.provider = settings.provider
            self.client = self
            self.is_async = True
            self.quota_exhausted = False
            self.limit = None
            self.usage = {
                "structured_generation_requests": 0, "json_repairs": 0, "provider_retries": 0,
                "request_seconds": 0.0, "token_usage_requests": 0, "input_tokens": 0,
                "output_tokens": 0, "total_tokens": 0, "quota_rejections": 0,
                "requests_blocked_by_quota": 0, "response_cache_hits": 0,
                "queue_wait_seconds": 0.0, "truncation_retries": 0, "resolved_models": [],
            }

        def generate(self, prompt, response_model):
            return asyncio.run(self.agenerate(prompt, response_model))

        async def agenerate(self, prompt, response_model):
            return await self.create(messages=[{"role": "user", "content": prompt}], response_model=response_model)

        def _map_provider_params(self):
            return {}

        async def create(self, *, messages, response_model, **kwargs):
            return await asyncio.wait_for(self._create(messages=messages, response_model=response_model), timeout=settings.timeout)

        async def _create(self, *, messages, response_model):
            schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
            instruction = (
                "Return exactly one valid JSON object matching the required schema. "
                "Do not include markdown or prose. Required JSON schema:\n" + schema
            )
            contents = [{"type": "text", "text": instruction}]
            for msg in messages:
                val = msg.get("content", "")
                contents.extend(val if isinstance(val, list) else [{"type": "text", "text": str(val)}])

            cache_path = None
            if response_cache:
                key = hashlib.sha256(json.dumps({
                    "provider": settings.provider, "model": settings.model,
                    "rubric": JUDGE_RUBRIC_VERSION, "messages": contents,
                }, sort_keys=True).encode()).hexdigest()
                cache_path = response_cache / f"{key}.json"
                saved = read_json(cache_path)
                if isinstance(saved, dict) and "response" in saved:
                    try:
                        res = response_model.model_validate(saved["response"])
                        self.usage["response_cache_hits"] += 1
                        if (tracker := _request_usage.get()) is not None:
                            tracker["provider_cache_hits"] += 1
                        return res
                    except ValueError:
                        pass

            last_error = None
            for attempt in range(settings.max_retries + 1):
                if self.quota_exhausted:
                    self.usage["requests_blocked_by_quota"] += 1
                    raise JudgeQuotaExceeded(self.limit)
                started = time.perf_counter()
                try:
                    self.usage["structured_generation_requests"] += 1
                    if (tracker := _request_usage.get()) is not None:
                        tracker["provider_requests"] += 1
                    options = {"format": ollama_json_schema(response_model)} if settings.provider == "ollama" else {}
                    response = await model.ainvoke([HumanMessage(content=contents)], **options)
                except Exception as exc:
                    if provider_status_code(exc) == 429:
                        self.quota_exhausted = True
                        self.usage["quota_rejections"] += 1
                    if attempt == settings.max_retries or (provider_status_code(exc) not in {500, 502, 503, 504} and not provider_timeout(exc)):
                        raise
                    self.usage["provider_retries"] += 1
                    await asyncio.sleep(0.5 * 2 ** attempt)
                    continue
                finally:
                    self.usage["request_seconds"] += time.perf_counter() - started

                meta = getattr(response, "usage_metadata", None) or {}
                if isinstance(meta, dict) and "total_tokens" in meta:
                    self.usage["token_usage_requests"] += 1
                    for k in ("input_tokens", "output_tokens", "total_tokens"):
                        if isinstance(meta.get(k), int):
                            self.usage[k] += meta[k]
                resp_meta = getattr(response, "response_metadata", None) or {}
                finish_reason = str(resp_meta.get("finish_reason") or resp_meta.get("done_reason", "")).upper()
                if finish_reason in {"MAX_TOKENS", "LENGTH"}:
                    if attempt < settings.max_retries:
                        self.usage["truncation_retries"] += 1
                        contents.append({"type": "text", "text": "The prior response was cut off. Redraft the complete JSON object concisely."})
                        continue
                    raise ValueError("Judge output was truncated; increase EVALUATION_MAX_TOKENS.")
                try:
                    result = _parse_structured_text(response.content, response_model)
                    if cache_path:
                        write_json(cache_path, {"response": result.model_dump(mode="json")})
                    return result
                except ValueError as exc:
                    last_error = exc
                    if attempt == settings.max_retries:
                        break
                    self.usage["json_repairs"] += 1
                    contents.append({"type": "text", "text": "Retry with ONLY the valid JSON object matching the schema."})
            raise ValueError(f"Judge JSON validation failed: {last_error}")

        async def close(self):
            if settings.provider == "ollama":
                await close_ollama_model(model)
                return
            for attr in ("http_async_client", "root_async_client"):
                client = getattr(model, attr, None)
                if client is not None and hasattr(client, "aclose"):
                    await client.aclose()

    return JSONJudgeLLM()


class RagasJudge:
    """Async context manager / wrapper for Ragas evaluation metrics."""

    def __init__(self, settings: JudgeSettings, *, embedding_model=None):
        self.settings = settings
        self.embedding_model = embedding_model
        self.client = None
        self.llm = None
        self.metrics = {}

    def prepare(self):
        from ragas.metrics.collections import (
            AnswerRelevancy, ContextPrecisionWithReference, ContextRecall,
            FactualCorrectness, Faithfulness, MultiModalFaithfulness,
        )

        key_name = JUDGE_API_KEYS.get(self.settings.provider)
        if key_name and not os.getenv(key_name):
            raise ValueError(f"{key_name} is required for judged evaluation.")
        self.llm = create_json_judge_llm(self.settings)
        self.client = self.llm
        self.metrics = {
            "faithfulness": Faithfulness(llm=self.llm),
            "multimodal_faithfulness": MultiModalFaithfulness(llm=self.llm),
            "factual_correctness": FactualCorrectness(llm=self.llm),
            "context_precision": ContextPrecisionWithReference(llm=self.llm),
            "context_recall": ContextRecall(llm=self.llm),
        }
        configure_context_precision(self.metrics["context_precision"])

        if self.embedding_model is not None:
            from ragas.embeddings.base import BaseRagasEmbedding
            emb = self.embedding_model

            class LocalEmbeddingAdapter(BaseRagasEmbedding):
                @lru_cache(maxsize=2048)
                def embed_text(self, text, **kwargs):
                    return emb.embed_query(text)

                async def aembed_text(self, text, **kwargs):
                    return await asyncio.to_thread(self.embed_text, text)

                def embed_texts(self, texts, **kwargs):
                    return emb.embed_documents(texts)

                async def aembed_texts(self, texts, **kwargs):
                    return await asyncio.to_thread(emb.embed_documents, texts)

            self.metrics["answer_relevancy"] = AnswerRelevancy(llm=self.llm, embeddings=LocalEmbeddingAdapter(), strictness=3)

    async def score(self, name, *, question, response, reference, contexts, image_contexts=()):
        usage = {"provider_requests": 0, "provider_cache_hits": 0}
        token = _request_usage.set(usage)
        try:
            res = await self._score(name, question=question, response=response, reference=reference, contexts=contexts, image_contexts=image_contexts)
            return {**res, **usage}
        except BaseException as exc:
            exc.judge_request_usage = usage
            raise
        finally:
            _request_usage.reset(token)

    async def _score(self, name, *, question, response, reference, contexts, image_contexts=()):
        if self.llm is None:
            raise RuntimeError("Prepare the judge before scoring.")
        if name == "abstention_accuracy":
            from pydantic import BaseModel
            class AbstentionVerdict(BaseModel):
                abstained: bool
                reason: str
            prompt = f"Set abstained=true if the response clearly states the source cannot answer.\nQuestion: {question}\nResponse: {response}"
            result = await self.llm.agenerate(prompt, AbstentionVerdict)
            return {"value": float(result.abstained), "reason": result.reason}

        metric = self.metrics[name]
        if name == "faithfulness":
            result = await metric.ascore(user_input=question, response=response, retrieved_contexts=list(contexts))
        elif name == "multimodal_faithfulness":
            result = await metric.ascore(response=response, retrieved_contexts=list(contexts) + list(image_contexts))
        elif name == "factual_correctness":
            result = await metric.ascore(response=response, reference=reference)
        elif name in {"context_precision", "context_recall"}:
            result = await metric.ascore(user_input=question, reference=reference, retrieved_contexts=list(contexts))
        elif name == "answer_relevancy":
            result = await metric.ascore(user_input=question, response=response)
        else:
            raise ValueError(f"Unknown evaluation metric: {name}")
        return {"value": validate_score(name, result.value), "reason": getattr(result, "reason", None)}

    async def close(self):
        if self.client is not None:
            await self.client.close()


def configure_context_precision(metric):
    """Allow partial support: a chunk supporting any material fact in a multi-part answer gets verdict=1."""
    prompt = metric.prompt
    prompt.instruction = (
        "Judge whether this context directly supports ANY material fact in the reference answer. "
        "Return verdict=1 if it supports at least one fact, even if other passages hold the rest. "
        "Return 0 only for unrelated or contradicting evidence. Give a brief reason and valid JSON."
    )
    examples = [
        ("What are the invoice total and payment deadline?", "Invoice A total is 7 USD.",
         "Invoice A totals 7 USD and payment is due in 30 days.", 1, "Supports the total."),
        ("What are the invoice total and payment deadline?", "Invoice B total is 9 USD.",
         "Invoice A totals 7 USD and payment is due in 30 days.", 0, "Different invoice."),
    ]
    prompt.examples = [(prompt.input_model(question=q, context=c, answer=a), prompt.output_model(verdict=v, reason=r)) for q, c, a, v, r in examples]
