"""Unified LLM client factory, multimodal prompt assembly, and grounded answer generation."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from backend.provider_limits import provider_status_code, provider_timeout

DEFAULT_GENERATION_MODEL = "gemini-3.1-flash-lite"


class GenerationError(RuntimeError):
    def __init__(self, message, *, partial_answer=None):
        super().__init__(message)
        self.partial_answer = partial_answer


def resolve_provider(provider=None):
    value = (provider or os.getenv("GENERATION_PROVIDER", "google")).strip().lower()
    if value not in {"google", "openai", "openrouter", "ollama"}:
        raise ValueError("GENERATION_PROVIDER must be google, openai, openrouter, or ollama.")
    return value


def openai_base_url():
    return (os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1").rstrip("/")


def model_name_for(provider=None, model_name=None):
    provider = resolve_provider(provider)
    if model_name:
        return model_name
    if provider == "ollama":
        return os.getenv("LOCAL_GENERATION_MODEL") or os.getenv("GENERATION_MODEL", "qwen3-vl:2b-instruct")
    return os.getenv("GENERATION_MODEL", DEFAULT_GENERATION_MODEL)


def generation_supports_images(provider=None, model_name=None):
    provider = resolve_provider(provider)
    if provider == "ollama":
        name = model_name_for(provider, model_name).lower().rsplit("/", 1)[-1]
        return not name.startswith(("gemma3:270m", "gemma3:1b"))
    return provider == "google"


async def close_ollama_model(model):
    for attr in ("_async_client", "_client"):
        client = getattr(getattr(model, attr, None), "_client", None)
        if client is not None:
            if hasattr(client, "aclose"):
                await client.aclose()
            elif hasattr(client, "close"):
                client.close()


def ollama_json_schema(response_model):
    """Strip string length bounds from Pydantic JSON schema for Ollama grammar compatibility."""
    def strip_bounds(value):
        if isinstance(value, dict):
            return {k: strip_bounds(v) for k, v in value.items() if k not in {"minLength", "maxLength"}}
        if isinstance(value, list):
            return [strip_bounds(item) for item in value]
        return value
    return strip_bounds(response_model.model_json_schema())


def create_openai_generation_model(model_name, timeout, max_tokens, *, asynchronous=False, base_url=None):
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI-compatible requests.")
    kwargs = {
        "model": model_name, "api_key": api_key, "base_url": base_url or openai_base_url(),
        "temperature": 0, "timeout": timeout, "max_retries": 0, "max_tokens": max_tokens,
    }
    if asynchronous:
        import httpx
        kwargs["http_async_client"] = httpx.AsyncClient(timeout=timeout)
    return ChatOpenAI(**kwargs)


def create_openrouter_generation_model(model_name, timeout, max_tokens, *, asynchronous=False):
    return create_openai_generation_model(
        model_name, timeout, max_tokens, asynchronous=asynchronous, base_url="https://openrouter.ai/api/v1"
    )


def _build_model(provider, model_name, timeout, max_output_tokens, thinking_budget=0, base_url=None):
    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for Google Gemini generation.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {
            "model": model_name, "google_api_key": api_key,
            "temperature": 0, "timeout": timeout, "max_retries": 0, "max_output_tokens": max_output_tokens,
        }
        normalized = model_name.lower().rsplit("/", 1)[-1]
        if normalized.startswith("gemini-2.5-flash") and "thinking_budget" in ChatGoogleGenerativeAI.model_fields:
            kwargs["thinking_budget"] = thinking_budget
        return ChatGoogleGenerativeAI(**kwargs)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name, temperature=0,
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            num_predict=max_output_tokens, num_ctx=4096, keep_alive="10m",
            client_kwargs={"timeout": timeout},
        )
    if provider == "openrouter":
        return create_openrouter_generation_model(model_name, timeout, max_output_tokens)
    return create_openai_generation_model(model_name, timeout, max_output_tokens, base_url=base_url)


def get_generation_model(model_name=None, *, provider=None, timeout_seconds=None):
    provider = resolve_provider(provider)
    timeout = timeout_seconds or int(os.getenv("GENERATION_REQUEST_TIMEOUT_SECONDS", os.getenv("MODEL_TIMEOUT_SECONDS", "60")))
    max_tokens = int(os.getenv("GENERATION_MAX_OUTPUT_TOKENS", "2048"))
    thinking_budget = int(os.getenv("GENERATION_THINKING_BUDGET", "0"))
    return _build_model(provider, model_name_for(provider, model_name), timeout, max_tokens, thinking_budget)


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item if isinstance(item, str) else item.get("text", "")
            for item in content if isinstance(item, (dict, str)) and (not isinstance(item, dict) or not item.get("thinking"))
        )
    return str(content or "")


def contextualize_question(query, chat_history=None, *, model=None):
    """Rewrite a follow-up question into a standalone query (from rag-for-beginners/4_history_aware_generation.py)."""
    if not chat_history:
        return query
    llm = model or get_generation_model()
    messages = [
        SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        *chat_history,
        HumanMessage(content=f"New question: {query}"),
    ]
    return content_text(llm.invoke(messages).content).strip() or query


def document_payload(document):
    original = document.metadata.get("original_content")
    if isinstance(original, str):
        try:
            original = json.loads(original)
        except (json.JSONDecodeError, TypeError):
            original = None
    original = original if isinstance(original, dict) else {}
    return {
        "raw_text": original.get("raw_text", document.page_content),
        "tables_html": original.get("tables_html", []),
        "images_base64": original.get("images_base64", []),
        "image_paths": original.get("image_paths", []),
    }


def document_key(document):
    return document.metadata.get("chunk_id") or hashlib.sha256(
        (document.page_content + json.dumps(document.metadata, sort_keys=True, default=str)).encode()
    ).hexdigest()


def _pages(document):
    pages = document.metadata.get("pages", document.metadata.get("page", ""))
    if isinstance(pages, str):
        try:
            decoded = json.loads(pages)
            if isinstance(decoded, list):
                return ", ".join(str(p) for p in decoded)
        except json.JSONDecodeError:
            pass
    return str(pages or "unknown")


def _resized_jpeg(data):
    from PIL import Image

    max_dim = int(os.getenv("GENERATION_IMAGE_MAX_DIM", "1280"))
    with Image.open(io.BytesIO(data)) as image:
        image.thumbnail((max_dim, max_dim))
        with io.BytesIO() as output:
            image.convert("RGB").save(output, "JPEG", quality=80)
            return base64.b64encode(output.getvalue()).decode("ascii")


def prepare_generation_context(chunks, query, *, provider=None, model_name=None):
    """Build multimodal prompt context (text, HTML tables, and images) from retrieved chunks."""
    provider = resolve_provider(provider)
    if not query or not query.strip():
        raise ValueError("Enter a non-empty question.")
    max_context = int(os.getenv("GENERATION_MAX_CONTEXT_CHARS", "12000"))
    max_images = int(os.getenv("GENERATION_MAX_IMAGES", "3"))
    supports_images = generation_supports_images(provider, model_name)

    contexts, chunk_ids, image_paths, images_base64 = [], [], [], []
    remaining = max_context
    for rank, doc in enumerate(chunks, 1):
        if remaining <= 0:
            break
        payload = document_payload(doc)
        source = "Combined PDF" if doc.metadata.get("source_pages") else (doc.metadata.get("source") or "document")
        header = (f"[Source {rank}; Combined PDF; combined pages {_pages(doc)}]"
                  if doc.metadata.get("source_pages") else f"[Source {rank}; {source}; pages {_pages(doc)}]")
        tables = "\n".join(str(t) for t in payload["tables_html"])
        evidence = str(payload["raw_text"] or "") + ("\nTABLE STRUCTURE:\n" + tables if tables else "")
        block = (header + "\n" + evidence)[:remaining]
        contexts.append(block)
        chunk_ids.append(document_key(doc))
        remaining -= len(block)

        if not supports_images:
            continue
        for img_path in payload["image_paths"]:
            if len(images_base64) >= max_images:
                break
            path = Path(img_path)
            if not path.is_absolute():
                asset_dir = doc.metadata.get("asset_directory")
                if not asset_dir:
                    continue
                path = Path(asset_dir) / path
            images_base64.append(_resized_jpeg(path.read_bytes()))
            image_paths.append(str(path.resolve()))
        for b64 in payload["images_base64"]:
            if len(images_base64) >= max_images:
                break
            data = base64.b64decode(b64, validate=True)
            images_base64.append(_resized_jpeg(data))
            image_paths.append("inline:" + hashlib.sha256(data).hexdigest())

    prompt = (
        "Answer the question using ONLY the supplied document evidence and images. "
        "Give a concise answer and cite source numbers and page numbers. Preserve exact numbers and units. "
        "If the evidence does not support an answer, say so; do not invent facts.\n\n"
        f"QUESTION: {query.strip()}\n\nEVIDENCE:\n" + "\n\n".join(contexts)
    )
    if any(doc.metadata.get("source_pages") for doc in chunks):
        prompt += "\n\nUse the displayed combined-PDF page numbers in citations. The source viewer maps them to the original PDF pages."
    message_content = [{"type": "text", "text": prompt}]
    message_content.extend({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in images_base64)
    return {
        "provider": provider, "prompt": prompt, "text_contexts": contexts,
        "chunk_ids": chunk_ids, "image_paths": image_paths, "images_base64": images_base64,
        "visual_evidence_omitted": False, "messages": [HumanMessage(content=message_content)],
    }


def generate_final_answer(chunks, query, *, provider=None, model_name=None,
                          progress_callback=None, token_callback=None, metadata_callback=None):
    """Generate a grounded multimodal answer with streaming and bounded transient retry."""
    provider = resolve_provider(provider)
    prepared = prepare_generation_context(chunks, query, provider=provider, model_name=model_name)
    if not prepared["text_contexts"] and not prepared["images_base64"]:
        return "The document evidence does not support an answer to this question."
    if progress_callback:
        progress_callback(0, 1, f"Generating with {model_name_for(provider, model_name)}...")

    model = get_generation_model(model_name, provider=provider)
    max_retries = int(os.getenv("GENERATION_MAX_RETRIES", "1"))
    parts, chunk, response = [], None, None
    for attempt in range(max_retries + 1):
        try:
            if token_callback:
                for chunk in model.stream(prepared["messages"]):
                    text = content_text(chunk.content)
                    if text:
                        parts.append(text)
                        token_callback(text)
                answer = "".join(parts)
            else:
                response = model.invoke(prepared["messages"])
                answer = content_text(response.content)
            break
        except Exception as exc:
            code = provider_status_code(exc)
            if parts or attempt == max_retries or (code not in {500, 502, 503, 504} and not provider_timeout(exc)):
                raise GenerationError(
                    f"{provider} generation failed ({type(exc).__name__}).",
                    partial_answer="".join(parts) if parts else None,
                ) from exc
            time.sleep(float(os.getenv("GENERATION_RETRY_DELAY_SECONDS", "1")))

    if not answer.strip():
        raise GenerationError("The generation provider returned no answer text.")
    if metadata_callback:
        final_msg = chunk if token_callback else response
        meta = getattr(final_msg, "response_metadata", None) or {}
        metadata_callback({"requested_model": model_name_for(provider, model_name),
                           "resolved_model": meta.get("model_name") or meta.get("model"),
                           "usage": getattr(final_msg, "usage_metadata", None) or {}})
    if progress_callback:
        progress_callback(1, 1, "Answer complete.")
    return answer
