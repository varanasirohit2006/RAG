"""Bounded availability probes before uncached proxy evaluation work."""
import hashlib
import json
import os
import time

from backend.generation_runtime import openai_base_url
from backend.provider_limits import provider_status_code, provider_timeout

_ready = {}


class ProviderUnavailable(RuntimeError):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


async def check_configured_models(models):
    if os.getenv('EVALUATION_PROVIDER_PREFLIGHT', 'false').lower() != 'true':
        return []
    if not os.getenv('OPENAI_BASE_URL', '').strip():
        return []
    from openai import AsyncOpenAI
    import httpx

    checks = []
    for provider, model in dict.fromkeys(models):
        if provider != 'openai':
            continue
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ProviderUnavailable('OPENAI_API_KEY is required for the configured proxy.')
        identity = (openai_base_url(), model, hashlib.sha256(api_key.encode()).hexdigest())
        cached = _ready.get(identity)
        if cached and time.monotonic() - cached[0] < 300:
            checks.append({**cached[1], 'cache_reused': True})
            continue
        try:
            async with AsyncOpenAI(api_key=api_key, base_url=openai_base_url(), timeout=20, max_retries=0,
                                   http_client=httpx.AsyncClient(timeout=20)) as client:
                reply = await client.chat.completions.create(model=model, temperature=0, max_tokens=64,
                    messages=[{'role': 'user', 'content': 'Return exactly {"ready":true} as JSON, without analysis or explanation.'}])
            if not reply.choices or reply.choices[0].finish_reason != 'stop' or json.loads(reply.choices[0].message.content or '').get('ready') is not True:
                raise ProviderUnavailable(f'{model} did not return a complete JSON response to the short availability check.')
        except ProviderUnavailable:
            raise
        except Exception as error:
            code = provider_status_code(error)
            if code == 429:
                hint = 'is rate/quota limited (HTTP 429). Check its upstream limit in FreeLLMAPI and retry after that limit resets.'
            elif code in {401, 403, 404, 503}:
                hint = f'is unavailable (HTTP {code}). Check that FreeLLMAPI has a usable upstream provider key and route for this fixed model.'
            elif provider_timeout(error):
                hint = 'did not answer the 20-second availability check. Check the proxy and upstream provider before starting a long run.'
            else:
                hint = f'failed the availability check ({type(error).__name__}). Check the proxy model configuration.'
            raise ProviderUnavailable(f'{model} {hint} Answer generation and scoring were not started.', code) from error
        check = {'model': model, 'resolved_model': reply.model, 'cache_reused': False}
        if len(_ready) >= 16:
            _ready.pop(next(iter(_ready)))
        _ready[identity] = (time.monotonic(), check)
        checks.append(check)
    return checks
