"""Simple provider HTTP status, timeout, and rate-limit helpers."""
import os
from pathlib import Path
from backend.files import PROJECT_ROOT


def cache_root():
    configured = os.getenv('RAG_EVALUATION_CACHE_DIR')
    return Path(configured).resolve() if configured else PROJECT_ROOT / 'evaluation_runs' / 'cache'


def provider_status_code(exc):
    for item in (exc, getattr(exc, '__cause__', None), getattr(exc, '__context__', None)):
        if item is None:
            continue
        code = getattr(item, 'status_code', None) or getattr(getattr(item, 'response', None), 'status_code', None)
        if code is not None:
            return int(code)
    text = str(exc)
    for code in (429, 503, 504, 502, 500, 401, 403, 404):
        if f' {code}' in text or f'code: {code}' in text or f'HTTP {code}' in text:
            return code
    return None


def provider_timeout(exc):
    for item in (exc, getattr(exc, '__cause__', None), getattr(exc, '__context__', None)):
        if item is not None and ('timeout' in type(item).__name__.lower() or 'timed out' in str(item).lower()):
            return True
    return False


def limit_from_body(body=None, headers=None):
    return {}


def record_openrouter_limit(exc):
    return getattr(exc, 'limit', None) or {}


def ensure_openrouter_available():
    return None


def limit_message(value=None):
    provider = (value or {}).get('provider') or 'Provider'
    return f'{provider} rate or quota limit reached (HTTP 429). Completed work is retained; retry after the limit resets.'


class OpenRouterLimitExceeded(RuntimeError):
    status_code = 429

    def __init__(self, value=None):
        self.limit = value or {}
        super().__init__(limit_message(self.limit))


class ProviderUnavailable(RuntimeError):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


async def check_configured_models(models):
    return []
