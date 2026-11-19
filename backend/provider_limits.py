"""Remember provider-reported free limits without persisting credentials or prose."""
import hashlib
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.files import PROJECT_ROOT, read_json, write_json

_lock = threading.Lock()


def provider_status_code(exc):
    """Read structured provider codes through wrappers; never infer them from prose."""
    seen = set()
    while exc is not None and id(exc) not in seen and len(seen) < 8:
        seen.add(id(exc))
        for value in (getattr(exc, "code", None), getattr(exc, "status_code", None),
                      getattr(getattr(exc, "response", None), "status_code", None)):
            if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
                return value
        exc = exc.__cause__ or exc.__context__
    return None


def cache_root():
    return Path(os.getenv('RAG_EVALUATION_CACHE_DIR', str(PROJECT_ROOT / 'evaluation_runs' / 'cache')))


def _state_path():
    key = os.getenv('OPENROUTER_API_KEY', '')
    if not key:
        return None
    fingerprint = hashlib.sha256(key.encode()).hexdigest()
    return cache_root() / 'provider_limits' / (fingerprint + '.json')


def _number(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def limit_from_body(body, headers=None, *, now=None):
    """Use the provider's reset timestamp; never invent a daily reset."""
    now = time.time() if now is None else now
    body = body if isinstance(body, dict) else {}
    error = body.get('error', body)
    error = error if isinstance(error, dict) else {}
    metadata = error.get('metadata', {})
    metadata = metadata if isinstance(metadata, dict) else {}
    headers = {str(k).lower(): v for k, v in {**(headers or {}), **(metadata.get('headers') or {})}.items()}
    reset = _number(headers.get('x-ratelimit-reset'))
    if reset and reset > 10**11:
        reset /= 1000
    retry_after = _number(headers.get('retry-after'))
    if not reset and retry_after is not None:
        reset = now + max(0, retry_after)
    daily = metadata.get('limit_source') == 'openrouter_free_tier_daily' or 'free-models-per-day' in str(error.get('message', ''))
    return {'provider': 'openrouter', 'status_code': 429, 'scope': 'daily' if daily else 'rate_or_quota',
            'limit': _number(headers.get('x-ratelimit-limit')),
            'remaining': _number(headers.get('x-ratelimit-remaining')),
            'reset_at': datetime.fromtimestamp(reset, timezone.utc).isoformat() if reset else None}


def record_openrouter_limit(exc):
    value = getattr(exc, 'limit', None)
    if not isinstance(value, dict):
        response = getattr(exc, 'response', None)
        value = limit_from_body(getattr(exc, 'body', None), getattr(response, 'headers', None))
    path = _state_path()
    if path and value.get('reset_at'):
        with _lock:
            write_json(path, value)
    return value


def current_openrouter_limit():
    path = _state_path()
    value = read_json(path) if path else None
    if not isinstance(value, dict) or not value.get('reset_at'):
        return None
    try:
        return value if datetime.fromisoformat(value['reset_at']).timestamp() > time.time() else None
    except (ValueError, TypeError):
        return None


def limit_message(value):
    value = value or {}
    message = 'OpenRouter daily free-model request allowance exhausted' if value.get('scope') == 'daily' else 'OpenRouter rate or quota limit reached'
    if value.get('provider') and value['provider'] != 'openrouter':
        message = value['provider'] + ' rate or quota limit reached'
    if value.get('limit') is not None:
        message += f" ({value['limit']} allowed, {value.get('remaining', 0)} remaining)"
    if value.get('reset_at'):
        message += f". Provider reset: {value['reset_at']}"
    return message + '. Completed work is retained; retry missing scores after the limit resets.'


class OpenRouterLimitExceeded(RuntimeError):
    status_code = 429

    def __init__(self, value):
        self.limit = value
        super().__init__(limit_message(value))


def ensure_openrouter_available():
    value = current_openrouter_limit()
    if value:
        raise OpenRouterLimitExceeded(value)
