from types import SimpleNamespace

import pytest

from backend import provider_limits as limits


def test_daily_limit_preserves_provider_reset_and_never_stores_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'private-test-secret')
    monkeypatch.setattr(limits.time, 'time', lambda: 1000)
    body = {'error': {'message': 'Rate limit exceeded: free-models-per-day', 'metadata': {
        'limit_source': 'openrouter_free_tier_daily', 'headers': {'X-RateLimit-Limit': '50',
            'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '2000000'}}}}
    # Real provider reset values are epoch milliseconds (modern dates).
    body['error']['metadata']['headers']['X-RateLimit-Reset'] = '1795046400000'
    value = limits.record_openrouter_limit(SimpleNamespace(body=body))
    assert value['scope'] == 'daily' and value['limit'] == 50 and value['remaining'] == 0
    assert value['reset_at'] == '2026-11-19T00:00:00+00:00'
    with pytest.raises(limits.OpenRouterLimitExceeded):
        limits.ensure_openrouter_available()
    stored = list(limits.cache_root().rglob('*.json'))
    assert len(stored) == 1 and 'private-test-secret' not in stored[0].read_text()
    monkeypatch.setattr(limits.time, 'time', lambda: 1795046401)
    assert limits.current_openrouter_limit() is None


def test_unknown_limit_does_not_invent_or_persist_a_reset(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    value = limits.record_openrouter_limit(SimpleNamespace(body={'message': 'Rate limited'}))
    assert value['reset_at'] is None and not list(limits.cache_root().rglob('*.json'))


def test_provider_names_are_preserved_in_limit_messages():
    assert 'google' in limits.limit_message({'provider': 'google'})
    assert 'OpenRouter' not in limits.limit_message({'provider': 'google'})
