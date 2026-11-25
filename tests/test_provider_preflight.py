import asyncio
import json

import httpx
import pytest

from backend import provider_preflight


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'private-preflight-test-key')
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://127.0.0.1:31415/v1')
    monkeypatch.setenv('EVALUATION_PROVIDER_PREFLIGHT', 'true')
    provider_preflight._ready.clear()
    yield
    provider_preflight._ready.clear()


@pytest.mark.parametrize('status', [401, 429, 503])
def test_unavailable_proxy_stops_at_the_probe_with_a_safe_actionable_error(configured, monkeypatch, status):
    clients = []

    async def send(client, request, **kwargs):
        clients.append(client)
        assert json.loads(request.content)['max_tokens'] == 64
        return httpx.Response(status, request=request, json={'error': {'message': 'private-preflight-test-key'}})

    monkeypatch.setattr(httpx.AsyncClient, 'send', send)
    with pytest.raises(provider_preflight.ProviderUnavailable) as error:
        asyncio.run(provider_preflight.check_configured_models([('openai', 'fixed-one'), ('openai', 'fixed-two')]))
    assert str(status) in str(error.value) and 'private-preflight-test-key' not in str(error.value)
    assert error.value.status_code == status
    assert len(clients) == 1 and clients[0].is_closed


def test_successful_route_probes_are_reused_across_job_event_loops(configured, monkeypatch):
    calls = []

    async def send(client, request, **kwargs):
        calls.append(request)
        return httpx.Response(200, request=request, json={'id': 'fixture', 'object': 'chat.completion',
            'model': 'fixed-one', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '{"ready":true}'}, 'finish_reason': 'stop'}]})

    monkeypatch.setattr(httpx.AsyncClient, 'send', send)
    first = asyncio.run(provider_preflight.check_configured_models([('openai', 'fixed-one')]))
    second = asyncio.run(provider_preflight.check_configured_models([('openai', 'fixed-one')]))
    assert first[0]['cache_reused'] is False and second[0]['cache_reused'] is True
    assert len(calls) == 1
    assert 'private-preflight-test-key' not in json.dumps(second)


def test_failed_transport_probe_has_no_hidden_retries(configured, monkeypatch):
    calls = []

    async def send(client, request, **kwargs):
        calls.append(request)
        raise httpx.ReadTimeout('Provider did not respond', request=request)

    monkeypatch.setattr(httpx.AsyncClient, 'send', send)
    with pytest.raises(provider_preflight.ProviderUnavailable, match='20-second availability check'):
        asyncio.run(provider_preflight.check_configured_models([('openai', 'fixed-one')]))
    assert len(calls) == 1
