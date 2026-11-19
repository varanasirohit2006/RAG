import json

import httpx
import pytest
from langchain_core.documents import Document

import generation_runtime as generation

MODEL = 'nvidia/nemotron-3-ultra-550b-a55b:free'


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv('GENERATION_PROVIDER', 'openrouter')
    monkeypatch.setenv('GENERATION_MODEL', MODEL)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'private-test-router-key')
    monkeypatch.setenv('OPENAI_API_KEY', 'different-test-key')
    generation._build_model.cache_clear()
    yield
    generation._build_model.cache_clear()


@pytest.mark.parametrize('model', ['nvidia/nemotron-3-ultra-550b-a55b', 'nemotron:free',
                                  'openrouter/auto', 'openrouter/auto:free', 'openrouter/free'])
def test_generation_cannot_select_paid_or_random_routes(settings, model):
    with pytest.raises(ValueError, match='fixed model ID'):
        generation.get_generation_model(model)


def test_generation_cannot_use_another_providers_key(settings, monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY')
    with pytest.raises(ValueError, match='OPENROUTER_API_KEY'):
        generation.get_generation_model()


@pytest.mark.parametrize('streaming', [False, True])
def test_real_openrouter_client_sends_only_free_route_and_streams_text(settings, monkeypatch, streaming):
    import langchain_openai

    requests = []
    answer = 'The invoice total is 7 USD [Source 1; page 2].'
    def respond(request):
        requests.append(request)
        if streaming:
            chunks = [{'id': 'fixture', 'object': 'chat.completion.chunk', 'model': MODEL,
                       'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': reason}]}
                      for text, reason in [(answer[:15], None), (answer[15:], None), ('', 'stop')]]
            content = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks) + 'data: [DONE]\n\n'
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=content)
        return httpx.Response(200, json={'id': 'fixture', 'object': 'chat.completion', 'model': MODEL,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': answer}, 'finish_reason': 'stop'}]})
    real = langchain_openai.ChatOpenAI
    transport = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(langchain_openai, 'ChatOpenAI', lambda **kwargs: real(**kwargs, http_client=transport))
    tokens = []
    docs = [Document(page_content='The invoice total is 7 USD.', metadata={'pages': [2], 'chunk_id': 'source'})]
    try:
        result = generation.generate_final_answer(docs, 'What is the invoice total?', token_callback=tokens.append if streaming else None)
        assert result == answer
        if streaming:
            assert ''.join(tokens) == answer
        assert len(requests) == 1
        request = requests[0]
        assert str(request.url) == 'https://openrouter.ai/api/v1/chat/completions'
        assert request.headers['authorization'] == 'Bearer private-test-router-key'
        body = json.loads(request.content)
        assert body['model'] == MODEL and 'models' not in body
        assert body['provider']['max_price'] == {'prompt': 0, 'completion': 0}
        assert body['reasoning'] == {'enabled': False}
        assert 'The invoice total is 7 USD.' in body['messages'][0]['content'][0]['text']
    finally:
        transport.close()


def test_ultra_receives_ocr_text_and_tables_without_loading_chart_images(settings):
    doc = Document(page_content='Extracted chart labels', metadata={'pages': [2], 'chunk_id': 'source',
        'original_content': json.dumps({'raw_text': 'OCR total: 7 USD',
            'tables_html': ['<table><tr><td>Total</td><td>7 USD</td></tr></table>'],
            'image_paths': ['missing-chart.jpg'], 'images_base64': ['invalid-image']})})
    context = generation.prepare_generation_context([doc], 'What is the total?')
    assert 'OCR total: 7 USD' in context['prompt'] and '<table>' in context['prompt']
    assert context['chunk_ids'] == ['source'] and context['visual_evidence_omitted']
    assert not context['image_paths'] and not context['images_base64']
    assert all(part['type'] == 'text' for part in context['messages'][0].content)
    assert 'Images are not supplied' in context['prompt']


def test_explicit_vision_model_keeps_its_capability_even_with_ultra_configured(settings):
    assert not generation.generation_supports_images('openrouter')
    assert generation.generation_supports_images('openrouter', 'nvidia/nemotron-3.5-super-vl:free')


def test_openrouter_rate_failure_is_reported_without_retry_or_secret(settings, monkeypatch):
    import langchain_openai

    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(429, json={'error': {'message': 'Provider limited: private-test-router-key'}})
    transport = httpx.Client(transport=httpx.MockTransport(respond))
    real = langchain_openai.ChatOpenAI
    monkeypatch.setattr(langchain_openai, 'ChatOpenAI', lambda **kwargs: real(**kwargs, http_client=transport))
    try:
        with pytest.raises(generation.GenerationError) as error:
            generation.generate_final_answer([Document(page_content='Source evidence')], 'Question?')
        assert '429' in str(error.value) and 'private-test-router-key' not in str(error.value)
        assert len(calls) == 1
    finally:
        transport.close()
