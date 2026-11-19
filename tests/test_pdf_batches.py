import hashlib
import json
from pathlib import Path

import pandas as pd
import pypdfium2 as pdfium
import pytest
from fastapi.testclient import TestClient

import api.main as routes
from pdf_batches import merge_pdfs, original_pages
from test_api import Database, HEADERS, await_job, env
from test_document_extraction import write_native_pdf
from test_vector_ingestion import LocalTestEmbeddings


def pdf_bytes(root, name, text):
    return write_native_pdf(root / name, [f'BT /F1 12 Tf 20 260 Td ({text}) Tj ET'], width=600).read_bytes()


@pytest.fixture
def services(env, monkeypatch):
    app, client, _, root = env
    calls = {'prepare': [], 'generate': [], 'evaluate': []}
    def prepare(file, sha, key, settings, job, **kwargs):
        calls['prepare'].append({'sha': sha, 'sources': kwargs.get('source_documents'), 'key': key})
        return {'database': Database(), 'index_path': 'db/chroma_' + key, 'stats': {'NarrativeText': 1},
                'timings': {}, 'extraction_cache_hit': False}
    async def generate(db_path, **kwargs):
        sha = kwargs['expected_document_hash']
        calls['generate'].append({'sha': sha, 'force': kwargs['force']})
        return {'document_hash': sha, 'questions': [{'id': sha[:8] + '-' + str(index),
            'question': f'Fixture question {index}?', 'reference_answer': 'Fixture answer', 'reference_pages': [1]}
            for index in range(kwargs['question_count'])], 'provenance': {'fixture': sha}}
    def evaluate(db_path, dataset, sha, profile, job, **kwargs):
        calls['evaluate'].append({'sha': sha, 'questions': dataset['questions'], 'key': job.key})
        directory = Path(kwargs['output_dir']) / job.key
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text(json.dumps({'profile': profile, 'status': 'completed', 'document_hash': sha}))
        (directory / 'summary.csv').write_text('strategy,mrr,mrr_count,samples\nVector,1,1,1\n')
        frame = pd.DataFrame()
        frame.attrs['manifest_path'] = str(directory / 'manifest.json')
        return frame
    monkeypatch.setattr(routes, 'prepare_document', prepare)
    monkeypatch.setattr(routes, 'generate_dataset', generate)
    monkeypatch.setattr(routes, 'evaluate_document', evaluate)
    return app, client, root, calls


def upload(client, root, count=3, mode='separate'):
    files = [('files', (f'pdf-{index}.pdf', pdf_bytes(root, f'fixture-{index}.pdf', f'PDF {index} reports a budget of {7 + index} USD with source evidence.')))
             for index in range(count)]
    return client.post('/api/documents/batch', headers=HEADERS, data={'mode': mode}, files=files)


def test_three_pdf_upload_uses_one_job_and_keeps_individual_documents(services):
    app, client, root, calls = services
    response = upload(client, root)
    assert response.status_code == 202
    data = response.json()
    assert len(data['documents']) == 3 and data['combined'] is None and data['errors'] == []
    assert 'file_path' not in response.text
    job = await_job(client, data['job']['id'])
    assert job['status'] == 'succeeded', job
    assert len(calls['prepare']) == 3 and len(app.state.jobs.entries) == 1
    ids = {doc['id'] for doc in data['documents']}
    assert all(doc['status'] == 'ready' for doc in client.get('/api/documents').json() if doc['id'] in ids)


def test_combined_upload_preserves_original_page_map_and_is_idempotent(services):
    _, client, root, calls = services
    data = upload(client, root, count=2, mode='combined').json()
    assert data['errors'] == [] and data['combined']
    job = await_job(client, data['job']['id'])
    assert job['status'] == 'succeeded', job
    group = data['combined']
    assert [source['page_start'] for source in group['source_documents']] == [1, 2]
    assert [source['page_count'] for source in group['source_documents']] == [1, 1]
    assert calls['prepare'][0]['sources'] == group['source_documents']
    assert len(calls['prepare']) == 1
    sources = original_pages([1, 2], group['source_documents'])
    assert [source['pages'] for source in sources] == [[1], [1]]
    assert [source['combined_pages'] for source in sources] == [[1], [2]]
    pdf = client.get(f"/api/documents/{group['id']}/pdf")
    with pdfium.PdfDocument(pdf.content) as document:
        assert len(document) == 2
    repeated = upload(client, root, count=2, mode='combined').json()
    assert repeated['combined']['id'] == group['id'] and repeated['job'] is None
    assert len(calls['prepare']) == 1


@pytest.mark.parametrize('mode', ['separate', 'combined'])
def test_batch_evaluation_creates_isolated_or_combined_reports(services, mode):
    app, client, root, calls = services
    data = upload(client, root).json()
    await_job(client, data['job']['id'])
    ids = [doc['id'] for doc in data['documents']]
    response = client.post('/api/evaluations/batch', headers=HEADERS, json={
        'document_ids': ids, 'mode': mode, 'question_count': 2, 'profile': 'full'})
    assert response.status_code == 202, response.text
    info = response.json()
    job = await_job(client, info['id'])
    assert job['status'] == 'succeeded', job
    result = job['result']
    assert result['mode'] == mode and result['status'] == 'completed'
    rows = result['batch_results']
    assert len(rows) == (3 if mode == 'separate' else 1)
    assert len(calls['generate']) == len(rows) == len(calls['evaluate'])
    assert len({row['run']['dataset_id'] for row in rows}) == len(rows)
    for row in rows:
        doc = next(doc for doc in client.get('/api/documents').json() if doc['id'] == row['document_id'])
        assert row['run']['manifest']['document_hash'] == doc['sha256']
        assert row['run']['document_id'] == doc['id']
        assert len(client.get(f"/api/documents/{doc['id']}/runs").json()) == 1
    if mode == 'combined':
        assert info['document_id'] not in ids
        doc = next(doc for doc in client.get('/api/documents').json() if doc['id'] == info['document_id'])
        assert len(doc['source_documents']) == 3
    else:
        assert {row['document_id'] for row in rows} == set(ids)


def test_batch_cannot_discover_another_workspaces_files(services):
    app, client, root, calls = services
    data = upload(client, root, count=2).json()
    await_job(client, data['job']['id'])
    before = len(calls['generate'])
    stranger = TestClient(app)
    try:
        response = stranger.post('/api/evaluations/batch', headers=HEADERS, json={
            'document_ids': [doc['id'] for doc in data['documents']], 'mode': 'combined'})
        assert response.status_code == 404
        assert len(calls['generate']) == before
    finally:
        stranger.close()


def test_invalid_file_and_batch_limits_are_visible(services):
    _, client, root, calls = services
    response = client.post('/api/documents/batch', headers=HEADERS, files=[
        ('files', ('good.pdf', pdf_bytes(root, 'good-fixture.pdf', 'Valid source text with an approved budget of 7 USD.'))),
        ('files', ('bad.pdf', b'not a PDF'))])
    data = response.json()
    assert len(data['documents']) == 1 and data['errors'][0]['name'] == 'bad.pdf'
    assert await_job(client, data['job']['id'])['status'] == 'succeeded'
    assert upload(client, root, count=4).status_code == 422
    assert len(calls['prepare']) == 1


def test_combined_mode_does_not_silently_drop_an_invalid_pdf(services):
    _, client, root, calls = services
    response = client.post('/api/documents/batch', headers=HEADERS, data={'mode': 'combined'}, files=[
        ('files', ('good.pdf', pdf_bytes(root, 'good-fixture.pdf', 'Valid source text with an approved budget of 7 USD.'))),
        ('files', ('bad.pdf', b'invalid PDF'))])
    data = response.json()
    assert data['combined'] is None and data['job'] is None and data['errors']
    assert calls['prepare'] == []


def test_batch_continues_after_one_document_preparation_failure(services, monkeypatch):
    _, client, root, calls = services
    data = upload(client, root).json()
    await_job(client, data['job']['id'])
    ids = [doc['id'] for doc in data['documents']]
    original = routes.generate_dataset
    async def generate(db_path, **kwargs):
        if kwargs['expected_document_hash'] == data['documents'][1]['sha256']:
            raise ValueError('This PDF could not produce verified questions.')
        return await original(db_path, **kwargs)
    monkeypatch.setattr(routes, 'generate_dataset', generate)
    response = client.post('/api/evaluations/batch', headers=HEADERS, json={'document_ids': ids, 'mode': 'separate', 'question_count': 1})
    job = await_job(client, response.json()['id'])
    assert job['status'] == 'succeeded' and job['result']['status'] == 'partial'
    results = job['result']['batch_results']
    assert 'run' in results[0] and 'error' in results[1] and 'run' in results[2]
    assert len(calls['evaluate']) == 2


def test_all_failed_evaluations_produce_errors_instead_of_fake_reports(services, monkeypatch):
    _, client, root, calls = services
    data = upload(client, root).json()
    await_job(client, data['job']['id'])
    async def unavailable(*args, **kwargs):
        raise RuntimeError('Provider unavailable; no verified questions.')
    monkeypatch.setattr(routes, 'generate_dataset', unavailable)
    response = client.post('/api/evaluations/batch', headers=HEADERS, json={
        'document_ids': [doc['id'] for doc in data['documents']], 'mode': 'separate', 'question_count': 1})
    job = await_job(client, response.json()['id'])
    assert job['result']['status'] == 'failed'
    assert all('error' in result and 'run' not in result for result in job['result']['batch_results'])
    assert calls['evaluate'] == []


def test_combined_index_has_original_and_combined_citations_without_paid_calls(tmp_path, monkeypatch):
    import document_extraction
    import rag_pipeline
    from api.runtime import source
    from generation_runtime import prepare_generation_context

    documents = []
    for index, text in enumerate(['Alpha approves a budget of 7 USD and identifies Cedar as the project owner.',
                                  'Beta schedules a review on Friday and approves a budget of 9 USD.']):
        path = write_native_pdf(tmp_path / f'pdf-{index}.pdf', [f'BT /F1 12 Tf 20 260 Td ({text}) Tj ET'], width=600)
        documents.append({'id': str(index) * 32, 'name': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'file_path': path.name})
    merged = merge_pdfs(documents, tmp_path)
    elements = document_extraction.extract_document(tmp_path / merged['file_path'])
    monkeypatch.setattr(rag_pipeline, 'get_embedding_model', lambda: LocalTestEmbeddings())
    directory = tmp_path / 'vectors'
    database = rag_pipeline.get_or_create_vector_store(elements, str(directory), document_hash=merged['sha256'], source_documents=merged['source_documents'])
    records = database.get()
    from langchain_core.documents import Document
    chunks = [Document(page_content=text, metadata=metadata) for text, metadata in zip(records['documents'], records['metadatas'])]
    assert all(len(source(chunk, 'group')['original_sources']) == 1 for chunk in chunks)
    originals = [item for chunk in chunks for item in source(chunk, 'group')['original_sources']]
    assert {item['document_id'] for item in originals} == {'0' * 32, '1' * 32}
    assert all(item['pages'] == [1] for item in originals)
    assert {page for item in originals for page in item['combined_pages']} == {1, 2}
    context = prepare_generation_context(chunks, 'Compare the two budgets.', provider='google')
    assert 'combined pages' in context['prompt'] and 'displayed combined-PDF page numbers' in context['prompt']
    index_manifest = json.loads((directory / 'index_manifest.json').read_text())
    assert index_manifest['source_documents'] == merged['source_documents'] and index_manifest['corpus_type'] == 'combined_pdfs'
