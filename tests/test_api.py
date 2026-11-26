"""HTTP contracts: workspace isolation, input validation, jobs, and durable reports."""
import json
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import backend.api.main as routes
from backend.api.runtime import Jobs, clean, read_summary

HEADERS = {'X-RAG-Client': '1'}
SHA = 'a' * 64


class Database:
    _collection = None

    def __init__(self):
        self._collection = self

    def count(self):
        return 2

    def get(self, **kwargs):
        return {'ids': ['c1', 'c2'], 'documents': ['Evidence one', 'Evidence two'], 'metadatas': [{'pages': '[2]'}, {'pages': '[3]'}]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    from backend import rag_pipeline
    monkeypatch.setattr(rag_pipeline, 'index_is_ready', lambda *args, **kwargs: True)
    monkeypatch.setattr(rag_pipeline, 'load_vector_store', lambda path: Database())
    app = routes.create_app(tmp_path, warm=False)
    with TestClient(app) as client:
        client.get('/api/bootstrap')
        owner = client.cookies.get('rag_workspace')
        (tmp_path / 'docs').mkdir(exist_ok=True)
        (tmp_path / 'docs' / f'{SHA}.pdf').write_bytes(b'%PDF-1.4 test')
        doc = app.state.catalog.put('document', owner, {'name': 'Test.pdf', 'sha256': SHA, 'status': 'ready', 'file_path': f'docs/{SHA}.pdf', 'index_path': 'db/chroma_test', 'stats': {}})
        yield app, client, doc, tmp_path


def await_job(client, key):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        value = client.get(f'/api/jobs/{key}').json()
        if value['status'] in {'succeeded','failed','cancelled'}:
            return value
        time.sleep(.01)
    pytest.fail('Job did not finish')


def test_workspace_isolation_and_secret_free_catalog(env):
    app, client, doc, _ = env
    response = client.get('/api/documents')
    assert response.json()[0]['id'] == doc['id']
    assert 'index_path' not in response.text and 'file_path' not in response.text
    assert 'httponly' in client.get('/api/health', cookies={'rag_workspace': 'invalid'}).headers['set-cookie'].lower()
    with TestClient(app) as stranger:
        assert stranger.get('/api/documents').json() == []
        for suffix in ['pdf', 'answers', 'chunks', 'datasets', 'runs']:
            assert stranger.get(f'/api/documents/{doc["id"]}/{suffix}').status_code == 404


def test_csrf_and_upload_validation(env):
    _, client, _, root = env
    assert client.post('/api/documents', files={'file': ('bad.pdf',b'bad')}).status_code == 403
    assert client.post('/api/documents', headers={**HEADERS,'Origin':'https://evil.example'}, files={'file': ('bad.pdf',b'bad')}).status_code == 403
    assert client.post('/api/documents', headers=HEADERS, files={'file': ('bad.pdf',b'bad')}).status_code == 422
    assert list((root / 'docs').glob('*.upload')) == []


def test_upload_reuses_pdf_and_serializes_preparation(env, monkeypatch):
    _, client, _, root = env
    monkeypatch.setattr(routes, 'prepare_document', lambda *args, **kwargs: {'database': Database(), 'index_path': 'db/chroma_new', 'stats': {'NarrativeText':2}, 'timings': {'index_seconds':.2}, 'extraction_cache_hit': True})
    response = client.post('/api/documents', headers=HEADERS, files={'file': ('../new.pdf',b'%PDF-1.4 new')})
    assert response.status_code == 202
    payload = response.json()
    assert payload['document']['name'] == 'new.pdf'
    job = await_job(client, payload['job']['id'])
    assert job['status'] == 'succeeded'
    assert job['result']['stats']['NarrativeText'] == 2
    assert 'database' not in job['result'] and 'index_path' not in job['result']
    assert list((root / 'docs').glob('*.upload')) == []
    again = client.post('/api/documents', headers=HEADERS, files={'file': ('new.pdf',b'%PDF-1.4 new')}).json()
    assert again['job'] is None
    assert again['document']['id'] == payload['document']['id']


def test_question_validation_and_chunk_pagination(env):
    _, client, doc, _ = env
    path = f'/api/documents/{doc["id"]}'
    assert client.post(path+'/answers', headers=HEADERS,json={'question':' '}).status_code == 422
    assert client.get(path+'/chunks?limit=100').status_code == 422
    chunks = client.get(path+'/chunks').json()
    assert chunks['total'] == 2 and chunks['items'][0]['pages'] == [2]
    assert client.get(path+'/assets/secret.env').status_code == 404


def test_answer_stream_result_and_durable_history(env, monkeypatch):
    app, client, doc, root = env
    def answer(db, question, job, **kwargs):
        job.progress(1,2,'Generating')
        job.token('Grounded answer')
        return {'query':question, 'answer':'Grounded answer [page 2]', 'documents':[Document(page_content='Evidence',metadata={'chunk_id':'c1','pages':'[2]'})], 'timings':{'generation_seconds':.2}}
    monkeypatch.setattr(routes, 'answer_question', answer)
    response = client.post(f'/api/documents/{doc["id"]}/answers',headers=HEADERS,json={'question':'Explain'})
    value = await_job(client, response.json()['id'])
    assert value['status'] == 'succeeded' and value['partial_text'] == 'Grounded answer'
    assert value['result']['sources'][0]['pages'] == [2]
    stream = client.get(f'/api/jobs/{value["id"]}/events')
    assert stream.headers['content-type'].startswith('text/event-stream')
    assert '"status": "succeeded"' in stream.text
    with TestClient(app) as stranger:
        assert stranger.get(f'/api/jobs/{value["id"]}').status_code == 404
        assert stranger.post(f'/api/jobs/{value["id"]}/cancel',headers=HEADERS).status_code == 404
    restarted = routes.create_app(root,warm=False)
    with TestClient(restarted) as other:
        other.cookies.set('rag_workspace',client.cookies.get('rag_workspace'))
        assert len(other.get(f'/api/documents/{doc["id"]}/answers').json()) == 1
        restored = other.get(f'/api/jobs/{value["id"]}').json()
        assert restored['status'] == 'succeeded' and restored['result'] == value['result']
        assert not restored['retryable']
        assert other.post(f'/api/jobs/{value["id"]}/retry', headers=HEADERS).status_code == 409


def test_dataset_hash_rejection_and_template(env):
    _, client, doc, _ = env
    path = f'/api/documents/{doc["id"]}/datasets'
    template = client.get(path+'/template').json()
    assert template['document_hash'] == SHA
    dataset = {'document_hash':'b'*64, 'questions':[{'question':'What?','reference_answer':'Answer','reference_pages':[2]}]}
    bad = client.post(path,headers=HEADERS,files={'file':('wrong.json',json.dumps(dataset))})
    assert bad.status_code == 422 and 'another PDF' in bad.json()['detail']
    dataset['document_hash'] = SHA
    good = client.post(path,headers=HEADERS,files={'file':('reviewed.json',json.dumps(dataset))})
    assert good.status_code == 201 and good.json()['count'] == 1


def test_evaluation_report_and_owned_exports(env, monkeypatch):
    app, client, doc, root = env
    def evaluate(*args, **kwargs):
        directory = root/'evaluation_runs'/'run1'
        directory.mkdir(parents=True)
        manifest = {'profile':'retrieval','document_hash':SHA,'status':'completed','index_path':'private path'}
        (directory/'manifest.json').write_text(json.dumps(manifest))
        (directory/'summary.csv').write_text('strategy,mrr,mrr_count\nVector,1,1\n')
        (directory/'samples.jsonl').write_text('{"status":"completed"}\n')
        frame = pd.DataFrame([{'strategy':'Vector','mrr':1}])
        frame.attrs['manifest_path'] = str(directory/'manifest.json')
        return frame
    monkeypatch.setattr(routes,'evaluate_document',evaluate)
    dataset = app.state.catalog.put('dataset',client.cookies.get('rag_workspace'),{'cases':[{'question':'What?','reference_answer':'Answer','reference_pages':[2],'document_hash':SHA}]},doc['id'])
    response = client.post(f'/api/documents/{doc["id"]}/evaluations',headers=HEADERS,json={'dataset_id':dataset['id'],'profile':'retrieval'})
    value = await_job(client,response.json()['id'])
    assert value['status'] == 'succeeded'
    run = value['result']
    assert run['summary'][0]['mrr'] == 1 and 'index_path' not in run['manifest']
    assert len(client.get(f'/api/documents/{doc["id"]}/runs').json()) == 1
    for kind in ('summary','samples','manifest'):
        assert client.get(f'/api/runs/{run["id"]}/export/{kind}').status_code == 200
    with TestClient(app) as stranger:
        assert stranger.get(f'/api/runs/{run["id"]}/export/summary').status_code == 404


def test_saved_synthetic_evaluation_preserves_reference_provenance(env, monkeypatch):
    app, client, doc, root = env
    provenance = {'origin': 'synthetic_source_verified', 'human_reviewed': False}
    warnings = ['Synthetic reference labels may omit other valid evidence.']
    dataset = app.state.catalog.put('dataset', client.cookies.get('rag_workspace'), {
        'origin': 'synthetic', 'provenance': provenance, 'warnings': warnings,
        'cases': [{'question': 'What is supported?', 'reference_answer': 'Evidence one',
                   'reference_pages': [2], 'document_hash': SHA}]}, doc['id'])
    def evaluate(db_path, received, document_hash, profile, job, **kwargs):
        assert received['provenance'] == provenance and received['warnings'] == warnings
        assert received['questions'][0]['document_hash'] == SHA
        directory = root / 'evaluation_runs' / 'synthetic-rerun'
        directory.mkdir(parents=True)
        manifest = {'profile': profile, 'status': 'completed', 'document_hash': SHA,
                    'dataset_provenance': received['provenance'], 'dataset_warnings': received['warnings']}
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        frame = pd.DataFrame()
        frame.attrs['manifest_path'] = str(directory / 'manifest.json')
        return frame
    monkeypatch.setattr(routes, 'evaluate_document', evaluate)
    response = client.post(f'/api/documents/{doc["id"]}/evaluations', headers=HEADERS,
                           json={'dataset_id': dataset['id'], 'profile': 'retrieval'})
    result = await_job(client, response.json()['id'])
    assert result['status'] == 'succeeded'
    assert result['result']['manifest']['dataset_provenance']['human_reviewed'] is False
    assert result['result']['manifest']['dataset_warnings'] == warnings


def test_jobs_are_bounded_cancelled_and_redacted():
    jobs, release = Jobs(), threading.Event()
    def waiting(job):
        while not release.wait(.01):
            job.check_cancelled()
    first = jobs.submit('owner','doc','answer',waiting)
    jobs.submit('owner','doc','answer',waiting)
    try:
        with pytest.raises(Exception) as error:
            jobs.submit('owner','doc','answer',waiting)
        assert error.value.status_code == 429
        jobs.get('owner',first['id'])['job'].cancel()
        jobs.get('owner',first['id'])['job'].future.result(timeout=2)
        assert jobs.snapshot('owner',first['id'])['status'] == 'cancelled'
    finally:
        release.set()
    def fail(job):
        raise ValueError('api_key=very-secret')
    failed = jobs.submit('other','doc','answer',fail)
    jobs.get('other',failed['id'])['job'].future.result(timeout=2)
    assert 'very-secret' not in jobs.snapshot('other',failed['id'])['error']
    assert clean({'score':float('nan'),'index_path':'hidden'}) == {'score':None}


def test_saved_summary_excludes_invalid_scores(tmp_path):
    path = tmp_path / 'summary.csv'
    path.write_text('strategy,mrr,mrr_count,answer_relevancy,recall_at_k\nVector,2,3,-0.1,0.5\n')
    rows, warnings = read_summary(path)
    assert rows[0]['mrr'] is None and rows[0]['recall_at_k'] == .5
    assert rows[0]['answer_relevancy'] == -.1
    assert 'mrr' in warnings[0]


def test_automatic_evaluation_generates_persists_and_exports_without_import(env, monkeypatch):
    app, client, doc, root = env
    calls = []
    async def generate(db_path, **kwargs):
        calls.append(('generate', kwargs['question_count']))
        assert kwargs['expected_document_hash'] == SHA
        assert Path(kwargs['output_dir']).is_relative_to(root)
        kwargs['progress_callback'](1, 1, 'Source verification complete')
        return {'document_hash': SHA, 'questions': [{
            'id': 'auto1', 'question': 'What is supported?', 'reference_answer': 'Evidence one',
            'reference_pages': [2], 'relevant_chunk_ids': ['c1'], 'category': 'single_fact',
            'supporting_quotes': [{'chunk_id': 'c1', 'quote': 'Evidence one'}],
            'verification': {'supported': True},
        }], 'provenance': {'origin': 'synthetic', 'verification_model': 'separate-judge'},
            'warnings': ['Automatically verified synthetic reference.'], 'cache_reused': False}
    def evaluate(db_path, dataset, document_hash, profile, job, **kwargs):
        calls.append(('evaluate', profile))
        assert dataset['questions'][0]['question'] == 'What is supported?'
        assert dataset['provenance']['origin'] == 'synthetic'
        directory = Path(kwargs['output_dir']) / 'automatic1'
        directory.mkdir(parents=True)
        manifest = {'profile': profile, 'document_hash': document_hash, 'status': 'completed',
                    'dataset_provenance': dataset['provenance']}
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        (directory / 'summary.csv').write_text('strategy,faithfulness,faithfulness_count\nReranking,0.9,1\n')
        (directory / 'samples.jsonl').write_text('{"status":"completed"}\n')
        frame = pd.DataFrame([{'strategy': 'Reranking', 'faithfulness': .9}])
        frame.attrs['manifest_path'] = str(directory / 'manifest.json')
        return frame
    monkeypatch.setattr(routes, 'generate_dataset', generate)
    monkeypatch.setattr(routes, 'evaluate_document', evaluate)
    path = f'/api/documents/{doc["id"]}/evaluations/automatic'
    response = client.post(path, headers=HEADERS, json={'question_count': 1})
    assert response.status_code == 202
    job = await_job(client, response.json()['id'])
    assert job['status'] == 'succeeded'
    assert calls == [('generate', 1), ('evaluate', 'full')]
    dataset = client.get(f'/api/documents/{doc["id"]}/datasets').json()[0]
    assert dataset['origin'] == 'synthetic'
    assert dataset['cases'][0]['supporting_quotes'][0]['quote'] == 'Evidence one'
    exported = client.get(f'/api/datasets/{dataset["id"]}/export')
    assert exported.status_code == 200 and exported.json()['document_hash'] == SHA
    result = job['result']
    assert result['dataset_id'] == dataset['id'] and result['automatic']['accepted_questions'] == 1
    assert client.get(f'/api/documents/{doc["id"]}/runs').json()[0]['automatic']['accepted_questions'] == 1
    with TestClient(app) as stranger:
        assert stranger.post(path, headers=HEADERS, json={'question_count': 1}).status_code == 404
        assert stranger.get(f'/api/datasets/{dataset["id"]}/export').status_code == 404
    assert len(calls) == 2


@pytest.mark.parametrize('count', [0, 51, 1.5, True, '2'])
def test_automatic_question_count_rejected_before_work(env, monkeypatch, count):
    _, client, doc, _ = env
    async def unexpected(*args, **kwargs):
        pytest.fail('Invalid count must not start model work')
    monkeypatch.setattr(routes, 'generate_dataset', unexpected)
    assert client.post(f'/api/documents/{doc["id"]}/evaluations/automatic',
        headers=HEADERS, json={'question_count': count}).status_code == 422


def test_automatic_generation_failure_does_not_evaluate_or_persist_dataset(env, monkeypatch):
    _, client, doc, _ = env
    async def generate(*args, **kwargs):
        raise ValueError('No source-verified questions could be generated.')
    monkeypatch.setattr(routes, 'generate_dataset', generate)
    monkeypatch.setattr(routes, 'evaluate_document', lambda *a, **k: pytest.fail('No evaluation without verified references'))
    response = client.post(f'/api/documents/{doc["id"]}/evaluations/automatic', headers=HEADERS, json={})
    job = await_job(client, response.json()['id'])
    assert job['status'] == 'failed' and 'source-verified' in job['error']
    assert client.get(f'/api/documents/{doc["id"]}/datasets').json() == []
    assert client.get(f'/api/documents/{doc["id"]}/runs').json() == []


def test_automatic_cancel_after_generation_does_not_start_evaluation(env, monkeypatch):
    app, client, doc, _ = env
    entered, release = threading.Event(), threading.Event()
    async def generate(*args, **kwargs):
        entered.set()
        release.wait(2)
        return {'document_hash': SHA, 'questions': []}
    monkeypatch.setattr(routes, 'generate_dataset', generate)
    monkeypatch.setattr(routes, 'evaluate_document', lambda *a, **k: pytest.fail('Cancelled generation cannot evaluate'))
    response = client.post(f'/api/documents/{doc["id"]}/evaluations/automatic', headers=HEADERS, json={})
    key = response.json()['id']
    try:
        assert entered.wait(2)
        assert client.post(f'/api/jobs/{key}/cancel', headers=HEADERS).status_code == 200
    finally:
        release.set()
    assert await_job(client, key)['status'] == 'cancelled'


def interrupted_snapshot():
    return {'status': 'running', 'done': 1, 'total': 3, 'message': 'Generating',
            'partial_text': 'An unfinished answer', 'elapsed_seconds': 2,
            'events': [[1, 'Generating']], 'result': None, 'error': None}


@pytest.mark.parametrize('operation', ['prepare', 'answer', 'retrieval', 'automatic', 'evaluation'])
def test_restart_restores_owned_jobs_and_explicit_retry_settings(env, monkeypatch, operation):
    from concurrent.futures import ThreadPoolExecutor
    app, client, doc, root = env
    owner, key = client.cookies.get('rag_workspace'), 'b' * 32
    calls, entered, release = [], threading.Event(), threading.Event()
    dataset = app.state.catalog.put('dataset', owner, {'cases': [{
        'id': 'q1', 'question': 'Original question?', 'reference_answer': 'Evidence one',
        'document_hash': SHA, 'reference_pages': [2]}]}, doc['id'])
    bodies = {'prepare': {}, 'answer': {'question': 'Original question?', 'rerank': False},
        'retrieval': {'question': 'Original question?', 'variations': ['Alternate query'],
                      'strategy': 'keyword', 'k': 7, 'top_n': 2, 'expand': False, 'generate': False},
        'automatic': {'question_count': 2, 'profile': 'retrieval', 'force': True},
        'evaluation': {'dataset_id': dataset['id'], 'profile': 'retrieval', 'force': True}}
    kind = 'evaluation' if operation in {'automatic', 'evaluation'} else operation
    app.state.catalog.save_job(key, owner, doc['id'], kind, interrupted_snapshot(),
                               request={'operation': operation, 'body': bodies[operation]})

    def hold(stage):
        calls.append(stage)
        entered.set()
        assert release.wait(3)

    def prepare(*args, **kwargs):
        assert kwargs['storage_root'] == root
        hold('prepare')
        return {'database': Database(), 'index_path': 'db/chroma_test', 'stats': {},
                'timings': {}, 'extraction_cache_hit': True}

    def answer(db, question, job, **kwargs):
        assert question == 'Original question?' and kwargs['use_reranker'] is False
        hold('answer')
        return {'query': question, 'answer': 'Evidence one [page 2]', 'documents': [], 'timings': {}}

    def retrieve(db, body, job, document_id):
        assert body.strategy == 'keyword' and body.k == 7 and body.top_n == 2
        assert body.variations == ['Alternate query'] and not body.expand and not body.generate
        hold('retrieval')
        return {'query': body.question, 'sources': []}

    async def generate(*args, **kwargs):
        assert kwargs['question_count'] == 2 and kwargs['force'] is False
        hold('automatic')
        return {'document_hash': SHA, 'questions': dataset['cases'], 'cache_reused': True}

    def evaluate(db_path, data, sha, profile, job, **kwargs):
        assert sha == SHA and profile == 'retrieval' and kwargs['force'] is False
        assert data['questions'][0]['question'] == 'Original question?'
        hold('evaluation')
        directory = root / 'evaluation_runs' / job.key
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text(json.dumps({'status': 'completed', 'profile': profile}))
        (directory / 'summary.csv').write_text('strategy,mrr\nVector,1\n')
        frame = pd.DataFrame()
        frame.attrs['manifest_path'] = str(directory / 'manifest.json')
        return frame

    monkeypatch.setattr(routes, 'prepare_document', prepare)
    monkeypatch.setattr(routes, 'answer_question', answer)
    monkeypatch.setattr(routes, 'retrieval_experiment', retrieve)
    monkeypatch.setattr(routes, 'generate_dataset', generate)
    monkeypatch.setattr(routes, 'evaluate_document', evaluate)
    restarted = routes.create_app(root, warm=False)
    with TestClient(restarted) as other:
        other.cookies.set('rag_workspace', owner)
        restored = other.get(f'/api/jobs/{key}').json()
        assert restored['status'] == 'failed' and restored['retryable'] and restored['recovery_required']
        assert restored['partial_text'] == 'An unfinished answer'
        listed = other.get('/api/jobs').json()
        assert len(listed) == 1 and listed[0]['id'] == key
        assert 'request' not in listed[0] and 'body' not in listed[0]
        assert calls == []  # Reading/restoring never starts a provider request.
        with TestClient(restarted) as stranger:
            assert stranger.get('/api/jobs').json() == []
            for suffix in ('', '/events'):
                assert stranger.get(f'/api/jobs/{key}{suffix}').status_code == 404
            assert stranger.post(f'/api/jobs/{key}/retry', headers=HEADERS).status_code == 404
        # The nested client above closes this app's lifespan; use a fresh runtime.
    restarted = routes.create_app(root, warm=False)
    with TestClient(restarted) as other:
        other.cookies.set('rag_workspace', owner)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(pool.map(lambda _: other.post(f'/api/jobs/{key}/retry', headers=HEADERS), range(2)))
            assert all(response.status_code == 202 for response in responses)
            child = responses[0].json()['id']
            assert child != key and responses[1].json()['id'] == child
            assert entered.wait(2)
            assert other.get(f'/api/jobs/{key}').json()['retry_job_id'] == child
            assert [row['id'] for row in other.get('/api/jobs').json()] == [child]
        finally:
            release.set()
        result = await_job(other, child)
        assert result['status'] == 'succeeded', result
        expected = ['automatic', 'evaluation'] if operation == 'automatic' else [operation]
        assert calls == expected
        assert other.get('/api/jobs').json() == []
        assert other.post(f'/api/jobs/{key}/retry', headers=HEADERS).json()['id'] == child
    again = routes.create_app(root, warm=False)
    with TestClient(again) as other:
        other.cookies.set('rag_workspace', owner)
        assert other.post(f'/api/jobs/{key}/retry', headers=HEADERS).json()['id'] == child
        assert other.get(f'/api/jobs/{child}').json()['status'] == 'succeeded'
        assert calls == expected


@pytest.mark.parametrize('kind', ['answer', 'evaluation'])
def test_committed_result_is_recovered_without_repeating_work(env, monkeypatch, kind):
    app, client, doc, root = env
    owner, key = client.cookies.get('rag_workspace'), 'c' * 32
    app.state.catalog.save_job(key, owner, doc['id'], kind, interrupted_snapshot(), request={'operation': kind})
    if kind == 'answer':
        payload = {'query': 'Already answered?', 'answer': 'Completed answer', 'sources': [], 'timings': {}}
        resource = 'answer'
    else:
        directory = root / 'evaluation_runs' / 'committed'
        directory.mkdir(parents=True)
        (directory / 'summary.csv').write_text('strategy,mrr\nVector,1\n')
        payload = {'manifest': {'status': 'completed'}, 'directory': 'evaluation_runs/committed'}
        resource = 'run'
    app.state.catalog.put(resource, owner, payload, doc['id'], resource_id=key)
    monkeypatch.setattr(routes, 'answer_question', lambda *a, **k: pytest.fail('A committed answer must not be regenerated'))
    monkeypatch.setattr(routes, 'evaluate_document', lambda *a, **k: pytest.fail('A committed report must not be rescored'))
    restarted = routes.create_app(root, warm=False)
    with TestClient(restarted) as other:
        other.cookies.set('rag_workspace', owner)
        restored = other.get(f'/api/jobs/{key}').json()
        assert restored['status'] == 'succeeded' and not restored['retryable']
        assert restored['error'] is None and restored['done'] == restored['total']
        assert restored['result']['id'] == key
        if kind == 'evaluation':
            assert restored['result']['summary'][0]['mrr'] == 1
            assert 'directory' not in restored['result']
        assert other.get('/api/jobs').json() == []
        assert other.post(f'/api/jobs/{key}/retry', headers=HEADERS).status_code == 409


def test_graceful_shutdown_retains_interruption_and_late_worker_updates_do_not_overwrite_it(tmp_path):
    from backend.api.storage import Catalog
    catalog = Catalog(tmp_path / 'catalog.sqlite3')
    jobs, entered = Jobs(catalog), threading.Event()
    def waiting(job):
        job.progress(1, 3, 'Waiting')
        job.token('Partial text')
        entered.set()
        assert job.cancellation_event.wait(3)
        job.check_cancelled()
    info = jobs.submit('owner', 'doc', 'answer', waiting, request={'operation': 'answer', 'body': {'question': 'Original?'}})
    assert entered.wait(2)
    jobs.shutdown()
    jobs.get('owner', info['id'])['job'].future.result(timeout=2)
    jobs.shutdown()  # Repeated shutdown must not replace interruption with cancellation.
    restored = Jobs(Catalog(catalog.path)).snapshot('owner', info['id'])
    assert restored['status'] == 'failed' and restored['recovery_required'] and restored['retryable']
    assert restored['partial_text'] == 'Partial text'
    assert restored['events'][0][1] == 'Waiting'
    with pytest.raises(Exception) as error:
        jobs.submit('owner', 'doc', 'answer', waiting)
    assert error.value.status_code == 503


def test_legacy_job_schema_is_migrated_and_old_jobs_remain_readable(tmp_path):
    import sqlite3
    from backend.api.storage import Catalog
    path = tmp_path / 'legacy.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE jobs (id TEXT PRIMARY KEY, owner TEXT, document_id TEXT, kind TEXT, snapshot TEXT)')
        db.execute('INSERT INTO jobs VALUES (?, ?, ?, ?, ?)', ('legacy', 'owner', 'doc', 'answer', json.dumps(interrupted_snapshot())))
    jobs = Jobs(Catalog(path))
    restored = jobs.snapshot('owner', 'legacy')
    assert restored['status'] == 'failed' and restored['recovery_required'] and not restored['retryable']
    assert jobs.pending('owner')[0]['id'] == 'legacy'
    assert jobs.pending('stranger') == []
    with pytest.raises(Exception) as error:
        jobs.retry('owner', 'legacy', lambda *args: pytest.fail('Legacy jobs have no saved request'))
    assert error.value.status_code == 409


@pytest.mark.parametrize('saved', [{'operation': 'retired'}, {'operation': 'answer', 'body': {'question': ''}}])
def test_stale_saved_settings_fail_without_creating_a_retry(env, saved):
    app, client, doc, root = env
    owner, key = client.cookies.get('rag_workspace'), 'e' * 32
    app.state.catalog.save_job(key, owner, doc['id'], 'answer', interrupted_snapshot(), request=saved)
    restarted = routes.create_app(root, warm=False)
    with TestClient(restarted) as other:
        other.cookies.set('rag_workspace', owner)
        response = other.post(f'/api/jobs/{key}/retry', headers=HEADERS)
        assert response.status_code == 409
        assert other.get(f'/api/jobs/{key}').json()['retry_job_id'] is None
        assert [job['id'] for job in other.get('/api/jobs').json()] == [key]
