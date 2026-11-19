"""Same-origin React API with local workspaces or verified OIDC accounts."""
import asyncio
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, model_validator

from api.runtime import Jobs, TERMINAL, clean, read_summary, retrieval_experiment, source
from api.storage import Catalog
from api.auth import Authentication, AuthSettings, PUBLIC_API_PATHS
from document_extraction import ExtractionSettings
from evaluation_data import EvaluationDataError, dataset_template, load_evaluation_dataset
from processing_jobs import answer_question, evaluate_document, extraction_cache_key, prepare_document
from pdf_batches import MAX_BATCH_PDFS, BatchStageJob, merge_pdfs


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    rerank: bool = True

    @model_validator(mode='after')
    def nonblank(self):
        if not self.question.strip():
            raise ValueError('Enter a question.')
        return self


class Evaluation(BaseModel):
    dataset_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    profile: Literal['preview', 'full', 'retrieval'] = 'preview'
    force: bool = False


class AutomaticEvaluation(BaseModel):
    question_count: int = Field(default=10, ge=1, le=50, strict=True)
    profile: Literal['preview', 'full', 'retrieval'] = 'full'
    force: bool = False


class BatchEvaluation(AutomaticEvaluation):
    document_ids: list[str] = Field(min_length=1, max_length=MAX_BATCH_PDFS)
    mode: Literal['separate', 'combined'] = 'separate'

    @model_validator(mode='after')
    def valid_documents(self):
        if len(set(self.document_ids)) != len(self.document_ids) or any(not re.fullmatch(r'[a-f0-9]{32}', key) for key in self.document_ids):
            raise ValueError('Select unique document IDs from your library.')
        if self.mode == 'combined' and len(self.document_ids) < 2:
            raise ValueError('Combine at least two different PDFs.')
        return self


async def generate_dataset(*args, **kwargs):
    """Lazy import keeps ordinary API startup free of generation work."""
    from evaluation_generation import generate_evaluation_dataset
    return await generate_evaluation_dataset(*args, **kwargs)


class Retrieval(Question):
    strategy: Literal['vector', 'keyword', 'hybrid'] = 'hybrid'
    variations: list[str] = Field(default_factory=list, max_length=3)
    expand: bool = False
    generate: bool = False
    k: int = Field(default=10, ge=1, le=30)
    top_n: int = Field(default=3, ge=1, le=10)
    vector_weight: float = Field(default=0.5, ge=0, le=1)
    rerank: bool = False

    @model_validator(mode='after')
    def limits(self):
        if self.top_n > self.k or any(len(q) > 4000 for q in self.variations):
            raise ValueError('Top N must not exceed K; query variations must be at most 4000 characters.')
        return self


def create_app(root=None, *, warm=True, auth_transport=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    load_dotenv(root / '.env')
    auth_settings = AuthSettings.from_env()
    catalog = Catalog(root / 'db' / 'web_catalog.sqlite3')
    auth = Authentication(auth_settings, catalog, transport=auth_transport)
    jobs, databases = Jobs(catalog), {}

    @asynccontextmanager
    async def lifespan(app):
        if warm:
            from bootstrap import warm_runtime
            warm_runtime()
        try:
            yield
        finally:
            jobs.shutdown()

    app = FastAPI(title='RAG Workspace', lifespan=lifespan)
    app.state.catalog, app.state.jobs = catalog, jobs
    app.state.auth = auth
    auth.register(app)

    @app.middleware('http')
    async def workspace(request, call_next):
        fresh = False
        if auth_settings.mode == 'oidc':
            account = auth.session(request)
            request.state.owner = account['id'] if account else ''
            if request.url.path.startswith('/api/') and request.url.path not in PUBLIC_API_PATHS and not account:
                return JSONResponse({'detail': 'Sign in to access your workspace.'}, status_code=401,
                                    headers={'Cache-Control': 'no-store'})
        else:
            owner = request.cookies.get('rag_workspace', '')
            fresh = not re.fullmatch(r'[a-f0-9]{32}', owner)
            request.state.owner = uuid.uuid4().hex if fresh else owner
        if request.url.path.startswith('/api/') and request.method not in ('GET', 'HEAD', 'OPTIONS'):
            origin = request.headers.get('origin')
            expected_origin = auth_settings.base_url if auth_settings.mode == 'oidc' else f'{request.url.scheme}://{request.headers.get("host")}'
            if request.headers.get('x-rag-client') != '1' or (origin and origin != expected_origin):
                return JSONResponse({'detail': 'Use the same-origin app to submit requests.'}, status_code=403,
                                    headers={'Cache-Control': 'no-store'})
        response = await call_next(request)
        if fresh:
            response.set_cookie('rag_workspace', request.state.owner, httponly=True, samesite='strict', secure=request.url.scheme == 'https', max_age=31536000)
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        if request.url.path.startswith('/api/auth/'):
            response.headers['Referrer-Policy'] = 'no-referrer'
        return response

    @app.exception_handler(EvaluationDataError)
    async def bad_dataset(request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=422)

    def owned(request, kind, key):
        result = catalog.get(kind, request.state.owner, key)
        if not result:
            raise HTTPException(404, f'{kind.capitalize()} not found.')
        return result

    def public_document(doc):
        return {k: v for k, v in doc.items() if k not in ('file_path', 'index_path')}

    def database(doc):
        from index_runtime import index_is_ready
        if not doc.get('index_path') or not index_is_ready(root / doc['index_path'], document_hash=doc['sha256']):
            raise HTTPException(409, 'Prepare this document before searching or evaluating it.')
        path = str(root / doc['index_path'])
        if path not in databases:
            from rag_pipeline import load_vector_store
            databases[path] = load_vector_store(path)
        return databases[path]

    def persist_doc(owner, doc, **changes):
        values = {k: v for k, v in doc.items() if k not in ('id', 'created_at')}
        return catalog.put('document', owner, {**values, **changes}, resource_id=doc['id'])

    def prepare_one(owner, doc, job):
        from dataclasses import replace
        settings = ExtractionSettings.from_env()
        sources = doc.get('source_documents')
        if sources:
            settings = replace(settings, max_pages=settings.max_pages * len(sources), max_file_mb=settings.max_file_mb * len(sources))
        identity = extraction_cache_key(doc['sha256'], settings) + ':fast-v2'
        if sources:
            identity += json.dumps(sources, sort_keys=True)
        key = hashlib.sha256(identity.encode()).hexdigest()
        persist_doc(owner, doc, status='preparing', preparation_error=None)
        try:
            options = {'source_documents': sources, 'storage_root': root} if sources else {}
            result = prepare_document(root / doc['file_path'], doc['sha256'], key, settings, job, **options)
            databases[str(root / result['index_path'])] = result['database']
            return persist_doc(owner, doc, status='ready', index_path=result['index_path'], stats=result['stats'],
                               timings=result['timings'], extraction_cache_hit=result['extraction_cache_hit'], preparation_error=None)
        except Exception as exc:
            persist_doc(owner, doc, status='needs_preparation', preparation_error=Jobs.safe_snapshot({'error': str(exc)})['error'])
            raise

    def preparation(owner, doc, *, retry_of=None):
        for entry in list(jobs.entries.values()):
            ids = (entry.get('request') or {}).get('document_ids', [])
            if entry['owner'] == owner and (entry['document_id'] == doc['id'] or doc['id'] in ids) and entry['kind'] == 'prepare' and entry['job'].snapshot()['status'] not in TERMINAL:
                raise HTTPException(409, 'This document is already being prepared.')
        def target(job):
            return public_document(prepare_one(owner, doc, job))
        return jobs.submit(owner, doc['id'], 'prepare', target,
                           request={'operation': 'prepare'}, retry_of=retry_of)

    def require_documents(owner, ids):
        documents = [catalog.get('document', owner, key) for key in ids]
        if any(doc is None for doc in documents):
            raise HTTPException(404, 'Document not found.')
        return documents

    def collection(owner, documents):
        if len(documents) < 2 or any(doc.get('source_documents') for doc in documents):
            raise HTTPException(422, 'Select two or three original PDFs to combine.')
        ids = [doc['id'] for doc in documents]
        existing = next((doc for doc in catalog.list('document', owner)
                         if [source['document_id'] for source in doc.get('source_documents', [])] == ids), None)
        if existing and (root / existing['file_path']).is_file():
            return existing
        try:
            values = merge_pdfs(documents, root, max_pages_per_file=ExtractionSettings.from_env().max_pages)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return catalog.put('document', owner, {**values, 'name': ('Combined: ' + ' + '.join(doc['name'] for doc in documents))[:200],
            'status': 'needs_preparation', 'stats': {}}, resource_id=existing['id'] if existing else hashlib.sha256(('combined:' + owner + json.dumps(ids)).encode()).hexdigest()[:32])

    def batch_preparation(owner, documents, *, retry_of=None):
        if not retry_of and all(doc.get('status') == 'ready' for doc in documents):
            return None
        def target(job):
            errors = []
            for index, doc in enumerate(documents):
                job.check_cancelled()
                if doc.get('status') == 'ready':
                    continue
                child = BatchStageJob(job, index, len(documents), doc['name'])
                try:
                    prepare_one(owner, doc, child)
                except Exception as exc:
                    job.check_cancelled()
                    errors.append(doc['name'] + ': ' + Jobs.safe_snapshot({'error': str(exc)})['error'])
            if errors:
                raise RuntimeError(f'{len(errors)} of {len(documents)} PDFs failed preparation. ' + '; '.join(errors))
            return public_document(catalog.get('document', owner, documents[0]['id']))
        for doc in documents:
            if doc.get('status') != 'ready':
                persist_doc(owner, doc, status='preparing')
        try:
            return jobs.submit(owner, documents[0]['id'], 'prepare', target,
                               request={'operation': 'prepare_batch', 'document_ids': [doc['id'] for doc in documents]}, retry_of=retry_of)
        except Exception:
            for doc in documents:
                if doc.get('status') != 'ready':
                    persist_doc(owner, doc, status=doc.get('status', 'needs_preparation'))
            raise

    @app.get('/api/health')
    def health():
        return {'status': 'ok'}

    @app.get('/api/bootstrap')
    def bootstrap(request: Request):
        owner = request.state.owner
        if auth_settings.mode == 'local' and catalog.claim_legacy(owner):
            from index_runtime import index_is_ready, read_index_manifest
            for path in sorted((root / 'docs').glob('*.pdf')):
                sha = path.stem
                if not re.fullmatch(r'[a-f0-9]{64}', sha):
                    continue
                candidates = [p for p in (root / 'db').glob('chroma_*') if index_is_ready(p, document_hash=sha)]
                index = candidates[0] if candidates else None
                doc = catalog.put('document', owner, {'name': 'RAG Evaluation Handbook.pdf' if sha == '65f6921adb3976f1c44d9eb1574d3b8ede189fc96d53fec0c878f483368571d0' else f'Document {sha[:8]}.pdf', 'sha256': sha, 'size_bytes': path.stat().st_size,
                    'file_path': str(path.relative_to(root)), 'index_path': str(index.relative_to(root)) if index else None,
                    'status': 'ready' if index else 'needs_preparation', 'stats': {}, 'chunk_count': (read_index_manifest(index) or {}).get('count') if index else 0})
                for manifest_path in (root / 'evaluation_runs').glob('*/manifest.json'):
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                        if manifest.get('document_hash') == sha:
                            catalog.put('run', owner, {'manifest': clean(manifest), 'directory': str(manifest_path.parent.relative_to(root))}, doc['id'])
                    except (ValueError, OSError):
                        continue
        return {'generation_model': os.getenv('GENERATION_MODEL', 'gemma-4-26b-a4b-it'), 'evaluation_model': os.getenv('EVALUATION_MODEL', 'gemma-4-26b-a4b-it'),
                'independent_reference_verifier': os.getenv('GENERATION_MODEL') != os.getenv('EVALUATION_MODEL'),
                'max_batch_pdfs': MAX_BATCH_PDFS,
                'max_file_mb': ExtractionSettings.from_env().max_file_mb, 'mode': auth_settings.mode, 'reranker_configured': bool(os.getenv('COHERE_API_KEY')) and os.getenv('RERANKER_PROVIDER', 'cohere') != 'none'}

    @app.get('/api/documents')
    def documents(request: Request):
        return [public_document(doc) for doc in catalog.list('document', request.state.owner)]

    async def receive_pdf(owner, file):
        limit = ExtractionSettings.from_env().max_file_mb * 1024 * 1024
        directory = root / 'docs'
        directory.mkdir(exist_ok=True)
        digest, size, header, temporary = hashlib.sha256(), 0, b'', None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False, suffix='.upload') as stream:
                temporary = Path(stream.name)
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f'PDF exceeds the {limit // 1024 // 1024} MB limit.')
                    if len(header) < 1024:
                        header += chunk[:1024-len(header)]
                    digest.update(chunk)
                    stream.write(chunk)
            if b'%PDF-' not in header:
                raise HTTPException(422, 'The uploaded file does not contain a PDF header.')
            sha = digest.hexdigest()
            path = directory / f'{sha}.pdf'
            if not path.exists():
                temporary.replace(path)
            existing = next((d for d in catalog.list('document', owner) if d['sha256'] == sha and not d.get('source_documents')), None)
            return existing or catalog.put('document', owner, {'name': (file.filename or 'Document.pdf').replace('\\', '/').split('/')[-1][:200], 'sha256': sha, 'size_bytes': size, 'file_path': str(path.relative_to(root)), 'status': 'needs_preparation', 'stats': {}})
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
            await file.close()

    @app.post('/api/documents', status_code=202)
    async def upload(request: Request, file: UploadFile = File(...)):
        doc = await receive_pdf(request.state.owner, file)
        return {'document': public_document(doc), 'job': None if doc['status'] == 'ready' else preparation(request.state.owner, doc)}

    @app.post('/api/documents/batch', status_code=202)
    async def upload_batch(request: Request, files: list[UploadFile] = File(...), mode: Literal['separate', 'combined'] = Form('separate')):
        try:
            if not 1 <= len(files) <= MAX_BATCH_PDFS:
                raise HTTPException(422, f'Upload one to {MAX_BATCH_PDFS} PDFs at a time.')
            documents, errors = [], []
            for file in files:
                try:
                    doc = await receive_pdf(request.state.owner, file)
                    if not any(item['id'] == doc['id'] for item in documents):
                        documents.append(doc)
                except HTTPException as exc:
                    errors.append({'name': (file.filename or 'PDF')[:200], 'error': str(exc.detail)})
            combined = None
            if mode == 'combined' and not errors:
                try:
                    combined = await asyncio.to_thread(collection, request.state.owner, documents)
                except HTTPException as exc:
                    errors.append({'name': 'Combined PDFs', 'error': str(exc.detail)})
            targets = [combined] if combined else documents
            work = None
            if targets and (mode == 'separate' or combined):
                work = batch_preparation(request.state.owner, targets)
            return {'documents': [public_document(doc) for doc in documents], 'combined': public_document(combined) if combined else None,
                    'job': work, 'errors': errors}
        finally:
            for file in files:
                await file.close()

    @app.post('/api/documents/{document_id}/prepare', status_code=202)
    def prepare(request: Request, document_id: str):
        return preparation(request.state.owner, owned(request, 'document', document_id))

    @app.get('/api/documents/{document_id}/pdf')
    def pdf(request: Request, document_id: str):
        doc = owned(request, 'document', document_id)
        return FileResponse(root / doc['file_path'], media_type='application/pdf', filename=doc['name'], content_disposition_type='inline')

    @app.get('/api/documents/{document_id}/assets/{name}')
    def asset(request: Request, document_id: str, name: str):
        doc = owned(request, 'document', document_id)
        if not re.fullmatch(r'[a-f0-9]{64}\.jpg', name) or not doc.get('index_path'):
            raise HTTPException(404, 'Image not found.')
        path = root / doc['index_path'] / 'assets' / name
        if not path.is_file():
            raise HTTPException(404, 'Image not found.')
        return FileResponse(path, media_type='image/jpeg')

    @app.get('/api/documents/{document_id}/chunks')
    def chunks(request: Request, document_id: str, offset: int = Query(0, ge=0), limit: int = Query(12, ge=1, le=50)):
        db = database(owned(request, 'document', document_id))
        from langchain_core.documents import Document
        records = db.get(offset=offset, limit=limit)
        return {'total': db._collection.count(), 'offset': offset, 'items': [source(Document(page_content=text, metadata={**(metadata or {}), 'chunk_id': key}), document_id) for key, text, metadata in zip(records['ids'], records['documents'], records['metadatas'])]}

    @app.post('/api/documents/{document_id}/answers', status_code=202)
    def ask(request: Request, document_id: str, body: Question):
        return start_answer(request.state.owner, owned(request, 'document', document_id), body)

    def start_answer(owner, doc, body, *, retry_of=None):
        document_id, db = doc['id'], database(doc)
        def target(job):
            result = answer_question(db, body.question.strip(), job, use_reranker=body.rerank)
            return catalog.put('answer', owner, {'query': result['query'], 'answer': result['answer'], 'sources': [source(d, document_id) for d in result['documents']], 'timings': clean(result['timings'])}, document_id, resource_id=job.key)
        return jobs.submit(owner, document_id, 'answer', target,
                           request={'operation': 'answer', 'body': body.model_dump()}, retry_of=retry_of)

    @app.get('/api/documents/{document_id}/answers')
    def answers(request: Request, document_id: str):
        owned(request, 'document', document_id)
        return catalog.list('answer', request.state.owner, document_id)

    @app.post('/api/documents/{document_id}/retrieval', status_code=202)
    def retrieve(request: Request, document_id: str, body: Retrieval):
        return start_retrieval(request.state.owner, owned(request, 'document', document_id), body)

    def start_retrieval(owner, doc, body, *, retry_of=None):
        document_id, db = doc['id'], database(doc)
        return jobs.submit(owner, document_id, 'retrieval', lambda job: retrieval_experiment(db, body, job, document_id),
                           request={'operation': 'retrieval', 'body': body.model_dump()}, retry_of=retry_of)

    @app.get('/api/documents/{document_id}/datasets/template')
    def template(request: Request, document_id: str):
        return JSONResponse(dataset_template(owned(request, 'document', document_id)['sha256']), headers={'Content-Disposition': 'attachment; filename="evaluation-template.json"'})

    def save_dataset(request, doc, content, format, name):
        cases = load_evaluation_dataset(content, format=format, expected_document_hash=doc['sha256'])
        return catalog.put('dataset', request.state.owner, {'name': name, 'origin': 'imported', 'cases': [c.to_dict() for c in cases], 'count': len(cases)}, doc['id'])

    @app.post('/api/documents/{document_id}/datasets', status_code=201)
    async def dataset_upload(request: Request, document_id: str, file: UploadFile = File(...)):
        doc = owned(request, 'document', document_id)
        content = await file.read(10*1024*1024 + 1)
        await file.close()
        return save_dataset(request, doc, content, Path(file.filename or '').suffix.lstrip('.').lower(), (file.filename or 'Dataset')[:200])

    @app.post('/api/documents/{document_id}/datasets/sample', status_code=201)
    def sample(request: Request, document_id: str):
        doc = owned(request, 'document', document_id)
        path = root / 'evaluation_datasets' / 'handbook-source-checked.json'
        if not path.is_file():
            raise HTTPException(404, 'Reviewed sample is not installed. Import your JSON or CSV dataset.')
        return save_dataset(request, doc, path.read_bytes(), 'json', 'Source-checked handbook questions')

    @app.get('/api/documents/{document_id}/datasets')
    def datasets(request: Request, document_id: str):
        owned(request, 'document', document_id)
        return catalog.list('dataset', request.state.owner, document_id)

    @app.get('/api/datasets/{dataset_id}/export')
    def dataset_export(request: Request, dataset_id: str):
        dataset = owned(request, 'dataset', dataset_id)
        value = {'questions': dataset['cases'], 'provenance': dataset.get('provenance'),
                 'warnings': dataset.get('warnings', []), 'origin': dataset.get('origin', 'imported')}
        if dataset['cases']:
            value['document_hash'] = dataset['cases'][0]['document_hash']
        return JSONResponse(clean(value), headers={'Content-Disposition': 'attachment; filename="evaluation-questions.json"'})

    def report(run):
        path = root / run['directory'] / 'summary.csv'
        rows, warnings = read_summary(path)
        manifest = {**run['manifest'], 'warnings': list(run['manifest'].get('warnings', [])) + warnings}
        from evaluation_diagnostics import report_diagnostics
        manifest, rows, failures = report_diagnostics(root / run['directory'], manifest, rows)
        return {k: v for k, v in run.items() if k != 'directory'} | {'manifest': manifest, 'summary': rows, 'metric_failures': failures}

    def save_run(owner, document_id, frame, dataset_id=None, automatic=None, *, job_id=None):
        path = Path(frame.attrs['manifest_path']).resolve()
        if not path.is_relative_to(root / 'evaluation_runs'):
            raise RuntimeError('Evaluation output is outside the configured report store.')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        run = catalog.put('run', owner, {'manifest': clean(manifest), 'document_id': document_id,
            'directory': str(path.parent.relative_to(root)),
            'dataset_id': dataset_id, 'automatic': automatic,
            'cache_reused': bool(frame.attrs.get('cache_reused'))}, document_id, resource_id=job_id)
        return report(run)

    @app.post('/api/documents/{document_id}/evaluations/automatic', status_code=202)
    def automatic_evaluate(request: Request, document_id: str, body: AutomaticEvaluation):
        return start_automatic_evaluation(request.state.owner, owned(request, 'document', document_id), body)

    def start_automatic_evaluation(owner, doc, body, *, retry_of=None):
        document_id = doc['id']
        database(doc)
        def target(job):
            return automatic_result(owner, doc, body, job)
        return jobs.submit(owner, document_id, 'evaluation', target,
                           request={'operation': 'automatic', 'body': body.model_dump()}, retry_of=retry_of)

    def automatic_result(owner, doc, body, job):
        document_id = doc['id']
        started = time.perf_counter()
        if isinstance(job, BatchStageJob):
            job.set_stage(1)
        job.progress(0, 1, 'Generating questions and checking their reference evidence')
        generated = asyncio.run(generate_dataset(str(root / doc['index_path']),
            expected_document_hash=doc['sha256'], question_count=body.question_count,
            progress_callback=job.progress, cancellation_event=job.cancellation_event,
            output_dir=str(root / 'evaluation_datasets' / 'generated'), force=body.force))
        job.check_cancelled()
        cases = load_evaluation_dataset(generated, expected_document_hash=doc['sha256'])
        raw = {case['id']: case for case in generated['questions']}
        rows = [{**raw[case.id], **case.to_dict()} for case in cases]
        provenance = clean(generated.get('provenance', {}))
        cached_dataset = next((d for d in catalog.list('dataset', owner, document_id)
            if d.get('origin') == 'synthetic' and d.get('provenance') == provenance and d.get('cases') == rows), None)
        dataset = cached_dataset or catalog.put('dataset', owner, {
            'name': f'Automatic source-verified questions ({len(cases)})', 'origin': 'synthetic',
            'cases': clean(rows), 'count': len(cases), 'provenance': provenance,
            'warnings': clean(generated.get('warnings', []))}, document_id)
        generation_seconds = round(time.perf_counter() - started, 3)
        job.check_cancelled()
        if isinstance(job, BatchStageJob):
            job.set_stage(2)
        job.progress(0, 1, f'Evaluating {len(cases)} automatically verified questions')
        frame = evaluate_document(str(root / doc['index_path']), generated, doc['sha256'],
            body.profile, job, force=body.force, output_dir=str(root / 'evaluation_runs'))
        automatic = {'question_generation_seconds': generation_seconds,
            'question_generation_cache_reused': bool(generated.get('cache_reused')),
            'requested_questions': body.question_count, 'accepted_questions': len(cases)}
        return save_run(owner, document_id, frame, dataset['id'], automatic, job_id=job.key)

    def start_batch_evaluation(owner, body, *, retry_of=None):
        documents = require_documents(owner, body.document_ids)
        if any(doc.get('source_documents') for doc in documents):
            raise HTTPException(422, 'Select original PDFs for a batch. An existing combined PDF can be evaluated individually.')
        targets = [collection(owner, documents)] if body.mode == 'combined' else documents
        for entry in list(jobs.entries.values()):
            saved = entry.get('request') or {}
            active_ids = saved.get('document_ids', saved.get('body', {}).get('document_ids', []))
            if entry['owner'] == owner and entry['job'].snapshot()['status'] not in TERMINAL and (
                entry['document_id'] in [doc['id'] for doc in targets] or set(active_ids) & set(body.document_ids)):
                raise HTTPException(409, 'Work is already running for a selected PDF. Wait or cancel it first.')
        def target(job):
            results, limited = [], None
            for index, original in enumerate(targets):
                job.check_cancelled()
                if limited:
                    results.append({'document_id': original['id'], 'name': original['name'], 'error': limited})
                    continue
                child = BatchStageJob(job, index, len(targets), original['name'], stages=3)
                doc = catalog.get('document', owner, original['id'])
                try:
                    if doc.get('status') != 'ready':
                        doc = prepare_one(owner, doc, child)
                    result = automatic_result(owner, doc, body, child)
                    results.append({'document_id': doc['id'], 'name': doc['name'], 'run': result})
                    if result['manifest'].get('provider_limit'):
                        limited = result['manifest']['provider_limit'].get('reason', 'The provider blocked further requests in this batch.')
                except (Exception, asyncio.CancelledError) as exc:
                    job.check_cancelled()
                    results.append({'document_id': original['id'], 'name': original['name'], 'error': Jobs.safe_snapshot({'error': str(exc)})['error']})
                job.progress((index + 1) * 100, len(targets) * 100, f'Finished PDF {index + 1}/{len(targets)}')
            status = 'completed' if all('run' in result and result['run']['manifest'].get('status') == 'completed' for result in results) else 'partial' if any('run' in result for result in results) else 'failed'
            return {'mode': body.mode, 'status': status, 'batch_results': results}
        for doc in targets:
            if doc.get('status') != 'ready':
                persist_doc(owner, doc, status='preparing')
        try:
            info = jobs.submit(owner, targets[0]['id'], 'evaluation_batch', target,
                               request={'operation': 'evaluation_batch', 'body': body.model_dump()}, retry_of=retry_of)
            return {**info, 'evaluation_setup': body.model_dump()}
        except Exception:
            for doc in targets:
                if doc.get('status') != 'ready':
                    persist_doc(owner, doc, status=doc.get('status', 'needs_preparation'))
            raise

    @app.post('/api/evaluations/batch', status_code=202)
    def batch_evaluate(request: Request, body: BatchEvaluation):
        return start_batch_evaluation(request.state.owner, body)

    @app.post('/api/documents/{document_id}/evaluations', status_code=202)
    def evaluate(request: Request, document_id: str, body: Evaluation):
        return start_evaluation(request.state.owner, owned(request, 'document', document_id), body)

    def start_evaluation(owner, doc, body, *, retry_of=None):
        document_id = doc['id']
        database(doc)
        dataset = catalog.get('dataset', owner, body.dataset_id)
        if not dataset:
            raise HTTPException(404, 'Dataset not found.')
        load_evaluation_dataset(dataset['cases'], expected_document_hash=doc['sha256'])
        evaluation_dataset = {'questions': dataset['cases'],
            'provenance': dataset.get('provenance'), 'warnings': dataset.get('warnings', [])}
        def target(job):
            frame = evaluate_document(str(root / doc['index_path']), evaluation_dataset, doc['sha256'], body.profile, job,
                force=body.force, output_dir=str(root / 'evaluation_runs'))
            return save_run(owner, document_id, frame, dataset['id'], job_id=job.key)
        return jobs.submit(owner, document_id, 'evaluation', target,
                           request={'operation': 'evaluation', 'body': body.model_dump()}, retry_of=retry_of)

    @app.get('/api/documents/{document_id}/runs')
    def runs(request: Request, document_id: str):
        owned(request, 'document', document_id)
        return [report(run) for run in catalog.list('run', request.state.owner, document_id)]

    @app.get('/api/runs/{run_id}/export/{kind}')
    def export(request: Request, run_id: str, kind: Literal['summary', 'samples', 'manifest']):
        run = owned(request, 'run', run_id)
        name = {'summary': 'summary.csv', 'samples': 'samples.jsonl', 'manifest': 'manifest.json'}[kind]
        path = root / run['directory'] / name
        if not path.is_file():
            raise HTTPException(404, 'This run has no completed export yet.')
        return FileResponse(path, filename=name)

    @app.get('/api/jobs')
    def pending_jobs(request: Request):
        return jobs.pending(request.state.owner)

    @app.post('/api/jobs/{job_id}/retry', status_code=202)
    def retry_job(request: Request, job_id: str):
        owner = request.state.owner
        def dispatch(entry, retry_of):
            doc = catalog.get('document', owner, entry['document_id'])
            if not doc:
                raise HTTPException(404, 'Document not found.')
            saved = entry['request']
            operation = saved.get('operation')
            if operation == 'prepare_batch':
                return batch_preparation(owner, require_documents(owner, saved['document_ids']), retry_of=retry_of)
            body = dict(saved.get('body', {}))
            # A retry reuses valid completed stages even if the original run forced recomputation.
            if operation in {'automatic', 'evaluation'}:
                body['force'] = False
            if operation == 'prepare':
                return preparation(owner, doc, retry_of=retry_of)
            if operation == 'evaluation_batch':
                body['force'] = False
                return start_batch_evaluation(owner, BatchEvaluation.model_validate(body), retry_of=retry_of)
            handlers = {'answer': (start_answer, Question), 'retrieval': (start_retrieval, Retrieval),
                        'automatic': (start_automatic_evaluation, AutomaticEvaluation),
                        'evaluation': (start_evaluation, Evaluation)}
            if operation not in handlers:
                raise HTTPException(409, 'This saved operation is no longer supported. Start a new operation from its page.')
            handler, schema = handlers[operation]
            try:
                validated = schema.model_validate(body)
            except ValidationError:
                raise HTTPException(409, 'The saved settings are no longer valid. Start a new operation from its page.')
            return handler(owner, doc, validated, retry_of=retry_of)
        return jobs.retry(owner, job_id, dispatch)

    @app.get('/api/jobs/{job_id}')
    def job(request: Request, job_id: str):
        return jobs.snapshot(request.state.owner, job_id)

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel(request: Request, job_id: str):
        jobs.get(request.state.owner, job_id)['job'].cancel()
        return jobs.snapshot(request.state.owner, job_id)

    @app.get('/api/jobs/{job_id}/events')
    async def events(request: Request, job_id: str):
        owner = request.state.owner
        jobs.get(owner, job_id)
        async def stream():
            while not await request.is_disconnected():
                if auth_settings.mode == 'oidc' and not auth.session(request):
                    return
                value = jobs.snapshot(owner, job_id)
                yield 'data: ' + json.dumps(value, allow_nan=False) + '\n\n'
                if value['status'] in TERMINAL:
                    return
                await asyncio.sleep(0.3)
        return StreamingResponse(stream(), media_type='text/event-stream', headers={'X-Accel-Buffering': 'no'})

    dist = root / 'frontend' / 'dist'
    if (dist / 'assets').is_dir():
        app.mount('/assets', StaticFiles(directory=dist / 'assets'), name='frontend-assets')

    @app.get('/{path:path}', include_in_schema=False)
    def frontend(path: str):
        if path.startswith('api/'):
            raise HTTPException(404, 'API route not found.')
        if not (dist / 'index.html').is_file():
            return JSONResponse({'detail': 'Build the frontend: cd frontend; npm ci; npm run build'}, status_code=503)
        return FileResponse(dist / 'index.html', headers={'Cache-Control': 'no-cache'})
    # A result may commit just before a process stops, before its terminal job snapshot.
    # Reconcile those artifacts by job ID without invoking any provider.
    for entry in catalog.pending_jobs():
        if not entry['snapshot'].get('recovery_required'):
            continue
        kind = {'answer': 'answer', 'evaluation': 'run'}.get(entry['kind'])
        result = catalog.get(kind, entry['owner'], entry['id']) if kind else None
        if result:
            if kind == 'run':
                result = report(result)
            value = {**entry['snapshot'], 'status': 'succeeded', 'message': 'Complete',
                     'done': entry['snapshot']['total'], 'result': result,
                     'error': None, 'recovery_required': False}
            catalog.save_job(entry['id'], entry['owner'], entry['document_id'], entry['kind'], jobs.safe_snapshot(value))
    return app
