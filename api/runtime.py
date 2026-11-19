"""Bounded jobs, browser-safe serializers and optional retrieval experiments."""
import json
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import HTTPException
from processing_jobs import BackgroundJob

TERMINAL = {'succeeded', 'failed', 'cancelled'}
METRICS = {'faithfulness', 'factual_correctness', 'answer_relevancy', 'context_precision',
           'context_recall', 'multimodal_faithfulness', 'abstention_accuracy',
           'precision_at_k', 'recall_at_k', 'mrr', 'ndcg_at_k', 'normalized_exact_match', 'token_f1',
           'hit_rate_at_k', 'map_at_k', 'citation_page_validity', 'citation_page_coverage', 'citation_source_validity',
           'failure_rate', 'partial_rate', 'error_rate', 'false_answer_rate', 'over_abstention_rate'}


def read_summary(path):
    """Legacy reports receive the same range checks as newly generated scores."""
    import pandas as pd
    from evaluation_data import validate_score
    warnings = []
    try:
        rows = pd.read_csv(path).to_dict(orient='records') if path.is_file() else []
    except pd.errors.EmptyDataError:
        rows = []
    except (OSError, ValueError, pd.errors.ParserError):
        return [], ['The saved metric summary could not be read. Inspect the run artifacts.']
    invalid = set()
    for row in rows:
        for name in METRICS:
            for key in (name, name + '_lower95', name + '_upper95'):
                if key not in row or pd.isna(row[key]):
                    continue
                try:
                    row[key] = validate_score(name, row[key])
                except ValueError:
                    row[key] = None
                    invalid.add(key)
    if invalid:
        warnings.append('Invalid saved metric values excluded from display: ' + ', '.join(sorted(invalid)))
    return clean(rows), warnings


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items() if not str(k).endswith(('_path', '_paths', '_directory'))}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, 'item'):
        return clean(value.item())
    return value


def source(document, document_id):
    from evaluation_data import document_id as chunk_id, document_pages
    from generation_runtime import document_payload
    payload = document_payload(document)
    assets = [Path(path).name for path in payload['image_paths']]
    return {'id': chunk_id(document), 'text': payload['raw_text'] or document.page_content,
            'pages': sorted(document_pages(document)), 'tables': payload['tables_html'],
            'images': [f'/api/documents/{document_id}/assets/{name}' for name in assets if re.fullmatch(r'[a-f0-9]{64}\.jpg', name)],
            'relevance_score': document.metadata.get('relevance_score'),
            'original_sources': json.loads(document.metadata.get('source_pages', '[]'))}


class SavedJob:
    def __init__(self, value):
        self.value = value

    def snapshot(self):
        return dict(self.value)

    def cancel(self):
        pass


class Jobs:
    def __init__(self, catalog=None):
        self.lock = threading.RLock()
        self.entries = {}
        self.catalog = catalog
        self.closed = False
        if catalog:
            catalog.recover_jobs()

    @staticmethod
    def safe_snapshot(value):
        value = clean(value)
        if value.get('error'):
            message = value['error']
            for name in ('GOOGLE_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'COHERE_API_KEY'):
                secret = os.getenv(name)
                if secret:
                    message = message.replace(secret, '[redacted]')
            if any(word in message.lower() for word in ('api_key', 'apikey', 'authorization', 'bearer', 'sk-')):
                message = 'Provider request failed. Check server credentials and provider status.'
            value['error'] = re.sub(r'[A-Za-z]:[\\/][^\s]+', '[server file]', message)[:600]
        return value

    def submit(self, owner, document_id, kind, target, *, request=None, retry_of=None):
        with self.lock:
            if self.closed:
                raise HTTPException(503, 'The server is stopping. Retry after it restarts.')
            active = [e for e in self.entries.values() if e['job'].snapshot()['status'] not in TERMINAL]
            if len(active) >= 4 or sum(e['owner'] == owner for e in active) >= 2:
                raise HTTPException(429, 'Work queue is full. Wait for a running job or cancel it.')
            for key in list(self.entries):
                if len(self.entries) >= 100 and self.entries[key]['job'].snapshot()['status'] in TERMINAL:
                    del self.entries[key]
            key = uuid.uuid4().hex
            last_saved = 0
            def persist(value):
                nonlocal last_saved
                if self.catalog:
                    with self.lock:
                        if self.closed:
                            return
                        now = time.monotonic()
                        if value['status'] in TERMINAL or now - last_saved >= 0.5:
                            self.catalog.save_job(key, owner, document_id, kind, self.safe_snapshot(value))
                            last_saved = now
            if self.catalog:
                self.catalog.save_job(key, owner, document_id, kind, {'status': 'queued', 'done': 0, 'total': 1,
                    'message': 'Queued', 'partial_text': '', 'elapsed_seconds': 0, 'events': [], 'result': None, 'error': None},
                    request=request, retry_of=retry_of)
            self.entries[key] = {'owner': owner, 'document_id': document_id, 'kind': kind,
                                 'request': request, 'job': BackgroundJob(target, key=key, on_change=persist)}
            return {'id': key, 'document_id': document_id, 'kind': kind}

    def get(self, owner, key):
        with self.lock:
            entry = self.entries.get(key)
            if not entry and self.catalog:
                stored = self.catalog.get_job(owner, key)
                if stored:
                    entry = {**stored, 'job': SavedJob(stored['snapshot'])}
            if not entry or entry['owner'] != owner:
                raise HTTPException(404, 'Job not found.')
            return entry

    def snapshot(self, owner, key):
        entry = self.get(owner, key)
        value = entry['job'].snapshot()
        stored = self.catalog.get_job(owner, key) if self.catalog else None
        retry_job_id = stored['retry_job_id'] if stored else None
        setup = entry.get('request', {}).get('body') if entry.get('request') and entry['kind'] == 'evaluation_batch' else None
        return {'id': key, 'kind': entry['kind'], 'document_id': entry['document_id'], 'evaluation_setup': setup,
                **self.safe_snapshot(value), 'retry_job_id': retry_job_id,
                'retryable': bool(value.get('recovery_required') and entry.get('request') and not retry_job_id)}

    def pending(self, owner):
        with self.lock:
            keys = [row['id'] for row in self.catalog.pending_jobs(owner)] if self.catalog else [
                key for key, entry in self.entries.items() if entry['owner'] == owner
                and entry['job'].snapshot()['status'] not in TERMINAL]
            return [self.snapshot(owner, key) for key in keys]

    def retry(self, owner, key, dispatch):
        """One explicit retry per interrupted attempt, including concurrent clicks."""
        with self.lock:
            entry = self.get(owner, key)
            stored = self.catalog.get_job(owner, key) if self.catalog else None
            if stored and stored['retry_job_id']:
                next_entry = self.get(owner, stored['retry_job_id'])
                return {'id': stored['retry_job_id'], 'document_id': next_entry['document_id'], 'kind': next_entry['kind']}
            if not entry['job'].snapshot().get('recovery_required') or not entry.get('request'):
                raise HTTPException(409, 'This job cannot be resumed. Start a new operation from its page.')
            return dispatch(entry, key)

    def shutdown(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for key, entry in self.entries.items():
                value = entry['job'].snapshot()
                if value['status'] in TERMINAL:
                    if self.catalog:
                        self.catalog.save_job(key, entry['owner'], entry['document_id'], entry['kind'], self.safe_snapshot(value))
                    continue
                if self.catalog:
                    value.update(status='failed', message='Interrupted by server restart', recovery_required=True,
                                 error='The server stopped before this job finished. Retry to reuse completed work; unfinished model requests may use API credits.')
                    self.catalog.save_job(key, entry['owner'], entry['document_id'], entry['kind'], self.safe_snapshot(value))
                entry['job'].cancel()


def retrieval_experiment(database, request, job, document_id):
    from generation_runtime import generate_final_answer, get_generation_model, content_text
    from retrieval_runtime import get_retrievers, get_reranker
    started = time.perf_counter()
    query = request.question.strip()
    variations = [q.strip() for q in request.variations if q.strip()]
    if request.expand:
        job.progress(0, 3, 'Generating query variations')
        response = get_generation_model().invoke(f'Give exactly three search-query paraphrases, one per line, without explanations, for: {query}')
        variations = [line.strip(' -0123456789.')[:4000] for line in content_text(response.content).splitlines() if line.strip()][:3]
    queries = list(dict.fromkeys([query, *variations]))[:4]
    vector, keyword, _, _ = get_retrievers(database, k=request.k)
    scores, documents, per_query = {}, {}, []
    for index, q in enumerate(queries):
        job.progress(index, len(queries) + 1, f'Searching query {index + 1} of {len(queries)}')
        lists = [(vector.invoke(q), request.vector_weight), (keyword.invoke(q), 1-request.vector_weight)] if request.strategy == 'hybrid' else [(vector.invoke(q) if request.strategy == 'vector' else keyword.invoke(q), 1)]
        local_scores, local_docs = {}, {}
        for ranked, weight in lists:
            for rank, doc in enumerate(ranked, 1):
                key = doc.metadata['chunk_id']
                local_docs[key] = documents[key] = doc
                local_scores[key] = local_scores.get(key, 0) + weight/(60+rank)
        ranked_keys = sorted(local_scores, key=lambda key: (-local_scores[key], key))[:request.k]
        per_query.append({'query': q, 'sources': [source(local_docs[key], document_id) for key in ranked_keys]})
        for rank, key in enumerate(ranked_keys, 1):
            scores[key] = scores.get(key, 0) + 1/(60+rank)
    selected = [documents[key] for key in sorted(scores, key=lambda key: (-scores[key], key))[:request.k]]
    warning = None
    if request.rerank:
        job.progress(len(queries), len(queries)+1, 'Reranking selected passages')
        try:
            reranker = get_reranker(top_n=request.top_n)
            if reranker:
                selected = list(reranker.compress_documents(selected, query))
            else:
                warning = 'Reranker is not configured; fused ranking used.'
        except Exception:
            warning = 'Reranker failed; fused ranking used.'
    selected = selected[:request.top_n]
    answer = generate_final_answer(selected, query, token_callback=job.token, progress_callback=job.progress) if request.generate else None
    return {'query': query, 'queries': queries, 'per_query': per_query, 'sources': [source(doc, document_id) for doc in selected],
            'answer': answer, 'warning': warning, 'timings': {'total_seconds': round(time.perf_counter()-started, 3)}}
