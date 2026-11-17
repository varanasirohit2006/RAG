"""Bounded jobs, browser-safe serializers and optional retrieval experiments."""
import json
import math
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
            'relevance_score': document.metadata.get('relevance_score')}


class Jobs:
    def __init__(self):
        self.lock = threading.RLock()
        self.entries = {}

    def submit(self, owner, document_id, kind, target):
        with self.lock:
            active = [e for e in self.entries.values() if e['job'].snapshot()['status'] not in TERMINAL]
            if len(active) >= 4 or sum(e['owner'] == owner for e in active) >= 2:
                raise HTTPException(429, 'Work queue is full. Wait for a running job or cancel it.')
            for key in list(self.entries):
                if len(self.entries) >= 100 and self.entries[key]['job'].snapshot()['status'] in TERMINAL:
                    del self.entries[key]
            key = uuid.uuid4().hex
            self.entries[key] = {'owner': owner, 'document_id': document_id, 'kind': kind, 'job': BackgroundJob(target)}
            return {'id': key, 'document_id': document_id, 'kind': kind}

    def get(self, owner, key):
        with self.lock:
            entry = self.entries.get(key)
            if not entry or entry['owner'] != owner:
                raise HTTPException(404, 'Job not found.')
            return entry

    def snapshot(self, owner, key):
        entry = self.get(owner, key)
        value = entry['job'].snapshot()
        if value['error']:
            # Provider exceptions can contain URLs or credentials. Expose an actionable,
            # bounded message and never serialize native clients or arbitrary exception reprs.
            message = value['error']
            if any(word in message.lower() for word in ('api_key', 'apikey', 'authorization', 'bearer', 'sk-')):
                message = 'Provider request failed. Check server credentials and provider status.'
            value['error'] = re.sub(r'[A-Za-z]:[\\/][^\s]+', '[server file]', message)[:600]
        return {'id': key, 'kind': entry['kind'], 'document_id': entry['document_id'], **clean(value)}


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
