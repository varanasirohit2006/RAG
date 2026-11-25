"""Explain recorded failures without replacing scores or their source evidence."""
import ast
import json
from pathlib import Path

from backend.provider_limits import limit_from_body, limit_message

QUALITY_METRICS = ('faithfulness', 'factual_correctness', 'answer_relevancy', 'context_precision', 'context_recall')


def legacy_limit(error):
    if not isinstance(error, str) or '429' not in error:
        return None
    start = error.find('{')
    if start < 0:
        return None
    try:
        body = ast.literal_eval(error[start:])
    except (ValueError, SyntaxError):
        return None
    return limit_from_body(body) if isinstance(body, dict) else None


def report_diagnostics(directory, manifest, rows):
    try:
        records = [json.loads(line) for line in (Path(directory) / 'samples.jsonl').read_text(encoding='utf-8').splitlines() if line]
    except (OSError, ValueError):
        return manifest, rows, []
    limit = manifest.get('provider_limit')
    from backend.api.runtime import Jobs
    sample_failures = [{'strategy': record.get('strategy'), 'question_id': record.get('question_id'),
                        'reason': Jobs.safe_snapshot({'error': record.get('error') or 'Sample failed before scoring.'})['error']}
                       for record in records if record.get('status') == 'failed']
    manifest = {**manifest, 'partial_samples': sum(record.get('status') == 'partial' for record in records),
                'answer_failed_samples': sum(record.get('status') == 'failed' and not record.get('response') for record in records),
                'sample_failures': sample_failures}
    failures = []
    for record in records:
        for name, metric in record.get('metrics', {}).items():
            if name not in QUALITY_METRICS or not isinstance(metric, dict):
                continue
            error = metric.get('error', '')
            found = legacy_limit(error)
            if found and found.get('reset_at'):
                limit = {**(limit or {}), **found, 'reason': limit_message(found)}
            if metric.get('status') != 'failed':
                continue
            code = metric.get('provider_status_code')
            if code == 429 or found:
                reason, category = limit_message(limit or found), 'provider_limit'
            elif 'TimeoutError' in error:
                configured = manifest.get('configuration', {}).get('metric_timeout_seconds') or manifest.get('configuration', {}).get('judge', {}).get('timeout')
                reason = error
                if error == 'TimeoutError: TimeoutError':
                    reason = f'The judge timed out under the configured {configured}-second metric budget.' if configured else 'The judge timed out while evaluating this metric.'
                category = 'timeout'
            else:
                reason = Jobs.safe_snapshot({'error': error or metric.get('reason') or 'Evaluation failed.'})['error']
                category = 'evaluation_error'
            failures.append({'strategy': record.get('strategy'), 'question_id': record.get('question_id'),
                             'question': record.get('question'), 'metric': name, 'category': category, 'reason': reason})
    rows = [dict(row) for row in rows]
    usable = [{**record, 'status': record.get('status', 'unknown')} for record in records
              if isinstance(record.get('strategy'), str) and isinstance(record.get('metrics', {}), dict)]
    if rows and usable and not any(key.endswith('_matched_count') for row in rows for key in row):
        from backend.advanced_evaluation import summarize_records
        recalculated = {row['strategy']: row for row in summarize_records(usable).to_dict(orient='records')}
        for row in rows:
            row.update({key: value for key, value in recalculated.get(row.get('strategy'), {}).items()
                        if '_matched' in key or '_fresh_' in key or key in {'generation_reused_samples', 'judge_reused_samples'}})
    for row in rows:
        matching = [record for record in records if record.get('strategy') == row.get('strategy')]
        for name in QUALITY_METRICS:
            row[name + '_blocked'] = sum(record.get('metrics', {}).get(name, {}).get('status') == 'skipped'
                and record.get('metrics', {}).get(name, {}).get('provider_status_code') == 429 for record in matching)
    if limit:
        manifest = {**manifest, 'provider_limit': {**limit, 'reason': limit_message(limit)}}
    from backend.api.runtime import clean
    return manifest, clean(rows), failures
