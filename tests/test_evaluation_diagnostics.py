import json

from backend.evaluation_diagnostics import report_diagnostics


def test_legacy_timeouts_and_daily_quota_are_reported_without_changing_scores(tmp_path):
    body = {'error': {'message': 'Rate limit exceeded: free-models-per-day', 'metadata': {
        'limit_source': 'openrouter_free_tier_daily', 'headers': {'X-RateLimit-Limit': '50',
            'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1795046400000'}}}}
    records = [{'strategy': 'Keyword', 'question_id': 'q1', 'metrics': {
        'factual_correctness': {'status': 'failed', 'error': 'TimeoutError: TimeoutError'},
        'faithfulness': {'status': 'completed', 'value': 1}}},
        {'strategy': 'Keyword', 'question_id': 'q2', 'metrics': {
            'faithfulness': {'status': 'failed', 'error': 'RateLimitError: Error code: 429 - ' + repr(body), 'provider_status_code': 429}}},
        {'strategy': 'Vector', 'question_id': 'q1', 'metrics': {
            'faithfulness': {'status': 'skipped', 'provider_status_code': 429}}}]
    path = tmp_path / 'samples.jsonl'
    path.write_text('\n'.join(json.dumps(record) for record in records))
    original = path.read_bytes()
    rows = [{'strategy': 'Keyword', 'faithfulness': 1.0}, {'strategy': 'Vector', 'faithfulness': None}]
    manifest, result, errors = report_diagnostics(tmp_path, {'configuration': {'judge': {'timeout': 90}}}, rows)
    assert path.read_bytes() == original and rows[0] == {'strategy': 'Keyword', 'faithfulness': 1.0}
    assert result[0]['faithfulness'] == 1.0 and result[1]['faithfulness'] is None
    assert result[1]['faithfulness_blocked'] == 1
    assert errors[0]['category'] == 'timeout' and '90-second' in errors[0]['reason']
    assert errors[1]['category'] == 'provider_limit'
    assert manifest['provider_limit']['scope'] == 'daily' and manifest['provider_limit']['remaining'] == 0
