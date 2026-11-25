"""Keep provider state and judging-step caches isolated from the user's runs."""
import pytest


@pytest.fixture(autouse=True)
def isolated_evaluation_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_EVALUATION_CACHE_DIR', str(tmp_path / 'evaluation-cache'))
    monkeypatch.setenv('APP_AUTH_MODE', 'local')
    monkeypatch.setenv('EVALUATION_PROVIDER_PREFLIGHT', 'false')
    for name in ('GENERATION_REQUEST_TIMEOUT_SECONDS', 'SYNTHETIC_REQUEST_TIMEOUT_SECONDS', 'EVALUATION_REQUEST_TIMEOUT'):
        monkeypatch.delenv(name, raising=False)
