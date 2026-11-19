"""Real signed OIDC fixtures, session recovery and API ownership without external calls."""
import base64
import hashlib
import json
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi.testclient import TestClient

from api.auth import AuthSettings, LOGIN_COOKIE, SESSION_COOKIE, digest
from api.main import create_app

ISSUER = 'https://identity.example'
BASE = 'https://rag.example'
CLIENT_ID = 'rag-web'
HEADERS = {'X-RAG-Client': '1', 'Origin': BASE}


@pytest.fixture(scope='module')
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class Provider:
    def __init__(self, signing_key):
        self.key = signing_key
        self.nonce = ''
        self.challenge = ''
        self.overrides = {}
        self.algorithm = 'RS256'
        self.kid = 'key-one'
        self.error = None
        self.requests = []
        self.metadata_overrides = {}

    def handle(self, request):
        self.requests.append((request.method, str(request.url)))
        path = request.url.path
        if path == '/.well-known/openid-configuration':
            return httpx.Response(200, json={
                'issuer': ISSUER, 'authorization_endpoint': ISSUER + '/authorize',
                'token_endpoint': ISSUER + '/token', 'jwks_uri': ISSUER + '/keys',
                'response_types_supported': ['code'], 'id_token_signing_alg_values_supported': ['RS256', 'ES256'],
                'code_challenge_methods_supported': ['S256'],
                'token_endpoint_auth_methods_supported': ['client_secret_basic', 'client_secret_post', 'none'],
                **self.metadata_overrides})
        if path == '/keys':
            algorithm = jwt.algorithms.ECAlgorithm if self.algorithm == 'ES256' else jwt.algorithms.RSAAlgorithm
            jwk = json.loads(algorithm.to_jwk(self.key.public_key()))
            return httpx.Response(200, json={'keys': [{**jwk, 'kid': self.kid, 'alg': self.algorithm, 'use': 'sig'}]})
        if path == '/token':
            if self.error:
                return httpx.Response(503, json={'error_description': self.error})
            form = parse_qs(request.content.decode())
            verifier = form['code_verifier'][0]
            assert base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode() == self.challenge
            assert form['redirect_uri'] == [BASE + '/api/auth/callback']
            if 'authorization' in request.headers:
                assert base64.b64decode(request.headers['authorization'].split()[1]).decode() == CLIENT_ID + ':provider-secret'
            else:
                assert form['client_id'] == [CLIENT_ID]
            claims = {'iss': ISSUER, 'sub': form['code'][0], 'aud': CLIENT_ID,
                      'iat': int(time.time()), 'exp': int(time.time()) + 3600, 'nonce': self.nonce,
                      'name': form['code'][0].title(), 'email': 'shared@example.test', 'email_verified': True,
                      **self.overrides}
            token = jwt.encode(claims, self.key if self.algorithm in {'RS256', 'ES256'} else 'untrusted-secret-for-algorithm-test',
                               algorithm=self.algorithm, headers={'kid': self.kid})
            return httpx.Response(200, json={'id_token': token, 'access_token': 'discard-this-access-token', 'token_type': 'Bearer'})
        pytest.fail(f'Unexpected provider request: {request.url}')


@pytest.fixture
def oidc(tmp_path, monkeypatch, signing_key):
    for key, value in {'APP_AUTH_MODE': 'oidc', 'OIDC_ISSUER': ISSUER, 'OIDC_CLIENT_ID': CLIENT_ID,
                       'OIDC_CLIENT_SECRET': 'provider-secret', 'APP_BASE_URL': BASE}.items():
        monkeypatch.setenv(key, value)
    provider = Provider(signing_key)
    transport = httpx.MockTransport(provider.handle)
    app = create_app(tmp_path, warm=False, auth_transport=transport)
    with TestClient(app, base_url=BASE, follow_redirects=False) as client:
        yield app, client, provider, transport, tmp_path


def begin(client, provider):
    response = client.get('/api/auth/login')
    assert response.status_code == 303
    params = parse_qs(urlsplit(response.headers['location']).query)
    assert params['response_type'] == ['code'] and params['scope'] == ['openid profile email']
    assert params['code_challenge_method'] == ['S256']
    provider.nonce, provider.challenge = params['nonce'][0], params['code_challenge'][0]
    cookie = response.headers['set-cookie'].lower()
    assert 'httponly' in cookie and 'secure' in cookie and 'samesite=lax' in cookie
    assert response.headers['cache-control'] == 'no-store' and response.headers['referrer-policy'] == 'no-referrer'
    return params['state'][0]


def sign_in(client, provider, subject='alice'):
    state = begin(client, provider)
    response = client.get('/api/auth/callback', params={'state': state, 'code': subject})
    assert response.status_code == 303 and response.headers['location'] == BASE + '/'
    return client.get('/api/auth/session').json()['user']


def test_oidc_requires_verified_account_and_ignores_workspace_cookie(oidc):
    app, client, _, _, root = oidc
    owner = 'a' * 32
    doc = app.state.catalog.put('document', owner, {'name': 'Private.pdf'})
    client.cookies.set('rag_workspace', owner)
    assert client.get('/api/health').json() == {'status': 'ok'}
    assert client.get('/api/auth/session').json() == {'mode': 'oidc', 'authenticated': False, 'user': None}
    paths = ['/api/bootstrap', '/api/documents', '/api/jobs', f'/api/documents/{doc["id"]}/pdf']
    for path in paths:
        response = client.get(path)
        assert response.status_code == 401 and response.headers['cache-control'] == 'no-store'
    assert client.post('/api/documents', headers=HEADERS).status_code == 401
    with app.state.catalog.connect() as db:
        assert db.execute("SELECT value FROM settings WHERE key='legacy_owner'").fetchone() is None


def test_returning_identity_owns_same_library_after_restart_and_new_browser(oidc):
    app, client, provider, transport, root = oidc
    user = sign_in(client, provider)
    assert user['name'] == 'Alice' and user['email'] == 'shared@example.test'
    (root / 'docs').mkdir()
    (root / 'docs' / 'private.pdf').write_bytes(b'%PDF-1.4 private')
    doc = app.state.catalog.put('document', user['id'], {'name': 'Private.pdf', 'file_path': 'docs/private.pdf'})
    session_token = client.cookies.get(SESSION_COOKIE)
    with app.state.catalog.connect() as db:
        row = db.execute('SELECT * FROM auth_sessions').fetchone()
        assert row['token_hash'] == digest(session_token) and session_token not in str(dict(row))
        assert 'discard-this-access-token' not in str(dict(row))
    restarted = create_app(root, warm=False, auth_transport=transport)
    with TestClient(restarted, base_url=BASE, follow_redirects=False) as other:
        other.cookies.set(SESSION_COOKIE, session_token)
        assert other.get('/api/auth/session').json()['user']['id'] == user['id']
        assert other.get('/api/documents').json()[0]['id'] == doc['id']
        other.cookies.clear()
        provider.overrides = {'name': 'Updated display name', 'email': 'changed@example.test'}
        again = sign_in(other, provider)
        assert again['id'] == user['id'] and again['name'] == 'Updated display name'
        assert other.get(f'/api/documents/{doc["id"]}/pdf').content == b'%PDF-1.4 private'


def test_two_accounts_cannot_read_submit_or_resume_each_others_work(oidc):
    app, client, provider, _, root = oidc
    alice = sign_in(client, provider)
    (root / 'docs').mkdir()
    (root / 'docs' / 'private.pdf').write_bytes(b'%PDF-1.4 private')
    doc = app.state.catalog.put('document', alice['id'], {'name': 'Secret.pdf', 'file_path': 'docs/private.pdf'})
    dataset = app.state.catalog.put('dataset', alice['id'], {'cases': [], 'name': 'Private cases'}, doc['id'])
    run = app.state.catalog.put('run', alice['id'], {'directory': 'evaluation_runs/private'}, doc['id'])
    key = 'b' * 32
    app.state.catalog.save_job(key, alice['id'], doc['id'], 'answer',
        {'status': 'failed', 'recovery_required': True}, request={'operation': 'answer', 'body': {'question': 'private'}})
    assert client.post('/api/auth/logout', headers=HEADERS).status_code == 200
    bob = sign_in(client, provider, 'bob')
    assert bob['id'] != alice['id'] and bob['email'] == alice['email']
    client.cookies.set('rag_workspace', alice['id'])
    assert client.get('/api/documents').json() == [] and client.get('/api/jobs').json() == []
    for suffix in ('pdf', 'answers', 'chunks', 'datasets', 'runs', 'datasets/template', 'assets/' + 'c' * 64 + '.jpg'):
        assert client.get(f'/api/documents/{doc["id"]}/{suffix}').status_code == 404
    for suffix, body in (('prepare', {}), ('answers', {'question': 'Private?'}), ('retrieval', {'question': 'Private?'}),
                         ('evaluations', {'dataset_id': dataset['id']}), ('evaluations/automatic', {'question_count': 1})):
        assert client.post(f'/api/documents/{doc["id"]}/{suffix}', headers=HEADERS, json=body).status_code == 404
    for kind in ('summary', 'samples', 'manifest'):
        assert client.get(f'/api/runs/{run["id"]}/export/{kind}').status_code == 404
    assert client.get(f'/api/jobs/{key}').status_code == 404
    assert client.get(f'/api/jobs/{key}/events').status_code == 404
    assert client.post(f'/api/jobs/{key}/cancel', headers=HEADERS).status_code == 404
    assert client.post(f'/api/jobs/{key}/retry', headers=HEADERS).status_code == 404
    assert client.post('/api/evaluations/batch', headers=HEADERS, json={'document_ids': [doc['id']]}).status_code == 404


def test_expiry_logout_and_session_rotation_are_effective_server_side(oidc):
    app, client, provider, _, _ = oidc
    sign_in(client, provider)
    first = client.cookies.get(SESSION_COOKIE)
    sign_in(client, provider)
    second = client.cookies.get(SESSION_COOKIE)
    assert second != first
    assert app.state.catalog.account_session(digest(first), time.time()) is None
    assert client.post('/api/auth/logout').status_code == 403
    assert client.post('/api/auth/logout', headers={**HEADERS, 'Origin': 'https://evil.example'}).status_code == 403
    assert client.post('/api/auth/logout', headers=HEADERS).json() == {'signed_out': True}
    client.cookies.set(SESSION_COOKIE, second)
    assert client.get('/api/documents').status_code == 401
    client.cookies.clear()
    sign_in(client, provider)
    with app.state.catalog.connect() as db:
        db.execute('UPDATE auth_sessions SET expires=?', (time.time() - 1,))
    assert client.get('/api/auth/session').json()['authenticated'] is False
    assert client.get('/api/documents').status_code == 401


@pytest.mark.parametrize('claims', [
    {'iss': 'https://impostor.example'}, {'aud': 'another-client'}, {'exp': 1}, {'iat': 9999999999},
    {'nonce': 'wrong-nonce'}, {'sub': ''}, {'azp': 'another-client'},
    {'aud': [CLIENT_ID, 'another-client']}, {'at_hash': 'wrong-token-hash'},
    {'exp': '9999999999'}, {'exp': float('inf')}, {'iat': float('nan')},
])
def test_invalid_signed_identity_is_rejected(oidc, claims):
    _, client, provider, _, _ = oidc
    provider.overrides = claims
    state = begin(client, provider)
    response = client.get('/api/auth/callback', params={'state': state, 'code': 'alice'})
    assert response.headers['location'] == BASE + '/?auth_error=signin_failed'
    assert client.get('/api/auth/session').json()['authenticated'] is False


def test_algorithm_confusion_and_wrong_signature_are_rejected(oidc, signing_key):
    _, client, provider, _, _ = oidc
    provider.algorithm = 'HS256'
    state = begin(client, provider)
    assert 'auth_error' in client.get('/api/auth/callback', params={'state': state, 'code': 'alice'}).headers['location']
    provider.algorithm = 'RS256'
    state = begin(client, provider)
    # Cache the trusted public key, then issue with a different private key.
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key()))
    provider.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    client.app.state.auth.keys = [{**public, 'kid': 'key-one'}]
    client.app.state.auth.keys_until = time.monotonic() + 300
    assert 'auth_error' in client.get('/api/auth/callback', params={'state': state, 'code': 'alice'}).headers['location']
    assert client.get('/api/auth/session').json()['authenticated'] is False


def test_login_state_is_bound_to_browser_expiring_and_single_use(oidc):
    app, client, provider, _, _ = oidc
    state = begin(client, provider)
    with TestClient(app, base_url=BASE, follow_redirects=False) as stranger:
        response = stranger.get('/api/auth/callback', params={'state': state, 'code': 'alice'})
        assert 'auth_error' in response.headers['location']
    assert client.get('/api/auth/callback', params={'state': state, 'code': 'alice'}).headers['location'] == BASE + '/'
    before = len(provider.requests)
    assert 'auth_error' in client.get('/api/auth/callback', params={'state': state, 'code': 'bob'}).headers['location']
    assert len(provider.requests) == before
    assert client.post('/api/auth/logout', headers=HEADERS).status_code == 200
    state = begin(client, provider)
    with app.state.catalog.connect() as db:
        db.execute('UPDATE login_attempts SET expires=?', (time.time() - 1,))
    assert 'auth_error' in client.get('/api/auth/callback', params={'state': state, 'code': 'alice'}).headers['location']
    assert client.get('/api/auth/session').json()['authenticated'] is False


def test_provider_errors_never_expose_tokens_or_secret_or_assign_legacy_data(oidc):
    app, client, provider, _, root = oidc
    provider.error = 'provider-secret very-sensitive-code'
    state = begin(client, provider)
    response = client.get('/api/auth/callback', params={'state': state, 'code': 'very-sensitive-code'})
    assert 'provider-secret' not in str(response.headers) + response.text
    assert 'very-sensitive-code' not in str(response.headers) + response.text
    provider.error = None
    (root / 'docs').mkdir()
    (root / 'docs' / ('d' * 64 + '.pdf')).write_bytes(b'%PDF-1.4 legacy')
    sign_in(client, provider)
    assert client.get('/api/bootstrap').json()['mode'] == 'oidc'
    assert client.get('/api/documents').json() == []
    with app.state.catalog.connect() as db:
        assert db.execute("SELECT value FROM settings WHERE key='legacy_owner'").fetchone() is None


def test_signing_key_rotation_refreshes_only_trusted_jwks_endpoint(oidc):
    app, client, provider, _, _ = oidc
    sign_in(client, provider)
    provider.kid = 'key-two'
    sign_in(client, provider)
    assert len([url for method, url in provider.requests if url == ISSUER + '/keys']) == 2
    assert app.state.auth.keys[0]['kid'] == 'key-two'


@pytest.mark.parametrize('overrides', [{'issuer': 'https://other.example'}, {'jwks_uri': 'http://unsafe.example/keys'},
                                      {'code_challenge_methods_supported': ['plain']},
                                      {'response_types_supported': None}, {'jwks_uri': 123},
                                      {'jwks_uri': 'http://127.0.0.1/keys'}])
def test_invalid_provider_metadata_fails_closed(oidc, overrides):
    _, client, provider, _, _ = oidc
    provider.metadata_overrides = overrides
    assert client.get('/api/auth/login').status_code == 503


@pytest.mark.parametrize('method', ['client_secret_post', 'none'])
def test_supported_token_authentication_methods(oidc, monkeypatch, method):
    _, _, provider, transport, root = oidc
    monkeypatch.setenv('OIDC_TOKEN_AUTH_METHOD', method)
    if method == 'none':
        monkeypatch.delenv('OIDC_CLIENT_SECRET')
    app = create_app(root, warm=False, auth_transport=transport)
    with TestClient(app, base_url=BASE, follow_redirects=False) as client:
        assert sign_in(client, provider)['name'] == 'Alice'


def test_authentication_configuration_and_local_compatibility(tmp_path, monkeypatch):
    app = create_app(tmp_path, warm=False)
    with TestClient(app) as client:
        assert client.get('/api/auth/session').json() == {'mode': 'local', 'authenticated': True, 'user': None}
        assert client.cookies.get('rag_workspace')
        assert client.get('/api/auth/login').status_code == 404
    monkeypatch.setenv('APP_AUTH_MODE', 'typo')
    with pytest.raises(ValueError, match='APP_AUTH_MODE'):
        AuthSettings.from_env()
    monkeypatch.setenv('APP_AUTH_MODE', 'oidc')
    for key in ('OIDC_ISSUER', 'OIDC_CLIENT_ID', 'APP_BASE_URL'):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match='requires'):
        AuthSettings.from_env()
    for key, value in {'OIDC_ISSUER': ISSUER, 'OIDC_CLIENT_ID': CLIENT_ID, 'OIDC_CLIENT_SECRET': 'secret',
                       'APP_BASE_URL': 'http://public.example'}.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match='HTTPS'):
        AuthSettings.from_env()
    monkeypatch.setenv('APP_BASE_URL', 'http://127.0.0.1:8501')
    assert AuthSettings.from_env().callback_url == 'http://127.0.0.1:8501/api/auth/callback'


def test_account_identity_uses_issuer_and_subject_not_email(oidc):
    app, _, _, _, _ = oidc
    catalog = app.state.catalog
    one = catalog.create_account_session(ISSUER, 'alice', 'Alice', 'same@example.test', 'first', time.time() + 60, time.time())
    two = catalog.create_account_session('https://other-issuer.example', 'alice', 'Alice', 'same@example.test', 'second', time.time() + 60, time.time())
    assert one['id'] != two['id']


def test_es256_identity_provider_is_supported(oidc, monkeypatch):
    _, _, provider, transport, root = oidc
    monkeypatch.setenv('OIDC_SIGNING_ALGORITHM', 'ES256')
    provider.algorithm = 'ES256'
    provider.key = ec.generate_private_key(ec.SECP256R1())
    app = create_app(root, warm=False, auth_transport=transport)
    with TestClient(app, base_url=BASE, follow_redirects=False) as client:
        assert sign_in(client, provider)['name'] == 'Alice'


def test_open_progress_stream_stops_after_session_revocation(oidc, monkeypatch):
    app, client, provider, _, _ = oidc
    user = sign_in(client, provider)
    key = 'e' * 32
    app.state.catalog.save_job(key, user['id'], 'd' * 32, 'answer',
        {'status': 'running', 'message': 'Generating', 'partial_text': 'Private progress'})
    original = app.state.jobs.snapshot
    token_hash = digest(client.cookies.get(SESSION_COOKIE))
    def revoke_after_first_frame(owner, job_id):
        value = original(owner, job_id)
        app.state.catalog.revoke_session(token_hash)
        return value
    monkeypatch.setattr(app.state.jobs, 'snapshot', revoke_after_first_frame)
    response = client.get(f'/api/jobs/{key}/events')
    assert response.status_code == 200 and response.text.count('data: ') == 1
    assert 'Private progress' in response.text
    assert client.get('/api/documents').status_code == 401
