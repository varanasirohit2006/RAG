"""OIDC authorization-code login; the browser receives only an opaque session cookie."""
import asyncio
import base64
import hashlib
import math
import os
import secrets
import time
from dataclasses import dataclass
from urllib.parse import urlencode, urlsplit

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

SESSION_COOKIE = 'rag_session'
LOGIN_COOKIE = 'rag_login'
PUBLIC_API_PATHS = {'/api/health', '/api/auth/session', '/api/auth/login',
                    '/api/auth/callback', '/api/auth/logout'}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def safe_url(value, *, origin=False):
    if not isinstance(value, str):
        raise ValueError('Authentication URLs must be strings.')
    parsed = urlsplit(value)
    loopback = parsed.hostname in {'localhost', '127.0.0.1', '::1'}
    if (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment
            or (parsed.scheme != 'https' and not (parsed.scheme == 'http' and loopback))
            or (origin and parsed.path not in ('', '/'))):
        raise ValueError('Authentication URLs must use HTTPS (HTTP is allowed only on loopback). APP_BASE_URL must be an origin.')
    return value.rstrip('/') if origin else value


@dataclass(frozen=True)
class AuthSettings:
    mode: str = 'local'
    issuer: str = ''
    client_id: str = ''
    client_secret: str = ''
    base_url: str = ''
    algorithm: str = 'RS256'
    token_auth_method: str = 'client_secret_basic'
    session_seconds: int = 28800

    @classmethod
    def from_env(cls):
        mode = os.getenv('APP_AUTH_MODE', 'local').strip().lower()
        if mode not in {'local', 'oidc'}:
            raise ValueError('APP_AUTH_MODE must be local or oidc.')
        if mode == 'local':
            return cls()
        issuer = os.getenv('OIDC_ISSUER', '').strip()
        client_id = os.getenv('OIDC_CLIENT_ID', '').strip()
        secret = os.getenv('OIDC_CLIENT_SECRET', '')
        base_url = os.getenv('APP_BASE_URL', '').strip()
        method = os.getenv('OIDC_TOKEN_AUTH_METHOD', 'client_secret_basic')
        algorithm = os.getenv('OIDC_SIGNING_ALGORITHM', 'RS256')
        if not issuer or not client_id or not base_url:
            raise ValueError('OIDC mode requires OIDC_ISSUER, OIDC_CLIENT_ID and APP_BASE_URL.')
        if method not in {'client_secret_basic', 'client_secret_post', 'none'} or (method != 'none' and not secret):
            raise ValueError('Set OIDC_CLIENT_SECRET or configure OIDC_TOKEN_AUTH_METHOD=none for a public PKCE client.')
        if algorithm not in {'RS256', 'ES256'}:
            raise ValueError('OIDC_SIGNING_ALGORITHM must be RS256 or ES256.')
        seconds = int(os.getenv('AUTH_SESSION_SECONDS', '28800'))
        if not 300 <= seconds <= 86400:
            raise ValueError('AUTH_SESSION_SECONDS must be between 300 and 86400.')
        return cls(mode, safe_url(issuer), client_id, secret, safe_url(base_url, origin=True), algorithm, method, seconds)

    @property
    def callback_url(self):
        return self.base_url + '/api/auth/callback'

    @property
    def secure(self):
        return self.base_url.startswith('https://')


class LoginError(Exception):
    """Deliberately excludes provider responses, credentials and authorization codes."""


class Authentication:
    def __init__(self, settings, catalog, *, transport=None):
        self.settings, self.catalog, self.transport = settings, catalog, transport
        self.metadata = None
        self.metadata_until = 0
        self.keys = None
        self.keys_until = 0
        self.lock = asyncio.Lock()

    def session(self, request):
        token = request.cookies.get(SESSION_COOKIE, '')
        if not token or len(token) > 128:
            return None
        return self.catalog.account_session(digest(token), time.time())

    def client(self):
        return httpx.AsyncClient(timeout=10, follow_redirects=False, transport=self.transport)

    async def fetch_json(self, client, method, url, **kwargs):
        try:
            async with client.stream(method, url, **kwargs) as response:
                response.raise_for_status()
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > 1024 * 1024:
                        raise LoginError()
                    chunks.append(chunk)
                import json
                value = json.loads(b''.join(chunks))
                if not isinstance(value, dict):
                    raise LoginError()
                return value
        except (httpx.HTTPError, ValueError) as exc:
            raise LoginError() from exc

    async def discovery(self, client):
        async with self.lock:
            if self.metadata and self.metadata_until > time.monotonic():
                return self.metadata
            value = await self.fetch_json(client, 'GET', self.settings.issuer.rstrip('/') + '/.well-known/openid-configuration')
            try:
                for key in ('response_types_supported', 'id_token_signing_alg_values_supported'):
                    if not isinstance(value.get(key), list):
                        raise LoginError()
                if (value.get('issuer') != self.settings.issuer
                        or 'code' not in value['response_types_supported']
                        or self.settings.algorithm not in value['id_token_signing_alg_values_supported']):
                    raise LoginError()
                for key in ('authorization_endpoint', 'token_endpoint', 'jwks_uri'):
                    safe_url(value[key])
                    if self.settings.secure and not value[key].startswith('https://'):
                        raise LoginError()
                supported_pkce = value.get('code_challenge_methods_supported')
                if supported_pkce is not None and (not isinstance(supported_pkce, list) or 'S256' not in supported_pkce):
                    raise LoginError()
                if self.settings.token_auth_method not in value.get('token_endpoint_auth_methods_supported', ['client_secret_basic']):
                    raise LoginError()
            except (KeyError, TypeError, ValueError) as exc:
                raise LoginError() from exc
            self.metadata, self.metadata_until = value, time.monotonic() + 300
            return value

    async def signing_keys(self, client, metadata, *, refresh=False):
        async with self.lock:
            if not refresh and self.keys and self.keys_until > time.monotonic():
                return self.keys
            value = await self.fetch_json(client, 'GET', metadata['jwks_uri'])
            if not isinstance(value.get('keys'), list):
                raise LoginError()
            self.keys, self.keys_until = value['keys'], time.monotonic() + 300
            return self.keys

    async def validate_identity(self, client, metadata, tokens, nonce):
        import jwt
        try:
            token = tokens['id_token']
            if not isinstance(token, str) or len(token) > 65536:
                raise LoginError()
            header = jwt.get_unverified_header(token)
            if header.get('alg') != self.settings.algorithm:
                raise LoginError()
            keys = await self.signing_keys(client, metadata)
            def matches(keys):
                return [key for key in keys if isinstance(key, dict)
                        and (not header.get('kid') or key.get('kid') == header['kid'])
                        and key.get('use', 'sig') == 'sig'
                        and key.get('alg', self.settings.algorithm) == self.settings.algorithm
                        and isinstance(key.get('key_ops', ['verify']), list)
                        and 'verify' in key.get('key_ops', ['verify'])]
            candidates = matches(keys)
            if not candidates and header.get('kid'):
                candidates = matches(await self.signing_keys(client, metadata, refresh=True))
            if len(candidates) != 1:
                raise LoginError()
            key = jwt.PyJWK.from_dict(candidates[0], algorithm=self.settings.algorithm).key
            claims = jwt.decode(token, key, algorithms=[self.settings.algorithm],
                issuer=self.settings.issuer, audience=self.settings.client_id, leeway=30,
                options={'require': ['iss', 'sub', 'aud', 'exp', 'iat', 'nonce']})
            if (not isinstance(claims['sub'], str) or not 1 <= len(claims['sub']) <= 255
                    or not isinstance(claims['nonce'], str) or not secrets.compare_digest(claims['nonce'], nonce)
                    or (isinstance(claims['aud'], list) and len(claims['aud']) > 1 and 'azp' not in claims)
                    or ('azp' in claims and claims['azp'] != self.settings.client_id)):
                raise LoginError()
            if any(type(claims[field]) not in (int, float) or not math.isfinite(claims[field]) for field in ('iat', 'exp')):
                raise LoginError()
            if 'at_hash' in claims:
                access_token = tokens.get('access_token')
                if not isinstance(access_token, str):
                    raise LoginError()
                expected = base64.urlsafe_b64encode(hashlib.sha256(access_token.encode('ascii')).digest()[:16]).rstrip(b'=').decode()
                if claims['at_hash'] != expected:
                    raise LoginError()
            return claims
        except (KeyError, TypeError, ValueError, OverflowError, jwt.PyJWTError) as exc:
            raise LoginError() from exc

    def register(self, app):
        @app.get('/api/auth/session')
        def session(request: Request):
            user = self.session(request) if self.settings.mode == 'oidc' else None
            return {'mode': self.settings.mode, 'authenticated': self.settings.mode == 'local' or user is not None, 'user': user}

        @app.get('/api/auth/login')
        async def login():
            if self.settings.mode != 'oidc':
                raise HTTPException(404, 'Account login is not enabled in local mode.')
            try:
                async with self.client() as client:
                    metadata = await self.discovery(client)
            except LoginError:
                raise HTTPException(503, 'Sign-in provider unavailable. Try again shortly.')
            state, binding, nonce, verifier = [secrets.token_urlsafe(32) for _ in range(4)]
            self.catalog.create_login_attempt(digest(state), digest(binding), nonce, verifier, time.time() + 600, time.time())
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
            parameters = {'client_id': self.settings.client_id, 'response_type': 'code',
                          'redirect_uri': self.settings.callback_url, 'scope': 'openid profile email',
                          'state': state, 'nonce': nonce, 'code_challenge': challenge,
                          'code_challenge_method': 'S256', 'prompt': 'select_account'}
            response = RedirectResponse(metadata['authorization_endpoint'] + '?' + urlencode(parameters), status_code=303)
            response.set_cookie(LOGIN_COOKIE, binding, httponly=True, secure=self.settings.secure,
                                samesite='lax', max_age=600, path='/api/auth')
            return response

        @app.get('/api/auth/callback')
        async def callback(request: Request):
            if self.settings.mode != 'oidc':
                raise HTTPException(404, 'Account login is not enabled in local mode.')
            state, binding = request.query_params.get('state', ''), request.cookies.get(LOGIN_COOKIE, '')
            attempt = self.catalog.consume_login_attempt(digest(state), digest(binding), time.time()) if state and binding and len(state) <= 128 and len(binding) <= 128 else None
            response = RedirectResponse(self.settings.base_url + '/?auth_error=signin_failed', status_code=303)
            response.delete_cookie(LOGIN_COOKIE, path='/api/auth', secure=self.settings.secure, httponly=True, samesite='lax')
            code = request.query_params.get('code', '')
            if not attempt or not code or len(code) > 8192 or request.query_params.get('error'):
                return response
            try:
                async with self.client() as client:
                    metadata = await self.discovery(client)
                    form = {'grant_type': 'authorization_code', 'code': code,
                            'redirect_uri': self.settings.callback_url, 'code_verifier': attempt['verifier']}
                    kwargs = {}
                    if self.settings.token_auth_method == 'client_secret_basic':
                        # OAuth Basic credentials are form-encoded before base64 encoding.
                        from urllib.parse import quote_plus
                        kwargs['auth'] = httpx.BasicAuth(quote_plus(self.settings.client_id), quote_plus(self.settings.client_secret))
                    else:
                        form['client_id'] = self.settings.client_id
                        if self.settings.token_auth_method == 'client_secret_post':
                            form['client_secret'] = self.settings.client_secret
                    tokens = await self.fetch_json(client, 'POST', metadata['token_endpoint'], data=form, **kwargs)
                    claims = await self.validate_identity(client, metadata, tokens, attempt['nonce'])
                name = claims.get('name')
                name = name[:120] if isinstance(name, str) and name.strip() else 'RAG account'
                email = claims.get('email') if claims.get('email_verified') is True else None
                email = email[:254] if isinstance(email, str) else None
                token = secrets.token_urlsafe(32)
                self.catalog.create_account_session(self.settings.issuer, claims['sub'], name, email,
                    digest(token), time.time() + self.settings.session_seconds, time.time())
                self.catalog.revoke_session(digest(request.cookies.get(SESSION_COOKIE, '')))
                response.headers['location'] = self.settings.base_url + '/'
                response.set_cookie(SESSION_COOKIE, token, httponly=True, secure=self.settings.secure,
                                    samesite='lax', max_age=self.settings.session_seconds, path='/')
            except LoginError:
                pass
            return response

        @app.post('/api/auth/logout')
        def logout(request: Request):
            self.catalog.revoke_session(digest(request.cookies.get(SESSION_COOKIE, '')))
            response = JSONResponse({'signed_out': True})
            response.delete_cookie(SESSION_COOKIE, secure=self.settings.secure, httponly=True, samesite='lax', path='/')
            response.delete_cookie(LOGIN_COOKIE, path='/api/auth', secure=self.settings.secure, httponly=True, samesite='lax')
            return response
