"""Local cookie-based workspace isolation for the single-user RAG Workspace."""
from dataclasses import dataclass
from fastapi.responses import JSONResponse

PUBLIC_API_PATHS = {'/api/health', '/api/auth/session', '/api/auth/logout'}


@dataclass(frozen=True)
class AuthSettings:
    mode: str = 'local'
    base_url: str = ''

    @classmethod
    def from_env(cls):
        return cls()


class Authentication:
    def __init__(self, settings: AuthSettings, catalog, transport=None):
        self.settings = settings
        self.catalog = catalog

    def session(self, request):
        return None

    def register(self, app):
        @app.get('/api/auth/session')
        def auth_session():
            return {'mode': 'local', 'authenticated': True, 'user': None}

        @app.post('/api/auth/logout')
        def auth_logout():
            response = JSONResponse({'ok': True})
            response.delete_cookie('rag_workspace', httponly=True, samesite='strict')
            return response
