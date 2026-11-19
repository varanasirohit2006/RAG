"""Small durable catalog. The content-addressed PDF/index stores stay unchanged."""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class Catalog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS resources (id TEXT PRIMARY KEY, kind TEXT, owner TEXT, document_id TEXT, created TEXT, payload TEXT)')
            db.execute('CREATE INDEX IF NOT EXISTS resources_owner ON resources(owner, kind, document_id)')
            db.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)')
            db.execute('''CREATE TABLE IF NOT EXISTS accounts (
                id TEXT PRIMARY KEY, issuer TEXT NOT NULL, subject TEXT NOT NULL,
                name TEXT NOT NULL, email TEXT, UNIQUE(issuer, subject))''')
            db.execute('''CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash TEXT PRIMARY KEY, account_id TEXT NOT NULL, expires REAL NOT NULL)''')
            db.execute('''CREATE TABLE IF NOT EXISTS login_attempts (
                state_hash TEXT PRIMARY KEY, binding_hash TEXT NOT NULL,
                nonce TEXT NOT NULL, verifier TEXT NOT NULL, expires REAL NOT NULL)''')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, owner TEXT NOT NULL, document_id TEXT NOT NULL, kind TEXT NOT NULL, snapshot TEXT NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner)')
            columns = {row['name'] for row in db.execute('PRAGMA table_info(jobs)')}
            for name in ('request', 'retry_job_id'):
                if name not in columns:
                    db.execute(f'ALTER TABLE jobs ADD COLUMN {name} TEXT')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, kind, owner, payload, document_id=None, resource_id=None):
        resource_id = resource_id or uuid.uuid4().hex
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO resources VALUES (?, ?, ?, ?, ?, ?)',
                       (resource_id, kind, owner, document_id, datetime.now(timezone.utc).isoformat(), json.dumps(payload, allow_nan=False)))
        return self.get(kind, owner, resource_id)

    @staticmethod
    def decode(row):
        return {'id': row['id'], 'created_at': row['created'], **json.loads(row['payload'])}

    def get(self, kind, owner, resource_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM resources WHERE id=? AND kind=? AND owner=?', (resource_id, kind, owner)).fetchone()
        return self.decode(row) if row else None

    def list(self, kind, owner, document_id=None):
        with self.connect() as db:
            if document_id is None:
                rows = db.execute('SELECT * FROM resources WHERE kind=? AND owner=? ORDER BY created DESC', (kind, owner)).fetchall()
            else:
                rows = db.execute('SELECT * FROM resources WHERE kind=? AND owner=? AND document_id=? ORDER BY created DESC', (kind, owner, document_id)).fetchall()
        return [self.decode(row) for row in rows]

    def claim_legacy(self, owner):
        with self.connect() as db:
            cursor = db.execute('INSERT OR IGNORE INTO settings VALUES (?, ?)', ('legacy_owner', owner))
            return cursor.rowcount == 1

    def create_login_attempt(self, state_hash, binding_hash, nonce, verifier, expires, now):
        with self.connect() as db:
            db.execute('DELETE FROM login_attempts WHERE expires<=?', (now,))
            db.execute('INSERT INTO login_attempts VALUES (?, ?, ?, ?, ?)',
                       (state_hash, binding_hash, nonce, verifier, expires))

    def consume_login_attempt(self, state_hash, binding_hash, now):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM login_attempts WHERE state_hash=? AND binding_hash=? AND expires>?',
                             (state_hash, binding_hash, now)).fetchone()
            if row:
                db.execute('DELETE FROM login_attempts WHERE state_hash=?', (state_hash,))
        return dict(row) if row else None

    def create_account_session(self, issuer, subject, name, email, token_hash, expires, now):
        with self.connect() as db:
            db.execute('DELETE FROM auth_sessions WHERE expires<=?', (now,))
            db.execute('''INSERT INTO accounts VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(issuer, subject) DO UPDATE SET name=excluded.name, email=excluded.email''',
                       (uuid.uuid4().hex, issuer, subject, name, email))
            row = db.execute('SELECT id, name, email FROM accounts WHERE issuer=? AND subject=?', (issuer, subject)).fetchone()
            db.execute('INSERT INTO auth_sessions VALUES (?, ?, ?)', (token_hash, row['id'], expires))
        return dict(row)

    def account_session(self, token_hash, now):
        with self.connect() as db:
            row = db.execute('''SELECT a.id, a.name, a.email FROM auth_sessions s
                JOIN accounts a ON a.id=s.account_id WHERE s.token_hash=? AND s.expires>?''',
                             (token_hash, now)).fetchone()
        return dict(row) if row else None

    def revoke_session(self, token_hash):
        with self.connect() as db:
            db.execute('DELETE FROM auth_sessions WHERE token_hash=?', (token_hash,))

    def save_job(self, key, owner, document_id, kind, snapshot, *, request=None, retry_of=None):
        with self.connect() as db:
            db.execute('''INSERT INTO jobs (id, owner, document_id, kind, snapshot, request)
                          VALUES (?, ?, ?, ?, ?, ?)
                          ON CONFLICT(id) DO UPDATE SET snapshot=excluded.snapshot''',
                       (key, owner, document_id, kind, json.dumps(snapshot, allow_nan=False),
                        json.dumps(request, allow_nan=False) if request is not None else None))
            if retry_of:
                cursor = db.execute('UPDATE jobs SET retry_job_id=? WHERE id=? AND owner=? AND retry_job_id IS NULL',
                                    (key, retry_of, owner))
                if cursor.rowcount != 1:
                    raise ValueError('This job already has a retry.')

    @staticmethod
    def decode_job(row):
        return {'id': row['id'], 'owner': row['owner'], 'document_id': row['document_id'],
                'kind': row['kind'], 'snapshot': json.loads(row['snapshot']),
                'request': json.loads(row['request']) if row['request'] else None,
                'retry_job_id': row['retry_job_id']}

    def get_job(self, owner, key):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=? AND owner=?', (key, owner)).fetchone()
        return self.decode_job(row) if row else None

    def pending_jobs(self, owner=None):
        with self.connect() as db:
            rows = db.execute('''SELECT * FROM jobs WHERE
                (json_extract(snapshot, '$.status') IN ('queued', 'running') OR
                 (json_extract(snapshot, '$.recovery_required') = 1 AND retry_job_id IS NULL))
                AND (? IS NULL OR owner=?) ORDER BY rowid DESC''', (owner, owner)).fetchall()
        return [self.decode_job(row) for row in rows]

    def recover_jobs(self):
        """An interrupted worker is never replayed automatically against paid APIs."""
        with self.connect() as db:
            for row in db.execute("SELECT id, snapshot FROM jobs WHERE json_extract(snapshot, '$.status') IN ('queued', 'running')").fetchall():
                value = json.loads(row['snapshot'])
                if value.get('status') in {'queued', 'running'}:
                    value.update(status='failed', message='Interrupted by server restart', recovery_required=True,
                                 error='The server stopped before this job finished. Retry to reuse completed work; unfinished model requests may use API credits.')
                    db.execute('UPDATE jobs SET snapshot=? WHERE id=?', (json.dumps(value, allow_nan=False), row['id']))
