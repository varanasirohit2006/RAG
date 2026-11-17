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
