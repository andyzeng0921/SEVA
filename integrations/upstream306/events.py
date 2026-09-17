"""SQLite persistence adapter for tasks, observations, cancellation and evidence."""
from contextlib import contextmanager
import json
import sqlite3
import uuid
from zeng_agent.models import utc_now_iso


class EventStore:
    def __init__(self, path):
        self.path = path
        with self.connect() as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT UNIQUE NOT NULL, task_id TEXT NOT NULL, kind TEXT NOT NULL,
                    captured_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS task_events ON events(task_id, seq);
                CREATE TABLE IF NOT EXISTS cancellations (task_id TEXT PRIMARY KEY, requested_at TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, task):
        with self.connect() as conn:
            conn.execute('INSERT OR REPLACE INTO tasks VALUES (?,?)',
                         (task['task_id'], json.dumps(task, ensure_ascii=False, allow_nan=False)))
        return task

    def get(self, task_id):
        with self.connect() as conn:
            row = conn.execute('SELECT payload FROM tasks WHERE id=?', (task_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def append(self, task_id, kind, payload):
        evidence_id = str(uuid.uuid4())
        with self.connect() as conn:
            conn.execute('INSERT INTO events(id,task_id,kind,captured_at,payload) VALUES (?,?,?,?,?)',
                (evidence_id, task_id, kind, utc_now_iso(), json.dumps(payload, ensure_ascii=False, allow_nan=False)))
        return evidence_id

    def events(self, task_id):
        with self.connect() as conn:
            rows = conn.execute('SELECT id,kind,captured_at,payload FROM events WHERE task_id=? ORDER BY seq', (task_id,)).fetchall()
        return [{'evidence_id': row[0], 'kind': row[1], 'captured_at': row[2], 'payload': json.loads(row[3])} for row in rows]

    def cancel(self, task_id):
        with self.connect() as conn:
            conn.execute('INSERT OR IGNORE INTO cancellations VALUES (?,?)', (task_id, utc_now_iso()))

    def cancelled(self, task_id):
        with self.connect() as conn:
            return conn.execute('SELECT 1 FROM cancellations WHERE task_id=?', (task_id,)).fetchone() is not None

    def running(self):
        with self.connect() as conn:
            rows = conn.execute('SELECT payload FROM tasks').fetchall()
        return [x for row in rows if (x := json.loads(row[0])).get('status') == 'running']
