"""Database layer for NeuroScan AI.

Uses PostgreSQL when the DATABASE_URL environment variable is set (hosting, e.g. Render),
otherwise a local SQLite file (app/instance/neuroscan.db). All SQL in the app is written with
'?' placeholders and portable syntax; this module adapts it for PostgreSQL.
Uploaded / processed scan images are stored in the database (images table), so the app works
on hosts without a persistent disk.
"""
import os
import sqlite3
from datetime import datetime

from flask import g

APP_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE = os.path.join(APP_DIR, "instance")
DB_PATH = os.path.join(INSTANCE, "neuroscan.db")
UPLOADS = os.path.join(INSTANCE, "uploads")  # temporary working folder for image processing
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
IS_PG = DATABASE_URL.startswith(("postgres://", "postgresql://"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_code TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin','doctor','patient')),
    patient_id INTEGER,
    department TEXT,
    license_no TEXT,
    phone TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_login TEXT
);
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_code TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    age INTEGER,
    sex TEXT,
    phone TEXT,
    blood_group TEXT,
    referred_by TEXT,
    history TEXT,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_code TEXT UNIQUE NOT NULL,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    modality TEXT NOT NULL,
    body_part TEXT,
    original_filename TEXT,
    prediction TEXT NOT NULL,
    label TEXT NOT NULL,
    abnormal INTEGER NOT NULL,
    confidence REAL NOT NULL,
    probabilities TEXT NOT NULL,
    details TEXT NOT NULL,
    inference_ms REAL,
    remarks TEXT,
    reviewed_by INTEGER REFERENCES users(id),
    reviewed_at TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_code TEXT NOT NULL REFERENCES analyses(analysis_code) ON DELETE CASCADE,
    name TEXT NOT NULL,
    data BLOB NOT NULL,
    UNIQUE (analysis_code, name)
);
CREATE INDEX IF NOT EXISTS idx_analyses_patient ON analyses(patient_id);
CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(user_id);
"""


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today():
    return datetime.now().strftime("%Y-%m-%d")


# ---------------------------------------------------------------- PostgreSQL adapter
class _Row(tuple):
    """Row supporting both row[0] and row['column'] like sqlite3.Row."""

    def __new__(cls, names, values):
        r = super().__new__(cls, values)
        r._map = dict(zip(names, values))
        return r

    def __getitem__(self, k):
        return self._map[k] if isinstance(k, str) else tuple.__getitem__(self, k)

    def keys(self):
        return self._map.keys()


class _Cursor:
    def __init__(self, cur, lastrowid=None):
        self._c, self.lastrowid = cur, lastrowid

    def _conv(self, row):
        if row is None:
            return None
        names = [d.name for d in self._c.description]
        return _Row(names, [bytes(v) if isinstance(v, memoryview) else v for v in row])

    def fetchone(self):
        return self._conv(self._c.fetchone())

    def fetchall(self):
        return [self._conv(r) for r in self._c.fetchall()]

    def __iter__(self):
        return iter(self.fetchall())


class _PgConnection:
    def __init__(self, url):
        import psycopg
        self._con = psycopg.connect(url)

    def execute(self, sql, params=()):
        sql = sql.replace("?", "%s")
        cur = self._con.cursor()
        if sql.lstrip().upper().startswith("INSERT INTO") and "RETURNING" not in sql.upper() and "INTO IMAGES" not in sql.upper():
            cur.execute(sql + " RETURNING id", tuple(params))
            return _Cursor(cur, cur.fetchone()[0])
        cur.execute(sql, tuple(params))
        return _Cursor(cur)

    def executescript(self, script):
        script = (script.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
                        .replace("BLOB", "BYTEA"))
        with self._con.cursor() as cur:
            cur.execute(script)

    def commit(self):
        self._con.commit()

    def close(self):
        self._con.close()


# ---------------------------------------------------------------- public API
def connect():
    os.makedirs(UPLOADS, exist_ok=True)
    if IS_PG:
        return _PgConnection(DATABASE_URL)
    os.makedirs(INSTANCE, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def get_db():
    if "db" not in g:
        g.db = connect()
    return g.db


def close_db(_e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    con = connect()
    con.executescript(SCHEMA)
    con.commit()
    con.close()


def reset_db():
    """Drop all tables (used by the demo seed script)."""
    con = connect()
    for t in ("images", "analyses", "patients", "users"):
        con.execute(f"DROP TABLE IF EXISTS {t}" + (" CASCADE" if IS_PG else ""))
    con.commit()
    con.close()


def next_code(db, table, column, prefix):
    row = db.execute(f"SELECT {column} FROM {table} WHERE {column} LIKE ? ORDER BY id DESC LIMIT 1", (prefix + "%",)).fetchone()
    n = int(row[0].rsplit("-", 1)[1]) + 1 if row else 1
    return f"{prefix}{n:04d}"


def save_images(db, code, folder, names):
    for n in names:
        with open(os.path.join(folder, n), "rb") as fh:
            db.execute("INSERT INTO images (analysis_code, name, data) VALUES (?,?,?)", (code, n, fh.read()))


def load_image(db, code, name):
    row = db.execute("SELECT data FROM images WHERE analysis_code=? AND name=?", (code, name)).fetchone()
    return bytes(row[0]) if row else None
