"""SQLite database layer for NeuroScan AI."""
import os
import sqlite3
from datetime import datetime

from flask import g

APP_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE = os.path.join(APP_DIR, "instance")
DB_PATH = os.path.join(INSTANCE, "neuroscan.db")
UPLOADS = os.path.join(INSTANCE, "uploads")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_code TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin','doctor','technician')),
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
"""


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def connect():
    os.makedirs(INSTANCE, exist_ok=True)
    os.makedirs(UPLOADS, exist_ok=True)
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


def next_code(db, table, column, prefix):
    row = db.execute(f"SELECT {column} FROM {table} WHERE {column} LIKE ? ORDER BY id DESC LIMIT 1", (prefix + "%",)).fetchone()
    n = int(row[0].rsplit("-", 1)[1]) + 1 if row else 1
    return f"{prefix}{n:04d}"
