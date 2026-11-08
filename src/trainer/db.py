"""SQLite persistence. The database holds metadata and paths; large artefacts live on disk by GUID.

Key: the 32-hex CCTV content GUID (stable across re-scrapes; assigned by the publisher).
Labels: date, headline, slug — columns, not keys.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Iterator, Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS segment (
    guid            TEXT PRIMARY KEY,
    broadcast_date  TEXT NOT NULL,          -- YYYY-MM-DD
    slug            TEXT,                   -- VIDE... page slug (label, not key)
    page_url        TEXT,
    headline        TEXT NOT NULL,
    duration_s      REAL,
    cctv_tags       TEXT,                   -- publisher-supplied, space-separated
    status          TEXT NOT NULL DEFAULT 'new',  -- new|audio|aligned|enriched|translated|ready|unusable|failed
    error           TEXT,
    qc_flags        TEXT,                   -- JSON list of alignment QC warnings
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_segment_date ON segment(broadcast_date);

CREATE TABLE IF NOT EXISTS recording (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    guid            TEXT NOT NULL REFERENCES segment(guid),
    para_idx        INTEGER NOT NULL,
    path            TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass
class Segment:
    guid: str
    broadcast_date: str
    slug: Optional[str]
    page_url: Optional[str]
    headline: str
    duration_s: Optional[float]
    cctv_tags: Optional[str]
    status: str
    error: Optional[str]
    qc_flags: Optional[str]

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Segment":
        return cls(**{k: r[k] for k in cls.__dataclass_fields__})


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    config.ensure_dirs()
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with connect() as con:
        con.executescript(SCHEMA)


def upsert_segment(
    guid: str,
    broadcast_date: str,
    headline: str,
    slug: Optional[str] = None,
    page_url: Optional[str] = None,
    duration_s: Optional[float] = None,
    cctv_tags: Optional[str] = None,
) -> None:
    with connect() as con:
        con.execute(
            """
            INSERT INTO segment (guid, broadcast_date, slug, page_url, headline, duration_s, cctv_tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(guid) DO UPDATE SET
                broadcast_date = excluded.broadcast_date,
                slug = excluded.slug,
                page_url = excluded.page_url,
                headline = excluded.headline,
                duration_s = COALESCE(excluded.duration_s, segment.duration_s),
                cctv_tags = COALESCE(excluded.cctv_tags, segment.cctv_tags),
                updated_at = datetime('now')
            """,
            (guid, broadcast_date, slug, page_url, headline, duration_s, cctv_tags),
        )


def set_status(guid: str, status: str, error: Optional[str] = None, qc_flags: Optional[str] = None) -> None:
    with connect() as con:
        con.execute(
            "UPDATE segment SET status = ?, error = ?, qc_flags = COALESCE(?, qc_flags), "
            "updated_at = datetime('now') WHERE guid = ?",
            (status, error, qc_flags, guid),
        )


def get_segment(guid: str) -> Optional[Segment]:
    with connect() as con:
        r = con.execute("SELECT * FROM segment WHERE guid = ?", (guid,)).fetchone()
        return Segment.from_row(r) if r else None


def list_segments(
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    status: Optional[str] = None,
) -> list[Segment]:
    clauses, params = [], []
    if date_from:
        clauses.append("broadcast_date >= ?")
        params.append(date_from.isoformat())
    if date_to:
        clauses.append("broadcast_date <= ?")
        params.append(date_to.isoformat())
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with connect() as con:
        rows = con.execute(
            f"SELECT * FROM segment {where} ORDER BY broadcast_date DESC, created_at ASC", params
        ).fetchall()
        return [Segment.from_row(r) for r in rows]


def add_recording(guid: str, para_idx: int, path: str) -> int:
    with connect() as con:
        cur = con.execute(
            "INSERT INTO recording (guid, para_idx, path) VALUES (?, ?, ?)", (guid, para_idx, path)
        )
        return int(cur.lastrowid)
