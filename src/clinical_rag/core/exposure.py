from __future__ import annotations
import sqlite3
from datetime import datetime
from pathlib import Path

def _ensure(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS exposure(
      request_id TEXT PRIMARY KEY,
      user_id TEXT,
      delivered INT DEFAULT 0,
      viewed INT DEFAULT 0,
      delivered_at TEXT,
      viewed_at TEXT
    );""")
    conn.commit()

def mark_pending(settings, request_id: str, user_id: str) -> None:
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(Path(settings.exposure_db_path)) as conn:
        _ensure(conn)
        conn.execute("INSERT OR REPLACE INTO exposure(request_id,user_id,delivered,viewed) VALUES (?,?,0,0)",
                     (request_id, user_id))
        conn.commit()

def mark_delivered(settings, request_id: str) -> None:
    with sqlite3.connect(Path(settings.exposure_db_path)) as conn:
        _ensure(conn)
        conn.execute("UPDATE exposure SET delivered=1, delivered_at=? WHERE request_id=?",
                     (datetime.utcnow().isoformat(timespec="seconds") + "Z", request_id))
        conn.commit()

def mark_viewed(settings, request_id: str) -> None:
    with sqlite3.connect(Path(settings.exposure_db_path)) as conn:
        _ensure(conn)
        conn.execute("UPDATE exposure SET viewed=1, viewed_at=? WHERE request_id=?",
                     (datetime.utcnow().isoformat(timespec="seconds") + "Z", request_id))
        conn.commit()
