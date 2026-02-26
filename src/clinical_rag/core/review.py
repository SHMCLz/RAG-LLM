from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

class ReviewBackend:
    def __init__(self, settings):
        self.db_path = Path(settings.review_db_path)
        self.standard_message = settings.standard_refer_message
        settings.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS review_queue(
              task_id INTEGER PRIMARY KEY AUTOINCREMENT,
              request_id TEXT,
              user_id TEXT,
              raw_query TEXT,
              draft TEXT,
              unsuitable TEXT,
              status TEXT,
              final_text TEXT,
              created_at TEXT,
              reviewed_at TEXT
            );""")
            conn.commit()

    def enqueue(self, request_id: str, user_id: str, raw_query: str, draft: str, unsuitable: List[str]) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO review_queue(request_id,user_id,raw_query,draft,unsuitable,status,created_at) VALUES (?,?,?,?,?,?,?)",
                (request_id, user_id, raw_query, draft, ",".join(unsuitable), "PENDING", datetime.utcnow().isoformat()+"Z"),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_pending(self, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT task_id, request_id, user_id, raw_query, unsuitable, status, created_at FROM review_queue WHERE status='PENDING' ORDER BY task_id ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_by_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM review_queue WHERE request_id=?", (request_id,)).fetchone()
        return dict(row) if row else None

    def approve(self, task_id: int, final_text: Optional[str]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE review_queue SET status=?, final_text=?, reviewed_at=? WHERE task_id=?",
                ("APPROVED", final_text, datetime.utcnow().isoformat()+"Z", task_id),
            )
            conn.commit()

    def replace(self, task_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE review_queue SET status=?, final_text=?, reviewed_at=? WHERE task_id=?",
                ("REPLACED", self.standard_message, datetime.utcnow().isoformat()+"Z", task_id),
            )
            conn.commit()
