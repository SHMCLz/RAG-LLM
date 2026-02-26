from __future__ import annotations
import json
from datetime import datetime
from typing import Any, Dict
from pathlib import Path

def audit_event(settings, request_id: str, event: str, payload: Dict[str, Any]) -> None:
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    settings.audit_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "request_id": request_id,
        "event": event,
        "payload": payload,
    }
    path = Path(settings.audit_dir) / "audit.log"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
