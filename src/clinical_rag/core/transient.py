from __future__ import annotations
import time
from typing import Any, Dict, Optional

class TransientStore:
    def __init__(self):
        self._d: Dict[str, tuple[float, Any]] = {}

    def put(self, obj: Any, ttl_seconds: int) -> str:
        eid = f"ext_{int(time.time()*1000)}"
        self._d[eid] = (time.time() + ttl_seconds, obj)
        return eid

    def get(self, eid: str) -> Optional[Any]:
        if eid not in self._d:
            return None
        exp, obj = self._d[eid]
        if time.time() > exp:
            self._d.pop(eid, None)
            return None
        return obj

transient_store = TransientStore()
