from __future__ import annotations
import requests
from typing import Any, Dict, Optional, List
from .transient import transient_store

def external_retrieve(settings, abstracted_query: str) -> Optional[Dict[str, Any]]:
    if not settings.enable_external_gateway:
        return None
    # Outbound-only gateway assumed; we only transmit abstracted query.
    r = requests.get(settings.external_gateway_url, params={"q": abstracted_query, "k": settings.external_k}, timeout=10)
    r.raise_for_status()
    results = r.json().get("results", [])
    items: List[Dict[str, str]] = []
    for it in results[: settings.external_k]:
        items.append({
            "title": str(it.get("title",""))[:200],
            "url": str(it.get("url",""))[:500],
            "text": str(it.get("content",""))[:4000],
        })
    eid = transient_store.put(items, ttl_seconds=settings.external_ttl_seconds)
    return {"evidence_id": eid, "items": items}
