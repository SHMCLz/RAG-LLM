from __future__ import annotations
import requests
from urllib.parse import urlparse
from typing import Any, Dict, Optional, List
from .transient import transient_store


def _host_allowed(url: str, allowlist: List[str]) -> bool:
    """Return True only if the result URL's host is on the curated allowlist."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in allowlist)


def external_retrieve(settings, abstracted_query: str) -> Optional[Dict[str, Any]]:
    """Threshold-gated, outbound-only external retrieval.

    Invoked only when local retrieval is insufficient (top score <= theta).
    Only the de-identified, abstracted query is transmitted through a
    controlled outbound-only gateway. Returned results are filtered against a
    curated allowlist of vetted medical sources (e.g. clinical guideline
    repositories, PubMed); any result from a non-allowlisted host is discarded.
    External content is used transiently and is never written into the local
    knowledge base.
    """
    if not settings.enable_external_gateway:
        return None

    # Outbound-only gateway: only the abstracted query leaves the intranet.
    r = requests.get(
        settings.external_gateway_url,
        params={"q": abstracted_query, "k": settings.external_k},
        timeout=10,
    )
    r.raise_for_status()
    results = r.json().get("results", [])

    allowlist = settings.external_source_allowlist
    items: List[Dict[str, str]] = []
    for it in results:
        url = str(it.get("url", ""))
        # Source filtering: keep only vetted allowlisted sources.
        if not _host_allowed(url, allowlist):
            continue
        items.append({
            "title": str(it.get("title", ""))[:200],
            "url": url[:500],
            "text": str(it.get("content", ""))[:4000],
        })
        if len(items) >= settings.external_k:
            break

    if not items:
        return None

    eid = transient_store.put(items, ttl_seconds=settings.external_ttl_seconds)
    return {"evidence_id": eid, "items": items}
