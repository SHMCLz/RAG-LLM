from __future__ import annotations
import uuid
from typing import Any, Dict, Optional

from .settings import Settings
from .audit import audit_event
from .exposure import mark_pending, mark_delivered, mark_viewed
from .privacy import deidentify, abstract_query
from .gateway import external_retrieve
from .llm import DeepSeekIntranetClient, trim_to_words
from .review import ReviewBackend
from .policies import classify_unsuitable

def build_prompt(settings: Settings, user_query: str, local_evidence, external_evidence) -> str:
    lines = []
    for seg in local_evidence:
        lines.append(f"[LOCAL:{seg.get('source','')}:{seg.get('pub_year','NA')}] {seg.get('text','')}")
    if external_evidence:
        for it in external_evidence.get("items", []):
            lines.append(f"[EXT:{it.get('url','')}] {it.get('text','')}")
    evidence = "\n".join(lines[:20])
    return f"""You are an AI system for patient education.
Write clearly and concisely for middle-aged and older adults.
Keep the answer within {settings.max_words} words.
If evidence is insufficient, say so and advise discussing with the physician.

Clinical context (system-fixed):
{settings.fixed_clinical_context}

Evidence:
{evidence}

Patient question:
{user_query}
""".strip()

class DeploymentPipeline:
    def __init__(self, settings: Settings, retriever):
        self.settings = settings
        self.retriever = retriever
        self.review = ReviewBackend(settings)
        self.llm = DeepSeekIntranetClient(settings.deepseek_base_url, settings.deepseek_api_key)

    def ask(self, user_id: str, raw_query: str) -> Dict[str, Any]:
        self.settings.assert_intranet_only()
        request_id = str(uuid.uuid4())

        deid = deidentify(raw_query)
        abstracted = abstract_query(deid)

        audit_event(self.settings, request_id, "query_received", {"user_id": user_id, "len": len(raw_query)})
        audit_event(self.settings, request_id, "query_deidentified", {"abstract": abstracted})

        local_hits, top_score = self.retriever.retrieve(abstracted)
        audit_event(self.settings, request_id, "local_retrieval_done", {"hits": len(local_hits), "top_score": float(top_score), "theta": self.settings.theta})

        external = None
        if top_score <= self.settings.theta:
            external = external_retrieve(self.settings, abstracted)
            audit_event(self.settings, request_id, "external_retrieval_triggered", {"used": external is not None})
        else:
            audit_event(self.settings, request_id, "external_retrieval_skipped", {})

        prompt = build_prompt(self.settings, deid, local_hits, external)
        draft = self.llm.generate(prompt, max_tokens=self.settings.max_tokens, temperature=self.settings.temperature)
        draft = trim_to_words(draft, self.settings.max_words)
        audit_event(self.settings, request_id, "draft_generated", {"chars": len(draft)})

        unsuitable = classify_unsuitable(raw_query)
        task_id = self.review.enqueue(request_id, user_id, raw_query, draft, unsuitable)
        audit_event(self.settings, request_id, "queued_for_human_review", {"task_id": task_id, "unsuitable": unsuitable})

        mark_pending(self.settings, request_id, user_id)
        return {"request_id": request_id, "task_id": task_id, "status": "queued_for_review"}

    def admin_pending(self, limit: int = 20):
        return self.review.list_pending(limit=limit)

    def admin_approve(self, task_id: int, final_text: Optional[str] = None):
        self.review.approve(task_id, final_text)

    def admin_replace(self, task_id: int):
        self.review.replace(task_id)

    def deliver(self, request_id: str) -> Dict[str, Any]:
        row = self.review.get_by_request(request_id)
        if not row:
            return {"status": "not_found"}
        if row["status"] == "PENDING":
            return {"status": "pending_review"}
        text = row["final_text"] or row["draft"] or ""
        mark_delivered(self.settings, request_id)
        mark_viewed(self.settings, request_id)
        audit_event(self.settings, request_id, "delivered_to_patient", {"chars": len(text)})
        return {"status": "delivered", "text": text, "review_status": row["status"]}
