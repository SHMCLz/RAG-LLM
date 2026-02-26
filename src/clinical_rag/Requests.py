"""Deployment-consistent request layer.

It routes queries through:
- de-identification + semantic abstraction
- hybrid retrieval (dense+sparse) on local KB
- threshold-gated external retrieval via outbound-only gateway (optional)
- local DeepSeek-R1 intranet inference
- mandatory human review queue

Public function:
  answer(user_id: str, query: str) -> dict with request_id/task_id/status
"""
from __future__ import annotations
from clinical_rag.deployment_entry import ask as _ask

def answer(user_id: str, query: str):
    return _ask(user_id, query)
