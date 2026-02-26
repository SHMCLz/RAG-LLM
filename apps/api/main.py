import os
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

from clinical_rag.core.settings import Settings
from clinical_rag.core.retrieval import LocalHybridRetriever
from clinical_rag.core.pipeline import DeploymentPipeline
from clinical_rag.core.kb_demo import demo_segments
from clinical_rag.core.kb_loader import load_segments_jsonl

app = FastAPI(title="Clinical RAG (Deployment-consistent)", version="4.0.0")

settings = Settings()
ADMIN_TOKEN = os.getenv('ADMIN_TOKEN','')

retriever = LocalHybridRetriever(settings)
segs = load_segments_jsonl(str(getattr(settings, 'kb_segments_path', './data/segments.jsonl')))
if not segs:
    segs = demo_segments()
retriever.load_segments(segs)
pipeline = DeploymentPipeline(settings, retriever)



def _check_admin(token: str | None):
    if not ADMIN_TOKEN:
        return
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail='unauthorized')


class AskRequest(BaseModel):
    user_id: str
    query: str

class AskResponse(BaseModel):
    request_id: str
    task_id: int
    status: str

class ApproveRequest(BaseModel):
    task_id: int
    final_text: Optional[str] = None

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        out = pipeline.ask(req.user_id, req.query)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/review/pending")
def review_pending(limit: int = 20, x_admin_token: str | None = None):
    _check_admin(x_admin_token)
    return {"pending": pipeline.admin_pending(limit=limit)}

@app.post("/review/approve")
def review_approve(req: ApproveRequest, x_admin_token: str | None = None):
    _check_admin(x_admin_token)
    # If unsuitable categories exist, admins can still approve, but in the deployment
    # the policy is often to replace with standard text.
    pipeline.admin_approve(req.task_id, req.final_text)
    return {"ok": True}

@app.post("/review/replace")
def review_replace(task_id: int, x_admin_token: str | None = None):
    _check_admin(x_admin_token)
    pipeline.admin_replace(task_id)
    return {"ok": True}

@app.get("/deliver")
def deliver(request_id: str):
    return pipeline.deliver(request_id)
