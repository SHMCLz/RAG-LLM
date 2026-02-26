import os
from clinical_rag.core.settings import Settings
from clinical_rag.core.retrieval import LocalHybridRetriever
from clinical_rag.core.pipeline import DeploymentPipeline
from clinical_rag.core.kb_demo import demo_segments
from clinical_rag.core.transient import transient_store

def _pipeline(tmp_path):
    os.environ["RUNTIME_DIR"] = str(tmp_path/"runtime")
    s = Settings()
    r = LocalHybridRetriever(s)
    r.load_segments(demo_segments())
    return s, DeploymentPipeline(s, r)

def test_mandatory_review_queue(tmp_path):
    s, p = _pipeline(tmp_path)
    out = p.ask("u1", "When can I shower?")
    assert out["status"] == "queued_for_review"
    delivered = p.deliver(out["request_id"])
    assert delivered["status"] in ("pending_review","not_found")

def test_transient_store_no_disk(tmp_path):
    eid = transient_store.put([{"url":"x","text":"y"}], ttl_seconds=10)
    assert transient_store.get(eid)[0]["text"] == "y"

def test_theta_gate_skips_external_by_default(tmp_path):
    os.environ["ENABLE_EXTERNAL_GATEWAY"] = "false"
    s, p = _pipeline(tmp_path)
    out = p.ask("u1", "When can I shower?")
    assert out["status"] == "queued_for_review"
