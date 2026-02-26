from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path

@dataclass
class Settings:
    # Threshold gate
    theta: float = float(os.getenv("THETA", "0.75"))
    # Retrieval parameters
    k_dense: int = int(os.getenv("K_DENSE", "8"))
    k_sparse: int = int(os.getenv("K_SPARSE", "20"))
    k_fused: int = int(os.getenv("K_FUSED", "8"))
    w_dense: float = float(os.getenv("W_DENSE", "0.65"))
    w_sparse: float = float(os.getenv("W_SPARSE", "0.35"))
    guideline_boost: float = float(os.getenv("GUIDELINE_BOOST", "1.25"))

    # Embeddings (deployment)
    dense_embed_model: str = os.getenv("DENSE_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")

    # External gateway
    enable_external_gateway: bool = os.getenv("ENABLE_EXTERNAL_GATEWAY", "false").lower() == "true"
    external_gateway_url: str = os.getenv("EXTERNAL_GATEWAY_URL", "http://localhost:9100/search")
    external_k: int = int(os.getenv("EXTERNAL_K", "3"))
    external_ttl_seconds: int = int(os.getenv("EXTERNAL_TTL_SECONDS", "600"))

    # Local LLM (DeepSeek-R1 intranet service)
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "http://localhost:9000").rstrip("/")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")

    # Prompt constraints
    fixed_clinical_context: str = os.getenv("FIXED_CLINICAL_CONTEXT", "patients scheduled for radical prostatectomy for prostate cancer")
    max_words: int = int(os.getenv("MAX_WORDS", "500"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "900"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))

    # Knowledge base
    kb_segments_path: str = os.getenv('KB_SEGMENTS_PATH', './data/segments.jsonl')
    index_dir: Path = Path(os.getenv('INDEX_DIR', './runtime/index'))

    # Runtime
    runtime_dir: Path = Path(os.getenv("RUNTIME_DIR", "./runtime"))
    @property
    def audit_dir(self) -> Path:
        return self.runtime_dir / "audit"

    @property
    def review_db_path(self) -> Path:
        return self.runtime_dir / "review.sqlite"

    @property
    def exposure_db_path(self) -> Path:
        return self.runtime_dir / "exposure.sqlite"

    @property
    def standard_refer_message(self) -> str:
        return "Please discuss this concern with your physician during the preoperative communication."

    def assert_intranet_only(self) -> None:
        # Prevent accidental use of public API bases in a deployment-consistent build.
        bad = ("api.openai.com", "api.deepseek.com", "openai.com", "deepseek.com")
        if any(b in self.deepseek_base_url for b in bad):
            raise RuntimeError(f"DEEPSEEK_BASE_URL must be an intranet endpoint, got: {self.deepseek_base_url}")
