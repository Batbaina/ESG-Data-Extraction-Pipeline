import os
from dataclasses import dataclass, asdict


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Model / runtime
    llm_backend: str = os.getenv("ESG_LLM_BACKEND", "ollama")
    model: str = os.getenv("ESG_MODEL", "qwen3:8b")
    ollama_url: str = os.getenv("ESG_OLLAMA_URL", "http://localhost:11434").rstrip("/")
    request_timeout_seconds: float = float(os.getenv("ESG_REQUEST_TIMEOUT_SECONDS", "240"))

    # Prompt
    prompt_version: str = os.getenv("ESG_PROMPT_VERSION", "v1")

    # Generation / reproducibility
    enable_thinking: bool = os.getenv("ESG_ENABLE_THINKING", "false").lower() in {"1", "true", "yes", "on"}
    temperature: float = float(os.getenv("ESG_TEMPERATURE", "0.0"))
    top_p: float = float(os.getenv("ESG_TOP_P", "0.9"))
    top_k: int = int(os.getenv("ESG_TOP_K", "40"))
    seed: int = int(os.getenv("ESG_SEED", "42"))
    num_ctx: int = int(os.getenv("ESG_NUM_CTX", "8192"))
    num_predict: int = int(os.getenv("ESG_NUM_PREDICT", "2048"))

    # Validation / retry
    max_retries: int = int(os.getenv("ESG_MAX_RETRIES", "2"))

    # Input preparation
    chunk_max_chars: int = int(os.getenv("ESG_CHUNK_MAX_CHARS", "6000"))
    chunk_overlap_chars: int = int(os.getenv("ESG_CHUNK_OVERLAP_CHARS", "500"))

    # Quality control
    qc_min_confidence: float = float(os.getenv("ESG_QC_MIN_CONFIDENCE", "0.70"))
    qc_min_lexical_grounding: float = float(os.getenv("ESG_QC_MIN_LEXICAL_GROUNDING", "0.45"))
    require_source_supported: bool = _env_bool("ESG_QC_REQUIRE_SOURCE_SUPPORTED", True)

    def generation_options(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }

    def public_dict(self) -> dict:
        return asdict(self)
