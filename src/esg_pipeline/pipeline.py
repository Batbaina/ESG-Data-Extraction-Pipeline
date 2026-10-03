import hashlib
from pathlib import Path
from datetime import datetime, timezone
from .config import Settings
from .prompts import load_prompt
from .loaders.tabular import load_tabular
from .loaders.text import load_text
from .extractors.pdf_docling import load_pdf
from .llm import create_llm_backend
from .quality.checks import check_unit
from .exporters import write_jsonl, write_review_xlsx, write_json


def _source_id(path: Path, locator: str, text: str):
    digest = hashlib.sha256((str(path.name) + locator + text).encode()).hexdigest()[:12]
    return f"{path.stem}-{digest}"


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_input(path: Path, settings: Settings, text_column="text", limit=None):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path, settings.chunk_max_chars, settings.chunk_overlap_chars, limit)
    if suffix in {".xlsx", ".csv"}:
        return load_tabular(path, text_column=text_column, limit=limit)
    if suffix in {".txt", ".md"}:
        return load_text(path, settings.chunk_max_chars, settings.chunk_overlap_chars, limit)
    raise ValueError("Supported input: PDF, XLSX, CSV, TXT, MD")


def run(path: Path, output_dir: Path, text_column="text", limit=None):
    settings = Settings()
    prompt = load_prompt(settings.prompt_version)
    generation = settings.generation_options()
    client = OllamaClient(
        settings.ollama_url, settings.model, generation,
        timeout=settings.request_timeout_seconds,
    )
    records = load_input(path, settings, text_column, limit)
    rows, ready = [], []

    generation_columns = {
        "model": settings.model,
        "prompt_version": settings.prompt_version,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "seed": settings.seed,
        "num_ctx": settings.num_ctx,
        "num_predict": settings.num_predict,
    }

    for rec in records:
        sid = _source_id(path, rec["source_locator"], rec["text"])
        base = {
            "source_id": sid,
            "source_file": path.name,
            "source_locator": rec["source_locator"],
            "source_text": rec["text"],
            **generation_columns,
        }
        try:
            result, retry_count, retry_errors, runtime = client.extract(
                prompt, rec["text"], settings.max_retries
            )
            if not result.usable:
                rows.append({
                    **base, "status": "valid_rejected", "usable": False,
                    "rejection_reason": result.rejection_reason or "no_usable_esg_knowledge",
                    "qc_pass": False, "qc_lexical_grounding": "",
                    "qc_reasons": "not_usable", "llm_retry_count": retry_count,
                })
                continue

            for unit in result.knowledge_units:
                qc = check_unit(
                    unit, rec["text"], settings.qc_min_confidence,
                    settings.qc_min_lexical_grounding, settings.require_source_supported,
                )
                row = {
                    **base, "status": "valid", "usable": True, "rejection_reason": "",
                    "pillar": unit.pillar, "topic": unit.topic,
                    "knowledge_type": unit.knowledge_type,
                    "source_facts": " | ".join(unit.source_facts),
                    "extracted_knowledge": unit.extracted_knowledge,
                    "confidence": unit.confidence,
                    "qc_pass": qc.passed,
                    "qc_lexical_grounding": qc.lexical_grounding_score,
                    "qc_reasons": " | ".join(qc.reasons),
                    "llm_retry_count": retry_count,
                }
                rows.append(row)
                if qc.passed:
                    ready.append(row)
        except Exception as exc:
            rows.append({
                **base, "status": "error", "usable": "", "rejection_reason": str(exc),
                "qc_pass": False, "qc_lexical_grounding": "",
                "qc_reasons": "pipeline_error", "llm_retry_count": settings.max_retries,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{path.stem}_{settings.prompt_version}_{stamp}"
    candidates = output_dir / f"{stem}_candidates.jsonl"
    ready_path = output_dir / f"{stem}_ready.jsonl"
    review = output_dir / f"{stem}_human_review.xlsx"
    manifest = output_dir / f"{stem}_run_manifest.json"

    write_jsonl(candidates, rows)
    write_jsonl(ready_path, ready)
    write_review_xlsx(review, rows)
    write_json(manifest, {
        "pipeline_version": "3.0.0",
        "created_at_utc": stamp,
        "input_file": str(path),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "record_count": len(records),
        "candidate_count": len(rows),
        "auto_qc_ready_count": len(ready),
        "prompt_sha256": _prompt_sha256(prompt),
        "settings": settings.public_dict(),
        "generation_options": generation,
        "outputs": {
            "candidates": str(candidates), "ready": str(ready_path), "human_review": str(review)
        },
    })
    return {
        "records": len(records), "candidate_rows": len(rows), "ready_rows": len(ready),
        "candidates": candidates, "ready": ready_path, "review": review, "manifest": manifest,
    }
