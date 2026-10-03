from pathlib import Path
from openpyxl import load_workbook
from esg_pipeline.exporters import write_review_xlsx


def test_review_workbook_contains_generation_parameters(tmp_path: Path):
    out = tmp_path / "review.xlsx"
    write_review_xlsx(out, [{
        "source_id":"x", "source_file":"a.pdf", "source_locator":"page:1", "source_text":"source",
        "status":"valid", "usable":True, "pillar":"Social", "topic":"Living wage",
        "knowledge_type":"company_specific", "source_facts":"fact", "extracted_knowledge":"A sufficiently long extracted knowledge statement.",
        "confidence":0.9, "qc_pass":True, "qc_lexical_grounding":0.9, "qc_reasons":"", "llm_retry_count":0,
        "model":"qwen3:8b", "prompt_version":"v1", "temperature":0.0, "top_p":0.9, "top_k":40,
        "seed":42, "num_ctx":8192, "num_predict":2048,
    }])
    wb = load_workbook(out, data_only=False)
    headers = [c.value for c in wb.active[1]]
    for expected in ["temperature","top_p","top_k","seed","num_ctx","num_predict","human_total"]:
        assert expected in headers
