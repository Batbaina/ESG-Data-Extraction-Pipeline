from pathlib import Path
from ..loaders.text import make_records

def load_pdf(path: Path, max_chars=6000, overlap=500, limit=None):
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise RuntimeError("PDF support requires: pip install -e '.[pdf]'") from exc

    converter = DocumentConverter()
    result = converter.convert(str(path))
    markdown = result.document.export_to_markdown(page_break_placeholder="<!-- page break -->")
    pages = markdown.split("<!-- page break -->")

    records = []
    for page_no, page in enumerate(pages, start=1):
        if not page.strip():
            continue
        records.extend(make_records(
            page, f"page:{page_no}", max_chars=max_chars, overlap=overlap,
            metadata={"page": page_no}
        ))
        if limit and len(records) >= limit:
            return records[:limit]
    return records
