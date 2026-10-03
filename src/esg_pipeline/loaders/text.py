import re
from pathlib import Path

def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, max_chars: int = 6000, overlap: int = 500):
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text] if text else []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for p in paras:
        candidate = (current + "\n\n" + p).strip() if current else p
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + "\n\n" + p).strip()
        else:
            start = 0
            step = max(1, max_chars - overlap)
            while start < len(p):
                chunks.append(p[start:start+max_chars])
                start += step
            current = ""
    if current:
        chunks.append(current)
    return chunks

def make_records(text: str, source_locator: str, max_chars=6000, overlap=500, metadata=None):
    return [
        {"text": c, "source_locator": f"{source_locator}/chunk:{i+1}", "metadata": metadata or {}}
        for i, c in enumerate(chunk_text(text, max_chars, overlap))
    ]

def load_text(path: Path, max_chars=6000, overlap=500, limit=None):
    text = path.read_text(encoding="utf-8")
    records = make_records(text, "text", max_chars, overlap)
    return records[:limit] if limit else records
