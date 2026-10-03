from pathlib import Path

def load_prompt(version: str) -> str:
    root = Path(__file__).resolve().parents[2]
    path = root / "prompts" / f"extraction_{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()
