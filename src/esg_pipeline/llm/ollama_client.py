import json
import httpx
from pydantic import ValidationError
from ..schemas.esg import ESGExtractionResult


class OllamaClient:
    def __init__(self, base_url: str, model: str, generation_options: dict, timeout: float = 240.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.generation_options = dict(generation_options)
        self.timeout = timeout

    def check(self):
        r = httpx.get(f"{self.base_url}/api/tags", timeout=10)
        r.raise_for_status()
        return r.json()

    def extract(self, system_prompt: str, source_text: str, max_retries: int = 2):
        schema = ESGExtractionResult.model_json_schema()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "SOURCE_TEXT:\n" + source_text},
        ]
        errors = []
        last_raw = None

        for attempt in range(max_retries + 1):
            payload = {
                "model": self.model,
                "stream": False,
                "format": schema,
                "messages": messages,
                "options": self.generation_options,
            }
            r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            response_json = r.json()
            content = response_json["message"]["content"]
            last_raw = content
            try:
                parsed = ESGExtractionResult.model_validate_json(content)
                runtime = {
                    "attempt": attempt,
                    "ollama_done_reason": response_json.get("done_reason"),
                    "total_duration": response_json.get("total_duration"),
                    "load_duration": response_json.get("load_duration"),
                    "prompt_eval_count": response_json.get("prompt_eval_count"),
                    "eval_count": response_json.get("eval_count"),
                }
                return parsed, attempt, errors, runtime
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                errors.append(str(exc))
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response failed schema validation. "
                        "Return ONLY a corrected JSON object matching the supplied schema. "
                        "Do not add new facts. Validation error: " + str(exc)
                    ),
                })

        raise RuntimeError(
            "Structured output validation failed after retries. "
            + " | ".join(errors)
            + (f" | last_raw={last_raw[:500]}" if last_raw else "")
        )
