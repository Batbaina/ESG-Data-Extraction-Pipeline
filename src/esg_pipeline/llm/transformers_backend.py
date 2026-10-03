import json

from pydantic import ValidationError

from .base import LLMBackend
from ..schemas.esg import ESGExtractionResult


class TransformersBackend(LLMBackend):
    """
    Hugging Face Transformers backend intended primarily for GPU execution
    on Google Colab, Kaggle, workstations, or GPU servers.

    The model is loaded lazily so importing the ESG pipeline does not require
    torch/transformers when the Ollama backend is being used.
    """

    def __init__(
        self,
        model: str,
        generation_options: dict,
        enable_thinking: bool = False,
    ):
        self.model_name = model
        self.generation_options = dict(generation_options)
        self.enable_thinking = enable_thinking

        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers backend requires torch and transformers. "
                "Install the Colab/GPU dependencies first."
            ) from exc

        self._torch = torch

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    def check(self):
        self._load()

        return {
            "backend": "transformers",
            "model": self.model_name,
            "device": str(self._model.device),
        }

    def extract(
        self,
        system_prompt: str,
        source_text: str,
        max_retries: int = 2,
    ):
        self._load()

        schema = ESGExtractionResult.model_json_schema()

        schema_instruction = (
            "\n\nReturn ONLY one valid JSON object matching this JSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt + schema_instruction,
            },
            {
                "role": "user",
                "content": "SOURCE_TEXT:\n" + source_text,
            },
        ]

        errors = []
        last_raw = None

        for attempt in range(max_retries + 1):

            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }

            # Qwen3 supports explicit non-thinking mode.
            if "qwen3" in self.model_name.lower():
                template_kwargs["enable_thinking"] = self.enable_thinking

            prompt = self._tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
            )

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self._model.device)

            options = self.generation_options

            max_new_tokens = int(
                options.get(
                    "max_new_tokens",
                    options.get("num_predict", 768),
                )
            )

            temperature = float(options.get("temperature", 0.0))
            do_sample = temperature > 0

            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }

            if do_sample:
                generate_kwargs.update(
                    {
                        "temperature": temperature,
                        "top_p": float(options.get("top_p", 0.9)),
                        "top_k": int(options.get("top_k", 40)),
                    }
                )

            with self._torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    **generate_kwargs,
                )

            generated_ids = output_ids[
                :,
                inputs["input_ids"].shape[1]:
            ]

            content = self._tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0].strip()

            last_raw = content

            # Defensive cleanup if a model surrounds JSON with Markdown.
            if content.startswith("```"):
                content = content.strip("`").strip()

                if content.lower().startswith("json"):
                    content = content[4:].strip()

            try:
                parsed = ESGExtractionResult.model_validate_json(content)

                runtime = {
                    "attempt": attempt,
                    "backend": "transformers",
                    "model": self.model_name,
                    "input_tokens": int(inputs["input_ids"].shape[1]),
                    "generated_tokens": int(generated_ids.shape[1]),
                    "enable_thinking": self.enable_thinking,
                }

                return parsed, attempt, errors, runtime

            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                errors.append(str(exc))

                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response failed schema validation. "
                            "Return ONLY a corrected JSON object matching the "
                            "required schema. Do not add new facts. "
                            "Validation error: "
                            + str(exc)
                        ),
                    }
                )

        raise RuntimeError(
            "Structured output validation failed after retries. "
            + " | ".join(errors)
            + (
                f" | last_raw={last_raw[:500]}"
                if last_raw
                else ""
            )
        )
