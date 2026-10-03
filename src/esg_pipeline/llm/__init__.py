def create_llm_backend(settings):
    backend = settings.llm_backend.lower().strip()

    if backend == "ollama":
        from .ollama_backend import OllamaBackend

        return OllamaBackend(
            settings.ollama_url,
            settings.model,
            settings.generation_options(),
            settings.request_timeout_seconds,
        )

    if backend == "transformers":
        from .transformers_backend import TransformersBackend

        return TransformersBackend(
            model=settings.model,
            generation_options=settings.generation_options(),
            enable_thinking=settings.enable_thinking,
        )

    raise ValueError(
        f"Unsupported ESG_LLM_BACKEND={settings.llm_backend!r}. "
        "Supported backends: ollama, transformers."
    )
