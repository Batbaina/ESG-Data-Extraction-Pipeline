from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Common interface implemented by all LLM inference backends."""

    @abstractmethod
    def check(self):
        """Check that the backend/model is available."""
        raise NotImplementedError

    @abstractmethod
    def extract(
        self,
        system_prompt: str,
        source_text: str,
        max_retries: int = 2,
    ):
        """Extract structured ESG information."""
        raise NotImplementedError
