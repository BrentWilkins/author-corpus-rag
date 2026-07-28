"""Optional OpenAI-compatible local text generation."""

from __future__ import annotations

from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class LocalModelSettings(BaseModel):
    """Validated settings for an OpenAI-compatible local model server."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=800, ge=1)
    timeout_seconds: float = Field(default=180.0, gt=0.0)
    reasoning_effort: ReasoningEffort = "none"


class OpenAICompatibleCompleter:
    """Generate text through a local OpenAI-compatible chat endpoint."""

    def __init__(self, settings: LocalModelSettings) -> None:
        """Initialize a reusable API client from validated settings."""
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout_seconds,
        )

    def __call__(self, prompt: str) -> str:
        """Return visible assistant content for one prompt."""
        response = self.client.chat.completions.create(
            model=self.settings.model_id,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort=self.settings.reasoning_effort,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content
