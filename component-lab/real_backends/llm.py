"""Minimal OpenAI-compatible adapter used only by optional live smoke tests."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI, OpenAIError


class LLMUnavailableError(RuntimeError):
    """Raised when the configured compatible gateway cannot serve a request."""


@dataclass(frozen=True, slots=True)
class LLMChatResult:
    """One bounded chat completion plus non-sensitive execution metadata."""

    model: str
    content: str
    elapsed_seconds: float
    finish_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return diagnostics without the full answer text."""
        return {
            "model": self.model,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "finish_reason": self.finish_reason,
            "response_chars": len(self.content),
            "usage": self.usage,
        }


class OpenAICompatibleLLM:
    """Read LLM_* settings and expose models, health, and a short chat call."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        api_key: str | None = None,
        timeout: float = 20.0,
        client: OpenAI | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must be non-empty")
        if not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout
        self.client = client or OpenAI(
            base_url=self.base_url,
            api_key=api_key or "local-anonymous",
            timeout=timeout,
            max_retries=0,
        )

    @classmethod
    def from_env(cls, **overrides: Any) -> "OpenAICompatibleLLM":
        """Build from LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL_ID."""
        base_url = overrides.pop("base_url", None) or os.environ.get("LLM_BASE_URL")
        model_id = overrides.pop("model_id", None) or os.environ.get("LLM_MODEL_ID")
        api_key = overrides.pop("api_key", None)
        if api_key is None:
            api_key = os.environ.get("LLM_API_KEY") or None
        if not base_url:
            raise ValueError("LLM_BASE_URL is required")
        if not model_id:
            raise ValueError("LLM_MODEL_ID is required")
        return cls(
            base_url=base_url,
            model_id=model_id,
            api_key=api_key,
            **overrides,
        )

    def list_models(self) -> tuple[str, ...]:
        """Return sorted model IDs from the gateway."""
        try:
            page = self.client.models.list()
        except OpenAIError as error:
            raise LLMUnavailableError(self._safe_error("model listing", error)) from error
        return tuple(sorted(model.id for model in page.data))

    def health(self) -> dict[str, Any]:
        """Verify model listing and report the sanitized endpoint."""
        models = self.list_models()
        return {
            "status": "ok",
            "endpoint": self.base_url,
            "model_count": len(models),
            "configured_model": self.model_id,
        }

    def chat(
        self,
        message: str,
        *,
        system: str | None = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> LLMChatResult:
        """Send one short chat request and return content without logging it."""
        if not message.strip():
            raise ValueError("message must be non-empty")
        if not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be within [1, 4096]")
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        started = time.perf_counter()
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except OpenAIError as error:
            raise LLMUnavailableError(self._safe_error("chat completion", error)) from error
        elapsed = time.perf_counter() - started

        choice = completion.choices[0] if completion.choices else None
        content = choice.message.content if choice and choice.message.content else ""
        usage: dict[str, Any] = {}
        if completion.usage is not None:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
        return LLMChatResult(
            model=completion.model or self.model_id,
            content=content,
            elapsed_seconds=elapsed,
            finish_reason=choice.finish_reason if choice else None,
            usage=usage,
        )

    @staticmethod
    def _safe_error(operation: str, error: OpenAIError) -> str:
        status = getattr(error, "status_code", None)
        suffix = f" HTTP {status}" if status is not None else ""
        return f"{operation} failed ({type(error).__name__}{suffix})"