"""Context packet, message, and budget configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from components.text import estimate_tokens


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class Message:
    """Minimal conversation message accepted by the gather stage."""

    content: str
    role: str
    timestamp: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("message content must be non-empty")
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("unsupported message role")


@dataclass(slots=True)
class ContextPacket:
    """One candidate context item with a budget and relevance score."""

    content: str
    timestamp: datetime
    token_count: int | None = None
    relevance_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("context content must be a non-empty string")
        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError("relevance_score must be within [0.0, 1.0]")
        if self.token_count is None:
            self.token_count = estimate_tokens(self.content)
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative")
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        else:
            self.timestamp = self.timestamp.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Validated GSSC budget and scoring configuration."""

    max_tokens: int = 3000
    reserve_ratio: float = 0.2
    min_relevance: float = 0.1
    enable_compression: bool = True
    recency_weight: float = 0.3
    relevance_weight: float = 0.7
    history_limit: int = 5

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= self.reserve_ratio < 1.0:
            raise ValueError("reserve_ratio must be within [0.0, 1.0)")
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError("min_relevance must be within [0.0, 1.0]")
        if abs(self.recency_weight + self.relevance_weight - 1.0) > 1e-9:
            raise ValueError("recency_weight + relevance_weight must equal 1.0")
        if self.history_limit < 0:
            raise ValueError("history_limit must be non-negative")

    @property
    def available_tokens(self) -> int:
        """Tokens available to the structured context after reserve headroom."""
        return max(1, int(self.max_tokens * (1.0 - self.reserve_ratio)))


class ContextBudgetError(RuntimeError):
    """Raised when structured context exceeds budget with compression disabled."""