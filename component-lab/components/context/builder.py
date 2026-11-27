"""Gather-Select-Structure-Compress context builder."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Callable, Sequence

from components.memory import MemoryManager
from components.rag import RAGTool
from components.text import estimate_tokens

from .models import (
    ContextBudgetError,
    ContextConfig,
    ContextPacket,
    Message,
    utc_now,
)


class ContextBuilder:
    """Build a deterministic context string inside a token budget."""

    def __init__(
        self,
        *,
        config: ContextConfig | None = None,
        memory: MemoryManager | None = None,
        rag: RAGTool | None = None,
        memory_scope: str | None = None,
        rag_namespace: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config or ContextConfig()
        self.memory = memory
        self.rag = rag
        self.memory_scope = memory_scope
        self.rag_namespace = rag_namespace
        self._clock = clock or utc_now

    def gather(
        self,
        user_query: str,
        *,
        conversation_history: Sequence[Message] | None = None,
        system_instructions: str | None = None,
        custom_packets: Sequence[ContextPacket] | None = None,
    ) -> list[ContextPacket]:
        """Collect system, memory, RAG, history, and custom candidates."""
        if not user_query.strip():
            raise ValueError("user_query must be non-empty")
        now = self._now()
        packets: list[ContextPacket] = []

        if system_instructions:
            packets.append(
                ContextPacket(
                    content=system_instructions,
                    timestamp=now,
                    relevance_score=1.0,
                    metadata={"type": "system_instruction", "priority": "high"},
                )
            )

        if self.memory is not None:
            for hit in self.memory.search(
                user_query,
                scope=self.memory_scope,
                limit=10,
                min_importance=0.3,
            ):
                packets.append(
                    ContextPacket(
                        content=hit.record.content,
                        timestamp=hit.record.updated_at,
                        relevance_score=hit.score,
                        metadata={
                            "type": "memory",
                            "memory_id": hit.record.id,
                            "memory_type": hit.record.memory_type,
                        },
                    )
                )

        if self.rag is not None:
            for hit in self.rag.search(
                user_query,
                top_k=5,
                namespace=self.rag_namespace,
                min_score=0.3,
            ):
                packets.append(
                    ContextPacket(
                        content=hit.document.text,
                        timestamp=now,
                        relevance_score=hit.score,
                        metadata={
                            "type": "rag_result",
                            "document_id": hit.document.id,
                            "namespace": hit.document.namespace,
                        },
                    )
                )

        if conversation_history and self.config.history_limit > 0:
            recent = list(conversation_history)[-self.config.history_limit :]
            for message in recent:
                packets.append(
                    ContextPacket(
                        content=f"{message.role}: {message.content}",
                        timestamp=message.timestamp,
                        relevance_score=0.6,
                        metadata={"type": "conversation_history", "role": message.role},
                    )
                )

        if custom_packets:
            packets.extend(custom_packets)
        return packets

    def select(
        self,
        packets: Sequence[ContextPacket],
        available_tokens: int,
    ) -> list[ContextPacket]:
        """Select packets by relevance/recency while respecting the budget."""
        if available_tokens < 0:
            raise ValueError("available_tokens must be non-negative")
        system_packets = [
            packet
            for packet in packets
            if packet.metadata.get("type") == "system_instruction"
        ]
        other_packets = [
            packet
            for packet in packets
            if packet.metadata.get("type") != "system_instruction"
        ]

        scored: list[tuple[float, int, ContextPacket]] = []
        for index, packet in enumerate(other_packets):
            if packet.relevance_score < self.config.min_relevance:
                continue
            recency = self._recency(packet.timestamp)
            combined = (
                self.config.relevance_weight * packet.relevance_score
                + self.config.recency_weight * recency
            )
            scored.append((combined, index, packet))
        scored.sort(key=lambda item: (-item[0], item[1]))

        selected = list(system_packets)
        used_tokens = sum(packet.token_count or 0 for packet in selected)
        for _, _, packet in scored:
            packet_tokens = packet.token_count or 0
            if used_tokens + packet_tokens <= available_tokens:
                selected.append(packet)
                used_tokens += packet_tokens
        return selected

    def structure(self, selected_packets: Sequence[ContextPacket], user_query: str) -> str:
        """Render selected packets into the fixed context template."""
        if not user_query.strip():
            raise ValueError("user_query must be non-empty")

        system: list[str] = []
        state: list[str] = []
        evidence: list[str] = []
        context: list[str] = []
        for packet in selected_packets:
            packet_type = packet.metadata.get("type", "general")
            if packet_type == "system_instruction":
                system.append(packet.content)
            elif packet_type in {"note", "state", "task_state"}:
                state.append(packet.content)
            elif packet_type in {"rag_result", "knowledge"}:
                evidence.append(packet.content)
            else:
                context.append(packet.content)

        sections: list[str] = []
        if system:
            sections.append("[Role & Policies]\n" + "\n".join(system))
        sections.append(f"[Task]\n{user_query}")
        if state:
            sections.append("[State]\n" + "\n".join(state))
        if evidence:
            sections.append("[Evidence]\n" + "\n---\n".join(evidence))
        if context:
            sections.append("[Context]\n" + "\n".join(context))
        sections.append(
            "[Output]\nAnswer from the supplied context and state when evidence is insufficient."
        )
        return "\n\n".join(sections)

    def compress(self, context: str, max_tokens: int) -> str:
        """Truncate text deterministically so the estimate never exceeds the cap."""
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if max_tokens == 0 or not context:
            return ""
        if estimate_tokens(context) <= max_tokens:
            return context

        marker = "\n[... content compressed ...]"
        marker_tokens = estimate_tokens(marker)
        if marker_tokens >= max_tokens:
            return self._fit_prefix(context, max_tokens).rstrip()
        prefix = self._fit_prefix(context, max_tokens - marker_tokens).rstrip()
        return f"{prefix}{marker}" if prefix else marker.strip()

    def build(
        self,
        user_query: str,
        *,
        conversation_history: Sequence[Message] | None = None,
        system_instructions: str | None = None,
        custom_packets: Sequence[ContextPacket] | None = None,
    ) -> str:
        """Run all four GSSC stages and return the final bounded context."""
        packets = self.gather(
            user_query,
            conversation_history=conversation_history,
            system_instructions=system_instructions,
            custom_packets=custom_packets,
        )
        budget = self.config.available_tokens
        selected = self.select(packets, budget)
        context = self.structure(selected, user_query)
        if estimate_tokens(context) <= budget:
            return context
        if not self.config.enable_compression:
            raise ContextBudgetError(
                f"context requires {estimate_tokens(context)} tokens but budget is {budget}"
            )
        return self.compress(context, budget)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _recency(self, timestamp: datetime) -> float:
        normalized = (
            timestamp.replace(tzinfo=timezone.utc)
            if timestamp.tzinfo is None
            else timestamp.astimezone(timezone.utc)
        )
        age_hours = max(0.0, (self._now() - normalized).total_seconds() / 3600.0)
        return max(0.1, min(1.0, math.exp(-0.1 * age_hours / 24.0)))

    @staticmethod
    def _fit_prefix(text: str, target_tokens: int) -> str:
        if target_tokens <= 0:
            return ""
        low = 0
        high = len(text)
        while low < high:
            midpoint = (low + high + 1) // 2
            if estimate_tokens(text[:midpoint]) <= target_tokens:
                low = midpoint
            else:
                high = midpoint - 1
        return text[:low]