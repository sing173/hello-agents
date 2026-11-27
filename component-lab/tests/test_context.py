"""Behavior tests for ContextPacket, budgets, GSSC stages, and compression."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from components.context import (
    ContextBudgetError,
    ContextBuilder,
    ContextConfig,
    ContextPacket,
    Message,
)
from components.memory import MemoryManager
from components.rag import RAGTool
from components.text import estimate_tokens


def packet(
    content: str,
    *,
    tokens: int,
    relevance: float,
    timestamp: datetime,
    kind: str = "general",
) -> ContextPacket:
    return ContextPacket(
        content=content,
        timestamp=timestamp,
        token_count=tokens,
        relevance_score=relevance,
        metadata={"type": kind},
    )


class ContextStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
        self.config = ContextConfig(
            max_tokens=100,
            reserve_ratio=0.0,
            min_relevance=0.0,
            relevance_weight=0.7,
            recency_weight=0.3,
        )
        self.builder = ContextBuilder(config=self.config, clock=lambda: self.now)

    def test_config_validates_budget_and_weights(self) -> None:
        self.assertEqual(100, self.config.available_tokens)
        with self.assertRaises(ValueError):
            ContextConfig(max_tokens=0)
        with self.assertRaises(ValueError):
            ContextConfig(reserve_ratio=1.0)
        with self.assertRaises(ValueError):
            ContextConfig(relevance_weight=0.8, recency_weight=0.3)
        with self.assertRaises(ValueError):
            ContextConfig(history_limit=-1)
        self.assertEqual(8, ContextConfig(max_tokens=10, reserve_ratio=0.2).available_tokens)

    def test_select_keeps_system_packets_and_skips_oversized_packets(self) -> None:
        system = packet(
            "system policy",
            tokens=4,
            relevance=1.0,
            timestamp=self.now,
            kind="system_instruction",
        )
        large = packet("large", tokens=50, relevance=1.0, timestamp=self.now)
        small = packet("small", tokens=5, relevance=0.8, timestamp=self.now)
        medium = packet("medium", tokens=10, relevance=0.9, timestamp=self.now)

        selected = self.builder.select([large, system, small, medium], available_tokens=20)

        self.assertEqual([system, medium, small], selected)
        self.assertLessEqual(sum(item.token_count or 0 for item in selected), 20)

    def test_system_packet_is_retained_even_when_over_budget(self) -> None:
        system = packet(
            "critical instructions",
            tokens=50,
            relevance=1.0,
            timestamp=self.now,
            kind="system_instruction",
        )
        selected = self.builder.select([system], available_tokens=5)
        self.assertEqual([system], selected)

    def test_selection_uses_recency_and_relevance_threshold(self) -> None:
        old = packet(
            "old relevant memory",
            tokens=5,
            relevance=0.8,
            timestamp=self.now - timedelta(days=30),
        )
        fresh = packet(
            "fresh relevant memory",
            tokens=5,
            relevance=0.8,
            timestamp=self.now,
        )
        irrelevant = packet(
            "irrelevant",
            tokens=5,
            relevance=0.01,
            timestamp=self.now,
        )
        config = ContextConfig(
            max_tokens=10,
            reserve_ratio=0.0,
            min_relevance=0.1,
            relevance_weight=0.7,
            recency_weight=0.3,
        )
        builder = ContextBuilder(config=config, clock=lambda: self.now)

        selected = builder.select([old, fresh, irrelevant], available_tokens=5)
        self.assertEqual([fresh], selected)

    def test_structure_groups_sections_in_template_order(self) -> None:
        selected = [
            packet("policy", tokens=1, relevance=1.0, timestamp=self.now, kind="system_instruction"),
            packet("remember Pandas", tokens=1, relevance=0.8, timestamp=self.now, kind="memory"),
            packet("use category dtype", tokens=1, relevance=0.9, timestamp=self.now, kind="rag_result"),
            packet("blocked on schema", tokens=1, relevance=0.7, timestamp=self.now, kind="task_state"),
        ]

        context = self.builder.structure(selected, "How do I save memory?")

        self.assertLess(context.index("[Role & Policies]"), context.index("[Task]"))
        self.assertLess(context.index("[Task]"), context.index("[State]"))
        self.assertLess(context.index("[State]"), context.index("[Evidence]"))
        self.assertLess(context.index("[Evidence]"), context.index("[Context]"))
        self.assertLess(context.index("[Context]"), context.index("[Output]"))
        self.assertIn("remember Pandas", context)
        self.assertIn("use category dtype", context)

    def test_compress_never_exceeds_budget_and_marks_truncation(self) -> None:
        text = "alpha beta gamma delta epsilon zeta eta theta " * 20
        compressed = self.builder.compress(text, 20)

        self.assertLessEqual(estimate_tokens(compressed), 20)
        self.assertIn("content compressed", compressed)
        self.assertEqual("", self.builder.compress(text, 0))
        tiny = self.builder.compress(text, 2)
        self.assertLessEqual(estimate_tokens(tiny), 2)

    def test_build_combines_sources_and_obeys_available_budget(self) -> None:
        memory = MemoryManager(clock=lambda: self.now)
        memory.add(
            "The user builds Pandas data pipelines and cares about RAM.",
            scope="user:alice",
            importance=0.9,
        )
        rag = RAGTool()
        rag.add_text(
            "Pandas category dtype reduces memory for repeated strings.",
            doc_id="pandas-memory",
        )
        config = ContextConfig(
            max_tokens=160,
            reserve_ratio=0.2,
            min_relevance=0.0,
            enable_compression=False,
        )
        builder = ContextBuilder(
            config=config,
            memory=memory,
            rag=rag,
            memory_scope="user:alice",
            clock=lambda: self.now,
        )

        context = builder.build(
            "How can I reduce Pandas memory?",
            conversation_history=[
                Message("I use Pandas daily", "user", self.now - timedelta(hours=1)),
                Message("Use suitable dtypes", "assistant", self.now - timedelta(minutes=50)),
            ],
            system_instructions="You are a Python data advisor.",
            custom_packets=[
                packet(
                    "Current task: audit memory",
                    tokens=5,
                    relevance=1.0,
                    timestamp=self.now,
                    kind="task_state",
                )
            ],
        )

        self.assertIn("[Role & Policies]", context)
        self.assertIn("Pandas daily", context)
        self.assertIn("category dtype", context)
        self.assertIn("audit memory", context)
        self.assertLessEqual(estimate_tokens(context), config.available_tokens)

    def test_build_compresses_or_raises_when_structure_exceeds_budget(self) -> None:
        config = ContextConfig(
            max_tokens=20,
            reserve_ratio=0.0,
            min_relevance=0.0,
            enable_compression=True,
        )
        builder = ContextBuilder(config=config, clock=lambda: self.now)
        long_custom = packet(
            "word " * 100,
            tokens=1,
            relevance=1.0,
            timestamp=self.now,
        )

        compressed = builder.build("Explain this", custom_packets=[long_custom])
        self.assertLessEqual(estimate_tokens(compressed), config.available_tokens)
        self.assertIn("content compressed", compressed)

        strict = ContextBuilder(
            config=ContextConfig(
                max_tokens=20,
                reserve_ratio=0.0,
                min_relevance=0.0,
                enable_compression=False,
            ),
            clock=lambda: self.now,
        )
        with self.assertRaises(ContextBudgetError):
            strict.build("Explain this", custom_packets=[long_custom])

    def test_gather_validates_query_and_honors_history_limit(self) -> None:
        with self.assertRaises(ValueError):
            self.builder.gather("   ")
        config = ContextConfig(history_limit=0)
        builder = ContextBuilder(config=config, clock=lambda: self.now)
        packets = builder.gather(
            "hello",
            conversation_history=[Message("old message", "user", self.now)],
        )
        self.assertEqual([], packets)


if __name__ == "__main__":
    unittest.main()