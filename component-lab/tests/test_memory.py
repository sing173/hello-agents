"""Behavior tests for the offline MemoryManager and MemoryTool."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from components.memory import MemoryManager, MemoryTool


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **delta: float) -> None:
        self.value += timedelta(**delta)


class MemoryManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
        self.memory = MemoryManager(clock=self.clock)

    def test_add_is_idempotent_for_normalized_content_within_scope(self) -> None:
        first = self.memory.add(
            "Python   developer",
            memory_type="semantic",
            scope="user:alice",
            importance=0.8,
        )
        duplicate = self.memory.add(
            " python developer ",
            memory_type="semantic",
            scope="user:alice",
            importance=0.2,
        )

        self.assertEqual(first.id, duplicate.id)
        self.assertEqual(0.8, duplicate.importance)
        self.assertEqual(1, len(self.memory))

    def test_scope_isolation_applies_to_reads_writes_and_removal(self) -> None:
        alice = self.memory.add("React frontend engineer", scope="user:alice")
        bob = self.memory.add("React frontend engineer", scope="user:bob")

        self.assertNotEqual(alice.id, bob.id)
        self.assertEqual([alice.id], [hit.record.id for hit in self.memory.search("React", scope="user:alice")])
        self.assertEqual(2, len(self.memory.search("React")))
        with self.assertRaises(KeyError):
            self.memory.get(alice.id, scope="user:bob")
        with self.assertRaises(KeyError):
            self.memory.update(alice.id, scope="user:bob", importance=0.9)
        self.assertFalse(self.memory.remove(alice.id, scope="user:bob"))
        self.assertEqual(2, len(self.memory))

    def test_search_ranks_matches_and_honors_filters(self) -> None:
        frontend = self.memory.add(
            "Alice builds React frontend interfaces",
            memory_type="semantic",
            importance=0.9,
        )
        backend = self.memory.add(
            "Bob maintains Python backend services",
            memory_type="semantic",
            importance=0.8,
        )
        self.memory.add(
            "Scratch note about frontend tickets",
            memory_type="working",
            importance=0.2,
        )

        hits = self.memory.search("React frontend", scope="global", limit=5)
        self.assertEqual(frontend.id, hits[0].record.id)
        self.assertGreater(hits[0].score, 0.0)
        self.assertNotIn(backend.id, [hit.record.id for hit in hits])

        filtered = self.memory.search("frontend", memory_type="working")
        self.assertEqual(1, len(filtered))
        self.assertEqual(0, len(self.memory.search("frontend", min_importance=0.95)))

    def test_update_preserves_id_and_remove_reports_missing(self) -> None:
        original = self.memory.add("Initial note", tags=("alpha",))
        self.clock.advance(minutes=5)
        updated = self.memory.update(
            original.id,
            content="Updated note",
            importance=0.75,
            tags=("alpha", "beta"),
        )

        self.assertEqual(original.id, updated.id)
        self.assertEqual(original.created_at, updated.created_at)
        self.assertGreater(updated.updated_at, original.updated_at)
        self.assertEqual(("alpha", "beta"), updated.tags)
        self.assertEqual(0, len(self.memory.search("Initial")))

        self.assertTrue(self.memory.remove(original.id))
        self.assertFalse(self.memory.remove(original.id))
        self.assertEqual(0, len(self.memory))

    def test_forget_supports_importance_time_and_capacity(self) -> None:
        self.memory.add("Low value", memory_type="working", importance=0.1)
        self.memory.add("High value", memory_type="working", importance=0.9)
        removed = self.memory.forget("importance_based", threshold=0.2)
        self.assertEqual(1, len(removed))
        self.assertEqual(1, len(self.memory))

        timed = MemoryManager(clock=self.clock)
        old = timed.add("Old event", memory_type="episodic", importance=0.8)
        self.clock.advance(days=31)
        fresh = timed.add("Fresh event", memory_type="episodic", importance=0.8)
        removed = timed.forget("time_based", scope="global", max_age_days=30)
        self.assertEqual((old.id,), removed)
        self.assertIn(fresh.id, [record.id for record in timed.list_records()])

        capacity = MemoryManager(clock=self.clock)
        capacity.add("Low", importance=0.1)
        capacity.add("High", importance=0.9)
        capacity.add("Medium", importance=0.5)
        capacity.forget("capacity_based", max_items=1)
        remaining = capacity.list_records()
        self.assertEqual(1, len(remaining))
        self.assertEqual("High", remaining[0].content)

    def test_consolidation_promotes_by_threshold_and_is_idempotent(self) -> None:
        important = self.memory.add(
            "Important working memory",
            memory_type="working",
            importance=0.8,
        )
        self.memory.add("Minor working memory", memory_type="working", importance=0.4)

        promoted = self.memory.consolidate(
            from_type="working",
            to_type="episodic",
            importance_threshold=0.7,
        )
        self.assertEqual((important.id,), promoted)
        self.assertEqual("episodic", self.memory.get(important.id).memory_type)
        self.assertEqual((), self.memory.consolidate(
            from_type="working",
            to_type="episodic",
            importance_threshold=0.7,
        ))

    def test_invalid_values_fail_loudly(self) -> None:
        with self.assertRaises(ValueError):
            self.memory.add("   ")
        with self.assertRaises(ValueError):
            self.memory.add("valid", memory_type="unknown")
        with self.assertRaises(ValueError):
            self.memory.add("valid", importance=1.5)
        with self.assertRaises(ValueError):
            self.memory.add("valid", scope="has space")

    def test_memory_tool_returns_json_for_tutorial_style_actions(self) -> None:
        tool = MemoryTool(default_scope="user:demo", clock=self.clock)
        added = json.loads(tool.run("add", content="Uses Pandas", importance=0.7))
        found = json.loads(tool.run("search", query="Pandas", limit=1))
        stats = json.loads(tool.run("stats"))

        self.assertEqual("user:demo", added["scope"])
        self.assertEqual(added["id"], found[0]["record"]["id"])
        self.assertEqual(1, stats["total"])


if __name__ == "__main__":
    unittest.main()