"""Behavior and safety tests for Markdown/YAML notes."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from components.notes import (
    NoteFormatError,
    NoteNotFoundError,
    NoteSecurityError,
    NoteTool,
)


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **delta: float) -> None:
        self.value += timedelta(**delta)


class NoteToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name) / "notes"
        self.clock = MutableClock(datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
        self.notes = NoteTool(self.workspace, clock=self.clock)

    def test_create_read_update_delete_round_trip(self) -> None:
        created = self.notes.create(
            "Project status",
            "## Done\n- data model\n\n## Next\n- API",
            note_type="task_state",
            tags=("refactor", "phase1"),
        )
        self.clock.advance(minutes=10)
        updated = self.notes.update(
            created.id,
            title="Project status v2",
            content="## Done\n- model and API",
            tags=("refactor", "phase2"),
        )

        self.assertEqual(created.id, updated.id)
        self.assertEqual(created.created_at, updated.created_at)
        self.assertGreater(updated.updated_at, created.updated_at)
        self.assertEqual("task_state", updated.note_type)
        self.assertEqual(("refactor", "phase2"), updated.tags)

        reread = self.notes.read(created.id)
        self.assertEqual(updated, reread)
        raw = (self.workspace / f"{created.id}.md").read_text(encoding="utf-8")
        self.assertTrue(raw.startswith("---\nid: "))
        self.assertIn("tags: [\"refactor\", \"phase2\"]", raw)
        self.assertTrue(raw.endswith("\n"))

        self.assertTrue(self.notes.delete(created.id))
        self.assertFalse(self.notes.delete(created.id))
        with self.assertRaises(NoteNotFoundError):
            self.notes.read(created.id)

    def test_notes_persist_across_tool_instances_and_list_in_creation_order(self) -> None:
        first = self.notes.create("First", "alpha", note_type="action", tags=("one",))
        self.clock.advance(seconds=2)
        second = self.notes.create("Second", "beta", note_type="blocker", tags=("two",))

        reopened = NoteTool(self.workspace)
        listed = reopened.list_notes()
        self.assertEqual([first.id, second.id], [note.id for note in listed])
        self.assertEqual([second.id], [note.id for note in reopened.list_notes(note_type="blocker")])
        self.assertEqual([first.id], [note.id for note in reopened.list_notes(tag="one")])

    def test_search_matches_titles_and_markdown_bodies(self) -> None:
        self.notes.create("Dependency conflict", "Resolve incompatible versions.")
        target = self.notes.create("Refactor plan", "Use a virtual environment for isolation.")
        hits = self.notes.search("virtual environment")
        self.assertEqual(target.id, hits[0].note.id)
        self.assertEqual(0, len(self.notes.search("quantum banana")))

    def test_plain_yaml_subset_from_upstream_style_is_accepted(self) -> None:
        note_id = "note_20260102_030405_7777"
        path = self.workspace / f"{note_id}.md"
        path.write_text(
            "---\n"
            f"id: {note_id}\n"
            "title: Plain metadata\n"
            "type: task_state\n"
            "tags: [refactor, phase1]\n"
            "created_at: 2026-01-02T03:04:05+00:00\n"
            "updated_at: 2026-01-02T03:04:05+00:00\n"
            "---\n\n"
            "Body text\n",
            encoding="utf-8",
        )

        note = self.notes.read(note_id)
        self.assertEqual("Plain metadata", note.title)
        self.assertEqual(("refactor", "phase1"), note.tags)
        self.assertEqual("Body text", note.content)

    def test_path_traversal_and_invalid_ids_are_rejected(self) -> None:
        for invalid in ("../../etc/passwd", "/etc/passwd", "note_bad", "note_20260102_030405"):
            with self.assertRaises(NoteSecurityError):
                self.notes.read(invalid)
        with self.assertRaises(NoteSecurityError):
            self.notes.delete("../victim")

    def test_symlink_escape_is_rejected(self) -> None:
        outside = Path(self.temporary.name) / "outside.md"
        outside.write_text("secret", encoding="utf-8")
        note_id = "note_20260102_030405_8888"
        link = self.workspace / f"{note_id}.md"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("symbolic links are unavailable")
        with self.assertRaises(NoteSecurityError):
            self.notes.read(note_id)

    def test_malformed_front_matter_fails_loudly(self) -> None:
        note_id = "note_20260102_030405_9999"
        path = self.workspace / f"{note_id}.md"

        path.write_text("No front matter\n", encoding="utf-8")
        with self.assertRaises(NoteFormatError):
            self.notes.read(note_id)

        path.write_text(f"---\nid: {note_id}\n", encoding="utf-8")
        with self.assertRaises(NoteFormatError):
            self.notes.read(note_id)

        path.write_text(
            "---\n"
            f"id: {note_id}\n"
            "title: Nested\n"
            "type: general\n"
            "tags:\n"
            "  - nested\n"
            "created_at: 2026-01-02T03:04:05+00:00\n"
            "updated_at: 2026-01-02T03:04:05+00:00\n"
            "---\n",
            encoding="utf-8",
        )
        with self.assertRaises(NoteFormatError):
            self.notes.read(note_id)

    def test_run_facade_returns_json_and_supports_empty_body(self) -> None:
        import json

        created = json.loads(self.notes.run("create", title="Empty", content=""))
        loaded = json.loads(self.notes.run("read", note_id=created["id"]))
        self.assertEqual("", loaded["content"])
        self.assertEqual(created["id"], loaded["id"])


if __name__ == "__main__":
    unittest.main()