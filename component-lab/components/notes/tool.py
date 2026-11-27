"""Markdown notes with a restricted YAML front-matter subset."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from components.text import hybrid_lexical_score

_NOTE_ID_PATTERN = re.compile(r"note_[0-9]{8}_[0-9]{6}_[0-9]{4}")
_FRONT_MATTER_KEYS = {
    "id",
    "title",
    "type",
    "tags",
    "created_at",
    "updated_at",
}


class NoteFormatError(ValueError):
    """Raised when a note file contains malformed front matter."""


class NoteSecurityError(ValueError):
    """Raised when a requested note id can resolve outside the workspace."""


class NoteNotFoundError(LookupError):
    """Raised when a valid note id does not identify an existing note."""


@dataclass(frozen=True, slots=True)
class Note:
    """A Markdown note and its validated metadata."""

    id: str
    title: str
    content: str
    note_type: str
    tags: tuple[str, ...]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class NoteHit:
    """A note paired with its lexical search score."""

    note: Note
    score: float


class NoteTool:
    """Create, read, update, delete, list, and search Markdown notes."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sequence = 0

    def create(
        self,
        title: str,
        content: str = "",
        *,
        note_type: str = "general",
        tags: Sequence[str] = (),
    ) -> Note:
        """Create a note in a new file and return its parsed record."""
        normalized_title = self._validate_title(title)
        normalized_type = self._validate_type(note_type)
        normalized_tags = self._validate_tags(tags)
        now = self._now()
        note_id = self._allocate_id(now)
        note = Note(
            id=note_id,
            title=normalized_title,
            content=content.strip("\n"),
            note_type=normalized_type,
            tags=normalized_tags,
            created_at=now,
            updated_at=now,
        )
        path = self._path_for_id(note_id)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(self._serialize(note))
        return note

    def read(self, note_id: str) -> Note:
        """Read one note by id."""
        path = self._path_for_id(note_id, must_exist=True)
        return self._parse_file(path)

    def update(
        self,
        note_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        note_type: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> Note:
        """Update selected fields and persist a new ``updated_at`` value."""
        current = self.read(note_id)
        updated = replace(
            current,
            title=current.title if title is None else self._validate_title(title),
            content=current.content if content is None else content.strip("\n"),
            note_type=(
                current.note_type if note_type is None else self._validate_type(note_type)
            ),
            tags=current.tags if tags is None else self._validate_tags(tags),
            updated_at=self._now(),
        )
        path = self._path_for_id(note_id, must_exist=True)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(self._serialize(updated))
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return updated

    def delete(self, note_id: str) -> bool:
        """Delete one note; a missing but valid id returns ``False``."""
        path = self._path_for_id(note_id)
        if path.is_symlink():
            raise NoteSecurityError("note files may not be symbolic links")
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_notes(
        self,
        *,
        note_type: str | None = None,
        tag: str | None = None,
        limit: int | None = None,
    ) -> tuple[Note, ...]:
        """List notes in stable creation order."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        notes: list[Note] = []
        for path in sorted(self.workspace.glob("note_*.md")):
            if not _NOTE_ID_PATTERN.fullmatch(path.stem):
                continue
            note = self._parse_file(self._path_for_id(path.stem, must_exist=True))
            if note_type is not None and note.note_type != note_type:
                continue
            if tag is not None and tag not in note.tags:
                continue
            notes.append(note)
        notes.sort(key=lambda note: (note.created_at, note.id))
        return tuple(notes if limit is None else notes[:limit])

    def search(self, query: str, *, limit: int = 5) -> tuple[NoteHit, ...]:
        """Search note titles and bodies with deterministic lexical scoring."""
        if not query.strip():
            raise ValueError("query must be non-empty")
        if limit < 0:
            raise ValueError("limit must be non-negative")
        hits: list[NoteHit] = []
        for note in self.list_notes():
            score = max(
                hybrid_lexical_score(query, note.title),
                hybrid_lexical_score(query, note.content),
            )
            if score > 0.0:
                hits.append(NoteHit(note=note, score=score))
        hits.sort(key=lambda hit: (-hit.score, hit.note.id))
        return tuple(hits[:limit])

    def run(self, action: str | Mapping[str, Any], **payload: Any) -> str:
        """Execute a tutorial-style action and return a JSON string."""
        fields = dict(action) if isinstance(action, Mapping) else {"action": action}
        fields.update(payload)
        name = fields.pop("action", None)
        if name == "create":
            note = self.create(
                fields.pop("title", ""),
                fields.pop("content", ""),
                note_type=fields.pop("note_type", "general"),
                tags=fields.pop("tags", ()),
            )
            return self._dump(self._note_dict(note))
        if name == "read":
            return self._dump(self._note_dict(self.read(fields.pop("note_id", ""))))
        if name == "update":
            note = self.update(
                fields.pop("note_id", ""),
                title=fields.pop("title", None),
                content=fields.pop("content", None),
                note_type=fields.pop("note_type", None),
                tags=fields.pop("tags", None),
            )
            return self._dump(self._note_dict(note))
        if name == "delete":
            return self._dump({"deleted": self.delete(fields.pop("note_id", ""))})
        if name == "list":
            notes = self.list_notes(
                note_type=fields.pop("note_type", None),
                tag=fields.pop("tag", None),
                limit=fields.pop("limit", None),
            )
            return self._dump([self._note_dict(note) for note in notes])
        if name == "search":
            hits = self.search(
                fields.pop("query", ""),
                limit=fields.pop("limit", 5),
            )
            return self._dump(
                [
                    {"score": hit.score, "note": self._note_dict(hit.note)}
                    for hit in hits
                ]
            )
        raise ValueError(f"unsupported note action: {name}")

    def _allocate_id(self, now: datetime) -> str:
        prefix = now.strftime("note_%Y%m%d_%H%M%S")
        while True:
            candidate = f"{prefix}_{self._sequence:04d}"
            self._sequence += 1
            if not self._path_for_id(candidate).exists():
                return candidate

    def _path_for_id(self, note_id: str, *, must_exist: bool = False) -> Path:
        if not isinstance(note_id, str) or not _NOTE_ID_PATTERN.fullmatch(note_id):
            raise NoteSecurityError(f"invalid note id: {note_id!r}")
        path = (self.workspace / f"{note_id}.md").resolve()
        if path.parent != self.workspace:
            raise NoteSecurityError("note path escapes the workspace")
        if path.is_symlink():
            raise NoteSecurityError("note files may not be symbolic links")
        if must_exist and (not path.exists() or not path.is_file()):
            raise NoteNotFoundError(note_id)
        return path

    def _parse_file(self, path: Path) -> Note:
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise NoteFormatError(f"note is not valid UTF-8: {path.name}") from error
        metadata, body = self._parse_markdown(raw, path.stem)
        return Note(
            id=metadata["id"],
            title=metadata["title"],
            content=body,
            note_type=metadata["type"],
            tags=tuple(metadata["tags"]),
            created_at=datetime.fromisoformat(metadata["created_at"]),
            updated_at=datetime.fromisoformat(metadata["updated_at"]),
        )

    @staticmethod
    def _parse_markdown(raw: str, expected_id: str) -> tuple[dict[str, Any], str]:
        lines = raw.splitlines()
        if not lines or lines[0] != "---":
            raise NoteFormatError("note must start with YAML front matter")
        try:
            closing_index = lines.index("---", 1)
        except ValueError as error:
            raise NoteFormatError("front matter is missing its closing delimiter") from error

        metadata: dict[str, Any] = {}
        for line in lines[1:closing_index]:
            if not line.strip():
                continue
            if line[:1].isspace() or ":" not in line:
                raise NoteFormatError("front matter supports only flat key/value lines")
            key, raw_value = line.split(":", 1)
            if key not in _FRONT_MATTER_KEYS:
                raise NoteFormatError(f"unsupported front matter key: {key}")
            if key in metadata:
                raise NoteFormatError(f"duplicate front matter key: {key}")
            metadata[key] = NoteTool._parse_scalar(raw_value.strip())

        required = {"id", "title", "type", "tags", "created_at", "updated_at"}
        missing = required - metadata.keys()
        if missing:
            raise NoteFormatError(f"missing front matter keys: {', '.join(sorted(missing))}")
        if metadata["id"] != expected_id:
            raise NoteFormatError("front matter id does not match the file name")
        if not isinstance(metadata["title"], str) or not metadata["title"].strip():
            raise NoteFormatError("title must be a non-empty string")
        if not isinstance(metadata["type"], str) or not metadata["type"].strip():
            raise NoteFormatError("type must be a non-empty string")
        if not isinstance(metadata["tags"], list) or not all(
            isinstance(tag, str) and tag.strip() for tag in metadata["tags"]
        ):
            raise NoteFormatError("tags must be a list of non-empty strings")
        for key in ("created_at", "updated_at"):
            if not isinstance(metadata[key], str):
                raise NoteFormatError(f"{key} must be an ISO-8601 string")
        body = "\n".join(lines[closing_index + 1 :]).strip("\n")
        return metadata, body

    @staticmethod
    def _parse_scalar(value: str) -> Any:
        if not value:
            return ""
        if value.startswith('"') or value.startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                if value.startswith("[") and value.endswith("]"):
                    inner = value[1:-1].strip()
                    if not inner:
                        return []
                    tags = [item.strip().strip("\"'") for item in inner.split(",")]
                    if any(not item for item in tags):
                        raise NoteFormatError("tags list contains an empty item")
                    return tags
                raise NoteFormatError("invalid quoted front matter value")
        if value[:1] in {"&", "*", "!", "|", ">", "{", "}", "%", "@", "`"}:
            raise NoteFormatError("unsupported YAML feature in front matter")
        return value.strip("\"'")

    @staticmethod
    def _serialize(note: Note) -> str:
        front_matter = [
            "---",
            f"id: {json.dumps(note.id, ensure_ascii=False)}",
            f"title: {json.dumps(note.title, ensure_ascii=False)}",
            f"type: {json.dumps(note.note_type, ensure_ascii=False)}",
            f"tags: {json.dumps(list(note.tags), ensure_ascii=False)}",
            f"created_at: {json.dumps(note.created_at.isoformat())}",
            f"updated_at: {json.dumps(note.updated_at.isoformat())}",
            "---",
            "",
        ]
        return "\n".join(front_matter) + note.content + "\n"

    @staticmethod
    def _validate_title(title: str) -> str:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        return title.strip()

    @staticmethod
    def _validate_type(note_type: str) -> str:
        if not isinstance(note_type, str) or not note_type.strip():
            raise ValueError("note_type must be a non-empty string")
        return note_type.strip()

    @staticmethod
    def _validate_tags(tags: Sequence[str]) -> tuple[str, ...]:
        result: list[str] = []
        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                raise ValueError("tags must contain non-empty strings")
            if tag.strip() not in result:
                result.append(tag.strip())
        return tuple(result)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _note_dict(note: Note) -> dict[str, Any]:
        return {
            "id": note.id,
            "title": note.title,
            "content": note.content,
            "note_type": note.note_type,
            "tags": list(note.tags),
            "created_at": note.created_at.isoformat(),
            "updated_at": note.updated_at.isoformat(),
        }

    @staticmethod
    def _dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)