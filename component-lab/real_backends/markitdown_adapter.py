"""Optional MarkItDown document adapter with explicit, sanitized failures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DocumentConversionError(RuntimeError):
    """MarkItDown could not convert a readable document."""


class EmptyDocumentError(DocumentConversionError):
    """The converter returned no usable text."""


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    content: str
    source_path: str
    format: str


class MarkItDownAdapter:
    """Convert local documents while preserving the caller's source path."""

    def __init__(self, *, converter: Any | None = None) -> None:
        if converter is None:
            try:
                from markitdown import MarkItDown
            except ImportError:
                raise ImportError(
                    "MarkItDownAdapter requires the 'real' extra: "
                    "pip install -e '.[real]'"
                ) from None
            converter = MarkItDown(enable_plugins=False)
        self.converter = converter

    def load_document(self, file_path: str | Path) -> LoadedDocument:
        """Return Markdown text and provenance for a readable local document."""
        original_source_path = str(file_path)
        if not original_source_path.strip():
            raise ValueError("file_path must be non-empty")
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"document does not exist: {original_source_path}")
        if not path.is_file():
            raise IsADirectoryError(f"document is not a regular file: {original_source_path}")

        try:
            result = self.converter.convert(path)
        except Exception as exc:
            raise DocumentConversionError(
                f"MarkItDown conversion failed for {original_source_path} "
                f"({exc.__class__.__name__})"
            ) from None

        content = getattr(result, "text_content", None)
        if not isinstance(content, str) or not content.strip():
            raise EmptyDocumentError(
                f"MarkItDown returned no usable text for {original_source_path}"
            )
        document_format = path.suffix.lower().lstrip(".") or "unknown"
        return LoadedDocument(
            content=content,
            source_path=original_source_path,
            format=document_format,
        )
