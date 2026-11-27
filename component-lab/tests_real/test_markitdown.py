"""Real MarkItDown conversion tests using temporary local documents."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from components.rag import RAGTool
from real_backends.markitdown_adapter import (
    DocumentConversionError,
    EmptyDocumentError,
    MarkItDownAdapter,
)


class MarkItDownAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter = MarkItDownAdapter()

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_markdown_conversion_preserves_source_path_and_format(self) -> None:
        path = self.directory / "knowledge.MD"
        path.write_text("# Graph Memory\n\nNeo4j stores evidence-bearing relations.\n", encoding="utf-8")
        loaded = self.adapter.load_document(path)

        self.assertIn("Graph Memory", loaded.content)
        self.assertIn("evidence-bearing relations", loaded.content)
        self.assertEqual(str(path), loaded.source_path)
        self.assertEqual("md", loaded.format)

    def test_plain_text_conversion(self) -> None:
        path = self.directory / "notes.txt"
        path.write_text("Agent memory uses scopes.\n第二行。\n", encoding="utf-8")
        loaded = self.adapter.load_document(path)

        self.assertIn("Agent memory uses scopes", loaded.content)
        self.assertIn("第二行", loaded.content)
        self.assertEqual(str(path), loaded.source_path)
        self.assertEqual("txt", loaded.format)

    def test_html_conversion(self) -> None:
        path = self.directory / "page.html"
        path.write_text(
            "<html><body><h1>Retrieval</h1><p>Namespace isolation matters.</p></body></html>",
            encoding="utf-8",
        )
        loaded = self.adapter.load_document(path)

        self.assertIn("Retrieval", loaded.content)
        self.assertIn("Namespace isolation", loaded.content)
        self.assertEqual("html", loaded.format)

    def test_missing_file_and_directory_have_clear_errors(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.adapter.load_document(self.directory / "missing.md")
        with self.assertRaises(IsADirectoryError):
            self.adapter.load_document(self.directory)

    def test_empty_converter_result_has_clear_error(self) -> None:
        path = self.directory / "empty.md"
        path.write_text(" \n", encoding="utf-8")
        converter = SimpleNamespace(
            convert=lambda _: SimpleNamespace(text_content=" \n\t")
        )
        adapter = MarkItDownAdapter(converter=converter)
        with self.assertRaises(EmptyDocumentError):
            adapter.load_document(path)

    def test_converter_failure_is_sanitized(self) -> None:
        def fail(_: Any) -> None:
            raise OSError("sensitive converter detail")

        adapter = MarkItDownAdapter(converter=SimpleNamespace(convert=fail))
        path = self.directory / "broken.pdf"
        path.write_bytes(b"not-a-pdf")
        with self.assertRaises(DocumentConversionError) as context:
            adapter.load_document(path)
        self.assertIn("OSError", str(context.exception))
        self.assertNotIn("sensitive converter detail", str(context.exception))

    def test_loaded_document_can_be_added_to_existing_rag(self) -> None:
        path = self.directory / "rag.txt"
        path.write_text("Graph neighbors expose evidence and direction.", encoding="utf-8")
        loaded = self.adapter.load_document(path)
        rag = RAGTool(default_namespace="documents")
        document = rag.add_text(
            loaded.content,
            doc_id="converted-document",
            namespace="documents",
            metadata={
                "source": "markitdown",
                "source_path": loaded.source_path,
                "format": loaded.format,
            },
        )
        duplicate = rag.add_text(
            loaded.content,
            doc_id=document.id,
            namespace=document.namespace,
            metadata=document.metadata,
        )

        self.assertEqual(document, duplicate)
        self.assertEqual(loaded.source_path, document.metadata["source_path"])
        self.assertEqual("txt", document.metadata["format"])
        self.assertTrue(rag.search("neighbors evidence"))


if __name__ == "__main__":
    unittest.main()
