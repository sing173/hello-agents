"""Convert a document with MarkItDown and feed its text to offline RAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.rag import RAGTool
from real_backends.markitdown_adapter import MarkItDownAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "file_path",
        nargs="?",
        type=Path,
        default=ROOT / "data" / "sample_knowledge.md",
        help="local document to convert (default: data/sample_knowledge.md)",
    )
    args = parser.parse_args()
    try:
        loaded = MarkItDownAdapter().load_document(args.file_path)
        rag = RAGTool(default_namespace="component-lab-smoke")
        document = rag.add_text(
            loaded.content,
            doc_id="smoke-markitdown-document",
            namespace="component-lab-smoke",
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
        report = {
            "status": "ok",
            "source_path": loaded.source_path,
            "format": loaded.format,
            "content_length": len(loaded.content),
            "rag_document_id": document.id,
            "rag_namespace": document.namespace,
            "idempotent": duplicate == document,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": exc.__class__.__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
