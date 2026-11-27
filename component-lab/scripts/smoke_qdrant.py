"""Live Qdrant smoke test; requires component-lab/.venv and a running Qdrant."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from real_backends.qdrant_store import QdrantRAGStore

load_dotenv(ROOT / ".env", override=False)

NAMESPACE = "component-lab-smoke"


def main() -> int:
    store = QdrantRAGStore.from_env()
    health = store.health()
    first = store.add_text(
        "Pandas memory optimization uses categorical dtypes to reduce RAM.",
        document_id="smoke-pandas",
        namespace=NAMESPACE,
        metadata={"source": "smoke-test"},
    )
    second = store.add_text(
        "Qdrant stores vectors and filters payload namespaces during retrieval.",
        document_id="smoke-qdrant",
        namespace=NAMESPACE,
        metadata={"source": "smoke-test"},
    )
    duplicate = store.add_text(
        "Pandas memory optimization uses categorical dtypes to reduce RAM.",
        document_id="smoke-pandas",
        namespace=NAMESPACE,
        metadata={"source": "smoke-test"},
    )
    hits = store.search(
        "How does Qdrant filter retrieval by namespace?",
        top_k=2,
        namespace=NAMESPACE,
    )
    stats = store.stats(namespace=NAMESPACE)
    if duplicate.id != first.id:
        raise RuntimeError("idempotent add returned a different document id")
    if not hits:
        raise RuntimeError("Qdrant search returned no hits")
    if stats["points"] != 2:
        raise RuntimeError(f"unexpected namespace point count: {stats['points']}")
    if any(hit.source != "qdrant" for hit in hits):
        raise RuntimeError("unexpected result source")

    report = {
        "status": "ok",
        "health": health,
        "collection": store.collection_name,
        "namespace": NAMESPACE,
        "written": [first.id, second.id],
        "idempotent": duplicate.id == first.id,
        "stats": stats,
        "top_hit": {
            "document_id": hits[0].document.id,
            "namespace": hits[0].document.namespace,
            "score": round(hits[0].score, 6),
            "source": hits[0].source,
            "collection": hits[0].collection,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())