"""Live Neo4j smoke test; requires component-lab/.venv and local Neo4j."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from real_backends.neo4j_store import Neo4jGraphStore

load_dotenv(ROOT / ".env", override=False)

NAMESPACE = "component-lab-smoke"


def http_health(url: str) -> int:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=10) as response:
        return int(response.status)


def main() -> int:
    try:
        with Neo4jGraphStore.from_env() as store:
            health = store.health()
            entity = store.upsert_entity(
                "smoke-neo4j",
                entity_type="Technology",
                properties={"name": "Neo4j", "description": "property graph database"},
                namespace=NAMESPACE,
            )
            duplicate = store.upsert_entity(
                "smoke-neo4j",
                entity_type="Technology",
                properties={"name": "Neo4j", "description": "property graph database"},
                namespace=NAMESPACE,
            )
            target = store.upsert_entity(
                "smoke-knowledge-graph",
                entity_type="Concept",
                properties={"name": "Knowledge Graph", "description": "entities and relations"},
                namespace=NAMESPACE,
            )
            relation = store.upsert_relation(
                "smoke-stores",
                source_id=entity.entity_id,
                target_id=target.entity_id,
                relation_type="STORES",
                source="component-lab-smoke",
                evidence="Neo4j stores graph entities and evidence-bearing relationships.",
                properties={"verified": True},
                namespace=NAMESPACE,
            )
            duplicate_relation = store.upsert_relation(
                "smoke-stores",
                source_id=entity.entity_id,
                target_id=target.entity_id,
                relation_type="STORES",
                source="component-lab-smoke",
                evidence="Neo4j stores graph entities and evidence-bearing relationships.",
                properties={"verified": True},
                namespace=NAMESPACE,
            )
            hits = store.search_entities("knowledge graph", namespace=NAMESPACE)
            neighbors = store.neighbors(entity.entity_id, namespace=NAMESPACE)
            if entity != duplicate or relation != duplicate_relation:
                raise RuntimeError("Neo4j upserts were not idempotent")
            if not any(hit.entity_id == target.entity_id for hit in hits):
                raise RuntimeError("Neo4j entity search missed the expected entity")
            if not any(item.relation.relation_id == relation.relation_id for item in neighbors):
                raise RuntimeError("Neo4j neighbor query missed the expected relationship")
            http_url = os.environ.get("NEO4J_HTTP_URL", "http://127.0.0.1:7474")
            report = {
                "status": "ok",
                "health": health,
                "http_url": http_url,
                "http_status": http_health(http_url.rstrip("/") + "/"),
                "namespace": NAMESPACE,
                "entity_ids": [entity.entity_id, target.entity_id],
                "idempotent": entity == duplicate and relation == duplicate_relation,
                "search_hits": [hit.entity_id for hit in hits],
                "neighbors": [
                    {
                        "entity_id": item.entity.entity_id,
                        "relation_id": item.relation.relation_id,
                        "direction": item.direction,
                        "source": item.relation.source,
                        "has_evidence": bool(item.relation.evidence),
                    }
                    for item in neighbors
                ],
            }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": exc.__class__.__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
