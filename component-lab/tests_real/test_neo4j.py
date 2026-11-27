"""Live Neo4j integration tests; requires component-lab/.env and local Neo4j."""

from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from dotenv import load_dotenv

from real_backends.neo4j_store import Neo4jGraphStore

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


class Neo4jAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        probe = None
        try:
            probe = Neo4jGraphStore.from_env()
            probe.health()
        except Exception as exc:
            raise unittest.SkipTest(
                f"live Neo4j is unavailable ({exc.__class__.__name__})"
            ) from None
        finally:
            if probe is not None:
                probe.close()

    def setUp(self) -> None:
        self.namespace = f"test-{uuid.uuid4().hex}"
        self.store = Neo4jGraphStore.from_env(default_namespace=self.namespace)

    def tearDown(self) -> None:
        self.store.close()

    def add_entity(
        self,
        entity_id: str,
        *,
        entity_type: str = "Concept",
        name: str | None = None,
        namespace: str | None = None,
    ) -> object:
        return self.store.upsert_entity(
            entity_id,
            entity_type=entity_type,
            properties={"name": name or entity_id},
            namespace=namespace,
        )

    def add_relation(
        self,
        relation_id: str,
        source_id: str,
        target_id: str,
        *,
        relation_type: str = "RELATES_TO",
        namespace: str | None = None,
    ) -> object:
        return self.store.upsert_relation(
            relation_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            source="tests_real",
            evidence=f"{source_id} {relation_type} {target_id}",
            properties={"verified": True},
            namespace=namespace,
        )

    def test_health_entity_relation_upsert_are_idempotent(self) -> None:
        health = self.store.health()
        self.assertEqual("ok", health["status"])
        self.assertEqual("neo4j", health["backend"])
        self.assertTrue(health["version"])

        self.add_entity("entity-a", name="Alpha Agent")
        self.add_entity("entity-b", name="Beta Graph")
        entity = self.store.search_entities("alpha agent", limit=5)[0]
        duplicate = self.add_entity("entity-a", name="Alpha Agent")
        self.assertEqual(entity, duplicate)

        relation = self.add_relation("relation-a", "entity-a", "entity-b")
        duplicate_relation = self.add_relation("relation-a", "entity-a", "entity-b")
        self.assertEqual(relation, duplicate_relation)
        self.assertEqual("tests_real", relation.source)
        self.assertIn("entity-a", relation.evidence)
        self.assertTrue(relation.properties["verified"])

    def test_namespace_isolates_entities_and_relations(self) -> None:
        other_namespace = f"test-{uuid.uuid4().hex}"
        for namespace in (self.namespace, other_namespace):
            self.add_entity("shared-source", name=f"source-{namespace}", namespace=namespace)
            self.add_entity("shared-target", name=f"target-{namespace}", namespace=namespace)
            self.add_relation("shared-relation", "shared-source", "shared-target", namespace=namespace)

        hits = self.store.search_entities("shared-source")
        self.assertEqual(1, len(hits))
        self.assertEqual(self.namespace, hits[0].namespace)
        self.assertEqual(f"source-{self.namespace}", hits[0].properties["name"])
        other_hits = self.store.search_entities("shared-source", namespace=other_namespace)
        self.assertEqual(1, len(other_hits))
        self.assertEqual(other_namespace, other_hits[0].namespace)
        self.assertEqual(
            (),
            self.store.search_entities("shared-source", namespace="namespace-that-does-not-exist"),
        )
        self.assertEqual(1, len(self.store.neighbors("shared-source")))
        self.assertEqual(
            1, len(self.store.neighbors("shared-source", namespace=other_namespace))
        )

    def test_neighbors_support_direction_type_and_limit(self) -> None:
        for entity_id in ("a", "b", "c", "d"):
            self.add_entity(entity_id)
        self.add_relation("a-works-b", "a", "b", relation_type="WORKS_WITH")
        self.add_relation("c-cites-a", "c", "a", relation_type="CITES")
        self.add_relation("b-works-d", "b", "d", relation_type="WORKS_WITH")

        both = self.store.neighbors("a", limit=10)
        self.assertEqual({"b", "c"}, {item.entity.entity_id for item in both})
        self.assertEqual({"out", "in"}, {item.direction for item in both})
        outgoing = self.store.neighbors("a", direction="out")
        self.assertEqual(("b",), tuple(item.entity.entity_id for item in outgoing))
        incoming = self.store.neighbors("a", direction="in")
        self.assertEqual(("c",), tuple(item.entity.entity_id for item in incoming))
        filtered = self.store.neighbors("a", relation_type="WORKS_WITH")
        self.assertEqual(("b",), tuple(item.entity.entity_id for item in filtered))
        self.assertEqual(1, len(self.store.neighbors("a", limit=1)))
        self.assertEqual((), self.store.neighbors("missing"))

    def test_relation_id_cannot_be_rebound_to_different_endpoints(self) -> None:
        for entity_id in ("a", "b", "c"):
            self.add_entity(entity_id)
        self.add_relation("stable", "a", "b")
        with self.assertRaises(ValueError):
            self.add_relation("stable", "a", "c")
        neighbors = self.store.neighbors("a")
        self.assertEqual(("b",), tuple(item.entity.entity_id for item in neighbors))

    def test_invalid_neighbor_direction_is_rejected(self) -> None:
        self.add_entity("a")
        with self.assertRaises(ValueError):
            self.store.neighbors("a", direction="sideways")


if __name__ == "__main__":
    unittest.main()
