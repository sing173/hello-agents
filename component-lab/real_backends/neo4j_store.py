"""Optional real Neo4j graph backend.

Nodes and relationships use properties rather than user-controlled Cypher
identifiers. Every identity includes a namespace, and all query values are bound
parameters. The module lives outside ``components/`` to preserve offline imports.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit

from neo4j import GraphDatabase, READ_ACCESS, RoutingControl, WRITE_ACCESS
from neo4j.exceptions import Neo4jError

_ENTITY_LABEL = "ComponentLabEntity"
_RELATION_LABEL = "ComponentLabRelation"
_ENTITY_RESERVED = {"namespace", "entity_id", "entity_type", "search_text"}
_RELATION_RESERVED = {
    "namespace",
    "relation_id",
    "relation_type",
    "source_id",
    "target_id",
    "source",
    "evidence",
}


class Neo4jBackendError(RuntimeError):
    """A sanitized Neo4j driver or query failure."""


@dataclass(frozen=True, slots=True)
class GraphEntity:
    namespace: str
    entity_id: str
    entity_type: str
    properties: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class GraphRelation:
    namespace: str
    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    source: str
    evidence: str
    properties: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class GraphNeighbor:
    entity: GraphEntity
    relation: GraphRelation
    direction: str


class Neo4jGraphStore:
    """Store namespace-isolated entities and evidence-bearing relationships."""

    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
        default_namespace: str = "default",
        timeout: float = 10.0,
        driver: Any | None = None,
    ) -> None:
        if not uri.startswith(("bolt://", "neo4j://", "bolt+s://", "neo4j+s://")):
            raise ValueError("uri must use a bolt:// or neo4j:// scheme")
        parsed_uri = urlsplit(uri)
        if parsed_uri.username is not None or parsed_uri.password is not None:
            raise ValueError("uri must not contain credentials")
        if not parsed_uri.hostname:
            raise ValueError("uri must include a host")
        if not user.strip():
            raise ValueError("user must be non-empty")
        if not password:
            raise ValueError("password must be non-empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._validate_namespace(default_namespace)
        self.uri = uri
        self.user = user
        self.database = database
        self.default_namespace = default_namespace
        self.driver = driver or GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=timeout,
            connection_acquisition_timeout=timeout,
            max_transaction_retry_time=timeout,
        )

    @classmethod
    def from_env(cls, **overrides: Any) -> "Neo4jGraphStore":
        """Build from NEO4J_* variables; callers load .env explicitly if desired."""
        password = os.environ.get("NEO4J_PASSWORD")
        if not password and "password" not in overrides:
            raise ValueError("NEO4J_PASSWORD is required")
        options: dict[str, Any] = {
            "uri": os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"),
            "user": os.environ.get("NEO4J_USER", "neo4j"),
            "password": password,
            "database": os.environ.get("NEO4J_DATABASE") or None,
            "default_namespace": os.environ.get("NEO4J_NAMESPACE", "default"),
            "timeout": float(os.environ.get("NEO4J_TIMEOUT", "10")),
        }
        options.update(overrides)
        return cls(**options)

    def close(self) -> None:
        """Release the underlying Neo4j driver."""
        self.driver.close()

    def __enter__(self) -> "Neo4jGraphStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def health(self) -> dict[str, Any]:
        """Verify Bolt connectivity and return sanitized server diagnostics."""
        try:
            self.driver.verify_connectivity()
            result = self.driver.execute_query(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions, edition ORDER BY name",
                database_=self.database,
                routing_=RoutingControl.READ,
            )
            components = []
            for record in result.records:
                components.append(
                    {
                        "name": record["name"],
                        "versions": list(record["versions"]),
                        "edition": record["edition"],
                    }
                )
            neo4j = next(
                (
                    component
                    for component in components
                    if str(component["name"]).lower().startswith("neo4j")
                ),
                {},
            )
            return {
                "status": "ok",
                "backend": "neo4j",
                "uri": self.uri,
                "database": self.database or "neo4j",
                "version": (neo4j.get("versions") or [None])[0],
                "edition": neo4j.get("edition"),
                "components": components,
            }
        except Exception as exc:
            raise Neo4jBackendError(
                f"Neo4j health check failed ({exc.__class__.__name__})"
            ) from None

    def upsert_entity(
        self,
        entity_id: str,
        *,
        entity_type: str,
        properties: Mapping[str, Any] | None = None,
        namespace: str | None = None,
    ) -> GraphEntity:
        """MERGE an entity by namespace/id and update its searchable properties."""
        target_namespace = self._namespace(namespace)
        self._validate_identifier("entity_id", entity_id)
        self._validate_identifier("entity_type", entity_type)
        entity_properties = self._properties(properties, _ENTITY_RESERVED)
        search_text = " ".join(
            [entity_id, entity_type, json.dumps(entity_properties, ensure_ascii=False, sort_keys=True)]
        )
        records = self._query(
            "MERGE (entity:ComponentLabEntity "
            "{namespace: $namespace, entity_id: $entity_id}) "
            "SET entity.entity_type = $entity_type, "
            "entity += $properties, entity.search_text = $search_text "
            "RETURN entity",
            {
                "namespace": target_namespace,
                "entity_id": entity_id,
                "entity_type": entity_type,
                "properties": entity_properties,
                "search_text": search_text,
            },
            write=True,
        )
        record = records[0] if records else None
        if record is None:
            raise Neo4jBackendError("Neo4j entity upsert returned no record")
        return self._entity(record["entity"], target_namespace)

    def upsert_relation(
        self,
        relation_id: str,
        *,
        source_id: str,
        target_id: str,
        relation_type: str,
        source: str,
        evidence: str,
        properties: Mapping[str, Any] | None = None,
        namespace: str | None = None,
    ) -> GraphRelation:
        """MERGE a relationship while preserving stable endpoints and provenance."""
        target_namespace = self._namespace(namespace)
        self._validate_identifier("relation_id", relation_id)
        self._validate_identifier("source_id", source_id)
        self._validate_identifier("target_id", target_id)
        self._validate_identifier("relation_type", relation_type)
        if not source.strip():
            raise ValueError("source must be non-empty")
        if not evidence.strip():
            raise ValueError("evidence must be non-empty")
        relation_properties = self._properties(properties, _RELATION_RESERVED)
        parameters = {
            "namespace": target_namespace,
            "relation_id": relation_id,
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "source": source,
            "evidence": evidence,
            "properties": relation_properties,
        }

        def write_transaction(tx: Any) -> Any:
            existing = tx.run(
                "MATCH ()-[relation:ComponentLabRelation "
                "{namespace: $namespace, relation_id: $relation_id}]->() "
                "RETURN relation.source_id AS source_id, "
                "relation.target_id AS target_id, "
                "relation.relation_type AS relation_type LIMIT 1",
                parameters,
            ).single()
            if existing is not None:
                current = (
                    existing["source_id"],
                    existing["target_id"],
                    existing["relation_type"],
                )
                expected = (source_id, target_id, relation_type)
                if current != expected:
                    raise ValueError("relation_id already exists with different endpoints or type")
            return tx.run(
                "MATCH (source:ComponentLabEntity "
                "{namespace: $namespace, entity_id: $source_id}), "
                "(target:ComponentLabEntity "
                "{namespace: $namespace, entity_id: $target_id}) "
                "MERGE (source)-[relation:ComponentLabRelation "
                "{namespace: $namespace, relation_id: $relation_id}]->(target) "
                "SET relation.source_id = $source_id, "
                "relation.target_id = $target_id, "
                "relation.relation_type = $relation_type, "
                "relation.source = $source, relation.evidence = $evidence, "
                "relation += $properties RETURN relation",
                parameters,
            ).single()

        try:
            with self.driver.session(
                database=self.database, default_access_mode=WRITE_ACCESS
            ) as session:
                record = session.execute_write(write_transaction)
        except ValueError:
            raise
        except Exception as exc:
            raise self._driver_error(exc) from None
        if record is None:
            raise ValueError("relation endpoints must exist in the same namespace")
        return self._relation(record["relation"], target_namespace)

    def search_entities(
        self,
        query: str,
        *,
        limit: int = 5,
        entity_type: str | None = None,
        namespace: str | None = None,
    ) -> tuple[GraphEntity, ...]:
        """Case-insensitive substring search over ids, types, and property text."""
        if not query.strip():
            raise ValueError("query must be non-empty")
        self._validate_limit(limit)
        target_namespace = self._namespace(namespace)
        records = self._query(
            "MATCH (entity:ComponentLabEntity {namespace: $namespace}) "
            "WHERE toLower(entity.search_text) CONTAINS $query "
            "AND ($entity_type IS NULL OR entity.entity_type = $entity_type) "
            "RETURN entity ORDER BY entity.entity_id LIMIT $limit",
            {
                "namespace": target_namespace,
                "query": query.strip().lower(),
                "entity_type": entity_type,
                "limit": limit,
            },
        )
        return tuple(self._entity(record["entity"], target_namespace) for record in records)

    def neighbors(
        self,
        entity_id: str,
        *,
        limit: int = 5,
        relation_type: str | None = None,
        direction: str = "both",
        namespace: str | None = None,
    ) -> tuple[GraphNeighbor, ...]:
        """Return incoming/outgoing/both neighbors with relationship evidence."""
        self._validate_identifier("entity_id", entity_id)
        self._validate_limit(limit)
        if direction not in {"in", "out", "both"}:
            raise ValueError("direction must be 'in', 'out', or 'both'")
        target_namespace = self._namespace(namespace)
        if limit == 0:
            return ()
        patterns = {
            "in": "(other:ComponentLabEntity {namespace: $namespace})-[relation]->(entity)",
            "out": "(entity)-[relation]->(other:ComponentLabEntity {namespace: $namespace})",
            "both": "(entity)-[relation]-(other:ComponentLabEntity {namespace: $namespace})",
        }
        direction_expression = {
            "in": "'in'",
            "out": "'out'",
            "both": "CASE WHEN startNode(relation) = entity THEN 'out' ELSE 'in' END",
        }[direction]
        records = self._query(
            "MATCH (entity:ComponentLabEntity "
            "{namespace: $namespace, entity_id: $entity_id}), "
            f"{patterns[direction]} "
            "WHERE relation.namespace = $namespace "
            "AND ($relation_type IS NULL OR relation.relation_type = $relation_type) "
            "RETURN other AS entity, relation AS relation, "
            f"{direction_expression} AS direction "
            "ORDER BY other.entity_id, relation.relation_id LIMIT $limit",
            {
                "namespace": target_namespace,
                "entity_id": entity_id,
                "relation_type": relation_type,
                "limit": limit,
            },
        )
        return tuple(
            GraphNeighbor(
                entity=self._entity(record["entity"], target_namespace),
                relation=self._relation(record["relation"], target_namespace),
                direction=str(record["direction"]),
            )
            for record in records
        )

    def _query(
        self,
        query: str,
        parameters: Mapping[str, Any],
        *,
        write: bool = False,
    ) -> list[Any]:
        try:
            result = self.driver.execute_query(
                query,
                parameters,
                database_=self.database,
                routing_=RoutingControl.WRITE if write else RoutingControl.READ,
            )
            return list(result.records)
        except Exception as exc:
            raise self._driver_error(exc) from None

    def _namespace(self, namespace: str | None) -> str:
        target = self.default_namespace if namespace is None else namespace
        self._validate_namespace(target)
        return target

    @staticmethod
    def _validate_namespace(namespace: str) -> None:
        Neo4jGraphStore._validate_identifier("namespace", namespace)

    @staticmethod
    def _validate_identifier(name: str, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        if any(character.isspace() for character in value):
            raise ValueError(f"{name} must not contain whitespace")

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if not isinstance(limit, int) or not 0 <= limit <= 100:
            raise ValueError("limit must be an integer within [0, 100]")

    @staticmethod
    def _properties(
        properties: Mapping[str, Any] | None, reserved: set[str]
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in (properties or {}).items():
            if key in reserved:
                raise ValueError(f"property {key!r} is reserved")
            if not isinstance(key, str) or not key:
                raise ValueError("property names must be non-empty strings")
            Neo4jGraphStore._validate_property_value(value)
            result[key] = value
        return result

    @staticmethod
    def _validate_property_value(value: Any) -> None:
        if isinstance(value, bool) or isinstance(value, str) or isinstance(value, int):
            return
        if isinstance(value, float) and math.isfinite(value):
            return
        if isinstance(value, list) and value:
            first_type = type(value[0])
            if first_type not in {str, int, float, bool}:
                raise ValueError("list properties may contain only scalar values")
            if any(type(item) is not first_type for item in value):
                raise ValueError("list properties must have one scalar type")
            if any(isinstance(item, float) and not math.isfinite(item) for item in value):
                raise ValueError("list properties must contain finite numbers")
            return
        raise ValueError("properties must contain Neo4j-compatible scalar values or scalar lists")

    @staticmethod
    def _entity(node: Any, namespace: str) -> GraphEntity:
        values = dict(node)
        return GraphEntity(
            namespace=namespace,
            entity_id=str(values["entity_id"]),
            entity_type=str(values["entity_type"]),
            properties={
                key: value
                for key, value in values.items()
                if key not in _ENTITY_RESERVED
            },
        )

    @staticmethod
    def _relation(relationship: Any, namespace: str) -> GraphRelation:
        values = dict(relationship)
        return GraphRelation(
            namespace=namespace,
            relation_id=str(values["relation_id"]),
            source_id=str(values["source_id"]),
            target_id=str(values["target_id"]),
            relation_type=str(values["relation_type"]),
            source=str(values["source"]),
            evidence=str(values["evidence"]),
            properties={
                key: value
                for key, value in values.items()
                if key not in _RELATION_RESERVED
            },
        )

    @staticmethod
    def _driver_error(exc: Exception) -> Neo4jBackendError:
        code = exc.code if isinstance(exc, Neo4jError) else None
        detail = code or exc.__class__.__name__
        return Neo4jBackendError(f"Neo4j operation failed ({detail})")
