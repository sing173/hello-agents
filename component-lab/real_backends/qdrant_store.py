"""Optional real Qdrant backend with deterministic local vectors.

The adapter is deliberately outside ``components/`` so importing the offline lab
never requires qdrant-client. Vectors are deterministic feature hashes, not
neural embeddings; Qdrant still owns storage, filtering, and cosine retrieval.
"""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from components.rag import RAGDocument
from components.text import normalize_text, tokenize

_POINT_NAMESPACE = uuid.UUID("f90de1a2-4b74-4f6e-8e89-9e4bfafed3cb")


@dataclass(frozen=True, slots=True)
class HashingEmbedder:
    """Signed feature hashing over the same deterministic tokens as offline RAG."""

    dimension: int = 384

    def __post_init__(self) -> None:
        if not 8 <= self.dimension <= 65536:
            raise ValueError("dimension must be within [8, 65536]")

    def embed(self, text: str) -> list[float]:
        """Return one L2-normalized vector for non-empty text."""
        tokens = tokenize(text)
        if not tokens:
            raise ValueError("text contains no searchable tokens")
        vector = [0.0] * self.dimension
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            raise ValueError("feature hashing produced an empty vector")
        return [value / norm for value in vector]


@dataclass(frozen=True, slots=True)
class QdrantHit:
    """One Qdrant result including storage provenance."""

    document: RAGDocument
    score: float
    source: str = "qdrant"
    collection: str = ""

    @property
    def text(self) -> str:
        """Return the matched payload text."""
        return self.document.text


class QdrantRAGStore:
    """Store and retrieve lab documents in one Qdrant collection."""

    def __init__(
        self,
        *,
        url: str = "http://127.0.0.1:6333",
        api_key: str | None = None,
        collection_name: str = "hello_agents_component_lab",
        vector_size: int = 384,
        timeout: float = 10.0,
        client: QdrantClient | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must be non-empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.url = url.rstrip("/")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.embedder = HashingEmbedder(dimension=vector_size)
        self.client = client or QdrantClient(
            url=self.url,
            api_key=api_key,
            timeout=timeout,
        )

    @classmethod
    def from_env(cls, **overrides: Any) -> "QdrantRAGStore":
        """Build from QDRANT_* variables without loading any secrets from files."""
        options: dict[str, Any] = {
            "url": os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
            "api_key": os.environ.get("QDRANT_API_KEY") or None,
            "collection_name": os.environ.get(
                "QDRANT_COLLECTION", "hello_agents_component_lab"
            ),
            "vector_size": int(os.environ.get("QDRANT_VECTOR_SIZE", "384")),
        }
        options.update(overrides)
        return cls(**options)

    def add_text(
        self,
        text: str,
        *,
        document_id: str | None = None,
        namespace: str = "default",
        metadata: Mapping[str, Any] | None = None,
    ) -> RAGDocument:
        """Idempotently upsert one text and return its stable document record."""
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("text must be non-empty")
        self._validate_namespace(namespace)
        target_id = document_id or self._content_id(namespace, normalized_text)
        if not target_id.strip() or any(character.isspace() for character in target_id):
            raise ValueError("document_id must be non-empty without whitespace")

        self._ensure_collection()
        point_id = self._point_id(namespace, target_id)
        existing = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if existing:
            payload = existing[0].payload or {}
            if normalize_text(str(payload.get("text", ""))) != normalize_text(normalized_text):
                raise ValueError("document_id already exists with different text")
            return RAGDocument(
                id=str(payload.get("document_id", target_id)),
                text=str(payload.get("text", normalized_text)),
                namespace=str(payload.get("namespace", namespace)),
                metadata=dict(payload.get("metadata") or {}),
            )

        document_metadata = dict(metadata or {})
        vector = self.embedder.embed(normalized_text)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "document_id": target_id,
                        "namespace": namespace,
                        "text": normalized_text,
                        "metadata": document_metadata,
                        "source": "qdrant",
                    },
                )
            ],
            wait=True,
        )
        return RAGDocument(
            id=target_id,
            text=normalized_text,
            namespace=namespace,
            metadata=document_metadata,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        namespace: str | None = None,
        min_score: float = 0.0,
    ) -> tuple[QdrantHit, ...]:
        """Search by cosine similarity, optionally filtered by payload namespace."""
        if not query.strip():
            raise ValueError("query must be non-empty")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be within [0.0, 1.0]")
        if top_k == 0 or not self.client.collection_exists(self.collection_name):
            return ()

        query_filter = None
        if namespace is not None:
            self._validate_namespace(namespace)
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="namespace",
                        match=qmodels.MatchValue(value=namespace),
                    )
                ]
            )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embedder.embed(query),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            score_threshold=min_score if min_score > 0.0 else None,
        )
        hits: list[QdrantHit] = []
        for point in response.points:
            payload = point.payload or {}
            if not payload.get("text"):
                continue
            document = RAGDocument(
                id=str(payload.get("document_id", point.id)),
                text=str(payload["text"]),
                namespace=str(payload.get("namespace", "default")),
                metadata=dict(payload.get("metadata") or {}),
            )
            hits.append(
                QdrantHit(
                    document=document,
                    score=float(point.score),
                    collection=self.collection_name,
                )
            )
        return tuple(hits)

    def stats(self, *, namespace: str | None = None) -> dict[str, Any]:
        """Return the current collection and optional namespace point count."""
        exists = self.client.collection_exists(self.collection_name)
        count = 0
        if exists:
            count_filter = None
            if namespace is not None:
                self._validate_namespace(namespace)
                count_filter = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="namespace",
                            match=qmodels.MatchValue(value=namespace),
                        )
                    ]
                )
            count = int(
                self.client.count(
                    collection_name=self.collection_name,
                    count_filter=count_filter,
                    exact=True,
                ).count
            )
        return {
            "backend": "qdrant",
            "collection": self.collection_name,
            "exists": exists,
            "points": count,
            "namespace": namespace,
            "vector_size": self.vector_size,
        }

    def health(self) -> dict[str, Any]:
        """Probe Qdrant collections without mutating storage."""
        collections = sorted(
            collection.name for collection in self.client.get_collections().collections
        )
        return {
            "status": "ok",
            "url": self.url,
            "collections": collections,
            "collection_exists": self.collection_name in collections,
        }

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self._validate_vector_size()
            return
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        except Exception:
            if not self.client.collection_exists(self.collection_name):
                raise
        self._validate_vector_size()

    def _validate_vector_size(self) -> None:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, Mapping):
            vector_config = next(iter(vectors.values())) if vectors else None
        else:
            vector_config = vectors
        actual_size = getattr(vector_config, "size", None)
        if actual_size is not None and int(actual_size) != self.vector_size:
            raise ValueError(
                f"collection vector size is {actual_size}, configured {self.vector_size}"
            )

    @staticmethod
    def _validate_namespace(namespace: str) -> None:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        if any(character.isspace() for character in namespace):
            raise ValueError("namespace must not contain whitespace")

    @staticmethod
    def _point_id(namespace: str, document_id: str) -> str:
        return str(uuid.uuid5(_POINT_NAMESPACE, f"{namespace}\0{document_id}"))

    @staticmethod
    def _content_id(namespace: str, text: str) -> str:
        digest = hashlib.sha256(f"{namespace}\0{normalize_text(text)}".encode()).hexdigest()
        return f"doc_{digest[:12]}"