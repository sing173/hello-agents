"""Offline RAG data models and index implementation."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

from components.text import normalize_text, tokenize


@dataclass(frozen=True, slots=True)
class RAGDocument:
    """One text document stored in a named offline namespace."""

    id: str
    text: str
    namespace: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RAGHit:
    """A retrieval result with its component scores."""

    document: RAGDocument
    score: float
    tfidf_score: float
    jaccard_score: float
    exact_match: bool

    @property
    def text(self) -> str:
        """Return the matched document text."""
        return self.document.text


@dataclass(frozen=True, slots=True)
class AnswerSkeleton:
    """Grounded prompt scaffold with no generated answer."""

    question: str
    hits: tuple[RAGHit, ...]
    prompt: str
    generator: str = "offline-skeleton"
    answer: None = None


class RAGTool:
    """Small in-process TF-IDF/Jaccard index with no embedding provider."""

    def __init__(self, *, default_namespace: str = "default") -> None:
        if not default_namespace.strip():
            raise ValueError("default_namespace must be non-empty")
        self.default_namespace = default_namespace
        self._documents: dict[tuple[str, str], RAGDocument] = {}

    def __len__(self) -> int:
        return len(self._documents)

    def add_text(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        namespace: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RAGDocument:
        """Add text idempotently; the same document id may not change content."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        target_namespace = namespace or self.default_namespace
        if not target_namespace.strip() or any(
            character.isspace() for character in target_namespace
        ):
            raise ValueError("namespace must be a non-empty string without whitespace")

        normalized_text = text.strip()
        target_id = doc_id or self._content_id(target_namespace, normalized_text)
        if not target_id.strip() or any(character.isspace() for character in target_id):
            raise ValueError("doc_id must be a non-empty string without whitespace")

        key = (target_namespace, target_id)
        existing = self._documents.get(key)
        if existing is not None:
            if normalize_text(existing.text) != normalize_text(normalized_text):
                raise ValueError("doc_id already exists with different text")
            return existing

        document = RAGDocument(
            id=target_id,
            text=normalized_text,
            namespace=target_namespace,
            metadata=dict(metadata or {}),
        )
        self._documents[key] = document
        return document

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        namespace: str | None = None,
        min_score: float = 0.0,
    ) -> tuple[RAGHit, ...]:
        """Rank documents by TF-IDF cosine, Jaccard, and exact containment."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be within [0.0, 1.0]")
        if top_k == 0:
            return ()

        documents = [
            document
            for document in self._documents.values()
            if namespace is None or document.namespace == namespace
        ]
        if not documents:
            return ()

        token_lists = [tokenize(document.text) for document in documents]
        query_tokens = tokenize(query)
        if not query_tokens:
            return ()

        idf = self._inverse_document_frequency(token_lists)
        query_vector = self._vector(query_tokens, idf)
        normalized_query = normalize_text(query)

        hits: list[RAGHit] = []
        for document, document_tokens in zip(documents, token_lists):
            document_vector = self._vector(document_tokens, idf)
            tfidf = self._cosine(query_vector, document_vector)
            query_set = set(query_tokens)
            document_set = set(document_tokens)
            intersection = query_set & document_set
            union = query_set | document_set
            jaccard = len(intersection) / len(union) if union else 0.0
            coverage = len(intersection) / len(query_set)
            exact = normalized_query in normalize_text(document.text)
            score = 0.65 * tfidf + 0.25 * jaccard + 0.10 * coverage
            if exact:
                score += 0.15
            score = min(1.0, score)
            if score >= min_score and score > 0.0:
                hits.append(
                    RAGHit(
                        document=document,
                        score=score,
                        tfidf_score=tfidf,
                        jaccard_score=jaccard,
                        exact_match=exact,
                    )
                )

        hits.sort(key=lambda hit: (-hit.score, hit.document.namespace, hit.document.id))
        return tuple(hits[:top_k])

    def ask_skeleton(
        self,
        question: str,
        *,
        top_k: int = 3,
        namespace: str | None = None,
        min_score: float = 0.0,
    ) -> AnswerSkeleton:
        """Return evidence and a grounded prompt without calling an LLM."""
        hits = self.search(
            question,
            top_k=top_k,
            namespace=namespace,
            min_score=min_score,
        )
        evidence_lines = [
            f"[{index}] ({hit.document.namespace}/{hit.document.id}, "
            f"score={hit.score:.3f}) {hit.document.text}"
            for index, hit in enumerate(hits, start=1)
        ]
        evidence = "\n".join(evidence_lines) if evidence_lines else "(no matching evidence)"
        prompt = (
            "[Task]\n"
            f"{question.strip()}\n\n"
            "[Evidence]\n"
            f"{evidence}\n\n"
            "[Output]\n"
            "Answer only from the evidence above. State when the evidence is insufficient. "
            "No model is invoked by this offline skeleton."
        )
        return AnswerSkeleton(question=question.strip(), hits=hits, prompt=prompt)

    def stats(self, *, namespace: str | None = None) -> dict[str, int]:
        """Return document counts for one namespace or the complete index."""
        return {
            "documents": sum(
                1
                for document in self._documents.values()
                if namespace is None or document.namespace == namespace
            )
        }

    def clear(self, *, namespace: str | None = None) -> int:
        """Remove documents from one namespace or all namespaces."""
        keys = [
            key
            for key in self._documents
            if namespace is None or key[0] == namespace
        ]
        for key in keys:
            del self._documents[key]
        return len(keys)

    @staticmethod
    def _content_id(namespace: str, text: str) -> str:
        digest = hashlib.sha256(f"{namespace}\0{normalize_text(text)}".encode()).hexdigest()
        return f"doc_{digest[:12]}"

    @staticmethod
    def _inverse_document_frequency(token_lists: list[tuple[str, ...]]) -> dict[str, float]:
        document_count = len(token_lists)
        frequencies: Counter[str] = Counter()
        for tokens in token_lists:
            frequencies.update(set(tokens))
        return {
            token: math.log((1 + document_count) / (1 + frequency)) + 1.0
            for token, frequency in frequencies.items()
        }

    @staticmethod
    def _vector(tokens: tuple[str, ...], idf: Mapping[str, float]) -> dict[str, float]:
        counts = Counter(tokens)
        fallback_idf = math.log(len(idf) + 1.0) + 1.0 if idf else 1.0
        return {
            token: count * idf.get(token, fallback_idf)
            for token, count in counts.items()
        }

    @staticmethod
    def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
        if not left or not right:
            return 0.0
        dot_product = sum(
            weight * right.get(token, 0.0) for token, weight in left.items()
        )
        left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
        right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
        if not left_norm or not right_norm:
            return 0.0
        return dot_product / (left_norm * right_norm)