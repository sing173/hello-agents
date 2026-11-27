"""Optional adapter tests; run only with the component-lab .venv."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from qdrant_client import QdrantClient

from real_backends.llm import OpenAICompatibleLLM
from real_backends.qdrant_store import HashingEmbedder, QdrantRAGStore


class QdrantAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = QdrantRAGStore(client=QdrantClient(location=":memory:"))

    def test_add_search_is_idempotent_and_namespace_filtered(self) -> None:
        first = self.store.add_text(
            "Pandas categorical dtypes reduce memory usage.",
            document_id="doc-a",
            namespace="team-a",
        )
        duplicate = self.store.add_text(
            "Pandas categorical dtypes reduce memory usage.",
            document_id="doc-a",
            namespace="team-a",
        )
        self.store.add_text(
            "Qdrant filters payload namespaces during vector retrieval.",
            document_id="doc-b",
            namespace="team-a",
        )
        self.store.add_text(
            "Unrelated graph traversal document.",
            document_id="doc-c",
            namespace="team-b",
        )

        self.assertEqual(first, duplicate)
        hits = self.store.search("namespace vector retrieval", namespace="team-a", top_k=2)
        self.assertEqual("doc-b", hits[0].document.id)
        self.assertEqual("qdrant", hits[0].source)
        self.assertEqual(2, self.store.stats(namespace="team-a")["points"])
        self.assertEqual(
            (),
            self.store.search("namespace vector retrieval", namespace="missing"),
        )

    def test_same_document_id_with_new_content_fails(self) -> None:
        self.store.add_text("Original text", document_id="stable", namespace="test")
        with self.assertRaises(ValueError):
            self.store.add_text("Changed text", document_id="stable", namespace="test")

    def test_hashing_embedder_is_stable_and_normalized(self) -> None:
        embedder = HashingEmbedder(dimension=64)
        first = embedder.embed("memory optimization for agents")
        second = embedder.embed("memory optimization for agents")
        norm = sum(value * value for value in first) ** 0.5

        self.assertEqual(first, second)
        self.assertAlmostEqual(1.0, norm, places=12)
        self.assertEqual(64, len(first))


class _FakeCompletions:
    def __init__(self) -> None:
        self.request: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.request = kwargs
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ready"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )


class _FakeModels:
    def list(self) -> SimpleNamespace:
        return SimpleNamespace(
            data=[SimpleNamespace(id="model-b"), SimpleNamespace(id="model-a")]
        )


class _FakeOpenAI:
    def __init__(self) -> None:
        self.models = _FakeModels()
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class LLMAdapterTests(unittest.TestCase):
    def test_models_and_bounded_chat_use_configured_model(self) -> None:
        fake = _FakeOpenAI()
        llm = OpenAICompatibleLLM(
            base_url="http://127.0.0.1:9999/v1",
            model_id="general-chat",
            client=fake,
        )

        self.assertEqual(("model-a", "model-b"), llm.list_models())
        result = llm.chat("hello", system="short", max_tokens=12)

        self.assertEqual("general-chat", result.model)
        self.assertEqual("ready", result.content)
        self.assertEqual(12, fake.completions.request["max_tokens"])
        self.assertEqual({"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}, result.usage)
        self.assertNotIn("hello", result.summary())
        self.assertNotIn("ready", result.summary())


if __name__ == "__main__":
    unittest.main()