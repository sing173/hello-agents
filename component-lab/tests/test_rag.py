"""Behavior tests for the offline TF-IDF/Jaccard RAG skeleton."""

from __future__ import annotations

import unittest

from components.rag import RAGTool


class RAGToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rag = RAGTool(default_namespace="notes")
        self.memory_doc = self.rag.add_text(
            "Memory optimization strategies reduce RAM usage in long running agents.",
            doc_id="memory",
        )
        self.graph_doc = self.rag.add_text(
            "Graph databases store entities and relationships for traversal.",
            doc_id="graph",
        )
        self.rag.add_text(
            "System monitoring tracks memory usage and CPU load.",
            doc_id="monitor",
        )

    def test_add_text_is_idempotent_and_namespaces_are_isolated(self) -> None:
        same = self.rag.add_text(
            "Memory optimization strategies reduce RAM usage in long running agents.",
            doc_id="memory",
        )
        other_namespace = self.rag.add_text(
            "Memory optimization strategies reduce RAM usage in long running agents.",
            namespace="other",
        )
        auto_one = self.rag.add_text("Unique text about vector indexes.")
        auto_two = self.rag.add_text("Unique text about vector indexes.")

        self.assertEqual(self.memory_doc, same)
        self.assertEqual(self.memory_doc.text, other_namespace.text)
        self.assertNotEqual(self.memory_doc.id, other_namespace.id)
        self.assertEqual(auto_one.id, auto_two.id)
        self.assertEqual(5, len(self.rag))
        self.assertEqual(1, self.rag.stats(namespace="other")["documents"])
        self.assertEqual(1, len(self.rag.search("Memory", namespace="other", min_score=0.01)))

    def test_add_text_rejects_reusing_id_for_different_content(self) -> None:
        with self.assertRaises(ValueError):
            self.rag.add_text("Completely different content.", doc_id="memory")
        with self.assertRaises(ValueError):
            self.rag.add_text("   ")

    def test_exact_query_ranks_matching_document_first(self) -> None:
        hits = self.rag.search("Memory optimization", top_k=2)

        self.assertEqual(self.memory_doc.id, hits[0].document.id)
        self.assertTrue(hits[0].exact_match)
        self.assertGreater(hits[0].tfidf_score, 0.0)
        self.assertGreater(hits[0].jaccard_score, 0.0)
        self.assertGreater(hits[0].score, hits[1].score)

    def test_morphological_and_cjk_near_queries_match_without_embeddings(self) -> None:
        near = self.rag.search("optimizing memories", top_k=1)
        self.assertEqual(self.memory_doc.id, near[0].document.id)
        self.assertFalse(near[0].exact_match)
        self.assertGreater(near[0].score, 0.0)

        self.rag.add_text(
            "记忆系统负责保存用户偏好，并支持上下文检索。",
            doc_id="memory-zh",
        )
        chinese = self.rag.search("记忆检索", top_k=1)
        self.assertEqual("memory-zh", chinese[0].document.id)
        self.assertFalse(chinese[0].exact_match)

    def test_search_filters_by_namespace_score_and_unrelated_query(self) -> None:
        self.assertEqual(1, len(self.rag.search("graph", namespace="notes")))
        self.assertEqual(0, len(self.rag.search("graph", namespace="missing")))
        self.assertEqual(0, len(self.rag.search("quantum banana", namespace="notes")))
        self.assertEqual(0, len(self.rag.search("graph", min_score=0.99)))
        with self.assertRaises(ValueError):
            self.rag.search("   ")
        with self.assertRaises(ValueError):
            self.rag.search("graph", min_score=2.0)

    def test_ask_skeleton_returns_grounded_prompt_without_an_answer(self) -> None:
        skeleton = self.rag.ask_skeleton("How can agents reduce memory use?", top_k=1)

        self.assertIsNone(skeleton.answer)
        self.assertEqual("offline-skeleton", skeleton.generator)
        self.assertEqual(self.memory_doc.id, skeleton.hits[0].document.id)
        self.assertIn("[Evidence]", skeleton.prompt)
        self.assertIn("No model is invoked", skeleton.prompt)

        empty = self.rag.ask_skeleton("Completely unrelated question")
        self.assertEqual((), empty.hits)
        self.assertIn("no matching evidence", empty.prompt)


if __name__ == "__main__":
    unittest.main()