
import unittest
from unittest.mock import Mock

from app.rag.models import Chunk, RAGResponse
from app.rag.rag_pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        # Create the pipeline without loading real models or FAISS.
        self.pipeline = RAGPipeline.__new__(RAGPipeline)

        self.pipeline.retriever = Mock()
        self.pipeline.context_builder = Mock()
        self.pipeline.generator = Mock()

    def test_empty_question_raises_error(self):
        with self.assertRaises(ValueError):
            self.pipeline.answer("   ")

        self.pipeline.retriever.retrieve.assert_not_called()
        self.pipeline.generator.generate.assert_not_called()

    def test_empty_retrieval_returns_fallback(self):
        self.pipeline.retriever.retrieve.return_value = []

        response = self.pipeline.answer(
            "Who founded the company?"
        )

        self.assertIsInstance(response, RAGResponse)
        self.assertEqual(
            response.answer,
            "I don't have enough information "
            "in the provided documents."
        )
        self.assertEqual(response.sources, [])
        self.assertEqual(response.retrieved_chunks, [])

        self.pipeline.context_builder.build_context.assert_not_called()
        self.pipeline.generator.generate.assert_not_called()

    def test_empty_context_returns_fallback(self):
        chunk = Chunk(
            chunk_id="test_0",
            document_id="test",
            content="Some retrieved text",
            source="test.txt",
            chunk_index=0
        )

        self.pipeline.retriever.retrieve.return_value = [
            (chunk, 0.8)
        ]
        self.pipeline.context_builder.build_context.return_value = "  "

        response = self.pipeline.answer(
            "What does the document say?"
        )

        self.assertEqual(
            response.answer,
            "I don't have enough information "
            "in the provided documents."
        )
        self.assertEqual(response.sources, [])
        self.pipeline.generator.generate.assert_not_called()

    def test_successful_answer_returns_structured_response(self):
        chunk = Chunk(
            chunk_id="test_0",
            document_id="test",
            content="Catalog V2 uses OAuth2.",
            source="catalog.txt",
            chunk_index=0
        )

        self.pipeline.retriever.retrieve.return_value = [
            (chunk, 0.91)
        ]
        self.pipeline.context_builder.build_context.return_value = (
            "Catalog V2 uses OAuth2."
        )
        self.pipeline.generator.generate.return_value = (
            "Catalog V2 uses OAuth2."
        )

        response = self.pipeline.answer(
            "What authentication does Catalog V2 use?"
        )

        self.assertIsInstance(response, RAGResponse)
        self.assertEqual(
            response.answer,
            "Catalog V2 uses OAuth2."
        )
        self.assertEqual(response.sources, ["catalog.txt"])
        self.assertEqual(len(response.retrieved_chunks), 1)
        self.assertEqual(
            response.retrieved_chunks[0].score,
            0.91
        )
        self.assertEqual(
            response.retrieved_chunks[0].chunk.chunk_id,
            "test_0"
        )

        self.pipeline.generator.generate.assert_called_once()

    def test_top_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            self.pipeline.answer("Test question", top_k=0)

        self.pipeline.retriever.retrieve.assert_not_called()

    def test_threshold_removes_low_score_chunks(self):
        strong_chunk = Chunk(
            chunk_id="strong",
            document_id="test",
            content="Relevant information",
            source="relevant.txt",
            chunk_index=0
        )

        weak_chunk = Chunk(
            chunk_id="weak",
            document_id="test",
            content="Unrelated information",
            source="unrelated.txt",
            chunk_index=1
        )

        self.pipeline.retriever.retrieve.return_value = [
            (strong_chunk, 0.80),
            (weak_chunk, 0.30)
        ]
        self.pipeline.context_builder.build_context.return_value = (
            "Relevant information"
        )
        self.pipeline.generator.generate.return_value = "Answer"

        response = self.pipeline.answer(
            "Question",
            min_score=0.50
        )

        self.assertEqual(len(response.retrieved_chunks), 1)
        self.assertEqual(
            response.retrieved_chunks[0].chunk.chunk_id,
            "strong"
        )

    def test_threshold_rejecting_all_chunks_abstains(self):
        chunk = Chunk(
            chunk_id="weak",
            document_id="test",
            content="Weak match",
            source="test.txt",
            chunk_index=0
        )

        self.pipeline.retriever.retrieve.return_value = [
            (chunk, 0.20)
        ]

        response = self.pipeline.answer(
            "Question",
            min_score=0.50
        )

        self.assertEqual(response.sources, [])
        self.assertEqual(response.retrieved_chunks, [])
        self.pipeline.generator.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)