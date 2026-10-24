import re
from rank_bm25 import BM25Okapi

from app.rag.models import Chunk


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Chunks cannot be empty")

        self.chunks = chunks
        self.tokenized_corpus = [
            self._tokenize(chunk.content)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[\w-]+\b", text.lower())

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> list[tuple[Chunk, float]]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return [
            (self.chunks[i], float(scores[i]))
            for i in ranked_indices[:top_k]
        ]