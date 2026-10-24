
import faiss
import numpy as np

from app.rag.models import Chunk
from app.rag.vector_store import InMemoryVectorStore


class FaissVectorStore:

    def __init__(self):
        self.index = None
        self.chunks: list[Chunk] = []

    def build(self, vector_store: InMemoryVectorStore) -> None:
        records = vector_store.get_all()

        if not records:
            raise ValueError("Cannot build an index with no vectors")

        embeddings = np.asarray(
            [record.embedding for record in records],
            dtype="float32"
        )

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain invalid values")

        dimension = embeddings.shape[1]

        if any(len(record.embedding) != dimension for record in records):
            raise ValueError("All embeddings must have the same dimension")

        # Copy so normalization doesn't mutate the original records.
        embeddings = embeddings.copy()
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        # FAISS ID i maps to chunks[i].
        self.chunks = [record.chunk for record in records]

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3
    ) -> list[tuple[Chunk, float]]:

        if self.index is None:
            raise RuntimeError("FAISS index has not been built or loaded")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        if query.shape[1] != self.index.d:
            raise ValueError(
                f"Query dimension {query.shape[1]} "
                f"does not match index dimension {self.index.d}"
            )

        if not np.isfinite(query).all():
            raise ValueError("Query embedding contains invalid values")

        query = query.copy()
        faiss.normalize_L2(query)

        k = min(top_k, self.index.ntotal)

        scores, ids = self.index.search(query, k)

        results = []

        for score, vector_id in zip(scores[0], ids[0]):
            if vector_id < 0:
                continue

            chunk = self.chunks[int(vector_id)]
            results.append((chunk, float(score)))

        return results

    def save(self, path: str) -> None:
        if self.index is None:
            raise RuntimeError("No FAISS index to save")

        faiss.write_index(self.index, path)

    def load(
        self,
        path: str,
        vector_store: InMemoryVectorStore
    ) -> None:
        index = faiss.read_index(path)
        records = vector_store.get_all()

        if index.ntotal != len(records):
            raise ValueError(
                "FAISS index and metadata record counts do not match"
            )

        if records and index.d != len(records[0].embedding):
            raise ValueError(
                "FAISS index dimension and metadata dimensions do not match"
            )

        self.index = index
        self.chunks = [record.chunk for record in records]

    def count(self) -> int:
        if self.index is None:
            return 0

        return self.index.ntotal