
from app.rag.embedding_service import EmbeddingService
from app.rag.faiss_vector_store import FaissVectorStore
from app.rag.vector_store import InMemoryVectorStore


class SemanticRetriever:

    def __init__(
        self,
        vector_store_path: str = "data/index/vector_store.json",
        faiss_index_path: str = "data/index/faiss.index"
    ):
        self.embedding_service = EmbeddingService()

        metadata_store = InMemoryVectorStore()
        metadata_store.load(vector_store_path)

        self.vector_store = FaissVectorStore()
        self.vector_store.load(
            faiss_index_path,
            metadata_store
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ):
        query_embedding = self.embedding_service.embed_text(
            query
        )

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        return results