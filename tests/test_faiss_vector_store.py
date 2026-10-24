
from app.rag.embedding_service import EmbeddingService
from app.rag.faiss_vector_store import FaissVectorStore
from app.rag.vector_store import InMemoryVectorStore

VECTOR_STORE_PATH = "data/index/vector_store.json"
FAISS_INDEX_PATH = "data/index/faiss.index"

# Load chunk metadata
vector_store = InMemoryVectorStore()
vector_store.load(VECTOR_STORE_PATH)

# Load the existing FAISS index
faiss_store = FaissVectorStore()
faiss_store.load(
    FAISS_INDEX_PATH,
    vector_store
)

print("FAISS vectors loaded:", faiss_store.count())

# Embed the query
embedding_service = EmbeddingService()
query = "What authentication does Catalog V2 use?"
query_embedding = embedding_service.embed_text(query)

# Search the loaded index
results = faiss_store.search(
    query_embedding,
    top_k=3
)

print("\nResults:\n")

for rank, (chunk, score) in enumerate(results, start=1):
    print("-----------------------------")
    print("Rank:", rank)
    print("Similarity:", round(score, 4))
    print("Source:", chunk.source)
    print("Chunk index:", chunk.chunk_index)
    print("Content:")
    print(chunk.content)