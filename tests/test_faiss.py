import faiss
import numpy as np

from app.rag.embedding_service import EmbeddingService
from app.rag.vector_store import InMemoryVectorStore


VECTOR_STORE_PATH = "data/index/vector_store.json"
FAISS_INDEX_PATH = "data/index/faiss.index"


# --------------------------------------------------
# Load our existing vector store
# --------------------------------------------------

vector_store = InMemoryVectorStore()

vector_store.load(
    VECTOR_STORE_PATH
)

records = vector_store.get_all()

embeddings = np.array(
    [record.embedding for record in records],
    dtype="float32"
)

print("Number of vectors:", embeddings.shape[0])
print("Embedding dimensions:", embeddings.shape[1])


# --------------------------------------------------
# Build FAISS index
# --------------------------------------------------

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("Vectors in FAISS index:", index.ntotal)


# --------------------------------------------------
# Save FAISS index
# --------------------------------------------------

faiss.write_index(
    index,
    FAISS_INDEX_PATH
)

print("FAISS index saved successfully.")


# --------------------------------------------------
# Load FAISS index
# --------------------------------------------------

loaded_index = faiss.read_index(
    FAISS_INDEX_PATH
)

print(
    "Vectors loaded from FAISS index:",
    loaded_index.ntotal
)


# --------------------------------------------------
# Query
# --------------------------------------------------

query = "What authentication does Catalog V2 use?"

embedding_service = EmbeddingService()

query_embedding = embedding_service.embed_text(
    query
)

query_embedding = np.array(
    [query_embedding],
    dtype="float32"
)

faiss.normalize_L2(query_embedding)


# --------------------------------------------------
# Search
# --------------------------------------------------

top_k = 3

scores, indices = loaded_index.search(
    query_embedding,
    top_k
)


print("\nQuery:")
print(query)

print("\nFAISS Results:")

for rank, (score, index_id) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    record = records[index_id]

    print("\n-----------------------------")
    print(f"Rank: {rank}")
    print(f"FAISS index ID: {index_id}")
    print(f"Similarity: {score:.4f}")
    print(f"Chunk index: {record.chunk.chunk_index}")
    print(f"Source: {record.chunk.source}")
    print("Content:")
    print(record.chunk.content)