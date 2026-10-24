from app.rag.retriever import SemanticRetriever


retriever = SemanticRetriever(
    vector_store_path="data/index/vector_store.json",
    faiss_index_path="data/index/faiss.index"
)

query = "What authentication does Catalog V2 use?"

results = retriever.retrieve(
    query,
    top_k=3
)

print("\nQuery:")
print(query)

print("\nRetrieved Results:")

for rank, (chunk, similarity) in enumerate(
    results,
    start=1
):
    print("\n-----------------------------")
    print(f"Rank: {rank}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Chunk index: {chunk.chunk_index}")
    print(f"Source: {chunk.source}")
    print("Content:")
    print(chunk.content)
