from app.rag.document_loader import load_text_documents
from app.rag.chunker import chunk_document
from app.rag.bm25_retriever import BM25Retriever

# 1. Load documents
documents = load_text_documents("data/raw")

# 2. Create chunks
chunks = []

for document in documents:
    chunks.extend(chunk_document(document))

print("Documents loaded:", len(documents))
print("Chunks created:", len(chunks))

# 3. Initialize BM25
retriever = BM25Retriever(chunks)

# 4. Search
question = "How do I fix CAT-V2-ERROR-4097?"
results = retriever.retrieve(question, top_k=3)

print("\nQUESTION:", question)
print("\nBM25 RESULTS")

for chunk, score in results:
    print("-" * 60)
    print("Source:", chunk.source)
    print("Chunk:", chunk.chunk_index)
    print("Score:", round(score, 4))
    print("Content:", chunk.content)