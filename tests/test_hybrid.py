from app.rag.document_loader import load_text_documents
from app.rag.chunker import chunk_document
from app.rag.hybrid_retriever import HybridRetriever

documents = load_text_documents("data/raw")

chunks = []
for document in documents:
    chunks.extend(chunk_document(document))

retriever = HybridRetriever(chunks)

questions = [
    "How do I fix CAT-V2-ERROR-4097?",
    "What authentication does Catalog V2 use?",
    "What should I do when the API rate limit is exceeded?"
]

for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)

    results = retriever.retrieve(question, top_k=3)

    for rank, (chunk, score) in enumerate(results, start=1):
        print("-" * 60)
        print("Rank:", rank)
        print("Source:", chunk.source)
        print("Chunk:", chunk.chunk_index)
        print("RRF score:", round(score, 5))
        print("Content:", chunk.content)