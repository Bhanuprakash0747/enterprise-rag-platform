
from app.rag.rag_pipeline import RAGPipeline


pipeline = RAGPipeline()

questions = [
    "What authentication does Catalog V2 use?",
    "How long are Catalog V2 access tokens valid?",
    "What does HTTP 429 mean?",
    "How do I resolve CAT-V2-ERROR-4097?",
    "Who founded the company that created Catalog V2?"
]

for question in questions:
    response = pipeline.answer(question, top_k=3)

    print("\n" + "=" * 70)
    print("QUESTION:", response.question)
    print("=" * 70)

    print("\nANSWER:")
    print(response.answer)

    print("\nRETRIEVAL EVIDENCE:")
    for rank, item in enumerate(
        response.retrieved_chunks, start=1
    ):
        print(f"\n[{rank}] Score: {item.score:.4f}")
        print(f"Source: {item.chunk.source}")
        print(f"Chunk: {item.chunk.chunk_index}")
        print(f"Text: {item.chunk.content}")

    print("\nRETRIEVED SOURCE FILES:")
    for source in response.sources:
        print("-", source)