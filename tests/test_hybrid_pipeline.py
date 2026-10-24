from app.rag.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

question = "How do I fix CAT-V2-ERROR-4097?"

response = pipeline.answer(
    question=question,
    top_k=3
)

print("\nQUESTION:", response.question)
print("\nANSWER:", response.answer)
print("\nSOURCES:")

for source in response.sources:
    print("-", source)

print("\nRETRIEVED EVIDENCE:")

for item in response.retrieved_chunks:
    print("-" * 60)
    print("Source:", item.chunk.source)
    print("Chunk:", item.chunk.chunk_index)
    print("RRF score:", round(item.score, 5))
    print("Content:", item.chunk.content)