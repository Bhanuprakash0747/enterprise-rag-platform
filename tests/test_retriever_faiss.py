
from app.rag.retriever import SemanticRetriever

retriever = SemanticRetriever()

questions = [
    "What authentication does Catalog V2 use?",
    "How long are access tokens valid?",
    "What does HTTP 429 mean?",
    "How do I resolve CAT-V2-ERROR-4097?"
]

for question in questions:
    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)

    results = retriever.retrieve(
        question,
        top_k=3
    )

    for rank, (chunk, score) in enumerate(
        results,
        start=1
    ):
        print("\nRank:", rank)
        print("Similarity:", round(score, 4))
        print("Source:", chunk.source)
        print("Chunk:", chunk.chunk_index)
        print("Content:", chunk.content)