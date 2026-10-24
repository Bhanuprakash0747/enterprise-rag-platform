
from app.rag.context_builder import ContextBuilder
from app.rag.retriever import SemanticRetriever


retriever = SemanticRetriever()
context_builder = ContextBuilder(max_chars=6000)

question = "What authentication does Catalog V2 use?"

results = retriever.retrieve(
    question,
    top_k=3
)

context = context_builder.build_context(results)

print("QUESTION:")
print(question)

print("\nASSEMBLED CONTEXT:")
print(context)

print("\nContext characters:", len(context))