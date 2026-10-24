
from app.rag.context_builder import ContextBuilder
from app.rag.prompt_template import build_rag_prompt
from app.rag.retriever import SemanticRetriever


retriever = SemanticRetriever()
builder = ContextBuilder()

question = "What authentication does Catalog V2 use?"

results = retriever.retrieve(question, top_k=3)
context = builder.build_context(results)

prompt = build_rag_prompt(question, context)

print(prompt)