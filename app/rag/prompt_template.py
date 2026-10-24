
def build_rag_prompt(question: str, context: str) -> str:
    return f"""You are an assistant for an enterprise knowledge platform.

Answer the user's question using only the provided context.

Rules:
1. Use only information supported by the context.
2. Do not invent facts or fill gaps with assumptions.
3. If the context does not contain the answer, say:
   "I don't have enough information in the provided documents."
4. If sources disagree, explain the disagreement.
5. Keep the answer clear and concise.
6. Mention the source filenames that support your answer.

Context:
<context>
{context}
</context>

Question:
{question}

Answer:
"""