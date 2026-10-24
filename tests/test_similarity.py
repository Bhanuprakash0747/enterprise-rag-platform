from app.rag.similarity import cosine_similarity


vector_a = [1, 0]
vector_b = [0.8, 0.6]

similarity = cosine_similarity(vector_a, vector_b)

print("Cosine similarity:", similarity)