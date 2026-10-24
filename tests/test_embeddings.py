from app.rag.embedding_service import EmbeddingService


embedding_service = EmbeddingService()

text = "Catalog V2 uses OAuth2 client credentials authentication."

embedding = embedding_service.embed_text(text)

print("Embedding dimensions:", len(embedding))
print("First 10 values:", embedding[:10])