from app.rag.document_loader import load_text_documents


documents = load_text_documents("data/raw")

print(f"Number of documents: {len(documents)}")

for document in documents:
    print("\n-----------------------------")
    print(f"ID: {document.document_id}")
    print(f"Title: {document.title}")
    print(f"Source: {document.source}")
    print("Content:")
    print(document.content)