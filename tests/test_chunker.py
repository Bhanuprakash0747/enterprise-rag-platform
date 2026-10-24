from app.rag.chunker import chunk_document
from app.rag.document_loader import load_text_documents


documents = load_text_documents("data/raw")

for document in documents:
    chunks = chunk_document(document)

    print("\n======================================")
    print(f"Document: {document.title}")
    print(f"Number of chunks: {len(chunks)}")

    for chunk in chunks:
        print("\n--------------------------------------")
        print(f"Chunk index: {chunk.chunk_index}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(chunk.content)