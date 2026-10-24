from pathlib import Path
from uuid import uuid4

from app.rag.models import Document


def load_text_documents(directory: str) -> list[Document]:
    documents = []

    data_directory = Path(directory)

    for file_path in data_directory.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")

        document = Document(
            document_id=str(uuid4()),
            title=file_path.stem,
            content=content,
            source=str(file_path),
        )

        documents.append(document)

    return documents