import re

from app.rag.models import Chunk, Document


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_document(
    document: Document,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Chunk]:

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences = split_sentences(document.content)

    chunks: list[Chunk] = []

    current_sentences: list[str] = []
    current_length = 0
    chunk_index = 0

    for sentence in sentences:

        sentence_length = len(sentence)

        if (
            current_sentences
            and current_length + sentence_length + 1 > chunk_size
        ):
            chunk_text = " ".join(current_sentences)

            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}_{chunk_index}",
                    document_id=document.document_id,
                    content=chunk_text,
                    source=document.source,
                    chunk_index=chunk_index,
                )
            )

            chunk_index += 1

            overlap_sentences: list[str] = []
            overlap_length = 0

            for previous_sentence in reversed(current_sentences):

                if overlap_length + len(previous_sentence) + 1 > chunk_overlap:
                    break

                overlap_sentences.insert(0, previous_sentence)
                overlap_length += len(previous_sentence) + 1

            current_sentences = overlap_sentences
            current_length = overlap_length

        current_sentences.append(sentence)
        current_length += sentence_length + 1

    if current_sentences:
        chunk_text = " ".join(current_sentences)

        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}_{chunk_index}",
                document_id=document.document_id,
                content=chunk_text,
                source=document.source,
                chunk_index=chunk_index,
            )
        )

    return chunks