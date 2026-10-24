import json
from dataclasses import asdict, dataclass

from app.rag.models import Chunk


@dataclass
class VectorRecord:
    chunk: Chunk
    embedding: list[float]


class InMemoryVectorStore:

    def __init__(self):
        self.records: list[VectorRecord] = []

    def add(
        self,
        chunk: Chunk,
        embedding: list[float]
    ) -> None:

        record = VectorRecord(
            chunk=chunk,
            embedding=embedding
        )

        self.records.append(record)

    def add_many(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]]
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        for chunk, embedding in zip(chunks, embeddings):
            self.add(chunk, embedding)

    def get_all(self) -> list[VectorRecord]:
        return self.records

    def count(self) -> int:
        return len(self.records)

    def save(self, path: str) -> None:

        data = [
            {
                "chunk": asdict(record.chunk),
                "embedding": record.embedding
            }
            for record in self.records
        ]

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file)

    def load(self, path: str) -> None:

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.records = []

        for item in data:

            chunk = Chunk(
                **item["chunk"]
            )

            record = VectorRecord(
                chunk=chunk,
                embedding=item["embedding"]
            )

            self.records.append(record)