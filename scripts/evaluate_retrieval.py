import json

from app.rag.retriever import SemanticRetriever


def load_evaluation_questions(
    path: str
) -> list[dict]:

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_recall_at_k(
    retriever: SemanticRetriever,
    questions: list[dict],
    k: int
) -> float:

    hits = 0

    for item in questions:

        results = retriever.retrieve(
            item["question"],
            top_k=k
        )

        expected_document = item["expected_document"]
        expected_chunk_index = item["expected_chunk_index"]

        found = any(
            chunk.source.endswith(
                f"{expected_document}.txt"
            )
            and chunk.chunk_index == expected_chunk_index
            for chunk, _ in results
        )

        if found:
            hits += 1

    return hits / len(questions)


if __name__ == "__main__":

    retriever = SemanticRetriever()

    questions = load_evaluation_questions(
        "data/evaluation/retrieval_questions.json"
    )

    print(f"Evaluation questions: {len(questions)}")

    for k in [1, 3, 5]:

        recall = evaluate_recall_at_k(
            retriever,
            questions,
            k
        )

        print(
            f"Recall@{k}: {recall:.2%}"
        )