
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Enterprise Knowledge Intelligence Platform",
    description="Hybrid RAG API with semantic and keyword retrieval",
    version="1.0.0",
)

pipeline = RAGPipeline()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=3, ge=1, le=10)


class EvidenceResponse(BaseModel):
    source: str
    chunk_index: int
    score: float
    content: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    evidence: list[EvidenceResponse]


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        response = pipeline.answer(
            question=request.question,
            top_k=request.top_k,
        )

        evidence = [
            EvidenceResponse(
                source=item.chunk.source,
                chunk_index=item.chunk.chunk_index,
                score=item.score,
                content=item.chunk.content,
            )
            for item in response.retrieved_chunks
        ]

        return AskResponse(
            question=response.question,
            answer=response.answer,
            sources=response.sources,
            evidence=evidence,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
