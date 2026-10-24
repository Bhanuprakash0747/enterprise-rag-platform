# Enterprise RAG Platform

A production-oriented Retrieval-Augmented Generation (RAG) platform that answers questions from enterprise documents using semantic search, keyword retrieval, and a locally hosted Large Language Model (LLM).

## Overview

Enterprise documents often contain useful information spread across multiple files. This project retrieves relevant document passages and uses them as context for generating grounded answers.

The system is designed to reduce unsupported answers by instructing the LLM to use only the supplied context and acknowledge when the documents do not contain sufficient information.

## Architecture
1. **Document ingestion** - Load source documents from `data/raw`.
2. **Chunking** - Split documents into smaller passages for retrieval.
3. **Embeddings** - Generate semantic embeddings using Sentence Transformers.
4. **Vector indexing** - Store embeddings and chunk metadata; use FAISS for similarity search.
5. **Hybrid retrieval** - Combine semantic search with BM25 using Reciprocal Rank Fusion (RRF).
6. **Context construction** - Assemble retrieved passages for the LLM.
7. **Answer generation** - Use Ollama with the local `llama3.2:3b` model.
8. **API delivery** - Expose question answering through FastAPI.

## Technology Stack

- Python 3.12
- FastAPI and Uvicorn
- Sentence Transformers
- FAISS
- BM25 keyword retrieval
- Reciprocal Rank Fusion
- Ollama and Llama 3.2
- Pydantic
- Python unittest

## Project Structure

```text
enterprise-rag-platform/
|-- app/
|   |-- main.py
|   `-- rag/
|       |-- bm25_retriever.py
|       |-- chunker.py
|       |-- context_builder.py
|       |-- document_loader.py
|       |-- embedding_service.py
|       |-- faiss_vector_store.py
|       |-- generation_service.py
|       |-- hybrid_retriever.py
|       |-- models.py
|       |-- prompt_template.py
|       |-- rag_pipeline.py
|       |-- retriever.py
|       `-- vector_store.py
|-- data/
|   |-- raw/
|   `-- index/
|-- scripts/
|-- tests/
|-- .env.example
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Prerequisites

- Python 3.12
- Ollama installed locally
- A compatible local Ollama model
- Internet access for the initial Python package and embedding-model downloads

No paid LLM API key is required when using local Ollama.

## Setup

### 1. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` if you have not created `.env` already.

Configure:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
RAG_TOP_K=3
```

Keep `.env` out of version control.

### 4. Start Ollama

Ensure Ollama is installed and running. Download the model if necessary:

```powershell
ollama pull llama3.2:3b
```

### 5. Add documents

Place supported source documents in `data/raw`.

### 6. Build the retrieval index

Run from the project root:

```powershell
python -m scripts.index_documents
```

This creates the persisted vector metadata and FAISS index under `data/index`.

### 7. Start the API

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## API Usage

### Health check

`GET /health`

Example response:

```json
{
  "status": "healthy"
}
```

### Ask a question

`POST /ask`

Example request:

```json
{
  "question": "What authentication does Catalog V2 use?",
  "top_k": 3
}
```

The response includes the question, generated answer, source filenames, and retrieved evidence passages.

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

## Retrieval Evaluation

An earlier retrieval evaluation on six questions produced these results:

| Metric | Result |
|---|---:|
| Recall@1 | 83.33% |
| Recall@3 | 100% |
| Recall@5 | 100% |

These are results from the existing evaluation set, not a guarantee of performance on unseen documents or questions.

## Key Features

- Semantic retrieval with vector embeddings
- Keyword retrieval using BM25
- Hybrid ranking using Reciprocal Rank Fusion
- Persistent FAISS indexing
- Locally hosted LLM generation
- Source-aware answers and retrieved evidence
- FastAPI endpoints for integration
- Environment-based configuration
- Retrieval evaluation scripts

## Limitations

- Answer quality depends on document quality and retrieval relevance.
- The local LLM requires sufficient system resources and may have higher latency on CPU-only machines.
- Retrieval metrics depend on the evaluation dataset.
- The system should be evaluated for security, access control, reliability, and performance before production deployment.

## Future Improvements

- Automated API integration tests
- Retrieval and answer-quality evaluation
- Authentication and authorization
- Structured logging and monitoring
- Document-level access control
- Performance benchmarking and deployment hardening

## License

Add an appropriate license before distributing the project publicly.
