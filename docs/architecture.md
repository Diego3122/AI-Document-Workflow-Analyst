# Elevated Architecture

## Goal

Build a recruiter-ready AI document workflow analyst that looks and behaves like a production-minded AI system, not a thin LLM wrapper.

## Layer map

`Streamlit UI -> FastAPI -> Ingestion/Retrieval/Agent Workflow -> Reliability/Safety -> Observability/Evaluation -> Streamlit UI`

### Frontend

The Streamlit UI handles uploads, query submission, citation display, confidence display, and human review decisions.

### API and orchestration

FastAPI owns request validation, route orchestration, startup bootstrap, and response shaping.

Core endpoints:

- `POST /documents/upload`
- `GET /documents`
- `POST /documents/index`
- `POST /query`
- `POST /query/review`
- `GET /metrics`
- `GET /health`

### Ingestion layer

The ingestion layer handles file persistence, parsing, chunking, embedding generation, vector persistence, and document metadata registration.

Important metadata:

- `document_id`
- `document_name`
- `page_number`
- `section`
- `chunk_id`

### Retrieval layer

The elevated design is meant to support vector retrieval, keyword retrieval, and optional reranking while always producing citation-ready evidence packs.

### Agent workflow

Describe the system as a controlled multi-step agent workflow:

- `RouterAgent`
- `RetrievalAgent`
- `SynthesisAgent`
- `ValidationAgent`

### Reliability and safety

This layer provides schema enforcement, retries, confidence thresholds, grounding checks, and human review triggers.

Confidence should blend:

- retrieval strength
- citation coverage
- schema validity
- contradiction or missing-evidence checks

### Observability and evaluation

SQLite should capture request traces, validation outcomes, review flags, and error states. Offline evaluation should compare chunking, retrieval settings, prompt versions, and model choices.
