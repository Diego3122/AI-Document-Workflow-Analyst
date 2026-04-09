# Phased Build Workflow

## Phase 1. Foundation and skeleton

- Repository structure
- FastAPI app shell
- Streamlit app shell
- shared Pydantic models
- config loading
- SQLite bootstrap
- health and workflow routes

## Phase 2. Document ingestion

- upload endpoint
- raw file storage
- parser service
- structure-aware chunking

## Phase 3. Retrieval and indexing

- embedding service
- Chroma persistence
- vector retrieval
- hybrid retrieval hooks

## Phase 4. Controlled agent workflow

- router agent
- retrieval agent
- synthesis agent
- validation agent

## Phase 5. Reliability and trust

- retry policy
- schema repair logic
- confidence thresholds
- human review triggers

## Phase 6. Observability and evaluation

- request logs
- retrieval traces
- metrics endpoint
- offline evaluation dataset

## Phase 7. Portfolio polish

- clean UI formatting
- domain-relevant sample documents
- stronger README and demo assets
