# AI Document Workflow Analyst

AI Document Workflow Analyst is a document question-answering app built with FastAPI and Streamlit. It lets a user upload documents, prepare them for search, ask questions against the indexed content, and review grounded answers with citations.

## What It Does

- uploads PDF and TXT documents
- parses and chunks document text for retrieval
- creates embeddings with Gemini, OpenAI, or a local fallback
- answers questions with cited evidence from prepared files
- flags lower-confidence results for review
- records metrics and evaluation data for debugging and iteration

## Stack

- FastAPI backend
- Streamlit frontend
- SQLite for app state and logs
- Chroma or SQLite-backed retrieval components
- Gemini, OpenAI, or local fallback providers
- Docker and Azure App Service deployment assets

## Local Run

Create a local `.env` from [`.env.example`](.env.example), then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.api.main:app --reload
```

Start the UI in a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run ui\streamlit_app.py
```

## Basic Workflow

1. Upload a PDF or TXT file.
2. Prepare the file for search.
3. Ask a question against one or more prepared files.
4. Review the answer, citations, and confidence signals.

## Configuration

The app supports:

- `LLM_PROVIDER=gemini|openai|auto`
- `EMBEDDING_PROVIDER=gemini|openai|auto`

For local development, use [`.env.example`](.env.example).  
For deployment settings, use [`production.env.example`](production.env.example) as a reference.

## Deployment

The repository includes a Docker-based deployment path for Azure App Service. The container runs:

- Streamlit on port `8501`
- FastAPI on `127.0.0.1:8000` inside the same container

Production deployment should use:

- Microsoft Entra authentication
- Azure App Service settings or Key Vault for secrets
- persistent storage for uploads, logs, and vector data

## Tests

Run the test suite with:

```powershell
python -m pytest -q
```
