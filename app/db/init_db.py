from __future__ import annotations

from app.db.sqlite import connect


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    indexed_at TEXT,
    last_error TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    owner_subject_id TEXT,
    owner_tenant_id TEXT
);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    document_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    section TEXT,
    text TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(document_id)
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    embedding_provider TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES document_chunks(chunk_id)
);

CREATE TABLE IF NOT EXISTS request_logs (
    request_id TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    task_type TEXT NOT NULL,
    document_ids_json TEXT NOT NULL DEFAULT '[]',
    metadata_filters_json TEXT NOT NULL DEFAULT '{}',
    requested_top_k INTEGER NOT NULL DEFAULT 5,
    task_type_hint TEXT,
    llm_model TEXT,
    latency_ms REAL NOT NULL,
    retrieved_chunk_ids TEXT NOT NULL,
    retrieval_stats_json TEXT NOT NULL DEFAULT '{}',
    structured_data_json TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL,
    needs_human_review INTEGER NOT NULL,
    validation_notes TEXT NOT NULL,
    schema_valid INTEGER NOT NULL,
    grounded INTEGER NOT NULL,
    citation_count INTEGER NOT NULL,
    citation_coverage REAL NOT NULL DEFAULT 0,
    evidence_strength REAL NOT NULL DEFAULT 0,
    citation_ids_valid INTEGER NOT NULL DEFAULT 1,
    answer_support_score REAL NOT NULL DEFAULT 0,
    insufficient_evidence INTEGER NOT NULL DEFAULT 0,
    vector_backend TEXT,
    embedding_backend TEXT,
    error_message TEXT,
    auth_subject_id TEXT,
    auth_tenant_id TEXT,
    auth_role TEXT,
    auth_type TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_decisions (
    review_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    action TEXT NOT NULL,
    edited_answer TEXT,
    reviewer_notes TEXT,
    reviewer_subject_id TEXT,
    reviewer_tenant_id TEXT,
    reviewer_role TEXT,
    reviewer_auth_type TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    evaluation_id TEXT PRIMARY KEY,
    dataset_path TEXT NOT NULL,
    total_cases INTEGER NOT NULL,
    passed_cases INTEGER NOT NULL,
    pass_rate REAL NOT NULL,
    task_type_match_rate REAL NOT NULL DEFAULT 0,
    average_latency_ms REAL NOT NULL,
    average_confidence REAL NOT NULL,
    review_rate REAL NOT NULL,
    grounded_rate REAL NOT NULL,
    citation_success_rate REAL NOT NULL,
    structured_success_rate REAL NOT NULL,
    failed_case_ids_json TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_case_results (
    evaluation_case_id TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    expected_task_type TEXT,
    task_type TEXT NOT NULL,
    task_type_match INTEGER NOT NULL DEFAULT 1,
    passed INTEGER NOT NULL,
    confidence REAL NOT NULL,
    latency_ms REAL NOT NULL,
    expected_review INTEGER,
    actual_review INTEGER NOT NULL,
    expected_citation_count INTEGER NOT NULL,
    actual_citation_count INTEGER NOT NULL,
    grounded INTEGER NOT NULL,
    citation_success INTEGER NOT NULL,
    structured_success INTEGER NOT NULL,
    notes_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(evaluation_id) REFERENCES evaluation_runs(evaluation_id)
);

CREATE TABLE IF NOT EXISTS provider_usage_events (
    event_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    operation TEXT NOT NULL,
    input_chars INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    usage_date TEXT NOT NULL,
    subject_key TEXT,
    subject_id TEXT,
    tenant_id TEXT,
    auth_type TEXT
);

CREATE TABLE IF NOT EXISTS api_request_events (
    event_id TEXT PRIMARY KEY,
    subject_kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id_chunk_index
ON document_chunks(document_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_documents_owner_subject_tenant
ON documents(owner_subject_id, owner_tenant_id, created_at);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_document_id
ON chunk_embeddings(document_id);

CREATE INDEX IF NOT EXISTS idx_request_logs_created_at
ON request_logs(created_at);

CREATE INDEX IF NOT EXISTS idx_request_logs_auth_subject_created_at
ON request_logs(auth_subject_id, auth_tenant_id, created_at);

CREATE INDEX IF NOT EXISTS idx_provider_usage_events_usage_date
ON provider_usage_events(usage_date, provider);

CREATE INDEX IF NOT EXISTS idx_provider_usage_events_subject_created_at
ON provider_usage_events(provider, subject_key, created_at);

CREATE INDEX IF NOT EXISTS idx_api_request_events_subject_created_at
ON api_request_events(subject_kind, subject_id, created_at);
"""


def _ensure_column(connection, table_name: str, column_name: str, definition: str) -> None:
    existing = {row['name'] for row in connection.execute(f'PRAGMA table_info({table_name})').fetchall()}
    if column_name not in existing:
        connection.execute(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}')


def apply_schema(connection) -> None:
    connection.executescript(SCHEMA_SQL)
    _ensure_column(connection, 'documents', 'page_count', 'INTEGER NOT NULL DEFAULT 0')
    _ensure_column(connection, 'documents', 'chunk_count', 'INTEGER NOT NULL DEFAULT 0')
    _ensure_column(connection, 'documents', 'indexed_at', 'TEXT')
    _ensure_column(connection, 'documents', 'last_error', 'TEXT')
    _ensure_column(connection, 'documents', 'metadata_json', "TEXT NOT NULL DEFAULT '{}' ")
    _ensure_column(connection, 'documents', 'owner_subject_id', 'TEXT')
    _ensure_column(connection, 'documents', 'owner_tenant_id', 'TEXT')
    _ensure_column(connection, 'document_chunks', 'document_name', "TEXT NOT NULL DEFAULT ''")
    _ensure_column(connection, 'document_chunks', 'source_uri', "TEXT NOT NULL DEFAULT ''")
    _ensure_column(connection, 'chunk_embeddings', 'embedding_provider', "TEXT NOT NULL DEFAULT 'unknown'")
    _ensure_column(connection, 'request_logs', 'document_ids_json', "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(connection, 'request_logs', 'metadata_filters_json', "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(connection, 'request_logs', 'requested_top_k', 'INTEGER NOT NULL DEFAULT 5')
    _ensure_column(connection, 'request_logs', 'task_type_hint', 'TEXT')
    _ensure_column(connection, 'request_logs', 'llm_model', 'TEXT')
    _ensure_column(connection, 'request_logs', 'retrieval_stats_json', "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(connection, 'request_logs', 'structured_data_json', "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(connection, 'request_logs', 'citation_coverage', 'REAL NOT NULL DEFAULT 0')
    _ensure_column(connection, 'request_logs', 'evidence_strength', 'REAL NOT NULL DEFAULT 0')
    _ensure_column(connection, 'request_logs', 'citation_ids_valid', 'INTEGER NOT NULL DEFAULT 1')
    _ensure_column(connection, 'request_logs', 'answer_support_score', 'REAL NOT NULL DEFAULT 0')
    _ensure_column(connection, 'request_logs', 'insufficient_evidence', 'INTEGER NOT NULL DEFAULT 0')
    _ensure_column(connection, 'request_logs', 'vector_backend', 'TEXT')
    _ensure_column(connection, 'request_logs', 'embedding_backend', 'TEXT')
    _ensure_column(connection, 'request_logs', 'auth_subject_id', 'TEXT')
    _ensure_column(connection, 'request_logs', 'auth_tenant_id', 'TEXT')
    _ensure_column(connection, 'request_logs', 'auth_role', 'TEXT')
    _ensure_column(connection, 'request_logs', 'auth_type', 'TEXT')
    _ensure_column(connection, 'review_decisions', 'reviewer_subject_id', 'TEXT')
    _ensure_column(connection, 'review_decisions', 'reviewer_tenant_id', 'TEXT')
    _ensure_column(connection, 'review_decisions', 'reviewer_role', 'TEXT')
    _ensure_column(connection, 'review_decisions', 'reviewer_auth_type', 'TEXT')
    _ensure_column(connection, 'evaluation_runs', 'task_type_match_rate', 'REAL NOT NULL DEFAULT 0')
    _ensure_column(connection, 'evaluation_case_results', 'expected_task_type', 'TEXT')
    _ensure_column(connection, 'evaluation_case_results', 'task_type_match', 'INTEGER NOT NULL DEFAULT 1')
    _ensure_column(connection, 'evaluation_case_results', 'grounded', 'INTEGER NOT NULL DEFAULT 0')
    _ensure_column(connection, 'evaluation_case_results', 'citation_success', 'INTEGER NOT NULL DEFAULT 0')
    _ensure_column(connection, 'evaluation_case_results', 'structured_success', 'INTEGER NOT NULL DEFAULT 0')
    _ensure_column(connection, 'provider_usage_events', 'subject_key', 'TEXT')
    _ensure_column(connection, 'provider_usage_events', 'subject_id', 'TEXT')
    _ensure_column(connection, 'provider_usage_events', 'tenant_id', 'TEXT')
    _ensure_column(connection, 'provider_usage_events', 'auth_type', 'TEXT')
    connection.executescript(INDEX_SQL)


def initialize_database() -> None:
    connection = connect(ensure_schema=False)
    try:
        apply_schema(connection)
        connection.commit()
    finally:
        connection.close()
