from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.models.schemas import Citation, DocumentStatus, ReviewAction, TaskType


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str


class AuthStatusResponse(BaseModel):
    subject_id: str
    tenant_id: str | None = None
    auth_type: str
    role: str
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    display_name: str | None = None
    can_upload_and_query: bool = False
    can_review: bool = False
    can_view_metrics: bool = False


class ChunkPreview(BaseModel):
    chunk_id: str
    chunk_index: int
    page_number: int | None = None
    section: str | None = None
    token_count: int
    preview: str


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    created_at: str
    page_count: int = 0
    chunk_count: int = 0
    indexed_at: str | None = None
    last_error: str | None = None


class DocumentDetailResponse(DocumentSummary):
    mime_type: str
    chunks: list[ChunkPreview] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus


class DocumentDeleteResponse(BaseModel):
    document_id: str
    status: str


class IndexDocumentResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    page_count: int
    chunk_count: int
    message: str
    warnings: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    request_id: str
    task_type: TaskType
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float
    needs_human_review: bool
    structured_data: dict[str, Any] = Field(default_factory=dict)
    validation_notes: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieval_stats: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float


class ReviewDecisionResponse(BaseModel):
    review_id: str
    request_id: str
    action: ReviewAction
    status: str


class EvaluationCaseResponse(BaseModel):
    case_id: str
    passed: bool
    expected_task_type: str | None = None
    task_type: str
    task_type_match: bool = True
    confidence: float
    latency_ms: float
    expected_review: bool | None = None
    actual_review: bool
    expected_citation_count: int = 0
    actual_citation_count: int = 0
    grounded: bool
    citation_success: bool
    structured_success: bool
    notes: list[str] = Field(default_factory=list)


class EvalRunResponse(BaseModel):
    evaluation_id: str
    dataset_path: str
    total_cases: int
    passed_cases: int
    pass_rate: float
    task_type_match_rate: float
    average_latency_ms: float
    average_confidence: float
    review_rate: float
    grounded_rate: float
    citation_success_rate: float
    structured_success_rate: float
    created_at: str
    failed_case_ids: list[str] = Field(default_factory=list)
    cases: list[EvaluationCaseResponse] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    total_requests: int
    workflow_error_count: int
    average_latency_ms: float
    average_confidence: float
    average_answer_support_score: float
    review_rate: float
    grounded_rate: float
    insufficient_evidence_rate: float
    schema_failure_count: int
    task_type_breakdown: dict[str, int] = Field(default_factory=dict)
    vector_backend_breakdown: dict[str, int] = Field(default_factory=dict)
    last_request_at: str | None = None
    latest_evaluation: EvalRunResponse | None = None
