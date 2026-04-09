from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    QUESTION_ANSWERING = 'question_answering'
    STRUCTURED_EXTRACTION = 'structured_extraction'
    SUMMARIZATION = 'summarization'
    RISK_FLAGGING = 'risk_flagging'
    UNSUPPORTED = 'unsupported'


class DocumentStatus(str, Enum):
    UPLOADED = 'uploaded'
    QUEUED = 'queued'
    INDEXED = 'indexed'
    FAILED = 'failed'


class ReviewAction(str, Enum):
    APPROVE = 'approve'
    EDIT = 'edit'
    REJECT = 'reject'


class AgentRoute(BaseModel):
    task_type: TaskType
    reason: str


class Citation(BaseModel):
    document_id: str
    document_name: str
    page_number: int | None = None
    chunk_id: str
    excerpt: str | None = None


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    storage_path: str
    mime_type: str
    status: DocumentStatus = DocumentStatus.UPLOADED
    text: str = ''
    page_count: int = 0
    chunk_count: int = 0
    page_texts: list[str] = Field(default_factory=list)
    indexed_at: str | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    owner_subject_id: str | None = None
    owner_tenant_id: str | None = None


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    page_number: int | None = None
    section: str | None = None
    text: str
    source_uri: str
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None = None
    section: str | None = None
    score: float | None = None
    keyword_score: float | None = None
    vector_score: float | None = None
    text: str


class EvidencePack(BaseModel):
    query: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    retrieval_stats: dict[str, Any] = Field(default_factory=dict)


class DraftAnswer(BaseModel):
    task_type: TaskType
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    structured_data: dict[str, Any] = Field(default_factory=dict)
    validation_notes: list[str] = Field(default_factory=list)


class StructuredSynthesis(BaseModel):
    answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    insufficiency_reason: str | None = None
    answer_style: str = 'grounded_answer'
    structured_data: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    schema_valid: bool
    citation_coverage: float
    evidence_strength: float
    grounded: bool
    citation_ids_valid: bool = True
    answer_support_score: float = 0.0
    insufficient_evidence: bool = False
    contradictions_detected: bool = False
    notes: list[str] = Field(default_factory=list)


class ValidatedResult(BaseModel):
    draft: DraftAnswer
    report: ValidationReport
    needs_human_review: bool
