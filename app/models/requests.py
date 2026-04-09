from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.schemas import ReviewAction, TaskType


DOCUMENT_ID_PATTERN = re.compile(r'^doc_[a-z0-9]{12}$')
REQUEST_ID_PATTERN = re.compile(r'^req_[a-z0-9]{12}$')
MAX_DOCUMENT_IDS = 10
MAX_QUERY_LENGTH = 2000
MAX_TEXT_FIELD_LENGTH = 5000
MAX_NOTE_LENGTH = 2000
MAX_METADATA_FILTERS = 10
MAX_METADATA_FILTER_LENGTH = 100
MAX_DATASET_PATH_LENGTH = 260
WHITESPACE_PATTERN = re.compile(r'\s+')


def sanitize_text(value: str | None, *, max_length: int) -> str | None:
    if value is None:
        return None
    cleaned = value.replace(chr(0), ' ')
    cleaned = WHITESPACE_PATTERN.sub(' ', cleaned).strip()
    if not cleaned:
        return None
    if len(cleaned) > max_length:
        raise ValueError(f'Value exceeds the maximum length of {max_length} characters.')
    return cleaned


class IndexDocumentRequest(BaseModel):
    document_id: str
    force_reindex: bool = False

    @field_validator('document_id')
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        cleaned = sanitize_text(value, max_length=32)
        if cleaned is None or not DOCUMENT_ID_PATTERN.fullmatch(cleaned):
            raise ValueError('Invalid document_id format.')
        return cleaned


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=MAX_QUERY_LENGTH)
    document_ids: list[str] = Field(default_factory=list)
    task_type_hint: TaskType | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    metadata_filters: dict[str, str] = Field(default_factory=dict)

    @field_validator('query')
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = sanitize_text(value, max_length=MAX_QUERY_LENGTH)
        if cleaned is None or len(cleaned) < 3:
            raise ValueError('Query must be at least 3 characters long.')
        return cleaned

    @field_validator('document_ids')
    @classmethod
    def validate_document_ids(cls, values: list[str]) -> list[str]:
        if len(values) > MAX_DOCUMENT_IDS:
            raise ValueError(f'No more than {MAX_DOCUMENT_IDS} document IDs may be provided.')

        cleaned_values: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = sanitize_text(value, max_length=32)
            if cleaned is None or not DOCUMENT_ID_PATTERN.fullmatch(cleaned):
                raise ValueError('Invalid document_id format.')
            if cleaned not in seen:
                cleaned_values.append(cleaned)
                seen.add(cleaned)
        return cleaned_values

    @field_validator('metadata_filters')
    @classmethod
    def validate_metadata_filters(cls, filters: dict[str, str]) -> dict[str, str]:
        if len(filters) > MAX_METADATA_FILTERS:
            raise ValueError(f'No more than {MAX_METADATA_FILTERS} metadata filters may be provided.')

        cleaned_filters: dict[str, str] = {}
        for raw_key, raw_value in filters.items():
            key = sanitize_text(raw_key, max_length=MAX_METADATA_FILTER_LENGTH)
            value = sanitize_text(raw_value, max_length=MAX_METADATA_FILTER_LENGTH)
            if key is None or value is None:
                raise ValueError('Metadata filter keys and values must be non-empty.')
            cleaned_filters[key] = value
        return cleaned_filters


class ReviewDecisionRequest(BaseModel):
    request_id: str
    action: ReviewAction
    edited_answer: str | None = None
    reviewer_notes: str | None = None

    @field_validator('request_id')
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        cleaned = sanitize_text(value, max_length=32)
        if cleaned is None or not REQUEST_ID_PATTERN.fullmatch(cleaned):
            raise ValueError('Invalid request_id format.')
        return cleaned

    @field_validator('edited_answer')
    @classmethod
    def validate_edited_answer(cls, value: str | None) -> str | None:
        return sanitize_text(value, max_length=MAX_TEXT_FIELD_LENGTH)

    @field_validator('reviewer_notes')
    @classmethod
    def validate_reviewer_notes(cls, value: str | None) -> str | None:
        return sanitize_text(value, max_length=MAX_NOTE_LENGTH)

    @model_validator(mode='after')
    def validate_edit_payload(self) -> ReviewDecisionRequest:
        if self.action == ReviewAction.EDIT and not self.edited_answer:
            raise ValueError('Edited answer text is required for edit decisions.')
        return self


class EvalRunRequest(BaseModel):
    dataset_path: str | None = None
    max_cases: int | None = Field(default=None, ge=1, le=100)

    @field_validator('dataset_path')
    @classmethod
    def validate_dataset_path(cls, value: str | None) -> str | None:
        return sanitize_text(value, max_length=MAX_DATASET_PATH_LENGTH)
