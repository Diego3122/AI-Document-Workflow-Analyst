from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import require_access
from app.core.utils import generate_id
from app.models.requests import QueryRequest, ReviewDecisionRequest
from app.models.responses import QueryResponse, ReviewDecisionResponse
from app.services.document_ingestion_service import DocumentIngestionService
from app.services.logging_service import LoggingService
from app.services.provider_guard_service import ProviderUsageLimitExceeded
from app.services.query_workflow_service import QueryWorkflowService
from app.services.security_service import AuthPrincipal


router = APIRouter(tags=['query'])


@router.post('/query', response_model=QueryResponse)
def run_query(
    request: QueryRequest,
    principal: AuthPrincipal | None = Depends(require_access('analyst', 'admin', required_scopes=('query.run',))),
) -> QueryResponse:
    request_id = generate_id('req')
    logging_service = LoggingService()
    document_service = DocumentIngestionService()
    accessible_document_ids = set(document_service.list_accessible_document_ids(principal=principal, require_indexed=True))

    if request.document_ids:
        requested_document_ids = list(dict.fromkeys(request.document_ids))
        if any(document_id not in accessible_document_ids for document_id in requested_document_ids):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='One or more requested documents were not found.')
    else:
        requested_document_ids = list(accessible_document_ids)

    if not requested_document_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='No accessible indexed documents are available for this query.',
        )

    effective_request = request.model_copy(update={'document_ids': requested_document_ids})

    try:
        response, report = QueryWorkflowService().execute(request=effective_request, request_id=request_id)
        logging_service.log_request(response=response, report=report, request=effective_request, principal=principal)
        return response
    except ProviderUsageLimitExceeded as exc:
        logging_service.log_failure(
            request_id=request_id,
            query=request.query,
            latency_ms=0.0,
            error_message=str(exc),
            principal=principal,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover
        logging_service.log_failure(
            request_id=request_id,
            query=request.query,
            latency_ms=0.0,
            error_message=str(exc),
            principal=principal,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='The query workflow failed.',
        ) from exc


@router.post('/query/review', response_model=ReviewDecisionResponse)
def submit_review_decision(
    request: ReviewDecisionRequest,
    principal: AuthPrincipal | None = Depends(require_access('reviewer', 'admin', required_scopes=('query.review',))),
) -> ReviewDecisionResponse:
    if request.action.value == 'edit' and not request.edited_answer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Edited answer text is required for edit decisions.',
        )

    review_id = LoggingService().create_review_id()
    LoggingService().log_review_decision(review_id=review_id, decision=request, principal=principal)
    return ReviewDecisionResponse(
        review_id=review_id,
        request_id=request.request_id,
        action=request.action,
        status='recorded',
    )
