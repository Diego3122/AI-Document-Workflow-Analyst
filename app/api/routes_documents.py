from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Path as FastApiPath, UploadFile, status

from app.api.dependencies import require_access
from app.models.requests import DOCUMENT_ID_PATTERN, IndexDocumentRequest
from app.models.responses import (
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentSummary,
    DocumentUploadResponse,
    IndexDocumentResponse,
)
from app.services.document_ingestion_service import DocumentIngestionService
from app.services.provider_guard_service import ProviderUsageLimitExceeded
from app.services.security_service import AuthPrincipal


router = APIRouter(prefix='/documents', tags=['documents'])
DocumentIdPath = Annotated[str, FastApiPath(pattern=DOCUMENT_ID_PATTERN.pattern)]
ALLOWED_UPLOAD_SUFFIXES = {'.pdf', '.txt'}


def _validate_filename(filename: str) -> str:
    cleaned = filename.strip()
    if not cleaned:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='A filename is required.')
    if len(cleaned) > 255:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Filenames must be 255 characters or fewer.')
    if cleaned != Path(cleaned).name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Filenames cannot include directory paths.')
    if any(ord(character) < 32 for character in cleaned):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Filename contains unsupported control characters.')
    suffix = Path(cleaned).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Only PDF and TXT files are supported.')
    return cleaned


def _validate_upload_content(filename: str, content: bytes) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix == '.pdf' and not content.startswith(b'%PDF-'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Uploaded PDF files must contain a valid PDF header.',
        )
    if suffix == '.txt':
        try:
            content.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='TXT uploads must be valid UTF-8 text.',
            ) from exc


@router.post(
    '/upload',
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    principal: AuthPrincipal | None = Depends(require_access('analyst', 'admin', required_scopes=('documents.write',))),
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    from app.core.config import get_settings

    settings = get_settings()
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='A filename is required.')

    filename = _validate_filename(file.filename)
    content = await file.read(settings.max_upload_size_bytes + 1)
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Uploaded file is empty.')
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f'Uploaded file exceeds the limit of {settings.max_upload_size_bytes} bytes.',
        )
    _validate_upload_content(filename, content)

    service = DocumentIngestionService()
    document = service.save_upload(
        filename=filename,
        content=content,
        mime_type=file.content_type or 'application/octet-stream',
        principal=principal,
    )
    return DocumentUploadResponse(
        document_id=document.document_id,
        filename=document.filename,
        status=document.status,
    )


@router.get('', response_model=list[DocumentSummary])
def list_documents(
    principal: AuthPrincipal | None = Depends(require_access('analyst', 'admin', required_scopes=('documents.read',))),
) -> list[DocumentSummary]:
    return DocumentIngestionService().list_documents(principal=principal)


@router.get('/{document_id}', response_model=DocumentDetailResponse)
def get_document(
    document_id: DocumentIdPath,
    principal: AuthPrincipal | None = Depends(require_access('analyst', 'admin', required_scopes=('documents.read',))),
) -> DocumentDetailResponse:
    document = DocumentIngestionService().get_document(document_id, principal=principal)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Document not found.')
    return document


@router.delete('/{document_id}', response_model=DocumentDeleteResponse)
def delete_document(
    document_id: DocumentIdPath,
    principal: AuthPrincipal | None = Depends(require_access('analyst', 'admin', required_scopes=('documents.write',))),
) -> DocumentDeleteResponse:
    document = DocumentIngestionService().delete_document(document_id, principal=principal)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Document not found.')
    return DocumentDeleteResponse(document_id=document.document_id, status='deleted')


@router.post('/index', response_model=IndexDocumentResponse)
def index_document(
    request: IndexDocumentRequest,
    principal: AuthPrincipal | None = Depends(require_access('analyst', 'admin', required_scopes=('documents.write',))),
) -> IndexDocumentResponse:
    service = DocumentIngestionService()
    try:
        document = service.index_document(
            document_id=request.document_id,
            principal=principal,
            force_reindex=request.force_reindex,
        )
    except ProviderUsageLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f'Indexing failed: {exc}',
        ) from exc

    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Document not found.')

    warnings: list[str] = []
    if document.chunk_count == 0:
        warnings.append('No chunks were produced during indexing.')

    return IndexDocumentResponse(
        document_id=document.document_id,
        status=document.status,
        page_count=document.page_count,
        chunk_count=document.chunk_count,
        message='Document parsing and chunk persistence completed.',
        warnings=warnings,
    )
