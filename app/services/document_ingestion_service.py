from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.config import get_settings
from app.core.utils import generate_id, iso_timestamp, safe_filename
from app.db.sqlite import get_connection
from app.models.responses import ChunkPreview, DocumentDetailResponse, DocumentSummary
from app.models.schemas import DocumentRecord, DocumentStatus
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.parser_service import ParserService
from app.services.security_service import AuthPrincipal
from app.services.vector_store_service import VectorStoreService


class DocumentIngestionService:
    """Handles upload persistence, parsing, chunking, and chunk metadata storage."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.parser_service = ParserService()
        self.chunking_service = ChunkingService()
        self.embedding_service = EmbeddingService()
        self.vector_store_service = VectorStoreService()

    def save_upload(
        self,
        filename: str,
        content: bytes,
        mime_type: str,
        principal: AuthPrincipal | None = None,
    ) -> DocumentRecord:
        self.purge_expired_documents()
        document_id = generate_id('doc')
        stored_name = f'{document_id}_{safe_filename(filename)}'
        storage_path = self.settings.uploads_dir / stored_name
        storage_path.write_bytes(content)
        now = iso_timestamp()
        owner_subject_id = principal.subject_id if principal is not None else None
        owner_tenant_id = principal.tenant_id if principal is not None else None

        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    filename,
                    storage_path,
                    mime_type,
                    status,
                    created_at,
                    updated_at,
                    page_count,
                    chunk_count,
                    indexed_at,
                    last_error,
                    metadata_json,
                    owner_subject_id,
                    owner_tenant_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    filename,
                    str(storage_path),
                    mime_type,
                    DocumentStatus.UPLOADED.value,
                    now,
                    now,
                    0,
                    0,
                    None,
                    None,
                    json.dumps({}),
                    owner_subject_id,
                    owner_tenant_id,
                ),
            )

        return DocumentRecord(
            document_id=document_id,
            filename=filename,
            storage_path=str(storage_path),
            mime_type=mime_type,
            status=DocumentStatus.UPLOADED,
            owner_subject_id=owner_subject_id,
            owner_tenant_id=owner_tenant_id,
        )

    def list_documents(self, principal: AuthPrincipal | None = None) -> list[DocumentSummary]:
        query = """
            SELECT document_id, filename, status, created_at, page_count, chunk_count, indexed_at, last_error
            FROM documents
        """
        parameters: list[str] = []
        query, parameters = self._apply_owner_filter(query, parameters, principal)
        query += ' ORDER BY created_at DESC'

        with get_connection() as connection:
            rows = connection.execute(query, parameters).fetchall()

        return [
            DocumentSummary(
                document_id=row['document_id'],
                filename=row['filename'],
                status=DocumentStatus(row['status']),
                created_at=row['created_at'],
                page_count=int(row['page_count'] or 0),
                chunk_count=int(row['chunk_count'] or 0),
                indexed_at=row['indexed_at'],
                last_error=row['last_error'],
            )
            for row in rows
        ]

    def list_accessible_document_ids(
        self,
        principal: AuthPrincipal | None = None,
        *,
        require_indexed: bool = False,
    ) -> list[str]:
        query = 'SELECT document_id FROM documents'
        parameters: list[str] = []
        query, parameters = self._apply_owner_filter(query, parameters, principal)
        if require_indexed:
            query += ' AND status = ?' if ' WHERE ' in query else ' WHERE status = ?'
            parameters.append(DocumentStatus.INDEXED.value)
        query += ' ORDER BY created_at DESC'

        with get_connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [row['document_id'] for row in rows]

    def get_document(
        self,
        document_id: str,
        principal: AuthPrincipal | None = None,
    ) -> DocumentDetailResponse | None:
        document = self._get_document_record(document_id, principal=principal)
        if document is None:
            return None

        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT document_id, filename, storage_path, mime_type, status, created_at, page_count, chunk_count, indexed_at, last_error
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

        if row is None:
            return None

        return DocumentDetailResponse(
            document_id=row['document_id'],
            filename=row['filename'],
            mime_type=row['mime_type'],
            status=DocumentStatus(row['status']),
            created_at=row['created_at'],
            page_count=int(row['page_count'] or 0),
            chunk_count=int(row['chunk_count'] or 0),
            indexed_at=row['indexed_at'],
            last_error=row['last_error'],
            chunks=self.list_chunks(document_id),
        )

    def index_document(
        self,
        document_id: str,
        principal: AuthPrincipal | None = None,
        force_reindex: bool = False,
    ) -> DocumentRecord | None:
        self.purge_expired_documents()
        document = self._get_document_record(document_id, principal=principal)
        if document is None:
            return None

        if document.status == DocumentStatus.INDEXED and document.chunk_count > 0 and not force_reindex:
            return document

        existing_chunk_ids = self._list_chunk_ids(document_id)
        self._update_document_status(document_id, DocumentStatus.QUEUED)
        try:
            parsed_document = self.parser_service.parse(document)
            if not parsed_document.text.strip():
                raise ValueError('No extractable text was found in the document.')

            chunks = self.chunking_service.chunk_document(parsed_document)
            if not chunks:
                raise ValueError('No chunks were produced from the parsed document.')

            if existing_chunk_ids:
                self.vector_store_service.delete_chunks(existing_chunk_ids)

            now = iso_timestamp()
            self._replace_document_chunks(document_id=document_id, chunks=chunks, created_at=now)
            embeddings = self.embedding_service.embed_texts([chunk.text for chunk in chunks])
            vector_backend = self.vector_store_service.index_chunks(
                chunks=chunks,
                embeddings=embeddings,
                embedding_provider=self.embedding_service.backend,
            )
            metadata_payload = {
                **parsed_document.metadata,
                'embedding_backend': self.embedding_service.backend,
                'vector_backend': vector_backend,
            }
            self._mark_indexed(
                document_id=document_id,
                page_count=parsed_document.page_count,
                chunk_count=len(chunks),
                indexed_at=now,
                metadata_payload=metadata_payload,
            )

            parsed_document.status = DocumentStatus.INDEXED
            parsed_document.chunk_count = len(chunks)
            parsed_document.indexed_at = now
            parsed_document.last_error = None
            parsed_document.metadata = metadata_payload
            parsed_document.owner_subject_id = document.owner_subject_id
            parsed_document.owner_tenant_id = document.owner_tenant_id
            return parsed_document
        except Exception:
            self.vector_store_service.delete_document(document_id)
            self._clear_document_chunks(document_id)
            self._mark_failed(document_id=document_id, error_message='Indexing failed unexpectedly.')
            raise

    def cleanup_invalid_documents(self) -> list[str]:
        uploads_root = self.settings.uploads_dir.resolve()
        with get_connection() as connection:
            rows = connection.execute('SELECT document_id, storage_path FROM documents').fetchall()

        invalid_document_ids: list[str] = []
        for row in rows:
            storage_path = Path(row['storage_path'])
            try:
                resolved_path = storage_path.resolve()
            except FileNotFoundError:
                resolved_path = storage_path

            is_within_uploads = uploads_root == resolved_path.parent or uploads_root in resolved_path.parents
            if not storage_path.exists() or not is_within_uploads:
                invalid_document_ids.append(row['document_id'])

        for document_id in invalid_document_ids:
            self.delete_document(document_id)
        return invalid_document_ids

    def cleanup_orphaned_uploads(self) -> list[str]:
        with get_connection() as connection:
            rows = connection.execute('SELECT storage_path FROM documents').fetchall()
        valid_storage_paths = {str(Path(row['storage_path']).resolve()) for row in rows}

        removed_files: list[str] = []
        for path in self.settings.uploads_dir.iterdir():
            if not path.is_file() or path.name.startswith('.'):
                continue
            if str(path.resolve()) not in valid_storage_paths:
                path.unlink(missing_ok=True)
                removed_files.append(path.name)
        return removed_files

    def purge_expired_documents(self) -> list[str]:
        if self.settings.document_retention_days is None:
            return []

        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.settings.document_retention_days)).isoformat()
        with get_connection() as connection:
            rows = connection.execute(
                'SELECT document_id FROM documents WHERE created_at < ? ORDER BY created_at ASC',
                (cutoff,),
            ).fetchall()

        purged_document_ids: list[str] = []
        for row in rows:
            if self.delete_document(row['document_id']) is not None:
                purged_document_ids.append(row['document_id'])
        return purged_document_ids

    def cleanup_orphaned_assets(self) -> dict[str, int]:
        purged_documents = self.purge_expired_documents()
        removed_documents = self.cleanup_invalid_documents()
        removed_files = self.cleanup_orphaned_uploads()
        cleanup_summary = self.vector_store_service.prune_orphaned_records()
        return {
            **cleanup_summary,
            'purged_documents': len(purged_documents),
            'removed_documents': len(removed_documents),
            'removed_upload_files': len(removed_files),
        }

    def reconcile_storage(self) -> dict[str, int]:
        return self.cleanup_orphaned_assets()

    def delete_document(
        self,
        document_id: str,
        principal: AuthPrincipal | None = None,
    ) -> DocumentRecord | None:
        document = self._get_document_record(document_id, principal=principal)
        if document is None:
            return None

        self.vector_store_service.delete_document(document_id)
        with get_connection() as connection:
            connection.execute('DELETE FROM document_chunks WHERE document_id = ?', (document_id,))
            connection.execute('DELETE FROM documents WHERE document_id = ?', (document_id,))
        path = Path(document.storage_path)
        if path.exists():
            path.unlink(missing_ok=True)
        return document

    def list_chunks(self, document_id: str) -> list[ChunkPreview]:
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, chunk_index, page_number, section, token_count, text
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
                """,
                (document_id,),
            ).fetchall()

        return [
            ChunkPreview(
                chunk_id=row['chunk_id'],
                chunk_index=int(row['chunk_index']),
                page_number=row['page_number'],
                section=row['section'],
                token_count=int(row['token_count']),
                preview=(row['text'][:220] + '...') if len(row['text']) > 220 else row['text'],
            )
            for row in rows
        ]

    def _get_document_record(
        self,
        document_id: str,
        principal: AuthPrincipal | None = None,
    ) -> DocumentRecord | None:
        query = """
            SELECT document_id, filename, storage_path, mime_type, status, page_count, chunk_count, indexed_at, last_error, metadata_json, owner_subject_id, owner_tenant_id
            FROM documents
            WHERE document_id = ?
        """
        parameters: list[str] = [document_id]
        query, parameters = self._apply_owner_filter(query, parameters, principal, needs_where=False)

        with get_connection() as connection:
            row = connection.execute(query, parameters).fetchone()

        if row is None:
            return None

        metadata = json.loads(row['metadata_json'] or '{}')
        return DocumentRecord(
            document_id=row['document_id'],
            filename=row['filename'],
            storage_path=row['storage_path'],
            mime_type=row['mime_type'],
            status=DocumentStatus(row['status']),
            page_count=int(row['page_count'] or 0),
            chunk_count=int(row['chunk_count'] or 0),
            indexed_at=row['indexed_at'],
            last_error=row['last_error'],
            metadata=metadata,
            owner_subject_id=row['owner_subject_id'],
            owner_tenant_id=row['owner_tenant_id'],
        )

    def _list_chunk_ids(self, document_id: str) -> list[str]:
        with get_connection() as connection:
            rows = connection.execute(
                'SELECT chunk_id FROM document_chunks WHERE document_id = ? ORDER BY chunk_index ASC',
                (document_id,),
            ).fetchall()
        return [row['chunk_id'] for row in rows]

    def _replace_document_chunks(self, document_id: str, chunks: list, created_at: str) -> None:
        with get_connection() as connection:
            connection.execute('DELETE FROM document_chunks WHERE document_id = ?', (document_id,))
            for chunk in chunks:
                connection.execute(
                    """
                    INSERT INTO document_chunks (
                        chunk_id,
                        document_id,
                        document_name,
                        chunk_index,
                        page_number,
                        section,
                        text,
                        source_uri,
                        token_count,
                        metadata_json,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.document_name,
                        chunk.chunk_index,
                        chunk.page_number,
                        chunk.section,
                        chunk.text,
                        chunk.source_uri,
                        chunk.token_count,
                        json.dumps(chunk.metadata),
                        created_at,
                    ),
                )

    def _mark_indexed(
        self,
        document_id: str,
        page_count: int,
        chunk_count: int,
        indexed_at: str,
        metadata_payload: dict[str, object],
    ) -> None:
        with get_connection() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = ?, page_count = ?, chunk_count = ?, indexed_at = ?, last_error = ?, metadata_json = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (
                    DocumentStatus.INDEXED.value,
                    page_count,
                    chunk_count,
                    indexed_at,
                    None,
                    json.dumps(metadata_payload),
                    indexed_at,
                    document_id,
                ),
            )

    def _clear_document_chunks(self, document_id: str) -> None:
        with get_connection() as connection:
            connection.execute('DELETE FROM document_chunks WHERE document_id = ?', (document_id,))

    def _update_document_status(self, document_id: str, status: DocumentStatus) -> None:
        with get_connection() as connection:
            connection.execute(
                'UPDATE documents SET status = ?, updated_at = ? WHERE document_id = ?',
                (status.value, iso_timestamp(), document_id),
            )

    def _mark_failed(self, document_id: str, error_message: str) -> None:
        with get_connection() as connection:
            connection.execute(
                'UPDATE documents SET status = ?, last_error = ?, page_count = 0, chunk_count = 0, indexed_at = NULL, updated_at = ? WHERE document_id = ?',
                (DocumentStatus.FAILED.value, error_message, iso_timestamp(), document_id),
            )

    def _apply_owner_filter(
        self,
        query: str,
        parameters: list[str],
        principal: AuthPrincipal | None,
        *,
        needs_where: bool = True,
    ) -> tuple[str, list[str]]:
        if principal is None or principal.is_admin:
            return query, parameters

        clause = ' WHERE ' if needs_where else ' AND '
        query += clause + 'owner_subject_id = ? AND COALESCE(owner_tenant_id, ?) = ?'
        tenant_id = principal.tenant_id or ''
        parameters.extend([principal.subject_id, '', tenant_id])
        return query, parameters
