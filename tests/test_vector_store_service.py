from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import get_settings
from app.core.utils import iso_timestamp
from app.db.init_db import initialize_database
from app.db.sqlite import get_connection
from app.models.schemas import ChunkRecord
from app.services.document_ingestion_service import DocumentIngestionService
from app.services.vector_store_service import VectorStoreService


pytest.importorskip('chromadb')


EMBEDDING = [0.1] * 128



def configure_chroma(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv('VECTOR_BACKEND', 'auto')
    monkeypatch.setenv('CHROMA_DIR', str(tmp_path / 'chroma'))
    monkeypatch.setenv('UPLOADS_DIR', str(tmp_path / 'uploads'))
    monkeypatch.setenv('LOGS_DIR', str(tmp_path / 'logs'))
    monkeypatch.setenv('SQLITE_PATH', str(tmp_path / 'logs' / 'app.db'))
    get_settings.cache_clear()
    settings = get_settings()
    settings.ensure_directories()
    initialize_database()



def test_chroma_query_ignores_and_prunes_orphaned_chunks(monkeypatch, tmp_path: Path) -> None:
    configure_chroma(monkeypatch, tmp_path)
    service = VectorStoreService()
    chunk = ChunkRecord(
        chunk_id='chunk_orphaned',
        document_id='doc_orphaned',
        document_name='orphaned.txt',
        chunk_index=0,
        page_number=1,
        section='PAYMENT TERMS',
        text='PAYMENT TERMS Net 30 days.',
        source_uri=str(tmp_path / 'uploads' / 'doc_orphaned_orphaned.txt'),
        token_count=4,
        metadata={},
    )

    backend = service.index_chunks([chunk], [EMBEDDING], embedding_provider='test')
    assert backend == 'chroma'

    results = service.query(query_embedding=EMBEDDING, top_k=3, document_ids=[])
    assert results == []

    collection = service._get_chroma_collection()
    snapshot = collection.get()
    assert 'chunk_orphaned' not in snapshot.get('ids', [])

    get_settings.cache_clear()



def test_force_reindex_replaces_old_chroma_chunks(monkeypatch, tmp_path: Path) -> None:
    configure_chroma(monkeypatch, tmp_path)
    service = DocumentIngestionService()
    document = service.save_upload(
        filename='contract.txt',
        content=b'PAYMENT TERMS\nInvoices are due within 30 days of receipt. Late fees apply after 45 days.',
        mime_type='text/plain',
    )

    first_index = service.index_document(document.document_id)
    assert first_index is not None
    second_index = service.index_document(document.document_id, force_reindex=True)
    assert second_index is not None

    with get_connection() as connection:
        current_chunk_ids = {
            row['chunk_id']
            for row in connection.execute(
                'SELECT chunk_id FROM document_chunks WHERE document_id = ?',
                (document.document_id,),
            ).fetchall()
        }

    collection = VectorStoreService()._get_chroma_collection()
    snapshot = collection.get(include=['metadatas'])
    chroma_chunk_ids = {
        chunk_id
        for chunk_id, metadata in zip(snapshot.get('ids', []), snapshot.get('metadatas', []), strict=False)
        if (metadata or {}).get('document_id') == document.document_id
    }

    assert chroma_chunk_ids == current_chunk_ids
    assert len(chroma_chunk_ids) == second_index.chunk_count

    get_settings.cache_clear()



def test_reconcile_storage_removes_orphaned_uploads_and_chroma_chunks(monkeypatch, tmp_path: Path) -> None:
    configure_chroma(monkeypatch, tmp_path)
    settings = get_settings()
    orphan_file = settings.uploads_dir / 'doc_orphaned_contract.txt'
    orphan_file.write_text('PAYMENT TERMS Net 30 days.', encoding='utf-8')

    vector_store = VectorStoreService()
    orphan_chunk = ChunkRecord(
        chunk_id='chunk_old',
        document_id='doc_old',
        document_name='contract.txt',
        chunk_index=0,
        page_number=1,
        section='PAYMENT TERMS',
        text='PAYMENT TERMS Net 30 days.',
        source_uri=str(orphan_file),
        token_count=4,
        metadata={},
    )
    vector_store.index_chunks([orphan_chunk], [EMBEDDING], embedding_provider='test')

    summary = DocumentIngestionService().reconcile_storage()

    assert summary['removed_upload_files'] == 1
    assert summary['removed_chroma_chunks'] == 1
    assert not orphan_file.exists()

    get_settings.cache_clear()


def test_cleanup_orphaned_assets_removes_documents_outside_live_uploads(monkeypatch) -> None:
    service = DocumentIngestionService()
    outside_dir = Path('tests/.pytest_runtime/debug_case/uploads')
    outside_dir.mkdir(parents=True, exist_ok=True)
    outside_path = outside_dir / 'doc_debug_contract.txt'
    outside_path.write_text('PAYMENT TERMS\nInvoices are due within 30 days.', encoding='utf-8')

    now = iso_timestamp()
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
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                'doc_debug',
                'contract.txt',
                str(outside_path.resolve()),
                'text/plain',
                'indexed',
                now,
                now,
                1,
                0,
                now,
                None,
                '{}',
            ),
        )

    monkeypatch.setattr(service.vector_store_service, 'prune_orphaned_records', lambda: {'removed_sqlite_embeddings': 0, 'removed_chroma_chunks': 0})
    summary = service.cleanup_orphaned_assets()

    with get_connection() as connection:
        row = connection.execute("SELECT document_id FROM documents WHERE document_id = 'doc_debug'").fetchone()

    assert summary['removed_documents'] == 1
    assert row is None
    assert outside_path.exists() is False

