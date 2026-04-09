from __future__ import annotations

import json
from pathlib import Path

from app.core.config import get_settings
from app.core.utils import iso_timestamp
from app.db.sqlite import get_connection
from app.services.document_ingestion_service import DocumentIngestionService
from app.services.vector_store_service import VectorStoreService



def test_reconcile_storage_removes_untracked_upload_file() -> None:
    service = DocumentIngestionService()
    tracked_document = service.save_upload(
        filename='kept.txt',
        content=b'PAYMENT TERMS\nInvoices are due within 30 days.',
        mime_type='text/plain',
    )
    orphan_path = service.settings.uploads_dir / 'orphan.txt'
    orphan_path.write_text('orphaned file', encoding='utf-8')

    summary = service.reconcile_storage()

    assert summary['removed_upload_files'] == 1
    assert not orphan_path.exists()
    assert Path(tracked_document.storage_path).exists()



def test_force_reindex_removes_old_chunk_embeddings() -> None:
    service = DocumentIngestionService()
    document = service.save_upload(
        filename='contract.txt',
        content=b'PAYMENT TERMS\nInvoices are due within 30 days.',
        mime_type='text/plain',
    )
    service.index_document(document.document_id)

    with get_connection() as connection:
        old_chunk_ids = {
            row['chunk_id']
            for row in connection.execute(
                'SELECT chunk_id FROM chunk_embeddings WHERE document_id = ?',
                (document.document_id,),
            ).fetchall()
        }

    Path(document.storage_path).write_text(
        'TERMINATION\nEither party may terminate with 15 days notice.',
        encoding='utf-8',
    )
    service.index_document(document.document_id, force_reindex=True)

    with get_connection() as connection:
        new_chunk_ids = {
            row['chunk_id']
            for row in connection.execute(
                'SELECT chunk_id FROM chunk_embeddings WHERE document_id = ?',
                (document.document_id,),
            ).fetchall()
        }

    assert old_chunk_ids
    assert new_chunk_ids
    assert old_chunk_ids.isdisjoint(new_chunk_ids)



def test_vector_store_reconcile_orphans_removes_sqlite_embeddings_without_live_docs_or_chunks() -> None:
    with get_connection() as connection:
        connection.execute('PRAGMA foreign_keys = OFF')
        connection.execute(
            """
            INSERT INTO chunk_embeddings (chunk_id, document_id, embedding_json, embedding_provider, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ('chunk_orphaned', 'doc_orphaned', json.dumps([0.1, 0.2]), 'local_hash', iso_timestamp()),
        )
        connection.execute('PRAGMA foreign_keys = ON')

    summary = VectorStoreService().reconcile_orphans()

    assert summary['removed_sqlite_embeddings'] == 1

    with get_connection() as connection:
        remaining = connection.execute(
            'SELECT COUNT(*) AS total FROM chunk_embeddings WHERE chunk_id = ?',
            ('chunk_orphaned',),
        ).fetchone()

    assert remaining['total'] == 0


def test_cleanup_orphaned_assets_purges_documents_past_retention_limit(monkeypatch) -> None:
    monkeypatch.setenv('DOCUMENT_RETENTION_DAYS', '30')
    get_settings.cache_clear()

    service = DocumentIngestionService()
    document = service.save_upload(
        filename='retained.txt',
        content=b'PAYMENT TERMS\nInvoices are due within 30 days.',
        mime_type='text/plain',
    )
    service.index_document(document.document_id)

    expired_timestamp = '2020-01-01T00:00:00+00:00'
    with get_connection() as connection:
        connection.execute(
            'UPDATE documents SET created_at = ?, updated_at = ? WHERE document_id = ?',
            (expired_timestamp, expired_timestamp, document.document_id),
        )

    summary = service.cleanup_orphaned_assets()

    with get_connection() as connection:
        document_row = connection.execute(
            'SELECT document_id FROM documents WHERE document_id = ?',
            (document.document_id,),
        ).fetchone()
        chunk_rows = connection.execute(
            'SELECT COUNT(*) AS total FROM document_chunks WHERE document_id = ?',
            (document.document_id,),
        ).fetchone()
        embedding_rows = connection.execute(
            'SELECT COUNT(*) AS total FROM chunk_embeddings WHERE document_id = ?',
            (document.document_id,),
        ).fetchone()

    assert summary['purged_documents'] == 1
    assert document_row is None
    assert int(chunk_rows['total']) == 0
    assert int(embedding_rows['total']) == 0
    assert not Path(document.storage_path).exists()

    get_settings.cache_clear()


