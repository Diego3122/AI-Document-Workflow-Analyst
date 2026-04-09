from __future__ import annotations

import json
import math
from typing import Any

from app.core.config import get_settings
from app.core.utils import iso_timestamp
from app.db.sqlite import get_connection
from app.models.schemas import ChunkRecord, RetrievedChunk


class VectorStoreService:
    """Vector storage with a Chroma path and a SQLite fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._active_backend = 'sqlite'

    @property
    def active_backend(self) -> str:
        return self._active_backend

    def index_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]], embedding_provider: str) -> str:
        if len(chunks) != len(embeddings):
            raise ValueError('Chunk and embedding counts must match.')
        if not chunks:
            return self._active_backend

        if self._should_use_chroma():
            try:
                self._index_with_chroma(chunks, embeddings)
                self._active_backend = 'chroma'
            except Exception:
                self._index_with_sqlite(chunks, embeddings, embedding_provider)
                self._active_backend = 'sqlite'
        else:
            self._index_with_sqlite(chunks, embeddings, embedding_provider)
            self._active_backend = 'sqlite'
        return self._active_backend

    def delete_document(self, document_id: str) -> None:
        self.delete_chunks(chunk_ids=[], document_id=document_id)

    def delete_chunks(self, chunk_ids: list[str], document_id: str | None = None) -> None:
        if not chunk_ids and not document_id:
            return

        with get_connection() as connection:
            if chunk_ids:
                placeholders = ', '.join('?' for _ in chunk_ids)
                connection.execute(f'DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})', chunk_ids)
            elif document_id is not None:
                connection.execute('DELETE FROM chunk_embeddings WHERE document_id = ?', (document_id,))

        if self._should_use_chroma():
            try:
                collection = self._get_chroma_collection()
                if chunk_ids:
                    collection.delete(ids=chunk_ids)
                elif document_id is not None:
                    collection.delete(where={'document_id': document_id})
            except Exception:
                pass

    def prune_orphaned_records(self) -> dict[str, int]:
        return self.reconcile_orphans()

    def reconcile_orphans(self) -> dict[str, int]:
        valid_document_ids = self._valid_document_ids()
        valid_chunk_ids = self._valid_chunk_ids()
        removed_sqlite_embeddings = self._delete_orphaned_sqlite_embeddings(valid_document_ids, valid_chunk_ids)
        removed_chroma_chunks = 0

        if self._should_use_chroma():
            try:
                collection = self._get_chroma_collection()
                snapshot = collection.get(include=['metadatas'])
                stale_chunk_ids: list[str] = []
                for chunk_id, metadata in zip(
                    snapshot.get('ids', []),
                    snapshot.get('metadatas', []),
                    strict=False,
                ):
                    document_id = (metadata or {}).get('document_id')
                    if chunk_id not in valid_chunk_ids or document_id not in valid_document_ids:
                        stale_chunk_ids.append(chunk_id)
                if stale_chunk_ids:
                    collection.delete(ids=stale_chunk_ids)
                removed_chroma_chunks = len(stale_chunk_ids)
            except Exception:
                pass

        return {
            'removed_sqlite_embeddings': removed_sqlite_embeddings,
            'removed_chroma_chunks': removed_chroma_chunks,
        }

    def query(self, query_embedding: list[float], top_k: int, document_ids: list[str] | None = None) -> list[RetrievedChunk]:
        document_ids = document_ids or []
        if self._should_use_chroma():
            try:
                results = self._query_with_chroma(query_embedding, top_k, document_ids)
                results = self._retain_live_retrieved_chunks(results, document_ids)
                self._active_backend = 'chroma'
                if results:
                    return results
            except Exception:
                pass
        self._active_backend = 'sqlite'
        return self._query_with_sqlite(query_embedding, top_k, document_ids)

    def _should_use_chroma(self) -> bool:
        if self.settings.vector_backend == 'sqlite':
            return False
        try:
            import chromadb  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_chroma_collection(self):
        import chromadb

        client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        return client.get_or_create_collection(name='document_chunks')

    def _index_with_chroma(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        collection = self._get_chroma_collection()
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    'document_id': chunk.document_id,
                    'document_name': chunk.document_name,
                    'chunk_index': chunk.chunk_index,
                    'page_number': chunk.page_number,
                    'section': chunk.section or '',
                    'source_uri': chunk.source_uri,
                }
                for chunk in chunks
            ],
        )

    def _query_with_chroma(self, query_embedding: list[float], top_k: int, document_ids: list[str]) -> list[RetrievedChunk]:
        collection = self._get_chroma_collection()
        where: dict[str, Any] | None = None
        if len(document_ids) == 1:
            where = {'document_id': document_ids[0]}
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)

        ids = results.get('ids', [[]])[0]
        distances = results.get('distances', [[]])[0] if results.get('distances') else []
        retrieved, orphaned_chunk_ids = self._filter_live_chunks(ids, distances, document_ids)

        if orphaned_chunk_ids:
            try:
                collection.delete(ids=orphaned_chunk_ids)
            except Exception:
                pass
        return retrieved

    def _filter_live_chunks(
        self,
        chunk_ids: list[str],
        distances: list[float],
        document_ids: list[str],
    ) -> tuple[list[RetrievedChunk], list[str]]:
        valid_rows = self._fetch_valid_chunk_rows(chunk_ids, document_ids)
        orphaned_chunk_ids: list[str] = []
        retrieved: list[RetrievedChunk] = []

        for index, chunk_id in enumerate(chunk_ids):
            row = valid_rows.get(chunk_id)
            if row is None:
                orphaned_chunk_ids.append(chunk_id)
                continue

            distance = distances[index] if index < len(distances) else 0.0
            score = round(1.0 / (1.0 + float(distance)), 4)
            retrieved.append(
                RetrievedChunk(
                    chunk_id=row['chunk_id'],
                    document_id=row['document_id'],
                    document_name=row['document_name'],
                    page_number=row['page_number'],
                    section=row['section'],
                    score=score,
                    vector_score=score,
                    text=row['text'],
                )
            )

        return retrieved, orphaned_chunk_ids

    def _retain_live_retrieved_chunks(self, chunks: list[RetrievedChunk], document_ids: list[str]) -> list[RetrievedChunk]:
        if not chunks:
            return []

        valid_rows = self._fetch_valid_chunk_rows([chunk.chunk_id for chunk in chunks], document_ids)
        retained: list[RetrievedChunk] = []
        for chunk in chunks:
            row = valid_rows.get(chunk.chunk_id)
            if row is None:
                continue
            retained.append(
                RetrievedChunk(
                    chunk_id=row['chunk_id'],
                    document_id=row['document_id'],
                    document_name=row['document_name'],
                    page_number=row['page_number'],
                    section=row['section'],
                    score=chunk.score,
                    keyword_score=chunk.keyword_score,
                    vector_score=chunk.vector_score,
                    text=row['text'],
                )
            )
        return retained

    def _index_with_sqlite(self, chunks: list[ChunkRecord], embeddings: list[list[float]], embedding_provider: str) -> None:
        now = iso_timestamp()
        with get_connection() as connection:
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO chunk_embeddings (
                        chunk_id,
                        document_id,
                        embedding_json,
                        embedding_provider,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        json.dumps(embedding),
                        embedding_provider,
                        now,
                    ),
                )

    def _query_with_sqlite(self, query_embedding: list[float], top_k: int, document_ids: list[str]) -> list[RetrievedChunk]:
        query = """
            SELECT dc.chunk_id, dc.document_id, dc.document_name, dc.page_number, dc.section, dc.text, ce.embedding_json
            FROM document_chunks dc
            INNER JOIN chunk_embeddings ce ON ce.chunk_id = dc.chunk_id
        """
        parameters: list[str] = []
        if document_ids:
            placeholders = ', '.join('?' for _ in document_ids)
            query += f' WHERE dc.document_id IN ({placeholders})'
            parameters.extend(document_ids)
        query += ' ORDER BY dc.chunk_index ASC'

        with get_connection() as connection:
            rows = connection.execute(query, parameters).fetchall()

        scored: list[RetrievedChunk] = []
        for row in rows:
            embedding = json.loads(row['embedding_json'])
            score = self._cosine_similarity(query_embedding, embedding)
            if score <= 0:
                continue
            scored.append(
                RetrievedChunk(
                    chunk_id=row['chunk_id'],
                    document_id=row['document_id'],
                    document_name=row['document_name'],
                    page_number=row['page_number'],
                    section=row['section'],
                    score=round(score, 4),
                    vector_score=round(score, 4),
                    text=row['text'],
                )
            )
        scored.sort(key=lambda item: item.score or 0.0, reverse=True)
        return scored[:top_k]

    def _fetch_valid_chunk_rows(self, chunk_ids: list[str], document_ids: list[str]) -> dict[str, Any]:
        if not chunk_ids:
            return {}

        placeholders = ', '.join('?' for _ in chunk_ids)
        query = f"""
            SELECT dc.chunk_id, dc.document_id, dc.document_name, dc.page_number, dc.section, dc.text
            FROM document_chunks dc
            INNER JOIN documents d ON d.document_id = dc.document_id
            WHERE dc.chunk_id IN ({placeholders})
        """
        parameters: list[str] = list(chunk_ids)
        if document_ids:
            document_placeholders = ', '.join('?' for _ in document_ids)
            query += f' AND dc.document_id IN ({document_placeholders})'
            parameters.extend(document_ids)

        with get_connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return {row['chunk_id']: row for row in rows}

    def _valid_document_ids(self) -> set[str]:
        with get_connection() as connection:
            rows = connection.execute('SELECT document_id FROM documents').fetchall()
        return {row['document_id'] for row in rows}

    def _valid_chunk_ids(self) -> set[str]:
        with get_connection() as connection:
            rows = connection.execute('SELECT chunk_id FROM document_chunks').fetchall()
        return {row['chunk_id'] for row in rows}

    def _delete_orphaned_sqlite_embeddings(self, valid_document_ids: set[str], valid_chunk_ids: set[str]) -> int:
        with get_connection() as connection:
            rows = connection.execute(
                'SELECT chunk_id, document_id FROM chunk_embeddings'
            ).fetchall()
            orphaned_chunk_ids = [
                row['chunk_id']
                for row in rows
                if row['chunk_id'] not in valid_chunk_ids or row['document_id'] not in valid_document_ids
            ]
            if orphaned_chunk_ids:
                placeholders = ', '.join('?' for _ in orphaned_chunk_ids)
                connection.execute(f'DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})', orphaned_chunk_ids)
        return len(orphaned_chunk_ids)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)
