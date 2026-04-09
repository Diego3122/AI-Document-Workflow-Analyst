from __future__ import annotations

import re

from app.models.schemas import EvidencePack, RetrievedChunk, TaskType
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService

STOPWORDS = {
    'a', 'an', 'and', 'are', 'at', 'be', 'by', 'for', 'from', 'how', 'in', 'is', 'it', 'of',
    'on', 'or', 'that', 'the', 'this', 'to', 'what', 'when', 'where', 'which', 'who', 'with',
}


class RetrievalService:
    """Hybrid retrieval with vector similarity and lightweight lexical gating."""

    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.vector_store_service = VectorStoreService()

    def retrieve(
        self,
        query: str,
        top_k: int,
        task_type: TaskType,
        document_ids: list[str],
    ) -> EvidencePack:
        query_embedding = self.embedding_service.embed_query(query)
        vector_results = self.vector_store_service.query(
            query_embedding=query_embedding,
            top_k=max(top_k * 3, top_k),
            document_ids=document_ids,
        )
        keyword_scores = self._keyword_scores(query=query, chunks=vector_results)
        combined = self._merge_scores(vector_results=vector_results, keyword_scores=keyword_scores, top_k=top_k)

        return EvidencePack(
            query=query,
            retrieved_chunks=combined,
            retrieval_stats={
                'task_type': task_type.value,
                'top_k': top_k,
                'document_filter_count': len(document_ids),
                'vector_candidates': len(vector_results),
                'matched_chunks': len(combined),
                'vector_backend': self.vector_store_service.active_backend,
                'embedding_backend': self.embedding_service.backend,
                'retriever': 'hybrid_vector_keyword',
            },
        )

    def _keyword_scores(self, query: str, chunks: list[RetrievedChunk]) -> dict[str, float]:
        query_terms = self._normalize_terms(query)
        return {
            chunk.chunk_id: self._score_chunk(query_terms=query_terms, text=chunk.text)
            for chunk in chunks
        }

    def _merge_scores(self, vector_results: list[RetrievedChunk], keyword_scores: dict[str, float], top_k: int) -> list[RetrievedChunk]:
        max_vector_score = max((chunk.score or 0.0 for chunk in vector_results), default=1.0)
        max_keyword_score = max(keyword_scores.values(), default=1.0)

        rescored: list[RetrievedChunk] = []
        for chunk in vector_results:
            raw_vector_score = chunk.score or 0.0
            raw_keyword_score = keyword_scores.get(chunk.chunk_id, 0.0)
            if raw_keyword_score == 0.0 and raw_vector_score < 0.35:
                continue

            vector_score = raw_vector_score / max(max_vector_score, 1e-9)
            keyword_score = raw_keyword_score / max(max_keyword_score, 1e-9)
            combined_score = round((0.75 * vector_score) + (0.25 * keyword_score), 4)
            if combined_score < 0.25:
                continue

            chunk.vector_score = round(vector_score, 4)
            chunk.keyword_score = round(keyword_score, 4)
            chunk.score = combined_score
            rescored.append(chunk)

        rescored.sort(key=lambda item: item.score or 0.0, reverse=True)
        return rescored[:top_k]

    def _normalize_terms(self, query: str) -> set[str]:
        return {
            term for term in re.findall(r'[a-zA-Z0-9]+', query.lower())
            if len(term) > 2 and term not in STOPWORDS
        }

    def _score_chunk(self, query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        text_terms = {
            term for term in re.findall(r'[a-zA-Z0-9]+', text.lower())
            if len(term) > 2 and term not in STOPWORDS
        }
        overlap = query_terms & text_terms
        if not overlap:
            return 0.0
        coverage = len(overlap) / len(query_terms)
        density = len(overlap) / max(len(text_terms), 1)
        return coverage + density
