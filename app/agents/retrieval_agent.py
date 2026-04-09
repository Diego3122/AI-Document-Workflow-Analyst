from __future__ import annotations

from app.models.schemas import EvidencePack, TaskType
from app.services.retrieval_service import RetrievalService


class RetrievalAgent:
    """Retrieval stage wrapper around the retrieval service."""

    def __init__(self, retrieval_service: RetrievalService | None = None) -> None:
        self.retrieval_service = retrieval_service or RetrievalService()

    def retrieve(
        self,
        query: str,
        top_k: int,
        task_type: TaskType,
        document_ids: list[str] | None = None,
    ) -> EvidencePack:
        return self.retrieval_service.retrieve(
            query=query,
            top_k=top_k,
            task_type=task_type,
            document_ids=document_ids or [],
        )
