from __future__ import annotations

from app.models.schemas import AgentRoute, Citation, DraftAnswer, EvidencePack
from app.services.llm_service import LLMService


class SynthesisAgent:
    """Generates a structured grounded draft from retrieved evidence."""

    def __init__(self, llm_service: LLMService | None = None) -> None:
        self.llm_service = llm_service or LLMService()

    def synthesize(self, query: str, route: AgentRoute, evidence: EvidencePack) -> DraftAnswer:
        synthesis = self.llm_service.generate_grounded_synthesis(
            query=query,
            task_type=route.task_type,
            evidence=evidence,
        )
        evidence_by_id = {chunk.chunk_id: chunk for chunk in evidence.retrieved_chunks}
        citations = [
            Citation(
                document_id=evidence_by_id[chunk_id].document_id,
                document_name=evidence_by_id[chunk_id].document_name,
                page_number=evidence_by_id[chunk_id].page_number,
                chunk_id=chunk_id,
                excerpt=evidence_by_id[chunk_id].text[:180],
            )
            for chunk_id in synthesis.cited_chunk_ids
            if chunk_id in evidence_by_id
        ]

        notes = [f'Routing reason: {route.reason}']
        if synthesis.insufficiency_reason:
            notes.append(synthesis.insufficiency_reason)
        notes.append(f'Synthesis style: {synthesis.answer_style}')

        return DraftAnswer(
            task_type=route.task_type,
            answer=synthesis.answer,
            citations=citations,
            confidence=synthesis.confidence,
            structured_data=synthesis.structured_data,
            validation_notes=notes,
        )
