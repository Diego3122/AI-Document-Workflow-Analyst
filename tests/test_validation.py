from __future__ import annotations

from app.agents.validation_agent import ValidationAgent
from app.models.schemas import Citation, DraftAnswer, EvidencePack, RetrievedChunk, TaskType


validation_agent = ValidationAgent()


def test_validation_rejects_citations_not_in_evidence() -> None:
    evidence = EvidencePack(
        query='What are the payment terms?',
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id='chunk_1',
                document_id='doc_1',
                document_name='contract.txt',
                page_number=1,
                text='Payment is due within 30 days.',
            )
        ],
        retrieval_stats={},
    )
    draft = DraftAnswer(
        task_type=TaskType.QUESTION_ANSWERING,
        answer='Payment is due within 30 days.',
        citations=[
            Citation(
                document_id='doc_1',
                document_name='contract.txt',
                page_number=1,
                chunk_id='chunk_missing',
            )
        ],
        confidence=0.8,
        structured_data={'supporting_points': ['Payment is due within 30 days.']},
    )

    validated = validation_agent.validate(draft=draft, evidence=evidence)
    assert validated.needs_human_review is True
    assert validated.report.citation_ids_valid is False


def test_validation_marks_insufficient_evidence_for_review() -> None:
    evidence = EvidencePack(query='What are the penalties?', retrieved_chunks=[], retrieval_stats={})
    draft = DraftAnswer(
        task_type=TaskType.QUESTION_ANSWERING,
        answer='I do not have enough indexed evidence to answer this reliably.',
        citations=[],
        confidence=0.2,
        structured_data={'status': 'insufficient_evidence'},
    )

    validated = validation_agent.validate(draft=draft, evidence=evidence)
    assert validated.needs_human_review is True
    assert validated.report.insufficient_evidence is True
