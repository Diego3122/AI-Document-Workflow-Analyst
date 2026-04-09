from __future__ import annotations

import re

from app.core.config import get_settings
from app.models.schemas import DraftAnswer, EvidencePack, ValidatedResult, ValidationReport


class ValidationAgent:
    """Applies deterministic validation rules before a response is returned."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def validate(self, draft: DraftAnswer, evidence: EvidencePack) -> ValidatedResult:
        notes = list(draft.validation_notes)
        chunk_count = len(evidence.retrieved_chunks)
        citation_count = len(draft.citations)
        evidence_ids = {chunk.chunk_id for chunk in evidence.retrieved_chunks}
        cited_ids = {citation.chunk_id for citation in draft.citations}

        schema_valid = bool(draft.answer) and isinstance(draft.structured_data, dict)
        citation_ids_valid = cited_ids.issubset(evidence_ids)
        citation_coverage = round(citation_count / max(chunk_count, 1), 2)
        evidence_strength = round(min(chunk_count / max(self.settings.retrieval_top_k, 1), 1.0), 2)
        insufficient_evidence = draft.structured_data.get('status') == 'insufficient_evidence' or chunk_count == 0
        answer_support_score = self._answer_support_score(draft.answer, evidence)
        grounded = (citation_count > 0 and citation_ids_valid and chunk_count > 0) or insufficient_evidence
        contradictions_detected = False

        if not schema_valid:
            notes.append('Draft answer is missing required structured fields.')
        if not citation_ids_valid:
            notes.append('Validation found citations that do not exist in the retrieved evidence pack.')
        if chunk_count == 0:
            notes.append('Validation found no retrieved evidence supporting the answer.')
        if citation_count == 0 and not insufficient_evidence:
            notes.append('Validation found no citations attached to the answer.')
        if insufficient_evidence:
            notes.append('The workflow marked this response as insufficiently supported by evidence.')
        if answer_support_score < 0.15 and not insufficient_evidence:
            notes.append('Answer text has weak lexical overlap with the retrieved evidence.')

        confidence_components = [
            draft.confidence,
            citation_coverage,
            evidence_strength,
            answer_support_score,
            1.0 if schema_valid else 0.0,
            1.0 if citation_ids_valid else 0.0,
            1.0 if grounded else 0.0,
        ]
        blended_confidence = round(sum(confidence_components) / len(confidence_components), 2)
        needs_human_review = (
            blended_confidence < self.settings.answer_confidence_threshold
            or not grounded
            or not schema_valid
            or insufficient_evidence
        )

        if blended_confidence < self.settings.human_review_threshold:
            notes.append('Confidence fell below the human review threshold.')
        elif needs_human_review:
            notes.append('Confidence fell below the normal auto-accept threshold.')

        draft.confidence = blended_confidence
        report = ValidationReport(
            schema_valid=schema_valid,
            citation_coverage=citation_coverage,
            evidence_strength=evidence_strength,
            grounded=grounded,
            citation_ids_valid=citation_ids_valid,
            answer_support_score=answer_support_score,
            insufficient_evidence=insufficient_evidence,
            contradictions_detected=contradictions_detected,
            notes=notes,
        )
        return ValidatedResult(draft=draft, report=report, needs_human_review=needs_human_review)

    def _answer_support_score(self, answer: str, evidence: EvidencePack) -> float:
        answer_terms = {term for term in re.findall(r'[a-zA-Z0-9]+', answer.lower()) if len(term) > 2}
        if not answer_terms:
            return 0.0
        evidence_terms = {
            term
            for chunk in evidence.retrieved_chunks
            for term in re.findall(r'[a-zA-Z0-9]+', chunk.text.lower())
            if len(term) > 2
        }
        if not evidence_terms:
            return 0.0
        return round(len(answer_terms & evidence_terms) / len(answer_terms), 2)
