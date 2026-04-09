from __future__ import annotations

import re
from typing import Iterable

from app.core.config import get_settings
from app.core.retries import transient_retry
from app.models.schemas import EvidencePack, StructuredSynthesis, TaskType
from app.services.provider_guard_service import ProviderGuardService, ProviderUsageLimitExceeded


STOPWORDS = {
    'about', 'after', 'again', 'against', 'all', 'also', 'and', 'any', 'are', 'because', 'been', 'before',
    'being', 'between', 'both', 'but', 'can', 'does', 'from', 'have', 'into', 'more', 'most', 'not', 'only',
    'other', 'such', 'than', 'that', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'those',
    'through', 'under', 'very', 'what', 'when', 'where', 'which', 'while', 'with', 'would', 'your', 'the',
}


class LLMService:
    """Structured grounded generation with provider-backed and deterministic local fallback paths."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider_guard = ProviderGuardService()

    def generate_grounded_synthesis(
        self,
        query: str,
        task_type: TaskType,
        evidence: EvidencePack,
    ) -> StructuredSynthesis:
        if not evidence.retrieved_chunks:
            return StructuredSynthesis(
                answer='I do not have enough indexed evidence to answer this reliably.',
                confidence=0.2,
                cited_chunk_ids=[],
                insufficiency_reason='No retrieved evidence was available for synthesis.',
                answer_style='insufficient_evidence',
                structured_data={
                    'task_type': task_type.value,
                    'supporting_chunk_ids': [],
                    'status': 'insufficient_evidence',
                    'evidence_gap': True,
                },
            )

        provider = self.settings.resolve_llm_provider()
        if provider == 'gemini':
            try:
                return self._generate_with_gemini(query=query, task_type=task_type, evidence=evidence)
            except ProviderUsageLimitExceeded:
                raise
            except Exception:
                return self._generate_locally(query=query, task_type=task_type, evidence=evidence)
        if provider == 'openai':
            try:
                return self._generate_with_openai(query=query, task_type=task_type, evidence=evidence)
            except ProviderUsageLimitExceeded:
                raise
            except Exception:
                return self._generate_locally(query=query, task_type=task_type, evidence=evidence)
        return self._generate_locally(query=query, task_type=task_type, evidence=evidence)

    @transient_retry()
    def _generate_with_gemini(
        self,
        query: str,
        task_type: TaskType,
        evidence: EvidencePack,
    ) -> StructuredSynthesis:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError('google-genai must be installed to use Gemini structured generation.') from exc

        user_prompt = self._user_prompt(query=query, evidence=evidence)
        system_prompt = self._system_prompt(task_type)
        self.provider_guard.check_and_record(
            provider='gemini',
            operation='llm',
            texts=[system_prompt, user_prompt],
        )
        client = genai.Client(api_key=self.settings.gemini_api_key)
        response = client.models.generate_content(
            model=self.settings.resolved_llm_model(),
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type='application/json',
                response_schema=StructuredSynthesis,
            ),
        )
        parsed = getattr(response, 'parsed', None)
        if parsed is None:
            text = getattr(response, 'text', None)
            if not text:
                raise RuntimeError('Gemini structured synthesis returned no parsed output.')
            parsed = StructuredSynthesis.model_validate_json(text)
        return self._normalize_structured_synthesis(parsed=parsed, task_type=task_type)

    @transient_retry()
    def _generate_with_openai(
        self,
        query: str,
        task_type: TaskType,
        evidence: EvidencePack,
    ) -> StructuredSynthesis:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError('openai must be installed to use structured generation.') from exc

        user_prompt = self._user_prompt(query=query, evidence=evidence)
        system_prompt = self._system_prompt(task_type)
        self.provider_guard.check_and_record(
            provider='openai',
            operation='llm',
            texts=[system_prompt, user_prompt],
        )
        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.responses.parse(
            model=self.settings.resolved_llm_model(),
            input=[
                {
                    'role': 'system',
                    'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': user_prompt,
                },
            ],
            text_format=StructuredSynthesis,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError('Structured synthesis parsing returned no parsed output.')
        return self._normalize_structured_synthesis(parsed=parsed, task_type=task_type)

    def _normalize_structured_synthesis(self, parsed: StructuredSynthesis, task_type: TaskType) -> StructuredSynthesis:
        parsed.structured_data.setdefault('task_type', task_type.value)
        parsed.structured_data.setdefault('supporting_chunk_ids', parsed.cited_chunk_ids)
        parsed.structured_data.setdefault('evidence_gap', False if parsed.cited_chunk_ids else True)
        if not parsed.cited_chunk_ids:
            parsed.structured_data.setdefault('status', 'insufficient_evidence')
        else:
            parsed.structured_data.setdefault('status', 'grounded')
        return parsed

    def _generate_locally(
        self,
        query: str,
        task_type: TaskType,
        evidence: EvidencePack,
    ) -> StructuredSynthesis:
        chunks = evidence.retrieved_chunks[:3]
        sentences = self._collect_sentences(chunks)
        cited_chunk_ids = [chunk.chunk_id for chunk in chunks]
        base_data = {
            'task_type': task_type.value,
            'supporting_chunk_ids': cited_chunk_ids,
            'evidence_gap': False,
        }

        if task_type == TaskType.STRUCTURED_EXTRACTION:
            fields = self._extract_fields(query=query, sentences=sentences)
            if fields:
                answer = 'Extracted fields: ' + '; '.join(f'{key}: {value}' for key, value in fields.items())
                confidence = 0.82
            else:
                answer = 'No matching fields could be extracted from the retrieved evidence.'
                confidence = 0.42
            return StructuredSynthesis(
                answer=answer,
                confidence=confidence,
                cited_chunk_ids=cited_chunk_ids if fields else [],
                insufficiency_reason=None if fields else 'No extraction fields matched the retrieved evidence.',
                answer_style='structured_extraction',
                structured_data={
                    **base_data,
                    'supporting_chunk_ids': cited_chunk_ids if fields else [],
                    'fields': fields,
                    'query': query,
                    'evidence_gap': not bool(fields),
                    'status': 'grounded' if fields else 'insufficient_evidence',
                },
            )

        if task_type == TaskType.SUMMARIZATION:
            bullet_summary = sentences[:3]
            return StructuredSynthesis(
                answer='Summary: ' + ' '.join(bullet_summary),
                confidence=0.74,
                cited_chunk_ids=cited_chunk_ids,
                answer_style='summary',
                structured_data={**base_data, 'bullets': bullet_summary, 'status': 'grounded'},
            )

        if task_type == TaskType.RISK_FLAGGING:
            risks = [sentence for sentence in sentences if self._looks_like_risk(sentence)]
            if risks:
                return StructuredSynthesis(
                    answer='Potential risks found in retrieved evidence: ' + ' '.join(risks[:3]),
                    confidence=0.71,
                    cited_chunk_ids=cited_chunk_ids,
                    answer_style='risk_flags',
                    structured_data={**base_data, 'risk_flags': risks[:5], 'status': 'grounded'},
                )
            return StructuredSynthesis(
                answer='No explicit risk indicators were identified in the retrieved evidence.',
                confidence=0.48,
                cited_chunk_ids=[],
                insufficiency_reason='Retrieved evidence did not contain clear risk indicators.',
                answer_style='insufficient_evidence',
                structured_data={
                    **base_data,
                    'supporting_chunk_ids': [],
                    'risk_flags': [],
                    'evidence_gap': True,
                    'status': 'insufficient_evidence',
                },
            )

        best_sentences = self._select_relevant_sentences(query=query, sentences=sentences)
        answer = ' '.join(best_sentences[:2]) if best_sentences else 'I could not find a grounded answer in the retrieved evidence.'
        return StructuredSynthesis(
            answer=answer,
            confidence=0.76 if best_sentences else 0.35,
            cited_chunk_ids=cited_chunk_ids if best_sentences else [],
            insufficiency_reason=None if best_sentences else 'Retrieved evidence was too weak to answer directly.',
            answer_style='grounded_answer' if best_sentences else 'insufficient_evidence',
            structured_data={
                **base_data,
                'supporting_chunk_ids': cited_chunk_ids if best_sentences else [],
                'supporting_points': best_sentences[:3],
                'evidence_gap': not bool(best_sentences),
                'status': 'grounded' if best_sentences else 'insufficient_evidence',
            },
        )

    def _system_prompt(self, task_type: TaskType) -> str:
        return (
            'You are a grounded document analyst. Answer only from the provided evidence. '
            'Do not use outside knowledge. Cite only chunk ids from the evidence. '
            'If the evidence is weak, say so clearly and mark the response as insufficient. '
            f'Task type: {task_type.value}.'
        )

    def _user_prompt(self, query: str, evidence: EvidencePack) -> str:
        evidence_lines = []
        for chunk in evidence.retrieved_chunks:
            location = f"page={chunk.page_number}" if chunk.page_number is not None else 'page=unknown'
            evidence_lines.append(
                f"chunk_id={chunk.chunk_id} | document={chunk.document_name} | {location} | text={chunk.text}"
            )
        return 'Query: ' + query + '\n\nEvidence:\n' + '\n'.join(evidence_lines)

    def _collect_sentences(self, chunks) -> list[str]:
        sentences: list[str] = []
        for chunk in chunks:
            sentences.extend(self._split_sentences(chunk.text))
        return [sentence for sentence in sentences if sentence]

    def _split_sentences(self, text: str) -> list[str]:
        raw_parts = text.replace('\n', ' ').split('.')
        return [part.strip() + '.' for part in raw_parts if part.strip()]

    def _select_relevant_sentences(self, query: str, sentences: Iterable[str]) -> list[str]:
        query_terms = self._normalize_terms(query)
        scored = []
        for sentence in sentences:
            sentence_terms = self._normalize_terms(sentence)
            overlap = len(query_terms & sentence_terms)
            scored.append((overlap, sentence))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [sentence for overlap, sentence in scored if overlap > 0]

    def _normalize_terms(self, text: str) -> set[str]:
        return {
            term
            for term in re.findall(r'[a-zA-Z0-9]+', text.lower())
            if len(term) > 2 and term not in STOPWORDS
        }

    def _extract_fields(self, query: str, sentences: list[str]) -> dict[str, str]:
        lowered = query.lower()
        fields: dict[str, str] = {}
        mappings = {
            'payment_terms': ('payment', 'invoice', 'due'),
            'termination': ('terminate', 'termination', 'notice'),
            'renewal': ('renew', 'renewal', 'term'),
            'fees': ('fee', 'fees', 'penalty', 'late'),
        }
        for field_name, keywords in mappings.items():
            if field_name.replace('_', ' ') in lowered or any(keyword in lowered for keyword in keywords):
                match = next((sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)), None)
                if match:
                    fields[field_name] = match
        if not fields and sentences:
            fields['primary_finding'] = sentences[0]
        return fields

    def _looks_like_risk(self, sentence: str) -> bool:
        risk_markers = ('penalty', 'liable', 'liability', 'breach', 'terminate', 'indemn', 'late fee', 'must')
        lowered = sentence.lower()
        return any(marker in lowered for marker in risk_markers)
