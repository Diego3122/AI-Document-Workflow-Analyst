from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.config import get_settings
from app.core.utils import iso_timestamp
from app.models.requests import QueryRequest
from app.models.responses import EvalRunResponse, EvaluationCaseResponse
from app.models.schemas import TaskType
from app.services.document_ingestion_service import DocumentIngestionService
from app.services.logging_service import LoggingService
from app.services.query_workflow_service import QueryWorkflowService


class EvalDatasetDocument(BaseModel):
    filename: str
    content: str
    mime_type: str = 'text/plain'


class EvalDatasetCase(BaseModel):
    case_id: str
    documents: list[EvalDatasetDocument] = Field(default_factory=list)
    query: str
    task_type_hint: TaskType | None = None
    expected_task_type: TaskType | None = None
    top_k: int = 5
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_structured_keys: list[str] = Field(default_factory=list)
    expected_review: bool | None = None
    min_citations: int = 0

    @model_validator(mode='before')
    @classmethod
    def normalize_legacy_payload(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        if 'documents' not in normalized:
            filename = normalized.pop('document_filename', None)
            document_text = normalized.pop('document_text', None)
            document_mime_type = normalized.pop('document_mime_type', 'text/plain')
            if filename is not None and document_text is not None:
                normalized['documents'] = [
                    {
                        'filename': filename,
                        'content': document_text,
                        'mime_type': document_mime_type,
                    }
                ]

        if 'expected_terms' in normalized and 'expected_answer_contains' not in normalized:
            normalized['expected_answer_contains'] = normalized.pop('expected_terms')
        if 'required_structured_keys' in normalized and 'expected_structured_keys' not in normalized:
            normalized['expected_structured_keys'] = normalized.pop('required_structured_keys')
        if 'expect_human_review' in normalized and 'expected_review' not in normalized:
            normalized['expected_review'] = normalized.pop('expect_human_review')
        return normalized

    @model_validator(mode='after')
    def validate_documents(self) -> EvalDatasetCase:
        if not self.documents:
            raise ValueError('Each evaluation case must define at least one document.')
        return self


class EvalService:
    """Runs offline evaluation cases against the same workflow used by the API."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logging_service = LoggingService()
        self.workflow_service = QueryWorkflowService()

    def run(self, dataset_path: Path | None = None, max_cases: int | None = None) -> EvalRunResponse:
        resolved_path = self._resolve_dataset_path(dataset_path)
        cases = self._load_cases(resolved_path)
        if max_cases is not None:
            cases = cases[:max_cases]
        if not cases:
            raise ValueError('Evaluation dataset contained no cases.')

        evaluation_id = self.logging_service.create_evaluation_id()
        created_at = iso_timestamp()
        case_results: list[EvaluationCaseResponse] = []
        grounded_passes = 0
        citation_passes = 0
        structured_passes = 0
        task_type_matches = 0

        for case in cases:
            case_result = self._run_case(case)
            case_results.append(case_result)
            grounded_passes += int(case_result.grounded)
            citation_passes += int(case_result.citation_success)
            structured_passes += int(case_result.structured_success)
            task_type_matches += int(case_result.task_type_match)

        total_cases = len(case_results)
        passed_cases = sum(1 for case in case_results if case.passed)
        failed_case_ids = [case.case_id for case in case_results if not case.passed]
        evaluation = EvalRunResponse(
            evaluation_id=evaluation_id,
            dataset_path=str(resolved_path),
            total_cases=total_cases,
            passed_cases=passed_cases,
            pass_rate=round(passed_cases / total_cases, 2),
            task_type_match_rate=round(task_type_matches / total_cases, 2),
            average_latency_ms=round(sum(case.latency_ms for case in case_results) / total_cases, 2),
            average_confidence=round(sum(case.confidence for case in case_results) / total_cases, 2),
            review_rate=round(sum(1 for case in case_results if case.actual_review) / total_cases, 2),
            grounded_rate=round(grounded_passes / total_cases, 2),
            citation_success_rate=round(citation_passes / total_cases, 2),
            structured_success_rate=round(structured_passes / total_cases, 2),
            created_at=created_at,
            failed_case_ids=failed_case_ids,
            cases=case_results,
        )
        self.logging_service.log_evaluation_run(evaluation)
        return evaluation

    def _resolve_dataset_path(self, dataset_path: Path | None) -> Path:
        resolved_path = Path(dataset_path or self.settings.sample_eval_dataset_path).expanduser().resolve()
        evals_root = self.settings.evals_dir.resolve()
        if evals_root != resolved_path.parent and evals_root not in resolved_path.parents:
            raise ValueError('Evaluation datasets must be located inside the configured evals directory.')
        return resolved_path

    def _load_cases(self, dataset_path: Path) -> list[EvalDatasetCase]:
        if not dataset_path.exists():
            raise FileNotFoundError(f'Evaluation dataset not found: {dataset_path}')

        cases: list[EvalDatasetCase] = []
        for line_number, line in enumerate(dataset_path.read_text(encoding='utf-8').splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON on evaluation dataset line {line_number}: {exc.msg}.') from exc
            try:
                cases.append(EvalDatasetCase.model_validate(payload))
            except ValidationError as exc:
                raise ValueError(f'Invalid evaluation case on line {line_number}: {exc}') from exc
        return cases

    def _run_case(self, case: EvalDatasetCase) -> EvaluationCaseResponse:
        ingestion_service = DocumentIngestionService()
        created_document_ids: list[str] = []
        cleanup_notes: list[str] = []
        case_response: EvaluationCaseResponse

        try:
            for document_input in case.documents:
                document = ingestion_service.save_upload(
                    filename=document_input.filename,
                    content=document_input.content.encode('utf-8'),
                    mime_type=document_input.mime_type,
                )
                created_document_ids.append(document.document_id)

            for document_id in created_document_ids:
                ingestion_service.index_document(document_id, force_reindex=True)

            request = QueryRequest(
                query=case.query,
                document_ids=created_document_ids,
                task_type_hint=case.task_type_hint,
                top_k=case.top_k,
            )
            response, report = self.workflow_service.execute(request=request, request_id=f'eval_{case.case_id}')

            notes: list[str] = []
            task_type_match = True
            if case.expected_task_type is not None and response.task_type != case.expected_task_type:
                task_type_match = False
                notes.append(
                    f'Expected task_type={case.expected_task_type.value} but got {response.task_type.value}.'
                )

            lowered_answer = response.answer.lower()
            missing_answers = [
                expected for expected in case.expected_answer_contains if expected.lower() not in lowered_answer
            ]
            if missing_answers:
                notes.append('Missing expected answer content: ' + ', '.join(missing_answers))

            missing_keys = [key for key in case.expected_structured_keys if key not in response.structured_data]
            structured_success = not missing_keys
            if missing_keys:
                notes.append('Missing structured keys: ' + ', '.join(missing_keys))

            citation_success = len(response.citations) >= case.min_citations
            if not citation_success:
                notes.append(f'Expected at least {case.min_citations} citations but found {len(response.citations)}.')

            if case.expected_review is not None and response.needs_human_review != case.expected_review:
                notes.append(f'Expected review={case.expected_review} but got review={response.needs_human_review}.')

            grounded = report.grounded
            if not grounded and not report.insufficient_evidence:
                notes.append('Validation report marked the answer as ungrounded.')

            case_response = EvaluationCaseResponse(
                case_id=case.case_id,
                passed=not notes,
                expected_task_type=case.expected_task_type.value if case.expected_task_type is not None else None,
                task_type=response.task_type.value,
                task_type_match=task_type_match,
                confidence=response.confidence,
                latency_ms=response.latency_ms,
                expected_review=case.expected_review,
                actual_review=response.needs_human_review,
                expected_citation_count=case.min_citations,
                actual_citation_count=len(response.citations),
                grounded=grounded,
                citation_success=citation_success,
                structured_success=structured_success,
                notes=notes,
            )
        except Exception as exc:
            expected_task_type = case.expected_task_type.value if case.expected_task_type is not None else None
            fallback_task_type = expected_task_type or (
                case.task_type_hint.value if case.task_type_hint is not None else 'workflow_error'
            )
            case_response = EvaluationCaseResponse(
                case_id=case.case_id,
                passed=False,
                expected_task_type=expected_task_type,
                task_type=fallback_task_type,
                task_type_match=case.expected_task_type is None,
                confidence=0.0,
                latency_ms=0.0,
                expected_review=case.expected_review,
                actual_review=True,
                expected_citation_count=case.min_citations,
                actual_citation_count=0,
                grounded=False,
                citation_success=False,
                structured_success=False,
                notes=[f'Case execution failed: {exc}'],
            )
        finally:
            for document_id in reversed(created_document_ids):
                try:
                    ingestion_service.delete_document(document_id)
                except Exception as cleanup_exc:  # pragma: no cover
                    cleanup_notes.append(f'Cleanup failed for {document_id}: {cleanup_exc}')

        if cleanup_notes:
            case_response.notes.extend(cleanup_notes)
            case_response.passed = False
            case_response.grounded = False
            case_response.citation_success = False
            case_response.structured_success = False

        return case_response
