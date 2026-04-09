from __future__ import annotations

import json

from app.core.config import get_settings
from app.core.utils import generate_id, iso_timestamp
from app.db.sqlite import get_connection
from app.models.requests import QueryRequest, ReviewDecisionRequest
from app.models.responses import EvalRunResponse, MetricsResponse, QueryResponse
from app.models.schemas import ValidationReport
from app.services.security_service import AuthPrincipal


class LoggingService:
    """Persists workflow traces, review decisions, and evaluation summaries to SQLite."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def create_review_id(self) -> str:
        return generate_id('rev')

    def create_evaluation_id(self) -> str:
        return generate_id('eval')

    def log_request(
        self,
        response: QueryResponse,
        report: ValidationReport,
        request: QueryRequest,
        principal: AuthPrincipal | None = None,
    ) -> None:
        auth_context = self._auth_context(principal)
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO request_logs (
                    request_id,
                    query_text,
                    task_type,
                    document_ids_json,
                    metadata_filters_json,
                    requested_top_k,
                    task_type_hint,
                    llm_model,
                    latency_ms,
                    retrieved_chunk_ids,
                    retrieval_stats_json,
                    structured_data_json,
                    confidence,
                    needs_human_review,
                    validation_notes,
                    schema_valid,
                    grounded,
                    citation_count,
                    citation_coverage,
                    evidence_strength,
                    citation_ids_valid,
                    answer_support_score,
                    insufficient_evidence,
                    vector_backend,
                    embedding_backend,
                    error_message,
                    auth_subject_id,
                    auth_tenant_id,
                    auth_role,
                    auth_type,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response.request_id,
                    request.query,
                    response.task_type.value,
                    json.dumps(request.document_ids),
                    json.dumps(request.metadata_filters),
                    request.top_k,
                    request.task_type_hint.value if request.task_type_hint is not None else None,
                    self.settings.resolved_llm_model(),
                    response.latency_ms,
                    json.dumps(response.retrieved_chunk_ids),
                    json.dumps(response.retrieval_stats),
                    json.dumps(response.structured_data),
                    response.confidence,
                    int(response.needs_human_review),
                    json.dumps(response.validation_notes),
                    int(report.schema_valid),
                    int(report.grounded),
                    len(response.citations),
                    report.citation_coverage,
                    report.evidence_strength,
                    int(report.citation_ids_valid),
                    report.answer_support_score,
                    int(report.insufficient_evidence),
                    str(response.retrieval_stats.get('vector_backend')) if response.retrieval_stats.get('vector_backend') is not None else None,
                    str(response.retrieval_stats.get('embedding_backend')) if response.retrieval_stats.get('embedding_backend') is not None else None,
                    None,
                    auth_context['auth_subject_id'],
                    auth_context['auth_tenant_id'],
                    auth_context['auth_role'],
                    auth_context['auth_type'],
                    iso_timestamp(),
                ),
            )

    def log_failure(
        self,
        request_id: str,
        query: str,
        latency_ms: float,
        error_message: str,
        principal: AuthPrincipal | None = None,
    ) -> None:
        auth_context = self._auth_context(principal)
        with get_connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO request_logs (
                    request_id,
                    query_text,
                    task_type,
                    document_ids_json,
                    metadata_filters_json,
                    requested_top_k,
                    task_type_hint,
                    llm_model,
                    latency_ms,
                    retrieved_chunk_ids,
                    retrieval_stats_json,
                    structured_data_json,
                    confidence,
                    needs_human_review,
                    validation_notes,
                    schema_valid,
                    grounded,
                    citation_count,
                    citation_coverage,
                    evidence_strength,
                    citation_ids_valid,
                    answer_support_score,
                    insufficient_evidence,
                    vector_backend,
                    embedding_backend,
                    error_message,
                    auth_subject_id,
                    auth_tenant_id,
                    auth_role,
                    auth_type,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    query,
                    'workflow_error',
                    json.dumps([]),
                    json.dumps({}),
                    0,
                    None,
                    self.settings.resolved_llm_model(),
                    latency_ms,
                    json.dumps([]),
                    json.dumps({}),
                    json.dumps({}),
                    0.0,
                    1,
                    json.dumps(['Query workflow failed before a final answer was returned.']),
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    0,
                    0.0,
                    1,
                    None,
                    None,
                    error_message,
                    auth_context['auth_subject_id'],
                    auth_context['auth_tenant_id'],
                    auth_context['auth_role'],
                    auth_context['auth_type'],
                    iso_timestamp(),
                ),
            )

    def log_review_decision(
        self,
        review_id: str,
        decision: ReviewDecisionRequest,
        principal: AuthPrincipal | None = None,
    ) -> None:
        auth_context = self._auth_context(principal)
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO review_decisions (
                    review_id,
                    request_id,
                    action,
                    edited_answer,
                    reviewer_notes,
                    reviewer_subject_id,
                    reviewer_tenant_id,
                    reviewer_role,
                    reviewer_auth_type,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    decision.request_id,
                    decision.action.value,
                    decision.edited_answer,
                    decision.reviewer_notes,
                    auth_context['auth_subject_id'],
                    auth_context['auth_tenant_id'],
                    auth_context['auth_role'],
                    auth_context['auth_type'],
                    iso_timestamp(),
                ),
            )

    def log_evaluation_run(self, evaluation: EvalRunResponse) -> None:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs (
                    evaluation_id,
                    dataset_path,
                    total_cases,
                    passed_cases,
                    pass_rate,
                    task_type_match_rate,
                    average_latency_ms,
                    average_confidence,
                    review_rate,
                    grounded_rate,
                    citation_success_rate,
                    structured_success_rate,
                    failed_case_ids_json,
                    summary_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation.evaluation_id,
                    evaluation.dataset_path,
                    evaluation.total_cases,
                    evaluation.passed_cases,
                    evaluation.pass_rate,
                    evaluation.task_type_match_rate,
                    evaluation.average_latency_ms,
                    evaluation.average_confidence,
                    evaluation.review_rate,
                    evaluation.grounded_rate,
                    evaluation.citation_success_rate,
                    evaluation.structured_success_rate,
                    json.dumps(evaluation.failed_case_ids),
                    evaluation.model_dump_json(),
                    evaluation.created_at,
                ),
            )
            for case in evaluation.cases:
                evaluation_case_id = f'{evaluation.evaluation_id}:{case.case_id}'
                connection.execute(
                    """
                    INSERT OR REPLACE INTO evaluation_case_results (
                        evaluation_case_id,
                        evaluation_id,
                        case_id,
                        expected_task_type,
                        task_type,
                        task_type_match,
                        passed,
                        confidence,
                        latency_ms,
                        expected_review,
                        actual_review,
                        expected_citation_count,
                        actual_citation_count,
                        grounded,
                        citation_success,
                        structured_success,
                        notes_json,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evaluation_case_id,
                        evaluation.evaluation_id,
                        case.case_id,
                        case.expected_task_type,
                        case.task_type,
                        int(case.task_type_match),
                        int(case.passed),
                        case.confidence,
                        case.latency_ms,
                        None if case.expected_review is None else int(case.expected_review),
                        int(case.actual_review),
                        case.expected_citation_count,
                        case.actual_citation_count,
                        int(case.grounded),
                        int(case.citation_success),
                        int(case.structured_success),
                        json.dumps(case.notes),
                        evaluation.created_at,
                    ),
                )

    def fetch_metrics(self) -> MetricsResponse:
        with get_connection() as connection:
            totals = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_requests,
                    SUM(CASE WHEN task_type = 'workflow_error' THEN 1 ELSE 0 END) AS workflow_error_count,
                    MAX(created_at) AS last_request_at
                FROM request_logs
                """
            ).fetchone()
            aggregate = connection.execute(
                """
                SELECT
                    COALESCE(AVG(latency_ms), 0) AS average_latency_ms,
                    COALESCE(AVG(confidence), 0) AS average_confidence,
                    COALESCE(AVG(answer_support_score), 0) AS average_answer_support_score,
                    COALESCE(AVG(needs_human_review), 0) AS review_rate,
                    COALESCE(AVG(grounded), 0) AS grounded_rate,
                    COALESCE(AVG(insufficient_evidence), 0) AS insufficient_evidence_rate,
                    SUM(CASE WHEN schema_valid = 0 THEN 1 ELSE 0 END) AS schema_failure_count
                FROM request_logs
                WHERE task_type != 'workflow_error'
                """
            ).fetchone()
            task_rows = connection.execute(
                'SELECT task_type, COUNT(*) AS count FROM request_logs GROUP BY task_type'
            ).fetchall()
            backend_rows = connection.execute(
                "SELECT COALESCE(vector_backend, 'unknown') AS backend, COUNT(*) AS count FROM request_logs WHERE task_type != 'workflow_error' GROUP BY COALESCE(vector_backend, 'unknown')"
            ).fetchall()

        return MetricsResponse(
            total_requests=int(totals['total_requests'] or 0),
            workflow_error_count=int(totals['workflow_error_count'] or 0),
            average_latency_ms=round(float(aggregate['average_latency_ms'] or 0.0), 2),
            average_confidence=round(float(aggregate['average_confidence'] or 0.0), 2),
            average_answer_support_score=round(float(aggregate['average_answer_support_score'] or 0.0), 2),
            review_rate=round(float(aggregate['review_rate'] or 0.0), 2),
            grounded_rate=round(float(aggregate['grounded_rate'] or 0.0), 2),
            insufficient_evidence_rate=round(float(aggregate['insufficient_evidence_rate'] or 0.0), 2),
            schema_failure_count=int(aggregate['schema_failure_count'] or 0),
            task_type_breakdown={row['task_type']: int(row['count']) for row in task_rows},
            vector_backend_breakdown={row['backend']: int(row['count']) for row in backend_rows},
            last_request_at=totals['last_request_at'],
            latest_evaluation=self.fetch_latest_evaluation(),
        )

    def fetch_latest_evaluation(self) -> EvalRunResponse | None:
        with get_connection() as connection:
            row = connection.execute(
                'SELECT summary_json FROM evaluation_runs ORDER BY created_at DESC LIMIT 1'
            ).fetchone()
        if row is None:
            return None
        return EvalRunResponse.model_validate_json(row['summary_json'])

    def _auth_context(self, principal: AuthPrincipal | None) -> dict[str, str | None]:
        if principal is None:
            return {
                'auth_subject_id': None,
                'auth_tenant_id': None,
                'auth_role': None,
                'auth_type': None,
            }
        return {
            'auth_subject_id': principal.subject_id,
            'auth_tenant_id': principal.tenant_id,
            'auth_role': principal.role,
            'auth_type': principal.auth_type,
        }
