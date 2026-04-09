from __future__ import annotations

from app.core.config import get_settings
from app.models.schemas import TaskType
from app.services.eval_service import EvalService
from app.services.logging_service import LoggingService



def _dataset_path(name: str):
    settings = get_settings()
    return settings.evals_dir / name



def test_eval_service_runs_current_dataset_shape() -> None:
    dataset_path = _dataset_path('eval_dataset.jsonl')
    dataset_path.write_text(
        '{"case_id":"eval_case_1","documents":[{"filename":"payment.txt","content":"PAYMENT TERMS\\nInvoices are due within 20 days.","mime_type":"text/plain"}],"query":"What are the payment terms?","expected_task_type":"question_answering","expected_answer_contains":["20 days"],"min_citations":1,"expected_review":false}\n',
        encoding='utf-8',
    )

    summary = EvalService().run(dataset_path=dataset_path)
    assert summary.total_cases == 1
    assert summary.passed_cases == 1
    assert summary.failed_case_ids == []
    assert summary.task_type_match_rate == 1.0
    assert summary.review_rate == 0.0
    assert summary.grounded_rate == 1.0
    assert summary.citation_success_rate == 1.0
    assert summary.structured_success_rate == 1.0
    assert summary.cases
    assert summary.cases[0].case_id == 'eval_case_1'
    assert summary.cases[0].passed is True
    assert summary.cases[0].task_type == 'question_answering'
    assert summary.cases[0].task_type_match is True
    assert summary.cases[0].citation_success is True
    assert summary.cases[0].structured_success is True

    latest = LoggingService().fetch_latest_evaluation()
    assert latest is not None
    assert latest.model_dump() == summary.model_dump()



def test_eval_service_accepts_legacy_dataset_shape() -> None:
    dataset_path = _dataset_path('legacy_eval_dataset.jsonl')
    dataset_path.write_text(
        '{"case_id":"legacy_case","document_filename":"payment.txt","document_text":"PAYMENT TERMS\\nInvoices are due within 20 days.","query":"What are the payment terms?","expected_task_type":"question_answering","expected_terms":["20 days"],"min_citations":1,"expect_human_review":false}\n',
        encoding='utf-8',
    )

    cases = EvalService()._load_cases(dataset_path)
    assert len(cases) == 1
    assert cases[0].documents[0].filename == 'payment.txt'
    assert cases[0].expected_answer_contains == ['20 days']
    assert cases[0].expected_review is False
    assert cases[0].expected_task_type == TaskType.QUESTION_ANSWERING



def test_eval_service_rejects_invalid_json_line() -> None:
    dataset_path = _dataset_path('invalid_json.jsonl')
    dataset_path.write_text('{not valid json}\n', encoding='utf-8')

    try:
        EvalService()._load_cases(dataset_path)
    except ValueError as exc:
        assert 'Invalid JSON' in str(exc)
    else:
        raise AssertionError('Expected ValueError for malformed evaluation JSON.')



def test_eval_service_rejects_invalid_case_schema() -> None:
    dataset_path = _dataset_path('invalid_case.jsonl')
    dataset_path.write_text('{"case_id":"missing_documents","query":"What happened?"}\n', encoding='utf-8')

    try:
        EvalService()._load_cases(dataset_path)
    except ValueError as exc:
        assert 'Invalid evaluation case' in str(exc)
    else:
        raise AssertionError('Expected ValueError for invalid evaluation case schema.')



def test_eval_service_missing_dataset_raises() -> None:
    missing_path = _dataset_path('missing.jsonl')
    try:
        EvalService().run(dataset_path=missing_path)
    except FileNotFoundError:
        assert True
    else:
        raise AssertionError('Expected FileNotFoundError for missing dataset path.')
