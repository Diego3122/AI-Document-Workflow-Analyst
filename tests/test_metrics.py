from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.utils import generate_id
from app.services.logging_service import LoggingService



def _upload_and_index(client: TestClient, filename: str, content: bytes) -> str:
    upload_response = client.post(
        '/documents/upload',
        files={'file': (filename, content, 'text/plain')},
    )
    assert upload_response.status_code == 201
    document_id = upload_response.json()['document_id']

    index_response = client.post('/documents/index', json={'document_id': document_id})
    assert index_response.status_code == 200
    return document_id



def test_metrics_endpoint_tracks_query_aggregates(client: TestClient) -> None:
    baseline = client.get('/metrics')
    assert baseline.status_code == 200
    baseline_payload = baseline.json()

    document_id = _upload_and_index(
        client,
        'metrics_payment_terms.txt',
        b'PAYMENT TERMS\nInvoices are due within 10 days of receipt.',
    )
    query_response = client.post(
        '/query',
        json={'query': 'What are the payment terms?', 'document_ids': [document_id]},
    )
    assert query_response.status_code == 200

    metrics_response = client.get('/metrics')
    assert metrics_response.status_code == 200
    payload = metrics_response.json()
    assert payload['total_requests'] == baseline_payload['total_requests'] + 1
    assert payload['average_confidence'] >= 0.0
    assert 'question_answering' in payload['task_type_breakdown']
    assert payload['vector_backend_breakdown']



def test_metrics_separate_workflow_failures_from_quality_averages(client: TestClient) -> None:
    document_id = _upload_and_index(
        client,
        'failure_metrics_terms.txt',
        b'PAYMENT TERMS\nInvoices are due within 25 days of receipt.',
    )
    query_response = client.post(
        '/query',
        json={'query': 'What are the payment terms?', 'document_ids': [document_id]},
    )
    assert query_response.status_code == 200

    baseline_payload = client.get('/metrics').json()
    baseline_backends = dict(baseline_payload['vector_backend_breakdown'])
    LoggingService().log_failure(
        request_id=generate_id('req'),
        query='synthetic failure for metrics coverage',
        latency_ms=0.0,
        error_message='synthetic test failure',
    )

    after_payload = client.get('/metrics').json()
    assert after_payload['workflow_error_count'] == baseline_payload['workflow_error_count'] + 1
    assert after_payload['total_requests'] == baseline_payload['total_requests'] + 1
    assert after_payload['average_confidence'] == baseline_payload['average_confidence']
    assert after_payload['grounded_rate'] == baseline_payload['grounded_rate']
    assert after_payload['vector_backend_breakdown'] == baseline_backends



def test_metrics_evaluate_endpoint_records_latest_evaluation(client: TestClient) -> None:
    settings = get_settings()
    dataset_path = settings.evals_dir / 'route_eval_dataset.jsonl'
    dataset_path.write_text(
        '{"case_id":"route_eval_case","documents":[{"filename":"invoice.txt","content":"PAYMENT TERMS\\nInvoice payment is due within 15 days.\\nLATE FEES\\nA 5 percent fee applies after 30 days.","mime_type":"text/plain"}],"query":"Extract the payment terms and fees as JSON fields.","expected_task_type":"structured_extraction","expected_structured_keys":["fields"],"min_citations":1,"expected_review":false}\n',
        encoding='utf-8',
    )

    eval_response = client.post('/metrics/evaluate', json={'dataset_path': dataset_path.name, 'max_cases': 1})
    assert eval_response.status_code == 200
    payload = eval_response.json()
    assert payload['total_cases'] == 1
    assert payload['passed_cases'] == 1
    assert payload['failed_case_ids'] == []
    assert payload['task_type_match_rate'] == 1.0
    assert payload['cases'][0]['case_id'] == 'route_eval_case'
    assert payload['cases'][0]['passed'] is True

    latest_response = client.get('/metrics/evaluations/latest')
    assert latest_response.status_code == 200
    latest_payload = latest_response.json()
    assert latest_payload == payload

    metrics_payload = client.get('/metrics').json()
    assert metrics_payload['latest_evaluation'] is not None
    assert metrics_payload['latest_evaluation']['evaluation_id'] == payload['evaluation_id']



def test_metrics_evaluate_uses_default_sample_dataset(client: TestClient) -> None:
    response = client.post('/metrics/evaluate', json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload['dataset_path'].endswith('sample_eval_dataset.jsonl')
    assert payload['total_cases'] == 3
    assert payload['passed_cases'] == payload['total_cases']
    assert payload['failed_case_ids'] == []
    assert payload['task_type_match_rate'] == 1.0
    assert payload['cases']
