from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import google.genai

from app.core.config import get_settings


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')


def make_hs256_token(claims: dict[str, object], *, secret: str) -> str:
    header = {'alg': 'HS256', 'typ': 'JWT'}
    encoded_header = _b64url(json.dumps(header, separators=(',', ':')).encode('utf-8'))
    encoded_payload = _b64url(json.dumps(claims, separators=(',', ':')).encode('utf-8'))
    signing_input = f'{encoded_header}.{encoded_payload}'.encode('ascii')
    signature = hmac.new(secret.encode('utf-8'), signing_input, hashlib.sha256).digest()
    return f'{encoded_header}.{encoded_payload}.{_b64url(signature)}'





class _FakeGeminiModels:
    calls: list[dict[str, object]] = []

    def embed_content(self, *, model: str, contents, config):
        batch = list(contents)
        self.__class__.calls.append({'model': model, 'contents': batch})
        return type('FakeEmbedResponse', (), {
            'embeddings': [
                type('FakeEmbedding', (), {'values': [float(index + 1)] * int(config.output_dimensionality or 1)})()
                for index, _ in enumerate(batch)
            ]
        })()


class _FakeGeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.models = _FakeGeminiModels()

def make_user_headers(subject: str, *, scopes: str = 'documents.read documents.write query.run health.read') -> dict[str, str]:
    now = int(time.time())
    token = make_hs256_token(
        {
            'sub': subject,
            'aud': 'api://doc-analyst',
            'iss': 'https://issuer.example.com',
            'exp': now + 600,
            'iat': now,
            'scp': scopes,
        },
        secret='dev-secret',
    )
    return {'Authorization': f'Bearer {token}'}


def test_health_endpoint(client) -> None:
    response = client.get('/health')
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ok'
    assert payload['app_name'] == 'AI Document Workflow Analyst'


def test_upload_and_index_txt_document(client) -> None:
    upload_response = client.post(
        '/documents/upload',
        files={
            'file': (
                'contract.txt',
                b'PAYMENT TERMS\nNet 30 days from invoice date.\n\nTERMINATION\nEither party may terminate with 30 days notice.',
                'text/plain',
            )
        },
    )
    assert upload_response.status_code == 201
    document_id = upload_response.json()['document_id']

    index_response = client.post('/documents/index', json={'document_id': document_id})
    assert index_response.status_code == 200
    index_payload = index_response.json()
    assert index_payload['status'] == 'indexed'
    assert index_payload['page_count'] == 1
    assert index_payload['chunk_count'] >= 1

    detail_response = client.get(f'/documents/{document_id}')
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload['status'] == 'indexed'
    assert detail_payload['chunk_count'] >= 1
    assert detail_payload['chunks'][0]['page_number'] == 1
    assert detail_payload['chunks'][0]['section'] == 'PAYMENT TERMS'


def test_reindex_is_idempotent_without_force(client) -> None:
    upload_response = client.post(
        '/documents/upload',
        files={
            'file': (
                'memo.txt',
                b'RENEWAL\nThe agreement renews annually unless terminated earlier.',
                'text/plain',
            )
        },
    )
    document_id = upload_response.json()['document_id']

    first_index = client.post('/documents/index', json={'document_id': document_id})
    second_index = client.post('/documents/index', json={'document_id': document_id})
    assert first_index.status_code == 200
    assert second_index.status_code == 200
    assert first_index.json()['chunk_count'] == second_index.json()['chunk_count']

    detail_response = client.get(f'/documents/{document_id}')
    detail_payload = detail_response.json()
    assert len(detail_payload['chunks']) == first_index.json()['chunk_count']


def test_unsupported_document_type_is_rejected_on_upload(client) -> None:
    upload_response = client.post(
        '/documents/upload',
        files={
            'file': (
                'spreadsheet.csv',
                b'col1,col2\nfoo,bar',
                'text/csv',
            )
        },
    )
    assert upload_response.status_code == 400
    assert 'Only PDF and TXT files are supported.' in upload_response.json()['detail']


def test_query_returns_grounded_answer_with_structured_data(client) -> None:
    upload_response = client.post(
        '/documents/upload',
        files={
            'file': (
                'payment_terms.txt',
                b'PAYMENT TERMS\nInvoices are due within 30 days of receipt. Late fees apply after 45 days.',
                'text/plain',
            )
        },
    )
    document_id = upload_response.json()['document_id']
    client.post('/documents/index', json={'document_id': document_id})

    query_response = client.post(
        '/query',
        json={
            'query': 'What are the payment terms?',
            'document_ids': [document_id],
            'top_k': 3,
        },
    )
    assert query_response.status_code == 200
    payload = query_response.json()
    assert payload['citations']
    assert payload['citations'][0]['document_id'] == document_id
    assert '30 days' in payload['answer']
    assert payload['needs_human_review'] is False
    assert 'supporting_points' in payload['structured_data']
    assert payload['retrieval_stats']['retriever'] == 'hybrid_vector_keyword'


def test_structured_extraction_returns_fields(client) -> None:
    upload_response = client.post(
        '/documents/upload',
        files={
            'file': (
                'invoice.txt',
                b'PAYMENT TERMS\nInvoice payment is due within 15 days.\nLATE FEES\nA 5 percent fee applies after 30 days.',
                'text/plain',
            )
        },
    )
    document_id = upload_response.json()['document_id']
    client.post('/documents/index', json={'document_id': document_id})

    query_response = client.post(
        '/query',
        json={
            'query': 'Extract the payment terms and fees as JSON fields.',
            'document_ids': [document_id],
        },
    )
    assert query_response.status_code == 200
    payload = query_response.json()
    assert payload['task_type'] == 'structured_extraction'
    assert 'fields' in payload['structured_data']
    assert payload['structured_data']['fields']
    assert payload['citations']


def test_query_document_filter_excludes_other_documents(client) -> None:
    payment_upload = client.post(
        '/documents/upload',
        files={
            'file': (
                'payment.txt',
                b'PAYMENT TERMS\nPayment due in 15 days.',
                'text/plain',
            )
        },
    )
    renewal_upload = client.post(
        '/documents/upload',
        files={
            'file': (
                'renewal.txt',
                b'RENEWAL\nThe contract renews automatically each January unless notice is given.',
                'text/plain',
            )
        },
    )
    payment_id = payment_upload.json()['document_id']
    renewal_id = renewal_upload.json()['document_id']
    client.post('/documents/index', json={'document_id': payment_id})
    client.post('/documents/index', json={'document_id': renewal_id})

    query_response = client.post(
        '/query',
        json={
            'query': 'When does the contract renew?',
            'document_ids': [renewal_id],
        },
    )
    assert query_response.status_code == 200
    payload = query_response.json()
    assert payload['citations']
    assert all(citation['document_id'] == renewal_id for citation in payload['citations'])
    assert 'renews automatically' in payload['answer']


def test_query_without_matching_evidence_flags_review(client) -> None:
    upload_response = client.post(
        '/documents/upload',
        files={
            'file': (
                'services.txt',
                b'SERVICES\nThe vendor provides bookkeeping support and monthly reporting.',
                'text/plain',
            )
        },
    )
    document_id = upload_response.json()['document_id']
    client.post('/documents/index', json={'document_id': document_id})

    query_response = client.post(
        '/query',
        json={
            'query': 'What are the cybersecurity breach penalties?',
            'document_ids': [document_id],
        },
    )
    assert query_response.status_code == 200
    payload = query_response.json()
    assert payload['citations'] == []
    assert payload['needs_human_review'] is True
    assert payload['structured_data'].get('status') == 'insufficient_evidence'
    assert payload['retrieval_stats']['matched_chunks'] == 0


def test_query_endpoint_returns_structured_response(client) -> None:
    response = client.post('/query', json={'query': 'What are the payment terms?'})
    assert response.status_code == 400
    assert 'accessible indexed documents' in response.json()['detail']


def test_index_returns_429_when_provider_char_limit_is_reached(client, monkeypatch) -> None:
    monkeypatch.setenv('GEMINI_API_KEY', 'test-gemini-key')
    monkeypatch.setenv('PROVIDER_DAILY_INPUT_CHAR_LIMIT', '10')
    get_settings.cache_clear()

    upload_response = client.post(
        '/documents/upload',
        files={
            'file': (
                'payment_terms.txt',
                b'PAYMENT TERMS\nInvoices are due within 30 days of receipt.',
                'text/plain',
            )
        },
    )
    document_id = upload_response.json()['document_id']

    response = client.post('/documents/index', json={'document_id': document_id})
    assert response.status_code == 429
    assert 'input character limit' in response.json()['detail']

    get_settings.cache_clear()


def test_query_returns_429_when_provider_request_limit_is_reached(client, monkeypatch) -> None:
    monkeypatch.setenv('GEMINI_API_KEY', 'test-gemini-key')
    monkeypatch.setenv('PROVIDER_MAX_REQUESTS_PER_WINDOW', '1')
    _FakeGeminiModels.calls = []
    monkeypatch.setattr(google.genai, 'Client', _FakeGeminiClient)
    get_settings.cache_clear()

    upload_response = client.post(
        '/documents/upload',
        files={
            'file': (
                'payment_terms.txt',
                b'PAYMENT TERMS\nInvoices are due within 30 days of receipt.',
                'text/plain',
            )
        },
    )
    document_id = upload_response.json()['document_id']
    index_response = client.post('/documents/index', json={'document_id': document_id})
    assert index_response.status_code == 200

    response = client.post(
        '/query',
        json={
            'query': 'What are the payment terms?',
            'document_ids': [document_id],
        },
    )
    assert response.status_code == 429
    assert 'request limit reached' in response.json()['detail']

    get_settings.cache_clear()


def test_jwt_users_have_independent_provider_request_budgets(client, monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'jwt')
    monkeypatch.setenv('JWT_AUDIENCES', 'api://doc-analyst')
    monkeypatch.setenv('JWT_SHARED_SECRET', 'dev-secret')
    monkeypatch.setenv('JWT_ALGORITHMS', 'HS256')
    monkeypatch.setenv('GEMINI_API_KEY', 'test-gemini-key')
    monkeypatch.setenv('EMBEDDING_PROVIDER', 'gemini')
    monkeypatch.setenv('LLM_PROVIDER', 'local')
    monkeypatch.setenv('PROVIDER_MAX_REQUESTS_PER_WINDOW', '')
    monkeypatch.setenv('PROVIDER_PER_USER_MAX_REQUESTS_PER_WINDOW', '1')
    _FakeGeminiModels.calls = []
    monkeypatch.setattr(google.genai, 'Client', _FakeGeminiClient)
    get_settings.cache_clear()

    user_a_headers = make_user_headers('user-a')
    user_b_headers = make_user_headers('user-b')

    upload_a1 = client.post(
        '/documents/upload',
        headers=user_a_headers,
        files={'file': ('alpha.txt', b'PAYMENT TERMS\nInvoices are due in 30 days.', 'text/plain')},
    )
    upload_b1 = client.post(
        '/documents/upload',
        headers=user_b_headers,
        files={'file': ('beta.txt', b'PAYMENT TERMS\nInvoices are due in 15 days.', 'text/plain')},
    )
    upload_a2 = client.post(
        '/documents/upload',
        headers=user_a_headers,
        files={'file': ('gamma.txt', b'PAYMENT TERMS\nInvoices are due in 45 days.', 'text/plain')},
    )

    document_a1 = upload_a1.json()['document_id']
    document_b1 = upload_b1.json()['document_id']
    document_a2 = upload_a2.json()['document_id']

    index_a1 = client.post('/documents/index', headers=user_a_headers, json={'document_id': document_a1})
    index_b1 = client.post('/documents/index', headers=user_b_headers, json={'document_id': document_b1})
    index_a2 = client.post('/documents/index', headers=user_a_headers, json={'document_id': document_a2})

    assert index_a1.status_code == 200
    assert index_b1.status_code == 200
    assert index_a2.status_code == 429
    assert 'this authenticated user' in index_a2.json()['detail']
    assert len(_FakeGeminiModels.calls) == 2

    get_settings.cache_clear()


def test_jwt_document_access_is_scoped_to_subject(client, monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'jwt')
    monkeypatch.setenv('JWT_AUDIENCES', 'api://doc-analyst')
    monkeypatch.setenv('JWT_SHARED_SECRET', 'dev-secret')
    monkeypatch.setenv('JWT_ALGORITHMS', 'HS256')
    get_settings.cache_clear()

    user_a_headers = make_user_headers('user-a')
    user_b_headers = make_user_headers('user-b')

    upload_response = client.post(
        '/documents/upload',
        headers=user_a_headers,
        files={
            'file': (
                'private.txt',
                b'PAYMENT TERMS\nInvoices are due within 30 days of receipt.',
                'text/plain',
            )
        },
    )
    assert upload_response.status_code == 201
    document_id = upload_response.json()['document_id']

    index_response = client.post('/documents/index', headers=user_a_headers, json={'document_id': document_id})
    assert index_response.status_code == 200

    list_response = client.get('/documents', headers=user_b_headers)
    assert list_response.status_code == 200
    assert list_response.json() == []

    detail_response = client.get(f'/documents/{document_id}', headers=user_b_headers)
    assert detail_response.status_code == 404

    query_response = client.post(
        '/query',
        headers=user_b_headers,
        json={'query': 'What are the payment terms?', 'document_ids': [document_id]},
    )
    assert query_response.status_code == 404

    get_settings.cache_clear()


def test_jwt_query_without_document_ids_uses_only_current_user_documents(client, monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'jwt')
    monkeypatch.setenv('JWT_AUDIENCES', 'api://doc-analyst')
    monkeypatch.setenv('JWT_SHARED_SECRET', 'dev-secret')
    monkeypatch.setenv('JWT_ALGORITHMS', 'HS256')
    get_settings.cache_clear()

    user_a_headers = make_user_headers('user-a')
    user_b_headers = make_user_headers('user-b')

    upload_a = client.post(
        '/documents/upload',
        headers=user_a_headers,
        files={'file': ('payment.txt', b'PAYMENT TERMS\nInvoices are due in 30 days.', 'text/plain')},
    )
    upload_b = client.post(
        '/documents/upload',
        headers=user_b_headers,
        files={'file': ('renewal.txt', b'RENEWAL\nThe agreement renews each January.', 'text/plain')},
    )
    document_a = upload_a.json()['document_id']
    document_b = upload_b.json()['document_id']
    client.post('/documents/index', headers=user_a_headers, json={'document_id': document_a})
    client.post('/documents/index', headers=user_b_headers, json={'document_id': document_b})

    query_response = client.post(
        '/query',
        headers=user_a_headers,
        json={'query': 'What are the payment terms?'}
    )
    assert query_response.status_code == 200
    payload = query_response.json()
    assert payload['citations']
    assert all(citation['document_id'] == document_a for citation in payload['citations'])
    assert '30 days' in payload['answer']

    get_settings.cache_clear()


def test_jwt_owner_can_delete_document_but_other_user_cannot(client, monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'jwt')
    monkeypatch.setenv('JWT_AUDIENCES', 'api://doc-analyst')
    monkeypatch.setenv('JWT_SHARED_SECRET', 'dev-secret')
    monkeypatch.setenv('JWT_ALGORITHMS', 'HS256')
    get_settings.cache_clear()

    owner_headers = make_user_headers('user-a')
    other_headers = make_user_headers('user-b')

    upload_response = client.post(
        '/documents/upload',
        headers=owner_headers,
        files={'file': ('private.txt', b'PAYMENT TERMS\nInvoices are due in 30 days.', 'text/plain')},
    )
    document_id = upload_response.json()['document_id']
    client.post('/documents/index', headers=owner_headers, json={'document_id': document_id})

    forbidden_delete = client.delete(f'/documents/{document_id}', headers=other_headers)
    owner_delete = client.delete(f'/documents/{document_id}', headers=owner_headers)
    detail_after_delete = client.get(f'/documents/{document_id}', headers=owner_headers)

    assert forbidden_delete.status_code == 404
    assert owner_delete.status_code == 200
    assert owner_delete.json()['status'] == 'deleted'
    assert detail_after_delete.status_code == 404

    get_settings.cache_clear()



def test_jwt_admin_allowlist_can_delete_another_users_document(client, monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'jwt')
    monkeypatch.setenv('JWT_AUDIENCES', 'api://doc-analyst')
    monkeypatch.setenv('JWT_SHARED_SECRET', 'dev-secret')
    monkeypatch.setenv('JWT_ALGORITHMS', 'HS256')
    monkeypatch.setenv('JWT_ADMIN_EMAILS', 'admin@example.com')
    get_settings.cache_clear()

    owner_headers = make_user_headers('user-a')
    now = int(time.time())
    admin_token = make_hs256_token(
        {
            'sub': 'admin-user',
            'aud': 'api://doc-analyst',
            'iss': 'https://issuer.example.com',
            'exp': now + 600,
            'iat': now,
            'preferred_username': 'admin@example.com',
        },
        secret='dev-secret',
    )
    admin_headers = {'Authorization': f'Bearer {admin_token}'}

    upload_response = client.post(
        '/documents/upload',
        headers=owner_headers,
        files={'file': ('shared.txt', b'PAYMENT TERMS\nInvoices are due in 30 days.', 'text/plain')},
    )
    document_id = upload_response.json()['document_id']

    delete_response = client.delete(f'/documents/{document_id}', headers=admin_headers)
    owner_detail = client.get(f'/documents/{document_id}', headers=owner_headers)

    assert delete_response.status_code == 200
    assert owner_detail.status_code == 404

    get_settings.cache_clear()
