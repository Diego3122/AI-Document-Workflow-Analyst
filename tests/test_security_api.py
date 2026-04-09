from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.core.config import get_settings


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')


def make_hs256_token(claims: dict[str, object], *, secret: str, kid: str | None = None) -> str:
    header = {'alg': 'HS256', 'typ': 'JWT'}
    if kid:
        header['kid'] = kid
    encoded_header = _b64url(json.dumps(header, separators=(',', ':')).encode('utf-8'))
    encoded_payload = _b64url(json.dumps(claims, separators=(',', ':')).encode('utf-8'))
    signing_input = f'{encoded_header}.{encoded_payload}'.encode('ascii')
    signature = hmac.new(secret.encode('utf-8'), signing_input, hashlib.sha256).digest()
    return f'{encoded_header}.{encoded_payload}.{_b64url(signature)}'


def test_health_requires_api_key_when_auth_enabled(client, monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'api_key')
    monkeypatch.setenv('ADMIN_API_KEYS', 'admin-key')
    get_settings.cache_clear()

    response = client.get('/health')

    assert response.status_code == 401
    assert 'API key' in response.json()['detail']

    get_settings.cache_clear()


def test_missing_api_key_attempts_are_rate_limited_by_ip(client, monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'api_key')
    monkeypatch.setenv('ADMIN_API_KEYS', 'admin-key')
    monkeypatch.setenv('API_RATE_LIMIT_REQUESTS_PER_IP', '1')
    monkeypatch.setenv('API_RATE_LIMIT_REQUESTS_PER_USER', '')
    get_settings.cache_clear()

    first = client.get('/health')
    second = client.get('/health')

    assert first.status_code == 401
    assert second.status_code == 429
    assert 'IP address' in second.json()['detail']

    get_settings.cache_clear()


def test_metrics_requires_admin_role(client, monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'api_key')
    monkeypatch.setenv('ANALYST_API_KEYS', 'analyst-key')
    monkeypatch.setenv('ADMIN_API_KEYS', 'admin-key')
    get_settings.cache_clear()

    response = client.get('/metrics', headers={'X-API-Key': 'analyst-key'})

    assert response.status_code == 403
    assert 'not authorized' in response.json()['detail']

    get_settings.cache_clear()


def test_rate_limiting_blocks_second_request_for_same_api_key(client, monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'api_key')
    monkeypatch.setenv('READER_API_KEYS', 'reader-key')
    monkeypatch.setenv('API_RATE_LIMIT_REQUESTS_PER_IP', '')
    monkeypatch.setenv('API_RATE_LIMIT_REQUESTS_PER_USER', '1')
    get_settings.cache_clear()

    first = client.get('/health', headers={'X-API-Key': 'reader-key'})
    second = client.get('/health', headers={'X-API-Key': 'reader-key'})

    assert first.status_code == 200
    assert second.status_code == 429
    assert 'API key' in second.json()['detail']

    get_settings.cache_clear()


def test_health_requires_bearer_token_when_jwt_enabled(client, monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'jwt')
    monkeypatch.setenv('JWT_AUDIENCES', 'api://doc-analyst')
    monkeypatch.setenv('JWT_SHARED_SECRET', 'dev-secret')
    monkeypatch.setenv('JWT_ALGORITHMS', 'HS256')
    get_settings.cache_clear()

    response = client.get('/health')

    assert response.status_code == 401
    assert 'bearer token' in response.json()['detail'].lower()

    get_settings.cache_clear()


def test_valid_jwt_with_scope_can_access_auth_me_and_health(client, monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'jwt')
    monkeypatch.setenv('JWT_AUDIENCES', 'api://doc-analyst')
    monkeypatch.setenv('JWT_SHARED_SECRET', 'dev-secret')
    monkeypatch.setenv('JWT_ALGORITHMS', 'HS256')
    get_settings.cache_clear()

    now = int(time.time())
    token = make_hs256_token(
        {
            'sub': 'user-123',
            'aud': 'api://doc-analyst',
            'iss': 'https://issuer.example.com',
            'exp': now + 600,
            'iat': now,
            'scp': 'health.read documents.read query.run documents.write',
            'preferred_username': 'alex@example.com',
        },
        secret='dev-secret',
    )
    headers = {'Authorization': f'Bearer {token}'}

    health = client.get('/health', headers=headers)
    me = client.get('/auth/me', headers=headers)

    assert health.status_code == 200
    assert me.status_code == 200
    payload = me.json()
    assert payload['subject_id'] == 'user-123'
    assert payload['auth_type'] == 'jwt'
    assert payload['can_upload_and_query'] is True
    assert payload['display_name'] == 'alex@example.com'

    get_settings.cache_clear()


def test_jwt_with_wrong_audience_is_rejected(client, monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'jwt')
    monkeypatch.setenv('JWT_AUDIENCES', 'api://doc-analyst')
    monkeypatch.setenv('JWT_SHARED_SECRET', 'dev-secret')
    monkeypatch.setenv('JWT_ALGORITHMS', 'HS256')
    get_settings.cache_clear()

    now = int(time.time())
    token = make_hs256_token(
        {
            'sub': 'user-123',
            'aud': 'api://wrong-audience',
            'iss': 'https://issuer.example.com',
            'exp': now + 600,
            'iat': now,
            'scp': 'health.read',
        },
        secret='dev-secret',
    )

    response = client.get('/health', headers={'Authorization': f'Bearer {token}'})

    assert response.status_code == 401
    assert 'audience' in response.json()['detail']

    get_settings.cache_clear()


def test_evaluation_rejects_dataset_path_outside_evals_dir(client, monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'api_key')
    monkeypatch.setenv('ADMIN_API_KEYS', 'admin-key')
    get_settings.cache_clear()

    response = client.post(
        '/metrics/evaluate',
        json={'dataset_path': r'..\outside.jsonl'},
        headers={'X-API-Key': 'admin-key'},
    )

    assert response.status_code == 400
    assert 'configured evals directory' in response.json()['detail']

    get_settings.cache_clear()


def test_livez_is_public_in_production(monkeypatch) -> None:
    monkeypatch.setenv('ENVIRONMENT', 'production')
    monkeypatch.setenv('AUTH_MODE', 'api_key')
    monkeypatch.setenv('ADMIN_API_KEYS', 'admin-key')
    monkeypatch.setenv('TRUSTED_HOSTS', 'testserver')
    monkeypatch.setenv('FORCE_HTTPS', 'false')
    get_settings.cache_clear()

    with TestClient(create_app()) as secured_client:
        response = secured_client.get('/livez')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}

    get_settings.cache_clear()


def test_docs_are_disabled_in_production_even_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv('ENVIRONMENT', 'production')
    monkeypatch.setenv('AUTH_MODE', 'api_key')
    monkeypatch.setenv('ADMIN_API_KEYS', 'admin-key')
    monkeypatch.setenv('TRUSTED_HOSTS', 'testserver')
    monkeypatch.setenv('ENABLE_API_DOCS', 'true')
    monkeypatch.setenv('FORCE_HTTPS', 'false')
    get_settings.cache_clear()

    with TestClient(create_app()) as secured_client:
        response = secured_client.get('/docs')

    assert response.status_code == 404

    get_settings.cache_clear()


def test_trusted_hosts_block_unexpected_host(monkeypatch) -> None:
    monkeypatch.setenv('ENVIRONMENT', 'production')
    monkeypatch.setenv('AUTH_MODE', 'api_key')
    monkeypatch.setenv('ADMIN_API_KEYS', 'admin-key')
    monkeypatch.setenv('TRUSTED_HOSTS', 'testserver')
    monkeypatch.setenv('FORCE_HTTPS', 'false')
    get_settings.cache_clear()

    with TestClient(create_app()) as secured_client:
        response = secured_client.get('/livez', headers={'host': 'evil.example.com'})

    assert response.status_code == 400

    get_settings.cache_clear()


def test_loopback_ui_host_is_allowed_even_when_not_explicitly_listed(monkeypatch) -> None:
    monkeypatch.setenv('ENVIRONMENT', 'production')
    monkeypatch.setenv('AUTH_MODE', 'api_key')
    monkeypatch.setenv('ADMIN_API_KEYS', 'admin-key')
    monkeypatch.setenv('TRUSTED_HOSTS', 'portal.example.com')
    monkeypatch.setenv('UI_API_BASE_URL', 'http://127.0.0.1:8000')
    monkeypatch.setenv('FORCE_HTTPS', 'false')
    get_settings.cache_clear()

    with TestClient(create_app()) as secured_client:
        response = secured_client.get('/livez', headers={'host': '127.0.0.1:8000'})

    assert response.status_code == 200

    get_settings.cache_clear()


def test_force_https_redirect_skips_loopback_ui_host(monkeypatch) -> None:
    monkeypatch.setenv('ENVIRONMENT', 'production')
    monkeypatch.setenv('AUTH_MODE', 'api_key')
    monkeypatch.setenv('ADMIN_API_KEYS', 'admin-key')
    monkeypatch.setenv('TRUSTED_HOSTS', 'portal.example.com')
    monkeypatch.setenv('UI_API_BASE_URL', 'http://127.0.0.1:8000')
    get_settings.cache_clear()

    with TestClient(create_app()) as secured_client:
        loopback_response = secured_client.get(
            '/livez',
            headers={'host': '127.0.0.1:8000'},
            follow_redirects=False,
        )
        public_response = secured_client.get(
            '/livez',
            headers={'host': 'portal.example.com'},
            follow_redirects=False,
        )

    assert loopback_response.status_code == 200
    assert public_response.status_code == 307
    assert public_response.headers['location'] == 'https://portal.example.com/livez'

    get_settings.cache_clear()


def test_cors_allows_authorization_header(monkeypatch) -> None:
    monkeypatch.setenv('ENVIRONMENT', 'production')
    monkeypatch.setenv('AUTH_MODE', 'jwt')
    monkeypatch.setenv('JWT_AUDIENCES', 'api://doc-analyst')
    monkeypatch.setenv('JWT_SHARED_SECRET', 'dev-secret')
    monkeypatch.setenv('JWT_ALGORITHMS', 'HS256')
    monkeypatch.setenv('TRUSTED_HOSTS', 'testserver')
    monkeypatch.setenv('CORS_ALLOWED_ORIGINS', 'https://portal.example.com')
    monkeypatch.setenv('FORCE_HTTPS', 'false')
    monkeypatch.setenv('JWT_REQUIRE_HTTPS_METADATA', 'false')
    get_settings.cache_clear()

    with TestClient(create_app()) as secured_client:
        response = secured_client.options(
            '/livez',
            headers={
                'Origin': 'https://portal.example.com',
                'Access-Control-Request-Method': 'GET',
                'Access-Control-Request-Headers': 'Authorization',
                'host': 'testserver',
            },
        )

    assert response.status_code == 200
    assert response.headers['access-control-allow-origin'] == 'https://portal.example.com'
    assert 'Authorization' in response.headers['access-control-allow-headers']

    get_settings.cache_clear()


def test_upload_rejects_path_like_filenames(client) -> None:
    response = client.post(
        '/documents/upload',
        files={'file': (r'..\secret.txt', b'hello', 'text/plain')},
    )

    assert response.status_code == 400
    assert 'directory paths' in response.json()['detail']


def test_upload_rejects_invalid_pdf_content(client) -> None:
    response = client.post(
        '/documents/upload',
        files={'file': ('fake.pdf', b'not-a-real-pdf', 'application/pdf')},
    )

    assert response.status_code == 400
    assert 'valid PDF header' in response.json()['detail']


def test_jwt_without_roles_or_scopes_defaults_to_basic_analyst_access(client, monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'jwt')
    monkeypatch.setenv('JWT_AUDIENCES', 'api://doc-analyst')
    monkeypatch.setenv('JWT_SHARED_SECRET', 'dev-secret')
    monkeypatch.setenv('JWT_ALGORITHMS', 'HS256')
    get_settings.cache_clear()

    now = int(time.time())
    token = make_hs256_token(
        {
            'sub': 'user-456',
            'aud': 'api://doc-analyst',
            'iss': 'https://issuer.example.com',
            'exp': now + 600,
            'iat': now,
            'preferred_username': 'portfolio-user@example.com',
        },
        secret='dev-secret',
    )
    headers = {'Authorization': f'Bearer {token}'}

    me = client.get('/auth/me', headers=headers)
    upload = client.post(
        '/documents/upload',
        headers=headers,
        files={'file': ('notes.txt', b'PAYMENT TERMS\nInvoices are due in 30 days.', 'text/plain')},
    )

    assert me.status_code == 200
    payload = me.json()
    assert payload['role'] == 'analyst'
    assert payload['can_upload_and_query'] is True
    assert payload['can_review'] is False
    assert upload.status_code == 201

    get_settings.cache_clear()



def test_jwt_admin_email_allowlist_grants_admin_access(client, monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'jwt')
    monkeypatch.setenv('JWT_AUDIENCES', 'api://doc-analyst')
    monkeypatch.setenv('JWT_SHARED_SECRET', 'dev-secret')
    monkeypatch.setenv('JWT_ALGORITHMS', 'HS256')
    monkeypatch.setenv('JWT_ADMIN_EMAILS', 'alex@example.com')
    get_settings.cache_clear()

    now = int(time.time())
    token = make_hs256_token(
        {
            'sub': 'admin-user',
            'aud': 'api://doc-analyst',
            'iss': 'https://issuer.example.com',
            'exp': now + 600,
            'iat': now,
            'preferred_username': 'alex@example.com',
        },
        secret='dev-secret',
    )
    headers = {'Authorization': f'Bearer {token}'}

    me = client.get('/auth/me', headers=headers)
    metrics = client.get('/metrics', headers=headers)

    assert me.status_code == 200
    payload = me.json()
    assert payload['role'] == 'admin'
    assert payload['can_view_metrics'] is True
    assert metrics.status_code == 200

    get_settings.cache_clear()
