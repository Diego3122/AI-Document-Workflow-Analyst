from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.db.sqlite import get_connection
from app.services.auth_context import pop_current_principal, push_current_principal
from app.services.provider_guard_service import ProviderGuardService, ProviderUsageLimitExceeded
from app.services.security_service import AuthPrincipal


def _principal(subject_id: str, *, tenant_id: str | None = 'tenant-a') -> AuthPrincipal:
    return AuthPrincipal(
        auth_type='jwt',
        subject_id=subject_id,
        role='authenticated',
        scopes=('documents.read', 'documents.write', 'query.run'),
        tenant_id=tenant_id,
        issuer='https://issuer.example.com',
    )


def test_provider_guard_blocks_second_request_in_window(monkeypatch) -> None:
    monkeypatch.setenv('PROVIDER_MAX_REQUESTS_PER_WINDOW', '1')
    get_settings.cache_clear()

    service = ProviderGuardService()
    service.check_and_record(provider='gemini', operation='llm', texts=['hello'])

    with pytest.raises(ProviderUsageLimitExceeded, match='request limit reached'):
        service.check_and_record(provider='gemini', operation='llm', texts=['again'])

    get_settings.cache_clear()


def test_provider_guard_blocks_when_estimated_cost_limit_would_be_exceeded(monkeypatch) -> None:
    monkeypatch.setenv('PROVIDER_DAILY_ESTIMATED_COST_LIMIT_USD', '0.01')
    monkeypatch.setenv('LLM_ESTIMATED_COST_PER_1K_CHARS_USD', '0.02')
    get_settings.cache_clear()

    service = ProviderGuardService()

    with pytest.raises(ProviderUsageLimitExceeded, match='estimated cost limit'):
        service.check_and_record(provider='gemini', operation='llm', texts=['x' * 600])

    get_settings.cache_clear()


def test_provider_guard_blocks_same_authenticated_user_after_personal_window_limit(monkeypatch) -> None:
    monkeypatch.setenv('PROVIDER_PER_USER_MAX_REQUESTS_PER_WINDOW', '1')
    get_settings.cache_clear()

    service = ProviderGuardService()
    token = push_current_principal(_principal('user-a'))
    try:
        service.check_and_record(provider='gemini', operation='embedding', texts=['hello'])
        with pytest.raises(ProviderUsageLimitExceeded, match='this authenticated user'):
            service.check_and_record(provider='gemini', operation='embedding', texts=['again'])
    finally:
        pop_current_principal(token)
        get_settings.cache_clear()


def test_provider_guard_allows_different_authenticated_users_to_use_separate_personal_buckets(monkeypatch) -> None:
    monkeypatch.setenv('PROVIDER_PER_USER_MAX_REQUESTS_PER_WINDOW', '1')
    get_settings.cache_clear()

    service = ProviderGuardService()
    token_a = push_current_principal(_principal('user-a'))
    try:
        service.check_and_record(provider='gemini', operation='embedding', texts=['alpha'])
    finally:
        pop_current_principal(token_a)

    token_b = push_current_principal(_principal('user-b'))
    try:
        service.check_and_record(provider='gemini', operation='embedding', texts=['beta'])
    finally:
        pop_current_principal(token_b)

    with get_connection() as connection:
        rows = connection.execute(
            "SELECT subject_key, subject_id FROM provider_usage_events WHERE provider = 'gemini' ORDER BY created_at"
        ).fetchall()
    assert len(rows) == 2
    assert rows[0]['subject_id'] == 'user-a'
    assert rows[1]['subject_id'] == 'user-b'
    assert rows[0]['subject_key'] != rows[1]['subject_key']

    get_settings.cache_clear()


def test_provider_guard_global_limit_still_applies_across_users(monkeypatch) -> None:
    monkeypatch.setenv('PROVIDER_MAX_REQUESTS_PER_WINDOW', '1')
    monkeypatch.setenv('PROVIDER_PER_USER_MAX_REQUESTS_PER_WINDOW', '2')
    get_settings.cache_clear()

    service = ProviderGuardService()
    token_a = push_current_principal(_principal('user-a'))
    try:
        service.check_and_record(provider='gemini', operation='llm', texts=['alpha'])
    finally:
        pop_current_principal(token_a)

    token_b = push_current_principal(_principal('user-b'))
    try:
        with pytest.raises(ProviderUsageLimitExceeded, match='current 60-second window'):
            service.check_and_record(provider='gemini', operation='llm', texts=['beta'])
    finally:
        pop_current_principal(token_b)
        get_settings.cache_clear()
