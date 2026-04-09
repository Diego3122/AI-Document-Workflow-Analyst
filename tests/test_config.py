from __future__ import annotations

import pytest

from app.core.config import get_settings


def test_settings_prefer_gemini_provider_when_key_present(monkeypatch) -> None:
    monkeypatch.setenv('GEMINI_API_KEY', 'test-gemini-key')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('LLM_PROVIDER', raising=False)
    monkeypatch.delenv('EMBEDDING_PROVIDER', raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.resolve_llm_provider() == 'gemini'
    assert settings.resolve_embedding_provider() == 'gemini'
    assert settings.resolved_llm_model() == 'gemini-2.5-flash'
    assert settings.resolved_embedding_model() == 'gemini-embedding-001'

    get_settings.cache_clear()


def test_provider_limit_settings_load_from_environment(monkeypatch) -> None:
    monkeypatch.setenv('PROVIDER_MAX_REQUESTS_PER_WINDOW', '2')
    monkeypatch.setenv('PROVIDER_DAILY_REQUEST_LIMIT', '5')
    monkeypatch.setenv('PROVIDER_DAILY_INPUT_CHAR_LIMIT', '2500')
    monkeypatch.setenv('PROVIDER_DAILY_ESTIMATED_COST_LIMIT_USD', '0.15')
    monkeypatch.setenv('LLM_ESTIMATED_COST_PER_1K_CHARS_USD', '0.02')
    monkeypatch.setenv('EMBEDDING_ESTIMATED_COST_PER_1K_CHARS_USD', '0.01')
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.provider_max_requests_per_window == 2
    assert settings.provider_daily_request_limit == 5
    assert settings.provider_daily_input_char_limit == 2500
    assert settings.provider_daily_estimated_cost_limit_usd == 0.15
    assert settings.llm_estimated_cost_per_1k_chars_usd == 0.02
    assert settings.embedding_estimated_cost_per_1k_chars_usd == 0.01

    get_settings.cache_clear()


def test_auth_settings_parse_api_key_lists(monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'api_key')
    monkeypatch.setenv('ADMIN_API_KEYS', 'admin-one, admin-two')
    monkeypatch.setenv('ANALYST_API_KEYS', 'analyst-one')
    monkeypatch.setenv('UI_API_KEY', 'admin-one')
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.auth_is_enabled() is True
    assert settings.resolved_auth_mode() == 'api_key'
    assert settings.admin_api_keys == ['admin-one', 'admin-two']
    assert settings.analyst_api_keys == ['analyst-one']
    assert settings.ui_api_key == 'admin-one'
    assert settings.api_key_role_map()['admin-one'] == 'admin'
    assert settings.api_key_role_map()['analyst-one'] == 'analyst'
    assert settings.ui_api_key_role() == 'admin'

    get_settings.cache_clear()


def test_jwt_settings_parse_lists(monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'jwt')
    monkeypatch.setenv('JWT_AUDIENCES', 'api://doc-analyst, another-audience')
    monkeypatch.setenv('JWT_ALGORITHMS', 'HS256')
    monkeypatch.setenv('JWT_SHARED_SECRET', 'dev-secret')
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.jwt_is_enabled() is True
    assert settings.jwt_audiences == ['api://doc-analyst', 'another-audience']
    assert settings.jwt_algorithms == ['HS256']
    settings.validate_jwt_settings()

    get_settings.cache_clear()


def test_production_settings_require_explicit_trusted_hosts(monkeypatch) -> None:
    monkeypatch.setenv('ENVIRONMENT', 'production')
    monkeypatch.setenv('AUTH_MODE', 'api_key')
    monkeypatch.setenv('ADMIN_API_KEYS', 'admin-one')
    monkeypatch.setenv('TRUSTED_HOSTS', '')
    get_settings.cache_clear()

    settings = get_settings()

    with pytest.raises(ValueError, match='TRUSTED_HOSTS'):
        settings.validate_production_settings()

    get_settings.cache_clear()


def test_public_ui_mode_requires_jwt_without_shared_key(monkeypatch) -> None:
    monkeypatch.setenv('ENVIRONMENT', 'production')
    monkeypatch.setenv('AUTH_MODE', 'api_key')
    monkeypatch.setenv('ADMIN_API_KEYS', 'admin-one')
    monkeypatch.setenv('TRUSTED_HOSTS', 'testserver')
    monkeypatch.setenv('UI_PUBLIC_ENABLED', 'true')
    monkeypatch.setenv('UI_API_KEY', 'admin-one')
    get_settings.cache_clear()

    settings = get_settings()

    with pytest.raises(ValueError, match='AUTH_MODE=jwt'):
        settings.validate_production_settings()

    get_settings.cache_clear()


def test_jwt_mode_requires_audience_and_secret_or_metadata(monkeypatch) -> None:
    monkeypatch.setenv('AUTH_MODE', 'jwt')
    monkeypatch.setenv('JWT_AUDIENCES', '')
    monkeypatch.setenv('JWT_SHARED_SECRET', '')
    monkeypatch.setenv('JWT_ISSUER', '')
    monkeypatch.setenv('JWT_JWKS_URL', '')
    get_settings.cache_clear()

    settings = get_settings()

    with pytest.raises(ValueError, match='JWT_AUDIENCES'):
        settings.validate_jwt_settings()

    get_settings.cache_clear()


def test_production_settings_use_app_service_storage_defaults_on_linux(monkeypatch) -> None:

    import app.core.config as config_module

    monkeypatch.setenv('ENVIRONMENT', 'production')
    monkeypatch.setenv('AUTH_MODE', 'api_key')
    monkeypatch.setenv('ADMIN_API_KEYS', 'admin-one')
    monkeypatch.setenv('TRUSTED_HOSTS', 'testserver')
    monkeypatch.setenv('FORCE_HTTPS', 'false')
    monkeypatch.delenv('UPLOADS_DIR', raising=False)
    monkeypatch.delenv('CHROMA_DIR', raising=False)
    monkeypatch.delenv('LOGS_DIR', raising=False)
    monkeypatch.delenv('EVALS_DIR', raising=False)
    monkeypatch.delenv('SQLITE_PATH', raising=False)
    monkeypatch.delenv('SAMPLE_EVAL_DATASET_PATH', raising=False)
    monkeypatch.setattr(config_module.os, 'name', 'posix', raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert str(settings.uploads_dir).replace('\\', '/') == '/home/site/data/uploads'
    assert str(settings.chroma_dir).replace('\\', '/') == '/home/site/data/chroma'
    assert str(settings.logs_dir).replace('\\', '/') == '/home/site/data/logs'
    assert str(settings.sqlite_path).replace('\\', '/') == '/home/site/data/logs/app.db'

    get_settings.cache_clear()



