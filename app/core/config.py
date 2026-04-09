from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ENVIRONMENTS = {'prod', 'production', 'release'}
DEVELOPMENT_ENVIRONMENTS = {'dev', 'development', 'debug', 'local'}
AUTH_MODE_VALUES = {'disabled', 'api_key', 'jwt', 'hybrid'}
DEFAULT_UPLOADS_DIR = PROJECT_ROOT / 'data' / 'uploads'
DEFAULT_CHROMA_DIR = PROJECT_ROOT / 'data' / 'chroma'
DEFAULT_LOGS_DIR = PROJECT_ROOT / 'data' / 'logs'
DEFAULT_EVALS_DIR = PROJECT_ROOT / 'data' / 'evals'
DEFAULT_SQLITE_PATH = DEFAULT_LOGS_DIR / 'app.db'
DEFAULT_SAMPLE_EVAL_DATASET_PATH = DEFAULT_EVALS_DIR / 'sample_eval_dataset.jsonl'
APP_SERVICE_DATA_ROOT = Path('/home/site/data')
APP_SERVICE_UPLOADS_DIR = APP_SERVICE_DATA_ROOT / 'uploads'
APP_SERVICE_CHROMA_DIR = APP_SERVICE_DATA_ROOT / 'chroma'
APP_SERVICE_LOGS_DIR = APP_SERVICE_DATA_ROOT / 'logs'
APP_SERVICE_EVALS_DIR = APP_SERVICE_DATA_ROOT / 'evals'
APP_SERVICE_SQLITE_PATH = APP_SERVICE_LOGS_DIR / 'app.db'
APP_SERVICE_SAMPLE_EVAL_DATASET_PATH = APP_SERVICE_EVALS_DIR / 'sample_eval_dataset.jsonl'
JWT_DEFAULT_ROLE_VALUES = {'authenticated', 'reader', 'analyst', 'reviewer', 'admin'}
LOOPBACK_HOSTS = {'127.0.0.1', 'localhost', '::1'}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    app_name: str = 'AI Document Workflow Analyst'
    environment: str = 'development'
    debug: bool = False
    uploads_dir: Path = Field(default=DEFAULT_UPLOADS_DIR)
    chroma_dir: Path = Field(default=DEFAULT_CHROMA_DIR)
    logs_dir: Path = Field(default=DEFAULT_LOGS_DIR)
    evals_dir: Path = Field(default=DEFAULT_EVALS_DIR)
    sqlite_path: Path = Field(default=DEFAULT_SQLITE_PATH)
    sample_eval_dataset_path: Path = Field(default=DEFAULT_SAMPLE_EVAL_DATASET_PATH)
    document_retention_days: int | None = Field(default=None, ge=1)
    ui_api_base_url: str = 'http://localhost:8000'
    ui_api_key: str | None = None
    ui_public_enabled: bool = False
    ui_forwarded_access_token_headers: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ['X-Ms-Token-Aad-Access-Token', 'Authorization']
    )
    enable_api_docs: bool = False
    cors_allowed_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    trusted_hosts: Annotated[list[str], NoDecode] = Field(default_factory=list)
    trust_proxy_headers: bool = False
    proxy_trusted_ips: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ['127.0.0.1'])
    force_https: bool | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    llm_provider: str = 'auto'
    embedding_provider: str = 'auto'
    llm_model: str | None = None
    embedding_model: str | None = None
    openai_llm_model: str = 'gpt-4.1-mini'
    openai_embedding_model: str = 'text-embedding-3-large'
    gemini_llm_model: str = 'gemini-2.5-flash'
    gemini_embedding_model: str = 'gemini-embedding-001'
    embedding_dimensions: int = 128
    vector_backend: str = 'auto'
    provider_rate_limit_window_seconds: int = 60
    provider_max_requests_per_window: int | None = None
    provider_daily_request_limit: int | None = None
    provider_daily_input_char_limit: int | None = None
    provider_daily_estimated_cost_limit_usd: float | None = None
    provider_per_user_max_requests_per_window: int | None = None
    provider_per_user_daily_request_limit: int | None = None
    provider_per_user_daily_input_char_limit: int | None = None
    provider_per_user_daily_estimated_cost_limit_usd: float | None = None
    llm_estimated_cost_per_1k_chars_usd: float = 0.0
    embedding_estimated_cost_per_1k_chars_usd: float = 0.0
    auth_mode: str = 'disabled'
    api_auth_enabled: bool = False
    reader_api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    analyst_api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    reviewer_api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    admin_api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    jwt_issuer: str | None = None
    jwt_audiences: Annotated[list[str], NoDecode] = Field(default_factory=list)
    jwt_jwks_url: str | None = None
    jwt_shared_secret: str | None = None
    jwt_algorithms: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ['RS256'])
    jwt_role_claim: str = 'roles'
    jwt_scope_claim: str = 'scp'
    jwt_subject_claim: str = 'sub'
    jwt_tenant_claim: str = 'tid'
    jwt_name_claim: str = 'preferred_username'
    jwt_clock_skew_seconds: int = 60
    jwt_require_https_metadata: bool = True
    jwt_default_role: str = 'analyst'
    jwt_admin_subjects: Annotated[list[str], NoDecode] = Field(default_factory=list)
    jwt_admin_emails: Annotated[list[str], NoDecode] = Field(default_factory=list)
    api_rate_limit_window_seconds: int = 60
    api_rate_limit_requests_per_ip: int | None = 60
    api_rate_limit_requests_per_user: int | None = 120
    max_upload_size_bytes: int = 10 * 1024 * 1024
    retrieval_top_k: int = 5
    answer_confidence_threshold: float = 0.75
    human_review_threshold: float = 0.50

    @field_validator('environment', mode='before')
    @classmethod
    def normalize_environment(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in PRODUCTION_ENVIRONMENTS:
                return 'production'
            if normalized in DEVELOPMENT_ENVIRONMENTS:
                return 'development'
            return normalized
        return value

    @field_validator('debug', mode='before')
    @classmethod
    def normalize_debug_flag(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {'release', 'prod', 'production', 'false', '0', 'no', 'off'}:
                return False
            if normalized in {'debug', 'dev', 'development', 'true', '1', 'yes', 'on'}:
                return True
        return value

    @field_validator('auth_mode', mode='before')
    @classmethod
    def normalize_auth_mode(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return 'disabled'
            if normalized not in AUTH_MODE_VALUES:
                raise ValueError('AUTH_MODE must be one of disabled, api_key, jwt, or hybrid.')
            return normalized
        return value

    @field_validator(
        'provider_max_requests_per_window',
        'provider_daily_request_limit',
        'provider_daily_input_char_limit',
        'provider_daily_estimated_cost_limit_usd',
        'provider_per_user_max_requests_per_window',
        'provider_per_user_daily_request_limit',
        'provider_per_user_daily_input_char_limit',
        'provider_per_user_daily_estimated_cost_limit_usd',
        'document_retention_days',
        'api_rate_limit_requests_per_ip',
        'api_rate_limit_requests_per_user',
        mode='before',
    )
    @classmethod
    def normalize_optional_numeric_settings(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator('force_https', mode='before')
    @classmethod
    def normalize_optional_bool(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return None
            if normalized in {'true', '1', 'yes', 'on'}:
                return True
            if normalized in {'false', '0', 'no', 'off'}:
                return False
        return value

    @field_validator(
        'ui_api_key',
        'jwt_issuer',
        'jwt_jwks_url',
        'jwt_shared_secret',
        mode='before',
    )
    @classmethod
    def normalize_optional_string(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator('jwt_default_role', mode='before')
    @classmethod
    def normalize_jwt_default_role(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return 'analyst'
            if normalized not in JWT_DEFAULT_ROLE_VALUES:
                raise ValueError(
                    'JWT_DEFAULT_ROLE must be one of authenticated, reader, analyst, reviewer, or admin.'
                )
            return normalized
        return value

    @field_validator(
        'reader_api_keys',
        'analyst_api_keys',
        'reviewer_api_keys',
        'admin_api_keys',
        'cors_allowed_origins',
        'trusted_hosts',
        'proxy_trusted_ips',
        'jwt_audiences',
        'jwt_algorithms',
        'jwt_admin_subjects',
        'jwt_admin_emails',
        'ui_forwarded_access_token_headers',
        mode='before',
    )
    @classmethod
    def normalize_csv_lists(cls, value: object) -> list[str] | object:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return value

    @field_validator('llm_provider', 'embedding_provider', mode='before')
    @classmethod
    def normalize_provider(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            aliases = {
                'google': 'gemini',
                'google_genai': 'gemini',
            }
            return aliases.get(normalized, normalized)
        return value

    @model_validator(mode='after')
    def apply_production_storage_defaults(self) -> Settings:
        if not self.is_production() or os.name == 'nt':
            return self

        if self.uploads_dir == DEFAULT_UPLOADS_DIR:
            self.uploads_dir = APP_SERVICE_UPLOADS_DIR
        if self.chroma_dir == DEFAULT_CHROMA_DIR:
            self.chroma_dir = APP_SERVICE_CHROMA_DIR
        if self.logs_dir == DEFAULT_LOGS_DIR:
            self.logs_dir = APP_SERVICE_LOGS_DIR
        if self.evals_dir == DEFAULT_EVALS_DIR:
            self.evals_dir = APP_SERVICE_EVALS_DIR
        if self.sqlite_path == DEFAULT_SQLITE_PATH:
            self.sqlite_path = APP_SERVICE_SQLITE_PATH
        if self.sample_eval_dataset_path == DEFAULT_SAMPLE_EVAL_DATASET_PATH:
            self.sample_eval_dataset_path = APP_SERVICE_SAMPLE_EVAL_DATASET_PATH
        return self

    def is_production(self) -> bool:
        return self.environment == 'production'

    def docs_are_enabled(self) -> bool:
        return self.enable_api_docs and not self.is_production() and not self.auth_is_enabled()

    def force_https_enabled(self) -> bool:
        if self.force_https is None:
            return self.is_production()
        return self.force_https

    def ui_api_hostname(self) -> str | None:
        parsed = urlparse(self.ui_api_base_url)
        hostname = parsed.hostname
        if hostname is None:
            return None
        cleaned = hostname.strip().lower()
        return cleaned or None

    def internal_api_loopback_hosts(self) -> list[str]:
        hostname = self.ui_api_hostname()
        if hostname not in LOOPBACK_HOSTS:
            return []

        hosts = ['127.0.0.1', 'localhost']
        if hostname == '::1':
            hosts.append('::1')
        return hosts

    def resolved_trusted_hosts(self) -> list[str]:
        if not self.trusted_hosts:
            return []

        allowed_hosts = list(self.trusted_hosts)
        for host in self.internal_api_loopback_hosts():
            if host not in allowed_hosts:
                allowed_hosts.append(host)
        return allowed_hosts

    def api_key_role_map(self) -> dict[str, str]:
        role_map: dict[str, str] = {}
        for role, keys in (
            ('reader', self.reader_api_keys),
            ('analyst', self.analyst_api_keys),
            ('reviewer', self.reviewer_api_keys),
            ('admin', self.admin_api_keys),
        ):
            for key in keys:
                role_map[key] = role
        return role_map

    def resolved_auth_mode(self) -> str:
        if self.auth_mode != 'disabled':
            return self.auth_mode
        if self.api_auth_enabled or bool(self.api_key_role_map()):
            return 'api_key'
        return 'disabled'

    def api_key_auth_is_enabled(self) -> bool:
        return self.resolved_auth_mode() in {'api_key', 'hybrid'} and bool(self.api_key_role_map())

    def jwt_is_enabled(self) -> bool:
        return self.resolved_auth_mode() in {'jwt', 'hybrid'}

    def auth_is_enabled(self) -> bool:
        return self.resolved_auth_mode() != 'disabled'

    def ui_api_key_role(self) -> str | None:
        if not self.ui_api_key or not self.api_key_auth_is_enabled():
            return None
        return self.api_key_role_map().get(self.ui_api_key)

    def streamlit_can_upload_and_query(self) -> bool:
        if not self.auth_is_enabled():
            return True
        return self.ui_api_key_role() in {'analyst', 'admin'}

    def streamlit_can_review(self) -> bool:
        if self.ui_public_enabled:
            return False
        return self.ui_api_key_role() in {'reviewer', 'admin'}

    def streamlit_can_view_metrics(self) -> bool:
        if self.ui_public_enabled:
            return False
        return self.ui_api_key_role() == 'admin'

    def validate_jwt_settings(self) -> None:
        if not self.jwt_is_enabled():
            return

        issues: list[str] = []
        if not self.jwt_audiences:
            issues.append('JWT_AUDIENCES must contain at least one accepted audience.')
        if not self.jwt_shared_secret and not (self.jwt_issuer or self.jwt_jwks_url):
            issues.append('JWT auth requires JWT_SHARED_SECRET or JWT_ISSUER/JWT_JWKS_URL.')
        if self.jwt_shared_secret and self.jwt_jwks_url and self.resolved_auth_mode() == 'jwt':
            issues.append('JWT auth should use either shared-secret validation or JWKS validation, not both.')
        if not self.jwt_algorithms:
            issues.append('JWT_ALGORITHMS must contain at least one signing algorithm.')
        if self.jwt_require_https_metadata:
            for label, value in (('JWT_ISSUER', self.jwt_issuer), ('JWT_JWKS_URL', self.jwt_jwks_url)):
                if value and not value.startswith('https://'):
                    issues.append(f'{label} must use HTTPS when JWT_REQUIRE_HTTPS_METADATA is enabled.')

        if issues:
            raise ValueError('Unsafe JWT settings: ' + ' '.join(issues))

    def validate_public_ui_settings(self) -> None:
        if not self.ui_public_enabled:
            return

        issues: list[str] = []
        if not self.auth_is_enabled():
            issues.append('UI_PUBLIC_ENABLED requires authentication to be enabled.')
        if self.resolved_auth_mode() != 'jwt':
            issues.append('UI_PUBLIC_ENABLED requires AUTH_MODE=jwt so each user has a validated identity.')
        if self.ui_api_key:
            issues.append('UI_PUBLIC_ENABLED cannot rely on UI_API_KEY because shared keys collapse all users together.')
        if not self.ui_forwarded_access_token_headers:
            issues.append('UI_PUBLIC_ENABLED requires at least one forwarded access-token header name.')

        if issues:
            raise ValueError('Unsafe public UI settings: ' + ' '.join(issues))

    def validate_production_settings(self) -> None:
        if not self.is_production():
            return

        issues: list[str] = []
        if not self.auth_is_enabled():
            issues.append('API authentication must be enabled in production.')
        if self.api_key_auth_is_enabled() and not self.admin_api_keys:
            issues.append('At least one admin API key must be configured when API-key auth is enabled in production.')
        if not self.trusted_hosts or '*' in self.trusted_hosts:
            issues.append('TRUSTED_HOSTS must contain explicit hostnames in production.')
        if '*' in self.cors_allowed_origins:
            issues.append('CORS_ALLOWED_ORIGINS cannot contain * in production.')

        try:
            self.validate_jwt_settings()
        except ValueError as exc:
            issues.append(str(exc))

        try:
            self.validate_public_ui_settings()
        except ValueError as exc:
            issues.append(str(exc))

        if issues:
            raise ValueError('Unsafe production settings: ' + ' '.join(issues))

    def resolve_llm_provider(self) -> str:
        if self.llm_provider != 'auto':
            if self.llm_provider == 'gemini' and self.gemini_api_key:
                return 'gemini'
            if self.llm_provider == 'openai' and self.openai_api_key:
                return 'openai'
            return 'local'

        if self.gemini_api_key:
            return 'gemini'
        if self.openai_api_key:
            return 'openai'
        return 'local'

    def resolve_embedding_provider(self) -> str:
        if self.embedding_provider != 'auto':
            if self.embedding_provider == 'gemini' and self.gemini_api_key:
                return 'gemini'
            if self.embedding_provider == 'openai' and self.openai_api_key:
                return 'openai'
            return 'local'

        if self.gemini_api_key:
            return 'gemini'
        if self.openai_api_key:
            return 'openai'
        return 'local'

    def resolved_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        provider = self.resolve_llm_provider()
        if provider == 'gemini':
            return self.gemini_llm_model
        if provider == 'openai':
            return self.openai_llm_model
        return 'local_fallback'

    def resolved_embedding_model(self) -> str:
        if self.embedding_model:
            return self.embedding_model
        provider = self.resolve_embedding_provider()
        if provider == 'gemini':
            return self.gemini_embedding_model
        if provider == 'openai':
            return self.openai_embedding_model
        return 'local_hash'

    def ensure_directories(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.evals_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
