from __future__ import annotations

import os
import shutil
from importlib import import_module
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

TEST_ROOT = Path(__file__).resolve().parent / '.pytest_runtime'
TEMP_ROOT = TEST_ROOT / 'temp'
UPLOADS_ROOT = TEST_ROOT / 'uploads'
CHROMA_ROOT = TEST_ROOT / 'chroma'
LOGS_ROOT = TEST_ROOT / 'logs'
CASE_TMP_ROOT = TEST_ROOT / 'case_tmp'
SQLITE_PATH = LOGS_ROOT / 'test_app.db'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALS_ROOT = TEST_ROOT / 'evals'
PROJECT_SAMPLE_EVAL_DATASET_PATH = PROJECT_ROOT / 'data' / 'evals' / 'sample_eval_dataset.jsonl'
SAMPLE_EVAL_DATASET_PATH = EVALS_ROOT / 'sample_eval_dataset.jsonl'
RESET_TABLES = [
    'review_decisions',
    'evaluation_case_results',
    'evaluation_runs',
    'request_logs',
    'api_request_events',
    'provider_usage_events',
    'chunk_embeddings',
    'document_chunks',
    'documents',
]

for directory in (TEMP_ROOT, UPLOADS_ROOT, CHROMA_ROOT, LOGS_ROOT, CASE_TMP_ROOT, EVALS_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

os.environ['TMP'] = str(TEMP_ROOT)
os.environ['TEMP'] = str(TEMP_ROOT)
os.environ['TMPDIR'] = str(TEMP_ROOT)
os.environ['DEBUG'] = 'false'
os.environ['ENVIRONMENT'] = 'development'
os.environ['ENABLE_API_DOCS'] = 'false'
os.environ['UI_PUBLIC_ENABLED'] = 'false'
os.environ['AUTH_MODE'] = 'disabled'
os.environ['CORS_ALLOWED_ORIGINS'] = ''
os.environ['TRUSTED_HOSTS'] = ''
os.environ['TRUST_PROXY_HEADERS'] = 'false'
os.environ['PROXY_TRUSTED_IPS'] = '127.0.0.1'
os.environ['FORCE_HTTPS'] = ''
os.environ['VECTOR_BACKEND'] = 'sqlite'
os.environ['OPENAI_API_KEY'] = ''
os.environ['GEMINI_API_KEY'] = ''
os.environ['API_AUTH_ENABLED'] = 'false'
os.environ['READER_API_KEYS'] = ''
os.environ['ANALYST_API_KEYS'] = ''
os.environ['REVIEWER_API_KEYS'] = ''
os.environ['ADMIN_API_KEYS'] = ''
os.environ['UI_API_KEY'] = ''
os.environ['UI_FORWARDED_ACCESS_TOKEN_HEADERS'] = 'Authorization,X-Ms-Token-Aad-Access-Token'
os.environ['JWT_ISSUER'] = ''
os.environ['JWT_AUDIENCES'] = ''
os.environ['JWT_JWKS_URL'] = ''
os.environ['JWT_SHARED_SECRET'] = ''
os.environ['JWT_ALGORITHMS'] = 'RS256'
os.environ['JWT_REQUIRE_HTTPS_METADATA'] = 'true'
os.environ['API_RATE_LIMIT_REQUESTS_PER_IP'] = ''
os.environ['API_RATE_LIMIT_REQUESTS_PER_USER'] = ''
os.environ['UPLOADS_DIR'] = str(UPLOADS_ROOT)
os.environ['CHROMA_DIR'] = str(CHROMA_ROOT)
os.environ['LOGS_DIR'] = str(LOGS_ROOT)
os.environ['EVALS_DIR'] = str(EVALS_ROOT)
os.environ['SQLITE_PATH'] = str(SQLITE_PATH)
os.environ['SAMPLE_EVAL_DATASET_PATH'] = str(SAMPLE_EVAL_DATASET_PATH)
os.environ['PROVIDER_MAX_REQUESTS_PER_WINDOW'] = ''
os.environ['PROVIDER_DAILY_REQUEST_LIMIT'] = ''
os.environ['PROVIDER_DAILY_INPUT_CHAR_LIMIT'] = ''
os.environ['PROVIDER_DAILY_ESTIMATED_COST_LIMIT_USD'] = ''
os.environ['PROVIDER_PER_USER_MAX_REQUESTS_PER_WINDOW'] = ''
os.environ['PROVIDER_PER_USER_DAILY_REQUEST_LIMIT'] = ''
os.environ['PROVIDER_PER_USER_DAILY_INPUT_CHAR_LIMIT'] = ''
os.environ['PROVIDER_PER_USER_DAILY_ESTIMATED_COST_LIMIT_USD'] = ''
os.environ['LLM_ESTIMATED_COST_PER_1K_CHARS_USD'] = '0'
os.environ['EMBEDDING_ESTIMATED_COST_PER_1K_CHARS_USD'] = '0'

get_settings = import_module('app.core.config').get_settings
main_module = import_module('app.api.main')
initialize_database = import_module('app.db.init_db').initialize_database
get_connection = import_module('app.db.sqlite').get_connection


def _clear_directory(directory: Path) -> None:
    if not directory.exists():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def reset_app_state() -> None:
    get_settings.cache_clear()
    settings = get_settings()
    settings.ensure_directories()
    initialize_database()

    with get_connection() as connection:
        for table in RESET_TABLES:
            connection.execute(f'DELETE FROM {table}')

    _clear_directory(settings.uploads_dir)
    _clear_directory(settings.chroma_dir)
    _clear_directory(CASE_TMP_ROOT)
    _clear_directory(settings.evals_dir)
    shutil.copyfile(PROJECT_SAMPLE_EVAL_DATASET_PATH, settings.sample_eval_dataset_path)

    yield

    get_settings.cache_clear()


@pytest.fixture()
def client() -> TestClient:
    get_settings.cache_clear()
    app = main_module.create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def tmp_path() -> Path:
    path = CASE_TMP_ROOT / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path
