from __future__ import annotations

from types import SimpleNamespace

import google.genai
import pytest

from app.core.config import get_settings
from app.db.sqlite import get_connection
from app.services import embedding_service as embedding_module
from app.services.embedding_service import EmbeddingService


class _FakeGeminiModels:
    calls: list[dict[str, object]] = []
    should_fail: bool = False

    def embed_content(self, *, model: str, contents, config):
        if self.__class__.should_fail:
            raise RuntimeError('gemini offline')
        batch = list(contents)
        self.__class__.calls.append(
            {
                'model': model,
                'contents': batch,
                'dimension': config.output_dimensionality,
            }
        )
        embeddings = [
            SimpleNamespace(values=[float(index + 1)] * int(config.output_dimensionality or 1))
            for index, _ in enumerate(batch)
        ]
        return SimpleNamespace(embeddings=embeddings)


class _FakeGeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.models = _FakeGeminiModels()


def test_gemini_embeddings_are_batched_into_single_guarded_request(monkeypatch) -> None:
    monkeypatch.setenv('GEMINI_API_KEY', 'test-gemini-key')
    monkeypatch.setenv('EMBEDDING_PROVIDER', 'gemini')
    monkeypatch.setenv('PROVIDER_MAX_REQUESTS_PER_WINDOW', '1')
    get_settings.cache_clear()

    _FakeGeminiModels.calls = []
    _FakeGeminiModels.should_fail = False
    monkeypatch.setattr(google.genai, 'Client', _FakeGeminiClient)

    service = EmbeddingService()
    vectors = service.embed_texts(['alpha', 'beta', 'gamma'])

    assert len(vectors) == 3
    assert len(_FakeGeminiModels.calls) == 1
    assert _FakeGeminiModels.calls[0]['contents'] == ['alpha', 'beta', 'gamma']

    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS request_count FROM provider_usage_events WHERE provider = 'gemini' AND operation = 'embedding'"
        ).fetchone()
    assert int(row['request_count'] or 0) == 1

    get_settings.cache_clear()


def test_gemini_embeddings_split_into_explicit_batches_when_limits_are_exceeded(monkeypatch) -> None:
    monkeypatch.setenv('GEMINI_API_KEY', 'test-gemini-key')
    monkeypatch.setenv('EMBEDDING_PROVIDER', 'gemini')
    get_settings.cache_clear()

    _FakeGeminiModels.calls = []
    _FakeGeminiModels.should_fail = False
    monkeypatch.setattr(google.genai, 'Client', _FakeGeminiClient)
    monkeypatch.setattr(embedding_module, 'GEMINI_EMBED_BATCH_MAX_TEXTS', 2)
    monkeypatch.setattr(embedding_module, 'GEMINI_EMBED_BATCH_MAX_CHARS', 100)

    service = EmbeddingService()
    vectors = service.embed_texts(['one', 'two', 'three', 'four', 'five'])

    assert len(vectors) == 5
    assert [call['contents'] for call in _FakeGeminiModels.calls] == [
        ['one', 'two'],
        ['three', 'four'],
        ['five'],
    ]

    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS request_count FROM provider_usage_events WHERE provider = 'gemini' AND operation = 'embedding'"
        ).fetchone()
    assert int(row['request_count'] or 0) == 3

    get_settings.cache_clear()


def test_explicit_gemini_provider_does_not_silently_fallback_on_failure(monkeypatch) -> None:
    monkeypatch.setenv('GEMINI_API_KEY', 'test-gemini-key')
    monkeypatch.setenv('EMBEDDING_PROVIDER', 'gemini')
    get_settings.cache_clear()

    _FakeGeminiModels.calls = []
    _FakeGeminiModels.should_fail = True
    monkeypatch.setattr(google.genai, 'Client', _FakeGeminiClient)

    service = EmbeddingService()
    with pytest.raises(RuntimeError, match='gemini offline'):
        service.embed_texts(['alpha'])

    get_settings.cache_clear()


def test_multi_chunk_gemini_indexing_uses_single_embedding_request(client, monkeypatch) -> None:
    monkeypatch.setenv('GEMINI_API_KEY', 'test-gemini-key')
    monkeypatch.setenv('EMBEDDING_PROVIDER', 'gemini')
    monkeypatch.setenv('LLM_PROVIDER', 'local')
    monkeypatch.setenv('PROVIDER_MAX_REQUESTS_PER_WINDOW', '1')
    get_settings.cache_clear()

    _FakeGeminiModels.calls = []
    _FakeGeminiModels.should_fail = False
    monkeypatch.setattr(google.genai, 'Client', _FakeGeminiClient)

    long_text = ('PAYMENT TERMS\n' + ' '.join(f'clause{i}' for i in range(700))).encode('utf-8')
    upload_response = client.post(
        '/documents/upload',
        files={'file': ('large_contract.txt', long_text, 'text/plain')},
    )
    assert upload_response.status_code == 201
    document_id = upload_response.json()['document_id']

    index_response = client.post('/documents/index', json={'document_id': document_id})
    assert index_response.status_code == 200
    assert index_response.json()['chunk_count'] > 1
    assert len(_FakeGeminiModels.calls) == 1
    assert len(_FakeGeminiModels.calls[0]['contents']) == index_response.json()['chunk_count']

    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS request_count FROM provider_usage_events WHERE provider = 'gemini' AND operation = 'embedding'"
        ).fetchone()
    assert int(row['request_count'] or 0) == 1

    get_settings.cache_clear()
