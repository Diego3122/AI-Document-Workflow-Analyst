from __future__ import annotations

import hashlib
import math
import re

from app.core.config import get_settings
from app.services.provider_guard_service import ProviderGuardService, ProviderUsageLimitExceeded


GEMINI_EMBED_BATCH_MAX_TEXTS = 32
GEMINI_EMBED_BATCH_MAX_CHARS = 12000


class EmbeddingService:
    """Embeds text with Gemini or OpenAI when configured, otherwise uses a deterministic local fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.dimension = self.settings.embedding_dimensions
        self._backend = 'local_hash'
        self.provider_guard = ProviderGuardService()

    @property
    def backend(self) -> str:
        return self._backend

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        provider = self.settings.resolve_embedding_provider()
        if provider == 'gemini':
            try:
                return self._embed_with_gemini(texts)
            except ProviderUsageLimitExceeded:
                raise
            except Exception:
                if self.settings.embedding_provider == 'gemini':
                    raise
        elif provider == 'openai':
            try:
                return self._embed_with_openai(texts)
            except ProviderUsageLimitExceeded:
                raise
            except Exception:
                if self.settings.embedding_provider == 'openai':
                    raise

        self._backend = 'local_hash'
        return [self._embed_with_local_hash(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    def _embed_with_gemini(self, texts: list[str]) -> list[list[float]]:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError('google-genai must be installed to use Gemini embeddings.') from exc

        client = genai.Client(api_key=self.settings.gemini_api_key)
        model_name = self.settings.resolved_embedding_model()
        vectors: list[list[float]] = []

        for batch in self._batched_texts(
            texts,
            max_texts=GEMINI_EMBED_BATCH_MAX_TEXTS,
            max_chars=GEMINI_EMBED_BATCH_MAX_CHARS,
        ):
            self.provider_guard.check_and_record(provider='gemini', operation='embedding', texts=batch)
            result = client.models.embed_content(
                model=model_name,
                contents=batch,
                config=types.EmbedContentConfig(output_dimensionality=self.dimension),
            )
            embeddings = getattr(result, 'embeddings', None)
            if not embeddings:
                raise RuntimeError('Gemini embeddings returned no vectors.')
            if len(embeddings) != len(batch):
                raise RuntimeError('Gemini embeddings returned an unexpected number of vectors.')

            for embedding in embeddings:
                values = getattr(embedding, 'values', None)
                if not values:
                    raise RuntimeError('Gemini embeddings returned an empty vector.')
                vectors.append(list(values))

        self._backend = 'gemini'
        return vectors

    def _embed_with_openai(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError('openai must be installed to use OpenAI embeddings.') from exc

        self.provider_guard.check_and_record(provider='openai', operation='embedding', texts=texts)
        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.embeddings.create(model=self.settings.resolved_embedding_model(), input=texts)
        self._backend = 'openai'
        return [item.embedding for item in response.data]

    def _batched_texts(self, texts: list[str], *, max_texts: int, max_chars: int) -> list[list[str]]:
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_chars = 0

        for text in texts:
            text_chars = len(text)
            if current_batch and (len(current_batch) >= max_texts or current_chars + text_chars > max_chars):
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(text)
            current_chars += text_chars

        if current_batch:
            batches.append(current_batch)
        return batches

    def _embed_with_local_hash(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for term in self._tokenize(text):
            digest = hashlib.sha256(term.encode('utf-8')).digest()
            index = int.from_bytes(digest[:4], 'big') % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [round(value / norm, 6) for value in vector]

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in re.findall(r'[a-zA-Z0-9]+', text.lower()) if len(token) > 1]
