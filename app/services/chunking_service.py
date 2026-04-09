from __future__ import annotations

from app.core.utils import generate_id
from app.models.schemas import ChunkRecord, DocumentRecord


class ChunkingService:
    """Page-aware chunking scaffold for business documents."""

    def chunk_document(
        self,
        document: DocumentRecord,
        target_words: int = 220,
        overlap_words: int = 40,
    ) -> list[ChunkRecord]:
        pages = document.metadata.get('pages') or [{'page_number': 1, 'text': document.text}]
        chunks: list[ChunkRecord] = []

        for page in pages:
            page_number = page.get('page_number')
            page_text = (page.get('text') or '').strip()
            if not page_text:
                continue

            section = self._detect_section(page_text)
            words = page_text.split()
            start = 0
            page_chunk_index = 0
            while start < len(words):
                end = min(start + target_words, len(words))
                chunk_words = words[start:end]
                chunk_text = ' '.join(chunk_words)
                chunks.append(
                    ChunkRecord(
                        chunk_id=generate_id('chunk'),
                        document_id=document.document_id,
                        document_name=document.filename,
                        chunk_index=len(chunks),
                        page_number=page_number,
                        section=section,
                        text=chunk_text,
                        source_uri=document.storage_path,
                        token_count=len(chunk_words),
                        metadata={'page_chunk_index': page_chunk_index},
                    )
                )
                page_chunk_index += 1
                if end == len(words):
                    break
                start = max(end - overlap_words, start + 1)
        return chunks

    def _detect_section(self, text: str) -> str | None:
        for line in text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if len(cleaned) <= 80 and (cleaned.isupper() or cleaned.istitle() or '.' not in cleaned):
                return cleaned
            break
        return None
