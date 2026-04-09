from __future__ import annotations

from pathlib import Path

from app.models.schemas import DocumentRecord


class ParserService:
    """Parses supported business document formats into plain text with page metadata."""

    def parse(self, document: DocumentRecord) -> DocumentRecord:
        file_path = Path(document.storage_path)
        suffix = file_path.suffix.lower()
        if suffix == '.txt':
            text = file_path.read_text(encoding='utf-8', errors='ignore').strip()
            pages = [{'page_number': 1, 'text': text}]
            return DocumentRecord(
                document_id=document.document_id,
                filename=document.filename,
                storage_path=str(file_path),
                mime_type='text/plain',
                status=document.status,
                text=text,
                page_count=1,
                metadata={**document.metadata, 'pages': pages, 'parser': 'text'},
            )
        if suffix == '.pdf':
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError('pypdf must be installed to parse PDF documents.') from exc

            reader = PdfReader(str(file_path))
            pages = [
                {
                    'page_number': index + 1,
                    'text': (page.extract_text() or '').strip(),
                }
                for index, page in enumerate(reader.pages)
            ]
            text = '\n\n'.join(page['text'] for page in pages if page['text']).strip()
            return DocumentRecord(
                document_id=document.document_id,
                filename=document.filename,
                storage_path=str(file_path),
                mime_type='application/pdf',
                status=document.status,
                text=text,
                page_count=len(pages),
                metadata={**document.metadata, 'pages': pages, 'parser': 'pypdf'},
            )
        raise ValueError(f'Unsupported document type: {suffix}')
