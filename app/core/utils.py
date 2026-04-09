from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


def generate_id(prefix: str) -> str:
    return f'{prefix}_{uuid4().hex[:12]}'


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(filename: str) -> str:
    return ''.join(character if character.isalnum() or character in {'.', '_', '-'} else '_' for character in filename)
