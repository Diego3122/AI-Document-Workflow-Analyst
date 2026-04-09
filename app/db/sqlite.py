from __future__ import annotations

import sqlite3
from contextlib import contextmanager

from app.core.config import get_settings


SCHEMA_SENTINEL_TABLE = 'documents'



def _ensure_schema(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (SCHEMA_SENTINEL_TABLE,),
    ).fetchone()
    if row is not None:
        return

    from app.db.init_db import apply_schema

    apply_schema(connection)
    connection.commit()



def connect(*, ensure_schema: bool = True) -> sqlite3.Connection:
    settings = get_settings()
    settings.ensure_directories()
    connection = sqlite3.connect(settings.sqlite_path)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys = ON')
    if ensure_schema:
        _ensure_schema(connection)
    return connection


@contextmanager
def get_connection():
    connection = connect()
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()
