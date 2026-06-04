import hashlib
import json
import logging
import sqlite3

logger = logging.getLogger(__name__)


def init_db(db_path: str) -> sqlite3.Connection:
    logger.info("Initializing SQLite database at: %s", db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id       TEXT PRIMARY KEY,
            document TEXT NOT NULL
        );
        """
    )
    conn.commit()
    return conn


def compute_document_id(appellant: str, respondent: str, year: int, month: int, day: int) -> str:
    content = f"{appellant}|{respondent}|{year}-{month:02d}-{day:02d}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_document(conn: sqlite3.Connection, doc_id: str, doc_payload: dict) -> None:
    logger.info("Writing document %s to SQLite", doc_id)
    doc_json = json.dumps(doc_payload, ensure_ascii=False)
    with conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.execute("INSERT INTO documents (id, document) VALUES (?, ?)", (doc_id, doc_json))
