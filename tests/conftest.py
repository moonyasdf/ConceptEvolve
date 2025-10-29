import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture
def in_memory_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE concepts (
            id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            draft_history TEXT,
            critique_history TEXT,
            system_requirements TEXT,
            generation INTEGER,
            parent_id TEXT,
            inspiration_ids TEXT,
            embedding TEXT,
            scores TEXT,
            combined_score REAL,
            island_idx INTEGER,
            timestamp REAL
        )
        """
    )
    conn.commit()
    try:
        yield conn, cursor
    finally:
        conn.close()
