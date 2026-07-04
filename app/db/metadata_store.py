"""
Asynchronous SQLite metadata store for Legal RAG using aiosqlite.

Migrates the massive in-memory chunks_metadata.json to a lightweight,
indexed SQLite database to eliminate OOM errors and reduce cold start time
on Vercel serverless functions.
"""
import logging
import aiosqlite
from pathlib import Path
from typing import Optional

from app.config import CHUNKS_METADATA_PATH

logger = logging.getLogger(__name__)

DB_PATH = Path(CHUNKS_METADATA_PATH).with_suffix(".db")

class MetadataStore:
    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = str(db_path)
        
    async def init_db(self):
        """Ensure the database and table exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    text TEXT NOT NULL,
                    article_number TEXT,
                    section TEXT,
                    act_name TEXT,
                    part TEXT,
                    page TEXT,
                    source_file TEXT
                )
            """)
            await db.commit()

    async def get_chunk_by_id(self, chunk_id: int) -> Optional[dict]:
        """Fetch a single chunk's metadata by its integer ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return dict(row)
                return None

    async def get_chunks_batch(self, chunk_ids: list[int]) -> list[dict]:
        """Fetch multiple chunks efficiently using an IN clause."""
        if not chunk_ids:
            return []
            
        placeholders = ",".join("?" * len(chunk_ids))
        query = f"SELECT * FROM chunks WHERE id IN ({placeholders})"
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, chunk_ids) as cursor:
                rows = await cursor.fetchall()
                # Create a lookup dictionary to preserve order
                lookup = {row["id"]: dict(row) for row in rows}
                # Return in the exact order requested, filtering missing ones
                return [lookup[cid] for cid in chunk_ids if cid in lookup]

# Global singleton instance
metadata_store = MetadataStore()
