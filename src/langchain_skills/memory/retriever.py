"""
Vector-based memory retriever using pgvector.

Stores pruned conversation fragments as embeddings in PostgreSQL,
enabling semantic recall of past conversation details.

Architecture:
  - Pruned messages → chunked by conversation turns → embedded → stored in memory_vectors
  - recall_memory tool → query embedding → cosine similarity search → return chunks
  - Topic index → extracted during summarization → injected in Tier 1
  - Optional topic pre-filter narrows search to relevant chunks

Chunking strategy:
  - Primary: by conversation turn (each Human+AI exchange = one chunk)
  - Fallback: if a single turn exceeds MAX_TURN_CHARS, split it with overlap
  - This preserves conversational coherence within each chunk

Embedding strategy:
  - Uses the configured embedding model (OpenAI-compatible API)
  - Falls back to ILIKE keyword search if embeddings are unavailable
"""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# A single conversation turn larger than this gets split further
MAX_TURN_CHARS = 1200
SPLIT_OVERLAP = 150


def _chunk_by_turns(messages: list) -> list[str]:
    """
    Chunk messages by conversation turns (Human+AI pairs).

    Each chunk is one complete exchange. If a turn is very long,
    it gets split further with overlap to stay within budget.
    """
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    chunks: list[str] = []
    current_turn_lines: list[str] = []

    def _content_str(msg) -> str:
        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text = (
                        block.get("text")
                        or block.get("thinking")
                        or block.get("reasoning")
                        or ""
                    )
                    if text:
                        parts.append(text)
            return "\n".join(parts)
        return str(content) if content else ""

    def flush():
        if not current_turn_lines:
            return
        text = "\n".join(current_turn_lines)
        if not text.strip():
            current_turn_lines.clear()
            return
        # Split oversized turns
        if len(text) > MAX_TURN_CHARS:
            chunks.extend(_split_long_text(text))
        else:
            chunks.append(text)
        current_turn_lines.clear()

    for msg in messages:
        role = type(msg).__name__.replace("Message", "")
        content = _content_str(msg)
        if not content.strip():
            continue

        # A new HumanMessage starts a new turn (flush previous)
        if isinstance(msg, HumanMessage) and current_turn_lines:
            flush()

        current_turn_lines.append(f"{role}: {content}")

    flush()
    return chunks


def _split_long_text(
    text: str,
    max_chars: int = MAX_TURN_CHARS,
    overlap: int = SPLIT_OVERLAP,
) -> list[str]:
    """Split a long text block into overlapping segments."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def _messages_to_text(messages: list) -> str:
    """Convert a list of LangChain messages into a single text block."""
    lines = []
    for msg in messages:
        role = type(msg).__name__.replace("Message", "")
        content = msg.content if isinstance(msg.content, str) else ""
        if isinstance(msg.content, list):
            parts = []
            for block in msg.content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text = (
                        block.get("text")
                        or block.get("thinking")
                        or block.get("reasoning")
                        or ""
                    )
                    if text:
                        parts.append(text)
            content = "\n".join(parts)
        if content.strip():
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


class MemoryRetriever:
    """
    Stores and retrieves conversation fragments using pgvector.

    Uses the LLM provider's embedding API for semantic search.
    Falls back to keyword search if embeddings are unavailable.
    """

    def __init__(
        self,
        pg_conn=None,
        embedding_model=None,
        embedding_dimensions: int = 0,
    ):
        self._pg_conn = pg_conn
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._table_ready = False
        self._setup_table()

    def _setup_table(self):
        """Create memory_vectors table with pgvector extension."""
        if not self._pg_conn:
            return
        try:
            with self._pg_conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # Determine vector dimension
                dim = self._embedding_dimensions or self._detect_dimensions()

                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS memory_vectors (
                        id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        chunk TEXT NOT NULL,
                        topics TEXT[] DEFAULT '{{}}',
                        embedding vector({dim}),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_vectors_thread
                    ON memory_vectors (thread_id)
                """)
                # GIN index on topics for array containment queries
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_vectors_topics
                    ON memory_vectors USING GIN (topics)
                """)
                # Topic index table (lightweight metadata per thread)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS memory_topics (
                        thread_id TEXT NOT NULL,
                        topic TEXT NOT NULL,
                        mention_count INT NOT NULL DEFAULT 1,
                        last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        PRIMARY KEY (thread_id, topic)
                    )
                """)
                self._table_ready = True
        except Exception as e:
            logger.warning("Failed to setup memory_vectors table: %s", e)

    def _detect_dimensions(self) -> int:
        """Detect embedding dimensions by doing a test embed."""
        if self._embedding_model:
            try:
                test = self._embedding_model.embed_query("test")
                dim = len(test)
                self._embedding_dimensions = dim
                return dim
            except Exception:
                pass
        # Default for common models
        return 1536

    def _embed(self, text: str) -> Optional[list[float]]:
        """Embed text using the configured embedding model."""
        if not self._embedding_model:
            return None
        try:
            return self._embedding_model.embed_query(text)
        except Exception as e:
            logger.warning("Embedding failed: %s", e)
            return None

    def _make_id(self, thread_id: str, chunk: str) -> str:
        """Deterministic ID for deduplication."""
        h = hashlib.sha256(f"{thread_id}:{chunk[:200]}".encode()).hexdigest()[:16]
        return f"mem-{h}"

    def store_messages(
        self,
        thread_id: str,
        messages: list,
        topics: Optional[list[str]] = None,
    ) -> int:
        """
        Chunk messages by conversation turns and store into the vector table.

        Returns the number of chunks stored.
        """
        if not self._pg_conn or not self._table_ready:
            return 0

        chunks = _chunk_by_turns(messages)
        if not chunks:
            return 0

        stored = 0
        topic_array = topics or []

        for chunk in chunks:
            chunk_id = self._make_id(thread_id, chunk)
            embedding = self._embed(chunk)

            try:
                with self._pg_conn.cursor() as cur:
                    if embedding:
                        cur.execute(
                            """
                            INSERT INTO memory_vectors (id, thread_id, chunk, topics, embedding)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                            """,
                            (chunk_id, thread_id, chunk, topic_array, embedding),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO memory_vectors (id, thread_id, chunk, topics)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                            """,
                            (chunk_id, thread_id, chunk, topic_array),
                        )
                stored += 1
            except Exception as e:
                logger.warning("Failed to store chunk: %s", e)

        # Update topic index
        if topic_array:
            self._update_topics(thread_id, topic_array)

        logger.info(
            "Stored %d turn-based chunks for thread %s (topics: %s)",
            stored, thread_id, topic_array,
        )
        return stored

    def _update_topics(self, thread_id: str, topics: list[str]):
        """Update the topic index for a thread."""
        if not self._pg_conn:
            return
        try:
            with self._pg_conn.cursor() as cur:
                for topic in topics:
                    cur.execute(
                        """
                        INSERT INTO memory_topics (thread_id, topic, mention_count, last_seen_at)
                        VALUES (%s, %s, 1, now())
                        ON CONFLICT (thread_id, topic) DO UPDATE SET
                            mention_count = memory_topics.mention_count + 1,
                            last_seen_at = now()
                        """,
                        (thread_id, topic),
                    )
        except Exception as e:
            logger.warning("Failed to update topics: %s", e)

    def load_topic_index(self, thread_id: str) -> list[str]:
        """Load the topic index for a thread (used by middleware for Tier 1 metadata)."""
        if not self._pg_conn:
            return []
        try:
            with self._pg_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT topic FROM memory_topics
                    WHERE thread_id = %s
                    ORDER BY last_seen_at DESC, mention_count DESC
                    LIMIT 20
                    """,
                    (thread_id,),
                )
                rows = cur.fetchall()
                return [
                    r["topic"] if isinstance(r, dict) else r[0]
                    for r in rows
                ]
        except Exception as e:
            logger.warning("Failed to load topic index: %s", e)
            return []

    def search(
        self,
        thread_id: str,
        query: str,
        top_k: int = 5,
        topic: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search for relevant conversation fragments.

        Args:
            thread_id: Which conversation to search in.
            query: The search query text.
            top_k: Maximum number of results.
            topic: Optional topic to pre-filter. If provided, only chunks
                   tagged with this topic (or similar) are searched.
                   If None, searches all chunks in the thread.

        Uses cosine similarity on embeddings, falls back to keyword search.
        Returns list of {chunk, score, topics}.
        """
        if not self._pg_conn or not self._table_ready:
            return []

        embedding = self._embed(query)

        if embedding:
            return self._vector_search(thread_id, embedding, top_k, topic=topic)
        else:
            return self._keyword_search(thread_id, query, top_k, topic=topic)

    def _vector_search(
        self,
        thread_id: str,
        embedding: list[float],
        top_k: int,
        topic: Optional[str] = None,
    ) -> list[dict]:
        """Search using cosine similarity, with optional topic pre-filter."""
        try:
            with self._pg_conn.cursor() as cur:
                if topic:
                    # Pre-filter: only search chunks whose topics array
                    # contains the given topic (case-insensitive via ILIKE on any element)
                    cur.execute(
                        """
                        SELECT chunk, topics,
                               1 - (embedding <=> %s::vector) AS score
                        FROM memory_vectors
                        WHERE thread_id = %s
                          AND embedding IS NOT NULL
                          AND EXISTS (
                              SELECT 1 FROM unnest(topics) AS t
                              WHERE t ILIKE '%%' || %s || '%%'
                          )
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (embedding, thread_id, topic, embedding, top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT chunk, topics,
                               1 - (embedding <=> %s::vector) AS score
                        FROM memory_vectors
                        WHERE thread_id = %s AND embedding IS NOT NULL
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (embedding, thread_id, embedding, top_k),
                    )
                return self._rows_to_results(cur.fetchall())
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            return self._keyword_search(thread_id, "", 0)

    def _keyword_search(
        self,
        thread_id: str,
        query: str,
        top_k: int,
        topic: Optional[str] = None,
    ) -> list[dict]:
        """Fallback: keyword-based search using PostgreSQL ILIKE."""
        if not query.strip():
            return []
        try:
            words = query.strip().split()
            conditions = " AND ".join(
                "chunk ILIKE '%%' || %s || '%%'" for _ in words
            )
            where = f"thread_id = %s AND ({conditions})"
            params: list = [thread_id, *words]

            if topic:
                where += " AND EXISTS (SELECT 1 FROM unnest(topics) AS t WHERE t ILIKE '%%' || %s || '%%')"
                params.append(topic)

            params.append(top_k)
            sql = f"""
                SELECT chunk, topics FROM memory_vectors
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT %s
            """
            with self._pg_conn.cursor() as cur:
                cur.execute(sql, params)
                return self._rows_to_results(cur.fetchall(), default_score=0.5)
        except Exception as e:
            logger.warning("Keyword search failed: %s", e)
            return []

    @staticmethod
    def _rows_to_results(rows, default_score: Optional[float] = None) -> list[dict]:
        """Convert DB rows to result dicts."""
        results = []
        for row in rows:
            if isinstance(row, dict):
                results.append({
                    "chunk": row["chunk"],
                    "topics": row.get("topics", []),
                    "score": float(row.get("score", default_score or 0)),
                })
            else:
                results.append({
                    "chunk": row[0],
                    "topics": row[1] if len(row) > 1 else [],
                    "score": float(row[2]) if len(row) > 2 and default_score is None else (default_score or 0),
                })
        return results

    def delete_thread(self, thread_id: str):
        """Delete all vectors and topics for a thread."""
        if not self._pg_conn:
            return
        try:
            with self._pg_conn.cursor() as cur:
                cur.execute("DELETE FROM memory_vectors WHERE thread_id = %s", (thread_id,))
                cur.execute("DELETE FROM memory_topics WHERE thread_id = %s", (thread_id,))
        except Exception as e:
            logger.warning("Failed to delete vectors for thread %s: %s", thread_id, e)
