"""
Conversation summarizer for Tier 1.

Generates and stores conversation summaries in PostgreSQL.
Supports recursive compression when summaries exceed budget.
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage

from .token_budget import estimate_tokens

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """You are a conversation summarizer. Summarize the following conversation messages concisely.
Focus on:
- Key topics discussed
- Important decisions made
- Relevant context for future conversation
- Tool calls and their outcomes (brief)

Output a concise summary in the same language as the conversation. Do NOT use markdown headers."""

COMPRESS_SYSTEM_PROMPT = """Compress the following summary to approximately 1/3 of its length.
Keep the most important information. Output in the same language."""

TOPIC_EXTRACTION_PROMPT = """Extract 3-8 short topic tags from the following conversation.
Each tag should be 1-4 words, describing a key topic, concept, or entity discussed.
Output ONLY a JSON array of strings, nothing else.
Example: ["Python基础", "数据库设计", "API认证", "用户管理"]"""


class ConversationSummarizer:
    """Generates and manages conversation summaries."""

    def __init__(self, pg_conn=None, llm=None, max_summary_tokens: int = 1000):
        self._pg_conn = pg_conn
        self._llm = llm
        self.max_summary_tokens = max_summary_tokens
        self._setup_table()

    def _setup_table(self):
        """Create conversation_summaries table if using PostgreSQL."""
        if not self._pg_conn:
            return
        try:
            with self._pg_conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_summaries (
                        thread_id TEXT PRIMARY KEY,
                        summary TEXT NOT NULL,
                        message_count INT NOT NULL DEFAULT 0,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                """)
        except Exception as e:
            logger.warning("Failed to create conversation_summaries table: %s", e)

    def load_summary(self, thread_id: str) -> Optional[str]:
        """Load existing summary for a thread from PostgreSQL."""
        if not self._pg_conn:
            return None
        try:
            with self._pg_conn.cursor() as cur:
                cur.execute(
                    "SELECT summary FROM conversation_summaries WHERE thread_id = %s",
                    (thread_id,),
                )
                row = cur.fetchone()
                if row:
                    return row["summary"] if isinstance(row, dict) else row[0]
        except Exception as e:
            logger.warning("Failed to load summary for thread %s: %s", thread_id, e)
        return None

    def save_summary(self, thread_id: str, summary: str, message_count: int):
        """Save summary to PostgreSQL."""
        if not self._pg_conn:
            return
        try:
            with self._pg_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_summaries (thread_id, summary, message_count, updated_at)
                    VALUES (%s, %s, %s, now())
                    ON CONFLICT (thread_id) DO UPDATE SET
                        summary = EXCLUDED.summary,
                        message_count = EXCLUDED.message_count,
                        updated_at = now()
                    """,
                    (thread_id, summary, message_count),
                )
        except Exception as e:
            logger.warning("Failed to save summary for thread %s: %s", thread_id, e)

    def generate_summary(self, messages: list) -> Optional[str]:
        """Generate a summary of the given messages using the LLM."""
        if not self._llm or not messages:
            return None

        # Build conversation text from messages
        lines = []
        for msg in messages:
            role = type(msg).__name__.replace("Message", "")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # Truncate very long messages for summarization
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role}: {content}")

        conversation_text = "\n".join(lines)

        try:
            response = self._llm.invoke([
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": conversation_text},
            ])
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning("Failed to generate summary: %s", e)
            return None

    def compress_summary(self, summary: str) -> Optional[str]:
        """Recursively compress a summary to fit within budget."""
        if not self._llm:
            return summary

        try:
            response = self._llm.invoke([
                {"role": "system", "content": COMPRESS_SYSTEM_PROMPT},
                {"role": "user", "content": summary},
            ])
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning("Failed to compress summary: %s", e)
            return summary

    def extract_topics(self, messages: list) -> list[str]:
        """Extract topic tags from messages using the LLM."""
        if not self._llm or not messages:
            return []

        lines = []
        for msg in messages:
            role = type(msg).__name__.replace("Message", "")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"{role}: {content}")

        conversation_text = "\n".join(lines)

        try:
            response = self._llm.invoke([
                {"role": "system", "content": TOPIC_EXTRACTION_PROMPT},
                {"role": "user", "content": conversation_text},
            ])
            raw = response.content if hasattr(response, "content") else str(response)
            # Parse JSON array from response
            import json
            # Try to find JSON array in the response
            raw = raw.strip()
            if raw.startswith("["):
                topics = json.loads(raw)
            else:
                # Try to extract array from markdown code block
                import re
                match = re.search(r'\[.*?\]', raw, re.DOTALL)
                if match:
                    topics = json.loads(match.group())
                else:
                    topics = []
            return [str(t).strip() for t in topics if isinstance(t, str) and t.strip()]
        except Exception as e:
            logger.warning("Failed to extract topics: %s", e)
            return []

    def generate_and_store(
        self,
        thread_id: str,
        messages_to_summarize: list,
        existing_summary: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate summary for new messages, merge with existing summary,
        and store the result.

        If the combined summary exceeds max_summary_tokens, recursively
        compress the old summary first.
        """
        if not messages_to_summarize:
            return existing_summary

        new_summary = self.generate_summary(messages_to_summarize)
        if not new_summary:
            return existing_summary

        if existing_summary:
            # Check if merge would exceed budget
            combined = f"{existing_summary}\n\n{new_summary}"
            if estimate_tokens(combined) > self.max_summary_tokens:
                # Compress old summary to ~1/3
                compressed_old = self.compress_summary(existing_summary)
                if compressed_old:
                    combined = f"{compressed_old}\n\n{new_summary}"
                # If still too long, compress the whole thing
                if estimate_tokens(combined) > self.max_summary_tokens:
                    combined = self.compress_summary(combined) or combined
            final_summary = combined
        else:
            final_summary = new_summary

        # Count total messages summarized
        msg_count = len(messages_to_summarize)
        self.save_summary(thread_id, final_summary, msg_count)

        return final_summary
