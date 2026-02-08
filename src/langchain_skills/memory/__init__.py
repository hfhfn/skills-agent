"""
Three-tier memory management system with RAG recall.

Provides context window management through a middleware that intercepts
messages before they reach the LLM, applying a three-tier strategy:

- Tier 1 (Summary + Topic Index): Long-term summary + recallable topics (~10% budget)
- Tier 2 (Condensed): Recent messages with thinking stripped (~35% budget)
- Tier 3 (Full): Most recent messages kept intact (~55% budget)

Additionally, pruned messages are stored as vectors in PostgreSQL (pgvector),
enabling semantic recall via the recall_memory tool.
"""

from .config import MemoryConfig
from .middleware import MemoryMiddleware, MemoryAgentMiddleware
from .retriever import MemoryRetriever
from .summarizer import ConversationSummarizer
from .condenser import condense_message
from .token_budget import (
    TokenBudget,
    calculate_budget,
    estimate_tokens,
    estimate_message_tokens,
)

__all__ = [
    "MemoryConfig",
    "MemoryMiddleware",
    "MemoryAgentMiddleware",
    "MemoryRetriever",
    "ConversationSummarizer",
    "TokenBudget",
    "calculate_budget",
    "condense_message",
    "estimate_tokens",
    "estimate_message_tokens",
]
